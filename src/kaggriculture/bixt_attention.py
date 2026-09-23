"""BiXT attention with bounded CUDA workspace and an opaque recomputing backward.

The external batch and optimizer update are unchanged. Only the independent
batch rows inside attention are tiled; each tile computes one similarity matrix
and reuses it for both normalizations. Neither scores nor probabilities survive
the operator call. The casts reproduce the dense BF16 primitive, including its
FP32 softmax and BF16 matrix-product boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

_BATCH_TILE = 128
_SCORE_TILE_ELEMENTS = 128 * 4 * 32 * 262


def _check_inputs(
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> None:
    values = (latent_references, token_references, latent_values, token_values)
    if any(value.device.type != "cuda" for value in values):
        raise ValueError("bounded BiXT attention requires CUDA tensors")
    if any(value.dtype != torch.bfloat16 for value in values):
        raise TypeError("bounded BiXT attention requires BF16 references and values")
    if any(value.ndim != 4 for value in values):
        raise ValueError("BiXT references and values must have shape [batch, heads, tokens, width]")
    if any(value.device != latent_references.device for value in (*values, token_valid)):
        raise ValueError("BiXT tensors must share one CUDA device")
    if (
        latent_references.shape != latent_values.shape
        or token_references.shape != token_values.shape
    ):
        raise ValueError("BiXT references and values must have matching shapes")
    batch, heads, latents, width = latent_references.shape
    if token_references.shape[:2] != (batch, heads) or token_references.shape[-1] != width:
        raise ValueError("BiXT streams must share batch, heads, and head width")
    tokens = token_references.shape[-2]
    if heads < 1 or latents < 1 or tokens < 1 or width < 1:
        raise ValueError("BiXT attention dimensions must be nonempty")
    if token_valid.dtype != torch.bool or token_valid.shape != (batch, tokens):
        raise ValueError("BiXT token validity must be a [batch, tokens] boolean tensor")
    if not 1 <= output_tokens <= tokens:
        raise ValueError("BiXT output_tokens must be between one and the data token count")


def _tile_rows(latent_references: Tensor, token_references: Tensor) -> int:
    elements_per_row = (
        latent_references.shape[1] * latent_references.shape[2] * token_references.shape[2]
    )
    return max(1, min(_BATCH_TILE, _SCORE_TILE_ELEMENTS // elements_per_row))


def _scores(latent_references: Tensor, token_references: Tensor) -> Tensor:
    # The BF16 matmul output rounds BEFORE widening and scaling, matching the
    # existing dense primitive rather than silently changing its precision.
    return (latent_references @ token_references.transpose(-1, -2)).float() * (
        latent_references.shape[-1] ** -0.5
    )


def _forward_tile(
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> tuple[Tensor, Tensor]:
    scores = _scores(latent_references, token_references)
    probabilities = scores.masked_fill(~token_valid[:, None, None, :], -float("inf")).softmax(-1)
    latent_output = probabilities.to(torch.bfloat16) @ token_values
    del probabilities
    probabilities = scores[:, :, :, :output_tokens].softmax(-2)
    token_output = probabilities.to(torch.bfloat16).transpose(-1, -2) @ latent_values
    token_output = token_output.masked_fill(~token_valid[:, None, :output_tokens, None], 0.0)
    return latent_output, token_output


@torch.library.custom_op(
    "kaggriculture::bounded_bixt_attention", mutates_args=(), device_types="cuda"
)
def _bounded_attention(
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> tuple[Tensor, Tensor]:
    _check_inputs(
        latent_references, token_references, latent_values, token_values, token_valid, output_tokens
    )
    latent_output = latent_values.new_empty(latent_values.shape)
    token_output = token_values.new_empty(
        (*token_values.shape[:2], output_tokens, token_values.shape[-1])
    )
    tile = _tile_rows(latent_references, token_references)
    for start in range(0, latent_references.shape[0], tile):
        rows = slice(start, start + tile)
        local_latent, local_token = _forward_tile(
            latent_references[rows],
            token_references[rows],
            latent_values[rows],
            token_values[rows],
            token_valid[rows],
            output_tokens,
        )
        latent_output[rows].copy_(local_latent)
        token_output[rows].copy_(local_token)
    return latent_output, token_output


@_bounded_attention.register_fake
def _fake_bounded_attention(
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> tuple[Tensor, Tensor]:
    _check_inputs(
        latent_references, token_references, latent_values, token_values, token_valid, output_tokens
    )
    return (
        latent_values.new_empty(latent_values.shape),
        token_values.new_empty((*token_values.shape[:2], output_tokens, token_values.shape[-1])),
    )


def _backward_tile(
    latent_gradient: Tensor | None,
    token_gradient: Tensor | None,
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    scores = _scores(latent_references, token_references)
    if latent_gradient is not None:
        probabilities = scores.masked_fill(~token_valid[:, None, None, :], -float("inf")).softmax(
            -1
        )
        token_value_gradient = probabilities.to(torch.bfloat16).transpose(-1, -2) @ latent_gradient
        probability_gradient = (latent_gradient @ token_values.transpose(-1, -2)).float()
        correction = (probabilities * probability_gradient).sum(-1, keepdim=True)
        score_gradient = probabilities * (probability_gradient - correction)
        del probabilities, probability_gradient, correction
    else:
        token_value_gradient = torch.zeros_like(token_values)
        score_gradient = torch.zeros_like(scores)
    if token_gradient is not None:
        token_gradient = token_gradient.masked_fill(
            ~token_valid[:, None, :output_tokens, None], 0.0
        )
        probabilities = scores[:, :, :, :output_tokens].softmax(-2)
        latent_value_gradient = probabilities.to(torch.bfloat16) @ token_gradient
        probability_gradient = (latent_values @ token_gradient.transpose(-1, -2)).float()
        correction = (probabilities * probability_gradient).sum(-2, keepdim=True)
        score_gradient[:, :, :, :output_tokens].add_(
            probabilities * (probability_gradient - correction)
        )
        del probabilities, probability_gradient, correction
    else:
        latent_value_gradient = torch.zeros_like(latent_values)
    # Both softmax cotangents accumulate in FP32. The gradient crosses the
    # original BF16 score-product boundary only after their sum and scaling.
    score_gradient = (score_gradient * latent_references.shape[-1] ** -0.5).to(torch.bfloat16)
    latent_reference_gradient = score_gradient @ token_references
    token_reference_gradient = score_gradient.transpose(-1, -2) @ latent_references
    return (
        latent_reference_gradient,
        token_reference_gradient,
        latent_value_gradient,
        token_value_gradient,
    )


@torch.library.custom_op(
    "kaggriculture::bounded_bixt_attention_backward", mutates_args=(), device_types="cuda"
)
def _bounded_attention_backward(
    latent_gradient: Tensor | None,
    token_gradient: Tensor | None,
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    inputs = (latent_references, token_references, latent_values, token_values)
    gradients = tuple(value.new_empty(value.shape) for value in inputs)
    tile = _tile_rows(latent_references, token_references)
    for start in range(0, latent_references.shape[0], tile):
        rows = slice(start, start + tile)
        local = _backward_tile(
            None if latent_gradient is None else latent_gradient[rows],
            None if token_gradient is None else token_gradient[rows],
            *(value[rows] for value in inputs),
            token_valid[rows],
            output_tokens,
        )
        for destination, value in zip(gradients, local, strict=True):
            destination[rows].copy_(value)
    return gradients


@_bounded_attention_backward.register_fake
def _fake_bounded_attention_backward(
    latent_gradient: Tensor | None,
    token_gradient: Tensor | None,
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return tuple(
        value.new_empty(value.shape)
        for value in (latent_references, token_references, latent_values, token_values)
    )


def _setup_context(ctx: object, inputs: tuple, output: tuple[Tensor, Tensor]) -> None:
    *values, output_tokens = inputs
    ctx.save_for_backward(*values)  # type: ignore[attr-defined]
    ctx.output_tokens = output_tokens  # type: ignore[attr-defined]
    ctx.set_materialize_grads(False)  # type: ignore[attr-defined]


@once_differentiable
def _backward(
    ctx: object, latent_gradient: Tensor | None, token_gradient: Tensor | None
) -> tuple[Tensor | None, ...]:
    if latent_gradient is None and token_gradient is None:
        return (None,) * 6
    gradients = _bounded_attention_backward(
        latent_gradient,
        token_gradient,
        *ctx.saved_tensors,
        ctx.output_tokens,  # type: ignore[attr-defined]
    )
    return (*gradients, None, None)


_bounded_attention.register_autograd(_backward, setup_context=_setup_context)


@torch.library.register_vmap(_bounded_attention)
def _vmap_bounded_attention(
    info: object,
    in_dims: Sequence[int | None],
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    output_tokens: int,
) -> tuple[tuple[Tensor, Tensor], tuple[int, int]]:
    lanes = info.batch_size  # type: ignore[attr-defined]
    flattened = []
    for value, dim in zip(
        (latent_references, token_references, latent_values, token_values, token_valid),
        in_dims[:5],
        strict=True,
    ):
        values = (
            value.unsqueeze(0).expand(lanes, *value.shape) if dim is None else value.movedim(dim, 0)
        )
        flattened.append(values.flatten(0, 1))
    outputs = _bounded_attention(*flattened, output_tokens)
    return tuple(value.unflatten(0, (lanes, -1)) for value in outputs), (0, 0)


def bounded_shared_score_attention(
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    *,
    output_tokens: int | None = None,
) -> tuple[Tensor, Tensor]:
    """CUDA BF16 simultaneous attention with bounded temporary storage.

    Inputs may be strided; outputs and gradients use contiguous storage.
    Each example must contain at least one valid input token. First-order
    gradients support either or both outputs. The operation remains opaque
    inside a fullgraph-compiled model; it does not change the PPO minibatch.
    """
    count = token_references.shape[-2] if output_tokens is None else output_tokens
    return _bounded_attention(
        latent_references, token_references, latent_values, token_values, token_valid, count
    )
