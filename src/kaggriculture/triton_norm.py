"""CUDA RMS normalization with batch-independent FP32 row arithmetic.

The custom operation deliberately hides the forward reduction from Inductor:
training and inference must not choose different reduction/fusion strategies.
Inputs may have arbitrary strides; outputs always have contiguous layout.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable


@triton.jit
def _rms_norm_kernel(
    inputs,
    weight,
    output,
    inverse_rms,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    COLUMN_STRIDE: tl.constexpr,
    WEIGHT_COLUMN_STRIDE: tl.constexpr,
    WEIGHT_LANE_STRIDE: tl.constexpr,
    ROWS_PER_LANE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    LANE_WEIGHTS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, BLOCK)
    valid = (rows[:, None] < ROWS) & (columns[None, :] < WIDTH)
    remaining = rows
    input_offset = tl.full((BLOCK_ROWS,), 0, tl.int32)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        input_offset += (remaining % SHAPE[dim]) * STRIDES[dim]
        remaining = remaining // SHAPE[dim]
    values = tl.load(
        inputs + input_offset[:, None] + columns[None, :] * COLUMN_STRIDE, valid, 0
    ).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(values * values, 1) / WIDTH + EPS)
    normalized = values * inverse[:, None]
    if HAS_WEIGHT:
        weight_offset = tl.full((BLOCK_ROWS,), 0, tl.int32)
        if LANE_WEIGHTS:
            weight_offset = (rows // ROWS_PER_LANE) * WEIGHT_LANE_STRIDE
        weights = tl.load(
            weight + weight_offset[:, None] + columns[None, :] * WEIGHT_COLUMN_STRIDE,
            valid,
            0,
        ).to(tl.float32)
        normalized = normalized * weights
    tl.store(output + rows[:, None] * WIDTH + columns[None, :], normalized, valid)
    tl.store(inverse_rms + rows, inverse, rows < ROWS)


def _check_inputs(inputs: Tensor, weight: Tensor | None) -> None:
    if inputs.ndim == 0 or inputs.shape[-1] == 0:
        raise ValueError("RMS normalization requires a nonempty final dimension")
    if inputs.device.type != "cuda":
        raise ValueError("Triton RMS normalization requires CUDA inputs")
    dtypes = (torch.float32, torch.bfloat16, torch.float16)
    if inputs.dtype not in dtypes:
        raise TypeError("Triton RMS normalization requires FP32, BF16, or FP16 inputs")
    if weight is not None:
        if weight.device != inputs.device or weight.dtype not in dtypes:
            raise ValueError("RMS normalization weights must be floating point on the input device")
        if weight.ndim not in (1, 2) or weight.shape[-1] != inputs.shape[-1]:
            raise ValueError("RMS normalization weights must match the final input dimension")
        if weight.ndim == 2 and (inputs.ndim < 2 or weight.shape[0] != inputs.shape[0]):
            raise ValueError("Vmapped RMS normalization weights must match the input lane count")


@torch.library.custom_op("kaggriculture::rms_norm", mutates_args=(), device_types="cuda")
def _rms_norm(inputs: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor]:
    _check_inputs(inputs, weight)
    width = inputs.shape[-1]
    rows = inputs.numel() // width
    output = torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype)
    inverse = torch.empty(inputs.shape[:-1], device=inputs.device, dtype=torch.float32)
    if rows:
        block = triton.next_power_of_2(width)
        # Pack narrow rows without letting batch size alter the reduction layout.
        # Up to 1024 elements per CTA gives widths 20/80 32/8 independent rows.
        block_rows = max(1, min(32, 1024 // block))
        _rms_norm_kernel[(triton.cdiv(rows, block_rows),)](
            inputs,
            weight,
            output,
            inverse,
            WIDTH=width,
            ROWS=rows,
            SHAPE=tuple(inputs.shape[:-1]),
            STRIDES=tuple(inputs.stride()[:-1]),
            COLUMN_STRIDE=inputs.stride(-1),
            WEIGHT_COLUMN_STRIDE=weight.stride(-1) if weight is not None else 0,
            WEIGHT_LANE_STRIDE=weight.stride(0) if weight is not None else 0,
            ROWS_PER_LANE=math.prod(inputs.shape[1:-1]),
            HAS_WEIGHT=weight is not None,
            LANE_WEIGHTS=weight is not None and weight.ndim == 2,
            EPS=eps,
            BLOCK=block,
            BLOCK_ROWS=block_rows,
            num_warps=4,  # pyright: ignore[reportCallIssue]
            enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
        )
    return output, inverse


@_rms_norm.register_fake
def _fake_rms_norm(inputs: Tensor, weight: Tensor | None, eps: float) -> tuple[Tensor, Tensor]:
    _check_inputs(inputs, weight)
    return (
        torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype),
        torch.empty(inputs.shape[:-1], device=inputs.device, dtype=torch.float32),
    )


def _setup_rms_norm_context(
    ctx: object,
    inputs: tuple[Tensor, Tensor | None, float],
    output: tuple[Tensor, Tensor],
) -> None:
    values, weight, _eps = inputs
    _normalized, inverse = output
    ctx.save_for_backward(values, weight, inverse)  # type: ignore[attr-defined]
    ctx.mark_non_differentiable(inverse)  # type: ignore[attr-defined]


@once_differentiable
def _backward_rms_norm(
    ctx: object, gradient: Tensor, _inverse_gradient: Tensor | None
) -> tuple[Tensor | None, Tensor | None, None]:
    inputs, weight, inverse = ctx.saved_tensors  # type: ignore[attr-defined]
    needs_input, needs_weight, _ = ctx.needs_input_grad  # type: ignore[attr-defined]
    inverse = inverse.unsqueeze(-1)
    normalized = inputs.float() * inverse
    gradient = gradient.float()
    input_gradient = None
    if needs_input:
        weighted_gradient = gradient
        if weight is not None:
            weights = weight.float()
            if weight.ndim == 2:
                weights = weights.reshape(
                    weight.shape[0], *((1,) * (inputs.ndim - 2)), weight.shape[-1]
                )
            weighted_gradient = gradient * weights
        projection = (weighted_gradient * normalized).mean(-1, keepdim=True)
        input_gradient = ((weighted_gradient - normalized * projection) * inverse).to(inputs.dtype)
    weight_gradient = None
    if weight is not None and needs_weight:
        # Keep lane-specific parameters separate; only the rows within a lane
        # contribute to its affine gradient. A single vector needs no reduction.
        reduction_dims = tuple(range(weight.ndim - 1, inputs.ndim - 1))
        weight_gradient = gradient * normalized
        if reduction_dims:
            weight_gradient = weight_gradient.sum(reduction_dims)
        weight_gradient = weight_gradient.to(weight.dtype)
    return input_gradient, weight_gradient, None


_rms_norm.register_autograd(_backward_rms_norm, setup_context=_setup_rms_norm_context)


@torch.library.register_vmap(_rms_norm)
def _vmap_rms_norm(
    info: object,
    in_dims: tuple[int | None, ...],
    inputs: Tensor,
    weight: Tensor | None,
    eps: float,
) -> tuple[tuple[Tensor, Tensor], tuple[int, int]]:
    input_dim, weight_dim, _eps_dim = in_dims
    if input_dim is None:
        values = inputs.unsqueeze(0).expand(info.batch_size, *inputs.shape)  # type: ignore[attr-defined]
    else:
        values = inputs.movedim(input_dim, 0)
    weights = weight.movedim(weight_dim, 0) if weight_dim is not None else weight
    return _rms_norm(values, weights, eps), (0, 0)


def rms_norm(inputs: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """Normalize the final dimension, optionally applying a vector affine weight.

    CUDA FP32/BF16/FP16 inputs and weights use FP32 normalization arithmetic;
    the result has the input dtype and contiguous layout. Leading and final
    strides, including expanded inputs, are supported without materializing a
    contiguous input. A single vmap level supports mapped or shared inputs and
    weights. Autograd supports first-order input and affine-weight gradients.
    """
    if weight is not None and weight.ndim != 1:
        raise ValueError("RMS normalization requires a one-dimensional affine weight")
    return _rms_norm(inputs, weight, eps)[0]
