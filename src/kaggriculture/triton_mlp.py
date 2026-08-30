"""CUDA-only fused ReLU-squared MLP kernels for structured models."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.tools.tensor_descriptor import TensorDescriptor

_E4M3_MAX = 448.0


def _require_cuda_bf16(*tensors: Tensor) -> None:
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("fused structured MLPs require CUDA")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("fused structured MLP activations and master weights must be BF16")


@triton.jit
def _linear_relu_square_kernel(
    input_descriptor,
    weight_descriptor,
    output_descriptor,
    auxiliary_descriptor,
    post_fp8_descriptor,
    partial_amax,
    dequant_scale,
    activation_scale,
    rows,
    outputs,
    inputs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FORWARD: tl.constexpr,
    USE_FP8: tl.constexpr,
    EMIT_FP8: tl.constexpr,
):
    program = tl.program_id(0)
    output_blocks = tl.cdiv(outputs, BLOCK_N)
    input_blocks = tl.cdiv(inputs, BLOCK_K)
    tiles = tl.cdiv(rows, BLOCK_M) * output_blocks
    output_tile = program - NUM_SMS
    local_amax = 0.0
    if EMIT_FP8:
        inverse_activation_scale = 1.0 / tl.load(activation_scale)

    for tile in tl.range(program, tiles, NUM_SMS, flatten=True):
        row_offset = (tile // output_blocks) * BLOCK_M
        output_offset = (tile % output_blocks) * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for input_block in range(input_blocks):
            input_offset = input_block * BLOCK_K
            values = input_descriptor.load([row_offset, input_offset])
            weights = weight_descriptor.load([output_offset, input_offset])
            accumulator = tl.dot(values, weights.T, accumulator)
        if USE_FP8:
            accumulator *= tl.load(dequant_scale)

        output_tile += NUM_SMS
        row_output = (output_tile // output_blocks) * BLOCK_M
        column_output = (output_tile % output_blocks) * BLOCK_N
        split = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        split = tl.permute(split, (0, 2, 1))
        first, second = tl.split(split)

        first = first.to(tl.bfloat16)
        second = second.to(tl.bfloat16)
        if FORWARD:
            first = tl.maximum(first, 0.0)
            second = tl.maximum(second, 0.0)
            first_post = first * first
            second_post = second * second
            output_descriptor.store([row_output, column_output], first_post)
            output_descriptor.store(
                [row_output, column_output + BLOCK_N // 2],
                second_post,
            )
            if EMIT_FP8:
                first_fp8 = tl.minimum(first_post * inverse_activation_scale, 448.0)
                second_fp8 = tl.minimum(second_post * inverse_activation_scale, 448.0)
                post_fp8_descriptor.store(
                    [row_output, column_output],
                    first_fp8.to(tl.float8e4nv),
                )
                post_fp8_descriptor.store(
                    [row_output, column_output + BLOCK_N // 2],
                    second_fp8.to(tl.float8e4nv),
                )
                local_amax = tl.maximum(
                    local_amax,
                    tl.max(tl.max(first_post.to(tl.float32), axis=1), axis=0),
                )
                local_amax = tl.maximum(
                    local_amax,
                    tl.max(tl.max(second_post.to(tl.float32), axis=1), axis=0),
                )
        else:
            first_post = auxiliary_descriptor.load([row_output, column_output])
            second_post = auxiliary_descriptor.load([row_output, column_output + BLOCK_N // 2])
            output_descriptor.store(
                [row_output, column_output],
                (2.0 * first * tl.sqrt(first_post.to(tl.float32))).to(tl.bfloat16),
            )
            output_descriptor.store(
                [row_output, column_output + BLOCK_N // 2],
                (2.0 * second * tl.sqrt(second_post.to(tl.float32))).to(tl.bfloat16),
            )

    if EMIT_FP8:
        tl.store(partial_amax + program, local_amax)


def _linear_relu_square(
    values: Tensor,
    weight: Tensor,
    post: Tensor | None = None,
    *,
    values_f8: Tensor | None = None,
    weight_f8: Tensor | None = None,
    dequant_scale: Tensor | None = None,
    activation_scale: Tensor | None = None,
    partial_amax: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    _require_cuda_bf16(values, weight)
    if values.ndim != 2 or weight.ndim != 2:
        raise ValueError("fused structured MLP kernels require two-dimensional inputs")
    rows, inputs = values.shape
    outputs, weight_inputs = weight.shape
    if inputs != weight_inputs:
        raise ValueError("fused structured MLP weight shape does not match its input")
    if inputs % 64 or outputs % 128:
        raise ValueError("fused structured MLP widths must be multiples of 64 and 128")

    use_fp8 = weight_f8 is not None
    emit_fp8 = activation_scale is not None
    if use_fp8 != (values_f8 is not None):
        raise ValueError("FP8 up projection requires both activation and weight copies")
    if emit_fp8 and not use_fp8:
        raise ValueError("FP8 down projection requires an FP8 up projection")
    if use_fp8 and dequant_scale is None:
        raise ValueError("FP8 up projection requires a dequantization scale")
    if emit_fp8 and partial_amax is None:
        raise ValueError("FP8 down projection requires per-SM activation maxima")

    result = torch.empty((rows, outputs), device=values.device, dtype=torch.bfloat16)
    block_m, block_n = 128, 128
    block_k = 128 if use_fp8 else 64
    forward = post is None
    auxiliary = (
        torch.empty((block_m, block_n // 2), device=values.device, dtype=torch.bfloat16)
        if forward
        else post
    )
    input_kernel = values_f8 if values_f8 is not None else values
    weight_kernel = weight_f8 if weight_f8 is not None else weight
    input_descriptor = TensorDescriptor.from_tensor(input_kernel, [block_m, block_k])
    weight_descriptor = TensorDescriptor.from_tensor(weight_kernel, [block_n, block_k])
    output_descriptor = TensorDescriptor.from_tensor(result, [block_m, block_n // 2])
    auxiliary_descriptor = TensorDescriptor.from_tensor(
        auxiliary,
        [block_m, block_n // 2],
    )

    if emit_fp8:
        assert activation_scale is not None
        assert partial_amax is not None
        post_fp8 = torch.empty(
            (rows, outputs),
            device=values.device,
            dtype=torch.float8_e4m3fn,
        )
        post_fp8_descriptor = TensorDescriptor.from_tensor(
            post_fp8,
            [block_m, block_n // 2],
        )
    else:
        post_fp8 = None
        post_fp8_descriptor = auxiliary_descriptor
        activation_scale = values
        partial_amax = values
    if dequant_scale is None:
        dequant_scale = values

    sms = torch.cuda.get_device_properties(values.device).multi_processor_count
    if emit_fp8 and partial_amax.numel() < sms:
        raise ValueError("FP8 activation-max buffer is smaller than the CUDA SM count")
    tiles = triton.cdiv(rows, block_m) * triton.cdiv(outputs, block_n)
    grid = (sms if emit_fp8 else min(sms, tiles),)
    _linear_relu_square_kernel[grid](
        input_descriptor,
        weight_descriptor,
        output_descriptor,
        auxiliary_descriptor,
        post_fp8_descriptor,
        partial_amax,
        dequant_scale,
        activation_scale,
        rows,
        outputs,
        inputs,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_SMS=sms,
        FORWARD=forward,
        USE_FP8=use_fp8,
        EMIT_FP8=emit_fp8,
        num_stages=2,  # pyright: ignore[reportCallIssue]
        num_warps=8,  # pyright: ignore[reportCallIssue]
    )
    if post_fp8 is not None:
        return result, post_fp8
    return result


@triton.jit
def _batched_linear_relu_square_kernel(
    input_descriptor,
    weight_descriptor,
    output_descriptor,
    rows,
    outputs,
    inputs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    lane = tl.program_id(0)
    tile = tl.program_id(1)
    output_blocks = tl.cdiv(outputs, BLOCK_N)
    row_offset = (tile // output_blocks) * BLOCK_M
    output_offset = (tile % output_blocks) * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for input_offset in range(0, inputs, BLOCK_K):
        values = input_descriptor.load([lane, row_offset, input_offset]).reshape(
            BLOCK_M,
            BLOCK_K,
        )
        weights = weight_descriptor.load([lane, output_offset, input_offset]).reshape(
            BLOCK_N,
            BLOCK_K,
        )
        accumulator = tl.dot(values, weights.T, accumulator)
    split = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
    split = tl.permute(split, (0, 2, 1))
    first, second = tl.split(split)
    first = tl.maximum(first.to(tl.bfloat16), 0.0)
    second = tl.maximum(second.to(tl.bfloat16), 0.0)
    output_descriptor.store(
        [lane, row_offset, output_offset],
        (first * first).reshape(1, BLOCK_M, BLOCK_N // 2),
    )
    output_descriptor.store(
        [lane, row_offset, output_offset + BLOCK_N // 2],
        (second * second).reshape(1, BLOCK_M, BLOCK_N // 2),
    )


@triton.jit
def _batched_linear_kernel(
    input_descriptor,
    weight_descriptor,
    output_descriptor,
    rows,
    outputs,
    inputs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    lane = tl.program_id(0)
    tile = tl.program_id(1)
    output_blocks = tl.cdiv(outputs, BLOCK_N)
    row_offset = (tile // output_blocks) * BLOCK_M
    output_offset = (tile % output_blocks) * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for input_offset in range(0, inputs, BLOCK_K):
        values = input_descriptor.load([lane, row_offset, input_offset]).reshape(
            BLOCK_M,
            BLOCK_K,
        )
        weights = weight_descriptor.load([lane, input_offset, output_offset]).reshape(
            BLOCK_K,
            BLOCK_N,
        )
        accumulator = tl.dot(values, weights, accumulator)
    output_descriptor.store(
        [lane, row_offset, output_offset],
        accumulator.to(tl.bfloat16).reshape(1, BLOCK_M, BLOCK_N),
    )


@torch.library.custom_op(
    "kaggriculture::batched_fused_relu_squared_mlp_bf16",
    mutates_args=(),
    device_types="cuda",
)
def _batched_fused_mlp(
    values: Tensor, up_weight: Tensor, down_weight: Tensor
) -> tuple[Tensor, Tensor]:
    _require_cuda_bf16(values, up_weight, down_weight)
    if values.ndim < 3 or up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("vmapped fused MLP requires one leading ensemble dimension")
    lanes = values.shape[0]
    original_shape = values.shape
    flat = values.flatten(1, -2).contiguous()
    if up_weight.shape[0] != lanes or down_weight.shape[0] != lanes:
        raise ValueError("vmapped fused MLP tensors must have the same lane count")
    hidden, width = up_weight.shape[1:]
    if width != flat.shape[-1] or down_weight.shape[1:] != (hidden, width):
        raise ValueError("vmapped fused MLP weight shapes do not match the activations")
    if width % 128 or hidden % 128:
        raise ValueError("vmapped fused MLP widths must be multiples of 128")

    rows = flat.shape[1]
    post = torch.empty((lanes, rows, hidden), device=values.device, dtype=torch.bfloat16)
    output = torch.empty((lanes, rows, width), device=values.device, dtype=torch.bfloat16)
    up_input = TensorDescriptor.from_tensor(flat, [1, 128, 128])
    up_weights = TensorDescriptor.from_tensor(up_weight.contiguous(), [1, 128, 128])
    post_descriptor = TensorDescriptor.from_tensor(post, [1, 128, 64])
    up_tiles = triton.cdiv(rows, 128) * triton.cdiv(hidden, 128)
    _batched_linear_relu_square_kernel[(lanes, up_tiles)](
        up_input,
        up_weights,
        post_descriptor,
        rows,
        hidden,
        width,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
        num_stages=2,  # pyright: ignore[reportCallIssue]
        num_warps=8,  # pyright: ignore[reportCallIssue]
    )
    down_input = TensorDescriptor.from_tensor(post, [1, 128, 128])
    down_weights = TensorDescriptor.from_tensor(down_weight.contiguous(), [1, 128, 128])
    output_descriptor = TensorDescriptor.from_tensor(output, [1, 128, 128])
    down_tiles = triton.cdiv(rows, 128) * triton.cdiv(width, 128)
    _batched_linear_kernel[(lanes, down_tiles)](
        down_input,
        down_weights,
        output_descriptor,
        rows,
        width,
        hidden,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
        num_stages=1,  # pyright: ignore[reportCallIssue]
        num_warps=8,  # pyright: ignore[reportCallIssue]
    )
    return output.view(original_shape), post


@_batched_fused_mlp.register_fake
def _fake_batched_fused_mlp(
    values: Tensor,
    up_weight: Tensor,
    _down_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    lanes = values.shape[0]
    rows = values.numel() // (lanes * values.shape[-1])
    post = values.new_empty((lanes, rows, up_weight.shape[1]))
    return values.new_empty(values.shape), post


@triton.jit
def _reduce_activation_scale_kernel(
    partial_amax,
    scale,
    partial_count: tl.constexpr,
    HEADROOM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(partial_amax + offsets, mask=offsets < partial_count, other=0.0)
    amax = tl.max(values, axis=0)
    tl.store(scale, tl.maximum(amax, 1.0e-12) * (HEADROOM / 448.0))


def reduce_mlp_activation_scale(
    partial_amax: Tensor,
    scale: Tensor,
    *,
    headroom: float = 1.8,
) -> None:
    block_size = triton.next_power_of_2(partial_amax.numel())
    _reduce_activation_scale_kernel[(1,)](
        partial_amax,
        scale,
        partial_count=partial_amax.numel(),
        HEADROOM=headroom,
        BLOCK_SIZE=block_size,
        num_stages=1,  # pyright: ignore[reportCallIssue]
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )


@triton.jit
def _quantize_transpose_down_weight_kernel(
    weight,
    output,
    next_scale,
    used_scale,
    partial_amax,
    hidden: tl.constexpr,
    width: tl.constexpr,
    width_tiles: tl.constexpr,
    tile_count: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile = tl.program_id(0)
    tile_h = tile // width_tiles
    tile_d = tile % width_tiles
    offsets_h = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offsets_h[:, None] < hidden) & (offsets_d[None, :] < width)
    values = tl.load(
        weight + offsets_h[:, None] * width + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(next_scale)
    quantized = tl.maximum(tl.minimum(values / scale, 448.0), -448.0)
    tl.store(
        output + offsets_d[:, None] * hidden + offsets_h[None, :],
        tl.trans(quantized).to(tl.float8e4nv),
        mask=tl.trans(mask),
    )
    tile_amax = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    tl.store(partial_amax + tile, tile_amax)
    if tile == 0:
        tl.store(used_scale, scale)


def quantize_transpose_mlp_down_weight(
    weight: Tensor,
    output_storage: Tensor,
    next_scale: Tensor,
    used_scale: Tensor,
    partial_amax: Tensor,
    *,
    headroom: float = 1.12,
) -> None:
    if weight.device.type != "cuda" or weight.dtype != torch.bfloat16:
        raise TypeError("FP8 down-projection refresh requires a CUDA BF16 weight")
    hidden, width = weight.shape
    if output_storage.shape != (width, hidden):
        raise ValueError("FP8 down-projection storage has the wrong transposed shape")
    block_h = block_d = 64
    width_tiles = triton.cdiv(width, block_d)
    tile_count = triton.cdiv(hidden, block_h) * width_tiles
    if partial_amax.numel() != tile_count:
        raise ValueError("FP8 down-projection amax storage has the wrong size")
    _quantize_transpose_down_weight_kernel[(tile_count,)](
        weight,
        output_storage,
        next_scale,
        used_scale,
        partial_amax,
        hidden=hidden,
        width=width,
        width_tiles=width_tiles,
        tile_count=tile_count,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_stages=1,  # pyright: ignore[reportCallIssue]
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    reduce_mlp_activation_scale(partial_amax, next_scale, headroom=headroom)


@torch.library.custom_op(
    "kaggriculture::fused_relu_squared_mlp_bf16",
    mutates_args=(),
    device_types="cuda",
)
def _fused_relu_squared_mlp_bf16(
    values: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    up_weight_bf16: Tensor,
    down_weight_bf16: Tensor,
) -> tuple[Tensor, Tensor]:
    """Opaque BF16 projection op with a native lane-batched vmap rule."""
    _require_cuda_bf16(values, up_weight_bf16, down_weight_bf16)
    original_shape = values.shape
    flat = values.reshape(-1, original_shape[-1]).contiguous()
    post = _linear_relu_square(flat, up_weight_bf16)
    assert isinstance(post, Tensor)
    output = post @ down_weight_bf16
    return output.view(original_shape), post


@_fused_relu_squared_mlp_bf16.register_fake
def _fake_fused_relu_squared_mlp_bf16(
    values: Tensor,
    _up_weight: Tensor,
    _down_weight: Tensor,
    up_weight_bf16: Tensor,
    _down_weight_bf16: Tensor,
) -> tuple[Tensor, Tensor]:
    rows = values.numel() // values.shape[-1]
    post = values.new_empty((rows, up_weight_bf16.shape[0]))
    return values.new_empty(values.shape), post


def _setup_fused_relu_squared_mlp_bf16_context(
    ctx: object,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor],
) -> None:
    values, _, _, up_weight_bf16, down_weight_bf16 = inputs
    _, post = output
    ctx.save_for_backward(  # type: ignore[attr-defined]
        values,
        up_weight_bf16,
        down_weight_bf16,
        post,
    )
    ctx.mark_non_differentiable(post)  # type: ignore[attr-defined]


@torch.library.custom_op(
    "kaggriculture::fused_relu_squared_mlp_bf16_backward",
    mutates_args=(),
    device_types="cuda",
)
def _fused_relu_squared_mlp_bf16_backward(
    gradient: Tensor,
    values: Tensor,
    up_weight_bf16: Tensor,
    down_weight_bf16: Tensor,
    post: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    flat_values = values.reshape(-1, values.shape[-1])
    flat_gradient = gradient.reshape(-1, gradient.shape[-1]).contiguous()
    down_gradient = post.T @ flat_gradient
    pre_gradient = _linear_relu_square(flat_gradient, down_weight_bf16, post)
    assert isinstance(pre_gradient, Tensor)
    up_gradient = pre_gradient.T @ flat_values
    input_gradient = pre_gradient @ up_weight_bf16
    return (
        input_gradient.view_as(values),
        up_gradient.float(),
        down_gradient.float(),
    )


@_fused_relu_squared_mlp_bf16_backward.register_fake
def _fake_fused_relu_squared_mlp_bf16_backward(
    _gradient: Tensor,
    values: Tensor,
    up_weight_bf16: Tensor,
    down_weight_bf16: Tensor,
    _post: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        values.new_empty(values.shape),
        up_weight_bf16.new_empty(up_weight_bf16.shape, dtype=torch.float32),
        down_weight_bf16.new_empty(down_weight_bf16.shape, dtype=torch.float32),
    )


def _backward_fused_relu_squared_mlp_bf16(
    ctx: object,
    gradient: Tensor,
    _post_gradient: Tensor | None,
) -> tuple[Tensor | None, ...]:
    values, up_weight_bf16, down_weight_bf16, post = ctx.saved_tensors  # type: ignore[attr-defined]
    input_gradient, up_gradient, down_gradient = _fused_relu_squared_mlp_bf16_backward(
        gradient,
        values,
        up_weight_bf16,
        down_weight_bf16,
        post,
    )
    return input_gradient, up_gradient, down_gradient, None, None


_fused_relu_squared_mlp_bf16.register_autograd(
    _backward_fused_relu_squared_mlp_bf16,
    setup_context=_setup_fused_relu_squared_mlp_bf16_context,
)


@torch.library.register_vmap(_fused_relu_squared_mlp_bf16)
def _vmap_fused_relu_squared_mlp_bf16(
    _info: object,
    in_dims: tuple[int | None, ...],
    values: Tensor,
    _up_weight: Tensor,
    _down_weight: Tensor,
    up_weight_bf16: Tensor,
    down_weight_bf16: Tensor,
) -> tuple[tuple[Tensor, Tensor], tuple[int, int]]:
    if in_dims != (0, 0, 0, 0, 0):
        raise RuntimeError("fused structured MLP vmap requires lane-major tensors")
    output, post = _batched_fused_mlp(
        values,
        up_weight_bf16,
        down_weight_bf16,
    )
    return (output, post), (0, 0)


class _FP8FusedReLUSquaredMLP(torch.autograd.Function):
    """FP8 up/down forward with the reference BF16 backward."""

    @staticmethod
    def forward(
        values: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        up_weight_bf16: Tensor,
        down_weight_bf16: Tensor,
        up_weight_f8: Tensor,
        up_weight_scale: Tensor,
        down_weight_f8: Tensor,
        down_weight_scale: Tensor,
        down_activation_scale: Tensor,
        partial_amax: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _require_cuda_bf16(values, up_weight_bf16, down_weight_bf16)
        original_shape = values.shape
        flat = values.reshape(-1, original_shape[-1]).contiguous()
        activation_amax = flat.detach().abs().amax().clamp_min(1.0e-12)
        input_scale = activation_amax.float() / _E4M3_MAX
        values_f8 = (flat.detach() / input_scale).to(torch.float8_e4m3fn)
        dequant_scale = input_scale * up_weight_scale
        projected = _linear_relu_square(
            flat,
            up_weight_bf16,
            values_f8=values_f8,
            weight_f8=up_weight_f8,
            dequant_scale=dequant_scale,
            activation_scale=down_activation_scale,
            partial_amax=partial_amax,
        )
        assert isinstance(projected, tuple)
        post, post_f8 = projected
        output = torch._scaled_mm(  # pyright: ignore[reportPrivateImportUsage]
            post_f8,
            down_weight_f8,
            out_dtype=torch.bfloat16,
            scale_a=down_activation_scale,
            scale_b=down_weight_scale,
            use_fast_accum=True,
        )
        reduce_mlp_activation_scale(partial_amax, down_activation_scale)
        return output.view(original_shape), post

    @staticmethod
    def setup_context(
        ctx: object,
        inputs: tuple[Tensor, ...],
        output: tuple[Tensor, Tensor],
    ) -> None:
        values, _, _, up_weight_bf16, down_weight_bf16, *_ = inputs
        _, post = output
        ctx.save_for_backward(  # type: ignore[attr-defined]
            values,
            up_weight_bf16,
            down_weight_bf16,
            post,
        )
        ctx.mark_non_differentiable(post)  # type: ignore[attr-defined]

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: object,
        gradient: Tensor,
        _post_gradient: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        values, up_weight_bf16, down_weight_bf16, post = ctx.saved_tensors  # type: ignore[attr-defined]
        flat_values = values.reshape(-1, values.shape[-1])
        flat_gradient = gradient.reshape(-1, gradient.shape[-1]).contiguous()
        down_gradient = post.T @ flat_gradient
        pre_gradient = _linear_relu_square(flat_gradient, down_weight_bf16, post)
        assert isinstance(pre_gradient, Tensor)
        up_gradient = pre_gradient.T @ flat_values
        input_gradient = pre_gradient @ up_weight_bf16
        return (
            input_gradient.view_as(values),
            up_gradient.float(),
            down_gradient.float(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_relu_squared_mlp(
    values: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    up_weight_bf16: Tensor,
    down_weight_bf16: Tensor,
    *,
    fp8_state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    """Apply the CUDA-native MLP; fused configurations have no portable path."""
    _require_cuda_bf16(values, up_weight_bf16, down_weight_bf16)
    if fp8_state is None:
        output, _ = _fused_relu_squared_mlp_bf16(
            values,
            up_weight,
            down_weight,
            up_weight_bf16,
            down_weight_bf16,
        )
        return output
    output, _ = _FP8FusedReLUSquaredMLP.apply(
        values,
        up_weight,
        down_weight,
        up_weight_bf16,
        down_weight_bf16,
        *fp8_state,
    )
    return output
