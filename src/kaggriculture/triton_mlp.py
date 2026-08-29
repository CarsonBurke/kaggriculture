"""Hardware-native fused ReLU-squared MLP kernels for structured models."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _linear_relu_square_kernel(
    input_descriptor,
    weight_descriptor,
    output_descriptor,
    auxiliary_descriptor,
    rows,
    outputs,
    inputs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FORWARD: tl.constexpr,
):
    program = tl.program_id(0)
    output_blocks = tl.cdiv(outputs, BLOCK_N)
    input_blocks = tl.cdiv(inputs, BLOCK_K)
    tiles = tl.cdiv(rows, BLOCK_M) * output_blocks
    output_tile = program - NUM_SMS

    for tile in tl.range(program, tiles, NUM_SMS, flatten=True):
        row_offset = (tile // output_blocks) * BLOCK_M
        output_offset = (tile % output_blocks) * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for input_block in range(input_blocks):
            input_offset = input_block * BLOCK_K
            values = input_descriptor.load([row_offset, input_offset])
            weights = weight_descriptor.load([output_offset, input_offset])
            accumulator = tl.dot(values, weights.T, accumulator)

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
            output_descriptor.store([row_output, column_output], first * first)
            output_descriptor.store([row_output, column_output + BLOCK_N // 2], second * second)
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


def _linear_relu_square(values: Tensor, weight: Tensor, post: Tensor | None = None) -> Tensor:
    rows, inputs = values.shape
    outputs, weight_inputs = weight.shape
    if inputs != weight_inputs:
        raise ValueError("fused structured MLP weight shape does not match its input")
    if values.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("fused structured MLP kernels require BF16 tensors")
    result = torch.empty((rows, outputs), device=values.device, dtype=values.dtype)
    block_m, block_n, block_k = 128, 128, 64
    forward = post is None
    auxiliary = (
        torch.empty((block_m, block_n // 2), device=values.device, dtype=values.dtype)
        if post is None
        else post
    )
    input_descriptor = TensorDescriptor.from_tensor(values, [block_m, block_k])
    weight_descriptor = TensorDescriptor.from_tensor(weight, [block_n, block_k])
    output_descriptor = TensorDescriptor.from_tensor(result, [block_m, block_n // 2])
    auxiliary_descriptor = TensorDescriptor.from_tensor(auxiliary, [block_m, block_n // 2])
    sms = torch.cuda.get_device_properties(values.device).multi_processor_count
    grid = (min(sms, triton.cdiv(rows, block_m) * triton.cdiv(outputs, block_n)),)
    _linear_relu_square_kernel[grid](
        input_descriptor,
        weight_descriptor,
        output_descriptor,
        auxiliary_descriptor,
        rows,
        outputs,
        inputs,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_SMS=sms,
        FORWARD=forward,
        num_stages=2,  # pyright: ignore[reportCallIssue]
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return result


class _FusedReLUSquaredMLP(torch.autograd.Function):
    """TMA-persistent up projection with activation-aware manual backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(
        values: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        original_shape = values.shape
        flat = values.reshape(-1, original_shape[-1]).contiguous()
        post = _linear_relu_square(flat, up_weight)
        output = post @ down_weight
        return output.view(original_shape), post

    @staticmethod
    def setup_context(
        ctx: object,
        inputs: tuple[Tensor, Tensor, Tensor],
        output: tuple[Tensor, Tensor],
    ) -> None:
        values, up_weight, down_weight = inputs
        _, post = output
        ctx.save_for_backward(values, up_weight, down_weight, post)  # type: ignore[attr-defined]
        ctx.mark_non_differentiable(post)  # type: ignore[attr-defined]

    @staticmethod
    def vmap(  # pyright: ignore[reportIncompatibleMethodOverride]
        info: object,
        in_dims: tuple[int | None, int | None, int | None],
        values: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[int, int]]:
        """Use batched matmuls when a frozen-policy ensemble vmaps this MLP."""
        batch_size = info.batch_size  # type: ignore[attr-defined]

        def batch(tensor: Tensor, dim: int | None) -> Tensor:
            if dim is None:
                return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
            return tensor.movedim(dim, 0)

        batched_values = batch(values, in_dims[0])
        batched_up = batch(up_weight, in_dims[1])
        batched_down = batch(down_weight, in_dims[2])
        original_shape = batched_values.shape
        flat = batched_values.flatten(1, -2)
        hidden = torch.relu(torch.bmm(flat, batched_up.transpose(1, 2)))
        post = hidden * hidden
        output = torch.bmm(post, batched_down)
        return (output.view(original_shape), post), (0, 0)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: object,
        gradient: Tensor,
        _post_gradient: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        values, up_weight, down_weight, post = ctx.saved_tensors  # type: ignore[attr-defined]
        flat_values = values.reshape(-1, values.shape[-1])
        flat_gradient = gradient.reshape(-1, gradient.shape[-1]).contiguous()
        down_gradient = post.T @ flat_gradient
        pre_gradient = _linear_relu_square(flat_gradient, down_weight, post)
        up_gradient = pre_gradient.T @ flat_values
        input_gradient = pre_gradient @ up_weight
        return input_gradient.view_as(values), up_gradient, down_gradient


def fused_relu_squared_mlp(
    values: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Apply the hardware-native bias-free MLP, with a portable eager fallback."""
    fused_dtypes = values.dtype == up_weight.dtype == down_weight.dtype == torch.bfloat16
    if values.device.type == "cuda" and (torch.is_autocast_enabled("cuda") or fused_dtypes):
        output, _ = _FusedReLUSquaredMLP.apply(values, up_weight, down_weight)
        return output
    hidden = torch.relu(values @ up_weight.T)
    return (hidden * hidden) @ down_weight
