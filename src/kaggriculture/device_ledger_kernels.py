"""Exact integer prefix scans with an explicit compiled CUDA kernel boundary.

Inductor's scan/reduction fusion can reuse an indirect-gather temporary outside
its loop scope (observed in the market quantity mask). This opaque scan keeps
all ledger work on GPU while preventing that invalid fusion. It also guarantees
int64 accumulation, independent of the surrounding neural compute dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _scan_kernel(
    inputs,
    output,
    WIDTH: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    COLUMN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Widen before multiplication: arbitrary tensor strides can exceed the
    # int32 address range even when the row and column counts are small.
    row = tl.program_id(0).to(tl.int64)
    remaining = row
    offset = tl.full((), 0, tl.int64)
    for dimension in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (remaining % SHAPE[dimension]) * STRIDES[dimension]
        remaining = remaining // SHAPE[dimension]
    columns = tl.arange(0, BLOCK)
    column_offsets = columns.to(tl.int64) * COLUMN_STRIDE
    values = tl.load(inputs + offset + column_offsets, columns < WIDTH, 0)
    total = tl.cumsum(values.to(tl.int64), axis=0)
    tl.store(output + row * WIDTH + columns, total, columns < WIDTH)


def _validate(inputs: Tensor) -> None:
    if inputs.device.type != "cuda" or inputs.dtype != torch.int64:
        raise ValueError("exact ledger scan requires CUDA int64 inputs")
    if inputs.ndim < 1 or not 1 <= inputs.shape[-1] <= 100:
        raise ValueError("exact ledger scan width must be in 1..100")


@torch.library.custom_op("kaggriculture::ledger_cumsum", mutates_args=(), device_types="cuda")
def ledger_cumsum(inputs: Tensor) -> Tensor:
    _validate(inputs)
    output = torch.empty(inputs.shape, device=inputs.device, dtype=torch.int64)
    width = inputs.shape[-1]
    rows = inputs.numel() // width
    if rows:
        _scan_kernel[(rows,)](
            inputs,
            output,
            WIDTH=width,
            SHAPE=tuple(inputs.shape[:-1]),
            STRIDES=tuple(inputs.stride()[:-1]),
            COLUMN_STRIDE=inputs.stride(-1),
            BLOCK=triton.next_power_of_2(width),
            num_warps=1 if width <= 32 else 4,
        )
    return output


@ledger_cumsum.register_fake
def _fake_scan(inputs: Tensor) -> Tensor:
    _validate(inputs)
    return torch.empty(inputs.shape, device=inputs.device, dtype=torch.int64)


@torch.library.register_vmap(ledger_cumsum)
def _vmap_scan(info: object, in_dims: tuple[int | None, ...], inputs: Tensor) -> tuple[Tensor, int]:
    dim = in_dims[0]
    values = (
        inputs.unsqueeze(0).expand(info.batch_size, *inputs.shape)
        if dim is None
        else inputs.movedim(dim, 0)
    )
    return ledger_cumsum(values), 0
