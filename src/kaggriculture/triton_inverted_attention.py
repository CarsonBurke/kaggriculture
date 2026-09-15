"""CUDA inverted memory attention with bounded-size score tiles.

Query slots compete independently in each query head for every memory key;
those probabilities are then normalized over memory for each query. All
reductions and saved normalization statistics are FP32. No global Q-by-M
scores/probabilities or expanded grouped-query K/V tensors are materialized.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable


@triton.jit
def _column_lse_kernel(
    query, key, query_valid, memory_valid, column_lse,
    HQ: tl.constexpr, HKV: tl.constexpr, Q: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
    QS: tl.constexpr, KS: tl.constexpr, QMS: tl.constexpr, MMS: tl.constexpr,
    HAS_MEMORY_MASK: tl.constexpr, SCALE: tl.constexpr,
    BQ: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr, PRECISION: tl.constexpr,
):
    memory = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.program_id(1)
    batch, head = lane // HQ, lane % HQ
    kv_head = head // (HQ // HKV)
    rows, features = tl.arange(0, BQ), tl.arange(0, BD)
    qmask = tl.load(query_valid + batch * QMS[0] + rows * QMS[1], rows < Q, False)
    mmask = memory < M
    if HAS_MEMORY_MASK:
        mmask = mmask & tl.load(
            memory_valid + batch * MMS[0] + memory * MMS[1], memory < M, False
        )
    q = tl.load(
        query + batch * QS[0] + head * QS[1]
        + rows[:, None] * QS[2] + features[None, :] * QS[3],
        qmask[:, None] & (features[None, :] < D), 0,
    )
    k = tl.load(
        key + batch * KS[0] + kv_head * KS[1]
        + memory[None, :] * KS[2] + features[:, None] * KS[3],
        mmask[None, :] & (features[:, None] < D), 0,
    )
    scores = tl.dot(q, k, input_precision=PRECISION) * SCALE
    scores = tl.where(qmask[:, None] & mmask[None, :], scores, -float("inf"))
    maximum = tl.max(scores, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    total = tl.sum(tl.exp(scores - safe_maximum[None, :]), 0)
    lse = safe_maximum + tl.log(tl.where(total > 0, total, 1.0))
    tl.store(column_lse + lane * M + memory, lse, memory < M)


@triton.jit
def _output_kernel(
    query, key, value, query_valid, memory_valid, column_lse,
    output, output_fp32, row_lse,
    HQ: tl.constexpr, HKV: tl.constexpr, Q: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
    QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
    QMS: tl.constexpr, MMS: tl.constexpr, HAS_MEMORY_MASK: tl.constexpr,
    SCALE: tl.constexpr, BQ: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
    PRECISION: tl.constexpr,
):
    lane = tl.program_id(0)
    batch, head = lane // HQ, lane % HQ
    kv_head = head // (HQ // HKV)
    rows, features = tl.arange(0, BQ), tl.arange(0, BD)
    qmask = tl.load(query_valid + batch * QMS[0] + rows * QMS[1], rows < Q, False)
    q = tl.load(
        query + batch * QS[0] + head * QS[1]
        + rows[:, None] * QS[2] + features[None, :] * QS[3],
        qmask[:, None] & (features[None, :] < D), 0,
    )
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    total = tl.zeros((BQ,), tl.float32)
    numerator = tl.zeros((BQ, BD), tl.float32)
    for start in range(tl.cdiv(M, BM)):
        memory = start * BM + tl.arange(0, BM)
        mmask = memory < M
        if HAS_MEMORY_MASK:
            mmask = mmask & tl.load(
                memory_valid + batch * MMS[0] + memory * MMS[1], memory < M, False
            )
        k = tl.load(
            key + batch * KS[0] + kv_head * KS[1]
            + memory[None, :] * KS[2] + features[:, None] * KS[3],
            mmask[None, :] & (features[:, None] < D), 0,
        )
        v = tl.load(
            value + batch * VS[0] + kv_head * VS[1]
            + memory[:, None] * VS[2] + features[None, :] * VS[3],
            mmask[:, None] & (features[None, :] < D), 0,
        ).to(tl.float32)
        column = tl.load(column_lse + lane * M + memory, memory < M, 0)
        logp = tl.dot(q, k, input_precision=PRECISION) * SCALE - column[None, :]
        logp = tl.where(qmask[:, None] & mmask[None, :], logp, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logp, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        rescale = tl.exp(maximum - safe_maximum)
        weights = tl.exp(logp - safe_maximum[:, None])
        numerator = numerator * rescale[:, None] + tl.dot(
            weights, v, input_precision=PRECISION
        )
        total = total * rescale + tl.sum(weights, 1)
        maximum = next_maximum
    denominator = tl.where(total > 0, total, 1.0)
    result = numerator / denominator[:, None]
    offsets = (lane * Q + rows[:, None]) * D + features[None, :]
    valid = (rows[:, None] < Q) & (features[None, :] < D)
    tl.store(output + offsets, result, valid)
    tl.store(output_fp32 + offsets, result, valid)
    logmass = tl.where(total > 0, maximum, 0.0) + tl.log(denominator)
    tl.store(row_lse + lane * Q + rows, logmass, rows < Q)


@triton.jit
def _backward_kv_kernel(
    query, key, value, query_valid, memory_valid, gradient,
    output_fp32, column_lse, row_lse, correction, key_gradient, value_gradient,
    HQ: tl.constexpr, HKV: tl.constexpr, Q: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
    QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr, GS: tl.constexpr,
    QMS: tl.constexpr, MMS: tl.constexpr, HAS_MEMORY_MASK: tl.constexpr,
    SCALE: tl.constexpr, BQ: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
    PRECISION: tl.constexpr,
):
    memory = tl.program_id(0) * BM + tl.arange(0, BM)
    kv_lane = tl.program_id(1)
    batch, kv_head = kv_lane // HKV, kv_lane % HKV
    rows, features = tl.arange(0, BQ), tl.arange(0, BD)
    qmask = tl.load(query_valid + batch * QMS[0] + rows * QMS[1], rows < Q, False)
    mmask = memory < M
    if HAS_MEMORY_MASK:
        mmask = mmask & tl.load(
            memory_valid + batch * MMS[0] + memory * MMS[1], memory < M, False
        )
    k = tl.load(
        key + batch * KS[0] + kv_head * KS[1]
        + memory[None, :] * KS[2] + features[:, None] * KS[3],
        mmask[None, :] & (features[:, None] < D), 0,
    )
    v = tl.load(
        value + batch * VS[0] + kv_head * VS[1]
        + memory[None, :] * VS[2] + features[:, None] * VS[3],
        mmask[None, :] & (features[:, None] < D), 0,
    ).to(tl.float32)
    dk = tl.zeros((BM, BD), tl.float32)
    dv = tl.zeros((BM, BD), tl.float32)
    # One program owns each KV tile and sums its GQA heads before the sole store.
    for group in range(HQ // HKV):
        head = kv_head * (HQ // HKV) + group
        lane = batch * HQ + head
        q = tl.load(
            query + batch * QS[0] + head * QS[1]
            + rows[:, None] * QS[2] + features[None, :] * QS[3],
            qmask[:, None] & (features[None, :] < D), 0,
        )
        g = tl.load(
            gradient + batch * GS[0] + head * GS[1]
            + rows[:, None] * GS[2] + features[None, :] * GS[3],
            qmask[:, None] & (features[None, :] < D), 0,
        ).to(tl.float32)
        o = tl.load(
            output_fp32 + (lane * Q + rows[:, None]) * D + features[None, :],
            qmask[:, None] & (features[None, :] < D), 0,
        )
        delta = tl.sum(g * o, 1)
        column = tl.load(column_lse + lane * M + memory, memory < M, 0)
        logmass = tl.load(row_lse + lane * Q + rows, rows < Q, 0)
        logp = tl.dot(q, k, input_precision=PRECISION) * SCALE - column[None, :]
        valid = qmask[:, None] & mmask[None, :]
        logp = tl.where(valid, logp, -float("inf"))
        p = tl.exp(logp)
        weights = tl.exp(logp - logmass[:, None])
        # T = P*dP = (P/mass)*(G.V - G.O). Both normalizations differentiate.
        t = weights * (tl.dot(g, v, input_precision=PRECISION) - delta[:, None])
        c = tl.sum(t, 0)
        ds = t - p * c[None, :]
        dk += tl.dot(tl.trans(ds), q.to(tl.float32), input_precision=PRECISION) * SCALE
        dv += tl.dot(tl.trans(weights), g, input_precision=PRECISION)
        tl.store(correction + lane * M + memory, c, memory < M)
    offsets = (kv_lane * M + memory[:, None]) * D + features[None, :]
    valid_output = (memory[:, None] < M) & (features[None, :] < D)
    tl.store(key_gradient + offsets, dk, valid_output)
    tl.store(value_gradient + offsets, dv, valid_output)


@triton.jit
def _backward_q_kernel(
    query, key, value, query_valid, memory_valid, gradient,
    output_fp32, column_lse, row_lse, correction, query_gradient,
    HQ: tl.constexpr, HKV: tl.constexpr, Q: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
    QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr, GS: tl.constexpr,
    QMS: tl.constexpr, MMS: tl.constexpr, HAS_MEMORY_MASK: tl.constexpr,
    SCALE: tl.constexpr, BQ: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
    PRECISION: tl.constexpr,
):
    lane = tl.program_id(0)
    batch, head = lane // HQ, lane % HQ
    kv_head = head // (HQ // HKV)
    rows, features = tl.arange(0, BQ), tl.arange(0, BD)
    qmask = tl.load(query_valid + batch * QMS[0] + rows * QMS[1], rows < Q, False)
    q = tl.load(
        query + batch * QS[0] + head * QS[1]
        + rows[:, None] * QS[2] + features[None, :] * QS[3],
        qmask[:, None] & (features[None, :] < D), 0,
    )
    g = tl.load(
        gradient + batch * GS[0] + head * GS[1]
        + rows[:, None] * GS[2] + features[None, :] * GS[3],
        qmask[:, None] & (features[None, :] < D), 0,
    ).to(tl.float32)
    o = tl.load(
        output_fp32 + (lane * Q + rows[:, None]) * D + features[None, :],
        qmask[:, None] & (features[None, :] < D), 0,
    )
    delta = tl.sum(g * o, 1)
    logmass = tl.load(row_lse + lane * Q + rows, rows < Q, 0)
    dq = tl.zeros((BQ, BD), tl.float32)
    for start in range(tl.cdiv(M, BM)):
        memory = start * BM + tl.arange(0, BM)
        mmask = memory < M
        if HAS_MEMORY_MASK:
            mmask = mmask & tl.load(
                memory_valid + batch * MMS[0] + memory * MMS[1], memory < M, False
            )
        k = tl.load(
            key + batch * KS[0] + kv_head * KS[1]
            + memory[None, :] * KS[2] + features[:, None] * KS[3],
            mmask[None, :] & (features[:, None] < D), 0,
        )
        v = tl.load(
            value + batch * VS[0] + kv_head * VS[1]
            + memory[None, :] * VS[2] + features[:, None] * VS[3],
            mmask[None, :] & (features[:, None] < D), 0,
        ).to(tl.float32)
        column = tl.load(column_lse + lane * M + memory, memory < M, 0)
        c = tl.load(correction + lane * M + memory, memory < M, 0)
        logp = tl.dot(q, k, input_precision=PRECISION) * SCALE - column[None, :]
        logp = tl.where(qmask[:, None] & mmask[None, :], logp, -float("inf"))
        p = tl.exp(logp)
        weights = tl.exp(logp - logmass[:, None])
        t = weights * (tl.dot(g, v, input_precision=PRECISION) - delta[:, None])
        ds = t - p * c[None, :]
        dq += tl.dot(ds, tl.trans(k).to(tl.float32), input_precision=PRECISION) * SCALE
    offsets = (lane * Q + rows[:, None]) * D + features[None, :]
    tl.store(query_gradient + offsets, dq, (rows[:, None] < Q) & (features[None, :] < D))


def _check_inputs(
    query: Tensor, key: Tensor, value: Tensor,
    query_valid: Tensor, memory_valid: Tensor | None, scale: float,
) -> None:
    if query.device.type != "cuda":
        raise ValueError("Inverted attention requires CUDA; no CPU fallback is provided")
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("Inverted attention requires BF16, FP16, or FP32 tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Inverted attention requires [B,H,Q,D] queries and [B,Hkv,M,D] keys/values")
    if key.device != query.device or value.device != query.device:
        raise ValueError("Inverted attention queries, keys, and values must share a CUDA device")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("Inverted attention queries, keys, and values must share a dtype")
    batch, heads, slots, width = query.shape
    if any(size <= 0 for size in query.shape) or any(size <= 0 for size in key.shape):
        raise ValueError("Inverted attention dimensions must be positive")
    if slots > 128 or width > 128:
        raise ValueError("Inverted attention supports at most 128 query slots and 128 head features")
    if key.shape != value.shape or key.shape[0] != batch or key.shape[3] != width:
        raise ValueError("Inverted attention keys/values must match each other and query batch/head width")
    if heads % key.shape[1]:
        raise ValueError("Inverted attention query heads must be divisible by KV heads")
    if query_valid.shape != (batch, slots):
        raise ValueError("Inverted attention query_valid must have shape [B,Q]")
    if query_valid.device != query.device or query_valid.dtype != torch.bool:
        raise TypeError("Inverted attention query_valid must be boolean on the query CUDA device")
    if memory_valid is not None:
        if memory_valid.shape != (batch, key.shape[2]):
            raise ValueError("Inverted attention memory_valid must have shape [B,M]")
        if memory_valid.device != query.device or memory_valid.dtype != torch.bool:
            raise TypeError("Inverted attention memory_valid must be boolean on the query CUDA device")
    if not math.isfinite(scale):
        raise ValueError("Inverted attention scale must be finite")


def _launch_options(
    query: Tensor, key: Tensor, query_valid: Tensor, memory_valid: Tensor | None, scale: float,
) -> dict:
    block_q = max(16, triton.next_power_of_2(query.shape[2]))
    block_d = max(16, triton.next_power_of_2(query.shape[3]))
    return {
        "HQ": query.shape[1], "HKV": key.shape[1], "Q": query.shape[2],
        "M": key.shape[2], "D": query.shape[3],
        "QS": query.stride(), "KS": key.stride(), "QMS": query_valid.stride(),
        "MMS": memory_valid.stride() if memory_valid is not None else (0, 0),
        "HAS_MEMORY_MASK": memory_valid is not None, "SCALE": scale,
        "BQ": block_q, "BM": 32, "BD": block_d,
        # FP32 oracle calls use IEEE products; three TF32 products preserve FP32
        # probability accuracy on tensor cores for the BF16/FP16 training path.
        "PRECISION": "ieee" if query.dtype == torch.float32 else "tf32x3",
        "num_warps": 8 if block_q * block_d >= 8192 else 4,
    }


@torch.library.custom_op("kaggriculture::inverted_attention", mutates_args=(), device_types="cuda")
def _inverted_attention(
    query: Tensor, key: Tensor, value: Tensor,
    query_valid: Tensor, memory_valid: Tensor | None, scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _check_inputs(query, key, value, query_valid, memory_valid, scale)
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    output_fp32 = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    column_lse = torch.empty(
        (query.shape[0], query.shape[1], key.shape[2]), dtype=torch.float32, device=query.device
    )
    row_lse = torch.empty(query.shape[:3], dtype=torch.float32, device=query.device)
    options = _launch_options(query, key, query_valid, memory_valid, scale)
    lanes = query.shape[0] * query.shape[1]
    _column_lse_kernel[(triton.cdiv(key.shape[2], options["BM"]), lanes)](
        query, key, query_valid, memory_valid, column_lse, **options
    )
    _output_kernel[(lanes,)](
        query, key, value, query_valid, memory_valid, column_lse,
        output, output_fp32, row_lse, VS=value.stride(), **options
    )
    return output, output_fp32, column_lse, row_lse


@_inverted_attention.register_fake
def _fake_inverted_attention(
    query: Tensor, key: Tensor, value: Tensor,
    query_valid: Tensor, memory_valid: Tensor | None, scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _check_inputs(query, key, value, query_valid, memory_valid, scale)
    return (
        torch.empty(query.shape, dtype=query.dtype, device=query.device),
        torch.empty(query.shape, dtype=torch.float32, device=query.device),
        torch.empty(
            (query.shape[0], query.shape[1], key.shape[2]), dtype=torch.float32, device=query.device
        ),
        torch.empty(query.shape[:3], dtype=torch.float32, device=query.device),
    )


@torch.library.custom_op(
    "kaggriculture::inverted_attention_backward", mutates_args=(), device_types="cuda"
)
def _inverted_attention_backward(
    query: Tensor, key: Tensor, value: Tensor,
    query_valid: Tensor, memory_valid: Tensor | None, gradient: Tensor,
    output_fp32: Tensor, column_lse: Tensor, row_lse: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    query_gradient = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    key_gradient = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    value_gradient = torch.empty(value.shape, dtype=value.dtype, device=value.device)
    correction = torch.empty(column_lse.shape, dtype=torch.float32, device=query.device)
    options = _launch_options(query, key, query_valid, memory_valid, scale)
    _backward_kv_kernel[(
        triton.cdiv(key.shape[2], options["BM"]), key.shape[0] * key.shape[1]
    )](
        query, key, value, query_valid, memory_valid, gradient,
        output_fp32, column_lse, row_lse, correction, key_gradient, value_gradient,
        VS=value.stride(), GS=gradient.stride(), **options
    )
    _backward_q_kernel[(query.shape[0] * query.shape[1],)](
        query, key, value, query_valid, memory_valid, gradient,
        output_fp32, column_lse, row_lse, correction, query_gradient,
        VS=value.stride(), GS=gradient.stride(), **options
    )
    return query_gradient, key_gradient, value_gradient


@_inverted_attention_backward.register_fake
def _fake_inverted_attention_backward(
    query: Tensor, key: Tensor, value: Tensor,
    query_valid: Tensor, memory_valid: Tensor | None, gradient: Tensor,
    output_fp32: Tensor, column_lse: Tensor, row_lse: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty(query.shape, dtype=query.dtype, device=query.device),
        torch.empty(key.shape, dtype=key.dtype, device=key.device),
        torch.empty(value.shape, dtype=value.dtype, device=value.device),
    )


def _setup_context(
    ctx: object,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, float],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    query, key, value, query_valid, memory_valid, scale = inputs
    _, output_fp32, column_lse, row_lse = output
    ctx.save_for_backward(  # type: ignore[attr-defined]
        query, key, value, query_valid, memory_valid, output_fp32, column_lse, row_lse
    )
    ctx.scale = scale  # type: ignore[attr-defined]
    ctx.mark_non_differentiable(output_fp32, column_lse, row_lse)  # type: ignore[attr-defined]


@once_differentiable
def _backward(
    ctx: object, gradient: Tensor,
    _output_fp32_gradient: Tensor | None,
    _column_lse_gradient: Tensor | None,
    _row_lse_gradient: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None, None]:
    query, key, value, query_valid, memory_valid, output_fp32, column_lse, row_lse = (
        ctx.saved_tensors  # type: ignore[attr-defined]
    )
    dq, dk, dv = _inverted_attention_backward(
        query, key, value, query_valid, memory_valid, gradient,
        output_fp32, column_lse, row_lse, ctx.scale,  # type: ignore[attr-defined]
    )
    needs = ctx.needs_input_grad  # type: ignore[attr-defined]
    return dq if needs[0] else None, dk if needs[1] else None, dv if needs[2] else None, None, None, None


_inverted_attention.register_autograd(_backward, setup_context=_setup_context)


def inverted_attention(
    query: Tensor, key: Tensor, value: Tensor,
    query_valid: Tensor, memory_valid: Tensor | None, scale: float,
) -> Tensor:
    """Apply query-competitive attention, then normalize each row over memory.

    Queries have shape [B,Hq,Q,D], keys/values [B,Hkv,M,D], and Hq must be
    divisible by Hkv. Each query head competes independently, using KV head
    h//(Hq/Hkv). Query and optional memory masks are boolean [B,Q] / [B,M].
    Invalid queries and all-masked rows produce exactly zero outputs and
    gradients; masked memory does not contribute. All tensor strides, including
    expanded zero strides, are supported without copying inputs.

    Requires same-device CUDA BF16/FP16/FP32 queries, keys, and values, positive
    dimensions, Q<=128, D<=128, and finite scale. M has no fixed upper bound.
    Output and first-order input gradients have their input dtype and contiguous
    layout. FP32 accumulators and saved unrounded outputs avoid differentiating
    a prematurely rounded quotient. FP32 calls use IEEE matrix products;
    low-precision calls use native input products and TF32x3 for FP32 products.
    Higher-order derivatives and vmap are not supported. No CPU/eager fallback.
    """
    _check_inputs(query, key, value, query_valid, memory_valid, scale)
    return _inverted_attention(query, key, value, query_valid, memory_valid, scale)[0]
