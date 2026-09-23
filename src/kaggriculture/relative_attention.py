"""Own-unit/own-tile relative bias without imposing coordinates on other relations."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from kaggriculture.constants import BOARD_SIZE, MAX_UNITS
from kaggriculture.structured import _fused_attention, _pad_head_width
from kaggriculture.tokens import TILE_COUNT


def unit_tile_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_valid: Tensor | None,
    bias: Tensor,
    positions: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Apply a shared per-head displacement table only to the own-farm relation.

    Training fuses table lookup into score computation. No-grad collection uses
    SDPA's vmap-compatible path for frozen opponent ensembles; its much smaller
    rollout batch can materialize the bias without retaining backward storage.
    """
    batch, heads, queries, head_dim = query.shape
    keys = key.shape[-2]
    if not torch.is_grad_enabled() or not query.is_cuda:
        q = torch.arange(queries, device=query.device)
        k = torch.arange(keys, device=query.device)
        xy = positions[:, q.clamp_max(MAX_UNITS - 1)]
        tile = k.clamp_max(TILE_COUNT - 1)
        dx = tile[None, None, :] % BOARD_SIZE - xy[..., 0, None] + BOARD_SIZE - 1
        dy = tile[None, None, :] // BOARD_SIZE - xy[..., 1, None] + BOARD_SIZE - 1
        index = dy * (2 * BOARD_SIZE - 1) + dx
        offset = bias[index].permute(0, 3, 1, 2)
        spatial = (q[:, None] < MAX_UNITS) & (k[None, :] < TILE_COUNT)
        offset = torch.where(spatial[None, None], offset, 0.0)
        if key_valid is not None:
            offset = offset.masked_fill(~key_valid[:, None, None, :], -float("inf"))
        return _fused_attention(
            query,
            key,
            value,
            offset.to(query.dtype),
            enable_gqa=heads != key.shape[1],
            scale=scale,
        )

    width = (head_dim + 7) // 8 * 8
    query, key, value = (_pad_head_width(tensor, width) for tensor in (query, key, value))
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    query = query.reshape(batch, kv_heads, groups * queries, width)

    def score_mod(score, b, h, q, k):
        unit = (q % queries).clamp_max(MAX_UNITS - 1)
        tile = k.clamp_max(TILE_COUNT - 1)
        dx = tile % BOARD_SIZE - positions[b, unit, 0] + BOARD_SIZE - 1
        dy = tile // BOARD_SIZE - positions[b, unit, 1] + BOARD_SIZE - 1
        index = dy * (2 * BOARD_SIZE - 1) + dx
        head = h * groups + q // queries
        offset = bias[index, head]
        score = score + torch.where((q % queries < MAX_UNITS) & (k < TILE_COUNT), offset, 0.0)
        if key_valid is not None:
            score = torch.where(key_valid[b, k], score, -float("inf"))
        return score

    attended = flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        scale=scale,
        kernel_options={"BACKEND": "TRITON", "BLOCK_M": 64, "BLOCK_N": 64},
    )
    return attended.reshape(batch, heads, queries, width)[..., :head_dim]
