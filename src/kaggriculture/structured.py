"""Structured farm transformer: the VIT_PLAN target architecture.

Replaces the convolutional trunk with semantic entity tokens (tiles, units,
economy) fused by a latent transformer core:

  tokenize -> shared farm-local blocks per farm -> opponent summary latents
  -> global latents cross-attend all context -> latent core -> unit / market
  / value decoders.

Output contracts are identical to the convolutional ``FarmActor`` and
``DistributionalCritic``: the sampler, rollout staging, replay, and frozen
ensembles consume both architectures interchangeably.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import BOARD_SIZE, CROPS, MAX_MARKET_ORDERS, MAX_UNITS, PRODUCTS
from kaggriculture.model import (
    ActorOutput,
    AxialRotaryEmbedding,
    ReluSquared,
    RMSNorm,
    _sdpa_inputs,
    factored_quantity_logits,
    initialize_policy_heads,
)
from kaggriculture.tokens import (
    CROP_PRIVATE_FIELDS,
    CROP_TOKEN_FIELDS,
    FARM_TOKEN_FIELDS,
    N_TILE_CONTINUOUS,
    N_UNIT_CONTINUOUS,
    PRODUCT_PRIVATE_FIELDS,
    PRODUCT_TOKEN_FIELDS,
    QUADRANT_COUNT,
    TILE_COUNT,
    TILE_KINDS,
    TILE_OCCUPANTS,
    TOWN_TOKEN_FIELDS,
    UNIT_ROLES,
    UNIT_TILE_GATHERS,
)
from kaggriculture.triton_mlp import (
    fused_relu_squared_mlp,
    quantize_transpose_mlp_down_weight,
)


@dataclass(frozen=True)
class StructuredConfig:
    """Target-architecture hyperparameters from VIT_PLAN."""

    model_dim: int = 128
    attention_heads: int = 4
    attention_kv_heads: int = 2
    ffn_multiplier: int = 2
    farm_blocks: int = 2
    opponent_latents: int = 8
    latents: int = 32
    core_layers: int = 8
    quantity_rank: int = 32
    global_refresh_layers: tuple[int, ...] = ()
    global_refresh_context: str = "none"
    input_reinject_layers: tuple[int, ...] = ()
    core_skip_source: int = 0
    core_skip_target: int = 0
    zero_init_branches: bool = False
    mudd_lite: bool = False
    fuse_market_decoder: bool = False
    fuse_unit_decoder: bool = False
    split_clock_token: bool = False
    global_modulation: bool = False
    fused_mlp: bool = False
    critic_core_layers: int = 0
    critic_latents: int = 0
    value_atoms: int = 101
    value_min: float = -2.2
    value_max: float = 2.2
    value_sigma_ratio: float = 0.75

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_refresh_layers", tuple(self.global_refresh_layers))
        object.__setattr__(self, "input_reinject_layers", tuple(self.input_reinject_layers))
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.attention_heads <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("attention_heads must evenly divide model_dim")
        if (self.model_dim // self.attention_heads) % 4:
            raise ValueError("attention head width must be divisible by 4 for axial RoPE")
        if (
            self.attention_kv_heads <= 0
            or self.attention_kv_heads > self.attention_heads
            or self.attention_heads % self.attention_kv_heads
        ):
            raise ValueError("attention KV heads must positively divide query heads")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if self.fused_mlp and (self.model_dim % 128 or self.model_dim * self.ffn_multiplier % 256):
            raise ValueError(
                "fused MLP requires model width divisible by 128 and hidden width by 256"
            )
        if self.farm_blocks <= 0:
            raise ValueError("farm_blocks must be positive")
        if self.opponent_latents <= 0:
            raise ValueError("opponent_latents must be positive")
        if self.latents <= 0:
            raise ValueError("latents must be positive")
        if self.core_layers <= 0:
            raise ValueError("core_layers must be positive")
        if self.quantity_rank <= 0:
            raise ValueError("quantity_rank must be positive")
        for name, layers in (
            ("global_refresh_layers", self.global_refresh_layers),
            ("input_reinject_layers", self.input_reinject_layers),
        ):
            if tuple(sorted(set(layers))) != layers:
                raise ValueError(f"{name} must be sorted and unique")
            if any(layer < 1 or layer > self.core_layers for layer in layers):
                raise ValueError(f"{name} must name layers in 1..core_layers")
        if self.global_refresh_context not in {"none", "economy", "all"}:
            raise ValueError("global_refresh_context must be none, economy, or all")
        if bool(self.global_refresh_layers) != (self.global_refresh_context != "none"):
            raise ValueError("global refresh layers and context must be enabled together")
        skip_enabled = bool(self.core_skip_source or self.core_skip_target)
        if skip_enabled and not (
            1 <= self.core_skip_source < self.core_skip_target <= self.core_layers
        ):
            raise ValueError("core skip must name an ordered source and target layer")
        if self.mudd_lite and self.core_layers < 6:
            raise ValueError("MUDD-lite needs at least six core layers")
        for name, value in (
            ("critic_core_layers", self.critic_core_layers),
            ("critic_latents", self.critic_latents),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.value_atoms < 2:
            raise ValueError("value_atoms must be at least 2")
        if not math.isfinite(self.value_min) or not math.isfinite(self.value_max):
            raise ValueError("value support bounds must be finite")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not math.isfinite(self.value_sigma_ratio) or self.value_sigma_ratio <= 0:
            raise ValueError("value_sigma_ratio must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredInputs(NamedTuple):
    """Batched token tensors for one player viewpoint (own farm first)."""

    tile_categorical: Tensor  # [B, 2 * TILE_COUNT, 6] int64
    tile_continuous: Tensor  # [B, 2 * TILE_COUNT, N_TILE_CONTINUOUS]
    unit_categorical: Tensor  # [B, MAX_UNITS, 4] int64
    unit_continuous: Tensor  # [B, MAX_UNITS, N_UNIT_CONTINUOUS]
    unit_active: Tensor  # [B, MAX_UNITS] bool
    unit_tile_gather: Tensor  # [B, MAX_UNITS, 5] int64 into own-farm tiles
    unit_tile_gather_valid: Tensor  # [B, MAX_UNITS, 5] bool
    products: Tensor  # [B, len(PRODUCTS), product feature width]
    crops: Tensor  # [B, len(CROPS), crop feature width]
    farms: Tensor  # [B, 2, len(FARM_TOKEN_FIELDS)]
    town: Tensor  # [B, len(TOWN_TOKEN_FIELDS)]


class CriticExtras(NamedTuple):
    """Opponent-side critic tensors stacked alongside StructuredInputs."""

    products: Tensor  # [B, len(PRODUCTS), len(PRODUCT_PRIVATE_FIELDS)]
    crops: Tensor  # [B, len(CROPS), len(CROP_PRIVATE_FIELDS)]
    unit_categorical: Tensor  # [B, MAX_UNITS, N_UNIT_CATEGORICAL] int64
    unit_continuous: Tensor  # [B, MAX_UNITS, N_UNIT_CONTINUOUS]
    unit_active: Tensor  # [B, MAX_UNITS] bool


def stack_structured(
    rows: list,
    device: torch.device | None = None,
) -> tuple[StructuredInputs, CriticExtras | None]:
    """Batch ``StructuredObservation`` bundles into model-ready tensors.

    Integer index arrays upcast to int64 for embedding lookups; continuous
    features keep their staged dtype (the embedders cast per forward). Critic
    extras are returned only when every row carries them.
    """
    if not rows:
        raise ValueError("cannot stack an empty batch")

    def stacked(field: str, dtype: torch.dtype | None = None) -> Tensor:
        tensor = torch.from_numpy(np.stack([getattr(row, field) for row in rows]))
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    inputs = StructuredInputs(
        tile_categorical=stacked("tile_categorical", torch.int64),
        tile_continuous=stacked("tile_continuous"),
        unit_categorical=stacked("unit_categorical", torch.int64),
        unit_continuous=stacked("unit_continuous"),
        unit_active=stacked("unit_active"),
        unit_tile_gather=stacked("unit_tile_gather", torch.int64),
        unit_tile_gather_valid=stacked("unit_tile_gather_valid"),
        products=stacked("products"),
        crops=stacked("crops"),
        farms=stacked("farms"),
        town=stacked("town"),
    )
    with_extras = sum(row.critic_products is not None for row in rows)
    if with_extras == 0:
        return inputs, None
    if with_extras != len(rows):
        raise ValueError("critic extras must be present on every row or none")
    extras = CriticExtras(
        products=stacked("critic_products"),
        crops=stacked("critic_crops"),
        unit_categorical=stacked("opponent_unit_categorical", torch.int64),
        unit_continuous=stacked("opponent_unit_continuous"),
        unit_active=stacked("opponent_unit_active"),
    )
    return inputs, extras


class GatedResidual(nn.Module):
    """Near-identity residual: branch output scaled by a learned channel gate."""

    def __init__(self, width: int, initial: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.full((width,), initial))

    def forward(self, residual: Tensor, branch: Tensor) -> Tensor:
        compute_dtype = branch.dtype
        return residual.to(compute_dtype) + self.gate.to(compute_dtype) * branch


#: Score elements -- batch * heads * query_tokens * key_tokens -- at or above
#: which attention runs through a fused SDPA kernel rather than an explicit
#: matmul/softmax/matmul the surrounding Inductor graph can fuse.
#:
#: The shipped path sent every attention to `scaled_dot_product_attention`, and
#: at this architecture's shapes that was the update's single largest kernel.
#: A compiled production minibatch (4040 rows) spent 32.9 ms of its 87.4 ms
#: actor forward+backward inside `_flash_attention_backward` against a 3.0 ms
#: forward: an 11x backward-to-forward ratio, where 2-3x is ordinary. Flash
#: amortizes its fixed cost over long sequences, and the longest context here
#: is 141 tokens.
#:
#: Isolated compiled forward+backward, bf16, RTX 5090, median of 15, at the
#: geometries the trunk and decoders actually run (`probe_attention*`):
#:
#:     geometry      batch  scores      flash   cuDNN(pad)   explicit
#:     core 32x32      4040   16.5M   1.735 ms    1.026 ms   0.793 ms
#:     core 32x32      8080   33.1M   3.855 ms    2.061 ms   2.040 ms
#:     latent 32x141   4040   72.9M   3.553 ms    2.564 ms   4.670 ms
#:     farm 100x100    2048   81.9M   1.622 ms    1.483 ms   1.868 ms
#:     farm 100x100    8080  323.2M  10.091 ms    7.010 ms  20.117 ms
#:
#: Flash is never the fastest cell. Explicit attention wins below roughly 32M
#: score elements and loses above it, because its cost is the materialized
#: score matrix while a fused kernel's is the fixed per-launch overhead. 32M
#: sits between the last cell explicit wins (33.1M, a tie) and the first it
#: clearly loses (72.9M), and it also bounds the materialized softmax.
#:
#: Both branches are exact rewrites of the same function, not approximations:
#: the explicit path is the algorithm SDPA implements, and the fused path zero-
#: pads the head width, which contributes nothing to QK^T and returns zeros in
#: the padded output channels that are then sliced away. Neither is bitwise
#: identical to the other -- `update_replay_parity` bounds that drift, and both
#: collection and update read this one implementation, so they move together.
#: Measured, the drift is nowhere near the bound: `update_replay_max_kl` moves
#: 3.12e-7 to 3.14e-7 against a 5e-3 gate, with every tail fraction still zero.
#:
#: End to end at production settings (`scripts/benchmark_ppo_iteration.py`,
#: 128 self-play + 64 league games, 2 epochs, 4 critic epochs, minibatch 4096,
#: median of five steady repeats, `artifacts/benchmarks/attn-*.jsonl`):
#:
#:                       rollout    update     total   iterations/hour
#:     before             18.540    41.617    60.272        59.73
#:     after              19.479    36.181    55.911        64.39
#:
#: The update, which is entirely compiled, takes the whole 13.1% the isolated
#: measurement predicted. The rollout runs eager and gives 5.1% back, which is
#: the same finding read from the other side: unfused, the explicit path is six
#: kernel launches where flash is one. The two do not cancel -- collection is
#: under a third of an iteration -- but they are why this is written as a
#: property of the geometry rather than of the model.
EXPLICIT_ATTENTION_SCORE_LIMIT = 32 << 20

#: Fused SDPA kernels reject a head width that is not a multiple of eight:
#: cuDNN and the memory-efficient backend refuse outright, and a masked call at
#: such a width has no fused kernel at all and silently decomposes to the math
#: backend. Production runs `model_dim` 80 over four heads, so its 20-wide
#: heads take exactly that decomposition today. Padding is a no-op for an
#: already-aligned width.
_FUSED_ATTENTION_HEAD_MULTIPLE = 8

#: cuDNN first: it is the fastest admissible backend at every measured geometry
#: above, and the only fused one that accepts a mask at these head widths.
#: Flash and math follow so an unsupported shape degrades instead of raising.
_FUSED_ATTENTION_BACKENDS = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.MATH,
)


def _expand_key_value(key: Tensor, value: Tensor, repeats: int) -> tuple[Tensor, Tensor]:
    """Materialize grouped-query key/value heads for explicit attention."""
    if repeats == 1:
        return key, value
    return key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)


def _explicit_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    *,
    repeats: int,
    scale: float,
) -> Tensor:
    """Attention as an explicit matmul/softmax/matmul, fusable by Inductor.

    The softmax reduces in fp32 exactly as every fused backend does
    internally, so this trades no precision for its speed.
    """
    key, value = _expand_key_value(key, value, repeats)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attention_mask is not None:
        valid_rows = attention_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~attention_mask, -torch.inf)
        # Avoid NaN softmax inputs for query rows whose entire context is
        # masked. SDPA defines both their output and gradient as zero.
        scores = torch.where(valid_rows, scores, 0.0)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if attention_mask is not None:
        probabilities = torch.where(attention_mask, probabilities, 0.0)
    return torch.matmul(probabilities.to(value.dtype), value)


def _pad_head_width(tensor: Tensor, width: int) -> Tensor:
    extra = width - tensor.shape[-1]
    return tensor if extra == 0 else nn.functional.pad(tensor, (0, extra))


def _fused_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    *,
    enable_gqa: bool,
    scale: float,
) -> Tensor:
    """One fused SDPA call, with the head width padded into kernel support.

    `scale` is stated rather than defaulted: SDPA derives its default from the
    padded width, which is not the width this attention is defined over.
    """
    head_dim = query.shape[-1]
    remainder = head_dim % _FUSED_ATTENTION_HEAD_MULTIPLE
    if remainder:
        width = head_dim + _FUSED_ATTENTION_HEAD_MULTIPLE - remainder
        query, key, value = (_pad_head_width(t, width) for t in (query, key, value))
    with sdpa_kernel(list(_FUSED_ATTENTION_BACKENDS), set_priority=True):
        attended = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    return attended[..., :head_dim] if remainder else attended


class Attention(nn.Module):
    """QK-normalized attention over an explicit context (self or cross)."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.kv_heads = config.attention_kv_heads
        self.head_dim = config.model_dim // self.heads
        self.query = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.key_value = nn.Linear(config.model_dim, 2 * self.kv_heads * self.head_dim, bias=False)
        self.query_norm = RMSNorm(self.head_dim)
        self.key_norm = RMSNorm(self.head_dim)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)
        if config.zero_init_branches:
            nn.init.zeros_(self.output.weight)

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        *,
        query_rotation: tuple[Tensor, Tensor] | None = None,
        key_rotation: tuple[Tensor, Tensor] | None = None,
        context_valid: Tensor | None = None,
    ) -> Tensor:
        batch, query_tokens, width = queries.shape
        key_tokens = context.shape[1]
        query = (
            self.query(queries).view(batch, query_tokens, self.heads, self.head_dim).transpose(1, 2)
        )
        key, value = (
            self.key_value(context)
            .view(batch, key_tokens, 2, self.kv_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        query = self.query_norm(query)
        key = self.key_norm(key)
        if query_rotation is not None:
            cosine, sine = query_rotation
            query = query * cosine.to(query.dtype) + AxialRotaryEmbedding._rotate_pairs(
                query
            ) * sine.to(query.dtype)
        if key_rotation is not None:
            cosine, sine = key_rotation
            key = key * cosine.to(key.dtype) + AxialRotaryEmbedding._rotate_pairs(key) * sine.to(
                key.dtype
            )
        attention_mask = None
        if context_valid is not None:
            attention_mask = context_valid.view(batch, 1, 1, key_tokens)
        query, key, value = _sdpa_inputs(query, key, value)
        scale = self.head_dim**-0.5
        scores = batch * self.heads * query_tokens * key_tokens
        if scores < EXPLICIT_ATTENTION_SCORE_LIMIT:
            attended = _explicit_attention(
                query,
                key,
                value,
                attention_mask,
                repeats=self.heads // self.kv_heads,
                scale=scale,
            )
        else:
            attended = _fused_attention(
                query,
                key,
                value,
                attention_mask,
                enable_gqa=self.heads != self.kv_heads,
                scale=scale,
            )
        attended = attended.to(dtype=queries.dtype)
        return self.output(attended.transpose(1, 2).reshape(batch, query_tokens, width))


class FeedForward(nn.Module):
    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        hidden = config.model_dim * config.ffn_multiplier
        self.input = nn.Linear(config.model_dim, hidden)
        self.activation = ReluSquared()
        self.output = nn.Linear(hidden, config.model_dim)
        if config.zero_init_branches:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.activation(self.input(inputs)))


class FusedFeedForward(nn.Module):
    """CUDA-native BF16/FP8 MLP with explicitly refreshed projection copies."""

    _up_weight_bf16: Tensor
    _down_weight_bf16: Tensor
    _up_weight_f8: Tensor
    _up_weight_scale: Tensor
    _down_weight_f8_storage: Tensor
    _down_weight_scale: Tensor
    _down_weight_next_scale: Tensor
    _down_activation_scale: Tensor
    _down_partial_amax: Tensor
    _down_weight_partial_amax: Tensor

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        hidden = config.model_dim * config.ffn_multiplier
        up_weight = torch.empty(hidden, config.model_dim)
        down_weight = torch.empty(hidden, config.model_dim)
        nn.init.kaiming_uniform_(up_weight, a=5**0.5)
        nn.init.kaiming_uniform_(down_weight.T, a=5**0.5)
        if config.zero_init_branches:
            nn.init.zeros_(down_weight)
        self.up_weight = nn.Parameter(up_weight)
        self.down_weight = nn.Parameter(down_weight)
        self.register_buffer(
            "_up_weight_bf16",
            up_weight.bfloat16(),
            persistent=False,
        )
        self.register_buffer(
            "_down_weight_bf16",
            down_weight.bfloat16(),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_f8",
            torch.empty(0, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer(
            "_up_weight_scale",
            torch.ones(1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_down_weight_f8_storage",
            torch.empty(0, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer(
            "_down_weight_scale",
            torch.ones(1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_down_weight_next_scale",
            torch.ones(1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_down_activation_scale",
            torch.ones(1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_down_partial_amax",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_down_weight_partial_amax",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self._fp8_ready = False
        self.register_load_state_dict_post_hook(self._invalidate_fp8_after_load)

    def _invalidate_fp8_after_load(
        self,
        _module: nn.Module,
        _incompatible_keys: object,
    ) -> None:
        self._fp8_ready = False
        self._up_weight_bf16.copy_(self.up_weight)
        self._down_weight_bf16.copy_(self.down_weight)

    @torch.no_grad()
    def refresh_fp8(self, *, bootstrap_down: bool = False) -> None:
        if self.up_weight.device.type != "cuda":
            raise RuntimeError("FP8 projection refresh requires CUDA")
        self._up_weight_bf16.copy_(self.up_weight)
        self._down_weight_bf16.copy_(self.down_weight)
        hidden, width = self.up_weight.shape
        sms = torch.cuda.get_device_properties(self.up_weight.device).multi_processor_count
        weight_tiles = ((hidden + 63) // 64) * ((width + 63) // 64)
        if self._up_weight_f8.shape != self.up_weight.shape:
            self._up_weight_f8 = torch.empty_like(
                self.up_weight,
                dtype=torch.float8_e4m3fn,
            )
            self._down_weight_f8_storage = torch.empty(
                width,
                hidden,
                dtype=torch.float8_e4m3fn,
                device=self.up_weight.device,
            )
            self._down_partial_amax = torch.empty(
                sms,
                dtype=torch.float32,
                device=self.up_weight.device,
            )
            self._down_weight_partial_amax = torch.empty(
                weight_tiles,
                dtype=torch.float32,
                device=self.up_weight.device,
            )

        up_scale = self._up_weight_bf16.float().abs().amax().clamp_min(1.0e-12) / 448.0
        self._up_weight_scale.copy_(up_scale)
        self._up_weight_f8.copy_(
            (self._up_weight_bf16 / self._up_weight_scale).to(torch.float8_e4m3fn)
        )
        if bootstrap_down:
            down_scale = self._down_weight_bf16.float().abs().amax().clamp_min(1.0e-12) / 448.0
            self._down_weight_next_scale.copy_(down_scale)
        quantize_transpose_mlp_down_weight(
            self._down_weight_bf16,
            self._down_weight_f8_storage,
            self._down_weight_next_scale,
            self._down_weight_scale,
            self._down_weight_partial_amax,
        )
        self._fp8_ready = True

    def forward(self, inputs: Tensor) -> Tensor:
        fp8_state = None
        if self.training:
            if not self._fp8_ready:
                raise RuntimeError("training a fused MLP requires refreshed FP8 projections")
            fp8_state = (
                self._up_weight_f8,
                self._up_weight_scale,
                self._down_weight_f8_storage.T,
                self._down_weight_scale,
                self._down_activation_scale,
                self._down_partial_amax,
            )
        return fused_relu_squared_mlp(
            inputs,
            self.up_weight,
            self.down_weight,
            self._up_weight_bf16,
            self._down_weight_bf16,
            fp8_state=fp8_state,
        )


def refresh_fused_mlp_fp8(
    module: nn.Module,
    *,
    bootstrap_down: bool | None = None,
) -> None:
    """Initialize missing FP8 state or refresh every projection after an update."""
    for child in module.modules():
        if isinstance(child, FusedFeedForward):
            if bootstrap_down is None and child._fp8_ready:
                continue
            child.refresh_fp8(
                bootstrap_down=not child._fp8_ready if bootstrap_down is None else bootstrap_down
            )


class Block(nn.Module):
    """Pre-norm attention + FFN block with near-identity gated residuals."""

    def __init__(
        self,
        config: StructuredConfig,
        *,
        residual_initial: float = 0.1,
        conditioned: bool = False,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.model_dim)
        self.attention = Attention(config)
        self.attention_gate = GatedResidual(config.model_dim, residual_initial)
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = FusedFeedForward(config) if config.fused_mlp else FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim, residual_initial)
        self.modulation = nn.Linear(config.model_dim, 4 * config.model_dim) if conditioned else None
        if self.modulation is not None:
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        queries: Tensor,
        context: Tensor | None = None,
        *,
        context_norm: nn.Module | None = None,
        query_rotation: tuple[Tensor, Tensor] | None = None,
        key_rotation: tuple[Tensor, Tensor] | None = None,
        context_valid: Tensor | None = None,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        attention_input = self.attention_norm(queries)
        ffn_scale = ffn_shift = None
        if self.modulation is not None:
            if conditioning is None:
                raise ValueError("conditioned block requires one vector per batch row")
            attention_scale, attention_shift, ffn_scale, ffn_shift = (
                self.modulation(conditioning).to(attention_input.dtype).chunk(4, dim=-1)
            )
            attention_input = attention_input * (
                1 + attention_scale.unsqueeze(1)
            ) + attention_shift.unsqueeze(1)
        elif conditioning is not None:
            raise ValueError("unconditioned block does not accept conditioning")
        if context is None:
            keys = attention_input
        else:
            keys = context_norm(context) if context_norm is not None else context
        hidden = self.attention_gate(
            queries,
            self.attention(
                attention_input,
                keys,
                query_rotation=query_rotation,
                key_rotation=key_rotation,
                context_valid=context_valid,
            ),
        )
        ffn_input = self.ffn_norm(hidden)
        if ffn_scale is not None and ffn_shift is not None:
            ffn_input = ffn_input * (1 + ffn_scale.unsqueeze(1)) + ffn_shift.unsqueeze(1)
        return self.ffn_gate(hidden, self.ffn(ffn_input))


class TileEmbedder(nn.Module):
    """Categorical embeddings plus a bounded-continuous MLP per tile token."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        width = config.model_dim
        self.kind = nn.Embedding(len(TILE_KINDS), width)
        self.occupant = nn.Embedding(len(TILE_OCCUPANTS), width)
        self.farm = nn.Embedding(2, width)
        self.row = nn.Embedding(BOARD_SIZE, width)
        self.column = nn.Embedding(BOARD_SIZE, width)
        self.quadrant = nn.Embedding(QUADRANT_COUNT, width)
        self.continuous = nn.Sequential(
            nn.Linear(N_TILE_CONTINUOUS, width),
            ReluSquared(),
            nn.Linear(width, width),
        )

    def forward(self, categorical: Tensor, continuous: Tensor) -> Tensor:
        return (
            self.kind(categorical[..., 0])
            + self.occupant(categorical[..., 1])
            + self.farm(categorical[..., 2])
            + self.row(categorical[..., 3])
            + self.column(categorical[..., 4])
            + self.quadrant(categorical[..., 5])
            + self.continuous(continuous.to(self.continuous[0].weight.dtype))
        )


class UnitEmbedder(nn.Module):
    """Unit identity, execution slot, position, inventory, and local tiles."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        width = config.model_dim
        self.role = nn.Embedding(len(UNIT_ROLES), width)
        self.slot = nn.Embedding(MAX_UNITS, width)
        self.row = nn.Embedding(BOARD_SIZE, width)
        self.column = nn.Embedding(BOARD_SIZE, width)
        self.farm = nn.Embedding(2, width)
        self.continuous = nn.Sequential(
            nn.Linear(N_UNIT_CONTINUOUS, width),
            ReluSquared(),
            nn.Linear(width, width),
        )
        self.gather_relation = nn.Embedding(len(UNIT_TILE_GATHERS), width)
        self.gather_projection = nn.Linear(len(UNIT_TILE_GATHERS) * width, width, bias=False)

    def local_tiles(self, farm_tiles: Tensor, gather: Tensor, gather_valid: Tensor) -> Tensor:
        """Gather each unit's HERE/NSEW encoded tiles: [B, U, 5, width]."""
        batch, units, slots = gather.shape
        width = farm_tiles.shape[-1]
        flat = gather.reshape(batch, units * slots, 1).expand(-1, -1, width)
        local = farm_tiles.gather(1, flat).view(batch, units, slots, width)
        local = torch.where(gather_valid.unsqueeze(-1), local, 0.0)
        return local + self.gather_relation.weight

    def forward(
        self,
        categorical: Tensor,
        continuous: Tensor,
        active: Tensor,
        local_tiles: Tensor,
        *,
        opponent: bool,
    ) -> Tensor:
        batch, units, slots, width = local_tiles.shape
        embedded = (
            self.role(categorical[..., 0])
            + self.slot(categorical[..., 1])
            + self.row(categorical[..., 2])
            + self.column(categorical[..., 3])
            + self.farm.weight[int(opponent)]
            + self.continuous(continuous.to(self.continuous[0].weight.dtype))
            + self.gather_projection(local_tiles.reshape(batch, units, slots * width))
        )
        return torch.where(active.unsqueeze(-1), embedded, 0.0)


class EconomyEmbedder(nn.Module):
    """Product, crop, farm-summary, and town tokens with identity embeddings."""

    def __init__(self, config: StructuredConfig, *, private_columns: bool) -> None:
        super().__init__()
        width = config.model_dim
        product_width = len(PRODUCT_TOKEN_FIELDS) + (
            len(PRODUCT_PRIVATE_FIELDS) if private_columns else 0
        )
        crop_width = len(CROP_TOKEN_FIELDS) + (len(CROP_PRIVATE_FIELDS) if private_columns else 0)
        self.private_columns = private_columns
        self.split_clock = config.split_clock_token
        self.product_identity = nn.Embedding(len(PRODUCTS), width)
        self.product_projection = nn.Linear(product_width, width)
        self.crop_identity = nn.Embedding(len(CROPS), width)
        self.crop_projection = nn.Linear(crop_width, width)
        self.farm_identity = nn.Embedding(2, width)
        self.farm_projection = nn.Linear(len(FARM_TOKEN_FIELDS), width)
        if self.split_clock:
            self.clock_projection = nn.Linear(6, width)
            self.town_projection = nn.Linear(len(TOWN_TOKEN_FIELDS) - 6, width)
        else:
            self.clock_projection = None
            self.town_projection = nn.Linear(len(TOWN_TOKEN_FIELDS), width)

    def forward(self, products: Tensor, crops: Tensor, farms: Tensor, town: Tensor) -> Tensor:
        dtype = self.product_projection.weight.dtype
        tokens = [
            self.product_projection(products.to(dtype)) + self.product_identity.weight,
            self.crop_projection(crops.to(dtype)) + self.crop_identity.weight,
            self.farm_projection(farms.to(dtype)) + self.farm_identity.weight,
        ]
        if self.clock_projection is None:
            tokens.append(self.town_projection(town.to(dtype)).unsqueeze(1))
        else:
            tokens.extend(
                (
                    self.clock_projection(town[..., :6].to(dtype)).unsqueeze(1),
                    self.town_projection(town[..., 6:].to(dtype)).unsqueeze(1),
                )
            )
        return torch.cat(tokens, dim=1)


class TrunkOutput(NamedTuple):
    """Everything the decoders and training auxiliaries read."""

    latents: Tensor  # [B, latents, model_dim], RMS-normalized
    own_patches: Tensor  # [B, 100, model_dim], post farm-local blocks
    opponent_patches: Tensor  # [B, 100, model_dim], post farm-local blocks
    opponent_summary: Tensor  # [B, opponent_latents, model_dim]
    unit_tokens: Tensor  # [B, MAX_UNITS, model_dim], inactive rows zeroed
    unit_local_tiles: Tensor  # [B, MAX_UNITS, 5, model_dim]
    economy_tokens: Tensor  # [B, economy tokens, model_dim]


class MuddLite(nn.Module):
    """One late dynamic residual route over four aligned latent streams."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.input = nn.Linear(width, 64)
        self.activation = ReluSquared()
        self.output = nn.Linear(64, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current: Tensor, sources: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        coefficients = 0.1 * self.output(self.activation(self.input(self.norm(current))))
        stacked = torch.stack(sources, dim=-2)
        return (coefficients.unsqueeze(-1) * stacked).sum(dim=-2)


class StructuredTrunk(nn.Module):
    """Shared encoder: farm-local blocks, opponent summary, latent core."""

    def __init__(self, config: StructuredConfig, *, private_columns: bool) -> None:
        super().__init__()
        self.config = config
        head_dim = config.model_dim // config.attention_heads
        self.rope = AxialRotaryEmbedding(head_dim)
        self.tiles = TileEmbedder(config)
        self.units = UnitEmbedder(config)
        self.economy = EconomyEmbedder(config, private_columns=private_columns)
        self.farm_local = nn.ModuleList(Block(config) for _ in range(config.farm_blocks))
        self.opponent_queries = nn.Parameter(
            torch.randn(config.opponent_latents, config.model_dim) * 0.02
        )
        self.opponent_summary = Block(config)
        self.opponent_context_norm = RMSNorm(config.model_dim)
        self.latent_queries = nn.Parameter(torch.randn(config.latents, config.model_dim) * 0.02)
        self.latent_read = Block(config)
        self.latent_context_norm = RMSNorm(config.model_dim)
        self.core = nn.ModuleList(
            Block(config, conditioned=config.global_modulation) for _ in range(config.core_layers)
        )
        self.core_norm = RMSNorm(config.model_dim)
        self.global_refresh = nn.ModuleDict(
            {
                str(layer): Block(config, residual_initial=0.0)
                for layer in config.global_refresh_layers
            }
        )
        self.global_context_norm = (
            RMSNorm(config.model_dim) if config.global_refresh_layers else None
        )
        self.reinject_norm = RMSNorm(config.model_dim) if config.input_reinject_layers else None
        self.reinject_gates = nn.ParameterDict(
            {
                str(layer): nn.Parameter(torch.zeros(config.model_dim))
                for layer in config.input_reinject_layers
            }
        )
        self.skip_gate = (
            nn.Parameter(torch.zeros(config.model_dim)) if config.core_skip_target else None
        )
        self.mudd = MuddLite(config.model_dim) if config.mudd_lite else None

    def encode_farms(self, tiles: Tensor, rotation: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Run shared farm-local blocks over both farms in one larger batch."""
        batch = tiles.shape[0]
        hidden = tiles.view(batch * 2, TILE_COUNT, self.config.model_dim)
        for block in self.farm_local:
            hidden = block(hidden, query_rotation=rotation, key_rotation=rotation)
        own, opponent = hidden.view(batch, 2, TILE_COUNT, self.config.model_dim).unbind(dim=1)
        return own, opponent

    def forward(
        self,
        inputs: StructuredInputs,
        *,
        opponent_units: Tensor | None = None,
        opponent_units_active: Tensor | None = None,
    ) -> TrunkOutput:
        batch = inputs.tile_categorical.shape[0]
        tiles = self.tiles(inputs.tile_categorical, inputs.tile_continuous)
        rotation = (
            self.rope.cosine.view(1, 1, TILE_COUNT, -1).expand(batch * 2, -1, -1, -1),
            self.rope.sine.view(1, 1, TILE_COUNT, -1).expand(batch * 2, -1, -1, -1),
        )
        own_tiles, opponent_tiles = self.encode_farms(tiles, rotation)

        local = self.units.local_tiles(
            own_tiles, inputs.unit_tile_gather, inputs.unit_tile_gather_valid
        )
        unit_tokens = self.units(
            inputs.unit_categorical,
            inputs.unit_continuous,
            inputs.unit_active,
            local,
            opponent=False,
        )
        economy_tokens = self.economy(inputs.products, inputs.crops, inputs.farms, inputs.town)

        summary = self.opponent_summary(
            self.opponent_queries.unsqueeze(0).expand(batch, -1, -1),
            opponent_tiles,
            context_norm=self.opponent_context_norm,
        )

        context_parts = [own_tiles, summary, unit_tokens, economy_tokens]
        valid_parts = [
            torch.ones(batch, own_tiles.shape[1], dtype=torch.bool, device=tiles.device),
            torch.ones(batch, summary.shape[1], dtype=torch.bool, device=tiles.device),
            inputs.unit_active,
            torch.ones(batch, economy_tokens.shape[1], dtype=torch.bool, device=tiles.device),
        ]
        if opponent_units is not None:
            if opponent_units_active is None:
                raise ValueError("opponent unit context requires its active mask")
            context_parts.append(opponent_units)
            valid_parts.append(opponent_units_active)
        context = torch.cat(context_parts, dim=1)
        context_valid = torch.cat(valid_parts, dim=1)

        latents = self.latent_read(
            self.latent_queries.unsqueeze(0).expand(batch, -1, -1),
            context,
            context_norm=self.latent_context_norm,
            context_valid=context_valid,
        )
        x0 = latents
        normalized_x0 = self.reinject_norm(x0) if self.reinject_norm is not None else None
        if self.config.global_refresh_context == "economy":
            global_context = economy_tokens
            global_valid = torch.ones(
                batch, economy_tokens.shape[1], dtype=torch.bool, device=tiles.device
            )
        elif self.config.global_refresh_context == "all":
            global_context = torch.cat((summary, unit_tokens, economy_tokens), dim=1)
            global_valid = torch.cat(
                (
                    torch.ones(batch, summary.shape[1], dtype=torch.bool, device=tiles.device),
                    inputs.unit_active,
                    torch.ones(
                        batch,
                        economy_tokens.shape[1],
                        dtype=torch.bool,
                        device=tiles.device,
                    ),
                ),
                dim=1,
            )
        else:
            global_context = global_valid = None
        conditioning = economy_tokens.mean(dim=1) if self.config.global_modulation else None
        snapshots: dict[int, Tensor] = {}
        for layer, block in enumerate(self.core, start=1):
            if self.mudd is not None and layer == self.config.core_layers:
                latents = latents + self.mudd(
                    latents,
                    (x0, snapshots[2], snapshots[5], latents),
                )
            latents = block(latents, conditioning=conditioning)
            if str(layer) in self.reinject_gates:
                gate = self.reinject_gates[str(layer)]
                assert normalized_x0 is not None
                latents = latents + gate * normalized_x0
            if layer == self.config.core_skip_target:
                assert self.skip_gate is not None
                latents = latents + self.skip_gate * snapshots[self.config.core_skip_source]
            if str(layer) in self.global_refresh:
                refresh = self.global_refresh[str(layer)]
                assert global_context is not None
                assert global_valid is not None
                assert self.global_context_norm is not None
                latents = refresh(
                    latents,
                    global_context,
                    context_norm=self.global_context_norm,
                    context_valid=global_valid,
                )
            snapshots[layer] = latents
        latents = self.core_norm(latents)
        return TrunkOutput(
            latents=latents,
            own_patches=own_tiles,
            opponent_patches=opponent_tiles,
            opponent_summary=summary,
            unit_tokens=unit_tokens,
            unit_local_tiles=local,
            economy_tokens=economy_tokens,
        )


class StructuredBelief(NamedTuple):
    """Typed actor representations exposed only to training auxiliaries."""

    own_patches: Tensor
    opponent_patches: Tensor
    opponent_summary: Tensor
    economy_entities: Tensor
    central_latents: Tensor
    unit_decisions: Tensor
    market_decisions: Tensor


class StructuredActor(nn.Module):
    """Decentralized structured policy with the FarmActor output contract."""

    def __init__(self, config: StructuredConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredConfig()
        self.config = config
        self.trunk = StructuredTrunk(config, private_columns=False)
        self.unit_decoder = Block(config)
        self.unit_local_decoder = None if config.fuse_unit_decoder else Block(config)
        # The trunk's latents leave core_norm already normalized; raw local
        # tiles and economy tokens each get their own context norm.
        self.local_context_norm = RMSNorm(config.model_dim)
        self.market_queries = nn.Embedding(MAX_MARKET_ORDERS, config.model_dim)
        self.market_decoder = Block(config)
        self.market_economy_decoder = None if config.fuse_market_decoder else Block(config)
        self.economy_context_norm = RMSNorm(config.model_dim)

        self.unit_head = nn.Sequential(
            RMSNorm(config.model_dim),
            nn.Linear(config.model_dim, N_UNIT_ACTIONS),
        )
        self.market_norm = RMSNorm(config.model_dim)
        self.market_kind = nn.Linear(config.model_dim, N_MARKET_KINDS)
        self.market_quantity_context = nn.Linear(config.model_dim, config.quantity_rank, bias=False)
        self.market_quantity_kind_gate = nn.Embedding(N_MARKET_KINDS, config.quantity_rank)
        self.market_quantity_value = nn.Embedding(N_QUANTITIES, config.quantity_rank)
        self.market_quantity_bias = nn.Parameter(torch.zeros(N_MARKET_KINDS, N_QUANTITIES))
        initialize_policy_heads(
            self.unit_head[-1],
            self.market_kind,
            self.market_quantity_context,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
        )

    def quantity_logits(self, quantity_context: Tensor, market_kinds: Tensor) -> Tensor:
        """Score exact quantities only for the already-selected market kind."""
        return factored_quantity_logits(
            quantity_context,
            market_kinds,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
            self.config.quantity_rank,
        )

    def forward_with_belief(self, inputs: StructuredInputs) -> tuple[ActorOutput, StructuredBelief]:
        batch = inputs.tile_categorical.shape[0]
        trunk = self.trunk(inputs)
        local = trunk.unit_local_tiles
        units, slots, width = local.shape[1], local.shape[2], local.shape[3]
        local_valid = inputs.unit_tile_gather_valid.clone()
        local_valid[..., 0] |= ~inputs.unit_active

        if self.unit_local_decoder is None:
            latent_context = trunk.latents[:, None].expand(-1, units, -1, -1)
            unit_context = torch.cat(
                (
                    latent_context,
                    self.local_context_norm(local),
                ),
                dim=2,
            ).reshape(batch * units, trunk.latents.shape[1] + slots, width)
            unit_valid = torch.cat(
                (
                    torch.ones(
                        batch,
                        units,
                        trunk.latents.shape[1],
                        dtype=torch.bool,
                        device=local.device,
                    ),
                    local_valid,
                ),
                dim=2,
            ).reshape(batch * units, trunk.latents.shape[1] + slots)
            unit_hidden = self.unit_decoder(
                trunk.unit_tokens.reshape(batch * units, 1, width),
                unit_context,
                context_valid=unit_valid,
            ).view(batch, units, width)
        else:
            unit_hidden = self.unit_decoder(trunk.unit_tokens, trunk.latents)
            unit_hidden = self.unit_local_decoder(
                unit_hidden.reshape(batch * units, 1, width),
                local.reshape(batch * units, slots, width),
                context_norm=self.local_context_norm,
                context_valid=local_valid.reshape(batch * units, slots),
            ).view(batch, units, width)
        unit_hidden = torch.where(inputs.unit_active.unsqueeze(-1), unit_hidden, 0.0)

        market_queries = self.market_queries.weight.unsqueeze(0).expand(batch, -1, -1)
        if self.market_economy_decoder is None:
            market_context = torch.cat(
                (
                    trunk.latents,
                    self.economy_context_norm(trunk.economy_tokens),
                ),
                dim=1,
            )
            market_hidden = self.market_decoder(market_queries, market_context)
        else:
            market_hidden = self.market_decoder(market_queries, trunk.latents)
            market_hidden = self.market_economy_decoder(
                market_hidden,
                trunk.economy_tokens,
                context_norm=self.economy_context_norm,
            )
        market_hidden = self.market_norm(market_hidden)
        output = ActorOutput(
            unit_logits=self.unit_head(unit_hidden).contiguous(),
            market_kind_logits=self.market_kind(market_hidden).contiguous(),
            market_quantity_context=self.market_quantity_context(market_hidden).contiguous(),
        )
        belief = StructuredBelief(
            own_patches=trunk.own_patches,
            opponent_patches=trunk.opponent_patches,
            opponent_summary=trunk.opponent_summary,
            economy_entities=trunk.economy_tokens,
            central_latents=trunk.latents,
            unit_decisions=unit_hidden,
            market_decisions=market_hidden,
        )
        return output, belief

    def forward(self, inputs: StructuredInputs) -> ActorOutput:
        return self.forward_with_belief(inputs)[0]


class StructuredCritic(nn.Module):
    """Centralized structured critic over both players' exact private state."""

    def __init__(self, config: StructuredConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredConfig()
        self.config = config
        critic_layers = config.critic_core_layers or config.core_layers
        skip_fits = config.core_skip_target <= critic_layers
        trunk_config = replace(
            config,
            core_layers=critic_layers,
            latents=config.critic_latents or config.latents,
            global_refresh_layers=tuple(
                layer for layer in config.global_refresh_layers if layer <= critic_layers
            ),
            global_refresh_context=(
                config.global_refresh_context
                if any(layer <= critic_layers for layer in config.global_refresh_layers)
                else "none"
            ),
            input_reinject_layers=tuple(
                layer for layer in config.input_reinject_layers if layer <= critic_layers
            ),
            core_skip_source=config.core_skip_source if skip_fits else 0,
            core_skip_target=config.core_skip_target if skip_fits else 0,
            mudd_lite=config.mudd_lite and critic_layers >= 6,
            critic_core_layers=0,
            critic_latents=0,
        )
        self.trunk = StructuredTrunk(trunk_config, private_columns=True)
        self.value_query = nn.Parameter(torch.randn(1, config.model_dim) * 0.02)
        self.value_decoder = Block(trunk_config)
        self.value_head = nn.Linear(config.model_dim, config.value_atoms)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.register_buffer(
            "support",
            torch.linspace(config.value_min, config.value_max, config.value_atoms),
            persistent=True,
        )

    def forward(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> Tensor:
        batch = inputs.tile_categorical.shape[0]
        # Opponent units attend as context only; their local tiles sit on the
        # opponent farm, which the latents already read through its tokens, so
        # their local context is the bare relation embedding.
        relation_only = self.trunk.units.gather_relation.weight.expand(
            batch, opponent_unit_categorical.shape[1], -1, -1
        )
        opponent_units = self.trunk.units(
            opponent_unit_categorical,
            opponent_unit_continuous,
            opponent_unit_active,
            relation_only,
            opponent=True,
        )
        trunk = self.trunk(
            inputs,
            opponent_units=opponent_units,
            opponent_units_active=opponent_unit_active,
        )
        value_hidden = self.value_decoder(
            self.value_query.unsqueeze(0).expand(batch, -1, -1),
            trunk.latents,
        )
        return self.value_head(value_hidden[:, 0]).contiguous()

    def value(self, logits: Tensor) -> Tensor:
        return (logits.float().softmax(dim=-1) * self.support.float()).sum(dim=-1)
