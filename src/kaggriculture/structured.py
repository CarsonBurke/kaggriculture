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
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
)
from kaggriculture.model import (
    ActorOutput,
    AxialRotaryEmbedding,
    Linear,
    ReluSquared,
    RMSNorm,
    _sdpa_inputs,
    categorical_value,
    categorical_value_support,
    factored_quantity_logits,
    initialize_policy_heads,
    softcap_value_logits,
)
from kaggriculture.tokens import (
    ANIMAL_PRIVATE_FIELDS,
    ANIMAL_TOKEN_FIELDS,
    CROP_PRIVATE_FIELDS,
    CROP_TOKEN_FIELDS,
    DEFAULT_OBSERVATION_SCHEMA_VERSION,
    N_TILE_CONTINUOUS,
    N_UNIT_CONTINUOUS,
    PRODUCT_PRIVATE_FIELDS,
    PRODUCT_TOKEN_FIELDS,
    QUADRANT_COUNT,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TILE_COUNT,
    TILE_KINDS,
    TILE_OCCUPANTS,
    TOWN_TOKEN_FIELDS,
    UNIT_ROLES,
    UNIT_TILE_GATHERS,
    farm_token_fields,
)
from kaggriculture.triton_mlp import (
    fused_relu_squared_mlp,
    quantize_transpose_mlp_down_weight,
)


@dataclass(frozen=True)
class StructuredConfig:
    """Target-architecture hyperparameters from VIT_PLAN."""

    observation_schema_version: int = DEFAULT_OBSERVATION_SCHEMA_VERSION
    action_interface: int = 1
    model_dim: int = 128
    attention_heads: int = 4
    attention_kv_heads: int = 2
    ffn_multiplier: int = 2
    farm_blocks: int = 2
    opponent_latents: int = 8
    latents: int = 32
    core_layers: int = 8
    quantity_rank: int = 32
    # Empty means disabled. Research variants that want the canonical 8-layer
    # refresh schedule opt into ``global_refresh_context="all"`` explicitly;
    # an empty layer tuple must not silently enable extra blocks.
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
    # Critic reads seed residual content from observations, not learned queries.
    critic_state_read: bool = False
    # Shared centralized decoder/readout, with one baseline per acting entity.
    per_entity_critic: bool = False
    # The actor's view of the opponent's board. Both farms run the shared
    # `farm_local` blocks and the opponent half reaches the rest of the actor
    # only as `opponent_summary`, so it is 30% of the actor's forward bytes at
    # production width -- half of `farm_local`, half of the tile embedding, and
    # the summary block. `scripts/ablate_opponent_farm.py` permutes that half
    # across a batch of real states and measures what the policy does about it:
    # mean unit KL 1.3e-5 and 0.04% of greedy actions changed at iteration 390,
    # against 3.80 and 59.9% for the same permutation of the actor's own farm.
    # The insensitivity is present at initialization and grows with training, so
    # it is a property of the architecture rather than a stage of learning.
    #
    # Disabling it is a learning change, not a speedup, and is off by default
    # until an A/B says otherwise: an actor that cannot see the opponent's board
    # cannot learn to react to it, and the ablation cannot distinguish "the
    # information is useless here" from "this path is too narrow to carry it".
    # The centralized critic is unaffected either way -- it reads both farms and
    # measurably uses them -- and the two trunks share no parameters.
    actor_opponent_farm: bool = True
    value_atoms: int = 255
    value_min: float = -2.2
    value_max: float = 2.2
    value_sigma_ratio: float = 3.0
    # Optional scalar ablation: half squared error without target clipping or
    # a readout softcap. The four value-support fields above are inert in it.
    scalar_value: bool = False

    def __post_init__(self) -> None:
        if self.action_interface not in (1, 2, 4):
            raise ValueError("action_interface must be 1, 2, or 4")
        if self.observation_schema_version not in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS:
            raise ValueError(
                "stale structured observation schema; fresh encoding and training required"
            )
        refresh_layers = tuple(self.global_refresh_layers)
        refresh_context = self.global_refresh_context
        if not refresh_layers and refresh_context == "all":
            refresh_layers = tuple(range(3, self.core_layers, 3))
            if not refresh_layers:
                refresh_context = "none"
        object.__setattr__(self, "global_refresh_layers", refresh_layers)
        object.__setattr__(self, "global_refresh_context", refresh_context)
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
    animals: Tensor  # [B, len(ANIMALS), animal feature width]
    crops: Tensor  # [B, len(CROPS), crop feature width]
    farms: Tensor  # [B, 2, len(FARM_TOKEN_FIELDS)]; models read their schema's prefix
    town: Tensor  # [B, len(TOWN_TOKEN_FIELDS)]


class CriticExtras(NamedTuple):
    """Opponent-side critic tensors stacked alongside StructuredInputs."""

    products: Tensor  # [B, len(PRODUCTS), len(PRODUCT_PRIVATE_FIELDS)]
    animals: Tensor  # [B, len(ANIMALS), len(ANIMAL_PRIVATE_FIELDS)]
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
        animals=stacked("animals"),
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
        animals=stacked("critic_animals"),
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


#: Memory-efficient CUDA attention requires head widths divisible by eight.
#: Padding preserves the original scale and is a no-op for aligned widths.
_FUSED_ATTENTION_HEAD_MULTIPLE = 8

#: Standard-deviation of a learned query bank that seeds a residual stream
#: rather than reading one. A query at unit RMS would dominate the block's
#: first residual sum; 0.02 is the small init every such bank here started
#: from. It is named because the optimizer has to know it: Adam's step is an
#: absolute per-element displacement, so a parameter living at 0.02 RMS moves
#: by fifty times the FRACTION of itself that a unit-RMS neighbour does under
#: one shared rate. `route_parameters` reads it back through
#: `adam_learning_rate_multipliers`.
SMALL_QUERY_INITIAL_SCALE = 0.02


def _pad_head_width(tensor: Tensor, width: int) -> Tensor:
    extra = width - tensor.shape[-1]
    return tensor if extra == 0 else nn.functional.pad(tensor, (0, extra))


def _masked_flex_attention(
    query: Tensor, key: Tensor, value: Tensor, key_valid: Tensor, scale: float
) -> Tensor:
    # Flex maps batch to CUDA grid Y, whose limit is 65535. Flattened per-unit
    # decoders exceed it at production minibatch sizes. Independent batch tiles
    # preserve the same fused operator and gradients without copying K/V.
    if query.shape[0] > 65535:
        outputs = []
        for start in range(0, query.shape[0], 32768):
            stop = start + 32768
            outputs.append(
                _masked_flex_attention(
                    query[start:stop],
                    key if key.shape[0] == 1 else key[start:stop],
                    value if value.shape[0] == 1 else value[start:stop],
                    key_valid if key_valid.shape[0] == 1 else key_valid[start:stop],
                    scale,
                )
            )
        return torch.cat(outputs, dim=0)

    def score_mod(score, b, h, q, k):
        mask_batch = 0 if key_valid.shape[0] == 1 else b
        return torch.where(key_valid[mask_batch, k], score, -float("inf"))

    return flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        scale=scale,
        # Default 128-row backward tiles overpad short folded entity queries.
        kernel_options={
            "BACKEND": "TRITON",
            **(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 64,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 64,
                }
                if query.shape[-1] <= 32 and query.shape[-2] <= 64 and key.shape[-2] <= 256
                else {}
            ),
        },
    )


def _fused_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    *,
    enable_gqa: bool,
    scale: float,
    backend: SDPBackend | None = None,
) -> Tensor:
    """One fused attention call, with the head width padded into kernel support.

    `scale` is stated rather than defaulted: SDPA derives its default from the
    padded width, which is not the width this attention is defined over.
    An explicit backend changes only the kernel; CUDA GQA still folds query
    groups without materializing repeated K/V heads.
    """
    head_dim = query.shape[-1]
    remainder = head_dim % _FUSED_ATTENTION_HEAD_MULTIPLE if query.is_cuda else 0
    if remainder:
        width = head_dim + _FUSED_ATTENTION_HEAD_MULTIPLE - remainder
        query, key, value = (_pad_head_width(t, width) for t in (query, key, value))
    fold_gqa = query.is_cuda and enable_gqa
    if fold_gqa:
        # Fold each query-head group into the sequence. Besides supporting equal-
        # head kernels, this avoids native GQA's expanded dK/dV intermediates.
        batch, heads, query_tokens, width = query.shape
        kv_heads = key.shape[1]
        grouped_tokens = (heads // kv_heads) * query_tokens
        query = query.reshape(batch, kv_heads, grouped_tokens, width)
        # Key-only masks already broadcast over the folded query sequence.
        # Query/head-specific masks must follow the original score layout.
        if (
            attention_mask is not None
            and attention_mask.ndim >= 2
            and (
                attention_mask.shape[-2] != 1
                or (attention_mask.ndim >= 3 and attention_mask.shape[-3] != 1)
            )
        ):
            attention_mask = attention_mask.expand(
                batch, heads, query_tokens, key.shape[-2]
            ).reshape(batch, kv_heads, grouped_tokens, key.shape[-2])
        enable_gqa = False
    if (
        backend is None
        and fold_gqa
        and query.dtype in (torch.float16, torch.bfloat16)
        and torch.is_grad_enabled()
        and attention_mask is not None
        and attention_mask.dtype == torch.bool
        and attention_mask.ndim == 4
        and attention_mask.shape[1:3] == (1, 1)
    ):
        # Fuse validity into dense scores; constructing sparse block metadata
        # does not pay for these short sequences. The decoding heuristic is
        # slower here. No-grad rollout retains SDPA's supported vmap batching.
        attended = _masked_flex_attention(query, key, value, attention_mask[:, 0, 0, :], scale)
    else:
        # Prefer Flash for dense attention and cuDNN for arbitrary masks. Keep
        # efficient CUDA for unsupported configurations, never a math fallback.
        # Explicit benchmark backends remain exclusive and fail loudly.
        backends: SDPBackend | list[SDPBackend]
        if backend is not None:
            backends = backend
        elif query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
            preferred = (
                SDPBackend.FLASH_ATTENTION if attention_mask is None else SDPBackend.CUDNN_ATTENTION
            )
            backends = [preferred, SDPBackend.EFFICIENT_ATTENTION]
        else:
            backends = SDPBackend.EFFICIENT_ATTENTION if query.is_cuda else SDPBackend.MATH
        with sdpa_kernel(backends, set_priority=True):
            attended = nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=scale,
                enable_gqa=enable_gqa,
            )
    if fold_gqa:
        attended = attended.reshape(batch, heads, query_tokens, attended.shape[-1])
    return attended[..., :head_dim] if remainder else attended


class Attention(nn.Module):
    """QK-normalized attention over an explicit context (self or cross)."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.kv_heads = config.attention_kv_heads
        self.head_dim = config.model_dim // self.heads
        self.query = Linear(config.model_dim, config.model_dim, bias=False)
        self.key_value = Linear(config.model_dim, 2 * self.kv_heads * self.head_dim, bias=False)
        self.query_norm = RMSNorm(self.head_dim)
        self.key_norm = RMSNorm(self.head_dim)
        self.output = Linear(config.model_dim, config.model_dim, bias=False)
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
        self.input = Linear(config.model_dim, hidden)
        self.activation = ReluSquared()
        self.output = Linear(hidden, config.model_dim)
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
    """Pre-norm block; state reads initialize residuals from normalized attention."""

    def __init__(
        self,
        config: StructuredConfig,
        *,
        residual_initial: float = 0.1,
        conditioned: bool = False,
        state_read: bool = False,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.model_dim)
        # A state read is the input projection, not a zero-initialized residual branch.
        attention_config = replace(config, zero_init_branches=False) if state_read else config
        self.attention = Attention(attention_config)
        self.attention_gate = (
            None if state_read else GatedResidual(config.model_dim, residual_initial)
        )
        self.read_norm = RMSNorm(config.model_dim, elementwise_affine=False) if state_read else None
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = FusedFeedForward(config) if config.fused_mlp else FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim, residual_initial)
        self.modulation = Linear(config.model_dim, 4 * config.model_dim) if conditioned else None
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
        if self.read_norm is not None and context is None:
            raise ValueError("a state read requires observation context")
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
        hidden = self.attention(
            attention_input,
            keys,
            query_rotation=query_rotation,
            key_rotation=key_rotation,
            context_valid=context_valid,
        )
        if self.read_norm is not None:
            hidden = self.read_norm(hidden)
        else:
            assert self.attention_gate is not None
            hidden = self.attention_gate(queries, hidden)
        ffn_input = self.ffn_norm(hidden)
        if ffn_scale is not None and ffn_shift is not None:
            ffn_input = ffn_input * (1 + ffn_scale.unsqueeze(1)) + ffn_shift.unsqueeze(1)
        return self.ffn_gate(hidden, self.ffn(ffn_input))


class _TinyVocabularyEmbedding(torch.autograd.Function):
    """Sum of embedding lookups whose tables have only a handful of rows.

    The forward is the ordinary gather. The backward is one dense GEMM of the
    transposed one-hot indices against the gradient: an embedding table this
    small turns the stock scatter-add backward into millions of atomic adds
    contending for a few hundred addresses, which at 1.6M tile tokens per
    minibatch cost more than the whole farm attention backward.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(indices: Tensor, *weights: Tensor) -> Tensor:
        embedded = weights[0][indices[..., 0]]
        for column, weight in enumerate(weights[1:], start=1):
            embedded = embedded + weight[indices[..., column]]
        return embedded

    @staticmethod
    def setup_context(ctx, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        indices, *weights = inputs
        ctx.save_for_backward(indices)
        ctx.rows = tuple(weight.shape[0] for weight in weights)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor | None, ...]:
        (indices,) = ctx.saved_tensors
        columns = len(ctx.rows)
        flat_gradient = gradient.reshape(-1, gradient.shape[-1])
        # One combined one-hot over the concatenated vocabularies: each token
        # row carries one 1 per table, so a single GEMM yields every table's
        # gradient stacked along the rows.
        offsets = torch.tensor(
            tuple(sum(ctx.rows[:column]) for column in range(columns)),
            device=indices.device,
            dtype=indices.dtype,
        )
        one_hot = nn.functional.one_hot(
            (indices.reshape(-1, columns) + offsets).reshape(-1), sum(ctx.rows)
        )
        one_hot = one_hot.view(-1, columns, sum(ctx.rows)).sum(dim=1).to(gradient.dtype)
        stacked = one_hot.t() @ flat_gradient
        return None, *stacked.split(ctx.rows, dim=0)


class TileEmbedder(nn.Module):
    """Categorical embeddings plus a bounded-continuous MLP per tile token.

    Tile tokens are laid out own farm first, ``y * BOARD_SIZE + x`` within a farm,
    so the farm, row, column, and quadrant columns are functions of the token slot
    alone (`tokens.TILE_SLOT_CATEGORICAL`). Their embeddings are therefore one
    ``[tokens, width]`` table added to every row instead of four per-token
    lookups, and their backward is a batch reduction rather than a scatter.
    """

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
            Linear(N_TILE_CONTINUOUS, width),
            ReluSquared(),
            Linear(width, width),
        )

    def slot_embedding(self, tokens: int, device: torch.device) -> Tensor:
        """Farm/row/column/quadrant embeddings for the leading ``tokens`` slots."""
        token = torch.arange(tokens, device=device)
        farm, position = token // TILE_COUNT, token % TILE_COUNT
        row, column = position // BOARD_SIZE, position % BOARD_SIZE
        half = BOARD_SIZE // 2
        quadrant = (row >= half).long() * 2 + (column >= half).long()
        return (
            self.farm.weight[farm]
            + self.row.weight[row]
            + self.column.weight[column]
            + self.quadrant.weight[quadrant]
        )

    def forward(self, categorical: Tensor, continuous: Tensor) -> Tensor:
        vocabulary = categorical[..., :2]
        weights = (self.kind.weight, self.occupant.weight)
        # The custom backward is only for a table that receives a gradient. A
        # frozen copy run under grad mode takes the plain gather, which Dynamo
        # also traces where it cannot trace this vararg Function with no input
        # requiring grad.
        embedded = (
            _TinyVocabularyEmbedding.apply(vocabulary, *weights)
            if torch.is_grad_enabled() and any(weight.requires_grad for weight in weights)
            else _TinyVocabularyEmbedding.forward(vocabulary, *weights)
        )
        projected = self.continuous(continuous.to(self.continuous[0].weight.dtype))
        slots = self.slot_embedding(categorical.shape[-2], categorical.device)
        # Enter the compute dtype here: the residual stream is already in it after
        # the first gated residual, so this only moves one rounding earlier while
        # halving the traffic and saved activations of the first farm block.
        return (embedded + slots + projected).to(projected.dtype)


class UnitEmbedder(nn.Module):
    """Unit identity, execution slot, position, inventory, and local tiles."""

    def __init__(
        self,
        config: StructuredConfig,
        *,
        local_init: bool = True,
        local_context: bool = True,
    ) -> None:
        super().__init__()
        width = config.model_dim
        self.role = nn.Embedding(len(UNIT_ROLES), width)
        self.slot = nn.Embedding(MAX_UNITS, width)
        self.row = nn.Embedding(BOARD_SIZE, width)
        self.column = nn.Embedding(BOARD_SIZE, width)
        self.farm = nn.Embedding(2, width)
        self.continuous = nn.Sequential(
            Linear(N_UNIT_CONTINUOUS, width),
            ReluSquared(),
            Linear(width, width),
        )
        self.gather_relation = (
            nn.Embedding(len(UNIT_TILE_GATHERS), width) if local_init or local_context else None
        )
        self.gather_projection = (
            Linear(len(UNIT_TILE_GATHERS) * width, width, bias=False) if local_init else None
        )

    def local_tiles(self, farm_tiles: Tensor, gather: Tensor, gather_valid: Tensor) -> Tensor:
        """Gather each unit's HERE/NSEW encoded tiles: [B, U, 5, width]."""
        if self.gather_relation is None:
            raise ValueError("unit embedder was constructed without local context")
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
        local_tiles: Tensor | None,
        *,
        opponent: bool,
    ) -> Tensor:
        embedded = (
            self.role(categorical[..., 0])
            + self.slot(categorical[..., 1])
            + self.row(categorical[..., 2])
            + self.column(categorical[..., 3])
            + self.farm.weight[int(opponent)]
            + self.continuous(continuous.to(self.continuous[0].weight.dtype))
        )
        if self.gather_projection is not None:
            if local_tiles is None:
                raise ValueError("local unit initialization requires gathered tiles")
            batch, units, slots, width = local_tiles.shape
            embedded = embedded + self.gather_projection(
                local_tiles.reshape(batch, units, slots * width)
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
        animal_width = len(ANIMAL_TOKEN_FIELDS) + (
            len(ANIMAL_PRIVATE_FIELDS) if private_columns else 0
        )
        self.private_columns = private_columns
        self.split_clock = config.split_clock_token
        self.product_identity = nn.Embedding(len(PRODUCTS), width)
        self.product_projection = Linear(product_width, width)
        self.animal_identity = nn.Embedding(len(ANIMALS), width)
        self.animal_projection = Linear(animal_width, width)
        self.crop_identity = nn.Embedding(len(CROPS), width)
        self.crop_projection = Linear(crop_width, width)
        self.farm_identity = nn.Embedding(2, width)
        # Staged farm tokens carry the newest schema's columns; this model reads
        # the prefix its own schema defines, so v3 weights see v3 inputs.
        self.farm_width = len(farm_token_fields(config.observation_schema_version))
        self.farm_projection = Linear(self.farm_width, width)
        if self.split_clock:
            self.clock_projection = Linear(6, width)
            self.town_projection = Linear(len(TOWN_TOKEN_FIELDS) - 6, width)
        else:
            self.clock_projection = None
            self.town_projection = Linear(len(TOWN_TOKEN_FIELDS), width)

    def forward(
        self, products: Tensor, animals: Tensor, crops: Tensor, farms: Tensor, town: Tensor
    ) -> Tensor:
        dtype = self.product_projection.weight.dtype
        tokens = [
            self.product_projection(products.to(dtype)) + self.product_identity.weight,
            self.animal_projection(animals.to(dtype)) + self.animal_identity.weight,
            self.crop_projection(crops.to(dtype)) + self.crop_identity.weight,
            self.farm_projection(farms[..., : self.farm_width].to(dtype))
            + self.farm_identity.weight,
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
        self.input = Linear(width, 64)
        self.activation = ReluSquared()
        self.output = Linear(64, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current: Tensor, sources: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        coefficients = 0.1 * self.output(self.activation(self.input(self.norm(current))))
        stacked = torch.stack(sources, dim=-2)
        return (coefficients.unsqueeze(-1) * stacked).sum(dim=-2)


def _token_mean(tokens: Tensor) -> Tensor:
    """Keep conditioning reduction order fixed across rollout/update batches.

    A batch-dependent parallel FP32 reduction can cross a BF16 midpoint before
    the modulation projection. The short token axis is accumulated in order;
    Inductor fuses the additions without allocating intermediate tensors.
    """
    total = tokens.select(-2, 0).float()
    for token in range(1, tokens.shape[-2]):
        total = total + tokens.select(-2, token).float()
    return (total / tokens.shape[-2]).to(tokens.dtype)


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
        state_read = private_columns and config.critic_state_read
        # The knob is the actor's alone. The centralized critic reads both farms
        # and `scripts/ablate_opponent_farm.py` shows it uses them, so private
        # columns keep the opponent path regardless of how the actor is built.
        self.opponent_farm = bool(private_columns) or config.actor_opponent_farm

        def small_query(tokens: int) -> nn.Parameter:
            queries = torch.randn(tokens, config.model_dim)
            return nn.Parameter(
                torch.nn.functional.rms_norm(queries, (config.model_dim,))
                if state_read
                else queries * SMALL_QUERY_INITIAL_SCALE
            )

        # Built only when read. Registering an unused query bank would hand the
        # optimizer parameters that never receive a gradient and would put them
        # in the checkpoint and the parameter count, where they read as capacity
        # the model does not have.
        if self.opponent_farm:
            self.opponent_queries = small_query(config.opponent_latents)
            self.opponent_summary = Block(config, state_read=state_read)
            self.opponent_context_norm = RMSNorm(config.model_dim)
        self.latent_queries = small_query(config.latents)
        # Read by `route_parameters`: only the banks actually left at the small
        # scale get the matching rate. A state read is normalized on the way in
        # and initialized at unit RMS, so it keeps the shared rate.
        multipliers = {} if state_read else {"latent_queries": SMALL_QUERY_INITIAL_SCALE}
        if multipliers and self.opponent_farm:
            multipliers["opponent_queries"] = SMALL_QUERY_INITIAL_SCALE
        self.adam_learning_rate_multipliers = multipliers
        self.latent_read = Block(config, state_read=state_read)
        self.latent_context_norm = RMSNorm(config.model_dim)
        self.core_input_norm = (
            RMSNorm(config.model_dim, elementwise_affine=False) if state_read else None
        )
        self.core = nn.ModuleList(
            Block(config, conditioned=config.global_modulation) for _ in range(config.core_layers)
        )
        self.core_norm = RMSNorm(config.model_dim)
        self.global_refresh = nn.ModuleDict(
            {str(layer): Block(config) for layer in config.global_refresh_layers}
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
        """Run shared farm-local blocks over every farm this trunk reads.

        With the opponent farm disabled the tile embedding already stopped at
        the own half, so the block batch is halved rather than computed and
        discarded, and the returned opponent patches are the zero-token tensor
        the rest of the trunk concatenates around.
        """
        batch = tiles.shape[0]
        farms = tiles.shape[1] // TILE_COUNT
        hidden = tiles.reshape(batch * farms, TILE_COUNT, self.config.model_dim)
        for block in self.farm_local:
            hidden = block(hidden, query_rotation=rotation, key_rotation=rotation)
        hidden = hidden.view(batch, farms, TILE_COUNT, self.config.model_dim)
        if farms == 1:
            return hidden.select(dim=1, index=0), hidden.new_zeros(batch, 0, self.config.model_dim)
        own, opponent = hidden.unbind(dim=1)
        return own, opponent

    def forward(
        self,
        inputs: StructuredInputs,
        *,
        opponent_units: Tensor | None = None,
        opponent_units_active: Tensor | None = None,
    ) -> TrunkOutput:
        batch = inputs.tile_categorical.shape[0]
        farms = 2 if self.opponent_farm else 1
        tiles = self.tiles(
            inputs.tile_categorical[:, : farms * TILE_COUNT],
            inputs.tile_continuous[:, : farms * TILE_COUNT],
        )
        rotation = (
            self.rope.cosine.view(1, 1, TILE_COUNT, -1).expand(batch * farms, -1, -1, -1),
            self.rope.sine.view(1, 1, TILE_COUNT, -1).expand(batch * farms, -1, -1, -1),
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
        economy_tokens = self.economy(
            inputs.products, inputs.animals, inputs.crops, inputs.farms, inputs.town
        )

        summary = (
            self.opponent_summary(
                self.opponent_queries.unsqueeze(0).expand(batch, -1, -1),
                opponent_tiles,
                context_norm=self.opponent_context_norm,
            )
            if self.opponent_farm
            else own_tiles.new_zeros(batch, 0, self.config.model_dim)
        )

        def all_valid(tokens: Tensor) -> Tensor:
            return torch.ones(batch, tokens.shape[1], dtype=torch.bool, device=tiles.device)

        # Paired so a part can never be added to one list and forgotten in the
        # other: a context token without its validity entry shifts every later
        # mask by one and silently reads the wrong tokens.
        parts: list[tuple[Tensor, Tensor]] = [(own_tiles, all_valid(own_tiles))]
        if self.opponent_farm:
            parts.append((summary, all_valid(summary)))
        parts.append((unit_tokens, inputs.unit_active))
        parts.append((economy_tokens, all_valid(economy_tokens)))
        if opponent_units is not None:
            if opponent_units_active is None:
                raise ValueError("opponent unit context requires its active mask")
            parts.append((opponent_units, opponent_units_active))
        context = torch.cat([tokens for tokens, _valid in parts], dim=1)
        context_valid = torch.cat([valid for _tokens, valid in parts], dim=1)

        latents = self.latent_read(
            self.latent_queries.unsqueeze(0).expand(batch, -1, -1),
            context,
            context_norm=self.latent_context_norm,
            context_valid=context_valid,
        )
        if self.core_input_norm is not None:
            latents = self.core_input_norm(latents)
        x0 = latents
        normalized_x0 = self.reinject_norm(x0) if self.reinject_norm is not None else None
        if self.config.global_refresh_context == "economy":
            global_context = economy_tokens
            global_valid = torch.ones(
                batch, economy_tokens.shape[1], dtype=torch.bool, device=tiles.device
            )
        elif self.config.global_refresh_context == "all":
            # Reuse the exact entity memory and mask from the initial read. This
            # includes own patches and critic-only opponent units; rebuilding a
            # narrower context here silently turned "all" into only summary,
            # own units, and economy, and paid for two extra concatenations.
            global_context = context
            global_valid = context_valid
        else:
            global_context = global_valid = None
        conditioning = _token_mean(economy_tokens) if self.config.global_modulation else None
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
    unit_decisions: Tensor  # Exact post-normalization input to the unit projection.
    market_decisions: Tensor  # Exact post-normalization input to the market heads.


class StructuredDecisionBelief(NamedTuple):
    """Normalized entity-decoder outputs consumed by the policy heads."""

    unit_decisions: Tensor
    market_decisions: Tensor


class StructuredCriticBelief(NamedTuple):
    """The normalized representation consumed by the critic's final value head."""

    value_decision: Tensor  # (B, N, D): global first, optionally units then orders.


class JepaBelief(NamedTuple):
    """Everything the shared `lejepa` backbone hands to its one world-model arm.

    The first two fields are the per-slot decision states; the last two are the
    encoded observation, which is what makes the objective a world model rather
    than a policy-readout regularizer. Under the shared backbone this is also the
    entire interface between the world model and its two consumers: the actor's
    heads and the critic's private tower both read these four tensors and nothing
    else, and both read them detached.
    """

    unit_decisions: Tensor  # (B, MAX_UNITS, D)
    market_decisions: Tensor  # (B, MAX_MARKET_ORDERS, D)
    economy: Tensor  # (B, economy tokens, D)
    tiles: Tensor  # (B, 2 * TILE_COUNT, D), own farm then opponent farm

    def detach(self) -> JepaBelief:
        """The same latents with the gradient cut, which is how consumers read them.

        The backbone is trained by `lejepa.jepa_horizon_loss` alone. Every path
        from a policy or value objective back into it passes through here, so the
        detach is one call in one place rather than a convention each consumer
        has to remember.
        """
        return JepaBelief(*(value.detach() for value in self))


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
            Linear(config.model_dim, N_UNIT_ACTIONS),
        )
        self.market_norm = RMSNorm(config.model_dim)
        self.market_kind = Linear(config.model_dim, N_MARKET_KINDS)
        self.market_quantity_context = Linear(config.model_dim, config.quantity_rank, bias=False)
        self.market_quantity_kind_gate = nn.Embedding(N_MARKET_KINDS, config.quantity_rank)
        quantity_rows = (
            7 if config.action_interface == 4 else N_QUANTITIES + (config.action_interface == 2)
        )
        self.market_quantity_value = nn.Embedding(quantity_rows, config.quantity_rank)
        self.market_quantity_bias = nn.Parameter(torch.zeros(N_MARKET_KINDS, quantity_rows))
        initialize_policy_heads(
            self.unit_head[-1],
            self.market_kind,
            self.market_quantity_context,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
            action_interface=config.action_interface,
        )

    def quantity_logits(
        self, quantity_context: Tensor, market_kinds: Tensor, quantity_mask: Tensor | None = None
    ) -> Tensor:
        """Score exact quantities only for the already-selected market kind."""
        return factored_quantity_logits(
            quantity_context,
            market_kinds,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
            self.config.quantity_rank,
            quantity_mask,
        )

    def _decode_entities(
        self,
        latents: Tensor,
        unit_tokens: Tensor,
        local: Tensor,
        economy_tokens: Tensor,
        inputs: StructuredInputs,
    ) -> StructuredDecisionBelief:
        batch = inputs.tile_categorical.shape[0]
        units, slots, width = local.shape[1], local.shape[2], local.shape[3]
        local_valid = inputs.unit_tile_gather_valid.clone()
        local_valid[..., 0] |= ~inputs.unit_active

        if self.unit_local_decoder is None:
            latent_context = latents[:, None].expand(-1, units, -1, -1)
            unit_context = torch.cat(
                (
                    latent_context,
                    self.local_context_norm(local),
                ),
                dim=2,
            ).reshape(batch * units, latents.shape[1] + slots, width)
            unit_valid = torch.cat(
                (
                    torch.ones(
                        batch,
                        units,
                        latents.shape[1],
                        dtype=torch.bool,
                        device=local.device,
                    ),
                    local_valid,
                ),
                dim=2,
            ).reshape(batch * units, latents.shape[1] + slots)
            unit_hidden = self.unit_decoder(
                unit_tokens.reshape(batch * units, 1, width),
                unit_context,
                context_valid=unit_valid,
            ).view(batch, units, width)
        else:
            unit_hidden = self.unit_decoder(unit_tokens, latents)
            unit_hidden = self.unit_local_decoder(
                unit_hidden.reshape(batch * units, 1, width),
                local.reshape(batch * units, slots, width),
                context_norm=self.local_context_norm,
                context_valid=local_valid.reshape(batch * units, slots),
            ).view(batch, units, width)
        unit_hidden = torch.where(inputs.unit_active.unsqueeze(-1), unit_hidden, 0.0)
        unit_hidden = self.unit_head[0](unit_hidden)

        market_queries = self.market_queries.weight.unsqueeze(0).expand(batch, -1, -1)
        if self.market_economy_decoder is None:
            market_context = torch.cat(
                (
                    latents,
                    self.economy_context_norm(economy_tokens),
                ),
                dim=1,
            )
            market_hidden = self.market_decoder(market_queries, market_context)
        else:
            market_hidden = self.market_decoder(market_queries, latents)
            market_hidden = self.market_economy_decoder(
                market_hidden,
                economy_tokens,
                context_norm=self.economy_context_norm,
            )
        market_hidden = self.market_norm(market_hidden)
        return StructuredDecisionBelief(unit_hidden, market_hidden)

    def encode_belief(self, inputs: StructuredInputs) -> StructuredBelief:
        trunk = self.trunk(inputs)
        decisions = self._decode_entities(
            trunk.latents,
            trunk.unit_tokens,
            trunk.unit_local_tiles,
            trunk.economy_tokens,
            inputs,
        )
        return StructuredBelief(
            own_patches=trunk.own_patches,
            opponent_patches=trunk.opponent_patches,
            opponent_summary=trunk.opponent_summary,
            economy_entities=trunk.economy_tokens,
            central_latents=trunk.latents,
            unit_decisions=decisions.unit_decisions,
            market_decisions=decisions.market_decisions,
        )

    def auxiliary_belief(self, inputs: StructuredInputs) -> StructuredDecisionBelief:
        # Keep the auxiliary path memory-safe at production minibatch size by
        # rematerializing the trunk. Both head-input tensors stay source-live.
        trunk = (
            checkpoint(self.trunk, inputs, use_reentrant=False)
            if torch.is_grad_enabled()
            else self.trunk(inputs)
        )
        return self._decode_entities(
            trunk.latents,
            trunk.unit_tokens,
            trunk.unit_local_tiles,
            trunk.economy_tokens,
            inputs,
        )

    def forward_with_auxiliary_belief(
        self, inputs: StructuredInputs
    ) -> tuple[ActorOutput, StructuredDecisionBelief]:
        belief = self.auxiliary_belief(inputs)
        return self.decode_belief(belief), belief

    def decode_belief(self, belief: StructuredBelief | StructuredDecisionBelief) -> ActorOutput:
        return ActorOutput(
            unit_logits=self.unit_head[-1](belief.unit_decisions).contiguous(),
            market_kind_logits=self.market_kind(belief.market_decisions).contiguous(),
            market_quantity_context=self.market_quantity_context(
                belief.market_decisions
            ).contiguous(),
        )

    def forward_with_belief(self, inputs: StructuredInputs) -> tuple[ActorOutput, StructuredBelief]:
        belief = self.encode_belief(inputs)
        return self.decode_belief(belief), belief

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
        value_query = torch.randn(1, config.model_dim)
        self.value_query = nn.Parameter(
            torch.nn.functional.rms_norm(value_query, (config.model_dim,))
            if config.critic_state_read
            else value_query * SMALL_QUERY_INITIAL_SCALE
        )
        if not config.critic_state_read:
            self.adam_learning_rate_multipliers = {
                "value_query": SMALL_QUERY_INITIAL_SCALE,
            }
        self.value_decoder = Block(trunk_config, state_read=config.critic_state_read)
        self.value_norm = RMSNorm(config.model_dim, eps=1e-5)
        self.value_head = Linear(config.model_dim, 1 if config.scalar_value else config.value_atoms)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        if config.per_entity_critic:
            self.entity_queries = nn.Embedding(MAX_UNITS + MAX_MARKET_ORDERS, config.model_dim)
        self.register_buffer(
            "support",
            categorical_value_support(config.value_min, config.value_max, config.value_atoms),
            persistent=True,
        )

    def encode_belief(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> StructuredCriticBelief:
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
        queries = self.value_query.unsqueeze(0).expand(batch, -1, -1)
        if self.config.per_entity_critic:
            unit_queries = trunk.unit_tokens + self.entity_queries.weight[:MAX_UNITS]
            market_queries = (
                _token_mean(trunk.economy_tokens)[:, None, :]
                + self.entity_queries.weight[MAX_UNITS:]
            )
            queries = torch.cat((queries, unit_queries, market_queries), dim=1)
        value_hidden = self.value_decoder(queries, trunk.latents)
        value_hidden = self.value_norm(value_hidden)
        return StructuredCriticBelief(value_decision=value_hidden)

    def decode_belief(self, belief: StructuredCriticBelief) -> Tensor:
        """Global logits (B, bins), including when the belief has entity slots."""
        readout = self.value_head(belief.value_decision[:, 0])
        return (readout if self.config.scalar_value else softcap_value_logits(readout)).contiguous()

    def decode_entity_belief(self, belief: StructuredCriticBelief) -> Tensor:
        """Entity logits (B, MAX_UNITS + MAX_MARKET_ORDERS, bins), never global."""
        if not self.config.per_entity_critic:
            raise ValueError("entity values require per_entity_critic")
        readout = self.value_head(belief.value_decision[:, 1:])
        return (readout if self.config.scalar_value else softcap_value_logits(readout)).contiguous()

    def forward_with_belief(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> tuple[Tensor, StructuredCriticBelief]:
        """One private trunk pass; global logits plus every normalized head input."""
        belief = self.encode_belief(
            inputs, opponent_unit_categorical, opponent_unit_continuous, opponent_unit_active
        )
        return self.decode_belief(belief), belief

    def forward(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> Tensor:
        return self.forward_with_belief(
            inputs,
            opponent_unit_categorical,
            opponent_unit_continuous,
            opponent_unit_active,
        )[0]

    def value(self, logits: Tensor) -> Tensor:
        if self.config.scalar_value:
            return logits.float().squeeze(-1)
        return categorical_value(logits, self.support)
