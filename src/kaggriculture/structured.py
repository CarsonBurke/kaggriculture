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
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import BOARD_SIZE, CROPS, MAX_MARKET_ORDERS, MAX_UNITS, PRODUCTS
from kaggriculture.model import (
    ActorOutput,
    AxialRotaryEmbedding,
    ReluSquared,
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


@dataclass(frozen=True)
class StructuredConfig:
    """Target-architecture hyperparameters from VIT_PLAN."""

    model_dim: int = 128
    attention_heads: int = 4
    ffn_multiplier: int = 4
    farm_blocks: int = 2
    opponent_latents: int = 8
    latents: int = 32
    core_layers: int = 8
    quantity_rank: int = 32
    value_atoms: int = 101
    value_min: float = -2.2
    value_max: float = 2.2
    value_sigma_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.attention_heads <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("attention_heads must evenly divide model_dim")
        if (self.model_dim // self.attention_heads) % 4:
            raise ValueError("attention head width must be divisible by 4 for axial RoPE")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
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
        if self.value_atoms < 2:
            raise ValueError("value_atoms must be at least 2")
        if not math.isfinite(self.value_min) or not math.isfinite(self.value_max):
            raise ValueError("value support bounds must be finite")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not math.isfinite(self.value_sigma_ratio) or self.value_sigma_ratio <= 0:
            raise ValueError("value_sigma_ratio must be finite and positive")

    def to_dict(self) -> dict[str, int | float]:
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
        return residual + self.gate * branch


class Attention(nn.Module):
    """QK-normalized attention over an explicit context (self or cross)."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.model_dim // self.heads
        self.query = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.key_value = nn.Linear(config.model_dim, 2 * config.model_dim, bias=False)
        self.query_norm = nn.RMSNorm(self.head_dim)
        self.key_norm = nn.RMSNorm(self.head_dim)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)

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
            .view(batch, key_tokens, 2, self.heads, self.head_dim)
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
        attended = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0
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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.activation(self.input(inputs)))


class Block(nn.Module):
    """Pre-norm attention + FFN block with near-identity gated residuals."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.model_dim)
        self.attention = Attention(config)
        self.attention_gate = GatedResidual(config.model_dim)
        self.ffn_norm = nn.RMSNorm(config.model_dim)
        self.ffn = FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim)

    def forward(
        self,
        queries: Tensor,
        context: Tensor | None = None,
        *,
        context_norm: nn.Module | None = None,
        query_rotation: tuple[Tensor, Tensor] | None = None,
        key_rotation: tuple[Tensor, Tensor] | None = None,
        context_valid: Tensor | None = None,
    ) -> Tensor:
        normalized = self.attention_norm(queries)
        if context is None:
            keys = normalized
        else:
            keys = context_norm(context) if context_norm is not None else context
        hidden = self.attention_gate(
            queries,
            self.attention(
                normalized,
                keys,
                query_rotation=query_rotation,
                key_rotation=key_rotation,
                context_valid=context_valid,
            ),
        )
        return self.ffn_gate(hidden, self.ffn(self.ffn_norm(hidden)))


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
        self.product_identity = nn.Embedding(len(PRODUCTS), width)
        self.product_projection = nn.Linear(product_width, width)
        self.crop_identity = nn.Embedding(len(CROPS), width)
        self.crop_projection = nn.Linear(crop_width, width)
        self.farm_identity = nn.Embedding(2, width)
        self.farm_projection = nn.Linear(len(FARM_TOKEN_FIELDS), width)
        self.town_projection = nn.Linear(len(TOWN_TOKEN_FIELDS), width)

    def forward(self, products: Tensor, crops: Tensor, farms: Tensor, town: Tensor) -> Tensor:
        dtype = self.product_projection.weight.dtype
        return torch.cat(
            (
                self.product_projection(products.to(dtype)) + self.product_identity.weight,
                self.crop_projection(crops.to(dtype)) + self.crop_identity.weight,
                self.farm_projection(farms.to(dtype)) + self.farm_identity.weight,
                self.town_projection(town.to(dtype)).unsqueeze(1),
            ),
            dim=1,
        )


class TrunkOutput(NamedTuple):
    """Everything the decoders read from the shared encoder."""

    latents: Tensor  # [B, latents, model_dim], RMS-normalized
    unit_tokens: Tensor  # [B, MAX_UNITS, model_dim], inactive rows zeroed
    unit_local_tiles: Tensor  # [B, MAX_UNITS, 5, model_dim]
    economy_tokens: Tensor  # [B, economy tokens, model_dim]


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
        self.opponent_context_norm = nn.RMSNorm(config.model_dim)
        self.latent_queries = nn.Parameter(torch.randn(config.latents, config.model_dim) * 0.02)
        self.latent_read = Block(config)
        self.latent_context_norm = nn.RMSNorm(config.model_dim)
        self.core = nn.ModuleList(Block(config) for _ in range(config.core_layers))
        self.core_norm = nn.RMSNorm(config.model_dim)
        board = torch.stack(
            torch.meshgrid(
                torch.arange(BOARD_SIZE),
                torch.arange(BOARD_SIZE),
                indexing="ij",
            )[::-1],
            dim=-1,
        ).reshape(1, TILE_COUNT, 2)
        self.register_buffer("board_positions", board, persistent=False)

    def encode_farm(self, tiles: Tensor, rotation: tuple[Tensor, Tensor]) -> Tensor:
        """Run the shared farm-local blocks over one farm's 100 tile tokens."""
        hidden = tiles
        for block in self.farm_local:
            hidden = block(hidden, query_rotation=rotation, key_rotation=rotation)
        return hidden

    def forward(
        self,
        inputs: StructuredInputs,
        *,
        opponent_units: Tensor | None = None,
        opponent_units_active: Tensor | None = None,
    ) -> TrunkOutput:
        batch = inputs.tile_categorical.shape[0]
        tiles = self.tiles(inputs.tile_categorical, inputs.tile_continuous)
        rotation = self.rope.rotation(self.board_positions.expand(batch, -1, -1))
        own_tiles = self.encode_farm(tiles[:, :TILE_COUNT], rotation)
        opponent_tiles = self.encode_farm(tiles[:, TILE_COUNT:], rotation)

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
        for block in self.core:
            latents = block(latents)
        latents = self.core_norm(latents)
        return TrunkOutput(
            latents=latents,
            unit_tokens=unit_tokens,
            unit_local_tiles=local,
            economy_tokens=economy_tokens,
        )


class StructuredActor(nn.Module):
    """Decentralized structured policy with the FarmActor output contract."""

    def __init__(self, config: StructuredConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredConfig()
        self.config = config
        self.trunk = StructuredTrunk(config, private_columns=False)
        self.unit_decoder = Block(config)
        self.unit_local_decoder = Block(config)
        # The trunk's latents leave core_norm already normalized; raw local
        # tiles and economy tokens each get their own context norm.
        self.local_context_norm = nn.RMSNorm(config.model_dim)
        self.market_queries = nn.Embedding(MAX_MARKET_ORDERS, config.model_dim)
        self.market_decoder = Block(config)
        self.market_economy_decoder = Block(config)
        self.economy_context_norm = nn.RMSNorm(config.model_dim)

        self.unit_head = nn.Sequential(
            nn.RMSNorm(config.model_dim),
            nn.Linear(config.model_dim, N_UNIT_ACTIONS),
        )
        self.market_norm = nn.RMSNorm(config.model_dim)
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

    def forward(self, inputs: StructuredInputs) -> ActorOutput:
        batch = inputs.tile_categorical.shape[0]
        trunk = self.trunk(inputs)

        unit_hidden = self.unit_decoder(trunk.unit_tokens, trunk.latents)
        # Relation-aware read of each unit's own and NSEW tiles: fold the
        # per-unit local context into a batch of 5-key attention windows.
        local = trunk.unit_local_tiles
        units, slots, width = local.shape[1], local.shape[2], local.shape[3]
        # Inactive units have no valid gathers; SDPA turns a fully masked row
        # into NaN, whose backward poisons shared parameters even though the
        # forward values are discarded below. Opening the HERE slot for them
        # keeps every attention row attendable.
        local_valid = inputs.unit_tile_gather_valid.clone()
        local_valid[..., 0] |= ~inputs.unit_active
        unit_hidden = self.unit_local_decoder(
            unit_hidden.reshape(batch * units, 1, width),
            local.reshape(batch * units, slots, width),
            context_norm=self.local_context_norm,
            context_valid=local_valid.reshape(batch * units, slots),
        ).view(batch, units, width)
        unit_hidden = torch.where(inputs.unit_active.unsqueeze(-1), unit_hidden, 0.0)

        market_hidden = self.market_decoder(
            self.market_queries.weight.unsqueeze(0).expand(batch, -1, -1),
            trunk.latents,
        )
        market_hidden = self.market_economy_decoder(
            market_hidden, trunk.economy_tokens, context_norm=self.economy_context_norm
        )
        market_hidden = self.market_norm(market_hidden)
        return ActorOutput(
            unit_logits=self.unit_head(unit_hidden).contiguous(),
            market_kind_logits=self.market_kind(market_hidden).contiguous(),
            market_quantity_context=self.market_quantity_context(market_hidden).contiguous(),
        )


class StructuredCritic(nn.Module):
    """Centralized structured critic over both players' exact private state."""

    def __init__(self, config: StructuredConfig | None = None) -> None:
        super().__init__()
        config = config or StructuredConfig()
        self.config = config
        self.trunk = StructuredTrunk(config, private_columns=True)
        self.value_query = nn.Parameter(torch.randn(1, config.model_dim) * 0.02)
        self.value_decoder = Block(config)
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
