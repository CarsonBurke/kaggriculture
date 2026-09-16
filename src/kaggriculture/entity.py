"""Entity-only policies: 26 decision states repeatedly read fixed farm memory."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.model import (
    ActorOutput,
    AxialRotaryEmbedding,
    Linear,
    RMSNorm,
    _sdpa_inputs,
    categorical_value,
    categorical_value_support,
    factored_quantity_logits,
    initialize_policy_heads,
    softcap_value_logits,
)
from kaggriculture.structured import (
    Attention,
    Block,
    EconomyEmbedder,
    FeedForward,
    FusedFeedForward,
    GatedResidual,
    StructuredCriticBelief,
    StructuredDecisionBelief,
    StructuredInputs,
    TileEmbedder,
    UnitEmbedder,
    _fused_attention,
    _token_mean,
)
from kaggriculture.tokens import OBSERVATION_SCHEMA_VERSION, TILE_COUNT


@dataclass(frozen=True)
class EntityConfig:
    """Only fields used by the entity architecture and its training heads."""

    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    model_dim: int = 96
    attention_heads: int = 4
    attention_kv_heads: int = 2
    ffn_multiplier: int = 2
    farm_blocks: int = 2
    core_layers: int = 4
    quantity_rank: int = 32
    global_modulation: bool = True
    zero_init_branches: bool = False
    fused_mlp: bool = False
    split_clock_token: bool = False
    shared_memory_kv: bool = True
    inter_attention_ffn: bool = False
    unit_local_readout: bool = False
    critic_readout_ffn: bool = False
    unit_local_init: bool = True
    tile_cross_rope: bool = False
    value_atoms: int = 255
    value_min: float = -2.2
    value_max: float = 2.2
    value_sigma_ratio: float = 3.0
    scalar_value: bool = False

    def __post_init__(self) -> None:
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("stale entity observation schema; fresh encoding required")
        for name in (
            "model_dim",
            "attention_heads",
            "attention_kv_heads",
            "ffn_multiplier",
            "farm_blocks",
            "core_layers",
            "quantity_rank",
            "value_atoms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_dim % self.attention_heads:
            raise ValueError("attention_heads must evenly divide model_dim")
        if (self.model_dim // self.attention_heads) % 4:
            raise ValueError("attention head width must be divisible by 4 for axial RoPE")
        if (
            self.attention_kv_heads >= self.attention_heads
            or self.attention_heads % self.attention_kv_heads
        ):
            raise ValueError("entity GQA requires fewer KV heads that divide query heads")
        if self.fused_mlp and (self.model_dim % 128 or self.model_dim * self.ffn_multiplier % 256):
            raise ValueError(
                "fused MLP requires model width divisible by 128 and hidden width by 256"
            )
        if self.split_clock_token:
            raise ValueError("entity memory requires the single town token")
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


class EntityMemory(nn.Module):
    """Normalize source memory once, then project shared or round-local K/V."""

    def __init__(self, config: EntityConfig, *, normalize: bool = True) -> None:
        super().__init__()
        self.kv_heads = config.attention_kv_heads
        self.head_dim = config.model_dim // config.attention_heads
        self.norm = RMSNorm(config.model_dim) if normalize else None
        self.key_value = Linear(config.model_dim, 2 * self.kv_heads * self.head_dim, bias=False)
        self.key_norm = RMSNorm(self.head_dim)

    def forward(
        self, memory: Tensor, tile_rotation: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        batch, tokens, _width = memory.shape
        key, value = (
            self.key_value(self.norm(memory) if self.norm is not None else memory)
            .view(batch, tokens, 2, self.kv_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        key = self.key_norm(key)
        if tile_rotation is not None:
            # Both farms use their own board coordinates; ownership stays encoded.
            tiles = key[:, :, : 2 * TILE_COUNT].unflatten(-2, (2, TILE_COUNT))
            cosine, sine = tile_rotation
            tiles = tiles * cosine.to(key.dtype) + AxialRotaryEmbedding._rotate_pairs(
                tiles
            ) * sine.to(key.dtype)
            key = torch.cat((tiles.flatten(-3, -2), key[:, :, 2 * TILE_COUNT :]), dim=-2)
        return key, value


class EntityMemoryRead(nn.Module):
    """Round-specific Q/output projections without another memory projection."""

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.kv_heads = config.attention_kv_heads
        self.head_dim = config.model_dim // self.heads
        self.query = Linear(config.model_dim, config.model_dim, bias=False)
        self.query_norm = RMSNorm(self.head_dim)
        self.output = Linear(config.model_dim, config.model_dim, bias=False)
        if config.zero_init_branches:
            nn.init.zeros_(self.output.weight)

    def forward(
        self,
        queries: Tensor,
        key: Tensor,
        value: Tensor,
        memory_valid: Tensor | None,
        unit_rotation: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch, tokens, width = queries.shape
        query = self.query(queries).view(batch, tokens, self.heads, self.head_dim)
        query = self.query_norm(query.transpose(1, 2))
        if unit_rotation is not None:
            # Market queries have no spatial coordinate and remain unrotated.
            units = query[:, :, :MAX_UNITS]
            cosine, sine = unit_rotation
            units = units * cosine.to(query.dtype) + AxialRotaryEmbedding._rotate_pairs(
                units
            ) * sine.to(query.dtype)
            query = torch.cat((units, query[:, :, MAX_UNITS:]), dim=-2)
        query, key, value = _sdpa_inputs(query, key, value)
        mask = None if memory_valid is None else memory_valid[:, None, None, :]
        attended = _fused_attention(
            query,
            key,
            value,
            mask,
            enable_gqa=self.heads != self.kv_heads,
            scale=self.head_dim**-0.5,
        )
        attended = attended.to(queries.dtype).transpose(1, 2).reshape(batch, tokens, width)
        return self.output(attended)


class EntityRound(nn.Module):
    """Self attention and memory cross attention, each optionally followed by an FFN."""

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.self_norm = RMSNorm(config.model_dim)
        self.self_attention = Attention(config)
        self.self_gate = GatedResidual(config.model_dim)
        self.cross_norm = RMSNorm(config.model_dim)
        self.cross_attention = EntityMemoryRead(config)
        self.cross_gate = GatedResidual(config.model_dim)
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = FusedFeedForward(config) if config.fused_mlp else FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim)
        self.memory = None if config.shared_memory_kv else EntityMemory(config, normalize=False)
        self.inter_ffn_norm = RMSNorm(config.model_dim) if config.inter_attention_ffn else None
        self.inter_ffn = (
            (FusedFeedForward(config) if config.fused_mlp else FeedForward(config))
            if config.inter_attention_ffn
            else None
        )
        self.inter_ffn_gate = (
            GatedResidual(config.model_dim) if config.inter_attention_ffn else None
        )
        self.modulation_chunks = 8 if config.inter_attention_ffn else 6
        self.modulation = (
            Linear(config.model_dim, self.modulation_chunks * config.model_dim)
            if config.global_modulation
            else None
        )
        if self.modulation is not None:
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        states: Tensor,
        key: Tensor,
        value: Tensor | None,
        state_valid: Tensor,
        memory_valid: Tensor | None,
        conditioning: Tensor | None,
        *,
        unit_rotation: tuple[Tensor, Tensor] | None = None,
        tile_rotation: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if self.memory is not None:
            key, value = self.memory(key, tile_rotation)
            states = states.to(key.dtype)
        if value is None:
            raise ValueError("shared entity round requires projected memory values")
        modulation = None
        if self.modulation is not None:
            if conditioning is None:
                raise ValueError("conditioned entity round requires economy conditioning")
            modulation = (
                self.modulation(conditioning).to(states.dtype).chunk(self.modulation_chunks, dim=-1)
            )
        valid = state_valid.unsqueeze(-1)
        self_input = self.self_norm(states)
        if modulation is not None:
            self_input = self_input * (1 + modulation[0][:, None]) + modulation[1][:, None]
        states = torch.where(
            valid,
            self.self_gate(
                states, self.self_attention(self_input, self_input, context_valid=state_valid)
            ),
            0.0,
        )
        if self.inter_ffn is not None:
            assert self.inter_ffn_norm is not None and self.inter_ffn_gate is not None
            inter_input = self.inter_ffn_norm(states)
            if modulation is not None:
                # Append channels: the original self/cross/final-FFN order stays fixed.
                inter_input = inter_input * (1 + modulation[6][:, None]) + modulation[7][:, None]
            states = torch.where(
                valid, self.inter_ffn_gate(states, self.inter_ffn(inter_input)), 0.0
            )
        cross_input = self.cross_norm(states)
        if modulation is not None:
            cross_input = cross_input * (1 + modulation[2][:, None]) + modulation[3][:, None]
        states = torch.where(
            valid,
            self.cross_gate(
                states, self.cross_attention(cross_input, key, value, memory_valid, unit_rotation)
            ),
            0.0,
        )
        ffn_input = self.ffn_norm(states)
        if modulation is not None:
            ffn_input = ffn_input * (1 + modulation[4][:, None]) + modulation[5][:, None]
        return torch.where(valid, self.ffn_gate(states, self.ffn(ffn_input)), 0.0)


class EntityTrunk(nn.Module):
    """Both farms are static memory; only 16 units and 10 order slots evolve."""

    def __init__(self, config: EntityConfig, *, private_columns: bool) -> None:
        super().__init__()
        self.config = config
        self.tiles = TileEmbedder(config)
        local_readout = config.unit_local_readout and not private_columns
        self.units = UnitEmbedder(
            config, local_init=config.unit_local_init, local_context=local_readout
        )
        self.economy = EconomyEmbedder(config, private_columns=private_columns)
        self.rope = AxialRotaryEmbedding(config.model_dim // config.attention_heads)
        self.farm_local = nn.ModuleList(Block(config) for _ in range(config.farm_blocks))
        # Unit-RMS lookup rows follow existing market-query optimizer ownership.
        self.market_queries = nn.Embedding(MAX_MARKET_ORDERS, config.model_dim)
        self.memory = EntityMemory(config) if config.shared_memory_kv else None
        self.memory_norm = None if config.shared_memory_kv else RMSNorm(config.model_dim)
        self.core = nn.ModuleList(EntityRound(config) for _ in range(config.core_layers))
        self.unit_local_decoder = Block(config) if local_readout else None
        self.local_context_norm = RMSNorm(config.model_dim) if local_readout else None

    def forward(
        self,
        inputs: StructuredInputs,
        opponent_units: Tensor | None = None,
        opponent_units_active: Tensor | None = None,
    ) -> Tensor:
        batch = inputs.tile_categorical.shape[0]
        width = self.config.model_dim
        tiles = self.tiles(inputs.tile_categorical, inputs.tile_continuous)
        farms = tiles.reshape(2 * batch, TILE_COUNT, width)
        rotation = (
            self.rope.cosine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
            self.rope.sine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
        )
        # No tile-validity mask: locked squares still carry real public state.
        for block in self.farm_local:
            farms = block(farms, query_rotation=rotation, key_rotation=rotation)
        tiles = farms.reshape(batch, 2 * TILE_COUNT, width)
        local = (
            self.units.local_tiles(
                tiles[:, :TILE_COUNT], inputs.unit_tile_gather, inputs.unit_tile_gather_valid
            )
            if self.config.unit_local_init or self.unit_local_decoder is not None
            else None
        )
        units = self.units(
            inputs.unit_categorical,
            inputs.unit_continuous,
            inputs.unit_active,
            local,
            opponent=False,
        )
        economy = self.economy(
            inputs.products, inputs.animals, inputs.crops, inputs.farms, inputs.town
        )
        economy_mean = _token_mean(economy)
        markets = economy_mean[:, None] + self.market_queries.weight
        states = torch.cat((units, markets), dim=1)
        state_valid = torch.cat(
            (
                inputs.unit_active,
                torch.ones(batch, MAX_MARKET_ORDERS, dtype=torch.bool, device=states.device),
            ),
            dim=1,
        )
        memory_valid = None
        memory = torch.cat((tiles, economy), dim=1)
        if opponent_units is not None:
            if opponent_units_active is None:
                raise ValueError("opponent unit memory requires an active mask")
            memory_valid = torch.cat(
                (
                    torch.ones(batch, memory.shape[1], dtype=torch.bool, device=memory.device),
                    opponent_units_active,
                ),
                dim=1,
            )
            memory = torch.cat((memory, opponent_units), dim=1)
        unit_rotation = tile_rotation = None
        if self.config.tile_cross_rope:
            positions = torch.stack(
                (inputs.unit_categorical[..., 3], inputs.unit_categorical[..., 2]), dim=-1
            )
            unit_rotation = self.rope.rotation(positions)
            tile_rotation = (self.rope.cosine, self.rope.sine)
        if self.memory is not None:
            key, value = self.memory(memory, tile_rotation)
            # Lookup embeddings seed FP32 tensors under autocast; enter the round
            # compute dtype once rather than promoting its adaptive RMS branches.
            states = states.to(key.dtype)
        else:
            # Normalize the common source once; only projected K/V are round-local.
            assert self.memory_norm is not None
            key, value = self.memory_norm(memory), None
        conditioning = economy_mean if self.config.global_modulation else None
        for block in self.core:
            states = block(
                states,
                key,
                value,
                state_valid,
                memory_valid,
                conditioning,
                unit_rotation=unit_rotation,
                tile_rotation=tile_rotation,
            )
        if self.unit_local_decoder is not None:
            assert local is not None and self.local_context_norm is not None
            units, markets = states.split((MAX_UNITS, MAX_MARKET_ORDERS), dim=1)
            slots = local.shape[2]
            local_valid = inputs.unit_tile_gather_valid.clone()
            # Keep inactive rows safe for attention, then discard their decoded state.
            local_valid[..., 0] |= ~inputs.unit_active
            units = self.unit_local_decoder(
                units.reshape(batch * MAX_UNITS, 1, width),
                local.reshape(batch * MAX_UNITS, slots, width),
                context_norm=self.local_context_norm,
                context_valid=local_valid.reshape(batch * MAX_UNITS, slots),
            ).view(batch, MAX_UNITS, width)
            units = torch.where(inputs.unit_active.unsqueeze(-1), units, 0.0)
            states = torch.cat((units, markets), dim=1)
        return states


class EntityActor(nn.Module):
    """Decentralized actor with direct, normalized decision-state readouts."""

    def __init__(self, config: EntityConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or EntityConfig()
        self.trunk = EntityTrunk(config, private_columns=False)
        self.unit_head = nn.Sequential(
            RMSNorm(config.model_dim), Linear(config.model_dim, N_UNIT_ACTIONS)
        )
        self.market_norm = RMSNorm(config.model_dim)
        self.market_kind = Linear(config.model_dim, N_MARKET_KINDS)
        self.market_quantity_context = Linear(config.model_dim, config.quantity_rank, bias=False)
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
        return factored_quantity_logits(
            quantity_context,
            market_kinds,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
            self.config.quantity_rank,
        )

    def _head_belief(self, states: Tensor) -> StructuredDecisionBelief:
        units, markets = states.split((MAX_UNITS, MAX_MARKET_ORDERS), dim=1)
        return StructuredDecisionBelief(self.unit_head[0](units), self.market_norm(markets))

    def encode_belief(self, inputs: StructuredInputs) -> StructuredDecisionBelief:
        return self._head_belief(self.trunk(inputs))

    def auxiliary_belief(self, inputs: StructuredInputs) -> StructuredDecisionBelief:
        states = (
            checkpoint(self.trunk, inputs, use_reentrant=False)
            if torch.is_grad_enabled()
            else self.trunk(inputs)
        )
        return self._head_belief(states)

    def decode_belief(self, belief: StructuredDecisionBelief) -> ActorOutput:
        return ActorOutput(
            unit_logits=self.unit_head[-1](belief.unit_decisions).contiguous(),
            market_kind_logits=self.market_kind(belief.market_decisions).contiguous(),
            market_quantity_context=self.market_quantity_context(
                belief.market_decisions
            ).contiguous(),
        )

    def forward_with_belief(
        self, inputs: StructuredInputs
    ) -> tuple[ActorOutput, StructuredDecisionBelief]:
        belief = self.encode_belief(inputs)
        return self.decode_belief(belief), belief

    def forward_with_auxiliary_belief(
        self, inputs: StructuredInputs
    ) -> tuple[ActorOutput, StructuredDecisionBelief]:
        belief = self.auxiliary_belief(inputs)
        return self.decode_belief(belief), belief

    def forward(self, inputs: StructuredInputs) -> ActorOutput:
        return self.forward_with_belief(inputs)[0]


class EntityCritic(nn.Module):
    """Independent private-state trunk with one GQA value query over 26 states."""

    def __init__(self, config: EntityConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or EntityConfig()
        self.trunk = EntityTrunk(config, private_columns=True)
        self.pool_norm = RMSNorm(config.model_dim)
        self.value_query = nn.Parameter(
            torch.nn.functional.rms_norm(torch.randn(1, config.model_dim), (config.model_dim,))
        )
        # This is a state read, not a residual branch with a live shortcut.
        readout_config = (
            replace(config, zero_init_branches=False) if config.zero_init_branches else config
        )
        self.pool_attention = Attention(readout_config)
        self.value_ffn_norm = RMSNorm(config.model_dim) if config.critic_readout_ffn else None
        self.value_ffn = (
            (FusedFeedForward(config) if config.fused_mlp else FeedForward(config))
            if config.critic_readout_ffn
            else None
        )
        self.value_ffn_gate = GatedResidual(config.model_dim) if config.critic_readout_ffn else None
        self.value_norm = RMSNorm(config.model_dim, eps=1e-5)
        self.value_head = Linear(config.model_dim, 1 if config.scalar_value else config.value_atoms)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
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
        # Private categorical/continuous features stay present without local init.
        # With it enabled, preserve the original relation-only opponent seed.
        relation_only = None
        if self.config.unit_local_init:
            assert self.trunk.units.gather_relation is not None
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
        states = self.trunk(inputs, opponent_units, opponent_unit_active)
        valid = torch.cat(
            (
                inputs.unit_active,
                torch.ones(batch, MAX_MARKET_ORDERS, dtype=torch.bool, device=states.device),
            ),
            dim=1,
        )
        query = self.value_query.unsqueeze(0).expand(batch, -1, -1)
        pooled = self.pool_attention(query, self.pool_norm(states), context_valid=valid)
        if self.value_ffn is not None:
            assert self.value_ffn_norm is not None and self.value_ffn_gate is not None
            pooled = self.value_ffn_gate(pooled, self.value_ffn(self.value_ffn_norm(pooled)))
        return StructuredCriticBelief(self.value_norm(pooled))

    def decode_belief(self, belief: StructuredCriticBelief) -> Tensor:
        readout = self.value_head(belief.value_decision[:, 0])
        return (readout if self.config.scalar_value else softcap_value_logits(readout)).contiguous()

    def forward_with_belief(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> tuple[Tensor, StructuredCriticBelief]:
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
            inputs, opponent_unit_categorical, opponent_unit_continuous, opponent_unit_active
        )[0]

    def value(self, logits: Tensor) -> Tensor:
        if self.config.scalar_value:
            return logits.float().squeeze(-1)
        return categorical_value(logits, self.support)
