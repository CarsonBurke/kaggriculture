"""Entity decision heads with a fixed-memory default and experimental reasoning cores."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.market_set import N_MARKET_SET_KINDS, MarketSetOrder
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
    SMALL_QUERY_INITIAL_SCALE,
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
from kaggriculture.tokens import (
    DEFAULT_OBSERVATION_SCHEMA_VERSION,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TILE_COUNT,
)


@dataclass(frozen=True)
class EntityConfig:
    """Only fields used by the entity architecture and its training heads."""

    observation_schema_version: int = DEFAULT_OBSERVATION_SCHEMA_VERSION
    action_interface: int = 1
    market_set_sell_order: str = "fixed"
    market_set_hire_last: bool = False
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
    shared_memory_kv: bool = False
    inter_attention_ffn: bool = False
    unit_local_readout: bool = False
    critic_readout_ffn: bool = False
    unit_local_init: bool = True
    tile_cross_rope: bool = False
    # Entity readout only; economic valuation states always read complete memory.
    critic_source_read: bool = True
    critic_architecture: str = "entity"
    memory_writeback: bool = False
    unit_tile_bias: bool = False
    bixt_latents: int = 0
    value_atoms: int = 255
    value_min: float = -2.2
    value_max: float = 2.2
    value_sigma_ratio: float = 3.0
    scalar_value: bool = False

    def __post_init__(self) -> None:
        if self.action_interface not in (1, 2, 3, 4):
            raise ValueError("action_interface must be 1, 2, 3, or 4")
        MarketSetOrder(self.market_set_sell_order, self.market_set_hire_last)
        if self.action_interface != 3 and (
            self.market_set_sell_order != "fixed" or self.market_set_hire_last
        ):
            raise ValueError("market set order applies only to action_interface=3")
        if self.critic_architecture not in ("entity", "economic", "forecast"):
            raise ValueError("critic_architecture must be 'entity', 'economic', or 'forecast'")
        if self.observation_schema_version not in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS:
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
        if self.unit_tile_bias and self.tile_cross_rope:
            raise ValueError("unit tile bias and tile cross RoPE are separate experiments")
        if self.memory_writeback and self.core_layers < 2:
            raise ValueError("memory writeback requires at least two entity rounds")
        if isinstance(self.bixt_latents, bool) or not isinstance(self.bixt_latents, int):
            raise ValueError("bixt_latents must be a nonnegative integer")
        if self.bixt_latents < 0:
            raise ValueError("bixt_latents must be a nonnegative integer")
        if self.bixt_latents:
            if self.critic_architecture != "entity":
                raise ValueError("BiXT requires critic_architecture='entity'")
            if self.core_layers < 2:
                raise ValueError(
                    "BiXT requires at least two rounds to exchange data through latents"
                )
            incompatible = (
                "global_modulation",
                "shared_memory_kv",
                "inter_attention_ffn",
                "unit_local_readout",
                "tile_cross_rope",
                "critic_source_read",
                "memory_writeback",
                "unit_tile_bias",
                "fused_mlp",
            )
            enabled = [name for name in incompatible if getattr(self, name)]
            if enabled:
                raise ValueError(
                    "BiXT ablation requires these flags disabled: " + ", ".join(enabled)
                )
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
        tile_bias: tuple[Tensor, Tensor] | None = None,
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
        if tile_bias is None:
            attended = _fused_attention(
                query,
                key,
                value,
                mask,
                enable_gqa=self.heads != self.kv_heads,
                scale=self.head_dim**-0.5,
            )
        else:
            from kaggriculture.relative_attention import unit_tile_attention

            attended = unit_tile_attention(
                query, key, value, memory_valid, *tile_bias, scale=self.head_dim**-0.5
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
        tile_bias: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if self.memory is not None:
            key, value = self.memory(key, tile_rotation)
            states = states.to(key.dtype)
        if value is None:
            raise ValueError("shared entity round requires projected memory values")
        modulation: tuple[Tensor, ...] | None = None
        if self.modulation is not None:
            if conditioning is None:
                raise ValueError("conditioned entity round requires economy conditioning")
            modulation = tuple(
                self.modulation(conditioning).to(states.dtype).chunk(self.modulation_chunks, dim=-1)
            )

        def modulated(norm: RMSNorm, inputs: Tensor, index: int) -> Tensor:
            # Each pre-norm fuses its `(1 + scale) * x + shift` conditioning.
            if modulation is None:
                return norm(inputs)
            return norm(inputs, modulation[index], modulation[index + 1])

        valid = state_valid.unsqueeze(-1)
        self_input = modulated(self.self_norm, states, 0)
        states = torch.where(
            valid,
            self.self_gate(
                states, self.self_attention(self_input, self_input, context_valid=state_valid)
            ),
            0.0,
        )
        if self.inter_ffn is not None:
            assert self.inter_ffn_norm is not None and self.inter_ffn_gate is not None
            # Appended channels: the original self/cross/final-FFN order stays fixed.
            inter_input = modulated(self.inter_ffn_norm, states, 6)
            states = torch.where(
                valid, self.inter_ffn_gate(states, self.inter_ffn(inter_input)), 0.0
            )
        cross_input = modulated(self.cross_norm, states, 2)
        cross_update = (
            self.cross_attention(cross_input, key, value, memory_valid, unit_rotation)
            if tile_bias is None
            else self.cross_attention(
                cross_input, key, value, memory_valid, unit_rotation, tile_bias
            )
        )
        states = torch.where(
            valid,
            self.cross_gate(states, cross_update),
            0.0,
        )
        ffn_input = modulated(self.ffn_norm, states, 4)
        return torch.where(valid, self.ffn_gate(states, self.ffn(ffn_input)), 0.0)


class EntityTrunk(nn.Module):
    """Decision-token trunk with fixed memory or an opt-in source-updating core."""

    def __init__(
        self,
        config: EntityConfig,
        *,
        private_columns: bool,
        market_count: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.market_count = (
            market_count
            if market_count is not None
            else N_MARKET_SET_KINDS
            if config.action_interface == 3
            else MAX_MARKET_ORDERS
        )
        self.tiles = TileEmbedder(config)
        local_readout = config.unit_local_readout and not private_columns
        self.units = UnitEmbedder(
            config, local_init=config.unit_local_init, local_context=local_readout
        )
        self.economy = EconomyEmbedder(config, private_columns=private_columns)
        self.rope = AxialRotaryEmbedding(config.model_dim // config.attention_heads)
        self.farm_local = nn.ModuleList(Block(config) for _ in range(config.farm_blocks))
        # Unit-RMS lookup rows follow existing market-query optimizer ownership.
        self.market_queries = nn.Embedding(self.market_count, config.model_dim)
        self.memory = EntityMemory(config) if config.shared_memory_kv else None
        self.memory_norm = (
            None if config.shared_memory_kv or config.bixt_latents else RMSNorm(config.model_dim)
        )
        self.bixt_latents = None
        if config.bixt_latents:
            from kaggriculture.bixt import BiXTRound

            self.bixt_latents = nn.Embedding(config.bixt_latents, config.model_dim)
            nn.init.trunc_normal_(self.bixt_latents.weight, std=SMALL_QUERY_INITIAL_SCALE)
            self.bixt_latents.adam_learning_rate_multipliers = {"weight": SMALL_QUERY_INITIAL_SCALE}
            self.core = nn.ModuleList(
                BiXTRound(
                    config,
                    refine_latents=index < config.core_layers - 1,
                    output_tokens=(
                        MAX_UNITS + self.market_count if index >= config.core_layers - 2 else None
                    ),
                )
                for index in range(config.core_layers)
            )
        else:
            self.core = nn.ModuleList(EntityRound(config) for _ in range(config.core_layers))
        # Additional source paths retain large activations at production B8192.
        # Replay complete rounds, including their private K/V projections; a
        # critic-only flag must not change actor execution. Fused MLPs maintain
        # delayed-scaling state and must not be replayed here.
        self.rematerialize = not config.fused_mlp and (
            config.memory_writeback
            or config.bixt_latents > 0
            or (private_columns and config.critic_source_read)
        )
        self.unit_local_decoder = Block(config) if local_readout else None
        self.local_context_norm = RMSNorm(config.model_dim) if local_readout else None
        self.source_writeback = Block(config) if config.memory_writeback else None
        self.writeback_context_norm = RMSNorm(config.model_dim) if config.memory_writeback else None
        self.tile_bias = (
            nn.Embedding(19 * 19, config.attention_heads) if config.unit_tile_bias else None
        )
        if self.tile_bias is not None:
            nn.init.zeros_(self.tile_bias.weight)

    def forward(
        self,
        inputs: StructuredInputs,
        opponent_units: Tensor | None = None,
        opponent_units_active: Tensor | None = None,
    ) -> Tensor:
        states, _memory, _memory_valid = self.forward_with_memory(
            inputs, opponent_units, opponent_units_active
        )
        return states

    def forward_with_memory(
        self,
        inputs: StructuredInputs,
        opponent_units: Tensor | None = None,
        opponent_units_active: Tensor | None = None,
        *,
        rematerialize_farms: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """States, source memory and its validity from one trunk pass.

        The farm blocks run over every tile token, and so hold most of what a
        training pass retains. `rematerialize_farms` replays them in backward
        instead, for the one caller whose update cannot afford to retain them
        (`LejepaBackbone.rematerialized`); every other pass keeps them, because
        the replay costs a second farm forward per step.
        """
        batch = inputs.tile_categorical.shape[0]
        width = self.config.model_dim
        tiles = self.tiles(inputs.tile_categorical, inputs.tile_continuous)
        farms = tiles.reshape(2 * batch, TILE_COUNT, width)
        rotation = (
            self.rope.cosine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
            self.rope.sine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
        )
        # No tile-validity mask: locked squares still carry real public state.
        rematerialize_farms = rematerialize_farms and torch.is_grad_enabled()
        for block in self.farm_local:
            farms = (
                checkpoint(
                    block,
                    farms,
                    use_reentrant=False,
                    query_rotation=rotation,
                    key_rotation=rotation,
                )
                if rematerialize_farms
                else block(farms, query_rotation=rotation, key_rotation=rotation)
            )
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
                torch.ones(batch, self.market_count, dtype=torch.bool, device=states.device),
            ),
            dim=1,
        )
        memory_valid = None
        # Source memory enters the compute dtype once, with the tiles that are
        # already in it: every consumer projects it under autocast anyway, and
        # an fp32 concatenation over ~236 tokens per row is the largest saved
        # activation of the trunk at production minibatch size.
        memory = torch.cat((tiles, economy.to(tiles.dtype)), dim=1)
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
            memory = torch.cat((memory, opponent_units.to(tiles.dtype)), dim=1)
        if self.bixt_latents is not None:
            tokens = torch.cat((states, memory), dim=1)
            valid = torch.cat(
                (
                    state_valid,
                    torch.ones_like(memory[..., 0], dtype=torch.bool)
                    if memory_valid is None
                    else memory_valid,
                ),
                dim=1,
            )
            latents = self.bixt_latents.weight.unsqueeze(0).expand(batch, -1, -1)
            for index, block in enumerate(self.core):
                if index == len(self.core) - 2:
                    # Later memory-token writes cannot reach the decision heads.
                    # Retain the last live memory for the shared trunk interface.
                    memory = tokens[:, MAX_UNITS + self.market_count :]
                latents, tokens = (
                    checkpoint(block, latents, tokens, valid, use_reentrant=False)
                    if self.rematerialize and torch.is_grad_enabled()
                    else block(latents, tokens, valid)
                )
                valid = valid[:, : tokens.shape[1]]
            return tokens, memory, memory_valid
        unit_rotation = tile_rotation = None
        if self.config.tile_cross_rope or self.tile_bias is not None:
            positions = torch.stack(
                (inputs.unit_categorical[..., 3], inputs.unit_categorical[..., 2]), dim=-1
            )
            if self.config.tile_cross_rope:
                unit_rotation = self.rope.rotation(positions)
                tile_rotation = (self.rope.cosine, self.rope.sine)
        tile_bias = None if self.tile_bias is None else (self.tile_bias.weight, positions)
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
        for index, block in enumerate(self.core):
            round_args = (
                states,
                key,
                value,
                state_valid,
                memory_valid,
                conditioning,
            )
            round_kwargs = {
                "unit_rotation": unit_rotation,
                "tile_rotation": tile_rotation,
                "tile_bias": tile_bias,
            }
            states = (
                checkpoint(block, *round_args, use_reentrant=False, **round_kwargs)
                if self.rematerialize and torch.is_grad_enabled()
                else block(*round_args, **round_kwargs)
            )
            if self.source_writeback is not None and index + 1 == len(self.core) // 2:
                writeback_kwargs = {
                    "context_norm": self.writeback_context_norm,
                    "context_valid": state_valid,
                }
                memory = (
                    checkpoint(
                        self.source_writeback,
                        memory,
                        states,
                        use_reentrant=False,
                        **writeback_kwargs,
                    )
                    if self.rematerialize and torch.is_grad_enabled()
                    else self.source_writeback(memory, states, **writeback_kwargs)
                )
                if memory_valid is not None:
                    memory = torch.where(memory_valid.unsqueeze(-1), memory, 0.0)
                if self.memory is not None:
                    key, value = self.memory(memory, tile_rotation)
                else:
                    key, value = self.memory_norm(memory), None
        if self.unit_local_decoder is not None:
            assert local is not None and self.local_context_norm is not None
            units, markets = states.split((MAX_UNITS, self.market_count), dim=1)
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
        return states, memory, memory_valid


class EntityActor(nn.Module):
    """Decentralized actor with direct, normalized decision-state readouts."""

    def __init__(self, config: EntityConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or EntityConfig()
        self.trunk = self._build_trunk(config)
        self._initialize_heads(config)

    def _build_trunk(self, config: EntityConfig) -> nn.Module:
        """The family's encoder, built before the heads so the RNG stream is fixed.

        A hook beside `_initialize_heads` rather than a reassignment afterwards:
        a family that swaps the trunk would otherwise draw one trunk's worth of
        initialization and discard it, moving every head's draw and with it the
        seed any two architectures are compared under.
        """
        return EntityTrunk(config, private_columns=False)

    def _initialize_heads(self, config: EntityConfig) -> None:
        """Shared factor readouts; each actor family supplies its own representation."""
        self.unit_head = nn.Sequential(
            RMSNorm(config.model_dim), Linear(config.model_dim, N_UNIT_ACTIONS)
        )
        self.market_norm = RMSNorm(config.model_dim)
        self.market_kind = (
            None if config.action_interface == 3 else Linear(config.model_dim, N_MARKET_KINDS)
        )
        self.market_quantity_context = Linear(config.model_dim, config.quantity_rank, bias=False)
        self.market_quantity_kind_gate = nn.Embedding(N_MARKET_KINDS, config.quantity_rank)
        quantity_rows = (
            7
            if config.action_interface == 4
            else N_QUANTITIES + (config.action_interface == 2) + 2 * (config.action_interface == 3)
        )
        self.market_quantity_value = nn.Embedding(quantity_rows, config.quantity_rank)
        self.market_quantity_bias = nn.Parameter(torch.zeros(N_MARKET_KINDS, quantity_rows))
        if config.action_interface == 3:
            order = MarketSetOrder(config.market_set_sell_order, config.market_set_hire_last)
            self.register_buffer(
                "market_set_kind_ids",
                torch.as_tensor([int(kind) for kind in order.decision_kinds]),
                persistent=False,
            )
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
        if self.config.action_interface == 3:
            raise ValueError("interface-3 actor uses market_set_logits")
        return factored_quantity_logits(
            quantity_context,
            market_kinds,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
            self.config.quantity_rank,
            quantity_mask,
        )

    def _head_belief(self, states: Tensor) -> StructuredDecisionBelief:
        units, markets = states.split((MAX_UNITS, self.trunk.market_count), dim=1)
        return StructuredDecisionBelief(self.unit_head[0](units), self.market_norm(markets))

    def market_set_logits(self, context: Tensor, mask: Tensor) -> Tensor:
        """Score 21 effective per-kind values, marginalizing the ALL alias."""
        if self.config.action_interface != 3:
            raise ValueError("market set logits require action_interface=3")
        if context.shape[-2:] != (N_MARKET_SET_KINDS, self.config.quantity_rank):
            raise ValueError("market set context must have 21 kind rows")
        if mask.shape != (*context.shape[:-1], N_QUANTITIES + 1):
            raise ValueError("market set mask must have 101 effective values")
        with torch.autocast(context.device.type, enabled=False):
            kinds = self.market_set_kind_ids
            gated = context.float() * (1.0 + self.market_quantity_kind_gate(kinds).float())
            values = self.market_quantity_value.weight.float()
            scores = self.market_quantity_bias[kinds].float()
            for rank in range(self.config.quantity_rank):
                scores = scores + gated[..., rank, None] * values[:, rank]
            maximum = mask.long().sum(-1).sub(1).clamp_min(0)
            at_max = scores[..., : N_QUANTITIES + 1].gather(-1, maximum[..., None])
            merged = torch.logaddexp(at_max, scores[..., N_QUANTITIES + 1 :])
            merged = torch.where(mask[..., 1:].any(-1, keepdim=True), merged, at_max)
            return scores[..., : N_QUANTITIES + 1].scatter(-1, maximum[..., None], merged)

    def encode_belief(self, inputs: StructuredInputs) -> StructuredDecisionBelief:
        return self._head_belief(self.trunk(inputs))

    def auxiliary_belief(
        self, inputs: StructuredInputs, *, rematerialize: bool = True
    ) -> StructuredDecisionBelief:
        """The decision belief an auxiliary reads; `rematerialize` replays the trunk."""
        states = (
            checkpoint(self.trunk, inputs, use_reentrant=False)
            if rematerialize and torch.is_grad_enabled()
            else self.trunk(inputs)
        )
        return self._head_belief(states)

    def decode_belief(self, belief: StructuredDecisionBelief) -> ActorOutput:
        context = self.market_quantity_context(belief.market_decisions).contiguous()
        return ActorOutput(
            unit_logits=self.unit_head[-1](belief.unit_decisions).contiguous(),
            market_kind_logits=(
                context[..., :0]
                if self.market_kind is None
                else self.market_kind(belief.market_decisions).contiguous()
            ),
            market_quantity_context=context,
        )

    def forward_with_belief(
        self, inputs: StructuredInputs
    ) -> tuple[ActorOutput, StructuredDecisionBelief]:
        belief = self.encode_belief(inputs)
        return self.decode_belief(belief), belief

    def forward_with_auxiliary_belief(
        self, inputs: StructuredInputs, *, rematerialize: bool = True
    ) -> tuple[ActorOutput, StructuredDecisionBelief]:
        belief = self.auxiliary_belief(inputs, rematerialize=rematerialize)
        return self.decode_belief(belief), belief

    def forward(self, inputs: StructuredInputs) -> ActorOutput:
        return self.forward_with_belief(inputs)[0]


class EntityCritic(nn.Module):
    """Pool independent entity/source states or dedicated economic valuation states.

    ``critic_source_read`` controls only the entity readout's source bypass.
    Economic rounds always read all sources and pool only their valuation states.
    """

    def __init__(self, config: EntityConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or EntityConfig()
        if config.critic_architecture in ("economic", "forecast"):
            from kaggriculture.economic_critic import EconomicCriticTrunk

            self.trunk = EconomicCriticTrunk(config)
        else:
            self.trunk = EntityTrunk(
                config,
                private_columns=True,
                market_count=MAX_MARKET_ORDERS if config.action_interface == 3 else None,
            )
        self.pool_norm = RMSNorm(config.model_dim)
        self.source_pool_norm = (
            RMSNorm(config.model_dim)
            if config.critic_source_read and config.critic_architecture == "entity"
            else None
        )
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
        self.forecast_heads = None
        if config.critic_architecture == "forecast":
            from kaggriculture.economic_forecasting import EconomicForecastHeads

            self.forecast_heads = EconomicForecastHeads(config)
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
        if self.config.critic_architecture in ("economic", "forecast"):
            states = self.trunk(
                inputs, opponent_unit_categorical, opponent_unit_continuous, opponent_unit_active
            )
            return self._economic_belief(states)
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
        states, memory, memory_valid = self.trunk.forward_with_memory(
            inputs, opponent_units, opponent_unit_active
        )
        valid = torch.cat(
            (
                inputs.unit_active,
                torch.ones(batch, self.trunk.market_count, dtype=torch.bool, device=states.device),
            ),
            dim=1,
        )
        pooled = (
            checkpoint(self._pool_belief, states, memory, valid, memory_valid, use_reentrant=False)
            if self.trunk.rematerialize and torch.is_grad_enabled()
            else self._pool_belief(states, memory, valid, memory_valid)
        )
        return StructuredCriticBelief(pooled)

    def _economic_belief(self, states: Tensor) -> StructuredCriticBelief:
        valid = torch.ones_like(states[..., 0], dtype=torch.bool)
        pooled = (
            checkpoint(self._pool_belief, states, states, valid, None, use_reentrant=False)
            if self.trunk.rematerialize and torch.is_grad_enabled()
            else self._pool_belief(states, states, valid, None)
        )
        return StructuredCriticBelief(pooled)

    def forward_with_forecasts(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
        *,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        market_active: Tensor,
        market_quantity_active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Share one state encoding; sampled actions enter only forecast heads.

        GAE and the value objective continue to use V(s). No action-conditioned
        forecast state is fed into that baseline, including during training.
        """
        if self.forecast_heads is None:
            raise ValueError("forecast forward requires critic_architecture='forecast'")
        states = self.trunk(
            inputs, opponent_unit_categorical, opponent_unit_continuous, opponent_unit_active
        )
        value = self.decode_belief(self._economic_belief(states))
        forecasts = self.forecast_heads(
            states,
            unit_actions,
            market_kinds,
            market_quantities,
            inputs.unit_categorical,
            inputs.unit_active,
            market_active,
            market_quantity_active,
        )
        return value, forecasts

    def _pool_belief(
        self, states: Tensor, memory: Tensor, valid: Tensor, memory_valid: Tensor | None
    ) -> Tensor:
        query = self.value_query.unsqueeze(0).expand(states.shape[0], -1, -1)
        context = self.pool_norm(states)
        if self.source_pool_norm is not None:
            context = torch.cat((context, self.source_pool_norm(memory)), dim=1)
            valid = torch.cat((valid, memory_valid), dim=1)
        pooled = self.pool_attention(query, context, context_valid=valid)
        if self.value_ffn is not None:
            assert self.value_ffn_norm is not None and self.value_ffn_gate is not None
            pooled = self.value_ffn_gate(pooled, self.value_ffn(self.value_ffn_norm(pooled)))
        return self.value_norm(pooled)

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
