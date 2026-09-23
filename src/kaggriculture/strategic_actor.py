"""Public global workspace and a shared categorical plan for a complete turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.entity import EntityActor, EntityConfig, EntityRound
from kaggriculture.model import ActorOutput, AxialRotaryEmbedding, Linear, RMSNorm
from kaggriculture.structured import (
    Block,
    EconomyEmbedder,
    StructuredInputs,
    TileEmbedder,
    UnitEmbedder,
)
from kaggriculture.tokens import TILE_COUNT


@dataclass(frozen=True)
class StrategicConfig(EntityConfig):
    workspace_states: int = 8
    plan_count: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("workspace_states", "plan_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            (
                self.bixt_latents,
                self.memory_writeback,
                self.unit_tile_bias,
                self.tile_cross_rope,
                self.unit_local_readout,
                self.shared_memory_kv,
            )
        ):
            raise ValueError("strategic workspace does not use entity-specific experimental paths")


class PlanChoice(NamedTuple):
    """Explicit randomness for collection, or recorded indices for PPO replay.

    All fields have shape [B]; negative indices request sampling. Random draws
    are generated outside compiled/captured forwards, including frozen ensembles.
    The old likelihood is replay metadata, never a neural feature.
    """

    indices: Tensor
    uniforms: Tensor
    temperatures: Tensor
    deterministic: Tensor
    old_logprobs: Tensor
    active: Tensor


class StrategicOutput(NamedTuple):
    unit_logits: Tensor
    market_kind_logits: Tensor
    market_quantity_context: Tensor
    plan: Tensor  # [B,3]: selected index, actual log probability, entropy


class PublicEncoder(nn.Module):
    """Spatial encoding once, with explicit units and public economic context."""

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.config = config
        self.tiles = TileEmbedder(config)
        self.units = UnitEmbedder(config, local_init=config.unit_local_init, local_context=False)
        self.economy = EconomyEmbedder(config, private_columns=False)
        self.farm_local = nn.ModuleList(Block(config) for _ in range(config.farm_blocks))
        self.rope = AxialRotaryEmbedding(config.model_dim // config.attention_heads)

    def forward(self, inputs: StructuredInputs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = inputs.tile_categorical.shape[0]
        farms = self.tiles(inputs.tile_categorical, inputs.tile_continuous).reshape(
            2 * batch, TILE_COUNT, self.config.model_dim
        )
        rotation = (
            self.rope.cosine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
            self.rope.sine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
        )
        for block in self.farm_local:
            farms = block(farms, query_rotation=rotation, key_rotation=rotation)
        tiles = farms.reshape(batch, 2 * TILE_COUNT, self.config.model_dim)
        local = (
            self.units.local_tiles(
                tiles[:, :TILE_COUNT], inputs.unit_tile_gather, inputs.unit_tile_gather_valid
            )
            if self.config.unit_local_init
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
        memory = torch.cat((tiles, economy, units), dim=1)
        valid = torch.cat(
            (torch.ones_like(memory[:, :-MAX_UNITS, 0], dtype=torch.bool), inputs.unit_active),
            dim=1,
        )
        return units, economy, memory, valid


class StrategicTrunk(nn.Module):
    def __init__(self, config: StrategicConfig) -> None:
        super().__init__()
        self.encoder = PublicEncoder(config)
        self.workspace = nn.Embedding(config.workspace_states, config.model_dim)
        self.memory_norm = RMSNorm(config.model_dim)
        self.core = nn.ModuleList(EntityRound(config) for _ in range(config.core_layers))
        self.decoder = Block(config)
        self.decoder_norm = RMSNorm(config.model_dim)
        self.market_queries = nn.Embedding(MAX_MARKET_ORDERS, config.model_dim)
        self.plans = nn.Embedding(config.plan_count, config.model_dim)
        self.plan_head = nn.Sequential(
            RMSNorm(config.model_dim), Linear(config.model_dim, config.plan_count)
        )

    def encode(self, inputs: StructuredInputs) -> tuple[Tensor, Tensor, Tensor]:
        units, economy, memory, memory_valid = self.encoder(inputs)
        summary = economy.mean(dim=1)
        workspace = summary[:, None] + self.workspace.weight
        valid = torch.ones_like(workspace[..., 0], dtype=torch.bool)
        memory = self.memory_norm(memory)
        for block in self.core:
            workspace = block(
                workspace,
                memory,
                None,
                valid,
                memory_valid,
                summary if self.encoder.config.global_modulation else None,
            )
        decisions = torch.cat((units, summary[:, None] + self.market_queries.weight), dim=1)
        return decisions, workspace, self.plan_head(workspace.mean(dim=1)).float()

    def decode(self, decisions: Tensor, workspace: Tensor, plans: Tensor) -> Tensor:
        return self.decoder(
            decisions + self.plans(plans)[:, None], workspace, context_norm=self.decoder_norm
        )


class StrategicActor(EntityActor):
    """One plan is sampled per turn and conditions every physical action head."""

    def __init__(self, config: StrategicConfig | None = None) -> None:
        config = config or StrategicConfig()
        nn.Module.__init__(self)
        self.config = config
        self.trunk = StrategicTrunk(config)
        self._initialize_heads(config)

    def _heads(self, states: Tensor) -> ActorOutput:
        units, markets = states.split((MAX_UNITS, MAX_MARKET_ORDERS), dim=1)
        markets = self.market_norm(markets)
        return ActorOutput(
            self.unit_head(units).contiguous(),
            self.market_kind(markets).contiguous(),
            self.market_quantity_context(markets).contiguous(),
        )

    def forward(
        self, inputs: StructuredInputs, choice: PlanChoice | None = None
    ) -> StrategicOutput:
        decisions, workspace, logits = self.trunk.encode(inputs)
        if choice is None:
            indices = logits.argmax(-1)
        else:
            logits = logits / choice.temperatures[:, None]
            probabilities = logits.softmax(-1)
            sampled = (probabilities.cumsum(-1) <= choice.uniforms[:, None]).sum(-1)
            sampled = sampled.clamp_max(self.config.plan_count - 1)
            sampled = torch.where(choice.deterministic, logits.argmax(-1), sampled)
            indices = torch.where(choice.indices >= 0, choice.indices, sampled).long()
        logprobs = logits.log_softmax(-1)
        selected = logprobs.gather(-1, indices[:, None]).squeeze(-1)
        entropy = -(logprobs.exp() * logprobs).sum(-1)
        heads = self._heads(self.trunk.decode(decisions, workspace, indices))
        return StrategicOutput(*heads, torch.stack((indices.float(), selected, entropy), dim=-1))

    def all_plans(self, inputs: StructuredInputs) -> tuple[ActorOutput, Tensor]:
        """Vectorized decoder branches for exact BC marginal likelihood."""
        decisions, workspace, logits = self.trunk.encode(inputs)
        batch, plans = logits.shape
        indices = torch.arange(plans, device=logits.device).expand(batch, -1).reshape(-1)
        decisions = decisions[:, None].expand(-1, plans, -1, -1).flatten(0, 1)
        workspace = workspace[:, None].expand(-1, plans, -1, -1).flatten(0, 1)
        return self._heads(self.trunk.decode(decisions, workspace, indices)), logits.log_softmax(-1)
