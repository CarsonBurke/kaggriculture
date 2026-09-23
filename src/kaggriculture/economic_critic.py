"""Dedicated valuation representation, independent of actor decision slots.

This is a value-representation ablation, not an economic forecasting objective.
The existing return and NextLat losses train it; no unobserved economic targets
are implied by its role-based valuation states.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from kaggriculture.constants import ANIMALS, CROPS, PRODUCTS
from kaggriculture.entity import EntityConfig, EntityMemory, EntityRound
from kaggriculture.model import AxialRotaryEmbedding, RMSNorm
from kaggriculture.structured import (
    Block,
    EconomyEmbedder,
    StructuredInputs,
    TileEmbedder,
    UnitEmbedder,
)
from kaggriculture.tokens import TILE_COUNT

ECONOMY_STATES = len(PRODUCTS) + len(ANIMALS) + len(CROPS) + 3
VALUATION_STATES = 4 + ECONOMY_STATES


def _workforce_mean(units: Tensor, active: Tensor) -> Tensor:
    """A vacant workforce has a zero summary and no padded-unit gradient."""
    count = active.sum(dim=1, keepdim=True).clamp_min(1).to(units.dtype)
    return units.sum(dim=1) / count


class EconomicCriticTrunk(nn.Module):
    """Farm/workforce/economy states repeatedly read complete centralized memory.

    Both players' tiles and units use the same independent critic encoders and
    explicit ownership embeddings. Own and opponent labor follow identical
    encoding paths: value does not depend on actor execution-slot queries or
    market-order capacity. All source tiles remain visible, including locked
    squares, while inactive units are excluded from summaries and attention.

    Entity-specific local initialization, local readout, spatial query bias/RoPE,
    source writeback, and source-read pooling flags continue to configure the
    actor/entity architecture; they do not alter this valuation representation.
    Width, farm blocks, reasoning depth, GQA, K/V sharing, modulation, and FFN
    configuration remain shared structural settings for both networks.
    """

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.config = config
        self.tiles = TileEmbedder(config)
        self.units = UnitEmbedder(config, local_init=False, local_context=False)
        self.economy = EconomyEmbedder(config, private_columns=True)
        self.rope = AxialRotaryEmbedding(config.model_dim // config.attention_heads)
        self.farm_local = nn.ModuleList(Block(config) for _ in range(config.farm_blocks))
        self.valuation_roles = nn.Embedding(VALUATION_STATES, config.model_dim)
        self.memory = EntityMemory(config) if config.shared_memory_kv else None
        self.memory_norm = None if config.shared_memory_kv else RMSNorm(config.model_dim)
        self.core = nn.ModuleList(EntityRound(config) for _ in range(config.core_layers))
        # Round-local projections over all 252 sources otherwise retain large
        # training activations. Stateful fused MLP scaling must not be replayed.
        self.rematerialize = not config.fused_mlp

    def forward(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> Tensor:
        batch = inputs.tile_categorical.shape[0]
        width = self.config.model_dim
        farms = self.tiles(inputs.tile_categorical, inputs.tile_continuous).reshape(
            2 * batch, TILE_COUNT, width
        )
        rotation = (
            self.rope.cosine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
            self.rope.sine.view(1, 1, TILE_COUNT, -1).expand(2 * batch, -1, -1, -1),
        )
        for block in self.farm_local:
            farms = block(farms, query_rotation=rotation, key_rotation=rotation)
        farms = farms.reshape(batch, 2, TILE_COUNT, width)
        own_units = self.units(
            inputs.unit_categorical,
            inputs.unit_continuous,
            inputs.unit_active,
            None,
            opponent=False,
        )
        opponent_units = self.units(
            opponent_unit_categorical,
            opponent_unit_continuous,
            opponent_unit_active,
            None,
            opponent=True,
        )
        economy = self.economy(
            inputs.products, inputs.animals, inputs.crops, inputs.farms, inputs.town
        )
        # Economy order ends in the two owned farm summaries and one town state.
        # Seed farm and workforce valuation with that player's current economy.
        farm_economy = economy[:, -3:-1]
        farm_states = farms.mean(dim=2) + farm_economy
        workforce_states = (
            torch.stack(
                (
                    _workforce_mean(own_units, inputs.unit_active),
                    _workforce_mean(opponent_units, opponent_unit_active),
                ),
                dim=1,
            )
            + farm_economy
        )
        states = torch.cat((farm_states, workforce_states, economy), dim=1)
        states = states + self.valuation_roles.weight
        memory = torch.cat((farms.flatten(1, 2), economy, own_units, opponent_units), dim=1)
        memory_valid = torch.cat(
            (
                torch.ones(
                    batch, 2 * TILE_COUNT + ECONOMY_STATES, dtype=torch.bool, device=memory.device
                ),
                inputs.unit_active,
                opponent_unit_active,
            ),
            dim=1,
        )
        state_valid = torch.ones_like(states[..., 0], dtype=torch.bool)
        if self.memory is not None:
            key, value = self.memory(memory)
            states = states.to(key.dtype)
        else:
            assert self.memory_norm is not None
            key, value = self.memory_norm(memory), None
        conditioning = economy.mean(dim=1) if self.config.global_modulation else None
        for block in self.core:
            args = states, key, value, state_valid, memory_valid, conditioning
            states = (
                checkpoint(block, *args, use_reentrant=False)
                if self.rematerialize and torch.is_grad_enabled()
                else block(*args)
            )
        return states
