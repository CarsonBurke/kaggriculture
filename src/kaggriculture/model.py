"""Compact convolutional actor and centralized distributional critic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    MarketKind,
    UnitAction,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, QUANTITY_BINS
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    residual_blocks: int = 3
    hidden: int = 192
    query_features: int = 24
    quantity_rank: int = 32
    value_atoms: int = 101
    value_min: float = -2.0
    value_max: float = 2.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ActorOutput(NamedTuple):
    unit_logits: Tensor
    market_kind_logits: Tensor
    market_quantity_logits: Tensor


def _group_count(width: int) -> int:
    if width <= 0:
        raise ValueError("model width must be positive")
    groups = min(8, width)
    while width % groups:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = _group_count(width)
        self.layers = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layers(inputs)


class SpatialEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Conv2d(BOARD_CHANNELS, config.width, 3, padding=1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(config.width) for _ in range(config.residual_blocks))
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(config.width), config.width), nn.SiLU()
        )

    def forward(self, board: Tensor) -> Tensor:
        return self.output(self.blocks(self.input(board)))


def _context_network(config: ModelConfig, global_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(global_features + 2 * config.width, config.hidden),
        nn.LayerNorm(config.hidden),
        nn.SiLU(),
        nn.Linear(config.hidden, config.hidden),
        nn.SiLU(),
    )


def _pooled_context(spatial: Tensor, global_features: Tensor, network: nn.Module) -> Tensor:
    mean = spatial.mean(dim=(-2, -1))
    maximum = spatial.amax(dim=(-2, -1))
    return network(torch.cat((global_features, mean, maximum), dim=-1))


class FarmActor(nn.Module):
    """Decentralized policy: public farms plus only the acting player's private state."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.spatial = SpatialEncoder(config)
        self.context = _context_network(config, GLOBAL_FEATURES)
        self.unit_head = nn.Sequential(
            nn.Linear(config.width + config.hidden + UNIT_FEATURES, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, N_UNIT_ACTIONS),
        )
        self.market_queries = nn.Embedding(MAX_MARKET_ORDERS, config.query_features)
        market_input = config.hidden + config.query_features
        self.market_trunk = nn.Sequential(
            nn.Linear(market_input, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU(),
        )
        self.market_kind = nn.Linear(config.hidden, N_MARKET_KINDS)
        # A dense hidden -> kind x exact-quantity head would become a material
        # fraction of this small policy at 100 quantities. Factor it through a
        # learned rank while retaining a fully expressive kind/quantity bias.
        self.market_quantity_context = nn.Linear(config.hidden, config.quantity_rank, bias=False)
        self.market_quantity_kind_gate = nn.Embedding(N_MARKET_KINDS, config.quantity_rank)
        self.market_quantity_value = nn.Embedding(N_QUANTITIES, config.quantity_rank)
        self.market_quantity_bias = nn.Parameter(torch.zeros(N_MARKET_KINDS, N_QUANTITIES))
        self._initialize_policy_heads()

    def _initialize_policy_heads(self) -> None:
        heads = (self.unit_head[-1], self.market_kind, self.market_quantity_context)
        for head in heads:
            nn.init.normal_(head.weight, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        nn.init.zeros_(self.market_quantity_kind_gate.weight)
        nn.init.normal_(self.market_quantity_value.weight, std=0.01)
        nn.init.zeros_(self.market_quantity_bias)

        # Legal masks already remove actions that cannot have an effect. Among
        # the remaining actions, favor completing an economic cycle over random
        # movement or destroying an investment. This is only an initialization:
        # every action retains non-zero probability and all weights are learned.
        with torch.no_grad():
            unit_bias = self.unit_head[-1].bias
            unit_bias[UnitAction.PASS] = -1.25
            unit_bias[UnitAction.DROP] = 2.0
            pickup_biases = {
                UnitAction.PICKUP_WHEAT_1: 1.0,
                UnitAction.PICKUP_WHEAT_2: 0.75,
                UnitAction.PICKUP_WHEAT_4: 0.25,
                UnitAction.PICKUP_WHEAT_8: -0.5,
                UnitAction.PICKUP_WHEAT_16: -1.25,
                UnitAction.PICKUP_FERTILIZER_1: 0.75,
                UnitAction.PICKUP_FERTILIZER_2: 0.0,
                UnitAction.PICKUP_FERTILIZER_4: -1.0,
                UnitAction.PICKUP_FERTILIZER_8: -2.0,
            }
            for animal in ("GOOSE", "COW", "SHEEP"):
                for quantity, bias in zip((1, 2, 3, 4), (0.75, -0.25, -1.0, -1.5), strict=True):
                    pickup_biases[UnitAction[f"PICKUP_{animal}_{quantity}"]] = bias
            for action, bias in pickup_biases.items():
                unit_bias[action] = bias
            unit_bias[UnitAction.PLACE_GOOSE : UnitAction.PLACE_SHEEP + 1] = 2.0
            unit_bias[UnitAction.PLANT_WHEAT : UnitAction.PLANT_MELON + 1] = 1.0
            unit_bias[UnitAction.WATER] = 2.0
            unit_bias[UnitAction.HARVEST] = 2.5
            unit_bias[UnitAction.FERTILIZE] = 1.0
            unit_bias[UnitAction.DIG] = -0.5
            unit_bias[UnitAction.BUILD_COOP : UnitAction.BUILD_PASTURE + 1] = -2.5
            unit_bias[UnitAction.FEED] = 2.0
            unit_bias[UnitAction.COLLECT_FERTILIZER] = 2.0
            unit_bias[UnitAction.CARE] = 1.0

            # A uniform non-STOP order is catastrophically expensive over 719
            # turns. Keep a roughly 95% opening STOP probability while putting
            # almost all initial exploration into cheap hires and seeds. Selling
            # is deliberately competitive with STOP whenever inventory exists,
            # because only banked money has terminal value.
            kind_bias = self.market_kind.bias
            kind_bias[MarketKind.STOP] = 4.5
            kind_bias[MarketKind.HIRE] = 1.0
            kind_bias[MarketKind.BUY_LAND] = -7.0
            kind_bias[MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_SEED_MELON + 1] = -1.0
            kind_bias[MarketKind.BUY_PRODUCT_WHEAT : MarketKind.BUY_PRODUCT_FERTILIZER + 1] = -3.0
            kind_bias[MarketKind.BUY_ANIMAL_GOOSE : MarketKind.BUY_ANIMAL_SHEEP + 1] = -4.0
            kind_bias[MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1] = 4.0

            quantities = torch.as_tensor(
                QUANTITY_BINS,
                device=self.market_quantity_bias.device,
                dtype=self.market_quantity_bias.dtype,
            )
            log_quantity = quantities.log()
            quantity_bias = self.market_quantity_bias
            quantity_bias[MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_SEED_MELON + 1].copy_(
                -2.0 * log_quantity
            )
            quantity_bias[MarketKind.BUY_PRODUCT_WHEAT : MarketKind.BUY_ANIMAL_SHEEP + 1].copy_(
                -2.5 * log_quantity
            )
            quantity_bias[MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1].copy_(
                0.5 * log_quantity
            )

    def _quantity_logits(self, market_hidden: Tensor) -> Tensor:
        quantity_context = self.market_quantity_context(market_hidden)
        # Multiplicative gating is the state x kind interaction. An additive
        # kind embedding would collapse after the dot product into a static
        # kind/quantity bias and could not size buys and sells differently as
        # inventory, cash, or price changes.
        quantity_features = quantity_context[:, :, None, :] * (
            1.0 + self.market_quantity_kind_gate.weight[None, None, :, :]
        )
        return (
            torch.einsum("bskr,qr->bskq", quantity_features, self.market_quantity_value.weight)
            + self.market_quantity_bias[None, None, :, :]
        )

    def forward(
        self,
        board: Tensor,
        global_features: Tensor,
        units: Tensor,
        unit_positions: Tensor,
    ) -> ActorOutput:
        spatial = self.spatial(board)
        context = _pooled_context(spatial, global_features, self.context)
        batch, unit_count = unit_positions.shape[:2]
        x = unit_positions[..., 0].clamp(0, spatial.size(-1) - 1)
        y = unit_positions[..., 1].clamp(0, spatial.size(-2) - 1)
        batch_index = torch.arange(batch, device=spatial.device)[:, None].expand(batch, unit_count)
        local = spatial[batch_index, :, y, x]
        expanded_context = context[:, None, :].expand(batch, unit_count, -1)
        unit_logits = self.unit_head(torch.cat((local, units, expanded_context), dim=-1))

        queries = self.market_queries.weight[None, :, :].expand(batch, -1, -1)
        market_context = context[:, None, :].expand(batch, MAX_MARKET_ORDERS, -1)
        market_hidden = self.market_trunk(torch.cat((market_context, queries), dim=-1))
        return ActorOutput(
            unit_logits=unit_logits,
            market_kind_logits=self.market_kind(market_hidden),
            market_quantity_logits=self._quantity_logits(market_hidden),
        )


class DistributionalCritic(nn.Module):
    """Separate centralized critic with access to both players' private inventories."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.spatial = SpatialEncoder(config)
        self.context = _context_network(config, CRITIC_FEATURES)
        self.value_head = nn.Linear(config.hidden, config.value_atoms)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.register_buffer(
            "support",
            torch.linspace(config.value_min, config.value_max, config.value_atoms),
            persistent=True,
        )

    def forward(self, board: Tensor, critic_features: Tensor) -> Tensor:
        spatial = self.spatial(board)
        context = _pooled_context(spatial, critic_features, self.context)
        return self.value_head(context)

    def value(self, logits: Tensor) -> Tensor:
        return (logits.float().softmax(dim=-1) * self.support.float()).sum(dim=-1)


def two_hot_value_targets(targets: Tensor, support: Tensor) -> Tensor:
    """Project scalar returns onto adjacent categorical critic atoms."""
    support_float = support.float()
    support_min = support_float[0]
    support_max = support_float[-1]
    clipped = targets.float().maximum(support_min).minimum(support_max)
    scale = (support.numel() - 1) / (support_max - support_min)
    positions = (clipped - support_min) * scale
    lower = positions.floor().long().clamp(0, support.numel() - 1)
    upper = positions.ceil().long().clamp(0, support.numel() - 1)
    upper_weight = positions - lower.float()
    lower_weight = 1.0 - upper_weight
    projected = torch.zeros(
        (*targets.shape, support.numel()), device=targets.device, dtype=torch.float32
    )
    projected.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    projected.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return projected


def distributional_value_loss(logits: Tensor, targets: Tensor, support: Tensor) -> Tensor:
    projected = two_hot_value_targets(targets, support)
    return -(projected * logits.float().log_softmax(dim=-1)).sum(dim=-1)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
