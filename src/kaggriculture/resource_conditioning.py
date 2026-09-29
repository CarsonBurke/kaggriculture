"""Small market-head residuals conditioned on the actual action-prefix ledger.

The encoder is evaluated once per turn. These heads are evaluated once per
market order, after unit effects and all preceding orders. Collectors must save
the same features with the sampled actions so PPO replays their distribution.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from kaggriculture.actions import N_MARKET_KINDS, MarketLedger
from kaggriculture.constants import CROPS, MAX_UNITS, PRIVATE_ITEMS, PRODUCTS, SHED_CAPACITY

RESOURCE_FEATURE_NAMES = (
    "money_signed_log",
    *(f"shed_{item}" for item in PRIVATE_ITEMS),
    *(f"market_{item}_signed_log" for item in PRODUCTS),
    "hires_today",
    "extra_land",
    *(f"seeds_{crop}_signed_log" for crop in CROPS),
)
RESOURCE_FEATURES = len(RESOURCE_FEATURE_NAMES)


def market_resource_features(ledger: MarketLedger) -> np.ndarray:
    """Encode exact pre-order public market and own-private resource state.

    Inventory may be negative; use signed logs rather than clamping away that
    information. Scales are fixed, with no running statistics or saturation.
    The feature ordering and scales are also the native sampler's wire format.
    """

    def signed_log(value: float) -> float:
        return math.copysign(math.log1p(abs(value)), value) / 12.0

    return np.asarray(
        [
            signed_log(ledger.money),
            *(ledger.shed.get(item, 0) / SHED_CAPACITY for item in PRIVATE_ITEMS),
            *(signed_log(ledger.inventory[item]) for item in PRODUCTS),
            ledger.hires / MAX_UNITS,
            ledger.extra_land / 3.0,
            *(signed_log(ledger.seeds.get(crop, 0)) for crop in CROPS),
        ],
        dtype=np.float32,
    )


class MarketResourceConditioner(nn.Module):
    """Zero-initialized, bias-free adjustments to existing market preferences.

    Quantity adjustments enter the existing context BEFORE its selected-kind
    gate and quantity projection. Thus they remain kind-conditioned and preserve
    ALL-alias marginalization. FP32 arithmetic matches the host/native decoder.
    Both matrices can be cached on the host once per rollout; no device roundtrip
    or additional encoder execution is needed between orders.
    """

    def __init__(self, quantity_rank: int) -> None:
        super().__init__()
        if quantity_rank <= 0:
            raise ValueError("quantity_rank must be positive")
        self.kind = nn.Linear(RESOURCE_FEATURES, N_MARKET_KINDS, bias=False)
        self.quantity = nn.Linear(RESOURCE_FEATURES, quantity_rank, bias=False)
        nn.init.zeros_(self.kind.weight)
        nn.init.zeros_(self.quantity.weight)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.shape[-1] != RESOURCE_FEATURES:
            raise ValueError(f"market resource features must have width {RESOURCE_FEATURES}")
        with torch.autocast(features.device.type, enabled=False):
            features = features.float()
            # Autocast disabling alone does not disable TF32 GEMM under the
            # trainer's global "high" setting. Fuse a short FP32 reduction,
            # matching the host decoder's feature order instead.
            kind_weights = self.kind.weight.float()
            quantity_weights = self.quantity.weight.float()
            kinds = features[..., 0, None] * kind_weights[:, 0]
            quantities = features[..., 0, None] * quantity_weights[:, 0]
            for index in range(1, RESOURCE_FEATURES):
                kinds = kinds + features[..., index, None] * kind_weights[:, index]
                quantities = quantities + features[..., index, None] * quantity_weights[:, index]
            return kinds, quantities

    def condition(
        self, kind_logits: Tensor, quantity_context: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Replay a batch of recorded pre-order ledgers without losing gradients."""
        if kind_logits.shape[:-1] != features.shape[:-1]:
            raise ValueError("kind logits and resource features must share leading dimensions")
        if quantity_context.shape[:-1] != features.shape[:-1]:
            raise ValueError("quantity context and resource features must share leading dimensions")
        kind_delta, quantity_delta = self(features)
        return kind_logits.float() + kind_delta, quantity_context.float() + quantity_delta


def replay_market_resources(
    observation: dict,
    unit_actions: np.ndarray,
    market_kinds: np.ndarray,
    market_quantities: np.ndarray,
) -> np.ndarray:
    """Teacher-force the exact physical action prefix for BC encoding."""
    from kaggriculture.actions import (
        MarketKind,
        UnitAction,
        _apply_ledger_order,
        apply_unit_shed_effect,
        apply_unit_tile_effect,
        copy_tile_grid,
    )
    from kaggriculture.constants import QUANTITY_BINS

    player = int(observation.get("player", 0) or 0)
    farm = observation["farms"][player]
    shed = dict((observation.get("private") or {}).get("shed") or {})
    tiles = copy_tile_grid(farm.get("tiles") or [])
    seeds = dict((observation.get("private") or {}).get("seeds") or {})
    live = 1 + len(farm.get("hands") or [])
    for index, action in enumerate(unit_actions[:live]):
        if UnitAction.PLANT_WHEAT <= action <= UnitAction.PLANT_MELON:
            crop = CROPS[int(action) - int(UnitAction.PLANT_WHEAT)]
            seeds[crop] = seeds.get(crop, 0) - 1
        apply_unit_shed_effect(observation, index, int(action), shed, tiles)
        apply_unit_tile_effect(observation, index, int(action), tiles)
    ledger = MarketLedger.from_observation(observation, shed=shed, seeds=seeds)
    rows = []
    active = True
    for kind, quantity in zip(market_kinds, market_quantities, strict=True):
        rows.append(market_resource_features(ledger))
        if int(kind) == int(MarketKind.STOP):
            active = False
        if active:
            _apply_ledger_order(
                observation, MarketKind(int(kind)), QUANTITY_BINS[int(quantity)], ledger
            )
    return np.stack(rows)
