"""Feed-forward, action-conditioned economic supervision for a state-value critic.

The forecast branch reads unpooled economic valuation states. Its sampled current
turn actions never enter the value readout. Future observations are labels only;
there is no latent target network, recurrent state, or future-action teacher input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
)
from kaggriculture.economic_critic import VALUATION_STATES
from kaggriculture.entity import EntityConfig
from kaggriculture.model import RMSNorm
from kaggriculture.structured import Attention, FeedForward
from kaggriculture.tokens import (
    TILE_CONTINUOUS_FIELDS,
    TILE_COUNT,
    TILE_KINDS,
    UNIT_CONTINUOUS_FIELDS,
)

FORECAST_HORIZONS = (1, 24, 96, 384)
FORECAST_CHANNELS = 8
# Matches EconomicCriticTrunk's [farm, workforce, product, animal, crop, farm, town].
_PRODUCT_START = 4
_ANIMAL_START = _PRODUCT_START + len(PRODUCTS)
_CROP_START = _ANIMAL_START + len(ANIMALS)
_FARM_START = _CROP_START + len(CROPS)
FORECAST_GROUPS = (
    (0, 2),
    (2, 4),
    (_PRODUCT_START, _ANIMAL_START),
    (_ANIMAL_START, _CROP_START),
    (_CROP_START, _FARM_START),
    (_FARM_START, _FARM_START + 2),
    (_FARM_START + 2, VALUATION_STATES),
)
ECONOMIC_FEATURE_MASK = np.zeros((VALUATION_STATES, FORECAST_CHANNELS), dtype=np.bool_)
for (_start, _stop), _channels in zip(FORECAST_GROUPS, (8, 3, 6, 4, 2, 3, 8), strict=True):
    ECONOMIC_FEATURE_MASK[_start:_stop, :_channels] = True


@dataclass(frozen=True)
class EconomicForecastTargets:
    """Compact CPU labels; flatten trajectory/time before device staging.

    Future indices index the flattened feature array. Invalid indices point at
    the source, so masked labels stay finite without out-of-bounds reads. Keeping
    one feature table avoids repeating 192 floats for every forecast horizon.
    """

    features: np.ndarray  # [trajectory, time, 24, 8], float16
    future_indices: np.ndarray  # [trajectory, time, horizon], int64
    valid: np.ndarray  # [trajectory, time, horizon], bool


def _economic_features(states: Mapping[str, np.ndarray], row: int) -> np.ndarray:
    """Fixed scales inherited from the observation encoder, never fitted to labels."""
    products = np.asarray(states["products"][row], dtype=np.float32)
    steps = products.shape[0]
    features = np.zeros((steps, VALUATION_STATES, FORECAST_CHANNELS), dtype=np.float32)
    tiles = np.asarray(states["tile_continuous"][row], dtype=np.float32).reshape(
        steps, 2, TILE_COUNT, -1
    )
    kinds = states["tile_categorical"][row, ..., 0].reshape(steps, 2, TILE_COUNT)
    features[:, :2, 0] = (kinds == TILE_KINDS.index("PLANT")).mean(axis=-1)
    features[:, :2, 1] = np.isin(
        kinds, (TILE_KINDS.index("COOP"), TILE_KINDS.index("PASTURE"))
    ).mean(axis=-1)
    # Board fractions preserve expansion/productive area instead of hiding it in
    # an occupied-tile denominator. Static geometry is deliberately not a target.
    for channel, field in enumerate(
        (
            "yield_fraction",
            "maturity_fraction",
            "harvest_ready",
            "watered_today",
            "fed_today",
            "decay_pressure",
        ),
        start=2,
    ):
        features[:, :2, channel] = tiles[..., TILE_CONTINUOUS_FIELDS.index(field)].mean(axis=-1)
    for farm, prefix in enumerate(("", "opponent_")):
        active = np.asarray(states[f"{prefix}unit_active"][row], dtype=np.bool_)
        continuous = np.asarray(states[f"{prefix}unit_continuous"][row], dtype=np.float32)
        features[:, 2 + farm, 0] = active.mean(axis=-1)
        for channel, field in enumerate(("holds_total", "shed_access"), start=1):
            features[:, 2 + farm, channel] = (
                continuous[..., UNIT_CONTINUOUS_FIELDS.index(field)] * active
            ).sum(axis=-1) / MAX_UNITS
    features[:, _PRODUCT_START:_ANIMAL_START, :2] = products[..., :2]
    features[:, _PRODUCT_START:_ANIMAL_START, 2:4] = products[..., 3:5]
    features[:, _PRODUCT_START:_ANIMAL_START, 4:6] = states["critic_products"][row]
    features[:, _ANIMAL_START:_CROP_START, :2] = states["animals"][row, ..., 1:3]
    features[:, _ANIMAL_START:_CROP_START, 2:4] = states["critic_animals"][row]
    features[:, _CROP_START:_FARM_START, 0] = states["crops"][row, ..., 1]
    features[:, _CROP_START:_FARM_START, 1] = states["critic_crops"][row, ..., 0]
    features[:, _FARM_START : _FARM_START + 2, :3] = states["farms"][row, ..., :3]
    # Shop supply is economic; the deterministic clock is an input, not a label.
    features[:, -1, :] = states["town"][row, ..., 6:14]
    return features


def build_economic_forecast_targets(
    states: Mapping[str, np.ndarray],
    valid: np.ndarray,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> EconomicForecastTargets:
    """Labels for s[t+h]-s[t], only if every intervening state is valid.

    Pass the learner's owned-valid mask for population updates. Trajectory and
    terminal boundaries are never crossed. The terminal successor is absent
    from RolloutBatch, so no terminal economic state is fabricated.
    """
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean [trajectory, time] array")
    if not horizons or any(
        isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons
    ):
        raise ValueError("forecast horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("forecast horizons must be distinct")
    trajectories, steps = valid.shape
    features = np.empty(
        (trajectories, steps, VALUATION_STATES, FORECAST_CHANNELS), dtype=np.float16
    )
    for row in range(trajectories):
        if valid[row].any():
            row_features = _economic_features(states, row)
            if not np.isfinite(row_features[valid[row]]).all():
                raise ValueError("economic features must be finite at valid states")
            row_features[~valid[row]] = 0
            features[row] = row_features
        else:
            features[row] = 0
    source = np.arange(trajectories * steps, dtype=np.int64).reshape(trajectories, steps)
    future = np.broadcast_to(source[..., None], (*valid.shape, len(horizons))).copy()
    mask = np.zeros_like(future, dtype=np.bool_)
    # Prefix counts detect interior holes, not merely valid endpoints.
    invalid_prefix = np.pad(np.cumsum(~valid, axis=1), ((0, 0), (1, 0)))
    for channel, horizon in enumerate(horizons):
        if horizon >= steps:
            continue
        windows_valid = invalid_prefix[:, horizon + 1 :] == invalid_prefix[:, : steps - horizon]
        mask[:, : steps - horizon, channel] = windows_valid
        future[:, : steps - horizon, channel] = np.where(
            windows_valid, source[:, horizon:], source[:, : steps - horizon]
        )
    return EconomicForecastTargets(features, future, mask)


def economic_forecast_loss(
    predictions: Tensor,
    targets: Tensor,
    valid: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Equal weight per horizon and economic family, excluding padded samples.

    Targets are deltas in fixed observable feature scales. A zero prediction is
    the persistence baseline and should be logged alongside forecast error.
    """
    mask = torch.as_tensor(ECONOMIC_FEATURE_MASK, device=predictions.device)
    errors = F.smooth_l1_loss(predictions.float(), targets.detach().float(), reduction="none")
    role_errors = (errors * mask).sum(dim=-1) / mask.sum(dim=-1)
    per_sample = torch.stack(
        [role_errors[..., start:stop].mean(dim=-1) for start, stop in FORECAST_GROUPS], dim=-1
    ).mean(dim=-1)
    weight = valid.float()
    if sample_weight is not None:
        weight = weight * sample_weight[:, None]
    horizon_count = weight.sum(dim=0)
    horizon_loss = (per_sample * weight).sum(dim=0) / horizon_count.clamp_min(1)
    present = horizon_count > 0
    return (horizon_loss * present).sum() / present.sum().clamp_min(1)


class _TypedForecastReadout(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(VALUATION_STATES, width, FORECAST_CHANNELS))
        self.bias = nn.Parameter(torch.zeros(VALUATION_STATES, FORECAST_CHANNELS))

    def forward(self, states: Tensor) -> Tensor:
        return torch.einsum("bhsd,sdc->bhsc", states, self.weight) + self.bias


class EconomicForecastHeads(nn.Module):
    """Typed multihorizon reads of economic states and this turn's action slots.

    Only this branch receives sampled actions. State-value pooling must use the
    input states directly, before this branch. Unknown opponent actions and all
    future policy actions are marginalized by regression to observed outcomes.
    """

    def __init__(self, config: EntityConfig, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> None:
        super().__init__()
        self.horizons = horizons
        width = config.model_dim
        self.state_norm = RMSNorm(width)
        self.horizon = nn.Embedding(len(horizons), width)
        self.unit_action = nn.Embedding(N_UNIT_ACTIONS, width)
        self.unit_slot = nn.Embedding(MAX_UNITS, width)
        self.unit_row = nn.Embedding(BOARD_SIZE, width)
        self.unit_column = nn.Embedding(BOARD_SIZE, width)
        self.market_kind = nn.Embedding(N_MARKET_KINDS, width)
        self.market_quantity = nn.Embedding(N_QUANTITIES, width)
        self.market_slot = nn.Embedding(MAX_MARKET_ORDERS, width)
        self.action_norm = RMSNorm(width)
        self.action_read = Attention(config)
        self.ffn_norm = RMSNorm(width)
        self.ffn = FeedForward(config)
        self.output_norm = RMSNorm(width)
        self.forecast_head = _TypedForecastReadout(width)

    def forward(
        self,
        states: Tensor,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        unit_categorical: Tensor,
        unit_active: Tensor,
        market_active: Tensor,
        market_quantity_active: Tensor,
    ) -> Tensor:
        units = (
            self.unit_action(unit_actions)
            + self.unit_slot.weight
            + self.unit_row(unit_categorical[..., 2])
            + self.unit_column(unit_categorical[..., 3])
        )
        markets = (
            self.market_kind(market_kinds)
            + self.market_slot.weight
            + self.market_quantity(market_quantities) * market_quantity_active[..., None]
        )
        actions = self.action_norm(torch.cat((units, markets), dim=1))
        action_valid = torch.cat((unit_active, market_active), dim=1).bool()
        queries = self.state_norm(states)[:, None] + self.horizon.weight[None, :, None]
        queries = queries.flatten(1, 2)
        queries = queries + self.action_read(queries, actions, context_valid=action_valid)
        queries = queries + self.ffn(self.ffn_norm(queries))
        queries = self.output_norm(queries).unflatten(1, (len(self.horizons), VALUATION_STATES))
        return self.forecast_head(queries)
