"""Reference implementation of the opt-in interface-3 market set.

The 21 decisions have stable kind identities. A value of zero omits the kind;
positive values are executed quantities (or HIRE count). The sampler's extra
ALL logit is marginalized into the largest legal positive value, so trajectories
store only effective values. This module does not change interface-1/2 actions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kaggriculture.actions import (
    N_QUANTITIES,
    QUANTIFIED_MARKET_KINDS,
    MarketKind,
    MarketLedger,
    _apply_ledger_order,
    _ledger_quantity_mask,
    market_order,
)
from kaggriculture.constants import (
    LAND_PRICES,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    fibonacci_hire_cost,
    market_price,
)

SELL_SET_KINDS = tuple(kind for kind in MarketKind if kind.name.startswith("SELL_"))
MARKET_SET_KINDS = (
    *SELL_SET_KINDS,
    MarketKind.HIRE,
    MarketKind.BUY_LAND,
    *(kind for kind in MarketKind if kind.name.startswith("BUY_SEED_")),
    *(kind for kind in MarketKind if kind.name.startswith("BUY_ANIMAL_")),
    *(kind for kind in MarketKind if kind.name.startswith("BUY_PRODUCT_")),
)
N_MARKET_SET_KINDS = len(MARKET_SET_KINDS)
MARKET_SET_MAX_VALUE = N_QUANTITIES
MARKET_SET_ALL_INDEX = N_QUANTITIES + 1
MARKET_SET_RAW_CHOICES = N_QUANTITIES + 2


@dataclass(frozen=True)
class MarketSetOrder:
    """The Stage-0b-selected convention, specified explicitly by every caller."""

    sell_order: str = "fixed"
    hire_last: bool = False

    def __post_init__(self) -> None:
        if self.sell_order not in {"fixed", "impact"}:
            raise ValueError(f"unknown sell order {self.sell_order!r}")

    @property
    def decision_kinds(self) -> tuple[MarketKind, ...]:
        if not self.hire_last:
            return MARKET_SET_KINDS
        return (*SELL_SET_KINDS, *MARKET_SET_KINDS[len(SELL_SET_KINDS) + 1 :], MarketKind.HIRE)


@dataclass(frozen=True)
class MarketSetFactors:
    """One market set with exact sequential support and effective BC labels."""

    values: np.ndarray  # [21], zero=none, positive=effective engine quantity
    masks: np.ndarray  # [21, 101], values 0..100
    active: np.ndarray  # [21], false when only zero is legal


def _hire_maximum(observation: dict[str, Any], ledger: MarketLedger, slots: int) -> int:
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if player >= len(farms):
        return 0
    farm = farms[player]
    current_hands = len(farm.get("hands") or [])
    new_hires = max(0, ledger.hires - int(farm.get("hires_today", 0) or 0))
    remaining_hands = max(0, MAX_UNITS - 1 - current_hands - new_hires)
    money = ledger.money
    count = 0
    while count < min(remaining_hands, MAX_MARKET_ORDERS - slots):
        cost = fibonacci_hire_cost(ledger.hires + count)
        if money < cost:
            break
        money -= cost
        count += 1
    return count


def market_set_value_mask(
    observation: dict[str, Any],
    kind: MarketKind,
    ledger: MarketLedger,
    slots: int,
) -> np.ndarray:
    """Exact own-ledger support for a kind after earlier set decisions.

    Slot zero is legal even when no order fits. Positive choices are prefix
    legal. The opponent can change market quotes at execution, as it can in
    interfaces 1/2; this mask uses the same own-ledger convention.
    """
    if kind == MarketKind.STOP:
        raise ValueError("STOP has no set decision")
    if not 0 <= slots <= MAX_MARKET_ORDERS:
        raise ValueError(f"invalid market slot usage {slots}")
    mask = np.zeros(MARKET_SET_MAX_VALUE + 1, dtype=np.bool_)
    mask[0] = True
    if slots == MAX_MARKET_ORDERS:
        return mask
    if kind == MarketKind.HIRE:
        mask[1 : _hire_maximum(observation, ledger, slots) + 1] = True
    elif kind == MarketKind.BUY_LAND:
        mask[1] = (
            ledger.extra_land < len(LAND_PRICES) and ledger.money >= LAND_PRICES[ledger.extra_land]
        )
    elif kind in QUANTIFIED_MARKET_KINDS:
        mask[1:] = _ledger_quantity_mask(observation, kind, ledger)
    else:
        raise ValueError(f"unknown market kind {kind!r}")
    return mask


def market_set_raw_mask(effective_mask: np.ndarray) -> np.ndarray:
    """Expose the ALL alias only when a positive effective value is legal."""
    if effective_mask.shape != (MARKET_SET_MAX_VALUE + 1,):
        raise ValueError("effective market set mask must have 101 choices")
    raw = np.zeros(MARKET_SET_RAW_CHOICES, dtype=np.bool_)
    raw[:-1] = effective_mask
    raw[-1] = bool(effective_mask[1:].any())
    return raw


def marginalize_market_set_all(raw_logits: np.ndarray, effective_mask: np.ndarray) -> np.ndarray:
    """Return logits over engine-equivalent values 0..100."""
    if raw_logits.shape[-1] != MARKET_SET_RAW_CHOICES:
        raise ValueError("raw market set logits must have 102 choices")
    if effective_mask.shape != (MARKET_SET_MAX_VALUE + 1,):
        raise ValueError("effective market set mask must have 101 choices")
    logits = np.array(raw_logits[..., :-1], copy=True)
    legal = np.flatnonzero(effective_mask[1:])
    if legal.size:
        maximum = int(legal[-1]) + 1
        logits[..., maximum] = np.logaddexp(logits[..., maximum], raw_logits[..., -1])
    return logits


def _sell_kinds(
    observation: dict[str, Any], values: dict[MarketKind, int], order: MarketSetOrder
) -> list[MarketKind]:
    sells = [kind for kind in SELL_SET_KINDS if values.get(kind, 0) > 0]
    if order.sell_order == "impact":
        market = observation.get("market") or {}
        inventory = market.get("inventory") or {}
        params = market.get("params")

        def impact(kind: MarketKind) -> int:
            item = kind.name.removeprefix("SELL_")
            stock = int(inventory.get(item, 0))
            quantity = values[kind]
            return quantity * max(
                0,
                market_price(item, stock, params) - market_price(item, stock + quantity, params),
            )

        sells.sort(key=lambda kind: (-impact(kind), int(kind)))
    return sells


def _compile_orders(
    observation: dict[str, Any], values: dict[MarketKind, int], order: MarketSetOrder
) -> list[list[Any]]:
    sells = _sell_kinds(observation, values, order)
    sequence = [
        *sells,
        *(kind for kind in order.decision_kinds if kind not in SELL_SET_KINDS),
    ]
    orders: list[list[Any]] = []
    for kind in sequence:
        value = values.get(kind, 0)
        if value == 0:
            continue
        if kind == MarketKind.HIRE:
            orders.extend([["HIRE"] for _ in range(value)])
        elif kind == MarketKind.BUY_LAND:
            orders.append(["BUY_LAND"])
        else:
            # Legacy helper indexes 1..100 by q-1.
            compiled = market_order(int(kind), value - 1)
            if compiled is None:
                raise ValueError(f"{kind.name} unexpectedly compiled to STOP")
            orders.append(compiled)
    if len(orders) > MAX_MARKET_ORDERS:
        raise ValueError("market set exceeds the ten-slot budget")
    return orders


def compile_market_set(
    observation: dict[str, Any],
    choices: Sequence[int],
    *,
    order: MarketSetOrder,
    post_unit_shed: dict[str, int] | None = None,
) -> tuple[list[list[Any]], MarketSetFactors]:
    """Validate a sampled set, update the own ledger, and compile engine slots."""
    kinds = order.decision_kinds
    if len(choices) != len(kinds):
        raise ValueError(f"expected {len(kinds)} market values, got {len(choices)}")
    ledger = MarketLedger.from_observation(
        observation, shed=None if post_unit_shed is None else dict(post_unit_shed)
    )
    values = np.zeros(len(kinds), dtype=np.uint8)
    masks = np.zeros((len(kinds), MARKET_SET_MAX_VALUE + 1), dtype=np.bool_)
    active = np.zeros(len(kinds), dtype=np.bool_)
    slots = 0
    selected: dict[MarketKind, int] = {}
    for index, (kind, choice) in enumerate(zip(kinds, choices, strict=True)):
        mask = market_set_value_mask(observation, kind, ledger, slots)
        value = int(choice)
        if not 0 <= value <= MARKET_SET_MAX_VALUE or not mask[value]:
            raise ValueError(f"{kind.name} value {value} is not legal after {slots} slots")
        values[index] = value
        masks[index] = mask
        active[index] = bool(mask[1:].any())
        selected[kind] = value
        if value:
            if kind == MarketKind.HIRE:
                for _ in range(value):
                    _apply_ledger_order(observation, kind, 1, ledger)
            else:
                _apply_ledger_order(observation, kind, value, ledger)
            slots += value if kind == MarketKind.HIRE else 1
    return _compile_orders(observation, selected, order), MarketSetFactors(values, masks, active)


def project_effective_market_set(
    observation: dict[str, Any],
    effective_orders: Sequence[Sequence[Any]],
    *,
    order: MarketSetOrder,
    post_unit_shed: dict[str, int] | None = None,
) -> tuple[list[list[Any]], MarketSetFactors]:
    """Create BC labels from *executed fills*, never requested quantities.

    The caller must obtain effective orders from an engine trace. Passing raw
    teacher requests can label no-op orders or unfilled HIREs as executed. The
    round trip catches values outside the own-ledger support, but cannot infer
    a partial fill from the request alone.
    """
    from kaggriculture.demonstrations import _parse_market_order

    if len(effective_orders) > MAX_MARKET_ORDERS:
        raise ValueError("effective market orders exceed the ten-slot budget")
    totals: dict[MarketKind, int] = {}
    for raw in effective_orders:
        parsed = _parse_market_order(raw)
        if parsed is None:
            continue
        kind, quantity = parsed
        totals[kind] = totals.get(kind, 0) + (
            1 if kind in (MarketKind.HIRE, MarketKind.BUY_LAND) else quantity
        )
    if totals.get(MarketKind.BUY_LAND, 0) > 1:
        raise ValueError("market set cannot represent repeated BUY_LAND")
    choices = [totals.get(kind, 0) for kind in order.decision_kinds]
    return compile_market_set(observation, choices, order=order, post_unit_shed=post_unit_shed)
