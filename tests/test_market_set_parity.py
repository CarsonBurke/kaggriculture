from __future__ import annotations

import numpy as np
import pytest
from kaggle_environments import make

from kaggriculture.actions import (
    MarketKind,
    MarketLedger,
    UnitAction,
    _apply_ledger_order,
    apply_unit_shed_effect,
    compile_action,
    unit_action_mask,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.demonstrations import _parse_market_order
from kaggriculture.market_set import MarketSetOrder, compile_market_set, market_set_value_mask
from kaggriculture.rust_env import load_native


@pytest.mark.parametrize("order", [MarketSetOrder(), MarketSetOrder("impact", True)])
def test_rust_market_set_masks_match_python_after_evolving_ledgers(order: MarketSetOrder) -> None:
    seeds = np.array([29, 30], dtype=np.uint64)
    native = load_native().BatchEnv(seeds)
    environments = [
        make("kaggriculture", configuration={"episodeSteps": 48, "seed": int(seed)})
        for seed in seeds
    ]
    for environment in environments:
        environment.reset(2)
    sell_opportunities = hire_opportunities = post_unit_changes = 0
    for step in range(36):
        units = np.zeros((len(seeds), 2, MAX_UNITS), dtype=np.uint8)
        values = np.zeros((len(seeds) * 2, 21), dtype=np.uint8)
        expected = []
        actions = []
        legacy_kinds = np.zeros((len(seeds), 2, MAX_MARKET_ORDERS), dtype=np.uint8)
        legacy_quantities = np.zeros_like(legacy_kinds)
        for game, environment in enumerate(environments):
            game_actions = []
            for seat in (0, 1):
                observation = environment.state[seat].observation
                row = game * 2 + seat
                shed = dict(observation["private"]["shed"])
                farmer_inventory = observation["private"]["inventories"][0]
                desired = (
                    UnitAction.PICKUP_WHEAT_1
                    if step % 3 == 1 and shed.get("WHEAT", 0)
                    else UnitAction.PLACE_WHEAT
                    if step % 3 == 2 and farmer_inventory.get("WHEAT", 0)
                    else UnitAction.PASS
                )
                if unit_action_mask(observation, 0)[desired]:
                    units[game, seat, 0] = desired
                    apply_unit_shed_effect(observation, 0, desired, shed)
                    post_unit_changes += desired != UnitAction.PASS
                ledger = MarketLedger.from_observation(observation, shed=dict(shed))
                slots = 0
                choices = []
                for kind in order.decision_kinds:
                    mask = market_set_value_mask(observation, kind, ledger, slots)
                    maximum = int(np.flatnonzero(mask)[-1])
                    value = 0
                    if maximum:
                        if kind.name.startswith("SELL_") and step % 2 == 1:
                            value = min(maximum, 20)
                            sell_opportunities += 1
                        elif kind == MarketKind.HIRE and step % 24 == 0:
                            value = min(maximum, 3)
                            hire_opportunities += 1
                        elif kind == MarketKind.BUY_LAND and step == 24:
                            value = 1
                        elif kind == MarketKind.BUY_PRODUCT_WHEAT and step % 3 == 0:
                            value = min(maximum, 8)
                        elif kind == MarketKind.BUY_PRODUCT_FERTILIZER and step % 6 == 0:
                            value = min(maximum, 5)
                        elif kind == MarketKind.BUY_SEED_WHEAT and step % 4 == 0:
                            value = min(maximum, 3)
                        elif kind == MarketKind.BUY_ANIMAL_GOOSE and step % 11 == 0:
                            value = min(maximum, 2)
                    choices.append(value)
                    if value:
                        if kind == MarketKind.HIRE:
                            for _ in range(value):
                                _apply_ledger_order(observation, kind, 1, ledger)
                        else:
                            _apply_ledger_order(observation, kind, value, ledger)
                        slots += value if kind == MarketKind.HIRE else 1
                compiled, factors = compile_market_set(
                    observation, choices, order=order, post_unit_shed=shed
                )
                for slot, command in enumerate(compiled):
                    parsed = _parse_market_order(command)
                    assert parsed is not None
                    kind, quantity = parsed
                    legacy_kinds[game, seat, slot] = kind
                    if kind not in (MarketKind.HIRE, MarketKind.BUY_LAND):
                        legacy_quantities[game, seat, slot] = quantity - 1
                values[row] = factors.values
                expected.append(factors)
                base = compile_action(
                    observation,
                    units[game, seat],
                    np.zeros(MAX_MARKET_ORDERS, dtype=np.uint8),
                    np.zeros(MAX_MARKET_ORDERS, dtype=np.uint8),
                )
                game_actions.append({**base, "market": compiled})
            actions.append(game_actions)
        actual = native.market_set_masks(
            units, values, sell_order=order.sell_order, hire_last=order.hire_last
        )
        np.testing.assert_array_equal(actual["values"], values)
        np.testing.assert_array_equal(actual["masks"], np.stack([row.masks for row in expected]))
        np.testing.assert_array_equal(actual["active"], np.stack([row.active for row in expected]))
        np.testing.assert_array_equal(actual["market_kinds"], legacy_kinds)
        np.testing.assert_array_equal(actual["market_quantities"], legacy_quantities)
        for environment, game_actions in zip(environments, actions, strict=True):
            environment.step(game_actions)
        native.step_factors(units, actual["market_kinds"], actual["market_quantities"])
    assert sell_opportunities > 0
    assert hire_opportunities > 0
    assert post_unit_changes > 0
