from __future__ import annotations

import numpy as np
from kaggle_environments import make

from kaggriculture.actions import (
    MarketKind,
    UnitAction,
    compile_action,
    market_kind_mask,
    quantity_mask,
    unit_action_mask,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS, QUANTITY_BINS


def _observation():
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7})
    return environment.reset(2)[0].observation


def test_initial_masks_exclude_impossible_field_operations() -> None:
    observation = _observation()

    mask = unit_action_mask(observation, 0)

    assert mask[UnitAction.PASS]
    assert mask[UnitAction.NORTH]
    assert mask[UnitAction.BUILD_PASTURE]
    assert not mask[UnitAction.PLANT_WHEAT]
    assert not mask[UnitAction.FEED]


def test_initial_market_mask_and_quantity_budget() -> None:
    observation = _observation()

    kinds = market_kind_mask(observation)
    quantities = quantity_mask(observation, MarketKind.BUY_ANIMAL_COW)

    assert kinds[MarketKind.HIRE]
    assert kinds[MarketKind.BUY_LAND]
    assert kinds[MarketKind.BUY_ANIMAL_COW]
    assert not kinds[MarketKind.SELL_MILK]
    assert len(QUANTITY_BINS) == 100
    assert quantities[:7].all()
    assert not quantities[7:].any()


def test_compiler_stops_market_queue_and_preserves_unit_count() -> None:
    observation = _observation()
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    kinds[:3] = (MarketKind.HIRE, MarketKind.BUY_SEED_WHEAT, MarketKind.STOP)

    action = compile_action(observation, units, kinds, quantities)

    assert action["farmer"] == ["PASS"]
    assert action["hands"] == []
    assert action["market"] == [["HIRE"], ["BUY_SEED", "WHEAT", 1]]


def test_compiler_reserves_shed_stock_across_unit_pickups() -> None:
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4]]
    observation["private"]["inventories"].append({})
    observation["private"]["shed"]["WHEAT"] = 20
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[:2] = (UnitAction.PICKUP_WHEAT_16, UnitAction.PICKUP_WHEAT_4)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    action = compile_action(observation, units, kinds, quantities)

    assert action["farmer"] == ["PICKUP", "WHEAT", 16]
    assert action["hands"] == [["PICKUP", "WHEAT", 4]]


def test_compiler_supports_small_pickups_for_multiple_hands() -> None:
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4]]
    observation["private"]["inventories"].append({})
    observation["private"]["shed"]["WHEAT"] = 3
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[:2] = (UnitAction.PICKUP_WHEAT_1, UnitAction.PICKUP_WHEAT_2)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    action = compile_action(observation, units, kinds, quantities)

    assert action["farmer"] == ["PICKUP", "WHEAT", 1]
    assert action["hands"] == [["PICKUP", "WHEAT", 2]]


def test_compiler_allows_build_then_place_on_same_turn() -> None:
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4]]
    observation["private"]["inventories"].append({"COW": 1})
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[:2] = (UnitAction.BUILD_PASTURE, UnitAction.PLACE_COW)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    action = compile_action(observation, units, kinds, quantities)

    assert action["farmer"] == ["BUILD_PASTURE"]
    assert action["hands"] == [["PLACE", "COW"]]


def test_compiler_allows_plant_then_water_on_same_turn() -> None:
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4]]
    observation["private"]["inventories"].append({})
    observation["private"]["seeds"]["WHEAT"] = 1
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[:2] = (UnitAction.PLANT_WHEAT, UnitAction.WATER)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    action = compile_action(observation, units, kinds, quantities)

    assert action["farmer"] == ["PLANT", "WHEAT"]
    assert action["hands"] == [["WATER"]]


def test_compiler_emits_large_exact_quantity_in_one_market_slot() -> None:
    observation = _observation()
    observation["private"]["shed"]["WHEAT"] = 53
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    kinds[0] = MarketKind.SELL_WHEAT
    quantities[0] = 52

    action = compile_action(observation, units, kinds, quantities)

    assert action["market"] == [["SELL", "WHEAT", 53]]


def test_drop_is_dominated_and_masked_when_shed_is_full() -> None:
    observation = _observation()
    observation["private"]["shed"]["WHEAT"] = 100
    observation["private"]["inventories"][0]["MILK"] = 1

    mask = unit_action_mask(observation, 0)

    assert not mask[UnitAction.DROP]
