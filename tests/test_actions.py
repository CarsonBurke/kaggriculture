from __future__ import annotations

import numpy as np
import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official

from kaggriculture.actions import (
    N_QUANTITIES,
    MarketKind,
    UnitAction,
    compile_action,
    market_kind_mask,
    market_order,
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


def test_every_pickup_variant_compiles_to_its_exact_quantity() -> None:
    observation = _observation()
    observation["private"]["shed"].update(
        {"WHEAT": 100, "FERTILIZER": 100, "GOOSE": 100, "COW": 100, "SHEEP": 100}
    )
    expected = {
        **{f"PICKUP_WHEAT_{quantity}": ("WHEAT", quantity) for quantity in (1, 2, 4, 8, 16)},
        **{f"PICKUP_FERTILIZER_{quantity}": ("FERTILIZER", quantity) for quantity in (1, 2, 4, 8)},
        **{
            f"PICKUP_{animal}_{quantity}": (animal, quantity)
            for animal in ("GOOSE", "COW", "SHEEP")
            for quantity in (1, 2, 3, 4)
        },
    }

    mask = unit_action_mask(observation, 0)

    assert len(expected) == 21
    for name, (item, quantity) in expected.items():
        action = UnitAction[name]
        assert mask[action]
        units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
        units[0] = action
        kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
        quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
        compiled = compile_action(observation, units, kinds, quantities)
        assert compiled["farmer"] == ["PICKUP", item, quantity]


@pytest.mark.parametrize(
    ("action", "item", "quantity"),
    [
        *[
            (UnitAction[f"PICKUP_WHEAT_{quantity}"], "WHEAT", quantity)
            for quantity in (1, 2, 4, 8, 16)
        ],
        *[
            (UnitAction[f"PICKUP_FERTILIZER_{quantity}"], "FERTILIZER", quantity)
            for quantity in (1, 2, 4, 8)
        ],
        *[
            (UnitAction[f"PICKUP_{animal}_{quantity}"], animal, quantity)
            for animal in ("GOOSE", "COW", "SHEEP")
            for quantity in (1, 2, 3, 4)
        ],
    ],
)
def test_every_pickup_variant_matches_official_engine(
    action: UnitAction, item: str, quantity: int
) -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 13})
    state = environment.reset(2)
    state[0].observation["private"]["shed"][item] = 100
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[0] = action
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    compiled = compile_action(state[0].observation, units, kinds, quantities)
    following = environment.step([compiled, {}])[0].observation

    assert following["private"]["shed"][item] == 100 - quantity
    assert following["private"]["inventories"][0][item] == quantity


@pytest.mark.parametrize("quantity", range(1, 101))
def test_every_exact_sell_quantity_matches_official_engine(quantity: int) -> None:
    observation = _observation()
    observation["private"]["shed"]["WOOL"] = 100
    farm = observation["farms"][0]
    expected_money = float(farm["money"])
    market_inventory = observation["market"]["inventory"]["WOOL"]
    for offset in range(quantity):
        expected_money += official.market_price("WOOL", market_inventory + offset)

    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7})
    state = environment.reset(2)
    state[0].observation["private"]["shed"]["WOOL"] = 100
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    kinds[0] = MarketKind.SELL_WOOL
    quantities[0] = quantity - 1
    action = compile_action(state[0].observation, units, kinds, quantities)

    following = environment.step([action, {}])[0].observation

    assert action["market"] == [["SELL", "WOOL", quantity]]
    assert following["private"]["shed"]["WOOL"] == 100 - quantity
    assert following["farms"][0]["money"] == expected_money


def test_pickup_masks_and_compilation_reserve_stock_sequentially() -> None:
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4], [4, 4]]
    observation["private"]["inventories"].extend(({}, {}))
    observation["private"]["shed"]["FERTILIZER"] = 7
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[:3] = (
        UnitAction.PICKUP_FERTILIZER_4,
        UnitAction.PICKUP_FERTILIZER_2,
        UnitAction.PICKUP_FERTILIZER_2,
    )
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    compiled = compile_action(observation, units, kinds, quantities)

    assert compiled["farmer"] == ["PICKUP", "FERTILIZER", 4]
    assert compiled["hands"] == [["PICKUP", "FERTILIZER", 2], ["PASS"]]


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


def test_every_exact_market_quantity_compiles_in_one_slot() -> None:
    assert N_QUANTITIES == 100
    for quantity_index in range(N_QUANTITIES):
        assert market_order(MarketKind.BUY_SEED_WHEAT, quantity_index) == [
            "BUY_SEED",
            "WHEAT",
            quantity_index + 1,
        ]
        assert market_order(MarketKind.SELL_WOOL, quantity_index) == [
            "SELL",
            "WOOL",
            quantity_index + 1,
        ]


def test_drop_is_dominated_and_masked_when_shed_is_full() -> None:
    observation = _observation()
    observation["private"]["shed"]["WHEAT"] = 100
    observation["private"]["inventories"][0]["MILK"] = 1

    mask = unit_action_mask(observation, 0)

    assert not mask[UnitAction.DROP]
