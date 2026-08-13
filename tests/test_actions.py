from __future__ import annotations

import numpy as np
import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    QUANTIFIED_MARKET_KINDS,
    MarketKind,
    UnitAction,
    apply_unit_shed_effect,
    apply_unit_tile_effect,
    compile_action,
    copy_tile_grid,
    market_kind_mask,
    market_order,
    quantity_mask,
    unit_action_mask,
)
from kaggriculture.constants import (
    CROPS,
    MARKET_I0,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
    QUANTITY_BINS,
)
from kaggriculture.policy import (
    MarketLedger,
    _apply_ledger_order,
    _ledger_kind_mask,
    _ledger_quantity_mask,
)
from kaggriculture.rust_env import load_native


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
        **{f"PICKUP_WHEAT_{quantity}": ("WHEAT", quantity) for quantity in range(1, 17)},
        **{f"PICKUP_FERTILIZER_{quantity}": ("FERTILIZER", quantity) for quantity in range(1, 9)},
        **{
            f"PICKUP_{animal}_{quantity}": (animal, quantity)
            for animal in ("GOOSE", "COW", "SHEEP")
            for quantity in (1, 2, 3, 4)
        },
    }

    mask = unit_action_mask(observation, 0)

    assert len(expected) == 36
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
        *[(UnitAction[f"PICKUP_WHEAT_{quantity}"], "WHEAT", quantity) for quantity in range(1, 17)],
        *[
            (UnitAction[f"PICKUP_FERTILIZER_{quantity}"], "FERTILIZER", quantity)
            for quantity in range(1, 9)
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
    observation["private"]["shed"]["WHEAT"] = 100
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[:2] = (UnitAction.BUILD_PASTURE, UnitAction.PLACE_COW)
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    action = compile_action(observation, units, kinds, quantities)

    assert action["farmer"] == ["BUILD_PASTURE"]
    assert action["hands"] == [["PLACE", "COW"]]


@pytest.mark.parametrize("position", ([4, 4], [5, 4]))
def test_place_animal_falls_back_to_shed_and_matches_official_engine(
    position: list[int],
) -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7})
    state = environment.reset(2)
    observation = state[0].observation
    observation["farms"][0]["farmer"] = position
    observation["private"]["inventories"][0].update({"WHEAT": 1, "COW": 1})
    observation["private"]["shed"]["WHEAT"] = 99
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[0] = UnitAction.PLACE_COW
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    mask = unit_action_mask(observation, 0)
    compiled = compile_action(observation, units, kinds, quantities)
    following = environment.step([compiled, {}])[0].observation

    assert mask[UnitAction.PLACE_COW]
    assert compiled["farmer"] == ["PLACE", "COW"]
    assert following["private"]["shed"]["WHEAT"] == 99
    assert following["private"]["shed"]["COW"] == 1
    assert following["private"]["inventories"][0] == {"WHEAT": 1}


def test_place_animal_prefers_matching_structure_when_shed_is_full() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7})
    state = environment.reset(2)
    observation = state[0].observation
    observation["farms"][0]["tiles"][4][4] = {"kind": "PASTURE"}
    observation["private"]["inventories"][0]["COW"] = 1
    observation["private"]["shed"]["WHEAT"] = 100
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[0] = UnitAction.PLACE_COW
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    mask = unit_action_mask(observation, 0)
    compiled = compile_action(observation, units, kinds, quantities)
    following = environment.step([compiled, {}])[0].observation

    assert mask[UnitAction.PLACE_COW]
    assert following["farms"][0]["tiles"][4][4]["animal"] == "COW"
    assert following["private"]["shed"]["COW"] == 0
    assert following["private"]["inventories"][0] == {}


def test_place_animal_reserves_last_shed_slot_across_units() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7})
    state = environment.reset(2)
    observation = state[0].observation
    observation["farms"][0]["hands"] = [[4, 4]]
    observation["private"]["inventories"][0]["COW"] = 1
    observation["private"]["inventories"].append({"COW": 1})
    observation["private"]["shed"]["WHEAT"] = 99
    units = np.full(MAX_UNITS, UnitAction.PASS, dtype=np.int64)
    units[:2] = UnitAction.PLACE_COW
    kinds = np.full(MAX_MARKET_ORDERS, MarketKind.STOP, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)

    compiled = compile_action(observation, units, kinds, quantities)
    following = environment.step([compiled, {}])[0].observation

    assert compiled["farmer"] == ["PLACE", "COW"]
    assert compiled["hands"] == [["PASS"]]
    assert following["private"]["shed"]["COW"] == 1
    assert following["private"]["inventories"] == [{}, {"COW": 1}]


def test_build_then_place_does_not_reserve_a_shed_slot() -> None:
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4]]
    observation["private"]["inventories"].append({"COW": 1})
    observation["private"]["shed"]["WHEAT"] = 99
    remaining_shed = dict(observation["private"]["shed"])
    tiles = copy_tile_grid(observation["farms"][0]["tiles"])

    apply_unit_tile_effect(observation, 0, UnitAction.BUILD_PASTURE, tiles)
    apply_unit_shed_effect(observation, 1, UnitAction.PLACE_COW, remaining_shed, tiles)
    apply_unit_tile_effect(observation, 1, UnitAction.PLACE_COW, tiles)

    assert sum(remaining_shed.values()) == 99
    assert remaining_shed["COW"] == 0
    assert tiles[4][4]["animal"] == "COW"


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


def _python_sequential_factor_masks(
    observation: dict,
    unit_actions: np.ndarray,
    market_kinds: np.ndarray,
    market_quantities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    player = int(observation.get("player", 0) or 0)
    farm = observation["farms"][player]
    unit_count = min(MAX_UNITS, 1 + len(farm.get("hands") or []))
    unit_masks = np.zeros((MAX_UNITS, N_UNIT_ACTIONS), dtype=np.bool_)
    unit_active = np.zeros(MAX_UNITS, dtype=np.bool_)
    remaining_seeds = dict(observation["private"].get("seeds") or {})
    remaining_shed = dict(observation["private"].get("shed") or {})
    tiles = copy_tile_grid(farm.get("tiles") or [])
    for unit in range(MAX_UNITS):
        if unit >= unit_count:
            unit_masks[unit, UnitAction.PASS] = True
            continue
        unit_active[unit] = True
        unit_masks[unit] = unit_action_mask(
            observation,
            unit,
            remaining_seeds,
            remaining_shed,
            tiles,
        )
        selected = int(unit_actions[unit])
        if selected >= N_UNIT_ACTIONS or not unit_masks[unit, selected]:
            selected = int(UnitAction.PASS)
        if UnitAction.PLANT_WHEAT <= selected <= UnitAction.PLANT_MELON:
            crop = CROPS[selected - int(UnitAction.PLANT_WHEAT)]
            remaining_seeds[crop] = remaining_seeds.get(crop, 0) - 1
        apply_unit_shed_effect(observation, unit, selected, remaining_shed, tiles)
        apply_unit_tile_effect(observation, unit, selected, tiles)

    market_inventory = (observation.get("market") or {}).get("inventory") or {}
    ledger = MarketLedger(
        money=float(farm.get("money", 0) or 0),
        shed=remaining_shed,
        hires=int(farm.get("hires_today", 0) or 0),
        extra_land=max(0, len(farm.get("unlocked_quadrants") or []) - 1),
        inventory={item: int(market_inventory.get(item, MARKET_I0)) for item in PRODUCTS},
    )
    kind_masks = np.zeros((MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.bool_)
    quantity_masks = np.zeros((MAX_MARKET_ORDERS, N_QUANTITIES), dtype=np.bool_)
    market_active = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
    quantity_active = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
    still_active = True
    for slot in range(MAX_MARKET_ORDERS):
        if not still_active:
            kind_masks[slot, MarketKind.STOP] = True
            quantity_masks[slot, 0] = True
            continue
        market_active[slot] = True
        kind_masks[slot] = _ledger_kind_mask(observation, ledger)
        raw_kind = int(market_kinds[slot])
        kind = (
            MarketKind(raw_kind)
            if raw_kind < N_MARKET_KINDS and kind_masks[slot, raw_kind]
            else MarketKind.STOP
        )
        if kind == MarketKind.STOP:
            quantity_masks[slot, 0] = True
            still_active = False
            continue
        quantity_masks[slot] = _ledger_quantity_mask(observation, kind, ledger)
        quantity_active[slot] = kind in QUANTIFIED_MARKET_KINDS
        raw_quantity = int(market_quantities[slot])
        quantity = (
            QUANTITY_BINS[raw_quantity]
            if raw_quantity < N_QUANTITIES and quantity_masks[slot, raw_quantity]
            else 1
        )
        _apply_ledger_order(observation, kind, quantity, ledger)
    return (
        unit_masks,
        kind_masks,
        quantity_masks,
        unit_active,
        market_active,
        quantity_active,
    )


def test_rust_factor_masks_match_python_ledgers_across_evolving_states() -> None:
    game_count = 8
    transitions = 300
    seeds = np.arange(game_count, dtype=np.uint64)
    environments = [
        make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": int(seed)},
            debug=False,
        )
        for seed in seeds
    ]
    for environment in environments:
        environment.reset(2)
    rust = load_native().BatchEnv(seeds)
    generator = np.random.default_rng(23_887)

    for _ in range(transitions):
        unit_actions = generator.integers(
            0,
            N_UNIT_ACTIONS,
            (game_count, 2, MAX_UNITS),
            dtype=np.uint8,
        )
        market_kinds = generator.integers(
            0,
            N_MARKET_KINDS,
            (game_count, 2, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        )
        market_quantities = generator.integers(
            0,
            N_QUANTITIES,
            (game_count, 2, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        )
        native = rust.factor_masks(unit_actions, market_kinds, market_quantities)
        expected = [
            _python_sequential_factor_masks(
                environment.state[player].observation,
                unit_actions[game, player],
                market_kinds[game, player],
                market_quantities[game, player],
            )
            for game, environment in enumerate(environments)
            for player in range(2)
        ]
        for index, name in enumerate(
            (
                "unit_masks",
                "market_kind_masks",
                "market_quantity_masks",
                "unit_active",
                "market_active",
                "market_quantity_active",
            )
        ):
            np.testing.assert_array_equal(
                np.stack([row[index] for row in expected]),
                native[name],
            )

        for game, environment in enumerate(environments):
            environment.step(
                [
                    compile_action(
                        environment.state[player].observation,
                        unit_actions[game, player],
                        market_kinds[game, player],
                        market_quantities[game, player],
                    )
                    for player in range(2)
                ]
            )
        rust.step_factors(unit_actions, market_kinds, market_quantities)
