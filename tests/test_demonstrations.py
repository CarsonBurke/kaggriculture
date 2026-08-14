from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from kaggle_environments import make

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.constants import QUANTITY_BINS, SEED_COST
from kaggriculture.demonstrations import (
    DemonstrationError,
    _canonical_unit_command,
    project_demonstration,
    verify_round_trip,
)


def _observation():
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7})
    return environment.reset(2)[0].observation


def _project(observation, action):
    projected = project_demonstration(observation, action)
    verify_round_trip(observation, action, projected)
    return projected


def test_simple_action_projects_and_round_trips() -> None:
    observation = _observation()
    action = {"farmer": ["NORTH"], "hands": [], "market": [["HIRE"]]}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.NORTH
    assert projected.market_kinds[0] == MarketKind.HIRE
    assert projected.market_kinds[1] == MarketKind.STOP
    assert projected.unit_active.sum() == 1
    # Orders slot 0 plus the trained STOP decision at slot 1.
    assert projected.market_active.sum() == 2
    assert projected.canonical_action == {
        "farmer": ["NORTH"],
        "hands": [],
        "market": [["HIRE"]],
    }


def test_decorated_arguments_reduce_to_engine_arity() -> None:
    # The official interpreter reads no arguments for FEED and friends; v27
    # emits decorated forms like ["FEED", "WHEAT"].
    assert _canonical_unit_command(["FEED", "WHEAT"]) == ["FEED"]
    assert _canonical_unit_command(["NORTH", "IGNORED"]) == ["NORTH"]
    assert _canonical_unit_command(["DROP", "ALL"]) == ["DROP"]
    assert _canonical_unit_command(["PICKUP", "WHEAT"]) == ["PICKUP", "WHEAT", 1]
    assert _canonical_unit_command(["PLACE", "GOOSE", 1]) == ["PLACE", "GOOSE"]


def test_place_shed_deposit_quantity_is_a_representability_gap() -> None:
    with pytest.raises(DemonstrationError, match="unrepresentable PLACE"):
        _canonical_unit_command(["PLACE", "GOOSE", 2])


def test_pickup_without_quantity_defaults_to_one() -> None:
    observation = _observation()
    observation["private"]["shed"]["WHEAT"] = 5
    action = {"farmer": ["PICKUP", "WHEAT"], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PICKUP_WHEAT_1
    assert projected.canonical_action["farmer"] == ["PICKUP", "WHEAT", 1]


def test_pickup_over_ask_clamps_to_shed_stock_like_the_engine() -> None:
    observation = _observation()
    observation["private"]["shed"]["WHEAT"] = 3
    action = {"farmer": ["PICKUP", "WHEAT", 4], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PICKUP_WHEAT_3
    assert projected.canonical_action["farmer"] == ["PICKUP", "WHEAT", 3]


def test_pickup_from_empty_shed_is_the_engine_no_op_pass() -> None:
    observation = _observation()
    observation["private"]["shed"].pop("WHEAT", None)
    action = {"farmer": ["PICKUP", "WHEAT", 2], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PASS
    assert projected.canonical_action["farmer"] == ["PASS"]


def test_masked_command_the_engine_would_no_op_projects_to_pass() -> None:
    observation = _observation()
    # The farmer starts on an empty tile: WATER is masked out and the engine
    # silently no-ops it (v27's open-loop trace emits such commands).
    action = {"farmer": ["WATER"], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PASS
    assert projected.canonical_action["farmer"] == ["PASS"]


def test_hire_beyond_the_unit_cap_is_a_representability_error() -> None:
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4]] * 15
    observation["farms"][0]["money"] = 1_000_000.0
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}

    with pytest.raises(DemonstrationError, match="16-unit cap"):
        project_demonstration(observation, action)


def test_market_over_ask_clamps_to_the_ledger_affordability_bound() -> None:
    observation = _observation()
    observation["farms"][0]["money"] = float(4 * SEED_COST["MELON"] + 10)
    action = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 7]]}

    projected = _project(observation, action)

    assert projected.market_kinds[0] == MarketKind.BUY_SEED_MELON
    assert QUANTITY_BINS[projected.market_quantities[0]] == 4
    assert projected.canonical_action["market"] == [["BUY_SEED", "MELON", 4]]


def test_zero_fill_market_order_is_dropped_and_later_orders_shift() -> None:
    observation = _observation()
    observation["private"]["shed"]["FERTILIZER"] = 0
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "FERTILIZER", 1], ["BUY_SEED", "WHEAT", 1]],
    }

    projected = _project(observation, action)

    assert projected.market_kinds[0] == MarketKind.BUY_SEED_WHEAT
    assert projected.market_kinds[1] == MarketKind.STOP
    assert projected.canonical_action["market"] == [["BUY_SEED", "WHEAT", 1]]


def test_market_queue_truncates_at_the_engine_cap() -> None:
    observation = _observation()
    observation["farms"][0]["money"] = 1_000_000_000.0
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 12}

    projected = _project(observation, action)

    assert (projected.market_kinds == MarketKind.HIRE).sum() == 10
    assert projected.market_active.all()
    assert len(projected.canonical_action["market"]) == 10


def test_verify_round_trip_rejects_tampered_factors() -> None:
    observation = _observation()
    action = {"farmer": ["NORTH"], "hands": [], "market": []}
    projected = project_demonstration(observation, action)
    tampered_units = projected.unit_actions.copy()
    tampered_units[0] = int(UnitAction.SOUTH)
    tampered = dataclasses.replace(projected, unit_actions=tampered_units)

    with pytest.raises(DemonstrationError, match="round trip diverged"):
        verify_round_trip(observation, action, tampered)


def test_more_demonstrated_hands_than_units_is_an_error() -> None:
    observation = _observation()
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

    with pytest.raises(DemonstrationError, match="hands"):
        project_demonstration(observation, action)


def test_every_step_of_a_real_episode_projects_for_both_seats() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 40, "seed": 11})
    environment.run(["starter", "starter"])
    steps = environment.steps
    pairs = 0
    for index in range(len(steps) - 1):
        for seat in (0, 1):
            observation = dict(steps[index][seat].observation)
            observation.setdefault("step", index)
            action = steps[index + 1][seat].action
            assert isinstance(action, dict)
            projected = _project(observation, action)
            assert projected.unit_masks[projected.unit_active, :].any(axis=1).all()
            pairs += 1
    assert pairs == 2 * (len(steps) - 1)


def test_projected_targets_always_satisfy_their_own_masks() -> None:
    observation = _observation()
    observation["private"]["shed"]["WHEAT"] = 3
    action = {
        "farmer": ["PICKUP", "WHEAT", 4],
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 2]],
    }

    projected = _project(observation, action)

    for unit in np.flatnonzero(projected.unit_active):
        assert projected.unit_masks[unit, projected.unit_actions[unit]]
    for slot in np.flatnonzero(projected.market_active):
        assert projected.market_kind_masks[slot, projected.market_kinds[slot]]
    for slot in np.flatnonzero(projected.market_quantity_active):
        assert projected.market_quantity_masks[slot, projected.market_quantities[slot]]
