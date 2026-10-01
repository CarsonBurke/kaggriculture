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
    perturb_action,
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


def test_place_product_deposit_projects_held_quantity() -> None:
    observation = _observation()
    observation["private"]["inventories"][0]["WOOL"] = 15
    action = {"farmer": ["PLACE", "WOOL", 15], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PLACE_WOOL
    assert projected.canonical_action["farmer"] == ["PLACE", "WOOL", 15]


def test_place_product_over_ask_clamps_to_held_like_the_engine() -> None:
    observation = _observation()
    observation["private"]["inventories"][0]["MILK"] = 3
    action = {"farmer": ["PLACE", "MILK", 6], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PLACE_MILK
    assert projected.canonical_action["farmer"] == ["PLACE", "MILK", 3]


def test_partial_product_deposit_is_unrepresentable_when_strict() -> None:
    # demand-advance4 keeps wheat back to FEED: ["PLACE", item] deposits one.
    observation = _observation()
    observation["private"]["inventories"][0]["WHEAT"] = 3
    action = {"farmer": ["PLACE", "WHEAT"], "hands": [], "market": []}

    with pytest.raises(DemonstrationError, match="round trip diverged"):
        _project(observation, action)


def test_partial_product_deposit_relabels_to_the_whole_deposit_when_opted_in() -> None:
    observation = _observation()
    observation["private"]["inventories"][0]["WHEAT"] = 3
    action = {"farmer": ["PLACE", "WHEAT"], "hands": [], "market": []}

    projected = project_demonstration(observation, action, deposit_all_products=True)
    verify_round_trip(observation, action, projected)

    assert projected.unit_actions[0] == UnitAction.PLACE_WHEAT
    assert projected.canonical_action["farmer"] == ["PLACE", "WHEAT", 3]
    assert projected.relabeled_partial_deposits == 1


def test_whole_product_deposit_is_not_counted_as_a_relabel() -> None:
    observation = _observation()
    observation["private"]["inventories"][0]["WOOL"] = 4
    action = {"farmer": ["PLACE", "WOOL", 4], "hands": [], "market": []}

    projected = project_demonstration(observation, action, deposit_all_products=True)

    assert projected.canonical_action["farmer"] == ["PLACE", "WOOL", 4]
    assert projected.relabeled_partial_deposits == 0


def test_empty_market_order_is_unread_like_the_engine() -> None:
    # The engine's `_parse_order([])` is None: a hole that reads nothing.
    observation = _observation()
    action = {"farmer": ["PASS"], "hands": [], "market": [[], ["HIRE"]]}

    projected = _project(observation, action)

    assert projected.market_kinds[0] == MarketKind.HIRE
    assert projected.market_kinds[1] == MarketKind.STOP
    assert projected.canonical_action["market"] == [["HIRE"]]


def test_unknown_unit_opcode_is_the_engine_no_op_pass() -> None:
    # Hosted agents emit ["NOOP"]; the interpreter falls through every branch.
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7})
    observation = environment.reset(2)[0].observation
    noop = {"farmer": ["NOOP"], "hands": [], "market": []}
    passing = {"farmer": ["PASS"], "hands": [], "market": []}
    after_noop = environment.step([noop, passing])[0].observation
    environment.reset(2)
    after_pass = environment.step([passing, passing])[0].observation
    assert after_noop == after_pass

    projected = _project(observation, noop)

    assert projected.unit_actions[0] == UnitAction.PASS
    assert projected.canonical_action["farmer"] == ["PASS"]


def test_unknown_market_opcode_is_unread_like_the_engine() -> None:
    # The engine's `_parse_order(["NOOP"])` is None, like an empty order.
    observation = _observation()
    action = {"farmer": ["PASS"], "hands": [], "market": [["NOOP"], ["HIRE"]]}

    projected = _project(observation, action)

    assert projected.market_kinds[0] == MarketKind.HIRE
    assert projected.canonical_action["market"] == [["HIRE"]]


def test_zero_quantity_market_order_is_discarded() -> None:
    observation = _observation()
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_SEED", "CARROT", 0], ["HIRE"]],
    }

    projected = _project(observation, action)

    assert projected.canonical_action["market"] == [["HIRE"]]


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


def test_pickup_of_an_item_the_shed_lacks_is_pass_whatever_the_item() -> None:
    # The engine clamps to shed stock before anything else, so a pickup the
    # factored space has no action for is still PASS when nothing moves.
    observation = _observation()
    observation["private"]["shed"].pop("MILK", None)
    action = {"farmer": ["PICKUP", "MILK", 2], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PASS


def test_pickup_of_a_product_the_shed_holds_is_unrepresentable() -> None:
    observation = _observation()
    observation["private"]["shed"]["MILK"] = 3
    action = {"farmer": ["PICKUP", "MILK", 2], "hands": [], "market": []}

    with pytest.raises(DemonstrationError, match="unknown pickup item"):
        project_demonstration(observation, action)


def test_masked_command_the_engine_would_no_op_projects_to_pass() -> None:
    observation = _observation()
    # The farmer starts on an empty tile: WATER is masked out and the engine
    # silently no-ops it (v27's open-loop trace emits such commands).
    action = {"farmer": ["WATER"], "hands": [], "market": []}

    projected = _project(observation, action)

    assert projected.unit_actions[0] == UnitAction.PASS
    assert projected.canonical_action["farmer"] == ["PASS"]


def _refertilize_observation():
    # The farmer stands on a crop already fertilized through day+2 and holds
    # fertilizer: the mask forbids FERTILIZE, yet the engine would spend one.
    observation = _observation()
    x, y = observation["farms"][0]["farmer"]
    observation["farms"][0]["tiles"][y][x] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "fertilized_until_day": int(observation["day"]) + 2,
    }
    observation["private"]["inventories"][0]["FERTILIZER"] = 1
    return observation


def test_redundant_fertilize_is_a_mask_divergence_when_strict() -> None:
    observation = _refertilize_observation()
    action = {"farmer": ["FERTILIZE"], "hands": [], "market": []}

    with pytest.raises(DemonstrationError, match="mask divergence"):
        _project(observation, action)


def test_redundant_fertilize_relabels_to_pass_when_opted_in() -> None:
    observation = _refertilize_observation()
    action = {"farmer": ["FERTILIZE"], "hands": [], "market": []}

    projected = project_demonstration(observation, action, redundant_fertilize_as_pass=True)
    verify_round_trip(observation, action, projected)

    assert projected.unit_actions[0] == UnitAction.PASS
    assert projected.canonical_action["farmer"] == ["PASS"]
    assert projected.relabeled_redundant_fertilize == 1


def test_perturbed_action_is_a_legal_single_deviation() -> None:
    """One decision changes; the rest executes exactly as the teacher wrote it."""
    observation = _observation()
    observation["farms"][0]["hands"] = [[4, 4], [5, 4]]
    # A decorated command the projection canonicalizes stays as written.
    action = {"farmer": ["PASS"], "hands": [["NORTH", "IGNORED"], ["PASS"]], "market": []}
    teacher = _project(observation, action)
    kinds = {"unit": 0, "market": 0}
    for seed in range(64):
        deviation = perturb_action(observation, action, np.random.default_rng(seed))
        if deviation is None:
            continue
        projected = _project(observation, deviation)
        written = [action["farmer"], *action["hands"]]
        executed = [deviation["farmer"], *deviation["hands"]]
        changed = [index for index, command in enumerate(written) if command != executed[index]]
        if deviation["market"] == action["market"]:
            assert len(changed) == 1
            assert int((projected.unit_actions != teacher.unit_actions).sum()) == 1
            kinds["unit"] += 1
        else:
            assert changed == [] and len(deviation["market"]) == 1
            kinds["market"] += 1
    assert kinds["unit"] >= 16 and kinds["market"] >= 16


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


def test_seed_over_ask_rejects_an_executed_fill_above_the_quantity_space() -> None:
    observation = _observation()
    observation["farms"][0]["money"] = float(101 * SEED_COST["WHEAT"])
    action = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 101]]}

    with pytest.raises(DemonstrationError, match="outside the factored quantity space"):
        project_demonstration(observation, action)


def test_seed_over_ask_projects_an_affordable_fill_within_the_quantity_space() -> None:
    observation = _observation()
    observation["farms"][0]["money"] = float(100 * SEED_COST["WHEAT"])
    action = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 101]]}

    projected = _project(observation, action)

    assert QUANTITY_BINS[projected.market_quantities[0]] == 100
    assert projected.canonical_action["market"] == [["BUY_SEED", "WHEAT", 100]]


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
