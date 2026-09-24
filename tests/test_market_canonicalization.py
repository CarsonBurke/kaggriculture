from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from kaggle_environments import make

from kaggriculture.demonstrations import DemonstrationError, canonicalize_market_orders
from kaggriculture.opponents import normalize_opponent


def _replay(*args, **kwargs):
    path = Path(__file__).resolve().parents[1] / "scripts/build_canonical_market_corpus.py"
    spec = importlib.util.spec_from_file_location("build_canonical_market_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._replay(*args, **kwargs)


def test_fixed_order_merges_quantities_and_preserves_hire_count() -> None:
    original = [
        ["BUY_SEED", "WHEAT", 2],
        ["SELL", "WOOL", 3],
        ["HIRE"],
        ["SELL", "WHEAT", 4],
        ["BUY_SEED", "WHEAT", 5],
        ["HIRE"],
        ["SELL", "WOOL", 1],
    ]
    assert canonicalize_market_orders({}, original) == [
        ["SELL", "WHEAT", 4],
        ["SELL", "WOOL", 4],
        ["HIRE"],
        ["HIRE"],
        ["BUY_SEED", "WHEAT", 7],
    ]


def test_impact_order_prioritizes_price_moving_sale() -> None:
    observation = {"market": {"inventory": {"WOOL": 10_000, "WHEAT": 10_000}}}
    orders = [["SELL", "WHEAT", 1], ["SELL", "WOOL", 20]]
    assert canonicalize_market_orders(observation, orders, sell_order="impact")[0] == [
        "SELL",
        "WOOL",
        20,
    ]


def test_invalid_land_repeat_is_explicit_representability_gap() -> None:
    with pytest.raises(DemonstrationError, match="multiple BUY_LAND"):
        canonicalize_market_orders({}, [["BUY_LAND"], ["BUY_LAND"]])


def test_zero_quantity_is_dropped() -> None:
    assert canonicalize_market_orders({}, [["SELL", "MILK", 0], ["HIRE"]]) == [["HIRE"]]


def test_animals_precede_products_in_proposed_interface() -> None:
    orders = [["BUY_PRODUCT", "WHEAT", 1], ["BUY_ANIMAL", "COW", 1]]
    assert canonicalize_market_orders({}, orders) == [
        ["BUY_ANIMAL", "COW", 1],
        ["BUY_PRODUCT", "WHEAT", 1],
    ]


def test_hire_last_moves_hires_after_all_purchases() -> None:
    orders = [["HIRE"], ["BUY_PRODUCT", "WHEAT", 2], ["SELL", "MILK", 1]]
    assert canonicalize_market_orders({}, orders, hire_last=True) == [
        ["SELL", "MILK", 1],
        ["BUY_PRODUCT", "WHEAT", 2],
        ["HIRE"],
    ]


def test_recorded_actions_replay_exactly_in_official_engine() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 17})
    environment.run(["pass", "pass"])
    raw = {
        "observations": [
            {"observation": {**dict(step[0].observation), "step": index}}
            for index, step in enumerate(environment.steps[:-1])
        ]
    }
    actions = [step[0].action for step in environment.steps[1:]]

    first_difference, _, statuses = _replay(raw, 17, 0, "pass", actions=actions)

    assert first_difference is None
    assert statuses == ["DONE", "DONE"]


def test_one_step_engine_branch_matches_original() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/audit_market_relabel_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_market_relabel_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 17})
    environment.reset(2)
    branch = module._branch(environment)

    assert module._signature(branch.step([{}, {}])) == module._signature(environment.step([{}, {}]))
    branch = module._branch(environment)
    assert module._signature(branch.step([{}, {}])) == module._signature(environment.step([{}, {}]))


def test_stage0b_wrapper_calls_public_teacher() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/probe_market_canonicalization.py"
    spec = importlib.util.spec_from_file_location("probe_market_canonicalization", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, teacher = normalize_opponent("public-v16")
    row = module._play((teacher, "pass", "control", 0, 0))
    assert row["old_orders"] > 0
    assert row["statuses"] == ["DONE", "DONE"]
