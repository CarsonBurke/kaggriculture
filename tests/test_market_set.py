from __future__ import annotations

import numpy as np
import pytest
from kaggle_environments import make

from kaggriculture.actions import MarketKind, MarketLedger
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.market_set import (
    MARKET_SET_ALL_INDEX,
    MARKET_SET_KINDS,
    MARKET_SET_RAW_CHOICES,
    MarketSetOrder,
    compile_market_set,
    marginalize_market_set_all,
    market_set_raw_mask,
    market_set_value_mask,
    project_effective_market_set,
)
from kaggriculture.market_set_trace import trace_effective_market_run, trace_effective_market_step


def _observation() -> dict:
    return (
        make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7}).reset(2)[0].observation
    )


def _choices(order: MarketSetOrder, **values: int) -> list[int]:
    return [values.get(kind.name, 0) for kind in order.decision_kinds]


def test_set_has_every_non_stop_kind_once() -> None:
    assert len(MARKET_SET_KINDS) == 21
    assert set(MARKET_SET_KINDS) == set(MarketKind) - {MarketKind.STOP}
    assert MarketSetOrder(hire_last=True).decision_kinds[-1] == MarketKind.HIRE


def test_zero_and_all_alias_have_distinct_meanings() -> None:
    observation = _observation()
    observation["private"]["shed"]["CARROT"] = 3
    mask = market_set_value_mask(
        observation, MarketKind.SELL_CARROT, MarketLedger.from_observation(observation), 0
    )
    assert mask[:4].all()
    assert not mask[4:].any()
    raw_mask = market_set_raw_mask(mask)
    assert raw_mask.shape == (MARKET_SET_RAW_CHOICES,)
    assert raw_mask[MARKET_SET_ALL_INDEX]

    raw_logits = np.full(MARKET_SET_RAW_CHOICES, -10.0)
    raw_logits[0] = 1.0
    raw_logits[3] = 2.0
    raw_logits[-1] = 3.0
    effective = marginalize_market_set_all(raw_logits, mask)
    assert effective[0] == 1.0
    assert effective[3] == pytest.approx(np.logaddexp(2.0, 3.0))
    assert effective[2] == -10.0


def test_hire_count_obeys_money_hand_cap_and_slot_budget() -> None:
    observation = _observation()
    farm = observation["farms"][0]
    farm["money"] = 4
    ledger = MarketLedger.from_observation(observation)
    mask = market_set_value_mask(observation, MarketKind.HIRE, ledger, 0)
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(
        market_set_value_mask(observation, MarketKind.HIRE, ledger, MAX_MARKET_ORDERS - 1)
    ).tolist() == [0, 1]

    farm["hands"] = [[4, 4]] * (MAX_UNITS - 2)
    assert np.flatnonzero(
        market_set_value_mask(observation, MarketKind.HIRE, ledger, 0)
    ).tolist() == [0, 1]


def test_compiler_respects_canonical_order_and_ledger_dependencies() -> None:
    observation = _observation()
    observation["private"]["shed"]["CARROT"] = 2
    observation["farms"][0]["money"] = 0
    order = MarketSetOrder(hire_last=False)
    compiled, factors = compile_market_set(
        observation,
        _choices(order, SELL_CARROT=2, HIRE=1, BUY_SEED_WHEAT=1),
        order=order,
    )
    assert compiled == [
        ["SELL", "CARROT", 2],
        ["HIRE"],
        ["BUY_SEED", "WHEAT", 1],
    ]
    assert factors.values.shape == (21,)
    assert factors.masks.shape == (21, 101)
    assert factors.active.shape == (21,)
    assert factors.masks[order.decision_kinds.index(MarketKind.BUY_SEED_WHEAT), 1]


def test_tenth_slot_turns_later_decisions_inactive() -> None:
    observation = _observation()
    observation["farms"][0]["money"] = 10_000
    order = MarketSetOrder()
    compiled, factors = compile_market_set(observation, _choices(order, HIRE=10), order=order)
    assert compiled == [["HIRE"]] * 10
    assert factors.active[order.decision_kinds.index(MarketKind.BUY_LAND)] == 0
    assert factors.masks[order.decision_kinds.index(MarketKind.BUY_LAND), 0]


def test_projection_requires_effective_hire_count() -> None:
    observation = _observation()
    observation["farms"][0]["money"] = 4
    order = MarketSetOrder()
    _, factors = project_effective_market_set(observation, [["HIRE"]] * 3, order=order)
    assert factors.values[order.decision_kinds.index(MarketKind.HIRE)] == 3
    with pytest.raises(ValueError, match="HIRE value 4 is not legal"):
        project_effective_market_set(observation, [["HIRE"]] * 4, order=order)


def test_impact_order_changes_compiled_sale_priority() -> None:
    observation = _observation()
    observation["private"]["shed"].update(WHEAT=1, WOOL=20)
    observation["market"]["inventory"].update(WHEAT=10_000, WOOL=10_000)
    order = MarketSetOrder(sell_order="impact")
    compiled, _ = compile_market_set(
        observation, _choices(order, SELL_WHEAT=1, SELL_WOOL=20), order=order
    )
    assert compiled[0] == ["SELL", "WOOL", 20]


def test_engine_trace_labels_partial_hire_fill() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7}, debug=True)
    environment.reset(2)
    environment.state[0].observation.farms[0]["money"] = 7
    before = environment.state[0].observation
    effective, result = trace_effective_market_step(environment, [{"market": [["HIRE"]] * 5}, {}])
    assert effective[0] == [["HIRE"]] * 4
    assert result[0].observation.farms[0]["hires_today"] == 4
    order = MarketSetOrder()
    compiled, factors = project_effective_market_set(before, effective[0], order=order)
    assert compiled == [["HIRE"]] * 4
    assert factors.values[order.decision_kinds.index(MarketKind.HIRE)] == 4


def test_engine_trace_labels_partial_trade_fill() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 7}, debug=True)
    environment.reset(2)
    environment.state[0].observation.farms[0]["money"] = 25
    before = environment.state[0].observation
    effective, result = trace_effective_market_step(
        environment, [{"market": [["BUY_SEED", "WHEAT", 4]]}, {}]
    )
    assert effective[0] == [["BUY_SEED", "WHEAT", 2]]
    assert result[0].observation.private["seeds"]["WHEAT"] == 2
    order = MarketSetOrder()
    compiled, factors = project_effective_market_set(before, effective[0], order=order)
    assert compiled == [["BUY_SEED", "WHEAT", 2]]
    assert factors.values[order.decision_kinds.index(MarketKind.BUY_SEED_WHEAT)] == 2


def test_live_trace_records_random_opponent_game() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 19}, debug=True)
    fills, steps = trace_effective_market_run(environment, ["starter", "random"])
    assert len(fills) == 7
    assert len(steps) == 8
    assert all(len(row) == 2 for row in fills)
    assert [str(state.status) for state in steps[-1]] == ["DONE", "DONE"]
