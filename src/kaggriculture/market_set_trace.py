"""Offline official-engine trace of effective market fills for interface-3 BC.

Use with ``make(..., debug=True)`` in a single process. This patches the
interpreter's market functions only during one synchronous step; production
rollout and inference do not import it.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as official

from kaggriculture.actions import MarketKind

_TRACE_LOCK = threading.Lock()


def trace_effective_market_step(
    environment: Any,
    actions: Sequence[dict[str, Any]],
    *,
    logs: list[dict[str, Any]] | None = None,
    step_fn: Callable[..., list[Any]] | None = None,
) -> tuple[list[list[list[Any]]], list[Any]]:
    """Advance one debug-engine step and return actually filled own orders.

    The returned orders aggregate successful commits by kind, preserving each
    kind's first executed position. In particular, failed HIRE or partial
    quantified requests contribute only their successful fills. The caller
    should replay from the original paired state, including the opponent.
    """
    if not environment.debug:
        raise ValueError("effective-fill tracing requires an official engine with debug=True")
    if len(actions) != 2:
        raise ValueError("market fill tracing requires both player actions")

    totals: list[dict[MarketKind, int]] = [defaultdict(int), defaultdict(int)]
    farm_seats: dict[int, int] = {}
    original_process = official._process_market
    original_commit = official._commit_unit
    original_hire = official._do_hire
    original_land = official._do_buy_land

    def process(state: list[Any], env: Any) -> None:
        farms = state[0].observation.farms
        farm_seats.update((id(farm), seat) for seat, farm in enumerate(farms))
        original_process(state, env)

    def commit(
        op: str,
        item: str,
        price: float,
        farm: Any,
        private: Any,
        market: Any,
        shed_capacity: int = 100,
    ) -> bool:
        filled = original_commit(op, item, price, farm, private, market, shed_capacity)
        if filled:
            name = f"{op}_{item}" if op != "SELL" else f"SELL_{item}"
            totals[farm_seats[id(farm)]][MarketKind[name]] += 1
        return filled

    def hire(farm: Any, private: Any, board_size: int, mult: int = 1) -> None:
        before = farm["hires_today"]
        original_hire(farm, private, board_size, mult)
        if farm["hires_today"] > before:
            totals[farm_seats[id(farm)]][MarketKind.HIRE] += 1

    def land(farm: Any, board_size: int) -> None:
        before = len(farm["unlocked_quadrants"])
        original_land(farm, board_size)
        if len(farm["unlocked_quadrants"]) > before:
            totals[farm_seats[id(farm)]][MarketKind.BUY_LAND] += 1

    with _TRACE_LOCK:
        official._process_market = process
        official._commit_unit = commit
        official._do_hire = hire
        official._do_buy_land = land
        try:
            result = (step_fn or environment.step)(list(actions), logs)
        finally:
            official._process_market = original_process
            official._commit_unit = original_commit
            official._do_hire = original_hire
            official._do_buy_land = original_land

    effective: list[list[list[Any]]] = []
    for player in totals:
        orders: list[list[Any]] = []
        for kind, count in player.items():
            if kind == MarketKind.HIRE:
                orders.extend([["HIRE"] for _ in range(count)])
            elif kind == MarketKind.BUY_LAND:
                orders.extend([["BUY_LAND"] for _ in range(count)])
            else:
                category, item = kind.name.rsplit("_", 1)
                orders.append([category, item, count])
        effective.append(orders)
    return effective, result


def trace_effective_market_run(
    environment: Any, players: Sequence[Any]
) -> tuple[list[list[list[list[Any]]]], list[list[Any]]]:
    """Record fills during a live debug-engine game, including random opponents."""
    if not environment.debug:
        raise ValueError("effective-fill tracing requires debug=True")
    original_step = environment.step
    fills: list[list[list[list[Any]]]] = []

    def traced_step(actions: list[Any], logs: list[dict[str, Any]] | None = None) -> list[Any]:
        effective, result = trace_effective_market_step(
            environment, actions, logs=logs, step_fn=original_step
        )
        fills.append(effective)
        return result

    environment.step = traced_step
    try:
        steps = environment.run(list(players))
    finally:
        environment.step = original_step
    return fills, steps
