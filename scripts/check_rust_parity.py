#!/usr/bin/env python3
"""Differentially replay compact factors through Rust and the official engine."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from kaggle_environments import make

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS, compile_action
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.rust_env import load_native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8, help="Number of consecutive games")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--factor-seed", type=int, default=20260812)
    parser.add_argument("--mode", choices=("pass", "random"), default="random")
    parser.add_argument("--debug-build", action="store_true")
    return parser.parse_args()


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _official_snapshot(environment: Any) -> dict[str, Any]:
    states = environment.state
    public = states[0].observation
    return _plain(
        {
            "step": int(public.get("step", 0) or 0),
            "day": int(public.get("day", 0) or 0),
            "hour": int(public.get("hour", 0) or 0),
            "done": bool(environment.done),
            "farms": public.get("farms") or [],
            "privates": [state.observation.get("private") or {} for state in states],
            "market": public.get("market") or {},
            "town": public.get("town") or {},
            "rewards": [state.reward for state in states],
            "statuses": [str(state.status) for state in states],
        }
    )


def _first_difference(left: Any, right: Any, path: str = "root") -> str | None:
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return f"{path} keys: official={sorted(left_keys)} rust={sorted(right_keys)}"
        for key in sorted(left_keys):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} length: official={len(left)} rust={len(right)}"
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_value, right_value, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path}: official={left!r} rust={right!r}"
    return None


def _factors(
    games: int,
    mode: str,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (games, 2)
    if mode == "pass":
        return (
            np.zeros((*shape, MAX_UNITS), dtype=np.uint8),
            np.zeros((*shape, MAX_MARKET_ORDERS), dtype=np.uint8),
            np.zeros((*shape, MAX_MARKET_ORDERS), dtype=np.uint8),
        )
    return (
        generator.integers(
            0,
            N_UNIT_ACTIONS,
            size=(*shape, MAX_UNITS),
            dtype=np.uint8,
        ),
        generator.integers(
            0,
            N_MARKET_KINDS,
            size=(*shape, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        ),
        generator.integers(
            0,
            N_QUANTITIES,
            size=(*shape, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        ),
    )


def _compare_snapshots(environments: list[Any], rust: Any, transition: int) -> None:
    for game, environment in enumerate(environments):
        official = _official_snapshot(environment)
        native = json.loads(rust.snapshot_json(game))
        difference = _first_difference(official, native)
        if difference is not None:
            raise AssertionError(
                f"parity divergence after transition {transition}, game {game}: {difference}"
            )


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if not 0 <= args.steps <= 719:
        raise ValueError("--steps must be between 0 and 719")
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.uint64)
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
    native = load_native(release=not args.debug_build)
    rust = native.BatchEnv(seeds)
    generator = np.random.default_rng(args.factor_seed)
    _compare_snapshots(environments, rust, transition=0)

    for transition in range(1, args.steps + 1):
        unit_actions, market_kinds, quantities = _factors(args.seeds, args.mode, generator)
        for game, environment in enumerate(environments):
            actions = [
                compile_action(
                    environment.state[player].observation,
                    unit_actions[game, player],
                    market_kinds[game, player],
                    quantities[game, player],
                )
                for player in range(2)
            ]
            environment.step(actions)
        rust.step_factors(unit_actions, market_kinds, quantities)
        _compare_snapshots(environments, rust, transition)

    print(
        json.dumps(
            {
                "games": args.seeds,
                "mode": args.mode,
                "transitions_per_game": args.steps,
                "joint_transitions": args.seeds * args.steps,
                "result": "exact parity",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
