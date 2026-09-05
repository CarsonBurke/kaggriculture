"""Persistent native/Python differential gate for a complete production episode.

Run from the repository root after a release build:
  PYTHONPATH=src .venv/bin/python rust/kagg_env/tests/parity_oracle.py
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from kaggle_environments import make

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    MarketKind,
    UnitAction,
    compile_action,
)
from kaggriculture.constants import DEFAULT_REWARD_GAMMA, MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.encoding import (
    encode_observation,
    pair_potential,
    shaped_pair_reward,
    terminal_pair_utility,
)
from kaggriculture.rollout import (
    _ANIMAL_STOCK_COLUMNS,
    _CROP_SEED_COLUMNS,
    _PRODUCT_STOCK_COLUMNS,
    _native_pair_rewards,
)
from kaggriculture.rust_env import load_native
from kaggriculture.tokens import encode_structured_observation

EPISODE_STEPS = 720
MAX_TRANSITIONS = EPISODE_STEPS - 1
ENCODING_NAMES = (
    "board",
    "global_features",
    "critic_features",
    "units",
    "unit_positions",
    "unit_active",
)
STRUCTURED_ENCODING_NAMES = (
    "tile_categorical",
    "tile_continuous",
    "unit_categorical",
    "unit_continuous",
    "unit_active",
    "unit_tile_gather",
    "unit_tile_gather_valid",
    "products",
    "animals",
    "crops",
    "farms",
    "town",
)


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
        if left.keys() != right.keys():
            return f"{path} keys: official={sorted(left)} native={sorted(right)}"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} length: official={len(left)} native={len(right)}"
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_value, right_value, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path}: official={left!r} native={right!r}"
    return None


def _compare_snapshots(environments: list[Any], native: Any, transition: int) -> None:
    for game, environment in enumerate(environments):
        official_snapshot = _official_snapshot(environment)
        native_snapshot = json.loads(native.snapshot_json(game))
        difference = _first_difference(official_snapshot, native_snapshot)
        if difference is not None:
            raise AssertionError(
                f"state divergence at transition {transition}, game {game}: {difference}"
            )


def _compare_encodings(environments: list[Any], encoded: dict[str, Any], transition: int) -> None:
    oracle = []
    for environment in environments:
        oracle.extend(
            (
                encode_observation(
                    environment.state[0].observation,
                    environment.state[1].observation.private,
                ),
                encode_observation(
                    environment.state[1].observation,
                    environment.state[0].observation.private,
                ),
            )
        )
    for name in ENCODING_NAMES:
        expected = np.stack([getattr(row, name) for row in oracle])
        actual = encoded[name]
        if not np.array_equal(expected, actual):
            difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            index = np.unravel_index(np.argmax(difference), difference.shape)
            raise AssertionError(
                f"encoding divergence at transition {transition}, {name}{index}: "
                f"official={expected[index]} native={actual[index]}"
            )


def _compare_structured_encodings(
    environments: list[Any], encoded: dict[str, Any], transition: int
) -> None:
    oracle = []
    for environment in environments:
        oracle.extend(
            (
                encode_structured_observation(
                    environment.state[0].observation,
                    environment.state[1].observation.private,
                ),
                encode_structured_observation(
                    environment.state[1].observation,
                    environment.state[0].observation.private,
                ),
            )
        )
    for name in STRUCTURED_ENCODING_NAMES:
        expected = np.stack([getattr(row, name) for row in oracle])
        np.testing.assert_array_equal(
            np.asarray(encoded[name]),
            expected,
            err_msg=f"structured encoding divergence at transition {transition}: {name}",
        )

    pair_rows = np.arange(len(oracle), dtype=np.int64) ^ 1
    derived = {
        "critic_products": np.asarray(encoded["products"])[pair_rows][:, :, _PRODUCT_STOCK_COLUMNS],
        "critic_animals": np.asarray(encoded["animals"])[pair_rows][:, :, _ANIMAL_STOCK_COLUMNS],
        "critic_crops": np.asarray(encoded["crops"])[pair_rows][:, :, _CROP_SEED_COLUMNS],
        "opponent_unit_categorical": np.asarray(encoded["unit_categorical"])[pair_rows],
        "opponent_unit_continuous": np.asarray(encoded["unit_continuous"])[pair_rows],
        "opponent_unit_active": np.asarray(encoded["unit_active"])[pair_rows],
    }
    for name, actual in derived.items():
        expected = np.stack([getattr(row, name) for row in oracle])
        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"structured critic encoding divergence at transition {transition}: {name}",
        )


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
            N_UNIT_ACTIONS,
            size=(*shape, MAX_UNITS),
            dtype=np.uint8,
        ),
        generator.integers(
            N_MARKET_KINDS,
            size=(*shape, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        ),
        generator.integers(
            N_QUANTITIES,
            size=(*shape, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        ),
    )


def assert_official_native_parity(
    *,
    games: int = 1,
    steps: int = MAX_TRANSITIONS,
    seed_start: int = 0,
    factor_seed: int = 20260812,
    mode: str = "random",
    exercise_submitted_cases: bool = True,
    build: bool = False,
    release: bool = True,
) -> dict[str, Any]:
    """Assert the complete state, encoding, potential, utility, and reward contract."""
    if games < 1:
        raise ValueError("games must be positive")
    if not 0 <= steps <= MAX_TRANSITIONS:
        raise ValueError(f"steps must be between 0 and {MAX_TRANSITIONS}")
    if mode not in {"pass", "random"}:
        raise ValueError("mode must be 'pass' or 'random'")

    seeds = np.arange(seed_start, seed_start + games, dtype=np.uint64)
    official = [
        make(
            "kaggriculture",
            configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
            debug=False,
        )
        for seed in seeds
    ]
    for environment in official:
        environment.reset(2)
    native = load_native(build=build, release=release).BatchEnv(seeds)
    encoded = native.encoded_buffers()
    potentials = np.asarray(
        [
            pair_potential(environment.state[0].observation, environment.state[1].observation)
            for environment in official
        ],
        dtype=np.float32,
    )
    generator = np.random.default_rng(factor_seed)

    for transition in range(steps + 1):
        _compare_snapshots(official, native, transition)
        native.encoded_into(encoded)
        _compare_encodings(official, encoded, transition)
        _compare_structured_encodings(official, native.structured(), transition)
        if transition == steps:
            break

        unit_actions, market_kinds, quantities = _factors(games, mode, generator)
        submitted_units: list[list[list[int]]] | None = None
        overflow_origins: list[tuple[int, int]] = []
        if exercise_submitted_cases and transition < 8:
            unit_actions.fill(int(UnitAction.PASS))
            market_kinds.fill(int(MarketKind.STOP))
            quantities.fill(0)
            if transition == 0:
                market_kinds[:, 0, :].fill(int(MarketKind.HIRE))
            elif transition == 1:
                market_kinds[:, 0, :6].fill(int(MarketKind.HIRE))
            elif transition == 3:
                market_kinds[:, 0, 0].fill(int(MarketKind.BUY_PRODUCT_WHEAT))
                market_kinds[:, 0, 2].fill(int(MarketKind.BUY_SEED_WHEAT))
                market_kinds[:, 0, 1].fill(int(MarketKind.BUY_PRODUCT_FERTILIZER))
                quantities[:, 0, 1].fill(1)
            submitted_units = []
            for game, environment in enumerate(official):
                game_units = [
                    [int(UnitAction.PASS)]
                    * (1 + len(environment.state[player].observation.farms[player].hands))
                    for player in range(2)
                ]
                if transition == 2:
                    farm = environment.state[0].observation.farms[0]
                    if len(farm.hands) != 16 or int(farm.money) != 417:
                        raise AssertionError(
                            f"overflow setup divergence before transition 3, game {game}: "
                            f"hands={len(farm.hands)} money={farm.money}"
                        )
                    overflow_origins.append(tuple(int(value) for value in farm.hands[15]))
                    game_units[0][16] = int(UnitAction.EAST)
                elif transition == 4:
                    game_units[0][0] = int(UnitAction.PICKUP_WHEAT_16)
                    game_units[0][4] = int(UnitAction.PICKUP_FERTILIZER_2)
                elif transition == 5:
                    game_units[0][4] = int(UnitAction.PLANT_WHEAT)
                elif transition in (6, 7):
                    game_units[0][4] = int(UnitAction.FERTILIZE)
                submitted_units.append(game_units)
        expected_potentials = np.empty(games, dtype=np.float32)
        expected_utilities = np.zeros(games, dtype=np.float32)
        expected_rewards = np.empty((games, 2), dtype=np.float32)
        for game, environment in enumerate(official):
            actions = [
                compile_action(
                    environment.state[player].observation,
                    unit_actions[game, player],
                    market_kinds[game, player],
                    quantities[game, player],
                )
                for player in range(2)
            ]
            if submitted_units is not None and transition == 2:
                actions[0]["hands"].append(["EAST"])
            if submitted_units is not None and transition == 4:
                actions[0]["farmer"] = ["PICKUP", "WHEAT", 16]
                actions[0]["hands"][3] = ["PICKUP", "FERTILIZER", 2]
            if submitted_units is not None and transition == 5:
                actions[0]["hands"][3] = ["PLANT", "WHEAT"]
            if submitted_units is not None and transition in (6, 7):
                actions[0]["hands"][3] = ["FERTILIZE"]
            environment.step(actions)
            if submitted_units is not None and transition == 2:
                actual = tuple(
                    int(value) for value in environment.state[0].observation.farms[0].hands[15]
                )
                origin = overflow_origins[game]
                expected = (origin[0] + 1, origin[1])
                if actual != expected:
                    raise AssertionError(
                        f"official overflow unit did not move at transition 3, game {game}: "
                        f"expected={expected} actual={actual}"
                    )
            if submitted_units is not None and transition == 4:
                private = environment.state[0].observation.private
                wheat = int(private.shed.get("WHEAT", 0))
                farmer_wheat = int(private.inventories[0].get("WHEAT", 0))
                hand_fertilizer = int(private.inventories[4].get("FERTILIZER", 0))
                if (wheat, farmer_wheat, hand_fertilizer) != (0, 1, 2):
                    raise AssertionError(
                        f"official partial pickup setup diverged in game {game}: "
                        f"shed_wheat={wheat} farmer_wheat={farmer_wheat} "
                        f"hand_fertilizer={hand_fertilizer}"
                    )
            if submitted_units is not None and transition in (6, 7):
                private = environment.state[0].observation.private
                remaining = int(private.inventories[4].get("FERTILIZER", 0))
                expected_remaining = 1 if transition == 6 else 0
                if remaining != expected_remaining:
                    raise AssertionError(
                        f"official repeated fertilize setup diverged in game {game}: "
                        f"expected={expected_remaining} actual={remaining}"
                    )
            if environment.done:
                utility = terminal_pair_utility(
                    environment.state[0].observation,
                    environment.state[1].observation,
                )
                next_potential = 0.0
                reward = shaped_pair_reward(
                    potentials[game],
                    None,
                    terminal_utility=utility,
                    gamma=DEFAULT_REWARD_GAMMA,
                )
                expected_utilities[game] = utility
            else:
                next_potential = pair_potential(
                    environment.state[0].observation,
                    environment.state[1].observation,
                )
                reward = shaped_pair_reward(
                    potentials[game],
                    next_potential,
                    gamma=DEFAULT_REWARD_GAMMA,
                )
            expected_potentials[game] = next_potential
            expected_rewards[game] = reward

        if submitted_units is None:
            native_step = native.step_factors(unit_actions, market_kinds, quantities)
        else:
            native_step = native.step_submitted(submitted_units, market_kinds, quantities)
        np.testing.assert_array_equal(
            native_step["previous_potentials"],
            potentials,
            err_msg=f"previous potential divergence at transition {transition + 1}",
        )
        np.testing.assert_array_equal(
            native_step["potentials"],
            expected_potentials,
            err_msg=f"potential divergence at transition {transition + 1}",
        )
        np.testing.assert_array_equal(
            native_step["terminal_utilities"],
            expected_utilities,
            err_msg=f"terminal utility divergence at transition {transition + 1}",
        )
        np.testing.assert_array_equal(
            _native_pair_rewards(native_step, DEFAULT_REWARD_GAMMA),
            expected_rewards,
            err_msg=f"reward divergence at transition {transition + 1}",
        )
        potentials = expected_potentials

    return {
        "games": games,
        "mode": mode,
        "transitions_per_game": steps,
        "joint_transitions": games * steps,
        "submitted_cases_exercised": exercise_submitted_cases and steps >= 8,
        "result": "exact state/conv+structured encoding/potential/utility/reward parity",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--steps", type=int, default=MAX_TRANSITIONS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--factor-seed", type=int, default=20260812)
    parser.add_argument("--mode", choices=("pass", "random"), default="random")
    parser.add_argument("--debug-build", action="store_true")
    parser.add_argument("--build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = assert_official_native_parity(
        games=args.games,
        steps=args.steps,
        seed_start=args.seed_start,
        factor_seed=args.factor_seed,
        mode=args.mode,
        build=args.build,
        release=not args.debug_build,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
