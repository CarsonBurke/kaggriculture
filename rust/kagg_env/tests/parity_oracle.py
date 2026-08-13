"""Persistent native/Python differential gate.

Run from the repository root after a release build:
  PYTHONPATH=src .venv/bin/python \
    rust/kagg_env/tests/parity_oracle.py --games 8 --steps 719
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from kaggle_environments import make

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    compile_action,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.encoding import encode_observation, pair_potential
from kaggriculture.rust_env import load_native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    seeds = np.arange(args.games, dtype=np.uint64)
    official = [
        make("kaggriculture", configuration={"seed": int(seed)}, debug=False) for seed in seeds
    ]
    for environment in official:
        environment.reset(2)
    native = load_native(release=True).BatchEnv(seeds)
    encoded_buffers = native.encoded_buffers()
    rng = np.random.default_rng(args.seed)

    for transition in range(args.steps + 1):
        native.encoded_into(encoded_buffers)
        encoded = encoded_buffers
        oracle = []
        potentials = []
        for environment in official:
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
            potentials.append(
                pair_potential(
                    environment.state[0].observation,
                    environment.state[1].observation,
                )
            )
        for name in (
            "board",
            "global_features",
            "critic_features",
            "units",
            "unit_positions",
            "unit_active",
        ):
            expected = np.stack([getattr(row, name) for row in oracle])
            if not np.array_equal(expected, encoded[name]):
                difference = np.abs(expected.astype(np.float64) - encoded[name].astype(np.float64))
                index = np.unravel_index(np.argmax(difference), difference.shape)
                raise AssertionError(
                    f"encoding divergence t={transition} {name}{index}: "
                    f"python={expected[index]} rust={encoded[name][index]}"
                )
        np.testing.assert_allclose(encoded["potentials"], potentials, atol=1e-7, rtol=0)
        if transition == args.steps:
            break
        unit = rng.integers(
            N_UNIT_ACTIONS,
            size=(args.games, 2, MAX_UNITS),
            dtype=np.uint8,
        )
        kinds = rng.integers(
            N_MARKET_KINDS,
            size=(args.games, 2, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        )
        quantities = rng.integers(
            N_QUANTITIES,
            size=(args.games, 2, MAX_MARKET_ORDERS),
            dtype=np.uint8,
        )
        for game, environment in enumerate(official):
            environment.step(
                [
                    compile_action(
                        environment.state[player].observation,
                        unit[game, player],
                        kinds[game, player],
                        quantities[game, player],
                    )
                    for player in range(2)
                ]
            )
        native.step_factors(unit, kinds, quantities)
        for game, environment in enumerate(official):
            python_state = json.loads(
                json.dumps(
                    {
                        "farms": environment.state[0].observation.farms,
                        "privates": [state.observation.private for state in environment.state],
                        "market": environment.state[0].observation.market,
                        "town": environment.state[0].observation.town,
                    }
                )
            )
            rust_state = json.loads(native.snapshot_json(game))
            for key, value in python_state.items():
                if rust_state[key] != value:
                    raise AssertionError(
                        f"state divergence t={transition + 1} game={game} key={key}"
                    )
    print(f"exact state/encoding/potential parity: {args.games} games x {args.steps} transitions")


if __name__ == "__main__":
    main()
