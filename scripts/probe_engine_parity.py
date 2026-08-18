#!/usr/bin/env python3
"""Localize where the native engine's state departs from `kaggle_environments`.

The scripted-v27 port emits actions identical to the real agent file given the
same observation, yet a native episode and an official episode on the same seed
end with different banks. That isolates the difference to the engines rather
than the agent, and this probe finds the first state field where they part.

Both engines are driven by the *same* action dicts, computed once from the
native state and submitted to each engine. Letting each engine ask the agent
about its own state would confound an engine difference with the agent's
state-dependent weed repair, which is exactly the ambiguity this probe exists
to remove. The first divergence is reported as a field path, because the money
gap at the end of an episode says nothing about which mechanic caused it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from audit_v27_parity import _reference_factors
from kaggle_environments import make

from kaggriculture.opponents import PUBLIC_V27_OPPONENT
from kaggriculture.rust_env import load_native

PLAYERS = 2
EPISODE_STEPS = 720
#: Absolute dollars below which a float difference is the two engines' own
#: accumulation order rather than a mechanic disagreeing.
MONEY_TOLERANCE = 1e-6


def _load_agent(path: Path) -> Any:
    """Import the agent file into a private module object."""
    spec = importlib.util.spec_from_file_location("kaggriculture_public_v27_probe", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the public v27 agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(snapshot: dict[str, Any], player: int) -> dict[str, Any]:
    """Shape the native state into the observation the real agent reads."""
    return {
        "player": player,
        "step": snapshot["step"],
        "day": snapshot["day"],
        "hour": snapshot["hour"],
        "farms": snapshot["farms"],
        "private": snapshot["privates"][player],
        "market": snapshot["market"],
        "town": snapshot["town"],
    }


def _differences(left: Any, right: Any, path: str, out: list[str], limit: int) -> None:
    """Collect field paths where two nested engine states disagree."""
    if len(out) >= limit:
        return
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                # A key one engine omits and the other carries as zero is a
                # serialization difference, not a mechanic: compare against the
                # absent side's implied zero rather than reporting a mismatch.
                _differences(left.get(key, 0), right.get(key, 0), f"{path}.{key}", out, limit)
                continue
            _differences(left[key], right[key], f"{path}.{key}", out, limit)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(f"{path}: length {len(left)} vs {len(right)}")
            return
        for index, (one, other) in enumerate(zip(left, right, strict=True)):
            _differences(one, other, f"{path}[{index}]", out, limit)
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if bool(left) != bool(right):
            out.append(f"{path}: {left} vs {right}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if abs(float(left) - float(right)) > MONEY_TOLERANCE:
            out.append(f"{path}: {left} vs {right}")
        return
    if left != right:
        out.append(f"{path}: {left!r} vs {right!r}")


def _native_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The comparable part of a native snapshot."""
    return {
        "farms": snapshot["farms"],
        "privates": snapshot["privates"],
        "market": snapshot["market"],
        "town": snapshot["town"],
        "day": snapshot["day"],
        "hour": snapshot["hour"],
    }


def _official_state(environment: Any) -> dict[str, Any]:
    """The same fields, read out of the official engine's own state."""
    shared = environment.state[0].observation
    return {
        "farms": json.loads(json.dumps(shared["farms"])),
        "privates": [
            json.loads(json.dumps(environment.state[player].observation["private"]))
            for player in range(PLAYERS)
        ],
        "market": json.loads(json.dumps(shared["market"])),
        "town": json.loads(json.dumps(shared["town"])),
        "day": shared["day"],
        "hour": shared["hour"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=PUBLIC_V27_OPPONENT)
    parser.add_argument("--seed", type=int, default=90001)
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS - 1)
    parser.add_argument(
        "--max-differences",
        type=int,
        default=12,
        help="field paths to report at the first divergent step",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent_path = args.agent.expanduser().resolve()
    if not agent_path.is_file():
        raise SystemExit(f"public v27 agent is unavailable: {agent_path}")
    module = load_native()
    agent = _load_agent(agent_path)

    native = module.BatchEnv(np.asarray([args.seed], dtype=np.uint64))
    official = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(args.seed)},
        debug=True,
    )
    official.reset(PLAYERS)

    first: dict[str, Any] | None = None
    for step in range(args.steps):
        snapshot = json.loads(native.snapshot_json(0, False))
        differences: list[str] = []
        _differences(
            _native_state(snapshot),
            _official_state(official),
            "state",
            differences,
            args.max_differences,
        )
        if differences and first is None:
            first = {"step": step, "differences": differences}
            break
        # Both engines receive the identical action, encoded from the one dict the
        # reference returned for the state they still agree on. Stepping native
        # from its own port instead would let a port disagreement be reported as
        # an engine bug, which is the confusion this probe exists to remove.
        submitted = [
            agent.agent(_observation(snapshot, player), official.configuration)
            for player in range(PLAYERS)
        ]
        units = np.zeros((1, PLAYERS, module.MAX_UNITS), dtype=np.uint8)
        kinds = np.zeros((1, PLAYERS, module.MAX_MARKET_ORDERS), dtype=np.uint8)
        quantities = np.zeros((1, PLAYERS, module.MAX_MARKET_ORDERS), dtype=np.uint8)
        for player, action in enumerate(submitted):
            active = 1 + len(snapshot["farms"][player].get("hands") or [])
            values, order_kinds, order_quantities = _reference_factors(
                action, active, f"seed {args.seed} step {step} seat {player}"
            )
            units[0, player, :active] = values
            kinds[0, player] = order_kinds
            quantities[0, player] = order_quantities
        # The reference's own dict, so the engine screen owed here is the
        # interpreter's -- our policy mask never produces these rows.
        native.step_factors(units, kinds, quantities, external=True)
        official.step(submitted)

    report = {
        "seed": int(args.seed),
        "steps_checked": args.steps if first is None else first["step"],
        "first_divergence": first,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    if first is None:
        print(f"seed {args.seed}: engines agree on every compared field for {args.steps} steps")
        return 0
    print(f"seed {args.seed}: first divergence at step {first['step']}")
    for line in first["differences"]:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
