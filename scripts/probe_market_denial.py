#!/usr/bin/env python
"""Measure whether a collapsed policy wins by destroying the shared economy.

`_relative_score(a, b) = (a - b) / (a + b)` is scale invariant, so a mirror
game scores 0.0 whether both banks hold 150,000 or 3,000, and it saturates
against a fixed opponent, so 57,000 against `starter` scores about as well as
143,000 does. Neither property tells a policy to stay rich. What that leaves
open is the direction of causation in a league game: the collapsed policy's
league score rate *rose* from 0.302 to 0.578 while league money fell 4.5x, and
only a head-to-head money matrix can say whether it learned to win by growing
its own margin or by pulling a market-dependent opponent down with it.

This runs every ordered pair of a set of actors and reports both banks, so a
diagonal (mirror) cell gives each actor's wealth against an equal and the
off-diagonal cells show what each does to the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kaggriculture.inference import load_actor_artifact
from kaggriculture.league import load_actor_snapshot
from kaggriculture.production import PRODUCTION_EPISODE_STEPS
from kaggriculture.registry import architecture_of
from kaggriculture.rollout import collect_mixed_play_rust
from kaggriculture.training import AnyActor

#: Kept at the production value: the reward telescopes to the terminal relative
#: bank score only over a full episode, so a shortened one measures a different
#: quantity than the run optimized.
_STEPS = PRODUCTION_EPISODE_STEPS


def _load(path: Path, device: torch.device) -> AnyActor:
    """Load a league snapshot or an exported actor artifact.

    Both file shapes are in play here: the league writes snapshots under
    `league/`, while a BC run exports the artifact shape inference reads. The
    two overlap on `model_config` and `actor`, so the snapshot loader's own
    strict key set is what distinguishes them -- a BC artifact additionally
    carries provenance and metrics, which that loader rejects outright.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if {"format_version", "iteration", "model_config", "actor"} <= set(payload) and not (
        set(payload) - {"format_version", "iteration", "model_config", "actor", "architecture"}
    ):
        return load_actor_snapshot(path, device=device)
    actor, _ = load_actor_artifact(path, device=device)
    return actor.eval().requires_grad_(False)


def _match(
    learner: AnyActor,
    opponent: AnyActor | str,
    *,
    games: int,
    seed: int,
    temperature: float,
    opponent_temperature: float,
) -> dict[str, Any]:
    """Play `games` league games and return both banks.

    League rows carry one learner seat each, so `final_money` is the learner's
    bank and `opponent_money` the frozen seat's. Seats alternate inside the
    wave, which matters here because the two seats do not face the same market.

    A string opponent names a native built-in, which occupies a reserved lane
    rather than a frozen-actor slot. Those are the denial-immune opponents: a
    `pass` seat holds its starting cash whatever the market does, so it is the
    control that says whether a policy is winning by earning or by destroying.
    """
    builtin = isinstance(opponent, str)
    rollout = collect_mixed_play_rust(
        learner,
        () if builtin else (opponent,),
        self_play_games=0,
        league_games=games,
        opponent_indices=None if builtin else np.zeros(games, dtype=np.int64),
        builtin_lanes=(opponent,) if builtin else (),
        seed_start=seed,
        episode_steps=_STEPS,
        temperature=temperature,
        opponent_temperature=opponent_temperature,
        sampling_seed=seed,
        forward_mode="eager",
        forward_autocast=False,
    )
    own = np.asarray(rollout.final_money, dtype=np.float64)
    other = np.asarray(rollout.opponent_money, dtype=np.float64)
    total = own + other
    relative = np.where(total == 0.0, 0.0, (own - other) / np.maximum(total, 1e-9))
    return {
        "games": int(own.size),
        "own_money_mean": float(own.mean()),
        "own_money_median": float(np.median(own)),
        "opponent_money_mean": float(other.mean()),
        "opponent_money_median": float(np.median(other)),
        "relative_score_mean": float(relative.mean()),
        "win_rate": float((own > other).mean()),
        # Per game, so a candidate objective can be scored on these banks
        # afterwards without replaying: comparing objectives on aggregates
        # would average the wrong way round for any nonlinear one.
        "own_money": own.tolist(),
        "opponent_money": other.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", action="append", required=True, help="name=path")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=90_001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opponent-temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--builtin",
        action="append",
        default=[],
        help="native built-in to face as an extra opponent column",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    actors: dict[str, AnyActor] = {}
    architectures: set[str] = set()
    for spec in args.actor:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got {spec!r}")
        actor = _load(Path(path), device)
        actors[name] = actor
        architectures.add(architecture_of(actor).name)
    if len(architectures) > 1:
        raise SystemExit(f"actors span several architectures: {sorted(architectures)}")

    rows: list[dict[str, Any]] = []
    opponents: dict[str, AnyActor | str] = {**actors, **{name: name for name in args.builtin}}
    for learner_name, learner in actors.items():
        for opponent_name, opponent in opponents.items():
            row: dict[str, Any] = {"learner": learner_name, "opponent": opponent_name}
            row.update(
                _match(
                    learner,
                    opponent,
                    games=args.games,
                    seed=args.seed,
                    temperature=args.temperature,
                    opponent_temperature=args.opponent_temperature,
                )
            )
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if not k.endswith("_money")}), flush=True
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "probe": "market-denial",
                "command": sys.argv,
                "episode_steps": _STEPS,
                "matches": rows,
            },
            indent=1,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
