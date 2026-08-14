#!/usr/bin/env python3
"""Extract projected demonstration episodes from official public-v27 games.

Runs complete kaggle_environments episodes with the public v27 agent in the
recorded seat(s), projects every recorded engine action through the exact
sequential legality ledger (`kaggriculture.demonstrations`), verifies the
factored round trip against the recorded dict, and stores one compressed
archive per episode-seat: raw observations (retokenizable for any future
architecture), factored targets, teacher-forced masks, and active flags.

Any representability gap or mask divergence aborts extraction with the
offending step — silent clamping would corrupt the dataset. CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.demonstrations import (
    DemonstrationError,
    project_demonstration,
    verify_round_trip,
)
from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.provenance import file_sha256, source_identity

DATASET_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128, help="environment seeds to play")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--teacher", default="public-v27", help="demonstrating agent (spec for opponents registry)"
    )
    parser.add_argument(
        "--opponent",
        default="public-v27",
        help="other seat; when it equals the teacher, both seats are recorded",
    )
    parser.add_argument("--episode-steps", type=int, default=720)
    return parser.parse_args()


def _seat_observation(steps: list[Any], step_index: int, seat: int) -> dict[str, Any]:
    observation = dict(steps[step_index][seat].observation)
    # The engine strips `step` from the non-first seat's stored schema; it is
    # positional, so restore it from the walk index.
    observation.setdefault("step", step_index)
    return observation


def extract_episode(
    environment_steps: list[Any],
    seat: int,
    *,
    episode_steps: int,
) -> dict[str, np.ndarray | bytes]:
    """Project one recorded seat of a complete episode into training arrays.

    `steps[t+1][seat].action` is the action applied to `steps[t][seat]`'s
    observation, so a T-step episode yields T-1 demonstration pairs.
    """
    if len(environment_steps) != episode_steps:
        raise DemonstrationError(
            f"episode has {len(environment_steps)} steps; expected {episode_steps}"
        )
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    projections = []
    for step_index in range(episode_steps - 1):
        observation = _seat_observation(environment_steps, step_index, seat)
        action = environment_steps[step_index + 1][seat].action
        if not isinstance(action, dict):
            raise DemonstrationError(f"step {step_index} seat {seat}: no recorded action")
        try:
            projected = project_demonstration(observation, action)
            verify_round_trip(observation, action, projected)
        except DemonstrationError as error:
            raise DemonstrationError(f"step {step_index} seat {seat}: {error}") from error
        opponent_private = environment_steps[step_index][1 - seat].observation.get("private")
        observations.append({"observation": observation, "opponent_private": opponent_private})
        actions.append(action)
        projections.append(projected)

    def stacked(name: str) -> np.ndarray:
        return np.stack([getattr(projection, name) for projection in projections])

    raw = {
        "observations": observations,
        "actions": actions,
    }
    return {
        "unit_actions": stacked("unit_actions"),
        "market_kinds": stacked("market_kinds"),
        "market_quantities": stacked("market_quantities"),
        "unit_masks": stacked("unit_masks"),
        "market_kind_masks": stacked("market_kind_masks"),
        "market_quantity_masks": stacked("market_quantity_masks"),
        "unit_active": stacked("unit_active"),
        "market_active": stacked("market_active"),
        "market_quantity_active": stacked("market_quantity_active"),
        "raw_json_zlib": zlib.compress(
            json.dumps(raw, separators=(",", ":"), allow_nan=False).encode("utf-8"), level=6
        ),
    }


def _play_episode(teacher: str, opponent: str, seed: int, episode_steps: int) -> list[Any]:
    from kaggle_environments import make

    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=False,
    )
    environment.run([teacher, opponent])
    if not environment.done:
        raise RuntimeError(f"seed {seed}: environment did not finish")
    for seat in (0, 1):
        status = str(environment.steps[-1][seat].status)
        if status != "DONE":
            raise RuntimeError(f"seed {seed}: seat {seat} ended with status {status}")
    return environment.steps


def _agent_digest(runnable: str) -> str | None:
    return None if runnable in BUILTIN_OPPONENTS else file_sha256(Path(runnable))


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.episode_steps != 720:
        raise ValueError("extraction needs at least one episode at the competition horizon")
    teacher_label, teacher = normalize_opponent(args.teacher)
    opponent_label, opponent = normalize_opponent(args.opponent)
    mirrored = teacher == opponent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    started = time.perf_counter()
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        steps = _play_episode(teacher, opponent, seed, args.episode_steps)
        final = steps[-1]
        for seat in (0, 1) if mirrored else (0,):
            arrays = extract_episode(steps, seat, episode_steps=args.episode_steps)
            name = f"episode-{seed:08d}-seat{seat}"
            np.savez_compressed(args.output_dir / f"{name}.npz", **arrays)
            episodes.append(
                {
                    "file": f"{name}.npz",
                    "seed": seed,
                    "seat": seat,
                    "steps": args.episode_steps - 1,
                    "teacher_money": float(final[seat].reward),
                    "opponent_money": float(final[1 - seat].reward),
                }
            )
        elapsed = time.perf_counter() - started
        print(
            f"seed {seed}: extracted ({elapsed:.1f}s elapsed, {len(episodes)} episode-seats)",
            flush=True,
        )

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "teacher": {"label": teacher_label, "sha256": _agent_digest(teacher)},
        "opponent": {"label": opponent_label, "sha256": _agent_digest(opponent)},
        "episode_steps": args.episode_steps,
        "seed_start": args.seed_start,
        "episodes": episodes,
        "extractor_source_identity": source_identity(),
        "command": sys.argv,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(episodes)} episode-seats and {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
