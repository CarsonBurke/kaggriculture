#!/usr/bin/env python3
"""Append diagnostic external-opponent evaluations of one league snapshot.

The training loop launches this worker on CPU after freezing a league
snapshot, so the learner's absolute strength against public reference
agents lands in the run's journal without stalling the GPU. Records are
diagnostics for steering training; they are never selection or calibration
evidence — finalist evaluation stays with evaluate_checkpoint.py and its
source-identity binding.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kaggriculture.league import load_actor_snapshot, snapshot_sha256
from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.policy import act_batch
from kaggriculture.provenance import file_sha256

_CAPTURE_LIMIT = 2_000


@dataclass(frozen=True)
class GameOutcome:
    seed: int
    snapshot_seat: int
    snapshot_money: float | None
    opponent_money: float | None
    error: str | None

    @property
    def complete(self) -> bool:
        return self.error is None

    @property
    def score(self) -> float:
        assert self.snapshot_money is not None and self.opponent_money is not None
        if self.snapshot_money > self.opponent_money:
            return 1.0
        if self.snapshot_money < self.opponent_money:
            return 0.0
        return 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="immutable league snapshot")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL journal to append to")
    parser.add_argument(
        "--opponents",
        default="starter,public-v27",
        help="comma-separated built-in names, v27 aliases, or agent file paths",
    )
    parser.add_argument("--seeds", type=int, default=2, help="seed pairs per opponent")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _play_game(
    agent, opponent: str, seed: int, snapshot_seat: int, episode_steps: int
) -> GameOutcome:
    from kaggle_environments import make

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            players: list[Any] = [opponent, opponent]
            players[snapshot_seat] = agent
            environment = make(
                "kaggriculture",
                configuration={"episodeSteps": episode_steps, "seed": seed},
                debug=False,
            )
            environment.run(players)
        final = environment.steps[-1]
        snapshot_money = _finite(final[snapshot_seat].reward)
        opponent_money = _finite(final[1 - snapshot_seat].reward)
        issues = []
        if not environment.done:
            issues.append("environment did not reach DONE")
        if snapshot_money is None or opponent_money is None:
            issues.append("a final reward is missing or non-finite")
        for seat in (0, 1):
            status = str(final[seat].status)
            if status != "DONE":
                issues.append(f"seat {seat} status is {status}")
        return GameOutcome(
            seed=seed,
            snapshot_seat=snapshot_seat,
            snapshot_money=snapshot_money,
            opponent_money=opponent_money,
            error="; ".join(issues) if issues else None,
        )
    except Exception as exc:  # external agents are outside our trust boundary
        detail = f"{type(exc).__name__}: {exc}"
        captured = (stdout.getvalue() + stderr.getvalue())[:_CAPTURE_LIMIT]
        if captured:
            detail = f"{detail}; captured: {captured}"
        return GameOutcome(
            seed=seed,
            snapshot_seat=snapshot_seat,
            snapshot_money=None,
            opponent_money=None,
            error=detail,
        )


def evaluate_opponent(
    agent,
    label: str,
    runnable: str,
    *,
    iteration: int,
    snapshot_name: str,
    snapshot_digest: str,
    seeds: range,
    episode_steps: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    outcomes = [
        _play_game(agent, runnable, seed, seat, episode_steps) for seed in seeds for seat in (0, 1)
    ]
    completed = [outcome for outcome in outcomes if outcome.complete]
    record: dict[str, Any] = {
        "event": "external_eval",
        "iteration": iteration,
        "snapshot": snapshot_name,
        "snapshot_sha256": snapshot_digest,
        "opponent": label,
        # The runnable, not the label, decides identity: a file opponent named
        # like a built-in still gets its digest recorded.
        "opponent_sha256": None if runnable in BUILTIN_OPPONENTS else file_sha256(Path(runnable)),
        "deterministic": True,
        "episode_steps": episode_steps,
        "seed_start": seeds.start,
        "games": len(outcomes),
        "completed_games": len(completed),
        "money_mean": (
            sum(outcome.snapshot_money for outcome in completed) / len(completed)
            if completed
            else None
        ),
        "opponent_money_mean": (
            sum(outcome.opponent_money for outcome in completed) / len(completed)
            if completed
            else None
        ),
        "score_rate": (
            sum(outcome.score for outcome in completed) / len(completed) if completed else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    errors = sorted({outcome.error for outcome in outcomes if outcome.error})
    if errors:
        record["errors"] = errors
    return record


def append_record(output: Path, record: dict[str, Any]) -> str:
    """Append one JSONL line to the diagnostics journal.

    The launcher runs at most one worker per training process, but a
    crash-and-resume can replay a probe and an orphaned worker from a killed
    run may still be appending. Readers must deduplicate last-wins on
    ``(iteration, opponent)`` and tolerate a torn trailing line.
    """
    rendered = json.dumps(record, sort_keys=True, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(rendered + "\n")
    return rendered


def main() -> None:
    args = parse_args()
    if args.seeds < 1 or args.episode_steps < 2:
        raise ValueError("external evaluation needs at least one seed and two episode steps")
    torch.set_num_threads(max(1, args.torch_threads))
    opponents = [normalize_opponent(spec) for spec in args.opponents.split(",") if spec]
    if not opponents:
        raise ValueError("at least one opponent is required")
    actor = load_actor_snapshot(args.snapshot)
    actor.eval()
    digest = snapshot_sha256(args.snapshot)

    def agent(observation: dict[str, Any]) -> dict[str, Any]:
        return act_batch(actor, [observation], deterministic=True).actions[0]

    for label, runnable in opponents:
        record = evaluate_opponent(
            agent,
            label,
            runnable,
            iteration=args.iteration,
            snapshot_name=args.snapshot.name,
            snapshot_digest=digest,
            seeds=range(args.seed_start, args.seed_start + args.seeds),
            episode_steps=args.episode_steps,
        )
        print(append_record(args.output, record), flush=True)


if __name__ == "__main__":
    main()
