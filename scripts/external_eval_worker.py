#!/usr/bin/env python3
"""Append diagnostic external-opponent evaluations of one provenance-bound actor.

The training loop launches this worker after committing a full checkpoint,
so the learner's absolute strength against public reference
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
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kaggriculture.evaluation import (
    DEVELOPMENT_SEED_START,
    SCORE_CONFIDENCE,
    artifact_seed_usage,
    bounded_mean_interval,
    seed_protocol,
)
from kaggriculture.inference import (
    actor_artifact_from_checkpoint,
    checkpoint_orientation,
    cpu_portable_actor,
    load_actor_artifact,
)
from kaggriculture.league import snapshot_sha256
from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.orientation import Orientation
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
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help=(
            "provenance-bound BC/exported actor or durable training checkpoint; "
            "--agents selects population members. Bare league snapshots are not evidence"
        ),
    )
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL journal to append to")
    parser.add_argument(
        "--agents",
        default="",
        help=(
            "comma-separated member indices to evaluate from a population checkpoint; "
            "empty evaluates the artifact's single actor. Members run sequentially in "
            "one process, because N concurrent CPU workers would contend with each "
            "other and with training for cores this probe is not entitled to"
        ),
    )
    parser.add_argument(
        "--opponents",
        default="starter,public-v27",
        help="comma-separated built-in names, v27 aliases, or agent file paths",
    )
    parser.add_argument("--seeds", type=int, default=2, help="seed pairs per opponent")
    parser.add_argument("--seed-start", type=int, default=DEVELOPMENT_SEED_START)
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
    artifact_name: str,
    artifact_digest: str,
    member: int | None,
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
        "artifact": artifact_name,
        "artifact_sha256": artifact_digest,
        # Which population member this row measures. Present and null for a
        # single-learner run, so a reader never has to guess whether a missing
        # key means "one learner" or "an older worker that did not record it".
        "agent": member,
        "opponent": label,
        # The runnable, not the label, decides identity: a file opponent named
        # like a built-in still gets its digest recorded.
        "opponent_sha256": None if runnable in BUILTIN_OPPONENTS else file_sha256(Path(runnable)),
        "deterministic": True,
        "episode_steps": episode_steps,
        "seed_start": seeds.start,
        "seed_count": len(seeds),
        "seed_clusters": len(seeds) if len(completed) == len(outcomes) else 0,
        "score_confidence": SCORE_CONFIDENCE,
        "score_rate_95ci": (
            list(
                bounded_mean_interval(
                    [
                        (outcomes[index].score + outcomes[index + 1].score) / 2
                        for index in range(0, len(outcomes), 2)
                    ]
                )
            )
            if len(completed) == len(outcomes)
            else None
        ),
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
    """Atomically append and durably flush one JSONL journal record.

    A crash-and-resume can replay a probe while an orphaned worker is still
    exiting. ``O_APPEND`` plus one ``write`` keeps each row indivisible across
    those processes; fsync makes the terminal completion row a durable
    acknowledgement of all rows written before it.
    """
    rendered = json.dumps(record, sort_keys=True, allow_nan=False)
    payload = (rendered + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short external-eval journal write: {written} of {len(payload)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return rendered


def completion_record(
    *,
    iteration: int,
    artifact_name: str,
    artifact_digest: str,
    members: Sequence[int | None],
    opponents: Sequence[str],
) -> dict[str, Any]:
    """Describe the exact Cartesian probe set committed by this worker."""
    return {
        "event": "external_eval_complete",
        "iteration": iteration,
        "artifact": artifact_name,
        "artifact_sha256": artifact_digest,
        "members": list(members),
        "opponents": list(opponents),
        "records": len(members) * len(opponents),
    }


def _members(spec: str) -> list[int | None]:
    """Member indices to evaluate, or a single unnamed actor.

    An empty spec is a single-learner run reading a league snapshot; anything
    else names members inside one population checkpoint. Duplicates would write
    two rows a reader must then deduplicate on a key it cannot distinguish, so
    they are refused rather than tolerated.
    """
    if not spec.strip():
        return [None]
    members = [int(part) for part in spec.split(",") if part.strip()]
    if not members:
        raise ValueError("--agents was given but named no member")
    if len(set(members)) != len(members):
        raise ValueError(f"--agents names a member twice: {spec}")
    if any(member < 0 for member in members):
        raise ValueError(f"--agents names a negative member: {spec}")
    return list(members)


def _load_member(artifact: Path, member: int | None) -> tuple[Any, Orientation]:
    """Load a provenance-bound actor in its recorded board orientation.

    Population checkpoints require a member index. Bare league snapshots carry
    no training-map exposure and are training opponents, not evaluation inputs.
    """
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"actor file is not a dictionary: {artifact}")
    artifact_seed_usage(payload)
    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping) and model_config.get("fused_mlp", False):
        orientation = checkpoint_orientation(payload, agent=member)
        artifact_payload = actor_artifact_from_checkpoint(payload, agent=member)
        return cpu_portable_actor(artifact_payload), orientation
    if member is not None:
        actor, loaded = load_actor_artifact(artifact, agent=member)
        return actor, checkpoint_orientation(loaded, agent=member)
    actor, _ = load_actor_artifact(artifact)
    return actor, checkpoint_orientation(payload)


def _agent_for(actor: Any, orientation: Orientation) -> Any:
    """A single-argument agent callable bound to one actor and its orientation.

    The arity is load-bearing and is why this is a factory rather than a closure
    written inline. `kaggle_environments` sizes the call with
    `getfullargspec`, which counts EVERY parameter including one carrying a
    default, and then invokes a two-parameter callable as
    `agent(observation, configuration)`. A loop-local closure that bound its
    actor as a default argument would therefore have the configuration dict
    passed in as the actor; the resulting TypeError is swallowed under
    `debug=False`, the seat submits nothing for all 720 steps, and the episode
    still reports DONE with the bank at exactly its 3000 starting money. That is
    indistinguishable from a policy that chose to do nothing, which is how the
    fault survives a green-looking journal. Measured: the bug produced
    `money_mean` 3000.0 and `score_rate` 0.0 against starter, public-v27 and
    public-v16 alike.

    A factory closes over its own scope, so the loop cannot rebind the actor
    behind the callable's back either -- the late-binding hazard the default
    argument was reaching for -- while the signature stays exactly one argument.
    """

    def agent(observation: dict[str, Any]) -> dict[str, Any]:
        return act_batch(actor, [observation], deterministic=True, orientation=orientation).actions[
            0
        ]

    return agent


def main() -> None:
    args = parse_args()
    if args.seeds < 1 or args.episode_steps < 2:
        raise ValueError("external evaluation needs at least one seed and two episode steps")
    torch.set_num_threads(max(1, args.torch_threads))
    opponents = [normalize_opponent(spec) for spec in args.opponents.split(",") if spec]
    if not opponents:
        raise ValueError("at least one opponent is required")
    labels = [label for label, _runnable in opponents]
    if len(set(labels)) != len(labels):
        raise ValueError("external evaluation opponents must have distinct journal labels")
    members = _members(args.agents)
    digest = snapshot_sha256(args.artifact)
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False, mmap=True)
    protocol = seed_protocol(
        "development",
        args.seed_start,
        args.seeds,
        usage=artifact_seed_usage(payload),
    )

    for member in members:
        actor, orientation = _load_member(args.artifact, member)
        actor = actor.eval()
        agent = _agent_for(actor, orientation)

        for label, runnable in opponents:
            record = evaluate_opponent(
                agent,
                label,
                runnable,
                iteration=args.iteration,
                artifact_name=args.artifact.name,
                artifact_digest=digest,
                member=member,
                seeds=range(args.seed_start, args.seed_start + args.seeds),
                episode_steps=args.episode_steps,
            )
            record["seed_protocol"] = protocol
            print(append_record(args.output, record), flush=True)
            if record["completed_games"] != record["games"]:
                raise RuntimeError(
                    f"external evaluation incomplete for member {member}, opponent {label}: "
                    f"{record['completed_games']} of {record['games']} games completed"
                )

    completed = completion_record(
        iteration=args.iteration,
        artifact_name=args.artifact.name,
        artifact_digest=digest,
        members=members,
        opponents=labels,
    )
    print(append_record(args.output, completed), flush=True)


if __name__ == "__main__":
    main()
