#!/usr/bin/env python3
"""Evaluate an actor artifact over paired seats against a fixed opponent."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import statistics
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.inference import CheckpointAgent, load_actor_artifact
from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.provenance import file_sha256, require_source_identity

EPISODE_STEPS = 720
DEFAULT_SEED_CLUSTERS = 32
_CAPTURE_LIMIT = 4_000


@dataclass(frozen=True)
class GameSpec:
    seed: int
    candidate_seat: int
    episode_steps: int = EPISODE_STEPS


@dataclass(frozen=True)
class GameResult:
    seed: int
    candidate_seat: int
    candidate_reward: float | None
    opponent_reward: float | None
    candidate_status: str
    opponent_status: str
    environment_done: bool
    steps: int
    expected_steps: int
    elapsed_seconds: float
    action_seconds: tuple[float, ...] = ()
    error: str | None = None
    captured_output: str | None = None

    @property
    def complete(self) -> bool:
        return (
            self.error is None
            and self.environment_done
            and self.steps == self.expected_steps
            and self.candidate_status == "DONE"
            and self.opponent_status == "DONE"
            and self.candidate_reward is not None
            and self.opponent_reward is not None
            and math.isfinite(self.candidate_reward)
            and math.isfinite(self.opponent_reward)
        )

    @property
    def margin(self) -> float | None:
        if not self.complete:
            return None
        assert self.candidate_reward is not None and self.opponent_reward is not None
        return self.candidate_reward - self.opponent_reward

    @property
    def outcome(self) -> float | None:
        margin = self.margin
        if margin is None:
            return None
        if margin > 0.0:
            return 1.0
        if margin < 0.0:
            return 0.0
        return 0.5


_WORKER_AGENT: CheckpointAgent | None = None
_WORKER_OPPONENT: str | None = None


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _artifact_provenance(path: Path) -> dict[str, Any]:
    """Validate the artifact before spawning workers and record stable identity."""
    actor, metadata = load_actor_artifact(path, device="cpu")
    del actor
    identity = require_source_identity(metadata.get("source_identity"))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "format_version": int(metadata["format_version"]),
        "iteration": int(metadata.get("iteration", 0)),
        "model_config": metadata["model_config"],
        "source_identity": identity,
        "run_provenance": metadata.get("run_provenance"),
    }


def _opponent_provenance(
    label: str,
    opponent: str,
    *,
    logical_path: str | None = None,
) -> dict[str, Any]:
    # The runnable, not the label, decides identity: a file opponent named
    # like a built-in must still record file provenance.
    if opponent in BUILTIN_OPPONENTS:
        return {"kind": "builtin", "name": label}
    path = Path(opponent).resolve()
    return {
        "kind": "python_file",
        "path": str(path) if logical_path is None else logical_path,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _opponent_identity(provenance: object) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("opponent provenance must be an object")
    if provenance.get("kind") == "builtin":
        if set(provenance) != {"kind", "name"} or provenance.get("name") not in BUILTIN_OPPONENTS:
            raise ValueError("built-in opponent provenance is invalid")
        return provenance
    if provenance.get("kind") == "python_file":
        if set(provenance) != {"kind", "path", "sha256", "size_bytes"}:
            raise ValueError("Python opponent provenance is invalid")
        digest = provenance.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(provenance.get("size_bytes")) is not int
            or provenance["size_bytes"] <= 0
        ):
            raise ValueError("Python opponent provenance identity is invalid")
        return {
            "kind": "python_file",
            "sha256": digest,
            "size_bytes": provenance["size_bytes"],
        }
    raise ValueError("opponent provenance kind is invalid")


def _selection_provenance(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    contents = path.read_bytes()
    payload = json.loads(contents.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("valid_for_selection") is not True:
        raise ValueError("selection report is not valid for finalist evaluation")
    if payload.get("best_output_sha256") != artifact["sha256"]:
        raise ValueError("selection report does not bind the finalist checkpoint bytes")
    if payload.get("source_identity") != artifact["source_identity"]:
        raise ValueError("selection report source identity does not match the finalist checkpoint")
    if payload.get("run_provenance") != artifact.get("run_provenance"):
        raise ValueError("selection report run provenance does not match the finalist checkpoint")
    opponents = payload.get("opponent_provenance")
    if not isinstance(opponents, dict) or not opponents:
        raise ValueError("selection report has no opponent provenance")
    normalized_opponents = {
        label: _opponent_identity(provenance) for label, provenance in opponents.items()
    }
    screening_seed_start = payload.get("seed_start")
    screening_seed_count = payload.get("seed_count")
    if (
        type(screening_seed_start) is not int
        or type(screening_seed_count) is not int
        or screening_seed_start < 0
        or screening_seed_count < 1
    ):
        raise ValueError("selection report screening seed range is invalid")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "best_output_sha256": artifact["sha256"],
        "run_provenance": artifact.get("run_provenance"),
        "opponent_provenance": normalized_opponents,
        "screening_seed_start": screening_seed_start,
        "screening_seed_count": screening_seed_count,
    }


def _initialize_worker(
    artifact: str,
    device: str,
    torch_threads: int,
    opponent: str,
) -> None:
    """Load one persistent candidate model per process, never per action or game."""
    global _WORKER_AGENT, _WORKER_OPPONENT
    _WORKER_AGENT = CheckpointAgent(
        Path(artifact),
        device=torch.device(device),
        torch_threads=torch_threads,
    )
    _WORKER_OPPONENT = opponent


def _make_environment(seed: int, episode_steps: int):
    from kaggle_environments import make

    return make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=False,
    )


def _captured_text(stdout: io.StringIO, stderr: io.StringIO) -> str | None:
    chunks = []
    if stdout.getvalue():
        chunks.append(f"stdout:\n{stdout.getvalue()}")
    if stderr.getvalue():
        chunks.append(f"stderr:\n{stderr.getvalue()}")
    if not chunks:
        return None
    text = "\n".join(chunks)
    if len(text) > _CAPTURE_LIMIT:
        return text[:_CAPTURE_LIMIT] + "\n...[truncated]"
    return text


def _reward(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _run_game(spec: GameSpec) -> GameResult:
    if _WORKER_AGENT is None or _WORKER_OPPONENT is None:
        raise RuntimeError("evaluation worker was not initialized")
    started = time.perf_counter()
    action_seconds: list[float] = []
    stdout = io.StringIO()
    stderr = io.StringIO()

    def timed_agent(observation):
        action_started = time.perf_counter()
        try:
            return _WORKER_AGENT(observation)
        finally:
            action_seconds.append(time.perf_counter() - action_started)

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            players: list[Any] = [_WORKER_OPPONENT, _WORKER_OPPONENT]
            players[spec.candidate_seat] = timed_agent
            environment = _make_environment(spec.seed, spec.episode_steps)
            environment.run(players)
        steps = len(environment.steps)
        final = environment.steps[-1]
        candidate = final[spec.candidate_seat]
        opponent = final[1 - spec.candidate_seat]
        candidate_reward = _reward(candidate.reward)
        opponent_reward = _reward(opponent.reward)
        candidate_status = str(candidate.status)
        opponent_status = str(opponent.status)
        done = bool(environment.done)
        issues = []
        if not done:
            issues.append("environment did not reach DONE")
        if steps != spec.episode_steps:
            issues.append(f"environment produced {steps} steps, expected {spec.episode_steps}")
        if candidate_status != "DONE":
            issues.append(f"candidate status is {candidate_status}")
        if opponent_status != "DONE":
            issues.append(f"opponent status is {opponent_status}")
        if candidate_reward is None:
            issues.append("candidate reward is missing or non-finite")
        if opponent_reward is None:
            issues.append("opponent reward is missing or non-finite")
        return GameResult(
            seed=spec.seed,
            candidate_seat=spec.candidate_seat,
            candidate_reward=candidate_reward,
            opponent_reward=opponent_reward,
            candidate_status=candidate_status,
            opponent_status=opponent_status,
            environment_done=done,
            steps=steps,
            expected_steps=spec.episode_steps,
            elapsed_seconds=time.perf_counter() - started,
            action_seconds=tuple(action_seconds),
            error="; ".join(issues) if issues else None,
            captured_output=_captured_text(stdout, stderr),
        )
    except Exception as exc:  # external agents are intentionally outside our trust boundary
        return GameResult(
            seed=spec.seed,
            candidate_seat=spec.candidate_seat,
            candidate_reward=None,
            opponent_reward=None,
            candidate_status="ERROR",
            opponent_status="UNKNOWN",
            environment_done=False,
            steps=0,
            expected_steps=spec.episode_steps,
            elapsed_seconds=time.perf_counter() - started,
            action_seconds=tuple(action_seconds),
            error=f"{type(exc).__name__}: {exc}",
            captured_output=_captured_text(stdout, stderr),
        )


def _run_seed_pair(seed: int) -> tuple[GameResult, GameResult]:
    """Keep both seats of a confidence-interval cluster in one worker task."""
    return (_run_game(GameSpec(seed, 0)), _run_game(GameSpec(seed, 1)))


def _mean_confidence_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def _timing_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def summarize(results: list[GameResult], seed_count: int) -> dict[str, Any]:
    requested_games = seed_count * 2
    completed = [result for result in results if result.complete]
    clustered: dict[int, list[GameResult]] = {}
    for result in completed:
        clustered.setdefault(result.seed, []).append(result)
    complete_pairs = [
        sorted(rows, key=lambda row: row.candidate_seat)
        for rows in clustered.values()
        if len(rows) == 2 and {row.candidate_seat for row in rows} == {0, 1}
    ]
    valid_for_selection = (
        len(results) == requested_games
        and len(completed) == requested_games
        and len(complete_pairs) == seed_count
    )
    action_seconds = [value for result in results for value in result.action_seconds]
    action_timing = _timing_summary(action_seconds)
    summary: dict[str, Any] = {
        "valid_for_selection": valid_for_selection,
        # Retain the original evaluator's field while making requested versus
        # completed games explicit for hardened selection consumers.
        "games": requested_games,
        "games_requested": requested_games,
        "games_returned": len(results),
        "games_completed": len(completed),
        "invalid_games": requested_games - len(completed),
        "seed_clusters_requested": seed_count,
        "complete_seat_pairs": len(complete_pairs),
        "incomplete_seed_clusters": seed_count - len(complete_pairs),
        "wall_seconds_sum": sum(result.elapsed_seconds for result in results),
        "candidate_action_seconds": action_timing,
        "action_seconds_mean": action_timing["mean"],
        "action_seconds_p99": action_timing["p99"],
        "action_seconds_max": action_timing["max"],
    }
    if not valid_for_selection:
        summary.update(
            {
                "wins": None,
                "ties": None,
                "losses": None,
                "score_rate": None,
                "score_rate_95ci": None,
                "mean_reward": None,
                "median_reward": None,
                "mean_margin": None,
                "median_margin": None,
                "margin_95ci": None,
                "seats": None,
                "seed_clusters": None,
                "seed_cluster_statistics": None,
            }
        )
        return summary

    outcomes = [float(result.outcome) for result in completed]
    margins = [float(result.margin) for result in completed]
    rewards = [float(result.candidate_reward) for result in completed]
    seed_outcomes = [
        statistics.fmean(float(row.outcome) for row in rows) for rows in complete_pairs
    ]
    seed_margins = [statistics.fmean(float(row.margin) for row in rows) for rows in complete_pairs]
    score_ci = _mean_confidence_interval(seed_outcomes)
    margin_ci = _mean_confidence_interval(seed_margins)
    seat_summaries = {}
    for seat in (0, 1):
        seat_rows = [result for result in completed if result.candidate_seat == seat]
        seat_summaries[str(seat)] = {
            "games": len(seat_rows),
            "score_rate": statistics.fmean(float(result.outcome) for result in seat_rows),
            "mean_margin": statistics.fmean(float(result.margin) for result in seat_rows),
        }
    summary.update(
        {
            "wins": sum(outcome == 1.0 for outcome in outcomes),
            "ties": sum(outcome == 0.5 for outcome in outcomes),
            "losses": sum(outcome == 0.0 for outcome in outcomes),
            "score_rate": statistics.fmean(seed_outcomes),
            "score_rate_95ci": [max(0.0, score_ci[0]), min(1.0, score_ci[1])],
            "mean_reward": statistics.fmean(rewards),
            "median_reward": statistics.median(rewards),
            "mean_margin": statistics.fmean(seed_margins),
            "median_margin": statistics.median(margins),
            "margin_95ci": list(margin_ci),
            "seats": seat_summaries,
            "seed_clusters": len(complete_pairs),
            "seed_cluster_statistics": [
                {
                    "seed": rows[0].seed,
                    "score_rate": statistics.fmean(float(row.outcome) for row in rows),
                    "mean_margin": statistics.fmean(float(row.margin) for row in rows),
                }
                for rows in sorted(complete_pairs, key=lambda rows: rows[0].seed)
            ],
        }
    )
    return summary


def _game_payload(result: GameResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "seat": result.candidate_seat,
        "candidate_seat": result.candidate_seat,
        "reward": result.candidate_reward,
        "candidate_reward": result.candidate_reward,
        "opponent_reward": result.opponent_reward,
        "candidate_status": result.candidate_status,
        "status": result.candidate_status,
        "opponent_status": result.opponent_status,
        "environment_done": result.environment_done,
        "steps": result.steps,
        "expected_steps": result.expected_steps,
        "complete": result.complete,
        "margin": result.margin,
        "outcome": result.outcome,
        "elapsed_seconds": result.elapsed_seconds,
        "candidate_action_seconds": _timing_summary(list(result.action_seconds)),
        "error": result.error,
        "captured_output": result.captured_output,
    }


def _normalize_json(value: Any, path: str = "payload") -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite JSON value at {path}: {value}")
    if isinstance(value, dict):
        return {str(key): _normalize_json(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(_normalize_json(payload), indent=2, sort_keys=True, allow_nan=False)


def _write_atomic(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        default="v27",
        help="built-in name, Python agent path, or 'v27' for the fixed public opponent",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEED_CLUSTERS,
        help="seed clusters; both seats run (use 128 for finalist evaluation)",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="optional checkpoint-selection evidence required by finalist packaging",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    artifact = args.artifact.expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    device = _resolve_device(args.device)
    if args.workers > 1 and device.type != "cpu":
        raise ValueError("parallel evaluation supports CPU only; use --workers 1 on accelerators")
    opponent_label, opponent = normalize_opponent(args.opponent)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="kaggriculture-evaluation-snapshot-") as name:
        snapshot_root = Path(name)
        artifact_snapshot = snapshot_root / "artifact.pt"
        shutil.copyfile(artifact, artifact_snapshot)
        artifact_provenance = _artifact_provenance(artifact_snapshot)
        artifact_provenance["path"] = str(artifact)
        selection = getattr(args, "selection_report", None)
        selection_provenance = (
            None if selection is None else _selection_provenance(selection, artifact_provenance)
        )
        if selection_provenance is not None:
            screening_start = selection_provenance["screening_seed_start"]
            screening_end = screening_start + selection_provenance["screening_seed_count"]
            finalist_end = args.seed_start + args.seeds
            if max(screening_start, args.seed_start) < min(screening_end, finalist_end):
                raise ValueError("finalist seeds overlap checkpoint-selection screening seeds")
        worker_opponent = opponent
        if opponent_label in BUILTIN_OPPONENTS:
            opponent_provenance = _opponent_provenance(opponent_label, opponent)
        else:
            opponent_snapshot = snapshot_root / "opponent.py"
            shutil.copyfile(opponent, opponent_snapshot)
            worker_opponent = str(opponent_snapshot)
            opponent_provenance = _opponent_provenance(
                opponent_label,
                worker_opponent,
                logical_path=opponent,
            )
        if selection_provenance is not None:
            selected_opponent = selection_provenance["opponent_provenance"].get(opponent_label)
            if selected_opponent != _opponent_identity(opponent_provenance):
                raise ValueError(
                    "finalist opponent does not match the checkpoint-selection evidence"
                )
        if args.workers == 1:
            _initialize_worker(
                str(artifact_snapshot),
                str(device),
                args.torch_threads,
                worker_opponent,
            )
            pairs = [_run_seed_pair(seed) for seed in seeds]
        else:
            context = mp.get_context("spawn")
            with context.Pool(
                processes=min(args.workers, len(seeds)),
                initializer=_initialize_worker,
                initargs=(
                    str(artifact_snapshot),
                    str(device),
                    args.torch_threads,
                    worker_opponent,
                ),
            ) as pool:
                pairs = list(pool.imap_unordered(_run_seed_pair, seeds))
    results = sorted(
        (result for pair in pairs for result in pair),
        key=lambda result: (result.seed, result.candidate_seat),
    )
    summary = summarize(results, args.seeds)
    return {
        "valid_for_selection": bool(summary["valid_for_selection"]),
        "artifact": str(artifact),
        "artifact_provenance": artifact_provenance,
        "opponent": opponent,
        "opponent_label": opponent_label,
        "opponent_provenance": opponent_provenance,
        "selection_provenance": selection_provenance,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "paired_seats": True,
        "workers": args.workers,
        "torch_threads_per_worker": args.torch_threads,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
        "games": [_game_payload(result) for result in results],
    }


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    payload["valid_for_selection"] = bool(payload["summary"]["valid_for_selection"])
    rendered = render_json(payload)
    print(rendered)
    if args.output:
        _write_atomic(args.output, rendered)
    if not payload["valid_for_selection"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
