#!/usr/bin/env python3
"""Summarize official BC evaluations and paired differences against a baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="run directories or eval-*.json files",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="run directory used for exact seed/seat paired reward differences",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=271_828)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    return parser.parse_args()


def _evaluation_paths(path: Path) -> tuple[Path, ...]:
    selected = path.expanduser().resolve()
    if selected.is_file():
        if selected.is_symlink():
            raise ValueError(f"evaluation is a symlink: {selected}")
        return (selected,)
    if not selected.is_dir():
        raise FileNotFoundError(selected)
    evaluations = tuple(sorted(selected.glob("eval-*.json")))
    if not evaluations:
        raise FileNotFoundError(f"no eval-*.json files directly below {selected}")
    if any(path.is_symlink() or not path.is_file() for path in evaluations):
        raise ValueError(f"evaluation directory contains a non-regular result: {selected}")
    return evaluations


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation is not an object: {path}")
    return payload


def _finite(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"evaluation field {field} is not numeric: {path}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"evaluation field {field} is not finite: {path}")
    return numeric


def _validate_evaluation(path: Path, payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    games = payload.get("games")
    if not isinstance(summary, dict) or not isinstance(games, list):
        raise ValueError(f"evaluation lacks summary/games: {path}")
    requested = int(summary.get("games", 0))
    if requested <= 0 or requested != len(games):
        raise ValueError(f"evaluation game count is inconsistent: {path}")
    if not payload.get("valid_for_selection") or int(summary.get("invalid_games", -1)) != 0:
        raise ValueError(f"evaluation is not valid for selection: {path}")
    artifact = payload.get("artifact_provenance")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str):
        raise ValueError(f"evaluation lacks artifact provenance: {path}")
    for field in ("score_rate", "mean_margin", "mean_reward"):
        _finite(summary.get(field), field=field, path=path)


def _opponent(payload: Mapping[str, Any], path: Path) -> str:
    label = payload.get("opponent_label")
    if not isinstance(label, str) or not label:
        raise ValueError(f"evaluation lacks opponent_label: {path}")
    return label


def load_run(path: Path) -> dict[str, Any]:
    """Load one run or evaluation file into a provenance-bound opponent panel."""
    evaluations: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    artifact_sha256: str | None = None
    for evaluation_path in _evaluation_paths(path):
        payload = _load_json(evaluation_path)
        _validate_evaluation(evaluation_path, payload)
        opponent = _opponent(payload, evaluation_path)
        if opponent in evaluations:
            raise ValueError(f"duplicate opponent {opponent!r} below {path}")
        current_sha = str(payload["artifact_provenance"]["sha256"])
        if artifact_sha256 is not None and artifact_sha256 != current_sha:
            raise ValueError(f"evaluation panel mixes actor artifacts below {path}")
        artifact_sha256 = current_sha
        evaluations[opponent] = payload
        files[opponent] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    return {
        "run": str(path.expanduser().resolve()),
        "artifact_sha256": artifact_sha256,
        "evaluations": evaluations,
        "evaluation_sha256": files,
    }


def _game_key(game: Mapping[str, Any]) -> tuple[int, int]:
    return int(game["seed"]), int(game.get("candidate_seat", game["seat"]))


def _paired_reward_by_seed(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[float]:
    candidate_games = {_game_key(game): game for game in candidate["games"]}
    baseline_games = {_game_key(game): game for game in baseline["games"]}
    if candidate_games.keys() != baseline_games.keys():
        raise ValueError("candidate and baseline do not share an exact seed/seat panel")
    seeds = sorted({seed for seed, _ in candidate_games})
    differences: list[float] = []
    for seed in seeds:
        seats = sorted(seat for game_seed, seat in candidate_games if game_seed == seed)
        differences.append(
            sum(
                _finite(
                    candidate_games[(seed, seat)]["reward"], field="reward", path=Path("candidate")
                )
                - _finite(
                    baseline_games[(seed, seat)]["reward"], field="reward", path=Path("baseline")
                )
                for seat in seats
            )
            / len(seats)
        )
    return differences


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_reward_difference(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Cluster bootstrap the mean candidate reward difference by paired seed."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    differences = _paired_reward_by_seed(candidate, baseline)
    generator = random.Random(bootstrap_seed)
    means = [
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(bootstrap_samples)
    ]
    return {
        "mean": sum(differences) / len(differences),
        "bootstrap_95ci": [_quantile(means, 0.025), _quantile(means, 0.975)],
        "seed_clusters": len(differences),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def summarize_run(
    run: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 271_828,
) -> dict[str, Any]:
    evaluations: Mapping[str, Any] = run["evaluations"]
    rows: dict[str, Any] = {}
    for opponent, evaluation in sorted(evaluations.items()):
        summary = evaluation["summary"]
        row: dict[str, Any] = {
            "valid_for_selection": True,
            "games": int(summary["games"]),
            "score_rate": float(summary["score_rate"]),
            "mean_margin": float(summary["mean_margin"]),
            "mean_reward": float(summary["mean_reward"]),
        }
        if baseline is not None:
            baseline_evaluation = baseline["evaluations"].get(opponent)
            if baseline_evaluation is None:
                raise ValueError(f"baseline lacks opponent {opponent!r}")
            row["paired_reward_delta"] = paired_reward_difference(
                evaluation,
                baseline_evaluation,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        rows[opponent] = row
    return {
        "run": run["run"],
        "artifact_sha256": run["artifact_sha256"],
        "evaluation_sha256": run["evaluation_sha256"],
        "opponents": rows,
    }


def _render(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"{'run':<30} {'opponent':<12} {'score':>7} {'margin':>12} "
        f"{'reward':>12} {'delta':>12} {'95% CI':>27}"
    ]
    for run in rows:
        name = Path(str(run["run"])).name
        for opponent, result in run["opponents"].items():
            paired = result.get("paired_reward_delta")
            delta = "-" if paired is None else f"{paired['mean']:,.1f}"
            interval = (
                "-"
                if paired is None
                else f"[{paired['bootstrap_95ci'][0]:,.1f}, {paired['bootstrap_95ci'][1]:,.1f}]"
            )
            lines.append(
                f"{name:<30} {opponent:<12} {result['score_rate']:7.3f} "
                f"{result['mean_margin']:12,.1f} {result['mean_reward']:12,.1f} "
                f"{delta:>12} {interval:>27}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    baseline = load_run(args.baseline) if args.baseline is not None else None
    rows = [
        summarize_run(
            load_run(path),
            baseline,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for path in args.runs
    ]
    payload = {"baseline": None if baseline is None else baseline["run"], "runs": rows}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return
    print(_render(rows))


if __name__ == "__main__":
    main()
