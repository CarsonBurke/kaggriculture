#!/usr/bin/env python3
"""Summarize one or more PPO metrics journals without loading checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="PPO run directory, metrics.jsonl journal, or config.json",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="actor iterations included in recent timing aggregates (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one compact JSON object per run instead of a human summary",
    )
    return parser.parse_args()


def _run_directory(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_dir() else path.parent


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"missing {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"no metrics in {path}")
    return rows


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row, key)) is not None]
    return statistics.fmean(values) if values else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row, key)) is not None]
    return statistics.median(values) if values else None


def summarize_run(path: Path, *, recent: int = 10) -> dict[str, Any]:
    if recent < 1:
        raise ValueError("recent window must be positive")
    run_directory = _run_directory(path)
    metrics = _read_metrics(run_directory / "metrics.jsonl")
    config = _read_json(run_directory / "config.json")
    arguments = config.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"expected arguments object in {run_directory / 'config.json'}")

    latest = metrics[-1]
    actor_rows = [row for row in metrics if (_number(row, "actor_updates") or 0.0) > 0.0]
    warmup_rows = [row for row in metrics if (_number(row, "actor_updates") or 0.0) == 0.0]
    recent_actor_rows = actor_rows[-recent:]
    money_rows = [row for row in actor_rows if _number(row, "money_mean") is not None]
    best_money = max(money_rows, key=lambda row: float(row["money_mean"])) if money_rows else None

    target = arguments.get("iterations")
    if not isinstance(target, int) or isinstance(target, bool):
        target = None
    return {
        "run": str(run_directory),
        "iteration": int(latest["iteration"]),
        "iterations_target": target,
        "actor_iterations": len(actor_rows),
        "actor_updates": sum(int(row.get("actor_updates", 0)) for row in actor_rows),
        "warmup_iterations": len(warmup_rows),
        "warmup_money_mean": _mean(warmup_rows, "money_mean"),
        "warmup_entropy_mean": _mean(warmup_rows, "rollout_entropy"),
        "best_actor_money": None if best_money is None else float(best_money["money_mean"]),
        "best_actor_money_iteration": None if best_money is None else int(best_money["iteration"]),
        "latest_money": _number(latest, "money_mean"),
        "latest_score": _number(latest, "score_rate"),
        "latest_entropy": _number(latest, "rollout_entropy"),
        "latest_actor_updates": int(latest.get("actor_updates", 0)),
        "latest_actor_minibatches": int(latest.get("actor_minibatches_intended", 0)),
        "latest_first_kl": _number(latest, "first_minibatch_approx_kl"),
        "latest_max_kl": _number(latest, "max_approx_kl"),
        "latest_kl_early_stop": int(latest.get("kl_early_stop", 0)),
        "latest_iteration_seconds": _number(latest, "iteration_seconds"),
        "latest_rollout_seconds": _number(latest, "rollout_seconds"),
        "latest_update_seconds": _number(latest, "update_seconds"),
        "recent_actor_window": len(recent_actor_rows),
        "recent_iteration_seconds_median": _median(recent_actor_rows, "iteration_seconds"),
        "recent_rollout_seconds_median": _median(recent_actor_rows, "rollout_seconds"),
        "recent_update_seconds_median": _median(recent_actor_rows, "update_seconds"),
    }


def _format_number(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def format_summary(summary: dict[str, Any]) -> str:
    iteration = _format_number(summary["iteration"])
    target = _format_number(summary["iterations_target"])
    return " ".join(
        (
            f"run={Path(summary['run']).name}",
            f"iter={iteration}/{target}",
            f"actor_iters={summary['actor_iterations']}",
            f"actor_updates={summary['actor_updates']}",
            f"money={_format_number(summary['latest_money'])}",
            f"best_money={_format_number(summary['best_actor_money'])}@{_format_number(summary['best_actor_money_iteration'])}",
            f"entropy={_format_number(summary['latest_entropy'])}",
            f"kl={_format_number(summary['latest_first_kl'])}/{_format_number(summary['latest_max_kl'])}",
            f"minibatches={summary['latest_actor_updates']}/{summary['latest_actor_minibatches']}",
            f"seconds={_format_number(summary['latest_iteration_seconds'])}",
            f"recent_update_median={_format_number(summary['recent_update_seconds_median'])}",
        )
    )


def main() -> None:
    args = parse_args()
    try:
        summaries = [summarize_run(path, recent=args.recent) for path in args.runs]
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for summary in summaries:
        if args.json:
            print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        else:
            print(format_summary(summary))


if __name__ == "__main__":
    main()
