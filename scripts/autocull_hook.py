#!/usr/bin/env python3
"""Decide whether a metrics journal has become uninformative at an eval boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from kaggriculture.telemetry import read_jsonl_snapshot

CONTINUE = 0
CULL = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--metric", default="holdout_nll")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--warmup", type=int, default=4, help="records before culling is possible")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="consecutive smoothed records without material improvement",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        required=True,
        help="absolute improvement needed to reset patience",
    )
    parser.add_argument(
        "--target",
        type=float,
        help="optional adequacy threshold; a run that reaches it is never culled",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="EMA weight on the newest held-out observation",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="best/latest useful checkpoint to bind into the cull record",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="durable decision file (default: JOURNAL.parent/autocull.json)",
    )
    parser.add_argument("--json", action="store_true", help="print the full decision record")
    return parser.parse_args()


def _metric_values(records: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for index, record in enumerate(records):
        if metric not in record:
            continue
        value = record[metric]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{metric} is not numeric in record {index}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{metric} is not finite in record {index}")
        values.append(numeric)
    return values


def _ema(values: Sequence[float], alpha: float) -> list[float]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("ema-alpha must be in (0, 1]")
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def _better(candidate: float, reference: float, delta: float, mode: Literal["min", "max"]) -> bool:
    return candidate <= reference - delta if mode == "min" else candidate >= reference + delta


def _target_reached(best: float, target: float | None, mode: Literal["min", "max"]) -> bool:
    if target is None:
        return False
    return best <= target if mode == "min" else best >= target


def decide(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    mode: Literal["min", "max"],
    warmup: int,
    patience: int,
    min_improvement: float,
    ema_alpha: float,
    target: float | None,
) -> dict[str, Any]:
    """Return an evidence-only cull decision over completed evaluation records."""
    if warmup < 1 or patience < 1:
        raise ValueError("warmup and patience must be positive")
    if not math.isfinite(min_improvement) or min_improvement <= 0.0:
        raise ValueError("min-improvement must be finite and positive")
    if target is not None and not math.isfinite(target):
        raise ValueError("target must be finite")
    raw = _metric_values(records, metric)
    smoothed = _ema(raw, ema_alpha)
    if not smoothed:
        return {
            "decision": "continue",
            "reason": "metric_not_observed",
            "observations": 0,
            "stale_observations": 0,
            "best_smoothed": None,
            "latest_smoothed": None,
        }

    material_reference = smoothed[0]
    observed_best = smoothed[0]
    last_improvement = 0
    for index, value in enumerate(smoothed[1:], start=1):
        observed_best = min(observed_best, value) if mode == "min" else max(observed_best, value)
        if _better(value, material_reference, min_improvement, mode):
            material_reference = value
            last_improvement = index
    stale = len(smoothed) - 1 - last_improvement
    raw_best = min(raw) if mode == "min" else max(raw)
    reached = _target_reached(raw_best, target, mode)
    enough_evidence = len(smoothed) >= warmup and stale >= patience
    should_cull = enough_evidence and not reached
    if should_cull:
        reason = "plateau_below_target" if target is not None else "plateau"
    elif reached:
        reason = "target_reached"
    elif len(smoothed) < warmup:
        reason = "warmup"
    else:
        reason = "improving_or_within_patience"
    return {
        "decision": "cull" if should_cull else "continue",
        "reason": reason,
        "observations": len(smoothed),
        "stale_observations": stale,
        "best_smoothed": observed_best,
        "best_raw": raw_best,
        "latest_smoothed": smoothed[-1],
        "latest_raw": raw[-1],
        "target_reached": reached,
    }


def _checkpoint_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    selected = path.expanduser().resolve()
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"checkpoint is not a regular file: {selected}")
    with selected.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(selected), "sha256": digest, "size_bytes": selected.stat().st_size}


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    journal = args.journal.expanduser().resolve()
    snapshot = read_jsonl_snapshot(journal)
    decision = decide(
        snapshot.records,
        metric=args.metric,
        mode=args.mode,
        warmup=args.warmup,
        patience=args.patience,
        min_improvement=args.min_improvement,
        ema_alpha=args.ema_alpha,
        target=args.target,
    )
    record = {
        "format_version": 1,
        "event": "AUTOCULL",
        "journal": {
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "records": len(snapshot.records),
        },
        "configuration": {
            "metric": args.metric,
            "mode": args.mode,
            "warmup": args.warmup,
            "patience": args.patience,
            "min_improvement": args.min_improvement,
            "ema_alpha": args.ema_alpha,
            "target": args.target,
        },
        "checkpoint": _checkpoint_record(args.checkpoint),
        **decision,
    }
    state = args.state or journal.parent / "autocull.json"
    _write_atomic(state, record)
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            f"AUTOCULL decision={record['decision']} reason={record['reason']} "
            f"metric={args.metric} observations={record['observations']} "
            f"stale={record['stale_observations']}"
        )
    raise SystemExit(CULL if record["decision"] == "cull" else CONTINUE)


if __name__ == "__main__":
    main()
