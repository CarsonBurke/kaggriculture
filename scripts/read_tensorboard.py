#!/usr/bin/env python3
"""Read TensorBoard scalar event files without starting a TensorBoard server."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdirs", nargs="+", type=Path)
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="exact scalar tag to include; repeat for multiple tags (default: every scalar)",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    return parser.parse_args()


def _event_directories(root: Path) -> tuple[Path, ...]:
    selected = root.expanduser().resolve()
    if not selected.is_dir():
        raise ValueError(f"TensorBoard logdir is not a directory: {selected}")
    directories: set[Path] = set()
    for event in selected.rglob("events.out.tfevents.*"):
        if event.is_symlink() or not event.is_file():
            raise ValueError(f"TensorBoard event path is not a regular file: {event}")
        directories.add(event.parent)
    if not directories:
        raise FileNotFoundError(f"no TensorBoard event files below {selected}")
    return tuple(sorted(directories))


def _finite(value: float, *, field: str, tag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite TensorBoard {field} for {tag}: {numeric}")
    return numeric


def read_scalars(root: Path, tags: set[str] | None = None) -> dict[str, Any]:
    """Return first/latest/min/max scalar values for every event run below root."""
    selected = root.expanduser().resolve()
    runs: list[dict[str, Any]] = []
    for directory in _event_directories(selected):
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
        available = sorted(accumulator.Tags().get("scalars", ()))
        chosen = available if not tags else [tag for tag in available if tag in tags]
        scalars: dict[str, Any] = {}
        for tag in chosen:
            events = accumulator.Scalars(tag)
            if not events:
                continue
            values = [_finite(event.value, field="value", tag=tag) for event in events]
            scalars[tag] = {
                "count": len(events),
                "first": {"step": int(events[0].step), "value": values[0]},
                "latest": {"step": int(events[-1].step), "value": values[-1]},
                "min": min(values),
                "max": max(values),
            }
        relative = directory.relative_to(selected).as_posix()
        runs.append({"run": "." if relative == "." else relative, "scalars": scalars})
    missing = sorted((tags or set()) - {tag for run in runs for tag in run["scalars"]})
    if missing:
        raise ValueError(f"requested scalar tags were absent: {', '.join(missing)}")
    return {"logdir": str(selected), "runs": runs}


def _render(report: dict[str, Any]) -> str:
    lines = [f"logdir: {report['logdir']}"]
    for run in report["runs"]:
        lines.append(f"run: {run['run']}")
        for tag, scalar in run["scalars"].items():
            latest = scalar["latest"]
            lines.append(
                f"  {tag}: {latest['value']:.8g} @ {latest['step']} "
                f"(n={scalar['count']}, min={scalar['min']:.8g}, max={scalar['max']:.8g})"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tags = set(args.tag) or None
    reports = [read_scalars(path, tags) for path in args.logdirs]
    if args.json:
        print(json.dumps({"tensorboard": reports}, indent=2, sort_keys=True, allow_nan=False))
        return
    print("\n\n".join(_render(report) for report in reports))


if __name__ == "__main__":
    main()
