#!/usr/bin/env python3
"""Compactly monitor MLQ pipelines and retry bounded transient launch failures."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

_TRANSIENT_FAILURE_MARKERS = (
    "transport endpoint is not connected",
    "runner lost",
    "daemon unavailable",
    "connection reset by peer",
    "broken pipe",
)
_TERMINAL_STATES = frozenset(("succeeded", "failed", "cancelled", "canceled"))
_UNSUCCESSFUL_TERMINAL_STATES = frozenset(("failed", "cancelled", "canceled"))


def _command(*arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"command returned a non-object: {arguments}")
    return payload


def _job(job_id: int) -> dict[str, Any]:
    return _command("mlq", "show", str(job_id), "--json")


def _is_transient_failure(job: dict[str, Any]) -> bool:
    if job.get("state") != "failed":
        return False
    reason = str(job.get("stateReason") or "").lower()
    return any(marker in reason for marker in _TRANSIENT_FAILURE_MARKERS)


def _render_job(job: dict[str, Any]) -> str:
    reason = f" ({job['stateReason']})" if job.get("stateReason") else ""
    attempts = int(job.get("attemptCount", len(job.get("attempts", ()))))
    suffix = f", attempts={attempts}" if attempts > 1 else ""
    return f"job {job['id']} {job['name']}: {job['state']}{reason}{suffix}"


def _number(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _render_report(path: Path) -> list[str]:
    if not path.exists():
        return [f"report {path}: pending"]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        return [f"report {path}: empty"]
    lines = [f"report {path}: {len(records)} records, last={records[-1].get('event', '?')}"]
    for record in records:
        if record.get("event") != "batch_summary":
            continue
        size = record.get("self_play_games", record.get("games"))
        if "steady_total_seconds_median" in record:
            lines.append(
                "  "
                f"games={size}: {_number(record.get('steady_total_seconds_median'))}s/iteration, "
                f"{_number(record.get('steady_iterations_per_hour_median'))}/hour"
            )
        else:
            lines.append(
                "  "
                f"games={size}: {_number(record.get('steady_state_complete_games_per_second'))} "
                "games/s"
            )
    iterations = [record for record in records if record.get("event") == "iteration"]
    if iterations:
        latest = iterations[-1]
        lines.append(
            "  latest: "
            f"games={latest.get('self_play_games')} {latest.get('phase')}, "
            f"total={_number(latest.get('total_seconds'))}s, "
            f"rollout={_number(latest.get('rollout_seconds'))}s, "
            f"update={_number(latest.get('update_seconds'))}s, "
            f"GPU={_number((latest.get('peak_cuda_bytes') or 0) / 2**30)}GiB"
        )
    return lines


def _render_report_cached(
    path: Path,
    cache: dict[Path, tuple[tuple[int, int] | None, list[str]]],
) -> list[str]:
    try:
        status = path.stat()
        signature: tuple[int, int] | None = (status.st_mtime_ns, status.st_size)
    except FileNotFoundError:
        signature = None
    cached = cache.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    rendered = _render_report(path)
    cache[path] = (signature, rendered)
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="*", type=int)
    parser.add_argument("--report", action="append", type=Path, default=[])
    parser.add_argument("--heal", action="store_true", help="retry transient launch failures")
    parser.add_argument("--watch", action="store_true", help="wait until every job is terminal")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-heals", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0.0 or args.max_heals < 0:
        raise ValueError("interval must be positive and max-heals non-negative")
    heals = {job_id: 0 for job_id in args.jobs}
    report_cache: dict[Path, tuple[tuple[int, int] | None, list[str]]] = {}
    previous = ""
    while True:
        jobs = [_job(job_id) for job_id in args.jobs]
        retried = False
        for job in jobs:
            job_id = int(job["id"])
            if args.heal and _is_transient_failure(job) and heals[job_id] < args.max_heals:
                _command("mlq", "retry", str(job_id), "--json")
                heals[job_id] += 1
                retried = True
        if retried:
            jobs = [_job(job_id) for job_id in args.jobs]
        lines = [_render_job(job) for job in jobs]
        for report in args.report:
            lines.extend(_render_report_cached(report, report_cache))
        rendered = "\n".join(lines)
        if rendered != previous:
            print(rendered, flush=True)
            previous = rendered
        if not args.watch:
            return 0
        healing_remaining = args.heal and any(
            _is_transient_failure(job) and heals[int(job["id"])] < args.max_heals for job in jobs
        )
        if all(job.get("state") in _TERMINAL_STATES for job in jobs) and not healing_remaining:
            return int(any(job.get("state") in _UNSUCCESSFUL_TERMINAL_STATES for job in jobs))
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
