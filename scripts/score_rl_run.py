#!/usr/bin/env python3
"""Rank RL runs by frozen-panel success: mean bank against public-v27.

In-league score rate is 0.5 by construction. This reads each run's
`metrics-external.jsonl` and reports the one number that can tell a living
economy from a collapse: best-member mean bank vs v27, with that member's
score rate and the worst member's v27 bank as health.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture.success import rank_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="run directories or metrics-external.jsonl files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the ranked records instead of the table",
    )
    return parser.parse_args()


def _journal(path: Path) -> Path:
    selected = path.expanduser()
    if selected.is_file():
        return selected.resolve()
    journal = selected / "metrics-external.jsonl"
    if not journal.is_file():
        raise FileNotFoundError(f"{selected}: no metrics-external.jsonl")
    return journal.resolve()


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f}"


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"

def main() -> None:
    args = parse_args()
    ranked = rank_runs(_journal(path) for path in args.runs)
    if args.json:
        print(json.dumps({"runs": ranked}, indent=2, sort_keys=True, allow_nan=False))
        return
    print(
        f"{'run':<36} {'iter':>5} {'agt':>3} {'success$':>10} "
        f"{'v27':>6} {'worst$':>10} {'starter$':>10} {'peak$':>10} {'peak@':>6}"
    )
    for row in ranked:
        latest = row["latest"]
        peak = row["peak"]
        name = Path(row["journal"]).parent.name
        print(
            f"{name:<36} {latest['iteration']:5d} {latest['best_agent']:3d} "
            f"{_fmt_money(latest['success']):>10} {_fmt_score(latest['v27_score']):>6} "
            f"{_fmt_money(latest['worst_v27_money']):>10} "
            f"{_fmt_money(latest['starter_money']):>10} "
            f"{_fmt_money(peak['success']):>10} {peak['iteration']:6d}"
        )


if __name__ == "__main__":
    main()
