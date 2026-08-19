#!/usr/bin/env python3
"""Greedily select steps whose market orders the public v27 plan is better without.

The scripted v27 agent ties itself exactly, so a step-wise cancellation that wins
is a measured edge over the strongest agent in sight.  One round scans every
remaining step on the training seeds, then the leader is re-measured on held-out
seeds and accepted only if it improves there: the scan is allowed to overfit, the
accept rule is not.

Both seats are the same scripted agent apart from the accepted cancellations, so
the margin is entirely attributable to them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from probe_plan_search import CODES, Plan, make_edit, play, summarize

from kaggriculture.opponents import PUBLIC_V27_OPPONENT
from kaggriculture.rust_env import load_native


def spec_for(stops: frozenset[int]) -> str:
    return "stop-step:" + ",".join(str(step) for step in sorted(stops)) if stops else "base"


def score(
    module: Any,
    plan: Plan,
    stops: frozenset[int],
    seeds: np.ndarray,
    opponent: str,
) -> dict[str, Any]:
    money = play(
        module,
        seeds,
        edit=make_edit(spec_for(stops), plan),
        base="scripted-v27",
        opponent=opponent,
        steps=plan.steps,
    )
    return summarize(money)


def rank(summary: dict[str, Any]) -> tuple[float, float]:
    """Order candidates by how often they win, then by the typical margin.

    The leaderboard's pairwise rating accumulates wins, so win rate is the
    objective; it plateaus at a coarse game count, which is why a round scans
    cheaply and then re-ranks its leaders on enough games to resolve the plateau.
    """

    return (summary["win_rate"], summary["margin_median"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path(PUBLIC_V27_OPPONENT))
    parser.add_argument("--opponent", default="scripted-v27", choices=sorted(CODES))
    parser.add_argument("--scan-games", type=int, default=64)
    parser.add_argument("--rerank-games", type=int, default=256)
    parser.add_argument("--train-seed-start", type=int, default=90_001)
    parser.add_argument("--holdout-games", type=int, default=192)
    parser.add_argument("--holdout-seed-start", type=int, default=1_500_000)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument(
        "--finalists",
        type=int,
        default=12,
        help="training leaders re-measured on the holdout each round",
    )
    parser.add_argument("--initial-stops", default="", help="comma-separated steps to start from")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = load_native()
    plan = Plan(args.agent)
    if plan.digest != module.V27_SOURCE_SHA256:
        raise SystemExit(f"plan digest {plan.digest} is not the engine's table")

    # The re-rank set extends the scan set, so a leader is re-measured on the
    # seeds that chose it plus more: cheaper than a disjoint set and it cannot
    # promote a step that only survives on seeds the scan never saw.
    training = max(args.scan_games, args.rerank_games)
    scan_seeds = np.arange(
        args.train_seed_start, args.train_seed_start + args.scan_games, dtype=np.uint64
    )
    rerank_seeds = np.arange(
        args.train_seed_start, args.train_seed_start + args.rerank_games, dtype=np.uint64
    )
    holdout = np.arange(
        args.holdout_seed_start, args.holdout_seed_start + args.holdout_games, dtype=np.uint64
    )
    if args.holdout_seed_start < args.train_seed_start + training:
        raise SystemExit("holdout seeds must not overlap the training seeds")

    stops = frozenset(int(part) for part in args.initial_stops.split(",") if part)
    accepted = score(module, plan, stops, holdout, args.opponent)
    history = [{"round": 0, "stops": sorted(stops), "holdout": accepted}]
    print(json.dumps(history[0]), flush=True)

    started = time.perf_counter()
    for round_index in range(1, args.rounds + 1):
        scanned = []
        for step in range(plan.steps):
            if step in stops:
                continue
            summary = score(module, plan, stops | {step}, scan_seeds, args.opponent)
            scanned.append((rank(summary), step))
        scanned.sort(reverse=True)

        # Re-rank the cheap scan's leaders on enough games to separate them, then
        # let the untouched holdout decide. Two stages, because a 64-game scan
        # cannot tell 0.92 from 0.95 and a 256-game scan of 719 steps is waste.
        reranked = []
        for _, step in scanned[: args.finalists]:
            summary = score(module, plan, stops | {step}, rerank_seeds, args.opponent)
            reranked.append((rank(summary), step, summary))
        reranked.sort(key=lambda item: item[0], reverse=True)

        chosen: dict[str, Any] | None = None
        for _, step, train_summary in reranked:
            trial = score(module, plan, stops | {step}, holdout, args.opponent)
            # The holdout decides, so a step that only wins on the scan seeds is
            # rejected rather than banked as progress.
            if rank(trial) > rank(accepted):
                chosen = {"step": step, "train": train_summary, "holdout": trial}
                break
        if chosen is None:
            print(
                json.dumps({"round": round_index, "stopped": "no candidate improved the holdout"}),
                flush=True,
            )
            break
        stops = stops | {chosen["step"]}
        accepted = chosen["holdout"]
        record = {
            "round": round_index,
            "added": chosen["step"],
            "stops": sorted(stops),
            "train": chosen["train"],
            "holdout": accepted,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "search": "plan-stops",
                "agent": str(args.agent),
                "agent_sha256": plan.digest,
                "opponent": args.opponent,
                "train_seed_start": args.train_seed_start,
                "scan_games": args.scan_games,
                "rerank_games": args.rerank_games,
                "holdout_seed_start": args.holdout_seed_start,
                "holdout_games": args.holdout_games,
                "stops": sorted(stops),
                "edit": spec_for(stops),
                "holdout": accepted,
                "history": history,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
