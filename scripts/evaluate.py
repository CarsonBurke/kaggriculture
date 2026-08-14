#!/usr/bin/env python3
"""Evaluate two Kaggriculture agents over paired seats and deterministic seeds."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kaggriculture.opponents import normalize_opponent


@dataclass(frozen=True)
class GameSpec:
    seed: int
    candidate_seat: int
    candidate: str
    opponent: str


@dataclass(frozen=True)
class GameResult:
    seed: int
    candidate_seat: int
    candidate_reward: float | None
    opponent_reward: float | None
    candidate_status: str
    opponent_status: str
    elapsed_seconds: float
    error: str | None = None

    @property
    def margin(self) -> float | None:
        if self.candidate_reward is None or self.opponent_reward is None:
            return None
        return self.candidate_reward - self.opponent_reward

    @property
    def outcome(self) -> float | None:
        margin = self.margin
        if margin is None:
            return None
        if margin > 0:
            return 1.0
        if margin < 0:
            return 0.0
        return 0.5


def _normalize_agent(agent: str) -> str:
    runnable: str
    _, runnable = normalize_opponent(agent)
    return runnable


def _run_game(spec: GameSpec) -> GameResult:
    started = time.perf_counter()
    try:
        from kaggle_environments import make

        agents = [spec.opponent, spec.opponent]
        agents[spec.candidate_seat] = spec.candidate
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": spec.seed},
            debug=False,
        )
        env.run(agents)
        final = env.steps[-1]
        candidate = final[spec.candidate_seat]
        opponent = final[1 - spec.candidate_seat]
        return GameResult(
            seed=spec.seed,
            candidate_seat=spec.candidate_seat,
            candidate_reward=(None if candidate.reward is None else float(candidate.reward)),
            opponent_reward=None if opponent.reward is None else float(opponent.reward),
            candidate_status=str(candidate.status),
            opponent_status=str(opponent.status),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # pragma: no cover - exercised by invalid external agents
        return GameResult(
            seed=spec.seed,
            candidate_seat=spec.candidate_seat,
            candidate_reward=None,
            opponent_reward=None,
            candidate_status="ERROR",
            opponent_status="UNKNOWN",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _mean_confidence_interval(values: list[float]) -> tuple[float, float]:
    """Normal-approximation 95% interval, useful as a compact noisy-run diagnostic."""
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def summarize(results: list[GameResult]) -> dict[str, Any]:
    completed = [result for result in results if result.outcome is not None]
    outcomes = [result.outcome for result in completed]
    margins = [result.margin for result in completed]
    rewards = [result.candidate_reward for result in completed]
    assert all(value is not None for value in outcomes + margins + rewards)
    numeric_outcomes = [float(value) for value in outcomes]
    numeric_margins = [float(value) for value in margins]
    numeric_rewards = [float(value) for value in rewards]
    clustered: dict[int, list[GameResult]] = {}
    for result in completed:
        clustered.setdefault(result.seed, []).append(result)
    complete_pairs = [
        rows
        for rows in clustered.values()
        if len(rows) == 2 and {row.candidate_seat for row in rows} == {0, 1}
    ]
    seed_outcomes = [
        statistics.fmean(float(result.outcome) for result in rows) for rows in complete_pairs
    ]
    seed_margins = [
        statistics.fmean(float(result.margin) for result in rows) for rows in complete_pairs
    ]
    score_ci = _mean_confidence_interval(seed_outcomes)
    margin_ci = _mean_confidence_interval(seed_margins)

    seat_summaries = {}
    for seat in (0, 1):
        seat_rows = [result for result in completed if result.candidate_seat == seat]
        seat_outcomes = [float(result.outcome) for result in seat_rows]
        seat_margins = [float(result.margin) for result in seat_rows]
        seat_summaries[str(seat)] = {
            "games": len(seat_rows),
            "score_rate": statistics.fmean(seat_outcomes) if seat_outcomes else math.nan,
            "mean_margin": statistics.fmean(seat_margins) if seat_margins else math.nan,
        }

    return {
        "games_requested": len(results),
        "games_completed": len(completed),
        "errors": len(results) - len(completed),
        "wins": sum(outcome == 1.0 for outcome in numeric_outcomes),
        "ties": sum(outcome == 0.5 for outcome in numeric_outcomes),
        "losses": sum(outcome == 0.0 for outcome in numeric_outcomes),
        "score_rate": statistics.fmean(numeric_outcomes) if numeric_outcomes else math.nan,
        "score_rate_95ci": [max(0.0, score_ci[0]), min(1.0, score_ci[1])],
        "mean_reward": statistics.fmean(numeric_rewards) if numeric_rewards else math.nan,
        "median_reward": statistics.median(numeric_rewards) if numeric_rewards else math.nan,
        "mean_margin": statistics.fmean(numeric_margins) if numeric_margins else math.nan,
        "median_margin": statistics.median(numeric_margins) if numeric_margins else math.nan,
        "margin_95ci": list(margin_ci),
        "seed_clusters": len(clustered),
        "complete_seat_pairs": len(complete_pairs),
        "incomplete_seed_clusters": len(clustered) - len(complete_pairs),
        "seats": seat_summaries,
        "wall_seconds_sum": sum(result.elapsed_seconds for result in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Agent .py path or built-in name")
    parser.add_argument("--opponent", required=True, help="Agent .py path or built-in name")
    parser.add_argument("--seeds", type=int, default=12, help="Number of seeds; both seats run")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    candidate = _normalize_agent(args.candidate)
    opponent = _normalize_agent(args.opponent)
    specs = [
        GameSpec(seed=seed, candidate_seat=seat, candidate=candidate, opponent=opponent)
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for seat in (0, 1)
    ]

    started = time.perf_counter()
    if args.workers == 1:
        results = [_run_game(spec) for spec in specs]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=min(args.workers, len(specs))) as pool:
            results = list(pool.imap_unordered(_run_game, specs))
        results.sort(key=lambda result: (result.seed, result.candidate_seat))

    payload = {
        "candidate": candidate,
        "opponent": opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summarize(results),
        "games": [
            asdict(result) | {"margin": result.margin, "outcome": result.outcome}
            for result in results
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
