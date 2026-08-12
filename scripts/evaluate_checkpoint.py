#!/usr/bin/env python3
"""Evaluate an actor artifact against a built-in or Python-file opponent."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from kaggle_environments import make

from kaggriculture.inference import CheckpointAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    opponent = args.opponent
    if opponent not in {"pass", "random", "starter"}:
        opponent = str(Path(opponent).expanduser().resolve())
    agent = CheckpointAgent(args.artifact, device=torch.device(args.device))
    games = []
    started = time.perf_counter()
    action_seconds = []

    def timed_agent(observation):
        action_started = time.perf_counter()
        action = agent(observation)
        action_seconds.append(time.perf_counter() - action_started)
        return action

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for seat in (0, 1):
            players = [opponent, opponent]
            players[seat] = timed_agent
            environment = make(
                "kaggriculture",
                configuration={"episodeSteps": 720, "seed": seed},
                debug=False,
            )
            environment.run(players)
            final = environment.steps[-1]
            reward = float(final[seat].reward)
            opponent_reward = float(final[1 - seat].reward)
            margin = reward - opponent_reward
            games.append(
                {
                    "seed": seed,
                    "seat": seat,
                    "reward": reward,
                    "opponent_reward": opponent_reward,
                    "margin": margin,
                    "outcome": 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5,
                    "status": str(final[seat].status),
                }
            )
    outcomes = [game["outcome"] for game in games]
    margins = [game["margin"] for game in games]
    rewards = [game["reward"] for game in games]
    sorted_times = sorted(action_seconds)
    payload = {
        "artifact": str(args.artifact.resolve()),
        "opponent": opponent,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {
            "games": len(games),
            "score_rate": statistics.fmean(outcomes),
            "wins": sum(outcome == 1.0 for outcome in outcomes),
            "ties": sum(outcome == 0.5 for outcome in outcomes),
            "losses": sum(outcome == 0.0 for outcome in outcomes),
            "mean_reward": statistics.fmean(rewards),
            "median_reward": statistics.median(rewards),
            "mean_margin": statistics.fmean(margins),
            "median_margin": statistics.median(margins),
            "action_seconds_mean": statistics.fmean(action_seconds),
            "action_seconds_p99": sorted_times[int(0.99 * (len(sorted_times) - 1))],
            "action_seconds_max": max(action_seconds),
        },
        "games": games,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
