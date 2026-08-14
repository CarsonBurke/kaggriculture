#!/usr/bin/env python3
"""Audit what an agent actually *does* over official-engine episodes.

Bank alone cannot tell a cloned policy apart from one that stumbled into
money: the BC acceptance gate asks whether the agent reproduces the
teacher's economic route — buys land, hires, runs animals, works crops —
not merely whether it banks well. This replays an actor on the real engine
and reports the route: the market orders it issued, the unit commands it
issued, and the farm it ended the episode holding.

Inference is CPU-only so an audit never contends with training on the GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from kaggriculture.constants import ANIMALS, CROPS
from kaggriculture.inference import CheckpointAgent
from kaggriculture.opponents import normalize_opponent
from kaggriculture.policy import act_batch
from kaggriculture.production import PRODUCTION_EPISODE_STEPS, PRODUCTION_TEMPERATURE

SELF_OPPONENT = "self"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="actor artifact or checkpoint")
    parser.add_argument("--seeds", type=int, default=8, help="episodes to audit")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--mode", choices=("deterministic", "sampled"), default="deterministic")
    parser.add_argument(
        "--opponent",
        default=SELF_OPPONENT,
        help=f"'{SELF_OPPONENT}' audits both seats of a mirror match; any other spec "
        "seats the opponent opposite the candidate",
    )
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--episode-steps", type=int, default=PRODUCTION_EPISODE_STEPS)
    parser.add_argument("--temperature", type=float, default=PRODUCTION_TEMPERATURE)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    return parser.parse_args()


def _final_farm_summary(farm: dict[str, Any]) -> dict[str, Any]:
    """What the agent is holding when the episode ends."""
    animals: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    structures: Counter[str] = Counter()
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crops[str(tile.get("crop"))] += 1
            elif kind == "ANIMAL":
                animals[str(tile.get("animal"))] += 1
            elif kind in ("COOP", "PASTURE"):
                structures[str(kind)] += 1
    return {
        "money": float(farm.get("money", 0) or 0),
        "unlocked_quadrants": len(farm.get("unlocked_quadrants") or []),
        "units": 1 + len(farm.get("hands") or []),
        "animal_tiles": {name: int(animals.get(name, 0)) for name in ANIMALS},
        "planted_tiles": {name: int(crops.get(name, 0)) for name in CROPS},
        "structure_tiles": {name: int(count) for name, count in sorted(structures.items())},
    }


def _episode_action_counts(steps: list[Any], seat: int) -> dict[str, dict[str, int]]:
    """Every market order and unit command the seat issued over the episode."""
    orders: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    for state in steps:
        action = state[seat].get("action")
        if not isinstance(action, dict):
            continue
        for order in action.get("market") or []:
            if order:
                orders["_".join(str(field) for field in order[:2])] += 1
        for command in [action.get("farmer"), *(action.get("hands") or [])]:
            if command:
                commands[str(command[0])] += 1
    return {
        "market_orders": dict(sorted(orders.items())),
        "unit_commands": dict(sorted(commands.items())),
    }


def audit_episode(
    agent: CheckpointAgent,
    *,
    seed: int,
    mode: str,
    episode_steps: int,
    temperature: float,
    opponent: str | None,
    candidate_seat: int,
) -> dict[str, Any]:
    """Play one engine episode and summarize the candidate seat's behavior."""
    generators = [
        np.random.default_rng(sequence) for sequence in np.random.SeedSequence(seed).spawn(2)
    ]

    def seat(player: int):
        def act(observation: dict[str, Any]) -> dict[str, Any]:
            if mode == "deterministic":
                return agent(observation)
            return act_batch(
                agent.actor,
                [observation],
                deterministic=False,
                temperature=temperature,
                generator=generators[player],
            ).actions[0]

        return act

    from kaggle_environments import make

    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=False,
    )
    if opponent is None:
        environment.run([seat(0), seat(1)])
        audited_seats = (0, 1)
    else:
        players: list[Any] = [opponent, opponent]
        players[candidate_seat] = seat(candidate_seat)
        environment.run(players)
        audited_seats = (candidate_seat,)
    farms = environment.steps[-1][0]["observation"]["farms"]
    return {
        "seed": seed,
        "steps": len(environment.steps),
        "seats": [
            {
                "seat": index,
                **_final_farm_summary(farms[index]),
                **_episode_action_counts(environment.steps, index),
            }
            for index in audited_seats
        ],
    }


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-seat records into one route description."""
    seats = [record for episode in episodes for record in episode["seats"]]
    if not seats:
        raise ValueError("audit produced no seat records")
    banks = [record["money"] for record in seats]
    orders: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    for record in seats:
        orders.update(record["market_orders"])
        commands.update(record["unit_commands"])

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "audited_seats": len(seats),
        "mean_bank": mean(banks),
        "median_bank": median(banks),
        "min_bank": min(banks),
        "max_bank": max(banks),
        "mean_unlocked_quadrants": mean([record["unlocked_quadrants"] for record in seats]),
        "mean_units": mean([record["units"] for record in seats]),
        "mean_animal_tiles": mean(
            [float(sum(record["animal_tiles"].values())) for record in seats]
        ),
        "mean_planted_tiles": mean(
            [float(sum(record["planted_tiles"].values())) for record in seats]
        ),
        "market_orders_per_seat": {
            name: count / len(seats) for name, count in sorted(orders.items())
        },
        "unit_commands_per_seat": {
            name: count / len(seats) for name, count in sorted(commands.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    if args.seed_start < 0:
        raise ValueError("seed start cannot be negative")
    if args.episode_steps < 1:
        raise ValueError("episode steps must be positive")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    artifact = args.artifact.expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact does not exist: {artifact}")
    opponent_label, opponent = (
        (SELF_OPPONENT, None)
        if args.opponent == SELF_OPPONENT
        else normalize_opponent(args.opponent)
    )
    agent = CheckpointAgent(artifact, device="cpu", torch_threads=args.torch_threads)
    episodes = [
        audit_episode(
            agent,
            seed=seed,
            mode=args.mode,
            episode_steps=args.episode_steps,
            temperature=args.temperature,
            opponent=opponent,
            candidate_seat=args.seat,
        )
        for seed in range(args.seed_start, args.seed_start + args.seeds)
    ]
    report = {
        "event": "behavior_audit",
        "artifact": str(artifact),
        "architecture": agent.metadata.get("architecture"),
        "iteration": int(agent.metadata.get("iteration", 0)),
        "opponent": opponent_label,
        "mode": args.mode,
        "episode_steps": args.episode_steps,
        "summary": summarize(episodes),
        "episodes": episodes,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: report[key] for key in report if key != "episodes"}, sort_keys=True))


if __name__ == "__main__":
    main()
