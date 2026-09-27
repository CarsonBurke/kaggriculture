#!/usr/bin/env python3
"""Paired full-episode tournament for standalone agents in the official engine.

Run local tournaments through mlq. This also runs in a private Kaggle CPU
notebook, using the same environment version as hosted competition games.
The official loader creates a fresh source namespace for each game. Workers
can optionally be isolated per game when module-level state is a concern.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def validate_episode(env, task: dict) -> tuple[list[str], list[float]]:
    """Reject failures even when the interpreter overwrites the final status."""
    final = env.steps[-1]
    statuses = [str(row.status) for row in final]
    money = [float(farm.money) for farm in final[0].observation.farms]
    failures = [
        (step, seat, str(row.status))
        for step, state in enumerate(env.steps)
        for seat, row in enumerate(state)
        if str(row.status) not in {"ACTIVE", "DONE"}
    ]
    if failures:
        raise RuntimeError(f"Agent failure during game: {task}, failures={failures[:5]}")
    if any(not isinstance(row.action, dict) for state in env.steps[1:] for row in state):
        raise RuntimeError(f"Missing or invalid action during game: {task}")
    if not env.done or len(env.steps) != 720 or statuses != ["DONE", "DONE"]:
        raise RuntimeError(
            f"Invalid full game: {task}, steps={len(env.steps)}, statuses={statuses}"
        )
    if (
        not all(math.isfinite(value) for value in money)
        or [float(row.reward) for row in final] != money
    ):
        raise RuntimeError(f"Terminal rewards disagree with banks: {task}")
    if "expected_money" in task and money != task["expected_money"]:
        raise RuntimeError(f"Replay reproduction failed: {task}, actual banks={money}")
    return statuses, money


def play_game(task: dict) -> dict:
    # Import only inside the worker: the parent never initializes all environments.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    from kaggle_environments import make

    started = time.monotonic()
    env = make(
        "kaggriculture", configuration={"seed": task["seed"], "episodeSteps": 720}, debug=False
    )
    env.run(task["paths"])
    statuses, money = validate_episode(env, task)
    # A PASS fallback can have DONE status. Keep activity visible in the report.
    active = [
        sum(
            bool(row[seat].action)
            and (
                row[seat].action.get("farmer") != ["PASS"]
                or bool(row[seat].action.get("market"))
                or any(a != ["PASS"] for a in row[seat].action.get("hands", []))
            )
            for row in env.steps[1:]
        )
        for seat in (0, 1)
    ]
    result = {
        "agents": task["agents"],
        "seed": task["seed"],
        "money": money,
        "statuses": statuses,
        "active_turns": active,
        "max_action_seconds": [
            max(
                (logs[seat].get("duration", 0.0) for logs in env.logs if len(logs) > seat),
                default=0.0,
            )
            for seat in (0, 1)
        ],
        "agent_stderr": [
            [
                {"step": step, "text": logs[seat]["stderr"][:2000]}
                for step, logs in enumerate(env.logs)
                if len(logs) > seat and logs[seat].get("stderr", "").strip()
            ][:5]
            for seat in (0, 1)
        ],
        "seconds": time.monotonic() - started,
    }
    del env
    gc.collect()
    return result


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        for seat in (0, 1):
            key = (row["agents"][seat], row["agents"][1 - seat])
            margin = row["money"][seat] - row["money"][1 - seat]
            groups.setdefault(key, []).append(
                {
                    "seed": row["seed"],
                    "seat": seat,
                    "score": float(margin > 0) + 0.5 * float(margin == 0),
                    "margin": margin,
                    "money": row["money"][seat],
                }
            )
    result = []
    for (agent, opponent), games in sorted(groups.items()):
        # Both orientations of a map are one cluster; report uncertainty on that unit.
        clusters: dict[int, list[float]] = {}
        for game in games:
            clusters.setdefault(game["seed"], []).append(game["score"])
        scores = [statistics.mean(values) for values in clusters.values()]
        se = statistics.stdev(scores) / len(scores) ** 0.5 if len(scores) > 1 else None
        result.append(
            {
                "agent": agent,
                "opponent": opponent,
                "games": len(games),
                "wins": sum(g["score"] == 1.0 for g in games),
                "ties": sum(g["score"] == 0.5 for g in games),
                "losses": sum(g["score"] == 0.0 for g in games),
                "seed_clusters": len(scores),
                "score": statistics.mean(scores),
                "score_cluster_standard_error": se,
                "seat_scores": [
                    statistics.mean(values) if values else None
                    for s in (0, 1)
                    for values in [[g["score"] for g in games if g["seat"] == s]]
                ],
                "money_mean": statistics.mean(g["money"] for g in games),
                "margin_mean": statistics.mean(g["margin"] for g in games),
                "margin_min": min(g["margin"] for g in games),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="append", required=True, help="NAME=PATH or NAME=builtin")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--games-per-process",
        type=int,
        default=1,
        choices=(0, 1),
        help="1 isolates games; 0 reuses workers for agents that reinitialize on load",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-version", default="1.32.7")
    parser.add_argument(
        "--tasks", type=Path, help="Explicit games with agents, seed, and optional expected_money"
    )
    args = parser.parse_args()
    if args.seeds < 2 or args.workers < 1:
        parser.error("Use at least two seed clusters and one worker")
    version = importlib.metadata.version("kaggle-environments")
    if version != args.required_version:
        raise RuntimeError(f"Expected environment {args.required_version}, got {version}")
    agents = dict(spec.split("=", 1) for spec in args.agent)
    if len(agents) < 2 or len(agents) != len(args.agent):
        parser.error("Need at least two uniquely named agents")
    identities = {}
    for name, path in agents.items():
        if path in {"starter", "random", "pass"}:
            identities[name] = {"builtin": path}
        else:
            file = Path(path).resolve(strict=True)
            agents[name] = str(file)
            identities[name] = {
                "path": str(file),
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
    tasks = [
        {"agents": list(order), "paths": [agents[name] for name in order], "seed": seed}
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for pair in itertools.combinations(agents, 2)
        for order in (pair, pair[::-1])
    ]
    if args.tasks is not None:
        tasks = json.loads(args.tasks.read_text())
        for task in tasks:
            if (
                len(task["agents"]) != 2
                or len(set(task["agents"])) != 2
                or type(task["seed"]) is not int
            ):
                raise ValueError(f"Invalid explicit game: {task}")
            task["paths"] = [agents[name] for name in task["agents"]]
    task_keys = [(tuple(task["agents"]), task["seed"]) for task in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("Duplicate ordered game and seed in tournament tasks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists():
        raise FileExistsError("Refusing to overwrite tournament evidence")
    rows = []
    with (
        journal.open("x") as stream,
        ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            # CPython 3.12 can deadlock when recycling workers after >1 task.
            # Use either full isolation or no recycling, never an intermediate count.
            max_tasks_per_child=args.games_per_process or None,
        ) as pool,
    ):
        for row in pool.map(play_game, tasks):
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({"completed": len(rows), "total": len(tasks), **row}), flush=True)
    for name, identity in identities.items():
        if "sha256" in identity:
            actual = hashlib.sha256(Path(agents[name]).read_bytes()).hexdigest()
            if actual != identity["sha256"]:
                raise RuntimeError(f"Agent source changed during evaluation: {name}")
    report = {
        "environment_version": version,
        "agents": identities,
        "games_per_process": args.games_per_process,
        "seed_start": None if args.tasks else args.seed_start,
        "seed_count": len({row["seed"] for row in rows}),
        "summary": summarize(rows),
        "games": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
