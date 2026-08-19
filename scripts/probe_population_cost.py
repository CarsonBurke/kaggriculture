#!/usr/bin/env python3
"""Cost and replay parity of a population wave against the mixed wave it replaces.

Stages 0(b) and 0(c) of `POPULATION_LEAGUE_PLAN.md`, measured together because
they read the same wave.

0(c) is the compute call the plan deliberately left open: cost parity runs
`G = 156` games for 312 trajectories, 78 per agent, and holds wall clock while
cutting per-agent data fourfold; data parity runs `G = 636` for 1,272
trajectories, 318 per agent, and pays in batch. The rollout forward is
launch-gap bound -- 859 launches summing 4.9 ms inside a measured 12.94 ms -- so
batch is expected to be the cheaper axis, and this measures whether that holds.

0(b) is a gate rather than a number. The population wave produces its behaviour
policy through `vmap` over `functional_call` on stacked weights, while the update
replays it through the plain module. If that shifts logprobs past the shipped
ceilings, the ratio in the surrogate measures the wrong thing, so every cell
carries `update_replay_parity` and a cell that fails is inadmissible whatever it
costs.

The actor must be trained: head sharpness drives replay divergence, and a
randomly initialized actor measures a number two orders of magnitude low. Weight
diversity across the population is irrelevant to both questions, so the same
artifact is loaded N times -- Stage 1 is where diversity is measured.

Host arena size is reported and enforced rather than discovered by an OOM: at
1,272 trajectories the staged wave is the largest allocation in the process, and
whether it fits is itself part of decision 2.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.inference import load_actor_artifact
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    PpoConfig,
    update_replay_parity,
)
from kaggriculture.production import (
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_ROLLOUT_BFLOAT16,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
)
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import (
    RolloutBatch,
    allocate_rollout_storage,
    collect_mixed_play_rust,
    collect_population_play_rust,
)
from kaggriculture.training import AnyActor

GIB = 1024**3


def _load(path: Path, device: torch.device) -> tuple[AnyActor, dict[str, Any]]:
    """Load one evaluation-mode actor, gradient-free: rollout takes no gradient."""
    actor, payload = load_actor_artifact(path, device=device)
    return actor.eval().requires_grad_(False), payload


def _arena_bytes(arena: dict[str, np.ndarray]) -> int:
    return sum(int(array.nbytes) for array in arena.values())


def _parity(
    actor: AnyActor,
    rollout: RolloutBatch,
    config: PpoConfig,
    rows: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = update_replay_parity(
        actor,
        rollout,
        rows=rows,
        minibatch_size=config.minibatch_size,
        compile_mode=config.update_compile_mode,
        autocast_enabled=config.use_bfloat16,
    )
    return {key: float(value) for key, value in metrics.items()}


def _summarize(
    label: str,
    seconds: list[float],
    parity: list[dict[str, float]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    worst_kl = max(record["update_replay_max_kl"] for record in parity)
    worst_first = max(record["update_replay_first_minibatch_kl"] for record in parity)
    cell = {
        "rollout_seconds_median": statistics.median(seconds),
        "rollout_seconds": seconds,
        "worst_update_replay_max_kl": worst_kl,
        "worst_first_minibatch_kl": worst_first,
        "kl_margin": MAX_UPDATE_REPLAY_KL / max(worst_kl, 1e-30),
        "first_minibatch_margin": MAX_FIRST_MINIBATCH_KL / max(worst_first, 1e-30),
        "admissible": worst_kl <= MAX_UPDATE_REPLAY_KL and worst_first <= MAX_FIRST_MINIBATCH_KL,
        **extra,
    }
    print(f"{label:28s} {json.dumps(cell, sort_keys=True)}", flush=True)
    return cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument(
        "--games",
        type=int,
        nargs="+",
        default=[156, 636],
        help="population wave sizes to measure; each must be a multiple of N*(N-1)",
    )
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260818)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument(
        "--arena-budget-gib",
        type=float,
        default=12.0,
        help="refuse a wave whose host arena exceeds this, rather than meeting the OOM",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this probe measures a CUDA configuration and needs a GPU")
    device = torch.device("cuda")

    actor, payload = _load(args.actor, device)
    architecture = resolve_architecture(payload)
    config = PpoConfig(minibatch_size=args.minibatch_size)
    # Cost and parity do not read the weights' diversity, only their sharpness.
    population = [actor] + [_load(args.actor, device)[0] for _ in range(args.population - 1)]

    cells: dict[str, Any] = {}

    # The wave being replaced, at exactly the shape production runs: one learner
    # forward over its own rows plus one stacked-ensemble forward over the frozen
    # rows. Its opponents are copies here, which costs what real snapshots cost --
    # the ensemble is stacked lane-wise regardless of what the lanes hold.
    mixed_trajectories = PRODUCTION_SELF_PLAY_GAMES * 2 + PRODUCTION_LEAGUE_GAMES
    mixed_arena = allocate_rollout_storage(
        architecture.name, trajectories=mixed_trajectories, horizon=args.episode_steps - 1
    )
    seconds: list[float] = []
    parity: list[dict[str, float]] = []
    mixed_rows = 0
    for repeat in range(args.repeats):
        seed = args.base_seed + repeat * 4096
        torch.cuda.synchronize()
        started = time.perf_counter()
        rollout = collect_mixed_play_rust(
            actor,
            opponents=population[1:],
            self_play_games=PRODUCTION_SELF_PLAY_GAMES,
            league_games=PRODUCTION_LEAGUE_GAMES,
            builtin_lanes=("pass", "random", "starter"),
            seed_start=seed,
            episode_steps=args.episode_steps,
            sampling_seed=seed ^ 0x5EED,
            forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
            forward_autocast=PRODUCTION_ROLLOUT_BFLOAT16,
            storage=mixed_arena,
        )
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        mixed_rows = int(rollout.trajectories)
        parity.append(_parity(actor, rollout, config))
    cells["mixed/production"] = _summarize(
        "mixed/production",
        seconds,
        parity,
        {
            "self_play_games": PRODUCTION_SELF_PLAY_GAMES,
            "league_games": PRODUCTION_LEAGUE_GAMES,
            "trajectories": mixed_rows,
            "trajectories_per_learner": mixed_rows,
            "arena_gib": _arena_bytes(mixed_arena) / GIB,
            "forwards_per_step": 2,
        },
    )
    del mixed_arena

    ordered_pairs = args.population * (args.population - 1)
    for games in args.games:
        label = f"population/G{games}"
        if games % ordered_pairs:
            cells[label] = {"error": f"{games} is not a multiple of {ordered_pairs} ordered pairs"}
            continue
        arena = allocate_rollout_storage(
            architecture.name, trajectories=games * 2, horizon=args.episode_steps - 1
        )
        arena_gib = _arena_bytes(arena) / GIB
        if arena_gib > args.arena_budget_gib:
            cells[label] = {
                "error": (
                    f"host arena {arena_gib:.1f} GiB exceeds the {args.arena_budget_gib} GiB budget"
                ),
                "arena_gib": arena_gib,
            }
            del arena
            print(f"{label:28s} {json.dumps(cells[label], sort_keys=True)}", flush=True)
            continue
        seconds = []
        parity = []
        rows = 0
        try:
            for repeat in range(args.repeats):
                seed = args.base_seed + repeat * 4096
                torch.cuda.synchronize()
                started = time.perf_counter()
                rollout = collect_population_play_rust(
                    population,
                    games=games,
                    seed_start=seed,
                    episode_steps=args.episode_steps,
                    sampling_seed=seed ^ 0x5EED,
                    forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
                    forward_autocast=PRODUCTION_ROLLOUT_BFLOAT16,
                    storage=arena,
                )
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - started)
                rows = int(rollout.trajectories)
                # Parity is read on one agent's rows. Every other row was sampled
                # by another agent's weights, so replaying the whole wave through
                # one module would report a divergence that is an artifact of the
                # wrong policy rather than of the vmap path under test.
                parity.append(
                    _parity(population[0], rollout, config, np.flatnonzero(rollout.agents == 0))
                )
        except Exception as error:  # a wave that cannot run is itself a result
            cells[label] = {"error": f"{type(error).__name__}: {error}", "arena_gib": arena_gib}
            print(f"{label:28s} {json.dumps(cells[label], sort_keys=True)}", flush=True)
            del arena
            continue
        cells[label] = _summarize(
            label,
            seconds,
            parity,
            {
                "games": games,
                "trajectories": rows,
                "trajectories_per_learner": games * 2 // args.population,
                "arena_gib": arena_gib,
                "forwards_per_step": 1,
            },
        )
        del arena

    baseline = cells.get("mixed/production", {}).get("rollout_seconds_median")
    report: dict[str, Any] = {
        "probe": "population-cost",
        "actor": str(args.actor),
        "population": args.population,
        "episode_steps": args.episode_steps,
        "repeats": args.repeats,
        "max_update_replay_kl": MAX_UPDATE_REPLAY_KL,
        "max_first_minibatch_kl": MAX_FIRST_MINIBATCH_KL,
        "cells": cells,
    }
    if baseline:
        report["seconds_over_mixed"] = {
            label: cell["rollout_seconds_median"] / baseline
            for label, cell in cells.items()
            if "rollout_seconds_median" in cell
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
