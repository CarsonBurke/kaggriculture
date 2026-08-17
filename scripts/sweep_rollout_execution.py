"""Measure collection cost and replay-parity drift across execution modes.

Choosing how the collection forward runs is a two-sided decision. One side is
cost: the forward is roughly two thirds of collection wall clock, and an isolated
benchmark puts Inductor at 1.8x fp32 and 3.0x with bfloat16 against eager. The
other side is drift: every mode moves the sampled behavior policy away from the
distribution the update path reconstructs, and `update_replay_parity` gates that
divergence. A speedup that fails the gate is not a speedup.

This measures both sides of the same wave, for each mode and precision, on a
real trained actor -- head sharpness drives the divergence, so a randomly
initialized actor measures a number two orders of magnitude low and would bless
a configuration that fails in production.

Reported per cell: median rollout seconds, and the two gated statistics with
their margins against `MAX_UPDATE_REPLAY_KL` and `MAX_FIRST_MINIBATCH_KL`. A
cell whose margin is below one is inadmissible regardless of its speed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from kaggriculture.inference import load_actor_artifact
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    PpoConfig,
    update_replay_parity,
)
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import (
    ROLLOUT_FORWARD_MODES,
    allocate_rollout_storage,
    collect_self_play_rust,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--games", type=int, default=112)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260817)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("this sweep measures a CUDA configuration and needs a GPU")

    actor, payload = load_actor_artifact(args.actor, device=device)
    actor.eval()
    architecture = resolve_architecture(payload)
    ppo_config = PpoConfig(minibatch_size=args.minibatch_size)
    arena = allocate_rollout_storage(
        architecture.name,
        trajectories=args.games * 2,
        horizon=args.episode_steps - 1,
    )

    cells: dict[str, Any] = {}
    for mode in ROLLOUT_FORWARD_MODES:
        for bf16 in (False, True):
            label = f"{mode}/{'bf16' if bf16 else 'fp32'}"
            seconds: list[float] = []
            parity: list[dict[str, float]] = []
            try:
                for repeat in range(args.repeats):
                    seed = args.base_seed + repeat * 4096
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    rollout = collect_self_play_rust(
                        actor,
                        games=args.games,
                        seed_start=seed,
                        episode_steps=args.episode_steps,
                        sampling_seed=seed ^ 0x5EED,
                        forward_mode=mode,
                        forward_autocast=bf16,
                        storage=arena,
                    )
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter() - started)
                    metrics = update_replay_parity(
                        actor,
                        rollout,
                        minibatch_size=ppo_config.minibatch_size,
                        compile_model=ppo_config.compile_update,
                        autocast_enabled=ppo_config.use_bfloat16,
                    )
                    parity.append({key: float(value) for key, value in metrics.items()})
            except Exception as error:  # a mode failing to run is itself a result
                cells[label] = {"error": f"{type(error).__name__}: {error}"}
                continue

            worst_kl = max(record["update_replay_max_kl"] for record in parity)
            worst_first = max(record["update_replay_first_minibatch_kl"] for record in parity)
            cells[label] = {
                "rollout_seconds_median": statistics.median(seconds),
                "rollout_seconds": seconds,
                "worst_update_replay_max_kl": worst_kl,
                "worst_first_minibatch_kl": worst_first,
                "kl_margin": MAX_UPDATE_REPLAY_KL / max(worst_kl, 1e-30),
                "first_minibatch_margin": MAX_FIRST_MINIBATCH_KL / max(worst_first, 1e-30),
                "admissible": worst_kl <= MAX_UPDATE_REPLAY_KL
                and worst_first <= MAX_FIRST_MINIBATCH_KL,
            }
            print(f"{label:22s} {json.dumps(cells[label], sort_keys=True)}", flush=True)

    baseline = cells.get("eager/fp32", {}).get("rollout_seconds_median")
    report = {
        "actor": str(args.actor),
        "games": args.games,
        "episode_steps": args.episode_steps,
        "repeats": args.repeats,
        "max_update_replay_kl": MAX_UPDATE_REPLAY_KL,
        "max_first_minibatch_kl": MAX_FIRST_MINIBATCH_KL,
        "cells": cells,
    }
    if baseline:
        report["speedup_over_eager_fp32"] = {
            label: baseline / cell["rollout_seconds_median"]
            for label, cell in cells.items()
            if "rollout_seconds_median" in cell
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
