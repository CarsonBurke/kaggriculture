#!/usr/bin/env python3
"""Fit two critics on one frozen-policy wave; this is an offline diagnostic, not PPO evidence."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.critic_diagnostics import critic_replay_arrays, terminal_outcomes
from kaggriculture.evaluation import artifact_seed_usage, seed_protocol
from kaggriculture.provenance import file_sha256, source_identity

HORIZON = 719
SELF_PLAY_GAMES = 128
LEAGUE_GAMES = 64
TRAJECTORIES = 2 * SELF_PLAY_GAMES + LEAGUE_GAMES


def split_seed_rows(seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hold out every fifth physical seed, keeping the two self-play seats together."""
    seeds = np.asarray(seeds)
    if seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer):
        raise ValueError("physical seeds must be a one-dimensional integer array")
    holdout = seeds % 5 == 0
    train_rows, holdout_rows = np.flatnonzero(~holdout), np.flatnonzero(holdout)
    if not train_rows.size or not holdout_rows.size:
        raise ValueError("physical-seed split requires nonempty train and holdout sets")
    return train_rows, holdout_rows


def fit_statistics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    rows: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Report actual R-squared, including bias, on outcome returns by time-to-go."""
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if (
        predictions.shape != (outcomes.size, HORIZON)
        or not np.isfinite(predictions).all()
        or not np.isfinite(outcomes).all()
    ):
        raise ValueError("critic predictions and terminal outcomes must be finite and complete")
    time_to_go = np.arange(HORIZON, 0, -1)
    windows = {
        "all": np.ones(HORIZON, dtype=bool),
        "ttg_1_32": time_to_go <= 32,
        "ttg_33_128": (time_to_go > 32) & (time_to_go <= 128),
        "ttg_129_512": (time_to_go > 128) & (time_to_go <= 512),
        "ttg_513_plus": time_to_go > 512,
    }
    result = {}
    for group in ("all", *sorted(set(groups.tolist()))):
        chosen = rows if group == "all" else rows[groups[rows] == group]
        if not chosen.size:
            continue
        target = outcomes[chosen]
        variance = float(np.var(target))
        result[group] = {}
        for label, window in windows.items():
            residual = predictions[chosen][:, window] - target[:, None]
            mse = float(np.mean(residual**2))
            result[group][label] = {
                "trajectories": int(chosen.size),
                "states": int(residual.size),
                "mse": mse,
                "r_squared": 1.0 - mse / variance if variance > 0 else None,
                "target_variance": variance,
                "prediction_bias": float(np.mean(residual)),
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=540)
    parser.add_argument("--seed-start", type=int, default=4_502_000)
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("epochs and max-seconds must be positive")
    seed_protocol("development", args.seed_start, SELF_PLAY_GAMES + LEAGUE_GAMES, usage=[])
    return args


def main() -> None:
    args = parse_args()
    import torch

    from kaggriculture.entity import EntityActor, EntityCritic
    from kaggriculture.inference import load_actor_artifact
    from kaggriculture.ppo import (
        PpoConfig,
        make_optimizers,
        make_structured_dynamics_optimizer,
        replay_behavior_values,
        update_ppo,
    )
    from kaggriculture.production import production_ppo_config
    from kaggriculture.rollout import collect_mixed_play_rust
    from kaggriculture.structured_dynamics import StructuredCriticDynamics

    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("common-policy fit requires queued CUDA with native BF16")
    started = time.perf_counter()
    torch.set_num_threads(1)
    actor_digest = file_sha256(args.actor)
    actor, metadata = load_actor_artifact(args.actor, device="cuda")
    if not isinstance(actor, EntityActor):
        raise ValueError("this matched source-read diagnostic requires an entity actor")
    actor.eval()
    initial_actor = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    protocol = seed_protocol(
        "development",
        args.seed_start,
        SELF_PLAY_GAMES + LEAGUE_GAMES,
        usage=artifact_seed_usage(metadata),
    )
    config = PpoConfig(**production_ppo_config(update_compile_mode="reduce-overhead"))
    if config.gamma != 1.0 or config.critic_gae_lambda != 1.0 or config.critic_epochs != 1:
        raise ValueError("common-policy fit requires Monte Carlo critic targets and one epoch")
    assignments = np.arange(LEAGUE_GAMES) % 3
    rollout = collect_mixed_play_rust(
        actor,
        opponents=(actor,),
        self_play_games=SELF_PLAY_GAMES,
        league_games=LEAGUE_GAMES,
        opponent_indices=assignments,
        builtin_lanes=("starter", "scripted-v27"),
        seed_start=args.seed_start,
        deterministic=False,
        temperature=1.0,
        opponent_temperature=1.0,
        episode_steps=720,
        reward_mode="terminal-outcome",
        sampling_seed=20260919,
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    if rollout.state_count != TRAJECTORIES * HORIZON or not rollout.valid.all():
        raise RuntimeError("common-policy diagnostic requires the full production-size wave")
    outcomes = terminal_outcomes(rollout)
    train_rows, holdout_rows = split_seed_rows(rollout.episode_seeds)
    groups = np.asarray(
        ["self_play"] * (2 * SELF_PLAY_GAMES)
        + [
            str(np.asarray(["bc_snapshot", "starter", "scripted-v27"])[lane])
            for lane in assignments
        ]
    )
    report: dict[str, Any] = {
        "format_version": 1,
        "source_identity": source_identity(),
        "scope": "Offline critic fit on one frozen-policy wave; not online learning evidence",
        "actor": {
            "path": str(args.actor.resolve()),
            "sha256": actor_digest,
            "source_identity": metadata["source_identity"],
            "model_config": metadata["model_config"],
        },
        "seed_protocol": protocol,
        "ppo": asdict(config),
        "epochs_requested": args.epochs,
        "max_seconds": args.max_seconds,
        "initialization_seed": 20260920,
        "sampling_seed": 20260919,
        "wave": {
            "self_play_games": SELF_PLAY_GAMES,
            "league_games": LEAGUE_GAMES,
            "states": rollout.state_count,
            "rollout_seconds": rollout.elapsed_seconds,
            "episode_seeds": rollout.episode_seeds.tolist(),
            "seats": rollout.seats.tolist(),
            "opponents": groups.tolist(),
            "outcomes": outcomes.tolist(),
            "train_rows": train_rows.tolist(),
            "holdout_rows": holdout_rows.tolist(),
            "split": "Hold out complete physical seeds divisible by five; never split seats",
        },
        "variants": {},
    }
    variants = {}
    baseline_weights = None
    for label, source_read in (("baseline", False), ("source_read", True)):
        torch.manual_seed(20260920)
        critic_config = replace(actor.config, critic_source_read=source_read)
        critic = EntityCritic(critic_config).to("cuda")
        if baseline_weights is None:
            baseline_weights = {
                name: value.detach().clone() for name, value in critic.state_dict().items()
            }
        else:
            state = critic.state_dict()
            for name, value in baseline_weights.items():
                state[name].copy_(value)
        torch.manual_seed(20260921)
        dynamics = (
            StructuredCriticDynamics(critic_config).to("cuda")
            if config.structured_critic_auxiliary_active
            else None
        )
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
        dynamics_optimizer = (
            make_structured_dynamics_optimizer(dynamics, config) if dynamics is not None else None
        )
        variants[label] = (critic, dynamics, actor_optimizer, critic_optimizer, dynamics_optimizer)
        report["variants"][label] = {"model_config": asdict(critic_config), "epochs": []}
    del baseline_weights

    def evaluate(critic: Any) -> dict[str, Any]:
        # Stage for this read only and release before update_ppo stages its training tensors.
        arrays = critic_replay_arrays(rollout)
        staged = {
            name: torch.from_numpy(array.reshape((-1, *array.shape[2:]))).to("cuda")
            for name, array in arrays.items()
        }
        predictions = (
            replay_behavior_values(
                critic,
                rollout.architecture,
                staged,
                compile_mode=config.update_compile_mode,
                autocast_enabled=True,
            )
            .cpu()
            .numpy()
            .reshape(TRAJECTORIES, HORIZON)
        )
        return {
            "train": fit_statistics(predictions, outcomes, train_rows, groups),
            "holdout": fit_statistics(predictions, outcomes, holdout_rows, groups),
        }

    def save_progress(completed_epochs: int) -> None:
        report["epochs_completed"] = completed_epochs
        report["complete"] = completed_epochs == args.epochs and report.get(
            "actor_unchanged", False
        )
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    for label, (critic, *_rest) in variants.items():
        report["variants"][label]["epochs"].append({"epoch": 0, **evaluate(critic)})
    save_progress(0)
    for epoch in range(1, args.epochs + 1):
        if time.perf_counter() - started >= args.max_seconds:
            break
        for label, bundle in variants.items():
            critic, dynamics, actor_optimizer, critic_optimizer, dynamics_optimizer = bundle
            metrics = update_ppo(
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                rollout,
                config,
                generator=np.random.default_rng(20260922 + epoch),
                actor_epochs=0,
                rows=train_rows,
                structured_critic_dynamics=dynamics,
                structured_critic_dynamics_optimizer=dynamics_optimizer,
                structured_critic_auxiliary=config.structured_critic_auxiliary_active,
                auxiliary_generator=np.random.default_rng(20260923 + epoch),
            )
            record = {"epoch": epoch, "update": metrics, **evaluate(critic)}
            report["variants"][label]["epochs"].append(record)
            print(
                json.dumps({"variant": label, "epoch": epoch, **record["holdout"]["all"]["all"]}),
                flush=True,
            )
        save_progress(epoch)
    if any(
        not torch.equal(initial_actor[name], value.detach().cpu())
        for name, value in actor.state_dict().items()
    ):
        raise RuntimeError("critic-only diagnostic mutated the frozen actor")
    if file_sha256(args.actor) != actor_digest:
        raise RuntimeError("actor artifact changed during the diagnostic")
    report["actor_unchanged"] = True
    report["stop_reason"] = (
        "epoch_budget" if report["epochs_completed"] == args.epochs else "soft_time_budget"
    )
    save_progress(report["epochs_completed"])


if __name__ == "__main__":
    main()
