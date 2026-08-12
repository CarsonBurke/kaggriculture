#!/usr/bin/env python3
"""Train a from-scratch Kaggriculture policy with self-play VAPO."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    ModelConfig,
    parameter_count,
)
from kaggriculture.rollout import (
    collect_frozen_opponent_play,
    collect_self_play,
    concatenate_rollouts,
)
from kaggriculture.training import (
    append_jsonl,
    load_checkpoint,
    rollout_diagnostics,
    save_checkpoint,
)
from kaggriculture.vapo import VapoConfig, make_optimizers, update_vapo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--league-games", type=int, default=16)
    parser.add_argument("--league-pool-size", type=int, default=8)
    parser.add_argument("--opponent-temperature", type=float, default=0.8)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--query-features", type=int, default=24)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--lr-warmup-steps", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--clip-low", type=float, default=0.80)
    parser.add_argument("--clip-high", type=float, default=1.28)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument(
        "--gae-lambda-alpha",
        type=float,
        default=0.0,
        help="0 uses exact Monte Carlo credit; positive values enable adaptive-lambda ablations",
    )
    parser.add_argument("--target-kl", type=float, default=0.08)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--no-bfloat16", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "iterations": args.iterations,
        "games": args.games,
        "league_pool_size": args.league_pool_size,
        "episode_steps": args.episode_steps,
        "checkpoint_every": args.checkpoint_every,
        "width": args.width,
        "residual_blocks": args.residual_blocks,
        "hidden": args.hidden,
        "query_features": args.query_features,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {', '.join(invalid)}")
    if args.episode_steps != 720:
        raise ValueError("training requires the competition horizon: --episode-steps 720")
    if args.league_games < 0:
        raise ValueError("league games cannot be negative")
    if not 0 < args.clip_low < 1 < args.clip_high:
        raise ValueError("clip interval must straddle one")
    if args.lr_warmup_steps < 0:
        raise ValueError("LR warmup steps cannot be negative")
    if args.gae_lambda_alpha < 0:
        raise ValueError("GAE lambda alpha cannot be negative")
    if args.temperature != 1.0:
        raise ValueError("on-policy VAPO currently requires --temperature 1.0")


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    args = parse_args()
    _validate_args(args)
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    model_config = ModelConfig(
        width=args.width,
        residual_blocks=args.residual_blocks,
        hidden=args.hidden,
        query_features=args.query_features,
    )
    vapo_config = VapoConfig(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        entropy_coefficient=args.entropy_coefficient,
        gae_lambda_alpha=args.gae_lambda_alpha,
        max_gradient_norm=args.max_gradient_norm,
        target_kl=args.target_kl,
        use_bfloat16=not args.no_bfloat16,
    )
    actor = FarmActor(model_config).to(device)
    critic = DistributionalCritic(model_config).to(device)
    opponent_actor = FarmActor(model_config).to(device)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, vapo_config)
    generator = np.random.default_rng(args.seed + 1)
    iteration = 0
    next_seed = args.seed
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            device=device,
        )
        if payload["model_config"] != model_config.to_dict():
            raise ValueError("resume checkpoint model configuration does not match arguments")
        if payload["vapo_config"] != asdict(vapo_config):
            raise ValueError("resume checkpoint VAPO configuration does not match arguments")
        iteration = int(payload["iteration"])
        next_seed = int(payload["next_seed"])
        if payload.get("training_rng") is not None:
            generator.bit_generator.state = payload["training_rng"]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "arguments": vars(args) | {"run_dir": str(args.run_dir), "resume": str(args.resume or "")},
        "model": model_config.to_dict(),
        "vapo": asdict(vapo_config),
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    writer = SummaryWriter(args.run_dir / "tensorboard")
    started = time.monotonic()

    initial_checkpoint = args.run_dir / "checkpoint-000000.pt"
    if iteration == 0 and not initial_checkpoint.exists():
        save_checkpoint(
            initial_checkpoint,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            model_config=model_config,
            vapo_config=vapo_config,
            iteration=0,
            next_seed=next_seed,
            metrics={"iteration": 0},
            training_rng_state=generator.bit_generator.state,
        )

    while iteration < args.iterations:
        if args.max_hours and (time.monotonic() - started) / 3600.0 >= args.max_hours:
            break
        self_play = collect_self_play(
            actor,
            critic,
            games=args.games,
            seed_start=next_seed,
            episode_steps=args.episode_steps,
            temperature=args.temperature,
            sampling_seed=args.seed + iteration * 1_000_003,
        )
        next_seed += args.games
        self_play_diagnostics = {
            f"self_play_{name}": value for name, value in rollout_diagnostics(self_play).items()
        }
        rollout_parts = [self_play]
        opponent_checkpoint = ""
        league_diagnostics = {}
        candidates = sorted(args.run_dir.glob("checkpoint-*.pt"))[-args.league_pool_size :]
        if args.league_games and candidates:
            selected = candidates[int(generator.integers(0, len(candidates)))]
            opponent_payload = torch.load(selected, map_location=device, weights_only=False)
            if opponent_payload["model_config"] != model_config.to_dict():
                raise ValueError(f"league checkpoint has incompatible model: {selected}")
            opponent_actor.load_state_dict(opponent_payload["actor"])
            league = collect_frozen_opponent_play(
                actor,
                critic,
                opponent_actor,
                games=args.league_games,
                seed_start=next_seed,
                episode_steps=args.episode_steps,
                temperature=args.temperature,
                opponent_temperature=args.opponent_temperature,
                sampling_seed=args.seed + iteration * 1_000_003 + 17,
            )
            next_seed += args.league_games
            league_diagnostics = {
                f"league_{name}": value for name, value in rollout_diagnostics(league).items()
            }
            rollout_parts.append(league)
            opponent_checkpoint = selected.name
        rollout = concatenate_rollouts(rollout_parts)
        del rollout_parts, self_play
        if args.league_games and candidates:
            del league
        update_metrics = update_vapo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            vapo_config,
            generator=generator,
        )
        iteration += 1
        metrics = {
            "iteration": iteration,
            "next_seed": next_seed,
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            "league_checkpoint": opponent_checkpoint,
            **rollout_diagnostics(rollout),
            **self_play_diagnostics,
            **league_diagnostics,
            **update_metrics,
        }
        if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError(f"non-finite training metric: {metrics}")
        append_jsonl(args.run_dir / "metrics.jsonl", metrics)
        for name, value in metrics.items():
            if isinstance(value, int | float):
                writer.add_scalar(name, value, iteration)
        writer.flush()
        print(json.dumps(metrics, sort_keys=True), flush=True)
        save_checkpoint(
            args.run_dir / "latest.pt",
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            model_config=model_config,
            vapo_config=vapo_config,
            iteration=iteration,
            next_seed=next_seed,
            metrics=metrics,
            training_rng_state=generator.bit_generator.state,
        )
        if iteration % args.checkpoint_every == 0:
            save_checkpoint(
                args.run_dir / f"checkpoint-{iteration:06d}.pt",
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                model_config=model_config,
                vapo_config=vapo_config,
                iteration=iteration,
                next_seed=next_seed,
                metrics=metrics,
                training_rng_state=generator.bit_generator.state,
            )
        del rollout
    writer.close()


if __name__ == "__main__":
    main()
