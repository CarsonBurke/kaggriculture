#!/usr/bin/env python3
"""Benchmark one complete native rollout plus VAPO replay at realistic batch sizes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig, parameter_count
from kaggriculture.provenance import source_identity
from kaggriculture.rollout import allocate_rollout_storage, collect_mixed_play_rust
from kaggriculture.rust_env import load_native
from kaggriculture.telemetry import TensorboardMirror
from kaggriculture.training import rollout_diagnostics
from kaggriculture.vapo import VapoConfig, make_optimizers, update_replay_parity, update_vapo

_REPORT_PATH: Path | None = None
_REPORT_LINES: list[str] = []
_REPORT_MIRROR: TensorboardMirror | None = None
_REPORT_TENSORBOARD_DIR: Path | None = None
PRODUCTION_EPISODE_STEPS = 720
PRODUCTION_ACTIVE_OPPONENTS = 2


def _configure_report(path: Path | None, tensorboard_dir: Path | None = None) -> None:
    global _REPORT_MIRROR, _REPORT_PATH, _REPORT_TENSORBOARD_DIR
    if _REPORT_MIRROR is not None:
        _REPORT_MIRROR.close()
        _REPORT_MIRROR = None
    _REPORT_PATH = None if path is None else path.expanduser().resolve()
    if tensorboard_dir is not None and _REPORT_PATH is None:
        raise ValueError("TensorBoard output requires a JSONL report path")
    if _REPORT_PATH is not None:
        default = _REPORT_PATH.parent / "tensorboard" / _REPORT_PATH.stem
        selected = default if tensorboard_dir is None else tensorboard_dir
        _REPORT_TENSORBOARD_DIR = selected.expanduser().resolve()
    else:
        _REPORT_TENSORBOARD_DIR = None
    _REPORT_LINES.clear()


def emit(payload: dict) -> None:
    """Write one finite, standards-compliant JSONL record."""

    global _REPORT_MIRROR

    def normalize(value, path: str):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite benchmark metric at {path}: {value}")
        if isinstance(value, dict):
            return {key: normalize(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [normalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    rendered = json.dumps(normalize(payload, "payload"), sort_keys=True, allow_nan=False)
    print(rendered, flush=True)
    if _REPORT_PATH is None:
        return
    _REPORT_LINES.append(rendered)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{_REPORT_PATH.name}.", suffix=".tmp", dir=_REPORT_PATH.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(_REPORT_LINES) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, _REPORT_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    if _REPORT_MIRROR is None:
        assert _REPORT_TENSORBOARD_DIR is not None
        _REPORT_MIRROR = TensorboardMirror(_REPORT_PATH, _REPORT_TENSORBOARD_DIR)
    else:
        _REPORT_MIRROR.record(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        default="64,112,128,256",
        help="comma-separated self-play games per iteration (each yields two trajectories)",
    )
    parser.add_argument(
        "--league-games",
        type=int,
        default=96,
        help="frozen-opponent games per iteration (each yields one learner trajectory)",
    )
    parser.add_argument(
        "--league-opponents",
        type=int,
        default=5,
        help="separate frozen actor forwards, matching initial + active + historical production",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="full rollout+update iterations per size; first is cold, later repeats are steady",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cnn-width", type=int, default=48)
    parser.add_argument("--cnn-blocks", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--transformer-layers", type=int, default=7)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--quantity-rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opponent-temperature", type=float, default=0.8)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--max-update-replay-error",
        type=float,
        default=1e-3,
        help=(
            "maximum unchanged-policy importance-ratio drift through the exact "
            "update-path forward; CUDA kernels are batch-shape stable only to "
            "roughly 1e-4, so this gates compiled/graphed update variants"
        ),
    )
    parser.add_argument("--compile-models", action="store_true")
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _verify_first_step_critic_features(rollout, self_play_games: int, seed_start: int) -> None:
    """Cross-validate stored critic features against a fresh native encode.

    Behavior values are replayed solely from stored `critic_features`, which
    nothing else consumes during collection. Recomputing the opening step from
    a fresh BatchEnv and demanding bitwise fp16 equality catches arena-layout
    or staging corruption across the whole merged wave at production scale.
    """
    league_games = rollout.trajectories - self_play_games * 2
    seeds = np.arange(seed_start, seed_start + self_play_games + league_games, dtype=np.uint64)
    fresh = np.asarray(load_native().BatchEnv(seeds).encoded()["critic_features"])
    self_rows = np.arange(self_play_games * 2, dtype=np.int64)
    league_seats = rollout.seats[self_play_games * 2 :].astype(np.int64)
    league_rows = 2 * (self_play_games + np.arange(league_games, dtype=np.int64)) + league_seats
    stored_rows = np.concatenate([self_rows, league_rows])
    if not np.array_equal(rollout.critic_features[:, 0], fresh[stored_rows].astype(np.float16)):
        raise RuntimeError("stored first-step critic features do not match a fresh native encode")


def _hardware_identity(device: torch.device) -> dict[str, object]:
    """Return stable runtime hardware metadata used to match benchmark modes."""
    identity: dict[str, object] = {
        "device_type": device.type,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type != "cuda":
        return identity
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    identity.update(
        {
            "device_index": index,
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    )
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is not None:
        identity["device_uuid"] = str(device_uuid)
    return identity


def _completion_record(game_counts: list[int], repeats: int) -> dict[str, object]:
    """Describe the exact batch/repeat Cartesian product completed by this run."""
    return {
        "event": "benchmark_complete",
        "completed": True,
        "self_play_game_counts": game_counts,
        "repeats": repeats,
        "completed_batches": [
            {
                "self_play_games": games,
                "completed_repeats": list(range(repeats)),
            }
            for games in game_counts
        ],
        "iteration_records": len(game_counts) * repeats,
        "batch_summaries": len(game_counts),
    }


def main() -> None:
    args = parse_args()
    _configure_report(args.output, args.tensorboard_dir)
    game_counts = [int(value) for value in args.games.split(",")]
    if not game_counts or any(value < 1 for value in game_counts):
        raise ValueError("--games must be a comma-separated list of positive integers")
    if len(set(game_counts)) != len(game_counts):
        raise ValueError("--games cannot contain duplicate batch sizes")
    if args.league_games < 0:
        raise ValueError("--league-games cannot be negative")
    if args.league_opponents < 1:
        raise ValueError("--league-opponents must be positive")
    if args.league_games and args.league_opponents > args.league_games:
        raise ValueError("--league-games must cover every --league-opponents policy")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least two to separate cold and steady iterations")
    if args.epochs < 1 or args.minibatch_size < 1:
        raise ValueError("epochs and minibatch size must be positive")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (args.temperature, args.opponent_temperature)
    ):
        raise ValueError("temperatures must be finite and positive")
    if args.temperature != 1.0:
        raise ValueError("on-policy VAPO benchmarking requires --temperature 1.0")
    if (
        not math.isfinite(args.max_update_replay_error)
        or args.max_update_replay_error <= 0.0
        or args.max_update_replay_error > 1e-3
    ):
        raise ValueError("--max-update-replay-error must be finite, positive, and at most 1e-3")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    model_config = ModelConfig(
        cnn_width=args.cnn_width,
        cnn_blocks=args.cnn_blocks,
        model_dim=args.model_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        ffn_multiplier=args.ffn_multiplier,
        quantity_rank=args.quantity_rank,
    )
    vapo_config = VapoConfig(
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
        use_bfloat16=not args.no_bfloat16,
        compile_update=args.compile_models,
    )
    identity = source_identity()
    emit(
        {
            "event": "configuration",
            "device": str(device),
            "hardware": _hardware_identity(device),
            "self_play_game_counts": game_counts,
            "league_games_per_iteration": args.league_games,
            "league_opponents": args.league_opponents,
            "league_initial_opponents": min(args.league_opponents, 1),
            "league_active_opponents": min(
                PRODUCTION_ACTIVE_OPPONENTS,
                max(args.league_opponents - 1, 0),
            ),
            "league_historical_opponents": max(
                args.league_opponents - 1 - PRODUCTION_ACTIVE_OPPONENTS,
                0,
            ),
            "episode_steps": PRODUCTION_EPISODE_STEPS,
            "physical_games_per_iteration": [games + args.league_games for games in game_counts],
            "repeats": args.repeats,
            "seed": args.seed,
            "source_digest": identity["sha256"],
            "source_identity": identity,
            "temperature": args.temperature,
            "opponent_temperature": args.opponent_temperature,
            "max_update_replay_error": args.max_update_replay_error,
            "precision": {
                "use_bfloat16": vapo_config.use_bfloat16,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "model": model_config.to_dict(),
            "vapo": asdict(vapo_config),
            "compile_models": args.compile_models,
            "torch": torch.__version__,
        }
    )

    for self_play_games in game_counts:
        # Keep initial parameters identical across batch sizes so throughput
        # comparisons are not confounded by a different policy/action mix.
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.empty_cache()
        actor = FarmActor(model_config).to(device)
        critic = DistributionalCritic(model_config).to(device)
        frozen_opponent_state = {
            name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
        }
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, vapo_config)
        generator = np.random.default_rng(args.seed)
        seed_cursor = args.seed
        physical_games = self_play_games + args.league_games
        # Production collects the whole mixed wave into one reusable pinned
        # arena; mirror that here so staging behavior matches training.
        arena = allocate_rollout_storage(
            self_play_games * 2 + args.league_games,
            PRODUCTION_EPISODE_STEPS - 1,
            pin_memory=device.type == "cuda",
        )
        repeat_payloads = []

        for repeat in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            iteration_started = time.perf_counter()

            opponents = []
            assignments = None
            opponent_temperatures = None
            deterministic_opponents = None
            if args.league_games:
                # Production reconstructs selected frozen actors from archive
                # snapshots on every iteration. Keep that object lifecycle in
                # the benchmark: compiled current actor/critic graphs persist,
                # while frozen-policy wrappers are fresh each repeat.
                opponents = [
                    FarmActor(model_config).to(device) for _ in range(args.league_opponents)
                ]
                for opponent in opponents:
                    opponent.load_state_dict(frozen_opponent_state)
                    opponent.requires_grad_(False)
                assignments = np.arange(args.league_games, dtype=np.int64) % len(opponents)
                generator.shuffle(assignments)
                deterministic_opponents = np.ones(len(opponents), dtype=np.bool_)
                deterministic_opponents[1:3] = False
                opponent_temperatures = np.ones(len(opponents), dtype=np.float32)
                opponent_temperatures[1:3] = args.opponent_temperature
            wave_seed_start = seed_cursor
            rollout = collect_mixed_play_rust(
                actor,
                opponents,
                self_play_games=self_play_games,
                league_games=args.league_games,
                opponent_indices=assignments,
                seed_start=wave_seed_start,
                episode_steps=PRODUCTION_EPISODE_STEPS,
                temperature=args.temperature,
                opponent_temperature=args.opponent_temperature,
                opponent_temperatures=opponent_temperatures,
                deterministic_opponents=deterministic_opponents,
                sampling_seed=int(generator.integers(0, np.iinfo(np.int64).max)),
                compile_models=args.compile_models,
                storage=arena,
            )
            seed_cursor += physical_games
            _synchronize(device)
            rollout_seconds = time.perf_counter() - iteration_started
            del opponents

            # Gate the update-path importance ratios at unchanged weights
            # before the update mutates the actor. This bounds the numeric
            # drift any update-path change (compilation, CUDA graphs, larger
            # minibatches) injects into clipping and the KL trust region.
            parity_started = time.perf_counter()
            parity = update_replay_parity(
                actor,
                rollout,
                minibatch_size=vapo_config.minibatch_size,
                compile_model=vapo_config.compile_update,
            )
            _synchronize(device)
            parity_seconds = time.perf_counter() - parity_started
            for component in ("unit", "kind", "quantity"):
                if parity[f"update_replay_{component}_active_count"] < 1:
                    raise RuntimeError(f"update replay parity saw no active {component} components")
            if parity["update_replay_max_ratio_error"] > args.max_update_replay_error:
                raise RuntimeError(
                    "update-path importance ratios drifted beyond "
                    f"{args.max_update_replay_error}: {parity['update_replay_max_ratio_error']}"
                )
            _verify_first_step_critic_features(rollout, self_play_games, wave_seed_start)

            update_started = time.perf_counter()
            update_metrics = update_vapo(
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                rollout,
                vapo_config,
                generator=generator,
            )
            _synchronize(device)
            update_seconds = time.perf_counter() - update_started
            if int(update_metrics["actor_updates"]) < 1:
                raise RuntimeError("benchmark iteration completed without an actor update")
            # The parity gate is benchmark-only instrumentation; production
            # iterations are rollout plus update, so the calibration decision
            # must be based on exactly that.
            total_seconds = rollout_seconds + update_seconds
            diagnostics = rollout_diagnostics(rollout)
            payload = {
                "event": "iteration",
                "phase": "cold_start" if repeat == 0 else "steady_state",
                "repeat": repeat,
                "self_play_games": self_play_games,
                "league_games": args.league_games,
                "physical_games": physical_games,
                "learner_trajectories": rollout.trajectories,
                "learner_states": rollout.states,
                "rollout_seconds": rollout_seconds,
                "update_replay_parity_seconds": parity_seconds,
                "update_seconds": update_seconds,
                "total_seconds": total_seconds,
                "physical_games_per_rollout_second": physical_games / rollout_seconds,
                "learner_states_per_rollout_second": rollout.states / rollout_seconds,
                "critic_replayed_states_per_second": (
                    rollout.states * vapo_config.epochs / update_seconds
                ),
                "iterations_per_hour": 3600.0 / total_seconds,
                "peak_cuda_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
                "process_lifetime_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "actor_parameters": parameter_count(actor),
                "critic_parameters": parameter_count(critic),
                "money_mean": diagnostics["money_mean"],
                "tie_fraction": diagnostics["tie_fraction"],
                **parity,
                **update_metrics,
            }
            emit(payload)
            repeat_payloads.append(payload)
            del rollout
            gc.collect()

        steady = repeat_payloads[1:]
        emit(
            {
                "event": "batch_summary",
                "self_play_games": self_play_games,
                "league_games": args.league_games,
                "physical_games": physical_games,
                "cold_total_seconds": repeat_payloads[0]["total_seconds"],
                "cold_iterations_per_hour": repeat_payloads[0]["iterations_per_hour"],
                "cold_physical_games_per_rollout_second": repeat_payloads[0][
                    "physical_games_per_rollout_second"
                ],
                "steady_total_seconds_median": statistics.median(
                    item["total_seconds"] for item in steady
                ),
                "steady_iterations_per_hour_median": statistics.median(
                    item["iterations_per_hour"] for item in steady
                ),
                "steady_physical_games_per_rollout_second_median": statistics.median(
                    item["physical_games_per_rollout_second"] for item in steady
                ),
                "steady_critic_replayed_states_per_second_median": statistics.median(
                    item["critic_replayed_states_per_second"] for item in steady
                ),
            }
        )
        del actor, critic, actor_optimizer, critic_optimizer, frozen_opponent_state, arena
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    emit(_completion_record(game_counts, args.repeats))
    _configure_report(None)


if __name__ == "__main__":
    main()
