#!/usr/bin/env python3
"""Profile the complete compiled production PPO update on fresh native waves.

Run only through MLQ. The first --updates call is cold (including compilation),
remaining calls are unprofiled steady updates. One additional fresh wave is
updated under CPU+CUDA profiling; its instrumented wall time is NOT throughput.
The Chrome trace is always exported beside --output. --kernels additionally
includes leaf CUDA kernel and CPU self-time tables in the report.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._dynamo
from benchmark_ppo_iteration import _hardware_identity, _verify_first_step_critic_state
from torch.profiler import ProfilerActivity, profile

from kaggriculture.inference import load_actor_artifact
from kaggriculture.model import parameter_count
from kaggriculture.modelargs import actor_model_config
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    MAX_VALUE_TARGET_SATURATED_FRACTION,
    UPDATE_COMPILE_MODES,
    PpoConfig,
    _fixed_minibatch_positions,
    make_optimizers,
    make_structured_dynamics_optimizer,
    replay_behavior_values,
    update_ppo,
    update_replay_parity,
)
from kaggriculture.production import (
    PRODUCTION_ARCHITECTURE,
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_TEMPERATURE,
    PRODUCTION_UPDATE_COMPILE_MODE,
    production_model_config,
    production_ppo_config,
)
from kaggriculture.provenance import file_sha256, source_identity
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import allocate_rollout_storage, collect_mixed_play_rust
from kaggriculture.structured_dynamics import StructuredCriticDynamics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=PRODUCTION_SELF_PLAY_GAMES)
    parser.add_argument("--league-games", type=int, default=PRODUCTION_LEAGUE_GAMES)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--init-actor-from", type=Path, required=True)
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=production_ppo_config(update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE)[
            "minibatch_size"
        ],
    )
    parser.add_argument(
        "--update-compile-mode",
        choices=[mode for mode in UPDATE_COMPILE_MODES if mode != "eager"],
        default=PRODUCTION_UPDATE_COMPILE_MODE,
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=2,
        help="unprofiled full updates, including one cold call; minimum two",
    )
    parser.add_argument("--kernels", action="store_true", help="include kernel and CPU tables")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _save_report(path: Path, report: dict[str, Any]) -> None:
    # Some diagnostic correlations are undefined; preserve standards-compliant JSON.
    def finite(value):
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(finite(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _memory() -> dict[str, int]:
    stats = torch.cuda.memory_stats()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocator_retries": stats.get("num_alloc_retries", 0),
        "allocator_ooms": stats.get("num_ooms", 0),
    }


def kernel_table(prof, trace_path: Path, top: int) -> dict[str, Any]:
    """Count actual Kineto kernel activities, never nested CPU device attribution."""
    trace = json.loads(trace_path.read_text())
    kernels: dict[str, dict[str, Any]] = {}
    transfers: dict[str, dict[str, Any]] = {}
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category == "kernel":
            table = kernels
        elif category in ("gpu_memcpy", "gpu_memset"):
            table = transfers
        else:
            continue
        row = table.setdefault(event["name"], {"name": event["name"], "count": 0, "total_ms": 0.0})
        row["count"] += 1
        row["total_ms"] += event["dur"] / 1000.0
    if not kernels:
        raise RuntimeError("profiler captured no CUDA kernel activities; attribution is invalid")
    total = sum(row["total_ms"] for row in kernels.values())
    ordered = sorted(kernels.values(), key=lambda row: row["total_ms"], reverse=True)
    for row in ordered:
        row["share"] = row["total_ms"] / total
    cpu = sorted(
        (event for event in prof.key_averages() if event.device_type.name == "CPU"),
        key=lambda event: event.self_cpu_time_total,
        reverse=True,
    )
    return {
        "accounting": "Chrome trace cat=kernel complete CUDA activities; overlapping streams sum",
        "kernel_device_total_ms": total,
        "kernel_launches": sum(row["count"] for row in ordered),
        "unique_kernel_names": len(ordered),
        "kernels": ordered[:top],
        "transfers_and_memsets": sorted(
            transfers.values(), key=lambda row: row["total_ms"], reverse=True
        )[:top],
        "cpu_accounting": "exclusive CPU self times; thread overlap is not wall time",
        "cpu_self_total_ms": sum(event.self_cpu_time_total for event in cpu) / 1000.0,
        "cpu_overhead": [
            {"name": event.key, "count": event.count, "self_ms": event.self_cpu_time_total / 1000.0}
            for event in cpu[:top]
        ],
    }


def _work_counts(metrics, rollout, config) -> dict[str, Any]:
    _, counts = _fixed_minibatch_positions(int(metrics["states"]), config.minibatch_size)
    actor_steps = int(metrics["actor_updates"])
    # The KL-rejected candidate still executes a forward/backward, but no step.
    actor_batches = actor_steps + int(metrics["kl_early_stop"])
    critic_steps = int(metrics["updates"])
    actor_row_counts = np.tile(counts, config.epochs)
    replay_chunk = inspect.signature(replay_behavior_values).parameters["chunk_size"].default
    replay_rows = rollout.trajectories * rollout.horizon
    return {
        "basis": "update_ppo counters plus its fixed padded minibatch partition",
        "valid_states": int(metrics["states"]),
        "minibatches_per_epoch": len(counts),
        "valid_rows_per_minibatch": counts.tolist(),
        "compiled_rows_per_minibatch": config.minibatch_size,
        "actor_minibatches_intended": int(metrics["actor_minibatches_intended"]),
        "actor_forward_backward_minibatches": actor_batches,
        "actor_optimizer_steps": actor_steps,
        "actor_optimizer_valid_rows": int(actor_row_counts[:actor_steps].sum()),
        "actor_forward_backward_padded_rows": actor_batches * config.minibatch_size,
        "critic_forward_backward_minibatches": critic_steps,
        "critic_optimizer_steps": critic_steps,
        "critic_optimizer_valid_rows": int(metrics["states"]) * int(metrics["epochs"]),
        "critic_forward_backward_padded_rows": critic_steps * config.minibatch_size,
        "actor_predictor_steps": 0,
        "critic_predictor_steps": int(metrics["structured_critic_predictor_updates"]),
        "critic_joint_auxiliary_updates": int(metrics["structured_critic_auxiliary_updates"]),
        "behavior_value_replay_rows": replay_rows,
        "behavior_value_replay_chunk_size": replay_chunk,
        "behavior_value_replay_forward_chunks": math.ceil(replay_rows / replay_chunk),
        "behavior_value_replay_padded_rows": math.ceil(replay_rows / replay_chunk) * replay_chunk,
        "kl_early_stop": bool(metrics["kl_early_stop"]),
    }


def _run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.updates < 2 or args.minibatch_size < 1 or args.top < 1:
        raise ValueError("require --updates >= 2, positive --minibatch-size and --top")
    if (args.games, args.league_games) != (PRODUCTION_SELF_PLAY_GAMES, PRODUCTION_LEAGUE_GAMES):
        raise ValueError("this profiler requires the full production self-play/league geometry")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 is required; no CPU or FP32 fallback")
    if torch._dynamo.config.disable:
        raise RuntimeError("TorchDynamo is disabled; compiled execution is required")
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.reset()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    architecture = resolve_architecture(PRODUCTION_ARCHITECTURE)
    model_config = architecture.build_config(production_model_config())
    config = PpoConfig(
        **{
            **production_ppo_config(update_compile_mode=args.update_compile_mode),
            "minibatch_size": args.minibatch_size,
        }
    )
    if not config.use_bfloat16 or config.structured_actor_auxiliary_active:
        raise ValueError("production must use BF16 with actor NextLat off")
    if (
        config.structured_critic_latent_coefficient,
        config.structured_critic_value_coefficient,
    ) != (1.0, 1.0):
        raise ValueError("production critic NextLat must remain enabled at 1/1")
    if PRODUCTION_ROLLOUT_FORWARD_MODE != "inductor_graph":
        raise ValueError("compiled CUDA-graph rollout is required")
    report.update(
        {
            "source_identity": source_identity(),
            "hardware": _hardware_identity(device),
            "torch": torch.__version__,
            "initial_actor_sha256": file_sha256(args.init_actor_from),
            "architecture": architecture.name,
            "model": model_config.to_dict(),
            "ppo": asdict(config),
            "rollout_forward_mode": PRODUCTION_ROLLOUT_FORWARD_MODE,
            "rollout_bfloat16": True,
            "episode_steps": PRODUCTION_EPISODE_STEPS,
            "temperature": PRODUCTION_TEMPERATURE,
            "precision_note": (
                "BF16 compute; production FP32 parameters, heads and reductions retained"
            ),
            "phase": "constructing_cuda_models",
        }
    )
    _save_report(args.output, report)
    # Even constructors inside the artifact helper allocate directly on CUDA.
    # CPU artifact deserialization/state snapshots are data, never CPU models.
    with torch.device(device):
        pretrained, payload = load_actor_artifact(args.init_actor_from, device=device)
        expected = actor_model_config(model_config)
        actual = actor_model_config(pretrained.config)
        if not isinstance(pretrained, architecture.actor_class) or actual != expected:
            raise ValueError("initial actor architecture/configuration does not match production")
        actor = architecture.actor_class(model_config)
        actor.load_state_dict(pretrained.state_dict())
        del pretrained, payload
        critic = architecture.critic_class(model_config)
        critic_dynamics = StructuredCriticDynamics(model_config)
    for module in (actor, critic, critic_dynamics):
        if any(
            tensor.device.type != "cuda" for tensor in (*module.parameters(), *module.buffers())
        ):
            raise RuntimeError("model construction left a non-CUDA parameter or buffer")
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    critic_dynamics_optimizer = make_structured_dynamics_optimizer(critic_dynamics, config)
    frozen_state = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    opponent_count = PRODUCTION_LEAGUE_ACTIVE_OPPONENTS + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS
    generator = np.random.default_rng(args.seed)
    auxiliary_generator = np.random.default_rng(args.seed + 2)
    arena = allocate_rollout_storage(
        architecture.name,
        args.games * 2 + args.league_games,
        PRODUCTION_EPISODE_STEPS - 1,
        pin_memory=True,
    )
    report.update(
        {
            "actor_parameters": parameter_count(actor),
            "critic_parameters": parameter_count(critic),
            "critic_predictor_parameters": parameter_count(critic_dynamics),
            "actor_predictor_parameters": 0,
            "frozen_opponents": opponent_count,
            "opponent_policy": "eight frozen copies of initial BC actor, reconstructed each wave",
            "fresh_wave_policy": "new native seeds and current learner weights for every update",
            "diagnostic_gradients": False,
        }
    )
    for index in range(args.updates + 1):
        instrumented = index == args.updates
        kind = (
            "profiled_update"
            if instrumented
            else ("cold_update" if index == 0 else "steady_update")
        )
        seed_start = args.seed + index * (args.games + args.league_games)
        record: dict[str, Any] = {
            "index": index,
            "kind": kind,
            "seed_start": seed_start,
            "instrumented": instrumented,
        }
        report["updates"].append(record)
        report["phase"] = f"{kind}:collecting_wave"
        _save_report(args.output, report)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.device(device):
            opponents = [architecture.actor_class(model_config) for _ in range(opponent_count)]
        for opponent in opponents:
            opponent.load_state_dict(frozen_state)
            opponent.requires_grad_(False)
        assignments = np.arange(args.league_games, dtype=np.int64) % opponent_count
        generator.shuffle(assignments)
        torch.cuda.synchronize()
        record["opponent_setup_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        rollout = collect_mixed_play_rust(
            actor,
            opponents,
            self_play_games=args.games,
            league_games=args.league_games,
            opponent_indices=assignments,
            seed_start=seed_start,
            episode_steps=PRODUCTION_EPISODE_STEPS,
            temperature=PRODUCTION_TEMPERATURE,
            opponent_temperature=PRODUCTION_TEMPERATURE,
            sampling_seed=int(generator.integers(0, np.iinfo(np.int64).max)),
            forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
            forward_autocast=True,
            storage=arena,
        )
        torch.cuda.synchronize()
        record["rollout_seconds"] = time.perf_counter() - started
        del opponent, opponents
        record["wave_memory"] = _memory()
        record.update(
            {
                "learner_states": rollout.state_count,
                "trajectories": rollout.trajectories,
                "transitions_per_trajectory": rollout.horizon,
            }
        )
        expected_states = (args.games * 2 + args.league_games) * (PRODUCTION_EPISODE_STEPS - 1)
        if rollout.state_count != expected_states:
            raise RuntimeError("native collection did not produce the complete production wave")
        report["phase"] = f"{kind}:replay_audit_outside_update"
        _save_report(args.output, report)
        started = time.perf_counter()
        parity = update_replay_parity(
            actor,
            rollout,
            minibatch_size=config.minibatch_size,
            compile_mode=config.update_compile_mode,
            autocast_enabled=True,
        )
        _verify_first_step_critic_state(rollout, args.games, seed_start)
        torch.cuda.synchronize()
        record["replay_audit_seconds"] = time.perf_counter() - started
        record["replay_checks"] = parity
        for component in ("unit", "kind", "quantity"):
            if parity[f"update_replay_{component}_active_count"] < 1:
                raise RuntimeError(f"replay audit saw no active {component} components")
        for key, limit in (
            ("update_replay_max_kl", MAX_UPDATE_REPLAY_KL),
            ("update_replay_max_tail_fraction", MAX_UPDATE_REPLAY_TAIL_FRACTION),
        ):
            if not math.isfinite(parity[key]) or parity[key] > limit:
                raise RuntimeError(f"replay audit failed {key}: {parity[key]} > {limit}")
        torch.cuda.reset_peak_memory_stats()
        record["memory_before_update"] = _memory()
        report["phase"] = f"{kind}:update_ppo"
        _save_report(args.output, report)

        def run_update(rollout=rollout, record=record):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start.record()
            metrics = update_ppo(
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                rollout,
                config,
                generator=generator,
                structured_actor_auxiliary=False,
                structured_critic_dynamics=critic_dynamics,
                structured_critic_dynamics_optimizer=critic_dynamics_optimizer,
                structured_critic_auxiliary=True,
                auxiliary_generator=auxiliary_generator,
                diagnostic_groups={
                    "self_play": np.arange(rollout.trajectories) < args.games * 2,
                    "league": np.arange(rollout.trajectories) >= args.games * 2,
                },
                diagnostic_gradients=False,
            )
            end.record()
            torch.cuda.synchronize()
            record["update_wall_seconds"] = time.perf_counter() - started
            record["update_cuda_span_ms"] = start.elapsed_time(end)
            record["update_metrics"] = metrics

        if instrumented:
            trace_path = args.output.with_suffix(".trace.json")
            record["trace"] = str(trace_path.resolve())
            record["timing_note"] = "instrumented CUDA/CPU spans; NOT throughput evidence"
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            try:
                with prof:
                    run_update()
            finally:
                prof.export_chrome_trace(str(trace_path))
        else:
            run_update()
        metrics = record["update_metrics"]
        record["work_counts"] = _work_counts(metrics, rollout, config)
        record["memory_after_update"] = _memory()
        record["allocator_retries_during_update"] = (
            record["memory_after_update"]["allocator_retries"]
            - record["memory_before_update"]["allocator_retries"]
        )
        _save_report(args.output, report)
        if instrumented:
            record["attribution"] = kernel_table(prof, trace_path, args.top if args.kernels else 0)
            del prof
        for key, limit in (
            ("first_minibatch_component_kl", MAX_FIRST_MINIBATCH_KL),
            ("value_target_saturated_fraction", MAX_VALUE_TARGET_SATURATED_FRACTION),
        ):
            if not math.isfinite(metrics[key]) or metrics[key] > limit:
                raise RuntimeError(f"update failed {key}: {metrics[key]} > {limit}")
        if (
            metrics["actor_updates"] < 1
            or metrics["structured_critic_predictor_updates"] != metrics["updates"]
        ):
            raise RuntimeError(
                "update failed to execute actor and every joint critic predictor step"
            )
        _save_report(args.output, report)
        print(
            json.dumps(
                {
                    "event": kind,
                    "index": index,
                    "update_wall_seconds": record["update_wall_seconds"],
                    "work_counts": record["work_counts"],
                }
            ),
            flush=True,
        )
        del rollout
        gc.collect()
    steady = [
        item["update_wall_seconds"] for item in report["updates"] if item["kind"] == "steady_update"
    ]
    report["unprofiled_summary"] = {
        "cold_update_seconds": report["updates"][0]["update_wall_seconds"],
        "steady_update_seconds": steady,
        "steady_update_seconds_median": statistics.median(steady),
        "note": "fresh waves; replay audit and profiler excluded; not a six-repeat calibration",
    }


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running",
        "phase": "initializing",
        "updates": [],
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "profiler_options": {
            "activities": ["CPU", "CUDA"],
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        },
    }
    _save_report(args.output, report)
    try:
        _run(args, report)
    except BaseException as error:
        report.update(
            {
                "status": "error",
                "error": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            }
        )
        if torch.cuda.is_initialized():
            try:
                report["failure_memory"] = _memory()
            except Exception as memory_error:
                report["failure_memory_error"] = str(memory_error)
        _save_report(args.output, report)
        raise
    report.update({"status": "complete", "phase": "complete"})
    _save_report(args.output, report)
    print(json.dumps({"event": "done", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
