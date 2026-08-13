#!/usr/bin/env python3
"""Benchmark complete native model-to-rollout collection at several batch sizes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig, parameter_count
from kaggriculture.policy import component_logprobs
from kaggriculture.provenance import source_identity
from kaggriculture.rollout import collect_self_play_rust
from kaggriculture.training import rollout_diagnostics

_REPORT_PATH: Path | None = None
_REPORT_LINES: list[str] = []


def _configure_report(path: Path | None) -> None:
    global _REPORT_PATH
    _REPORT_PATH = None if path is None else path.expanduser().resolve()
    _REPORT_LINES.clear()


def emit(payload: dict) -> None:
    """Write one strict, machine-readable JSONL record."""
    rendered = json.dumps(payload, sort_keys=True, allow_nan=False)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="16,32,64,128")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--query-features", type=int, default=24)
    parser.add_argument("--compile-models", action="store_true")
    parser.add_argument("--replay-minibatch-size", type=int, default=2048)
    parser.add_argument("--max-replay-error", type=float, default=5e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def replay_diagnostics(
    actor: FarmActor,
    critic: DistributionalCritic,
    rollout,
    minibatch_size: int,
) -> dict[str, float | int]:
    """Measure behavior/eager-replay drift over every stored policy factor."""
    device = next(actor.parameters()).device
    flat_valid = rollout.valid.reshape(-1)
    indices = np.flatnonzero(flat_valid)

    def flattened(name: str) -> np.ndarray:
        values = getattr(rollout, name)
        return values.reshape(-1, *values.shape[2:])

    def tensor(name: str, dtype: torch.dtype, row_indices: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(flattened(name)[row_indices], device=device, dtype=dtype)

    maximum_logprob_error = {"unit": 0.0, "kind": 0.0, "quantity": 0.0}
    maximum_ratio_error = {"unit": 0.0, "kind": 0.0, "quantity": 0.0}
    active_counts = {"unit": 0, "kind": 0, "quantity": 0}
    maximum_value_error = 0.0
    for start in range(0, indices.size, minibatch_size):
        selected = indices[start : start + minibatch_size]
        board = tensor("board", torch.float32, selected)
        output = actor(
            board,
            tensor("global_features", torch.float32, selected),
            tensor("units", torch.float32, selected),
            tensor("unit_positions", torch.long, selected),
        )
        market_kinds = tensor("market_kinds", torch.long, selected)
        replayed = component_logprobs(
            output,
            actor.quantity_logits(output.market_quantity_context, market_kinds),
            tensor("unit_actions", torch.long, selected),
            market_kinds,
            tensor("market_quantities", torch.long, selected),
            tensor("unit_masks", torch.bool, selected),
            tensor("market_kind_masks", torch.bool, selected),
            tensor("market_quantity_masks", torch.bool, selected),
        )[:3]
        for name, new, old_name, active_name in (
            ("unit", replayed[0], "old_unit_logprobs", "unit_active"),
            ("kind", replayed[1], "old_market_kind_logprobs", "market_active"),
            (
                "quantity",
                replayed[2],
                "old_market_quantity_logprobs",
                "market_quantity_active",
            ),
        ):
            active = tensor(active_name, torch.bool, selected)
            active_counts[name] += int(active.sum())
            if not bool(active.any()):
                continue
            difference = new[active] - tensor(old_name, torch.float32, selected)[active]
            if not bool(torch.isfinite(difference).all()):
                raise FloatingPointError(f"non-finite {name} replay difference")
            maximum_logprob_error[name] = max(
                maximum_logprob_error[name], float(difference.abs().max())
            )
            maximum_ratio_error[name] = max(
                maximum_ratio_error[name], float((difference.exp() - 1.0).abs().max())
            )
        values = critic.value(critic(board, tensor("critic_features", torch.float32, selected)))
        value_difference = values - torch.as_tensor(
            rollout.old_values.reshape(-1)[selected],
            device=device,
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(value_difference).all()):
            raise FloatingPointError("non-finite critic replay difference")
        maximum_value_error = max(maximum_value_error, float(value_difference.abs().max()))

    return {
        **{
            f"replay_{name}_logprob_max_abs_error": value
            for name, value in maximum_logprob_error.items()
        },
        **{
            f"replay_{name}_ratio_max_abs_error": value
            for name, value in maximum_ratio_error.items()
        },
        **{f"replay_{name}_active_count": value for name, value in active_counts.items()},
        "replay_value_max_abs_error": maximum_value_error,
    }


def main() -> None:
    args = parse_args()
    _configure_report(args.output)
    game_counts = [int(value) for value in args.games.split(",")]
    if not game_counts or any(value < 1 for value in game_counts):
        raise ValueError("--games must be a comma-separated list of positive integers")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.compile_models and args.repeats < 2:
        raise ValueError("compiled benchmarks require at least two repeats (cold and steady)")
    if args.replay_minibatch_size < 1:
        raise ValueError("--replay-minibatch-size must be positive")
    if not np.isfinite(args.max_replay_error) or args.max_replay_error <= 0.0:
        raise ValueError("--max-replay-error must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    config = ModelConfig(
        width=args.width,
        residual_blocks=args.residual_blocks,
        hidden=args.hidden,
        query_features=args.query_features,
    )
    actor = FarmActor(config).to(device).eval()
    critic = DistributionalCritic(config).to(device).eval()
    emit(
        {
            "event": "configuration",
            "device": str(device),
            "compile_models": args.compile_models,
            "max_replay_error": args.max_replay_error,
            "actor_parameters": parameter_count(actor),
            "critic_parameters": parameter_count(critic),
            "model": config.to_dict(),
            "torch": torch.__version__,
            "source_identity": source_identity(),
        }
    )

    summaries = []
    seed_cursor = args.seed
    for games in game_counts:
        rates = []
        state_rates = []
        for repeat in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            rollout = collect_self_play_rust(
                actor,
                critic,
                games=games,
                seed_start=seed_cursor,
                sampling_seed=args.seed ^ (games << 16) ^ repeat,
                compile_models=args.compile_models,
            )
            seed_cursor += games
            diagnostics = rollout_diagnostics(rollout)
            games_per_second = games / rollout.elapsed_seconds
            states_per_second = rollout.states / rollout.elapsed_seconds
            rates.append(games_per_second)
            state_rates.append(states_per_second)
            payload = {
                "event": "repeat",
                "games": games,
                "repeat": repeat,
                "phase": "cold_start" if repeat == 0 else "steady_state",
                "complete_games_per_second": games_per_second,
                "learner_states_per_second": states_per_second,
                "elapsed_seconds": rollout.elapsed_seconds,
                "peak_cuda_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
                "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "money_mean": diagnostics["money_mean"],
                "money_min": diagnostics["money_min"],
                "money_max": diagnostics["money_max"],
                "tie_fraction": diagnostics["tie_fraction"],
                "rollout_entropy": diagnostics["rollout_entropy"],
            }
            if repeat == 0:
                replay = replay_diagnostics(
                    actor,
                    critic,
                    rollout,
                    args.replay_minibatch_size,
                )
                for name in ("unit", "kind", "quantity"):
                    if replay[f"replay_{name}_active_count"] < 1:
                        raise RuntimeError(f"benchmark did not exercise the {name} policy head")
                    for metric in ("logprob", "ratio"):
                        error = replay[f"replay_{name}_{metric}_max_abs_error"]
                        if error > args.max_replay_error:
                            raise AssertionError(
                                f"{name} {metric} replay error {error:.9g} exceeds "
                                f"{args.max_replay_error:.9g}"
                            )
                if replay["replay_value_max_abs_error"] > args.max_replay_error:
                    raise AssertionError(
                        f"critic replay error {replay['replay_value_max_abs_error']:.9g} "
                        f"exceeds {args.max_replay_error:.9g}"
                    )
                payload.update(replay)
            emit(payload)
            del rollout
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summary = {
            "games": games,
            "cold_start_complete_games_per_second": rates[0],
            "steady_state_complete_games_per_second": statistics.median(rates[1:] or rates),
            "steady_state_learner_states_per_second": statistics.median(
                state_rates[1:] or state_rates
            ),
            "complete_games_per_second_median": statistics.median(rates),
            "complete_games_per_second_min": min(rates),
            "learner_states_per_second_median": statistics.median(state_rates),
        }
        summaries.append(summary)
        emit({"event": "batch_summary", **summary})
    emit({"event": "summary", "batches": summaries})


if __name__ == "__main__":
    main()
