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

from kaggriculture.model import FarmActor, ModelConfig, parameter_count
from kaggriculture.policy import component_logprobs
from kaggriculture.provenance import source_identity
from kaggriculture.rollout import collect_self_play_rust
from kaggriculture.telemetry import TensorboardMirror
from kaggriculture.training import rollout_diagnostics

_REPORT_PATH: Path | None = None
_REPORT_LINES: list[str] = []
_REPORT_MIRROR: TensorboardMirror | None = None
_REPORT_TENSORBOARD_DIR: Path | None = None


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
    """Write one strict, machine-readable JSONL record."""
    global _REPORT_MIRROR
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
    if _REPORT_MIRROR is None:
        assert _REPORT_TENSORBOARD_DIR is not None
        _REPORT_MIRROR = TensorboardMirror(_REPORT_PATH, _REPORT_TENSORBOARD_DIR)
    else:
        _REPORT_MIRROR.record(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="16,32,64,128")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cnn-width", type=int, default=48)
    parser.add_argument("--cnn-blocks", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--transformer-layers", type=int, default=7)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--quantity-rank", type=int, default=32)
    parser.add_argument("--compile-models", action="store_true")
    parser.add_argument("--replay-minibatch-size", type=int, default=2048)
    parser.add_argument(
        "--max-replay-error",
        type=float,
        default=1e-3,
        help=(
            "maximum unchanged-policy log-ratio drift across collection and PPO replay; "
            "CUDA kernels are batch-shape stable only to roughly 1e-4"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def replay_diagnostics(
    actor: FarmActor,
    rollout,
    minibatch_size: int,
) -> dict[str, float | int]:
    """Measure unchanged-policy behavior/replay drift over every stored factor.

    Collection and PPO replay intentionally use different batch shapes. CUDA
    convolution and GEMM kernels are not bitwise batch-shape invariant, even in
    eager mode, so this is a semantic importance-ratio guard rather than a
    same-kernel floating-point identity check. Native categorical correctness
    is covered separately against the exact host logits supplied to Rust.
    """
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
    }


def main() -> None:
    args = parse_args()
    _configure_report(args.output, args.tensorboard_dir)
    game_counts = [int(value) for value in args.games.split(",")]
    if not game_counts or any(value < 1 for value in game_counts):
        raise ValueError("--games must be a comma-separated list of positive integers")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.compile_models and args.repeats < 2:
        raise ValueError("compiled benchmarks require at least two repeats (cold and steady)")
    if args.replay_minibatch_size < 1:
        raise ValueError("--replay-minibatch-size must be positive")
    if (
        not np.isfinite(args.max_replay_error)
        or args.max_replay_error <= 0.0
        or args.max_replay_error > 1e-3
    ):
        raise ValueError("--max-replay-error must be finite, positive, and at most 1e-3")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    config = ModelConfig(
        cnn_width=args.cnn_width,
        cnn_blocks=args.cnn_blocks,
        model_dim=args.model_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        ffn_multiplier=args.ffn_multiplier,
        quantity_rank=args.quantity_rank,
    )
    actor = FarmActor(config).to(device).eval()
    emit(
        {
            "event": "configuration",
            "device": str(device),
            "compile_models": args.compile_models,
            "max_replay_error": args.max_replay_error,
            "actor_parameters": parameter_count(actor),
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
    _configure_report(None)


if __name__ == "__main__":
    main()
