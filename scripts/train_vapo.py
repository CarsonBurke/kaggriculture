#!/usr/bin/env python3
"""Train a from-scratch Kaggriculture policy with self-play VAPO."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from kaggriculture.league import (
    SnapshotRef,
    SnapshotSelection,
    copy_actor_snapshot,
    list_actor_snapshots,
    load_actor_snapshot,
    save_actor_snapshot,
    select_snapshot_mix,
    snapshot_sha256,
)
from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    ModelConfig,
    parameter_count,
)
from kaggriculture.provenance import (
    require_source_identity,
    run_provenance_from_decision,
    source_identity,
    validate_run_provenance,
)
from kaggriculture.rollout import (
    RolloutBatch,
    collect_frozen_opponents_play_rust,
    collect_self_play_rust,
    concatenate_rollouts,
)
from kaggriculture.telemetry import TensorboardMirror
from kaggriculture.training import (
    append_iteration_jsonl,
    load_checkpoint,
    metrics_journal_iteration,
    rollout_diagnostics,
    save_checkpoint,
)
from kaggriculture.vapo import VapoConfig, make_optimizers, update_vapo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=112)
    parser.add_argument("--league-games", type=int, default=96)
    parser.add_argument("--league-active-opponents", type=int, default=2)
    parser.add_argument("--league-historical-opponents", type=int, default=2)
    parser.add_argument("--league-active-pool-size", type=int, default=16)
    parser.add_argument("--opponent-temperature", type=float, default=0.8)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--cnn-width", type=int, default=48)
    parser.add_argument("--cnn-blocks", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--transformer-layers", type=int, default=7)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--quantity-rank", type=int, default=32)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--lr-warmup-steps", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--clip-low", type=float, default=0.80)
    parser.add_argument("--clip-high", type=float, default=1.28)
    parser.add_argument(
        "--gae-lambda-alpha",
        type=float,
        default=0.0,
        help="0 uses exact Monte Carlo credit; positive values enable adaptive-lambda ablations",
    )
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--compile-models", action="store_true")
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument(
        "--expected-source-digest",
        help="require the immutable source digest selected by the calibration launcher",
    )
    parser.add_argument(
        "--calibration-decision",
        type=Path,
        help="exact calibration decision to bind into every production checkpoint",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "iterations": args.iterations,
        "games": args.games,
        "league_active_pool_size": args.league_active_pool_size,
        "episode_steps": args.episode_steps,
        "checkpoint_every": args.checkpoint_every,
        "cnn_width": args.cnn_width,
        "cnn_blocks": args.cnn_blocks,
        "model_dim": args.model_dim,
        "transformer_layers": args.transformer_layers,
        "attention_heads": args.attention_heads,
        "ffn_multiplier": args.ffn_multiplier,
        "quantity_rank": args.quantity_rank,
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
    if args.league_active_opponents < 0 or args.league_historical_opponents < 0:
        raise ValueError("league opponent counts cannot be negative")
    if args.league_games and not (args.league_active_opponents or args.league_historical_opponents):
        raise ValueError("league games require at least one active or historical opponent")
    configured_opponents = 1 + args.league_active_opponents + args.league_historical_opponents
    if args.league_games and args.league_games < configured_opponents:
        raise ValueError(
            "league games must cover the initial anchor and every configured "
            f"active/historical opponent ({configured_opponents})"
        )
    if not math.isfinite(args.opponent_temperature) or args.opponent_temperature <= 0.0:
        raise ValueError("opponent temperature must be finite and positive")
    if not 0 < args.clip_low < 1 < args.clip_high:
        raise ValueError("clip interval must straddle one")
    if args.lr_warmup_steps < 0:
        raise ValueError("LR warmup steps cannot be negative")
    if args.gae_lambda_alpha < 0:
        raise ValueError("GAE lambda alpha cannot be negative")
    if not math.isfinite(args.max_hours) or args.max_hours < 0.0:
        raise ValueError("max hours must be finite and non-negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    if args.temperature != 1.0:
        raise ValueError("on-policy VAPO currently requires --temperature 1.0")
    if args.expected_source_digest is not None and (
        len(args.expected_source_digest) != 64
        or any(character not in "0123456789abcdef" for character in args.expected_source_digest)
    ):
        raise ValueError("expected source digest must be 64 lowercase hexadecimal characters")
    if (args.expected_source_digest is None) != (args.calibration_decision is None):
        raise ValueError(
            "expected source digest and calibration decision must be supplied together"
        )


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_run_provenance(
    path: Path | None,
    current_source_identity: dict[str, object],
    *,
    bind_command: bool,
) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    contents = resolved.read_bytes()
    decision = json.loads(contents.decode("utf-8"))
    if not isinstance(decision, dict):
        raise ValueError("calibration decision must be a JSON object")
    if decision.get("source_identity") != current_source_identity:
        raise ValueError("calibration decision source identity does not match current source")
    expected_command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    if bind_command and decision.get("training_command") != expected_command:
        raise ValueError("calibration decision does not bind this exact training command")
    return run_provenance_from_decision(decision)


def _balanced_assignments(games: int, opponents: int, generator: np.random.Generator) -> np.ndarray:
    """Assign nearly equal game counts to each selected opponent."""
    if games < 1 or opponents < 1:
        raise ValueError("balanced assignments require positive games and opponents")
    assignments = np.arange(games, dtype=np.int64) % opponents
    generator.shuffle(assignments)
    return assignments


def _league_opponent_diagnostics(
    league: RolloutBatch,
    assignments: np.ndarray,
    selections: list[SnapshotSelection],
) -> dict[str, float | int | str]:
    diagnostics: dict[str, float | int | str] = {}
    margins = league.final_money - league.opponent_money
    outcomes = (margins > 0).astype(np.float32) - (margins < 0).astype(np.float32)
    for index, selection in enumerate(selections):
        selected = assignments == index
        prefix = f"league_opponent_{selection.ref.iteration:08d}"
        diagnostics[f"{prefix}_category"] = selection.category
        diagnostics[f"{prefix}_games"] = int(selected.sum())
        diagnostics[f"{prefix}_score_rate"] = float(((outcomes[selected] + 1.0) / 2.0).mean())
        diagnostics[f"{prefix}_mean_margin"] = float(margins[selected].mean())
    return diagnostics


def _select_league_opponents(
    args: argparse.Namespace,
    refs: Sequence[SnapshotRef],
    iteration: int,
    generator: np.random.Generator,
) -> list[SnapshotSelection]:
    """Select a bounded opponent mix without consuming RNG when league play is off."""
    if not args.league_games:
        return []
    selections = select_snapshot_mix(
        refs,
        current_iteration=max(1, iteration),
        active_count=args.league_active_opponents,
        historical_count=args.league_historical_opponents,
        active_pool_size=args.league_active_pool_size,
        generator=generator,
    )
    if len(selections) > args.league_games:
        raise ValueError("league game budget cannot cover the selected opponent mix")
    return selections


def _training_data_config(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Return every non-model setting that can alter future rollout data."""
    return {
        "games": args.games,
        "league_games": args.league_games,
        "league_active_opponents": args.league_active_opponents,
        "league_historical_opponents": args.league_historical_opponents,
        "league_active_pool_size": args.league_active_pool_size,
        "opponent_temperature": args.opponent_temperature,
        "episode_steps": args.episode_steps,
        "temperature": args.temperature,
        "compile_models": args.compile_models,
        "device_type": device.type,
        "device_index": (
            torch.cuda.current_device()
            if device.type == "cuda" and device.index is None
            else device.index
        ),
        "cuda_device_count": torch.cuda.device_count(),
    }


def _checkpoint_values_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _checkpoint_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list | tuple) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _checkpoint_values_equal(first, second)
            for first, second in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _validate_league_manifest(
    manifest: object,
    *,
    current_iteration: int,
) -> dict[int, str]:
    if not isinstance(manifest, dict):
        raise ValueError("resume checkpoint has no valid league snapshot manifest")
    validated: dict[int, str] = {}
    for iteration, digest in manifest.items():
        if type(iteration) is not int or not 0 <= iteration <= current_iteration:
            raise ValueError("resume checkpoint has an invalid league snapshot iteration")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("resume checkpoint has an invalid league snapshot digest")
        validated[iteration] = digest
    expected_iterations = set(range(current_iteration + 1))
    if set(validated) != expected_iterations:
        raise ValueError(
            "resume checkpoint league manifest is incomplete; expected one snapshot per iteration"
        )
    return validated


def _restore_league_archive(
    *,
    checkpoint: Path,
    destination: Path,
    manifest: dict[int, str],
    model_config: ModelConfig,
) -> None:
    """Restore exactly the immutable archive bound to a training checkpoint."""
    source_directory = checkpoint.resolve().parent / "league"
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {f"league-actor-{iteration:08d}.pt" for iteration in manifest}
    existing_refs = list_actor_snapshots(destination)
    existing_by_name = {ref.path.name: ref for ref in existing_refs}
    unexpected = set(existing_by_name) - expected_names
    trailing_name = f"league-actor-{max(manifest) + 1:08d}.pt"
    disallowed = unexpected - {trailing_name}
    if disallowed or len(unexpected) > 1:
        raise ValueError(
            "resume destination contains league snapshots outside the checkpoint manifest; "
            "use a fresh --run-dir when rewinding a run"
        )
    if trailing_name in unexpected:
        # A kill after committing snapshot K+1 but before atomically replacing
        # latest.pt leaves exactly this recoverable state. It remains
        # ineligible while replaying iteration K; save_actor_snapshot later
        # requires the regenerated actor state to match exactly.
        load_actor_snapshot(
            existing_by_name[trailing_name].path,
            expected_model_config=model_config,
            device="cpu",
        )
    for iteration, expected_digest in sorted(manifest.items()):
        name = f"league-actor-{iteration:08d}.pt"
        target = destination / name
        if not target.exists():
            source = source_directory / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"checkpoint league sidecar is incomplete; missing snapshot: {source}"
                )
            copy_actor_snapshot(
                source,
                destination,
                expected_model_config=model_config,
            )
        else:
            load_actor_snapshot(
                target,
                expected_model_config=model_config,
                device="cpu",
            )
        actual_digest = snapshot_sha256(target)
        if actual_digest != expected_digest:
            raise ValueError(f"league snapshot digest mismatch: {target}")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    current_source_identity = source_identity()
    if (
        args.expected_source_digest is not None
        and current_source_identity["sha256"] != args.expected_source_digest
    ):
        raise ValueError(
            "current source digest does not match the calibration decision: "
            f"{current_source_identity['sha256']} != {args.expected_source_digest}"
        )
    run_provenance = _load_run_provenance(
        args.calibration_decision,
        current_source_identity,
        bind_command=args.resume is None,
    )
    if run_provenance is not None and (
        run_provenance["calibration"]["compile_models"] is not args.compile_models
    ):
        raise ValueError("training compile mode does not match calibration run provenance")
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
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
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        gae_lambda_alpha=args.gae_lambda_alpha,
        max_gradient_norm=args.max_gradient_norm,
        target_kl=args.target_kl,
        use_bfloat16=not args.no_bfloat16,
    )
    training_data_config = _training_data_config(args, device)
    actor = FarmActor(model_config).to(device)
    critic = DistributionalCritic(model_config).to(device)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, vapo_config)
    generator = np.random.default_rng(args.seed + 1)
    iteration = 0
    next_seed = args.seed
    resume_payload = None
    if args.resume:
        resume_payload = load_checkpoint(
            args.resume,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            device=device,
        )
        if resume_payload["model_config"] != model_config.to_dict():
            raise ValueError("resume checkpoint model configuration does not match arguments")
        if resume_payload["vapo_config"] != asdict(vapo_config):
            raise ValueError("resume checkpoint VAPO configuration does not match arguments")
        if resume_payload.get("training_data_config") != training_data_config:
            raise ValueError(
                "resume checkpoint data-generation configuration does not match arguments"
            )
        require_source_identity(resume_payload.get("source_identity"))
        checkpoint_run_provenance = validate_run_provenance(resume_payload.get("run_provenance"))
        if checkpoint_run_provenance is not None and (
            checkpoint_run_provenance["calibration"]["compile_models"] is not args.compile_models
        ):
            raise ValueError("resume checkpoint compile mode does not match calibration")
        if run_provenance is None:
            run_provenance = checkpoint_run_provenance
        elif checkpoint_run_provenance != run_provenance:
            raise ValueError("resume checkpoint run provenance does not match calibration")
        iteration = int(resume_payload["iteration"])
        next_seed = int(resume_payload["next_seed"])
        if resume_payload.get("training_rng") is not None:
            generator.bit_generator.state = resume_payload["training_rng"]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    journal_iteration = metrics_journal_iteration(args.run_dir / "metrics.jsonl")
    if resume_payload is not None and journal_iteration > iteration:
        raise ValueError(
            "resume destination has committed metrics newer than the checkpoint; "
            "use a fresh --run-dir when rewinding a run"
        )
    initial_checkpoint = args.run_dir / "checkpoint-000000.pt"
    if (
        iteration == 0
        and initial_checkpoint.exists()
        and (args.resume is None or initial_checkpoint.resolve() != args.resume.resolve())
    ):
        raise FileExistsError(
            f"refusing to trust a pre-existing initial checkpoint: {initial_checkpoint}; "
            "use a fresh --run-dir or resume that exact checkpoint"
        )
    league_directory = args.run_dir / "league"
    league_snapshot_manifest: dict[int, str] = {}
    if resume_payload is not None:
        league_snapshot_manifest = _validate_league_manifest(
            resume_payload.get("league_snapshot_manifest"),
            current_iteration=iteration,
        )
        _restore_league_archive(
            checkpoint=args.resume,
            destination=league_directory,
            manifest=league_snapshot_manifest,
            model_config=model_config,
        )
    serialized_arguments = {
        name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
    }
    serialized_arguments["resume"] = str(args.resume or "")
    configuration = {
        "arguments": serialized_arguments,
        "model": model_config.to_dict(),
        "vapo": asdict(vapo_config),
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "device": str(device),
        "torch_version": torch.__version__,
        "source_identity": current_source_identity,
        "run_provenance": run_provenance,
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if resume_payload is not None:
        destination_latest = args.run_dir / "latest.pt"
        if destination_latest.exists():
            destination_payload = torch.load(
                destination_latest,
                map_location="cpu",
                weights_only=False,
            )
            if not _checkpoint_values_equal(destination_payload, resume_payload):
                raise FileExistsError(
                    f"latest checkpoint conflicts with resume state: {destination_latest}"
                )
        else:
            save_checkpoint(
                destination_latest,
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                model_config=model_config,
                vapo_config=vapo_config,
                iteration=iteration,
                next_seed=next_seed,
                metrics=resume_payload["metrics"],
                training_rng_state=generator.bit_generator.state,
                training_data_config=training_data_config,
                league_snapshot_manifest=league_snapshot_manifest,
                source_identity=current_source_identity,
                run_provenance=run_provenance,
            )
    if resume_payload is not None and iteration > 0:
        numbered_checkpoint = args.run_dir / f"checkpoint-{iteration:06d}.pt"
        if iteration % args.checkpoint_every == 0:
            if numbered_checkpoint.exists():
                numbered_payload = torch.load(
                    numbered_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
                if not _checkpoint_values_equal(numbered_payload, resume_payload):
                    raise FileExistsError(
                        f"numbered checkpoint conflicts with resume state: {numbered_checkpoint}"
                    )
            else:
                save_checkpoint(
                    numbered_checkpoint,
                    actor=actor,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    model_config=model_config,
                    vapo_config=vapo_config,
                    iteration=iteration,
                    next_seed=next_seed,
                    metrics=resume_payload["metrics"],
                    training_rng_state=generator.bit_generator.state,
                    training_data_config=training_data_config,
                    league_snapshot_manifest=league_snapshot_manifest,
                    source_identity=current_source_identity,
                    run_provenance=run_provenance,
                )
        append_iteration_jsonl(args.run_dir / "metrics.jsonl", resume_payload["metrics"])
    writer = TensorboardMirror(
        args.run_dir / "metrics.jsonl",
        args.run_dir / "tensorboard",
        writer_factory=lambda path: SummaryWriter(path),
    )
    started = time.monotonic()

    # The snapshot for the checkpoint's current actor is installed before the
    # checkpoint is written, so every manifest is complete and portable with
    # its immutable ``league/`` sidecar directory.
    current_snapshot = save_actor_snapshot(league_directory, actor, iteration)
    current_digest = snapshot_sha256(current_snapshot.path)
    previous_digest = league_snapshot_manifest.get(iteration)
    if previous_digest is not None and previous_digest != current_digest:
        raise ValueError("resume checkpoint actor does not match its current league snapshot")
    league_snapshot_manifest[iteration] = current_digest

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
            training_data_config=training_data_config,
            league_snapshot_manifest=league_snapshot_manifest,
            source_identity=current_source_identity,
            run_provenance=run_provenance,
        )

    while iteration < args.iterations:
        if args.max_hours and (time.monotonic() - started) / 3600.0 >= args.max_hours:
            break
        iteration_started = time.monotonic()
        self_play_sampling_seed = int(generator.integers(0, np.iinfo(np.int64).max))
        self_play = collect_self_play_rust(
            actor,
            critic,
            games=args.games,
            seed_start=next_seed,
            episode_steps=args.episode_steps,
            temperature=args.temperature,
            sampling_seed=self_play_sampling_seed,
            compile_models=args.compile_models,
        )
        next_seed += args.games
        self_play_diagnostics = {
            f"self_play_{name}": value for name, value in rollout_diagnostics(self_play).items()
        }
        rollout_parts = [self_play]
        opponent_checkpoint = ""
        league_diagnostics = {}
        league = None
        selections = _select_league_opponents(
            args,
            list_actor_snapshots(league_directory),
            iteration,
            generator,
        )
        if args.league_games and selections:
            opponents = [
                load_actor_snapshot(
                    selection.ref.path,
                    expected_model_config=model_config,
                    device=device,
                )
                for selection in selections
            ]
            assignments = _balanced_assignments(
                args.league_games,
                len(opponents),
                generator,
            )
            opponent_temperatures = np.asarray(
                [
                    args.opponent_temperature if row.category == "active" else 1.0
                    for row in selections
                ],
                dtype=np.float32,
            )
            deterministic_opponents = np.asarray(
                [row.category != "active" for row in selections],
                dtype=np.bool_,
            )
            league_sampling_seed = int(generator.integers(0, np.iinfo(np.int64).max))
            league = collect_frozen_opponents_play_rust(
                actor,
                critic,
                opponents,
                games=args.league_games,
                opponent_indices=assignments,
                seed_start=next_seed,
                episode_steps=args.episode_steps,
                temperature=args.temperature,
                opponent_temperature=args.opponent_temperature,
                opponent_temperatures=opponent_temperatures,
                deterministic_opponents=deterministic_opponents,
                sampling_seed=league_sampling_seed,
                compile_models=args.compile_models,
            )
            next_seed += args.league_games
            league_diagnostics = {
                f"league_{name}": value for name, value in rollout_diagnostics(league).items()
            }
            league_diagnostics.update(_league_opponent_diagnostics(league, assignments, selections))
            rollout_parts.append(league)
            opponent_checkpoint = ",".join(row.ref.path.name for row in selections)
            del opponents
        rollout = concatenate_rollouts(rollout_parts)
        del rollout_parts, self_play
        if league is not None:
            del league
        update_started = time.monotonic()
        update_metrics = update_vapo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            vapo_config,
            generator=generator,
        )
        if int(update_metrics["actor_updates"]) < 1:
            raise RuntimeError("VAPO iteration completed without an actor update")
        update_seconds = time.monotonic() - update_started
        iteration += 1
        metrics = {
            "iteration": iteration,
            "next_seed": next_seed,
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            "iteration_seconds": time.monotonic() - iteration_started,
            "update_seconds": update_seconds,
            "critic_replayed_states_per_second": (
                rollout.states * vapo_config.epochs / max(update_seconds, 1e-9)
            ),
            "league_checkpoint": opponent_checkpoint,
            **rollout_diagnostics(rollout),
            **self_play_diagnostics,
            **league_diagnostics,
            **update_metrics,
        }
        if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError(f"non-finite training metric: {metrics}")
        current_snapshot = save_actor_snapshot(league_directory, actor, iteration)
        league_snapshot_manifest[iteration] = snapshot_sha256(current_snapshot.path)
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
            training_data_config=training_data_config,
            league_snapshot_manifest=league_snapshot_manifest,
            source_identity=current_source_identity,
            run_provenance=run_provenance,
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
                training_data_config=training_data_config,
                league_snapshot_manifest=league_snapshot_manifest,
                source_identity=current_source_identity,
                run_provenance=run_provenance,
            )
        append_iteration_jsonl(args.run_dir / "metrics.jsonl", metrics)
        writer.record(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        del rollout
    writer.close()


if __name__ == "__main__":
    main()
