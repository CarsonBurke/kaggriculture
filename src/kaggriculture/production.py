"""Production training configuration shared by every launch path."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from kaggriculture.model import ModelConfig
from kaggriculture.provenance import repository_root
from kaggriculture.vapo import VapoConfig

PRODUCTION_SELF_PLAY_GAMES = 112
PRODUCTION_LEAGUE_GAMES = 96
PRODUCTION_LEAGUE_INITIAL_OPPONENTS = 1
PRODUCTION_LEAGUE_ACTIVE_OPPONENTS = 2
PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS = 2
PRODUCTION_LEAGUE_ACTIVE_POOL_SIZE = 16
PRODUCTION_EPISODE_STEPS = 720
PRODUCTION_CHECKPOINT_EVERY = 5
PRODUCTION_TEMPERATURE = 1.0
PRODUCTION_OPPONENT_TEMPERATURE = 0.8


def production_model_config() -> dict[str, int | float]:
    return ModelConfig().to_dict()


def production_vapo_config(*, compiled: bool) -> dict[str, int | float | bool]:
    return asdict(
        VapoConfig(epochs=1, minibatch_size=2048, target_kl=0.03, compile_update=compiled)
    )


def require_repository_launcher(script_file: Path) -> None:
    """Reject launchers running outside the tree that provides kaggriculture.

    A launcher script from one checkout combined with an importable package
    from another would exec that other tree's train_vapo.py while stamping its
    source identity — silently launching foreign code.  Both trees must agree.
    """
    expected = repository_root() / "scripts"
    actual = script_file.resolve().parent
    if actual != expected:
        raise RuntimeError(
            f"launcher lives in {actual} but the imported kaggriculture package "
            f"belongs to {expected.parent}; refusing a mixed-tree launch"
        )


def resolve_resume_checkpoint(run_directory: Path) -> Path | None:
    """Return the run's atomic latest checkpoint, or None for a fresh start."""
    latest_checkpoint = run_directory / "latest.pt"
    if latest_checkpoint.is_symlink() or (
        latest_checkpoint.exists() and not latest_checkpoint.is_file()
    ):
        raise ValueError(f"training resume checkpoint is not a regular file: {latest_checkpoint}")
    return latest_checkpoint if latest_checkpoint.is_file() else None


def build_training_command(
    run_directory: Path,
    *,
    iterations: int,
    max_hours: float,
    seed: int,
    compile_models: bool,
    expected_source_digest: str | None = None,
    calibration_decision: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    """Build the exact production train_vapo.py invocation."""
    if (expected_source_digest is None) != (calibration_decision is None):
        raise ValueError("source digest and calibration decision must be provided together")
    model = production_model_config()
    vapo = production_vapo_config(compiled=compile_models)
    command = [
        sys.executable,
        str(repository_root() / "scripts" / "train_vapo.py"),
        "--run-dir",
        str(run_directory),
        "--iterations",
        str(iterations),
        "--max-hours",
        str(max_hours),
        "--seed",
        str(seed),
    ]
    if expected_source_digest is not None and calibration_decision is not None:
        command.extend(
            (
                "--expected-source-digest",
                expected_source_digest,
                "--calibration-decision",
                str(calibration_decision),
            )
        )
    command.extend(
        (
            "--device",
            "cuda",
            "--games",
            str(PRODUCTION_SELF_PLAY_GAMES),
            "--league-games",
            str(PRODUCTION_LEAGUE_GAMES),
            "--league-active-opponents",
            str(PRODUCTION_LEAGUE_ACTIVE_OPPONENTS),
            "--league-historical-opponents",
            str(PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS),
            "--league-active-pool-size",
            str(PRODUCTION_LEAGUE_ACTIVE_POOL_SIZE),
            "--opponent-temperature",
            str(PRODUCTION_OPPONENT_TEMPERATURE),
            "--episode-steps",
            str(PRODUCTION_EPISODE_STEPS),
            "--temperature",
            str(PRODUCTION_TEMPERATURE),
            "--checkpoint-every",
            str(PRODUCTION_CHECKPOINT_EVERY),
            "--cnn-width",
            str(model["cnn_width"]),
            "--cnn-blocks",
            str(model["cnn_blocks"]),
            "--model-dim",
            str(model["model_dim"]),
            "--transformer-layers",
            str(model["transformer_layers"]),
            "--attention-heads",
            str(model["attention_heads"]),
            "--ffn-multiplier",
            str(model["ffn_multiplier"]),
            "--quantity-rank",
            str(model["quantity_rank"]),
            "--actor-lr",
            str(vapo["actor_learning_rate"]),
            "--critic-lr",
            str(vapo["critic_learning_rate"]),
            "--lr-warmup-steps",
            str(vapo["lr_warmup_steps"]),
            "--weight-decay",
            str(vapo["weight_decay"]),
            "--epochs",
            str(vapo["epochs"]),
            "--minibatch-size",
            str(vapo["minibatch_size"]),
            "--clip-low",
            str(vapo["clip_low"]),
            "--clip-high",
            str(vapo["clip_high"]),
            "--gamma",
            str(vapo["gamma"]),
            "--actor-gae-lambda",
            str(vapo["actor_gae_lambda"]),
            "--target-kl",
            str(vapo["target_kl"]),
            "--max-gradient-norm",
            str(vapo["max_gradient_norm"]),
        )
    )
    if compile_models:
        command.append("--compile-models")
    if resume_checkpoint is not None:
        command.extend(("--resume", str(resume_checkpoint)))
    return command
