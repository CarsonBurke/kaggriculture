"""Production training configuration shared by every launch path."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from kaggriculture.model import ModelConfig
from kaggriculture.ppo import PpoConfig
from kaggriculture.provenance import repository_root

PRODUCTION_SELF_PLAY_GAMES = 112
PRODUCTION_LEAGUE_GAMES = 96
PRODUCTION_LEAGUE_ACTIVE_OPPONENTS = 2
PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS = 2
PRODUCTION_LEAGUE_ACTIVE_POOL_SIZE = 16
PRODUCTION_EPISODE_STEPS = 720
PRODUCTION_CHECKPOINT_EVERY = 5
PRODUCTION_TEMPERATURE = 1.0
PRODUCTION_OPPONENT_TEMPERATURE = 0.8
# The collection forward is ~64% of a wave's wall clock, and measurement picked
# both of these rather than taste. Production 112-game waves, real BC actor:
# the rollout sweep moves 8.91 s (eager/fp32) -> 5.36 s (inductor/bf16), 1.66x,
# while the shipped 4-wave `scripts/audit_replay_parity.py` gate on the
# league-mixed path moves worst max_kl 1.9089e-03 -> 2.2786e-04, 8.4x lower
# drift: the update path is already Inductor + bf16, so matching its backend
# and precision cancels most of the collect/update gap instead of widening it.
#
# They are not the same kind of setting. The forward mode is the calibrated
# rollout knob -- the chain varies it and attributes a speedup to it -- so it
# reaches the command as a parameter, and this constant is only what the
# uncalibrated direct launch (`scripts/launch_production.py`) states. The
# precision is not a knob: it is pinned identical on every chain node, so it
# is fixed configuration and this constant is the value.
# `ROLLOUT_FORWARD_MODES` owns the valid mode strings; train_ppo.py's
# `--rollout-forward-mode` choices validate whatever is passed, exercised by
# the launcher round-trip test.
PRODUCTION_ROLLOUT_FORWARD_MODE = "inductor"
PRODUCTION_ROLLOUT_BFLOAT16 = True
# Deterministic starter/public-v27 probes every N committed iterations give the
# journal an absolute progress axis that self-play score rates cannot provide.
# The opponents are emitted explicitly so the launch command is the complete
# record; unavailable ones are dropped at launch with a warning, never fatal.
PRODUCTION_EXTERNAL_EVAL_EVERY = 10
PRODUCTION_EXTERNAL_EVAL_OPPONENTS = "starter,public-v27"


def production_model_config() -> dict[str, int | float]:
    return ModelConfig().to_dict()


def production_ppo_config(*, compile_update: bool) -> dict[str, int | float | bool]:
    return asdict(
        PpoConfig(
            epochs=1,
            critic_epochs=4,
            minibatch_size=2048,
            target_kl=0.03,
            compile_update=compile_update,
        )
    )


def require_repository_launcher(script_file: Path) -> None:
    """Reject launchers running outside the tree that provides kaggriculture.

    A launcher script from one checkout combined with an importable package
    from another would exec that other tree's train_ppo.py while stamping its
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
    rollout_forward_mode: str,
    compile_update: bool,
    rollout_bfloat16: bool = PRODUCTION_ROLLOUT_BFLOAT16,
    expected_source_digest: str | None = None,
    calibration_decision: Path | None = None,
    resume_checkpoint: Path | None = None,
    initial_actor: Path | None = None,
    critic_warmup_iterations: int | None = None,
) -> list[str]:
    """Build the exact production train_ppo.py invocation."""
    if (expected_source_digest is None) != (calibration_decision is None):
        raise ValueError("source digest and calibration decision must be provided together")
    # A warm start initializes iteration zero; a resume continues a run that
    # already has an actor. train_ppo rejects the pair, and it must fail here
    # rather than after the launcher has already rewritten the run's evidence.
    if initial_actor is not None and resume_checkpoint is not None:
        raise ValueError("a resumed run already has an actor; --init-actor-from initializes one")
    if critic_warmup_iterations is not None and initial_actor is None:
        raise ValueError("critic warmup applies only to a warm-started run")
    if critic_warmup_iterations is not None and not 0 < critic_warmup_iterations < iterations:
        raise ValueError("critic warmup must be positive and leave iterations for the actor")
    model = production_model_config()
    ppo = production_ppo_config(compile_update=compile_update)
    command = [
        sys.executable,
        str(repository_root() / "scripts" / "train_ppo.py"),
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
            "--external-eval-every",
            str(PRODUCTION_EXTERNAL_EVAL_EVERY),
            "--external-eval-opponents",
            PRODUCTION_EXTERNAL_EVAL_OPPONENTS,
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
            str(ppo["actor_learning_rate"]),
            "--critic-lr",
            str(ppo["critic_learning_rate"]),
            "--lr-warmup-steps",
            str(ppo["lr_warmup_steps"]),
            "--weight-decay",
            str(ppo["weight_decay"]),
            "--epochs",
            str(ppo["epochs"]),
            "--critic-epochs",
            str(ppo["critic_epochs"]),
            "--minibatch-size",
            str(ppo["minibatch_size"]),
            "--clip-low",
            str(ppo["clip_low"]),
            "--clip-high",
            str(ppo["clip_high"]),
            "--gamma",
            str(ppo["gamma"]),
            "--actor-gae-lambda",
            str(ppo["actor_gae_lambda"]),
            "--target-kl",
            str(ppo["target_kl"]),
            "--max-gradient-norm",
            str(ppo["max_gradient_norm"]),
        )
    )
    if compile_update:
        command.append("--compile-update")
    # Stated unconditionally, in both directions: these move the sampled
    # behavior policy, so the command has to be the complete record of the
    # collection backend and precision a run actually used rather than leaning
    # on whatever train_ppo currently defaults to.
    command.extend(("--rollout-forward-mode", rollout_forward_mode))
    command.append("--rollout-bfloat16" if rollout_bfloat16 else "--no-rollout-bfloat16")
    if initial_actor is not None:
        command.extend(("--init-actor-from", str(initial_actor)))
        if critic_warmup_iterations is not None:
            command.extend(("--critic-warmup-iterations", str(critic_warmup_iterations)))
    if resume_checkpoint is not None:
        command.extend(("--resume", str(resume_checkpoint)))
    return command
