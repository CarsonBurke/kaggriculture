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
# Engine reference agents admitted to the TRAINING league, and the lanes
# reserved for them. They are here because self-play alone never produced an
# opponent the learner had to beat from outside its own lineage: the cloned
# policy loses every game to `starter`, a twenty-eight-line carrot loop, and
# a league it can only lose to itself cannot notice. The lanes are a ceiling,
# not a floor -- `select_league_mix` contests each reserved lane against one
# more snapshot on the shared PFSP weight, so beaten built-ins hand their
# lanes back to the snapshot strata without anyone editing this constant.
# Evaluation stays separate: `PRODUCTION_EXTERNAL_EVAL_OPPONENTS` below is a
# diagnostic probe and shares nothing with these lanes.
PRODUCTION_LEAGUE_BUILTIN_OPPONENTS = "pass,random,starter"
PRODUCTION_LEAGUE_BUILTIN_LANES = 3
PRODUCTION_EPISODE_STEPS = 720
PRODUCTION_CHECKPOINT_EVERY = 5
# Every seat in a wave decodes at this one temperature, learner and league
# alike. Splitting them is what a separate opponent temperature did, and it
# was not neutral: the learner sampled at 1.0 while active lanes ran 0.8 and
# historical lanes ran argmax, so an identical snapshot was a strictly better
# executor of the learner's own policy and iteration 40 scored 0.302 in league
# lanes against copies of itself. A knob whose only correct value is the
# learner's temperature is not a knob.
PRODUCTION_TEMPERATURE = 1.0
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
# The update knob's counterpart to the mode above, and the same kind of setting:
# the calibrated knob reaches the command as a parameter, and this constant is
# only what the uncalibrated direct launch states. `default` is Inductor's
# fusion without CUDA graph capture, which is the mode production has been
# running; the three modes above it in `UPDATE_COMPILE_MODES` add graph capture
# or benchmarked kernel selection and are not yet measured on this update path
# -- `scripts/profile_update_backends.py` is what measures them. Stating the
# mode that has run rather than the fastest mode nobody has timed is the whole
# point of the direct launch being uncalibrated: the chain is what earns a
# change here.
PRODUCTION_UPDATE_COMPILE_MODE = "default"
# Deterministic starter/public-v27 probes every N committed iterations give the
# journal an absolute progress axis that self-play score rates cannot provide.
# The opponents are emitted explicitly so the launch command is the complete
# record; unavailable ones are dropped at launch with a warning, never fatal.
PRODUCTION_EXTERNAL_EVAL_EVERY = 10
PRODUCTION_EXTERNAL_EVAL_OPPONENTS = "starter,public-v27"


def production_model_config() -> dict[str, int | float]:
    return ModelConfig().to_dict()


def production_ppo_config(*, update_compile_mode: str) -> dict[str, int | float | bool | str]:
    """The schedule the calibrated launcher runs and every benchmark measures.

    `critic_epochs` is 2 because that is where the measurement leaves it, not
    because it is the actor count doubled. `scripts/probe_critic_epochs.py`
    refits the critic on 80% of the GAMES in a production wave and scores the
    held-out 20%, splitting by `episode_seeds` rather than by state: states along
    one trajectory are near-duplicates, so a random state split puts copies of
    fitted states in the holdout and reports memorization as generalization.
    That distinction is the whole result. Under a state-level reading the critic
    looks like it reaches 0.86 explained variance; on a game-level holdout it
    explains 0.5-2% of value variance at EVERY epoch count from 1 to 8.

    So no epoch measurably buys generalization. The remaining holdout explained
    variance gain after epoch 0 is +0.0017 +/- 0.0021 over five repeats from a
    fresh critic and -0.0442 +/- 0.0786 over three from a warm one, both inside
    their own spread, while fit explained variance climbs to 0.87 and the
    memorization gap to 1.15. A warm critic's holdout explained variance goes
    from +0.005 to -0.283 across eight epochs -- worse than predicting the
    holdout mean.

    Two is a DELIBERATE OVERSPEND, not the optimum, and the number it costs is
    known. On the distributional loss the critic actually optimizes, the marginal
    per-epoch change in holdout loss is -0.081 +/- 0.049 for epoch 1 and
    +0.072 +/- 0.021 for epoch 2 once the critic is warm, so epoch 2 gives back
    slightly more than epoch 1 wins, and epoch 1 is the best epoch in all three
    warm repeats. A fresh critic wants three epochs (epoch 3 worth -0.087 +/-
    0.030, epoch 4 break-even at +0.003 +/- 0.018); a warm one wants one. The two
    regimes have different optima and this constant has to serve both.

    What buys the overspend is the one thing no single-rollout curve can measure:
    the critic also accumulates fit ACROSS iterations, and the same probe shows
    data dominating passes -- eight epochs over one wave leaves explained variance
    at 0.002, while four epochs over each of six disjoint waves reaches 0.54 and
    0.79 on two of three seeds for the same 24 gradient epochs. A second pass is
    held as insurance against one pass per iteration failing to keep up over 500
    of them, at a measured price of 4.6 s per iteration and a small known
    degradation in the warm-regime holdout fit. Four cost 18.5 s of the 28.0 s
    update phase; two costs 9.2 s and takes the iteration from 32.5 s to 23.2 s.
    """
    return asdict(
        PpoConfig(
            epochs=1,
            critic_epochs=2,
            minibatch_size=2048,
            target_kl=PpoConfig.target_kl,
            update_compile_mode=update_compile_mode,
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
    update_compile_mode: str,
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
    ppo = production_ppo_config(update_compile_mode=update_compile_mode)
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
            "--league-builtin-opponents",
            PRODUCTION_LEAGUE_BUILTIN_OPPONENTS,
            "--league-builtin-lanes",
            str(PRODUCTION_LEAGUE_BUILTIN_LANES),
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
            "--entropy-coefficient",
            str(ppo["entropy_coefficient"]),
            "--max-gradient-norm",
            str(ppo["max_gradient_norm"]),
        )
    )
    # Stated unconditionally, all three: the collection backend and precision
    # move the sampled behavior policy and the update mode moves the graphs that
    # consume it, so the command has to be the complete record of what a run
    # actually used rather than leaning on whatever train_ppo currently defaults
    # to. `--update-compile-mode` carries the whole update knob; no boolean
    # projection of it is emitted, because two names for one decision give the
    # command two places to disagree with the calibration it was launched from.
    command.extend(("--rollout-forward-mode", rollout_forward_mode))
    command.append("--rollout-bfloat16" if rollout_bfloat16 else "--no-rollout-bfloat16")
    command.extend(("--update-compile-mode", update_compile_mode))
    if initial_actor is not None:
        command.extend(("--init-actor-from", str(initial_actor)))
        if critic_warmup_iterations is not None:
            command.extend(("--critic-warmup-iterations", str(critic_warmup_iterations)))
    if resume_checkpoint is not None:
        command.extend(("--resume", str(resume_checkpoint)))
    return command
