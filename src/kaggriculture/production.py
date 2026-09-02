"""Production training configuration shared by every launch path."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from kaggriculture.model import ModelConfig
from kaggriculture.ppo import PpoConfig
from kaggriculture.provenance import repository_root

# A mirror self-play game contributes two current-policy trajectories; a
# frozen-league game contributes one. 128 * 2 : 64 is therefore the intended
# 80% current self-play / 20% past-and-reference-opponent training-data split.
PRODUCTION_SELF_PLAY_GAMES = 128
PRODUCTION_LEAGUE_GAMES = 64
PRODUCTION_LEAGUE_ACTIVE_OPPONENTS = 2
# Historical lanes are the log-age PFSP draw. `_sample_log_age_strata` takes
# one opponent per occupied log2 age bucket, then refills leftover seats from
# remaining members of those buckets. A 500-iteration archive outside the
# 16-deep active window occupies five rungs (ages 17-31, 32-63, 64-127,
# 128-255, 256-511). Two seats cover two rungs, so the older archive sits
# idle. Six seats fill every rung of that ladder plus one PFSP refill.
PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS = 6
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
#
# `scripted-v27` is the strength this leaderboard actually fields, replayed from
# the public agent's own 719-step plan. It belongs here because the PFSP weight
# is `(1 - score_rate)^2`: the three engine agents the learner already beats in
# every game weigh nothing and hand their lanes to snapshots, so before v27 the
# reserved lanes were spent on opponents that had stopped teaching anything.
PRODUCTION_LEAGUE_BUILTIN_OPPONENTS = "pass,random,starter,scripted-v27"
PRODUCTION_LEAGUE_BUILTIN_LANES = 3
PRODUCTION_EPISODE_STEPS = 720
PRODUCTION_CHECKPOINT_SECONDS = 420
# Every seat in a wave decodes at this one temperature, learner and league
# alike. Splitting them is what a separate opponent temperature did, and it
# was not neutral: the learner sampled at 1.0 while active lanes ran 0.8 and
# historical lanes ran argmax, so an identical snapshot was a strictly better
# executor of the learner's own policy and iteration 40 scored 0.302 in league
# lanes against copies of itself. A knob whose only correct value is the
# learner's temperature is not a knob.
PRODUCTION_TEMPERATURE = 1.0
# The two-shard GQA collector cannot reuse the old single-wave compile decision.
# The earlier reading of that -- an Inductor benchmark that spent 575.51 s in
# CPU-side compilation without reaching the first iteration -- was the right
# observation and the wrong diagnosis. It is not slow compilation. Both the
# shipped source and the current one stall identically at production shapes:
# every Inductor compile worker at 0% CPU, no codegen written for minutes,
# allocated device memory frozen byte-for-byte, and all 58 threads of the
# process in `futex_wait`. Waiting longer is not the fix, and the 192-game,
# 230080-state rollout it never reached takes 19.258 s eager.
#
# Graph capture is *not* the mechanism, which an intermediate version of this
# comment asserted. `inductor_default` -- Inductor's fusion with `mode="default"`
# and no CUDA graphs anywhere -- was measured and hangs the same way: 1107 s in,
# 54 of 58 threads in `futex_wait`, the subprocess compile pool idle at 0.2% CPU,
# zero cache files written in the preceding two minutes, and the GPU at 0%
# rather than the fifth utilization `reduce-overhead` leaves behind. Removing
# capture removes that difference and nothing else.
#
# What both modes share is that `collect_mixed_play_rust` runs its shards on a
# two-worker `ThreadPoolExecutor`, and `torch.compile` returns a lazy wrapper --
# so compilation is first entered from inside *both* shard threads at once, on
# the first step, before either has produced code. Concurrent entry is the
# common factor; the specific lock cycle is not established here, and the
# evidence rules capture out rather than ruling a particular lock in.
#
# That pointed the fix at ordering rather than at the backend, and ordering was
# most of it: driving one shard's first step through to completion before the
# peer's (`_pipeline_ready` / `_pipeline_wait` in `rollout.py`) is what lets
# `inductor_default` run a full wave at all, which it now does. `reduce-overhead`
# needed a second, unrelated fix on top -- `cudagraph_trees` keys generations off
# a process-global counter while keeping a tree manager per thread, so one
# shard's mark retires the other shard's live outputs and the read raises
# `accessing tensor output of CUDAGraphs that has been overwritten` --  and after
# that fix it still wedges, twice, for twenty-five minutes of idle exclusive GPU
# between them and no stack either time. That mode has had enough of this
# machine.
#
# `graph` is the replacement, and it does not go through `torch.compile` at all.
# `rollout._CapturedStep` captures one `torch.cuda.CUDAGraph` per shard over the
# step's forward region and replays it for the remaining 719 steps. That is
# available because the collector already holds every input at a fixed address
# for the life of a wave, and because owning the graph removes the entire
# question of who decides when a recording is retired. It therefore owes bitwise
# equality with eager rather than the semantic bound the compiled modes settle
# for, and `test_a_captured_collection_reproduces_the_eager_one_exactly` holds
# it to exactly that on a wave carrying both self-play and league rows.
#
# An end-to-end arm moved it. On one frozen tree at production shapes, against
# the same-tree eager control, `graph` takes the rollout from 20.383 s to
# 7.270 s and the iteration from 47.507 s to 33.368 s -- 75.78 to 107.89
# iterations an hour. Parity is untouched: `update_replay_max_kl` peaks at
# 7.8e-7 against a 5e-3 gate with every tail fraction zero, which is what a mode
# that replays the identical kernels should look like.
#
# Peak allocation rises 128 MiB an iteration for the first eight collections and
# is then flat for the rest -- five consecutive repeats at exactly zero. That is
# the league pool filling to its eight opponents and the allocator reaching
# steady state, not the capture pool accumulating; the run was carried to 14
# repeats specifically because four cannot tell those apart, and a leak here
# would have exhausted this card around iteration 160 of a 500-iteration run.
#
# BF16 remains fixed: it matches the update precision and is exercised by the
# CUDA GQA/pipeline test. `ROLLOUT_FORWARD_MODES` owns the valid mode strings;
# train_ppo.py validates whatever the calibrated launcher passes.
PRODUCTION_ROLLOUT_FORWARD_MODE = "graph"
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
# Each committed recovery checkpoint gets a deterministic external probe, giving
# the journal an absolute progress axis that self-play score rates cannot
# provide. The checkpoint event itself is the trigger, so a population worker
# can never be pointed at a missing or mutable artifact. The opponents are
# emitted explicitly so the launch command is the complete record; unavailable
# ones are dropped at launch with a warning, never fatal.
#
# `public-v16` is here because it is the strongest reference on hand: measured in
# the official engine over three seeds and both seat orders it beat `public-v27`
# 6/6, median bank 77,261 against 59,489. An absolute axis anchored only on
# agents we already beat would saturate exactly where the interesting failure
# lives. `starter` stays as the cheap floor that catches total collapse.
PRODUCTION_EXTERNAL_EVAL_OPPONENTS = "starter,public-v27,public-v16"


def production_model_config() -> dict[str, int | float]:
    return ModelConfig().to_dict()


def production_ppo_config(*, update_compile_mode: str) -> dict[str, int | float | bool | str]:
    """The schedule the calibrated launcher runs and every benchmark measures.

    With 230,080 states, a 4096-row ceiling produces 57 balanced minibatches per
    epoch. Two actor epochs and four critic epochs therefore run 114 actor and
    228 critic optimizer steps: effectively the same step counts as the former
    2048-row one-actor/two-critic schedule, while replaying each collected state
    twice for the actor and four times for the critic.

    This deliberately trades the prior throughput optimum for larger device
    work. The batch sweep found 4096 2.2-4.8% slower than 2048 and 8192 unable
    to fit. The critic holdout probe also found weak generalization from later
    passes, so the extra critic reuse must be judged by end-to-end policy
    evaluation rather than by fit explained variance.
    """
    return asdict(
        PpoConfig(
            critic_epochs=4,
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
    population: int = 1,
    games: int = PRODUCTION_SELF_PLAY_GAMES,
    rollout_bfloat16: bool = PRODUCTION_ROLLOUT_BFLOAT16,
    expected_source_digest: str | None = None,
    calibration_decision: Path | None = None,
    resume_checkpoint: Path | None = None,
    initial_actors: Sequence[Path] = (),
    critic_warmup_iterations: int | None = None,
) -> list[str]:
    """Build the exact production train_ppo.py invocation."""
    if (expected_source_digest is None) != (calibration_decision is None):
        raise ValueError("source digest and calibration decision must be provided together")
    # A warm start initializes iteration zero; a resume continues a run that
    # already has an actor. train_ppo rejects the pair, and it must fail here
    # rather than after the launcher has already rewritten the run's evidence.
    if initial_actors and resume_checkpoint is not None:
        raise ValueError("a resumed run already has an actor; --init-actor-from initializes one")
    if critic_warmup_iterations is not None and not initial_actors:
        raise ValueError("critic warmup applies only to a warm-started run")
    if critic_warmup_iterations is not None and not 0 < critic_warmup_iterations < iterations:
        raise ValueError("critic warmup must be positive and leave iterations for the actor")
    if population < 1:
        raise ValueError("a population needs at least one member")
    # Every ordered pairing must get the same number of games or seat bias
    # survives into the advantage, so the wave size is a multiple of N(N-1). No
    # production constant states it because the plan's Stage 0 measures the two
    # candidate sizes -- 156 at cost parity, 636 at data parity -- and the choice
    # is a compute call, so the caller states which one it launched.
    pairings = population * (population - 1)
    if population > 1 and games % pairings:
        raise ValueError(
            f"a population of {population} has {pairings} ordered pairings, so its wave "
            f"size must be a multiple of {pairings}; {games} is not"
        )
    if population > 1 and initial_actors and len(initial_actors) != population:
        raise ValueError(
            f"a population of {population} takes one initial actor per member or none "
            f"at all; {len(initial_actors)} were given"
        )
    if len({artifact.expanduser().resolve() for artifact in initial_actors}) != len(initial_actors):
        raise ValueError("each member needs its own initial actor; identical members score 0.5")
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
    # A population wave has no frozen or built-in lanes, so those flags are
    # emitted as the absence they are rather than left at the single-learner
    # values. A committed recovery checkpoint remains the run's absolute
    # measurement: one worker probes every member from each immutable event and
    # can see a member's bank falling while its relative score rate rises.
    league = population == 1
    command.extend(
        (
            "--device",
            "cuda",
            "--population",
            str(population),
            "--games",
            str(games),
            "--league-games",
            str(PRODUCTION_LEAGUE_GAMES if league else 0),
            "--league-active-opponents",
            str(PRODUCTION_LEAGUE_ACTIVE_OPPONENTS if league else 0),
            "--league-historical-opponents",
            str(PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS if league else 0),
            "--league-active-pool-size",
            str(PRODUCTION_LEAGUE_ACTIVE_POOL_SIZE),
            "--league-builtin-opponents",
            PRODUCTION_LEAGUE_BUILTIN_OPPONENTS if league else "",
            "--league-builtin-lanes",
            str(PRODUCTION_LEAGUE_BUILTIN_LANES if league else 0),
            "--episode-steps",
            str(PRODUCTION_EPISODE_STEPS),
            "--temperature",
            str(PRODUCTION_TEMPERATURE),
            "--checkpoint-seconds",
            str(PRODUCTION_CHECKPOINT_SECONDS),
            "--external-eval",
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
            "--critic-gae-lambda",
            str(ppo["critic_gae_lambda"]),
            "--target-kl",
            str(ppo["target_kl"]),
            "--optimizer",
            str(ppo["optimizer"]),
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
    # One flag per member, in agent order, since a population's members are
    # distinguished only by which artifact each starts from.
    for artifact in initial_actors:
        command.extend(("--init-actor-from", str(artifact)))
    if initial_actors and critic_warmup_iterations is not None:
        command.extend(("--critic-warmup-iterations", str(critic_warmup_iterations)))
    if resume_checkpoint is not None:
        command.extend(("--resume", str(resume_checkpoint)))
    return command
