"""Production training configuration shared by every launch path."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaggriculture.evaluation import DEVELOPMENT_SEED_START
from kaggriculture.provenance import repository_root
from kaggriculture.registry import STRUCTURED

PRODUCTION_ARCHITECTURE = STRUCTURED
PRODUCTION_CRITIC_WARMUP_ITERATIONS = 5
PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS = 40

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
# Collection uses one physical-game shard: all 192 games enter one `BatchEnv`,
# one whole-wave actor/league graph, and one Rayon game-parallel native step.
# There is no Python thread split, no duplicated first-step compilation, and no
# shard join. The 719 environment transitions remain causally sequential, but
# every game's work within each transition is parallel.
#
# `inductor_graph` is the collector-owned `torch.cuda.CUDAGraph` over the
# Inductor-fused (`mode="default"`) actor/league forward. The collector holds
# its packed inputs at fixed device addresses for the wave, compiles during the
# capture warmup, captures the complete fused forward once, and replays it
# thereafter. Owning the graph avoids Inductor cudagraph-tree generation
# bookkeeping; fusing first is what removes the ~1,900 eager kernels per step
# that the plain `graph` mode replays unchanged. Matched six-repeat MLQ runs at
# production shape (artifacts/benchmarks/rollout-mode-{graph,inductor_graph}
# .jsonl): steady rollout median 6.72 s -> 3.90 s (42%), and because the fused
# kernels are the update path's own, the update-replay joint KL fell from
# 8.3e-5/6.6e-5 to 5.4e-5/3.8e-5 with a zero tail fraction in both.
#
# Structured BF16 collection uses a native-BF16 inference replica while the
# trainable actor remains FP32. Quantity heads stay FP32. The GPU/native action
# boundary transfers only unit/kind utilities, low-rank quantity context, and
# one scalar quantity draw per slot. At production shape the quantity part is
# 506,880 bytes per step instead of the former 8,448,000-byte all-kind prefix
# table, a 16.67x reduction. Two pinned host encodings share one fixed device
# input block, so encoding and the next H2D submission happen before CPU
# trajectory storage without changing graph addresses.
# The learner and frozen-opponent forwards use independent CUDA streams inside
# that single graph and rejoin before sampling. At 192 physical games, a matched
# four-repeat MLQ run reduced the three-post-cold steady rollout median from
# 9.0817 s with sequential forwards to 8.2462 s (9.2%); maximum update-replay KL
# was 1.612e-5 and the tail fraction was zero.
#
# BF16 matches the update precision and is guarded by replay-parity KL and tail
# gates. `ROLLOUT_FORWARD_MODES` owns the valid mode strings; train_ppo.py
# validates the configured mode.
PRODUCTION_ROLLOUT_FORWARD_MODE = "inductor_graph"
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


def production_model_config() -> dict[str, Any]:
    """Return the complete JSON-persisted structured production model contract."""
    from kaggriculture.structured import StructuredConfig

    config = StructuredConfig(
        model_dim=80,
        attention_heads=4,
        attention_kv_heads=2,
        ffn_multiplier=2,
        farm_blocks=2,
        opponent_latents=8,
        latents=32,
        core_layers=8,
        quantity_rank=32,
        global_refresh_layers=(),
        global_refresh_context="none",
        # nanogpt residual transports. Gates start at 0 so they are identity
        # until trained: x0 into every core layer, U-net skip 3→6, late MUDD.
        input_reinject_layers=(1, 2, 3, 4, 5, 6, 7, 8),
        core_skip_source=3,
        core_skip_target=6,
        zero_init_branches=False,
        mudd_lite=True,
        fuse_market_decoder=True,
        fuse_unit_decoder=False,
        split_clock_token=False,
        global_modulation=True,
        fused_mlp=False,
        critic_core_layers=0,
        critic_latents=0,
        value_atoms=101,
        value_min=-2.2,
        value_max=2.2,
        value_sigma_ratio=0.75,
    ).to_dict()
    # Benchmark reports pass through JSON before the calibrated launcher reads
    # them, so tuple-valued layer schedules are lists in the persisted contract.
    return {
        name: list(value) if isinstance(value, tuple) else value for name, value in config.items()
    }


def production_ppo_config(
    *, update_compile_mode: str
) -> dict[str, int | float | bool | str | None]:
    """The schedule the calibrated launcher runs and every benchmark measures.

    With 230,080 states, a 4096-row ceiling produces 57 balanced minibatches per
    epoch. Production is one actor epoch and one critic epoch on the same wave:
    a second same-wave critic pass memorized holdout, and a second actor pass
    is a replay at a KL that does not bind. Independent CUDA streams overlap
    each paired actor/critic minibatch. NextLat shares that trunk pass: h_t and
    h_{t+1} come from contiguous episode runs, and p_ψ steps on the same
    backward as PPO.
    """
    from kaggriculture.ppo import PpoConfig

    return asdict(
        PpoConfig(
            critic_epochs=PpoConfig.epochs,
            target_kl=PpoConfig.target_kl,
            update_compile_mode=update_compile_mode,
            # The actor predicts future policy distributions. The critic uses
            # both latent regression and decoded categorical-value matching.
            structured_decision_coefficient=1.0,
            structured_patch_coefficient=0.0,
            structured_economy_coefficient=0.0,
            structured_opponent_summary_coefficient=0.0,
            structured_opponent_patch_coefficient=0.0,
            structured_decision_horizon=1,
            structured_patch_horizon=1,
            structured_critic_latent_coefficient=1.0,
            structured_critic_value_coefficient=1.0,
            structured_critic_horizon=1,
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


def resolve_resume_checkpoint(
    run_directory: Path,
    requested_checkpoint: Path | None = None,
) -> Path | None:
    """Resolve an explicit checkpoint or the run's atomic latest checkpoint."""
    checkpoint = (
        run_directory / "latest.pt"
        if requested_checkpoint is None
        else requested_checkpoint.expanduser()
    )
    if checkpoint.is_symlink() or (checkpoint.exists() and not checkpoint.is_file()):
        raise ValueError(f"training resume checkpoint is not a regular file: {checkpoint}")
    if checkpoint.is_file():
        return checkpoint.resolve() if requested_checkpoint is not None else checkpoint
    if requested_checkpoint is not None:
        raise FileNotFoundError(checkpoint)
    return None


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
    if initial_actors and critic_warmup_iterations is None:
        critic_warmup_iterations = PRODUCTION_CRITIC_WARMUP_ITERATIONS
    if critic_warmup_iterations is not None and not initial_actors:
        raise ValueError("critic warmup applies only to a warm-started run")
    if (
        critic_warmup_iterations is not None
        and critic_warmup_iterations > PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS
    ):
        raise ValueError(
            "critic warmup cannot exceed the "
            f"{PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS}-iteration readiness deadline"
        )
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
    if resume_checkpoint is None and len(initial_actors) != population:
        if population == 1:
            raise ValueError("a fresh production run requires exactly one BC actor or --resume")
        raise ValueError(
            f"a fresh production population needs one BC actor per member; "
            f"{len(initial_actors)} were given for {population} members"
        )
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
            "--external-eval-seed-start",
            str(DEVELOPMENT_SEED_START),
            "--architecture",
            PRODUCTION_ARCHITECTURE,
            "--model-dim",
            str(model["model_dim"]),
            "--attention-heads",
            str(model["attention_heads"]),
            "--attention-kv-heads",
            str(model["attention_kv_heads"]),
            "--ffn-multiplier",
            str(model["ffn_multiplier"]),
            "--farm-blocks",
            str(model["farm_blocks"]),
            "--opponent-latents",
            str(model["opponent_latents"]),
            "--latents",
            str(model["latents"]),
            "--core-layers",
            str(model["core_layers"]),
            "--quantity-rank",
            str(model["quantity_rank"]),
            "--global-refresh-layers",
            ",".join(str(layer) for layer in model["global_refresh_layers"]),
            "--global-refresh-context",
            str(model["global_refresh_context"]),
            "--input-reinject-layers",
            ",".join(str(layer) for layer in model["input_reinject_layers"]),
            "--core-skip-source",
            str(model["core_skip_source"]),
            "--core-skip-target",
            str(model["core_skip_target"]),
            "--zero-init-branches",
            str(model["zero_init_branches"]).lower(),
            "--mudd-lite",
            str(model["mudd_lite"]).lower(),
            "--fuse-market-decoder",
            str(model["fuse_market_decoder"]).lower(),
            "--fuse-unit-decoder",
            str(model["fuse_unit_decoder"]).lower(),
            "--split-clock-token",
            str(model["split_clock_token"]).lower(),
            "--global-modulation",
            str(model["global_modulation"]).lower(),
            "--fused-mlp",
            str(model["fused_mlp"]).lower(),
            "--critic-core-layers",
            str(model["critic_core_layers"]),
            "--critic-latents",
            str(model["critic_latents"]),
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
            "--nextlat-max-gradient-norm",
            str(ppo["nextlat_max_gradient_norm"]),
            "--structured-decision-coefficient",
            str(ppo["structured_decision_coefficient"]),
            "--structured-patch-coefficient",
            str(ppo["structured_patch_coefficient"]),
            "--structured-economy-coefficient",
            str(ppo["structured_economy_coefficient"]),
            "--structured-opponent-summary-coefficient",
            str(ppo["structured_opponent_summary_coefficient"]),
            "--structured-opponent-patch-coefficient",
            str(ppo["structured_opponent_patch_coefficient"]),
            "--structured-decision-horizon",
            str(ppo["structured_decision_horizon"]),
            "--structured-patch-horizon",
            str(ppo["structured_patch_horizon"]),
            "--structured-critic-latent-coefficient",
            str(ppo["structured_critic_latent_coefficient"]),
            "--structured-critic-value-coefficient",
            str(ppo["structured_critic_value_coefficient"]),
            "--structured-critic-horizon",
            str(ppo["structured_critic_horizon"]),
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
