"""Production training configuration shared by every launch path."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaggriculture.constants import DEFAULT_REWARD_MODE
from kaggriculture.evaluation import DEVELOPMENT_SEED_START
from kaggriculture.modelargs import model_config_arguments
from kaggriculture.opponents import HELDOUT_REFERENCE_AGENTS, LEAGUE_REFERENCE_AGENTS
from kaggriculture.outcome_value import validate_outcome_objective
from kaggriculture.provenance import UNCOMPILED_UPDATE_COMPILE_MODE, repository_root
from kaggriculture.registry import LEJEPA, resolve_architecture
from kaggriculture.rollout import REWARD_MODES

# The attached LeJEPA world model under its own objective, with the family's
# default configuration. Its eight-epoch WDL clone already beats V27 97.5% of
# games at argmax, and it is the model every September 27 PPO stage fine-tuned
# (docs/experiments/core-model-2026-09-26.md; artifacts/probes/ppo-ablations-20260927).
PRODUCTION_ARCHITECTURE = LEJEPA
PRODUCTION_CRITIC_WARMUP_ITERATIONS = 10
PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS = 40

# A mirror self-play game contributes two current-policy trajectories; a
# frozen-league or reference-agent game contributes one. 128 * 2 : (64 + 40)
# is therefore about 71% current self-play and 29% past and reference
# opponents.
PRODUCTION_SELF_PLAY_GAMES = 128
PRODUCTION_LEAGUE_GAMES = 64
PRODUCTION_LEAGUE_SELECTION = "hardness"
PRODUCTION_LEAGUE_ACTIVE_OPPONENTS = 2
# Hardness pools the active/historical budget with the admitted built-in
# budget: with no built-ins, eight distinct opponents, including discovery and
# stale screening.
# The explicit stratified ablation uses two active and six log-age slots.
PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS = 6
PRODUCTION_LEAGUE_ACTIVE_POOL_SIZE = 16
# Native reference agents give the learner opponents outside its own lineage.
# In hardness mode these names compete with snapshots in one pool; this lane
# count adds to the total budget, and zero disables built-ins. In stratified
# mode each reserved built-in lane instead contests one additional snapshot
# using PFSP weights, releasing easy-agent lanes back to the active stratum.
# Production admits none: pass, random and starter are floors every clone
# clears, and scripted-v27 replays one hardcoded action per step, so none of
# them answers what the learner does. The dynamic reference agents below replace
# them. External evaluation remains a separate fixed diagnostic panel.
PRODUCTION_LEAGUE_BUILTIN_OPPONENTS = ""
PRODUCTION_LEAGUE_BUILTIN_LANES = 0
# Fixed league lanes against public agents that react to the state
# (`kaggriculture.opponents.LEAGUE_REFERENCE_AGENTS`), played natively with
# exact official-engine parity, split evenly over agents and seats and on top of
# the snapshot league. Eight games per agent per wave.
PRODUCTION_LEAGUE_SCRIPT_OPPONENTS = LEAGUE_REFERENCE_AGENTS
PRODUCTION_LEAGUE_SCRIPT_GAMES = 8 * len(LEAGUE_REFERENCE_AGENTS)
# Script games go two per agent, the rest by (1 - estimated score)^2, so an
# agent the learner beats every time stops taking games from ones it does not.
PRODUCTION_LEAGUE_SCRIPT_ALLOCATION = "hardness"
# Two of the eight hardness lanes screen the most uncertain stale snapshots two
# games apiece instead of refreshing one with a lane's eight. Simulated over a
# 1,300-iteration run, this found a forgotten opponent in about 6-9 iterations
# rather than about 70, and missed 52-77% of forgetting episodes rather than
# 94-100%, at no cost in how hard the hardness lanes' opponents were.
PRODUCTION_LEAGUE_SCREEN_LANES = 2
# The archive keeps the latest sixteen snapshots and older ones at spacing
# that doubles with age (`kaggriculture.league.retired_snapshots`), about 116
# of 1,300 iterations plus any the learner is not clearly beating, and deletes
# the rest, so a long run's disk use and selection pool stay bounded.
PRODUCTION_LEAGUE_ARCHIVE_RECENT = 16
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
# The opponents are the held-out reference agents
# (`kaggriculture.opponents.HELDOUT_REFERENCE_AGENTS`): the strongest public
# family and three others the league never plays, so the axis measures transfer
# to the ladder rather than progress against the lanes the learner trains on.
PRODUCTION_EXTERNAL_EVAL_OPPONENTS = ",".join(HELDOUT_REFERENCE_AGENTS)
# The fixed native development panel (`kaggriculture.architecture_panel`) every
# 25 actor-active waves: synchronous argmax and sampled strength against starter
# and V27 plus the critic's calibration, the milestone cadence the September 27
# stages were compared on. It culls only a run that has lost strength from its
# initialization and stalled on both signals; it never selects.
PRODUCTION_ARCHITECTURE_PANEL = 25


def production_architecture_panel(
    *,
    population: int,
    autocull: bool,
    device: str,
    update_compile_mode: str,
    reward_mode: str,
    gamma: float,
) -> int:
    """The panel interval a launch defaults to: production's wherever it can run.

    The panel scores one learner in compiled CUDA BF16 and is calibrated on
    undiscounted terminal outcomes, and it is the other stop rule beside
    autocull. A launch outside those conditions -- a population, a CPU or eager
    update, a shaped or discounted objective -- defaults to no panel, as every
    run did before it was adopted; an explicit interval there is still refused.
    """
    hosted = (
        population == 1
        and not autocull
        and device.startswith("cuda")
        and update_compile_mode != UNCOMPILED_UPDATE_COMPILE_MODE
        and reward_mode == "terminal-outcome"
        and gamma == 1.0
    )
    return PRODUCTION_ARCHITECTURE_PANEL if hosted else 0


def production_model_config() -> dict[str, Any]:
    """Return the complete JSON-persisted production model contract."""
    return resolve_architecture(PRODUCTION_ARCHITECTURE).config_class().to_dict()


def production_ppo_config(
    *,
    update_compile_mode: str,
    architecture: str = PRODUCTION_ARCHITECTURE,
    critic_architecture: str | None = None,
) -> dict[str, int | float | bool | str | None]:
    """The schedule the calibrated launcher runs and every benchmark measures.

    `PpoConfig` owns the algorithm's defaults and `family_ppo_defaults` the
    family's objective and backbone rate; for the production family together
    they are the measured recipe of artifacts/probes/ppo-stage2-20260927, arm
    lambda-1-actor-lr-5e-5. Another family, as a historical campaign names,
    takes the same schedule under its own family's settings; an unstated
    critic is that family's default. Production is one actor epoch and one
    critic epoch on the same wave:
    a second same-wave critic pass memorized holdout, and a second actor pass
    is a replay at a KL that does not bind. Actor and critic run on the same
    CUDA stream to reuse their activation allocation pool without eviction.
    Both NextLat objectives are opt-in. Ordinary PPO shuffles individual states;
    an active self-predictive objective, NextLat or LeJEPA, instead groups
    contiguous episode runs for successor targets.
    """
    from kaggriculture.ppo import PpoConfig, family_ppo_defaults

    return asdict(
        PpoConfig(
            **family_ppo_defaults(architecture, critic_architecture),
            critic_epochs=PpoConfig.epochs,
            target_kl=PpoConfig.target_kl,
            update_compile_mode=update_compile_mode,
            # Preserve the head's boost over ordinary Adam groups as base rates change.
            critic_head_learning_rate=(
                PpoConfig.critic_learning_rate * PpoConfig.adam_learning_rate_ratio * (25.0 / 3.0)
            ),
            # Cross-run evidence favors ordinary value fitting: critic NextLat
            # adds little prediction beyond persistence, while the off recipe
            # preserves substantially more deployed strength. See
            # docs/experiments/run-comparison-2026-09-18.md for the evidence and confounds. The
            # `lejepa` world-model objective excludes them regardless, and the
            # terms' horizons, inert while they are off, keep `PpoConfig`'s
            # defaults as the measured recipe did.
            structured_latent_coefficient=0.0,
            structured_decision_coefficient=0.0,
            structured_critic_latent_coefficient=0.0,
            structured_critic_value_coefficient=0.0,
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
    reward_mode: str = DEFAULT_REWARD_MODE,
    architecture: str = PRODUCTION_ARCHITECTURE,
    model_config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build the exact production train_ppo.py invocation.

    `architecture` and `model_config` (that family's defaults when omitted)
    let a campaign run the production schedule on another family; the command
    then carries that family's own model flags and PPO settings.
    """
    if (expected_source_digest is None) != (calibration_decision is None):
        raise ValueError("source digest and calibration decision must be provided together")
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}")
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
    family = resolve_architecture(architecture)
    model = dict(model_config) if model_config is not None else family.config_class().to_dict()
    ppo = production_ppo_config(
        update_compile_mode=update_compile_mode,
        architecture=architecture,
        critic_architecture=model.get("critic_architecture"),
    )
    if model.get("wdl_value"):
        validate_outcome_objective(reward_mode, ppo["gamma"], ppo["critic_gae_lambda"])
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
        "--reward-mode",
        reward_mode,
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
    panel = production_architecture_panel(
        population=population,
        autocull=False,
        device="cuda",
        update_compile_mode=update_compile_mode,
        reward_mode=reward_mode,
        gamma=float(ppo["gamma"]),
    )
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
            "--league-selection",
            PRODUCTION_LEAGUE_SELECTION,
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
            *(
                part
                for name in (PRODUCTION_LEAGUE_SCRIPT_OPPONENTS if league else ())
                for part in ("--league-script-opponent", name)
            ),
            "--league-script-games",
            str(PRODUCTION_LEAGUE_SCRIPT_GAMES if league else 0),
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
            "--architecture-panel",
            str(panel),
            "--architecture",
            architecture,
            *model_config_arguments(family, model),
            "--actor-lr",
            str(ppo["actor_learning_rate"]),
            "--critic-lr",
            str(ppo["critic_learning_rate"]),
            "--critic-head-lr",
            str(ppo["critic_head_learning_rate"]),
            "--lr-warmup-steps",
            str(ppo["lr_warmup_steps"]),
            "--epochs",
            str(ppo["epochs"]),
            "--critic-epochs",
            str(ppo["critic_epochs"]),
            "--minibatch-size",
            str(ppo["minibatch_size"]),
            "--policy-loss-reduction",
            str(ppo["policy_loss_reduction"]),
            "--policy-ratio-scope",
            str(ppo["policy_ratio_scope"]),
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
            "--structured-latent-coefficient",
            str(ppo["structured_latent_coefficient"]),
            "--structured-decision-coefficient",
            str(ppo["structured_decision_coefficient"]),
            "--structured-decision-horizon",
            str(ppo["structured_decision_horizon"]),
            "--structured-critic-latent-coefficient",
            str(ppo["structured_critic_latent_coefficient"]),
            "--structured-critic-value-coefficient",
            str(ppo["structured_critic_value_coefficient"]),
            "--economic-forecast-coefficient",
            str(ppo["economic_forecast_coefficient"]),
            "--structured-critic-horizon",
            str(ppo["structured_critic_horizon"]),
        )
    )
    # The world-model objective exists only in `lejepa`; any other family
    # refuses it, and the parser resolves it to its absence there.
    if architecture == LEJEPA:
        command.extend(
            (
                "--jepa-prediction-coefficient",
                str(ppo["jepa_prediction_coefficient"]),
                "--jepa-sigreg-coefficient",
                str(ppo["jepa_sigreg_coefficient"]),
                "--jepa-reward-coefficient",
                str(ppo["jepa_reward_coefficient"]),
                "--jepa-horizon",
                str(ppo["jepa_horizon"]),
            )
        )
    # None is "the actor's rate", which has no flag spelling; the parser's own
    # default states it.
    if ppo["structured_learning_rate"] is not None:
        command.extend(("--structured-learning-rate", str(ppo["structured_learning_rate"])))
    command.append(
        "--rematerialize-actor-update"
        if ppo["rematerialize_actor_update"]
        else "--no-rematerialize-actor-update"
    )
    command.append(
        "--structured-critic-gradient-balance"
        if ppo["structured_critic_gradient_balance"]
        else "--no-structured-critic-gradient-balance"
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
