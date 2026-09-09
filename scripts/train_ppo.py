#!/usr/bin/env python3
"""Train a from-scratch Kaggriculture policy with self-play PPO."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from kaggriculture.evaluation import (
    DEVELOPMENT_SEED_START,
    ONLINE_RL_SEED_START,
    artifact_seed_usage,
    validate_seed_interval,
)
from kaggriculture.inference import load_actor_artifact
from kaggriculture.league import (
    PFSP_UNMEASURED_SCORE_RATE,
    BuiltinSelection,
    FrozenActorPool,
    LeagueSelection,
    SnapshotRef,
    SnapshotSelection,
    copy_actor_snapshot,
    list_actor_snapshots,
    load_actor_snapshot,
    save_actor_snapshot,
    save_actor_state_snapshot,
    select_league_mix,
    snapshot_sha256,
)
from kaggriculture.model import ModelConfig, parameter_count
from kaggriculture.modelargs import add_model_config_arguments, model_config_from_args
from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.policy import mean_off_diagonal, population_disagreement
from kaggriculture.ppo import (
    DEFAULT_ACTOR_GAE_LAMBDA,
    DEFAULT_CRITIC_GAE_LAMBDA,
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    MAX_VALUE_TARGET_SATURATED_FRACTION,
    UPDATE_COMPILE_MODES,
    Actor,
    PpoConfig,
    actor_forward_args,
    make_optimizers,
    make_structured_dynamics_optimizer,
    update_ppo,
    update_replay_parity,
)
from kaggriculture.production import (
    PRODUCTION_CRITIC_WARMUP_ITERATIONS,
    PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
)
from kaggriculture.provenance import (
    CALIBRATION_KNOBS,
    file_sha256,
    require_source_identity,
    run_provenance_from_decision,
    source_identity,
    validate_run_provenance,
)
from kaggriculture.registry import ARCHITECTURES, CONV_ENTITY, STRUCTURED, resolve_architecture
from kaggriculture.rollout import (
    ROLLOUT_FORWARD_MODES,
    RolloutBatch,
    allocate_rollout_storage,
    collect_mixed_play_rust,
    collect_population_play_rust,
    slice_trajectories,
)
from kaggriculture.rust_env import toolchain_identity
from kaggriculture.structured import StructuredConfig
from kaggriculture.structured_dynamics import StructuredCriticDynamics, StructuredDynamics
from kaggriculture.telemetry import (
    TensorboardMirror,
    population_agent_field,
    population_disagreement_field,
    population_disagreement_pair_field,
    population_head_to_head_field,
    read_jsonl_snapshot,
)
from kaggriculture.training import (
    TrainingAgent,
    append_iteration_jsonl,
    checkpoint_payload,
    cpu_state_copy,
    install_immutable_checkpoint,
    load_checkpoint,
    metrics_journal_iteration,
    replace_checkpoint_alias,
    rollout_diagnostics,
    save_checkpoint,
    training_rng_states,
    write_immutable_checkpoint,
)

MIN_CHECKPOINT_SECONDS = 300.0
MAX_CHECKPOINT_SECONDS = 600.0
DEFAULT_CHECKPOINT_SECONDS = 420.0
DEFAULT_CRITIC_EPOCHS = PpoConfig.epochs
DEFAULT_CRITIC_WARMUP_ITERATIONS = PRODUCTION_CRITIC_WARMUP_ITERATIONS
CRITIC_WARMUP_READY_MONTE_CARLO_EV = 0.10
MAX_CRITIC_WARMUP_ITERATIONS = PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS


class RecoveryCheckpointTimer:
    """Monotonic periodic checkpoint timer, advanced at committed boundaries."""

    def __init__(self, interval_seconds: float, *, clock: Callable[[], float]) -> None:
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._last_committed_at = clock()

    def due(self, now: float | None = None) -> bool:
        current = self._clock() if now is None else now
        return current - self._last_committed_at >= self.interval_seconds

    def committed(self, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        if current < self._last_committed_at:
            raise ValueError("monotonic checkpoint clock moved backwards")
        self._last_committed_at = current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help=f"games per wave; defaults to {PRODUCTION_SELF_PLAY_GAMES} for a single "
        "learner and 13 copies of every ordered pairing for a population",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=1,
        help=(
            "concurrently learning agents; every game pairs two distinct members "
            "and every seat optimizes its own absolute economy. 1 is the single "
            "learner with mirror self-play and the frozen league, and N > 1 "
            "replaces both: --games must then be a multiple of N * (N - 1) so "
            "every ordered pairing appears equally often and seat bias cancels"
        ),
    )
    parser.add_argument(
        "--league-games",
        type=int,
        default=None,
        help=f"frozen-league games per wave; defaults to {PRODUCTION_LEAGUE_GAMES} for "
        "a single learner and 0 for a population (which has no frozen lane)",
    )
    parser.add_argument("--league-active-opponents", type=int, default=2)
    parser.add_argument(
        "--league-historical-opponents",
        type=int,
        default=6,
        help=(
            "log-age PFSP snapshot lanes per wave; defaults to 6 so a 500-iteration "
            "archive fills every occupied log2 rung outside the active window"
        ),
    )
    parser.add_argument("--league-active-pool-size", type=int, default=16)
    parser.add_argument(
        "--league-builtin-opponents",
        default="",
        help="comma-separated engine reference agents admitted to the training league",
    )
    parser.add_argument(
        "--league-builtin-lanes",
        type=int,
        default=0,
        help="league lanes reserved for admitted built-ins; unwon lanes go to snapshots",
    )
    parser.add_argument(
        "--external-eval",
        action="store_true",
        help="evaluate each committed recovery checkpoint against external agents",
    )
    parser.add_argument(
        "--external-eval-opponents",
        default="starter,public-v27",
        help="comma-separated opponents forwarded to external_eval_worker.py",
    )
    parser.add_argument("--external-eval-seeds", type=int, default=2)
    parser.add_argument("--external-eval-seed-start", type=int, default=DEVELOPMENT_SEED_START)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=ONLINE_RL_SEED_START)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=DEFAULT_CHECKPOINT_SECONDS,
        help="seconds between periodic recovery checkpoints (300-600)",
    )
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=CONV_ENTITY,
        help="actor/critic family; each family's structural flags default to that "
        "family's model configuration and a flag from another family is rejected",
    )
    add_model_config_arguments(parser)
    # Direct launches read PPO defaults from the algorithm configuration rather
    # than maintaining a second schedule in this parser.
    parser.add_argument("--actor-lr", type=float, default=PpoConfig.actor_learning_rate)
    parser.add_argument("--critic-lr", type=float, default=PpoConfig.critic_learning_rate)
    parser.add_argument("--lr-warmup-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=PpoConfig.epochs)
    parser.add_argument(
        "--critic-epochs",
        type=int,
        default=DEFAULT_CRITIC_EPOCHS,
        help="total critic epochs (>= --epochs; defaults to the same one pass)",
    )
    parser.add_argument("--minibatch-size", type=int, default=PpoConfig.minibatch_size)
    parser.add_argument("--clip-low", type=float, default=0.80)
    parser.add_argument("--clip-high", type=float, default=1.28)
    parser.add_argument(
        "--gamma",
        type=float,
        default=PpoConfig.gamma,
        help="shared reward-shaping and PPO discount; defaults to 0.997",
    )
    parser.add_argument(
        "--actor-gae-lambda",
        type=float,
        default=DEFAULT_ACTOR_GAE_LAMBDA,
        help=(
            "policy GAE lambda; defaults to VAPO's formula 1-1/(0.05*719) at the "
            "fixed competition horizon. Critic targets use --critic-gae-lambda"
        ),
    )
    parser.add_argument(
        "--critic-gae-lambda",
        type=float,
        default=DEFAULT_CRITIC_GAE_LAMBDA,
        help="critic GAE lambda; defaults to 1.0 (VAPO decoupled GAE, unbiased return)",
    )
    parser.add_argument("--target-kl", type=float, default=PpoConfig.target_kl)
    # Sourced from the dataclass rather than restated, so the justification
    # recorded there cannot drift out of agreement with what the CLI ships.
    parser.add_argument(
        "--optimizer",
        choices=("normuon", "adamw"),
        default=PpoConfig.optimizer,
        help=(
            "matrix optimizer: 'normuon' spectrally normalizes every hidden "
            "matrix's update and leaves gains, biases and heads on Adam, "
            "'adamw' is element-wise throughout. The learning rates above are "
            "in the chosen optimizer's units -- a NorMuon rate is the fraction "
            "of itself a matrix moves per step, an Adam rate is a per-element "
            "step, and they differ by sqrt(fan_in)"
        ),
    )
    parser.add_argument("--nextlat-max-gradient-norm", type=float, default=1.0)
    # Two phases, two knobs, decided separately by calibration: the collection
    # forward and the update compile different graphs, and their measured
    # speedups on the conv model fall on opposite sides of the threshold.
    # Neither knob is a boolean, and for the same reason on both sides -- the
    # decision is which execution mode, and the modes are not one measurement.
    # On the collection side that is measured: on 112-game waves with
    # a real BC actor the isolated forward runs 4.907 ms eager, 5.309 ms
    # cudagraphs and 2.720 ms inductor in fp32, so the old boolean's
    # `cudagraphs` was a pessimization dressed as an optimization. On the update
    # side the modes differ in whether they capture CUDA graphs and whether they
    # benchmark kernel selection, so a boolean could not have said which of the
    # four ran even when it said `true`.
    parser.add_argument(
        "--update-compile-mode",
        choices=UPDATE_COMPILE_MODES,
        default="default",
        help="execution mode of the update-path forward/backward, and the whole compile "
        "decision for the update: eager does not compile; default `default`, Inductor "
        "fusion without CUDA graph capture",
    )
    parser.add_argument("--no-bfloat16", action="store_true")
    # Collection execution and precision are explicit and independent of the
    # update backend. Production uses a collector-owned whole-wave CUDA graph
    # and a native-BF16 structured inference replica; the trainable actor and
    # quantity heads remain FP32. Replay-parity gates enforce the numerical
    # contract rather than assuming an execution backend is equivalent.
    parser.add_argument(
        "--rollout-forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        default=PRODUCTION_ROLLOUT_FORWARD_MODE,
        help="collection forward backend; defaults to the measured explicit CUDA graph "
        "path, which avoids TorchInductor's threaded cold-start deadlock",
    )
    parser.add_argument(
        "--rollout-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run structured collection with a native bf16 inference replica; default enabled",
    )
    parser.add_argument(
        "--deterministic-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "require deterministic PyTorch/CUDA algorithms and disable cuDNN "
            "benchmarking; recorded in checkpoints and fixed across resume"
        ),
    )
    parser.add_argument(
        "--structured-decision-coefficient",
        type=float,
        default=PpoConfig.structured_decision_coefficient,
        help="weight on structured future-decision decode KL",
    )
    parser.add_argument(
        "--structured-patch-coefficient",
        type=float,
        default=PpoConfig.structured_patch_coefficient,
        help="weight on normalized future own-patch feature L1",
    )
    parser.add_argument(
        "--structured-economy-coefficient",
        type=float,
        default=PpoConfig.structured_economy_coefficient,
        help="weight on normalized future economy-entity feature L1",
    )
    parser.add_argument(
        "--structured-opponent-summary-coefficient",
        type=float,
        default=PpoConfig.structured_opponent_summary_coefficient,
        help="weight on normalized future opponent-summary feature L1",
    )
    parser.add_argument(
        "--structured-opponent-patch-coefficient",
        type=float,
        default=PpoConfig.structured_opponent_patch_coefficient,
        help="weight on normalized future opponent-patch feature L1",
    )
    parser.add_argument(
        "--structured-critic-latent-coefficient",
        type=float,
        default=PpoConfig.structured_critic_latent_coefficient,
        help="weight on critic-belief NextLat latent prediction",
    )
    parser.add_argument(
        "--structured-critic-value-coefficient",
        type=float,
        default=PpoConfig.structured_critic_value_coefficient,
        help="weight on decoded future-value prediction from critic NextLat",
    )
    parser.add_argument(
        "--structured-decision-horizon",
        type=int,
        default=PpoConfig.structured_decision_horizon,
    )
    parser.add_argument(
        "--structured-patch-horizon",
        type=int,
        default=PpoConfig.structured_patch_horizon,
    )
    parser.add_argument(
        "--structured-critic-horizon",
        type=int,
        default=PpoConfig.structured_critic_horizon,
    )
    parser.add_argument(
        "--structured-learning-rate",
        type=float,
        default=PpoConfig.structured_learning_rate,
        help="actor predictor optimizer rate; defaults to --actor-lr",
    )
    parser.add_argument(
        "--structured-critic-learning-rate",
        type=float,
        default=PpoConfig.structured_critic_learning_rate,
        help="critic predictor optimizer rate; defaults to --critic-lr",
    )
    parser.add_argument(
        "--expected-source-digest",
        help="require the immutable source digest selected by the calibration launcher",
    )
    parser.add_argument(
        "--calibration-decision",
        type=Path,
        help="exact calibration decision to bind into every production checkpoint",
    )
    parser.add_argument(
        "--init-actor-from",
        type=Path,
        action="append",
        help="actor artifact (e.g. a BC clone) whose weights initialize a fresh run's "
        "actor; repeat it once per --population member, since four agents from one "
        "checkpoint are numerically identical and their games are mirrors scoring 0",
    )
    parser.add_argument(
        "--critic-warmup-iterations",
        type=int,
        help=(
            "minimum critic-only iterations before actor release; after this floor, "
            "the actor remains frozen until the previous fresh-wave pre-update Monte "
            f"Carlo-return EV reaches {CRITIC_WARMUP_READY_MONTE_CARLO_EV:.2f}; "
            f"defaults to {DEFAULT_CRITIC_WARMUP_ITERATIONS} for a fresh warm start"
        ),
    )
    args = parser.parse_args()
    if args.init_actor_from is not None and args.critic_warmup_iterations is None:
        args.critic_warmup_iterations = DEFAULT_CRITIC_WARMUP_ITERATIONS
    # A population wave is all learners; the single-learner defaults (128 live
    # games plus 64 frozen) are not a valid population configuration, so they
    # must not be the implicit ones. An explicit flag still wins either way.
    if args.games is None:
        args.games = (
            args.population * (args.population - 1) * 13
            if args.population > 1
            else PRODUCTION_SELF_PLAY_GAMES
        )
    if args.league_games is None:
        args.league_games = 0 if args.population > 1 else PRODUCTION_LEAGUE_GAMES
    return args


def _recorded_argument(value: object) -> object:
    """One parsed CLI argument as `config.json` records it.

    `--init-actor-from` is repeatable, so its value is a list of paths where every
    other Path-valued flag is scalar, and both have to reach JSON as strings.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_recorded_argument(entry) for entry in value]
    return value


def _initial_actor_paths(args: argparse.Namespace) -> list[Path]:
    """The pretrained artifacts a fresh run's members start from, in agent order."""
    return list(args.init_actor_from or ())


def _validate_population(args: argparse.Namespace) -> None:
    """Refuse a population whose members cannot produce a usable gradient.

    Three silent defects are refused. A wave whose game count is not a multiple
    of N(N-1) cannot give every ordered pairing the same number of games, so seat
    bias survives into the advantage. Members initialized from the same artifact
    are one policy rather than four learners. And a configured frozen or built-in
    lane would violate the population collector's all-learners contract.
    """
    initial = _initial_actor_paths(args)
    resolved = [path.expanduser().resolve() for path in initial]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "--init-actor-from must name a different artifact per agent; "
            "four learners require four independently trained initial policies"
        )
    if args.population == 1:
        if len(initial) > 1:
            raise ValueError(
                "a single learner has one actor to initialize; --population admits more"
            )
        return
    pairings = args.population * (args.population - 1)
    if args.games % pairings:
        raise ValueError(
            f"a population of {args.population} has {pairings} ordered pairings, so "
            f"--games must be a multiple of {pairings}; {args.games} is not"
        )
    if initial and len(initial) != args.population:
        raise ValueError(
            f"--population {args.population} takes one --init-actor-from per agent or "
            f"none at all; {len(initial)} were given"
        )
    # Every seat in a population wave is a learner, so nothing in it reads the
    # frozen archive or plays an engine reference agent. Leaving either configured
    # would accept a launch command claiming opponents the run never meets.
    if args.league_games or args.league_builtin_lanes or _league_builtin_opponents(args):
        raise ValueError(
            "a population wave has no frozen or built-in lanes; pass --league-games 0 "
            "and no built-in opponents"
        )


def _validate_args(args: argparse.Namespace) -> None:
    # Model-configuration flags are validated by the config dataclass itself,
    # so every entry point that builds one gets the same rules.
    positive = {
        "iterations": args.iterations,
        "population": args.population,
        "games": args.games,
        "league_active_pool_size": args.league_active_pool_size,
        "episode_steps": args.episode_steps,
        "checkpoint_seconds": args.checkpoint_seconds,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
    }
    if args.critic_epochs is not None:
        positive["critic_epochs"] = args.critic_epochs
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {', '.join(invalid)}")
    if (
        not math.isfinite(args.checkpoint_seconds)
        or not MIN_CHECKPOINT_SECONDS <= args.checkpoint_seconds <= MAX_CHECKPOINT_SECONDS
    ):
        raise ValueError(
            "checkpoint seconds must be finite and between "
            f"{MIN_CHECKPOINT_SECONDS:g} and {MAX_CHECKPOINT_SECONDS:g}"
        )
    if args.episode_steps != 720:
        raise ValueError("training requires the competition horizon: --episode-steps 720")
    # The k3 estimator is non-negative, and the trust region stops on
    # `batch_kl > target_kl`, so zero admits only the exactly-parity first
    # minibatch and anything negative admits nothing at all. Either collapses
    # the update to near-zero optimizer steps silently rather than erroring.
    if not math.isfinite(args.target_kl) or args.target_kl <= 0.0:
        raise ValueError("target KL must be finite and positive")
    structured_actor_coefficients = (
        args.structured_decision_coefficient,
        args.structured_patch_coefficient,
        args.structured_economy_coefficient,
        args.structured_opponent_summary_coefficient,
        args.structured_opponent_patch_coefficient,
    )
    structured_critic_coefficients = (
        args.structured_critic_latent_coefficient,
        args.structured_critic_value_coefficient,
    )
    structured_coefficients = structured_actor_coefficients + structured_critic_coefficients
    if not all(math.isfinite(value) and value >= 0.0 for value in structured_coefficients):
        raise ValueError("structured auxiliary coefficients must be finite and nonnegative")
    structured_active = any(structured_coefficients)
    if structured_active and args.architecture != STRUCTURED:
        raise ValueError("structured auxiliary coefficients require --architecture structured")
    structured_horizons = (
        args.structured_decision_horizon,
        args.structured_patch_horizon,
        args.structured_critic_horizon,
    )
    if any(horizon < 0 for horizon in structured_horizons):
        raise ValueError("structured auxiliary horizons cannot be negative")
    if args.structured_decision_coefficient and args.structured_decision_horizon < 1:
        raise ValueError("structured decision horizon must be positive when decision KL is active")
    if any(structured_actor_coefficients[1:]) and args.structured_patch_horizon < 1:
        raise ValueError(
            "structured patch horizon must be positive when feature prediction is active"
        )
    if any(structured_critic_coefficients) and args.structured_critic_horizon < 1:
        raise ValueError(
            "structured critic horizon must be positive when critic auxiliary is active"
        )
    if args.structured_learning_rate is not None and (
        not math.isfinite(args.structured_learning_rate) or args.structured_learning_rate <= 0.0
    ):
        raise ValueError("structured learning rate must be finite and positive")
    if args.structured_critic_learning_rate is not None and (
        not math.isfinite(args.structured_critic_learning_rate)
        or args.structured_critic_learning_rate <= 0.0
    ):
        raise ValueError("structured critic learning rate must be finite and positive")
    if args.league_games < 0:
        raise ValueError("league games cannot be negative")
    if args.league_active_opponents < 0 or args.league_historical_opponents < 0:
        raise ValueError("league opponent counts cannot be negative")
    if args.league_builtin_lanes < 0:
        raise ValueError("league built-in lane budget cannot be negative")
    builtins = _league_builtin_opponents(args)
    unknown = sorted(set(builtins) - BUILTIN_OPPONENTS)
    if unknown:
        raise ValueError(f"unknown built-in league opponents: {', '.join(unknown)}")
    if len(set(builtins)) != len(builtins):
        raise ValueError("built-in league opponents must be distinct")
    # A reserved lane with nothing admitted to fill it is silently nothing, and
    # admitted agents with no reserved lane never play. Either is a launch
    # command that does not mean what it says.
    if bool(builtins) != bool(args.league_builtin_lanes):
        raise ValueError(
            "--league-builtin-opponents and --league-builtin-lanes must be set together"
        )
    configured_opponents = (
        args.league_active_opponents
        + args.league_historical_opponents
        + min(args.league_builtin_lanes, len(builtins))
    )
    if args.league_games and not configured_opponents:
        raise ValueError("league games require at least one active, historical, or built-in lane")
    if args.external_eval and args.external_eval_seeds < 1:
        raise ValueError("external evaluation needs positive seeds")
    if args.external_eval:
        validate_seed_interval(
            "development", args.external_eval_seed_start, args.external_eval_seeds
        )
    if args.league_games and args.league_games < configured_opponents:
        raise ValueError(
            "league games must cover the initial anchor and every configured "
            f"active/historical/built-in lane ({configured_opponents})"
        )
    _validate_population(args)
    if args.critic_warmup_iterations is not None and args.critic_warmup_iterations < 0:
        raise ValueError("critic warmup iterations cannot be negative")
    if args.init_actor_from is not None and args.resume is not None:
        raise ValueError(
            "--init-actor-from initializes a fresh run; a resumed run's actor "
            "comes from its checkpoint"
        )
    # The warmup count is part of the warm start and is persisted with it, so a
    # resume restores it rather than restating it. Without that, a crash inside
    # the warmup window would silently relaunch with no warmup at all -- the
    # actor would start taking steps against a critic that never finished
    # fitting, which is the single failure this protocol exists to prevent.
    if args.critic_warmup_iterations is not None and args.resume is not None:
        raise ValueError(
            "--critic-warmup-iterations belongs to the run being resumed and is "
            "restored from its checkpoint"
        )
    if args.critic_warmup_iterations is not None and not _initial_actor_paths(args):
        raise ValueError("critic warmup applies only to a warm-started run")
    # A warmup at least as long as the run freezes the actor for its whole
    # life, and the stalled-actor guard that would otherwise catch zero actor
    # updates is deliberately suppressed while the warmup is active, so the
    # run would end silently identical to the clone it started from.
    if (
        args.critic_warmup_iterations is not None
        and args.critic_warmup_iterations >= args.iterations
    ):
        raise ValueError("critic warmup must leave iterations for the actor to train in")
    if (
        args.critic_warmup_iterations is not None
        and args.critic_warmup_iterations > MAX_CRITIC_WARMUP_ITERATIONS
    ):
        raise ValueError(
            f"critic warmup minimum cannot exceed the {MAX_CRITIC_WARMUP_ITERATIONS}-iteration "
            "readiness deadline"
        )
    if not 0 < args.clip_low < 1 < args.clip_high:
        raise ValueError("clip interval must straddle one")
    if args.lr_warmup_steps < 0:
        raise ValueError("LR warmup steps cannot be negative")
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not math.isfinite(args.actor_gae_lambda) or not 0.0 <= args.actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")
    if not math.isfinite(args.critic_gae_lambda) or not 0.0 <= args.critic_gae_lambda <= 1.0:
        raise ValueError("critic GAE lambda must be finite and in [0, 1]")
    if not math.isfinite(args.max_hours) or args.max_hours < 0.0:
        raise ValueError("max hours must be finite and non-negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    validate_seed_interval(
        "online_rl",
        args.seed,
        max(1, (args.games + (args.league_games if args.population == 1 else 0)) * args.iterations),
    )
    if args.temperature != 1.0:
        raise ValueError("on-policy PPO currently requires --temperature 1.0")
    if args.expected_source_digest is not None and (
        len(args.expected_source_digest) != 64
        or any(character not in "0123456789abcdef" for character in args.expected_source_digest)
    ):
        raise ValueError("expected source digest must be 64 lowercase hexadecimal characters")
    if (args.expected_source_digest is None) != (args.calibration_decision is None):
        raise ValueError(
            "expected source digest and calibration decision must be supplied together"
        )


def _load_initial_actor(
    path: Path,
    actor: torch.nn.Module,
    architecture_name: str,
    model_config: ModelConfig | StructuredConfig,
    device: torch.device,
) -> dict[str, object]:
    """Initialize a fresh run's actor from a pretrained artifact (BC warm start).

    The artifact must carry exactly this run's model configuration. The critic
    and both optimizers deliberately start fresh — a clone brings no value
    function — and the pre-loop league snapshot then seeds the frozen-opponent
    archive with the pretrained policy automatically, so the learner must keep
    beating its own starting point.
    """
    pretrained, payload = load_actor_artifact(path, device)
    artifact_architecture = resolve_architecture(payload)
    if artifact_architecture.name != architecture_name:
        raise ValueError("initial actor artifact architecture does not match arguments")
    artifact_config = artifact_architecture.build_config(payload["model_config"]).to_dict()
    if artifact_config != model_config.to_dict():
        raise ValueError("initial actor artifact model configuration does not match arguments")
    actor.load_state_dict(pretrained.state_dict())
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "format_version": payload["format_version"],
        "iteration": int(payload.get("iteration", 0)),
        "bc_provenance": payload.get("bc_provenance"),
        "seed_usage": artifact_seed_usage(payload),
        # Equality with the current tree would be unusable here -- any edit
        # changes it, and the clone is deliberately produced before the run.
        # Recording which tree tokenized the demonstrations is free, and
        # without it the checkpoint cannot say where its weights came from.
        "source_identity": payload.get("source_identity"),
    }


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


def _balanced_assignments(
    games: int,
    opponents: int,
    generator: np.random.Generator,
    *,
    seed_start: int = 0,
) -> np.ndarray:
    """Balance total games and seed-determined seats for every opponent."""
    if games < 1 or opponents < 1:
        raise ValueError("balanced assignments require positive games and opponents")

    totals = np.bincount(np.arange(games, dtype=np.int64) % opponents, minlength=opponents)
    seat_zero_games = (games + (seed_start % 2 == 0)) // 2
    seat_zero_counts = totals // 2
    odd_opponents = np.flatnonzero(totals % 2)
    extras = seat_zero_games - int(seat_zero_counts.sum())
    if extras:
        odd_opponents = generator.permutation(odd_opponents)
        seat_zero_counts[odd_opponents[:extras]] += 1
    seat_one_counts = totals - seat_zero_counts

    assignments = np.empty(games, dtype=np.int64)
    seats = (seed_start + np.arange(games, dtype=np.int64)) % 2
    for seat, counts in ((0, seat_zero_counts), (1, seat_one_counts)):
        values = np.repeat(np.arange(opponents, dtype=np.int64), counts)
        generator.shuffle(values)
        assignments[seats == seat] = values
    return assignments


# Blend of the previous estimate and this iteration's measured score rate for
# PFSP opponent weighting; each measurement covers only ~20-50 games, so the
# estimate keeps some memory while still down-weighting a beaten opponent
# quickly. The prior seeds the blend, so a single clean sweep can never pin
# an estimate at exactly 1.0 and permanently retire an opponent.
LEAGUE_SCORE_RATE_EMA = 0.5
# Unsampled snapshot estimates decay toward the unmeasured prior because an
# increasingly stale frozen policy may no longer represent its last result.
# Built-ins never change, so their retirement evidence remains authoritative
# until they are sampled again.
LEAGUE_SCORE_RATE_DECAY = 0.05
# Iterations between sampling-vs-update replay audits. The divergence this
# audits is a function of how sharp the policy has become -- a randomly
# initialized actor measures it near 4e-7 where a behavior-cloned one measures
# 2e-3 -- so auditing only at iteration zero measures it at the one moment it
# is smallest and never rechecks as training sharpens the heads. The audit is
# a full extra replay forward over the rollout, roughly five percent of an
# iteration, so this cadence costs about two parts in a thousand.
REPLAY_PARITY_AUDIT_INTERVAL = 25
# How far a parity measurement must exceed the same staging configuration's
# previous measurement before a breach is read as a defect rather than drift.
#
# Two things push a breach past the bound and they call for opposite
# responses. A staging defect -- a buffer the wave failed to rewrite, a dtype
# that stopped matching, a compiled graph specialized on the wrong shape --
# is a step change: the divergence it produces has nothing to do with the
# divergence numerics produce, and the original failure that motivated this
# gate measured a likelihood error of 20.7 against a healthy 3e-3. Drift is
# the opposite: the divergence is a function of how sharp the heads have
# become, it grows monotonically as RL sharpens them, and it crosses the
# bound by a hair. So the discriminator is a derivative, not a level, and the
# comparison is against what this configuration measured one interval ago --
# which is why the previous measurement is checkpointed rather than held in
# process memory, where every restart would forget it.
#
# One consequence is worth stating outright because the naming hides it:
# MAX_UPDATE_REPLAY_KL is the launch check and the warning line, not the
# in-run abort line. A running process aborts at five times the previous
# audit, so with the clone's 1.9e-3 the real abort line is 9.5e-3. The bound
# is the fatal one only where there is no previous measurement -- a fresh run,
# or a staging configuration this run has never exercised.
#
# Five, because the two populations are separated by orders of magnitude and
# the factor only has to sit between them. On the drift side it cannot fire
# by accident: sustained 5x growth per 25 iterations compounds to 5^20, about
# 1e14, over a 500-iteration run, so no amount of real sharpening produces
# it. On the defect side the step it has to clear is the difference between a
# staging path that works and one that does not, which is not a factor of
# five. Note that the baseline advances after every audit including a warned
# one, so drift is always measured against recent drift and can never
# accumulate into a false defect.
#
# The blind spot the factor opens is a window in *severity*, and it is worth
# stating precisely because the tempting argument from extent is simply wrong.
# A component enters the tail statistic only once it disagrees by more than
# UPDATE_REPLAY_TAIL_LOGPROB, so a defect of any extent whatsoever is
# tail-silent while its severity stays under that, and a gate cannot escalate
# what it cannot see. That leaves arithmetic on the mean: a defect over a
# share f at severity d adds f * k3(d), so it warns rather than aborts
# whenever that lands between the bound and the step change, and the tail
# stays blind to it throughout while d <= 2.5. Against the clone's 1.9e-3 the
# smallest extent that reaches the band is f = 3.6e-4, some 530 components or
# an eighth of one trajectory, and the windows are wide: one step index across
# every trajectory warns silently from a 4.8x likelihood error to an 8.6x one,
# one trajectory in three hundred and twenty from 3.1x to 5.1x.
#
# What makes that acceptable is where the abort line falls, not that the
# window is empty. For every structural unit big enough to reach the band it
# sits at 1.6 to 2.2 nats, and a staging path that is genuinely broken --
# replaying one state's logits against another state's sampled action, on a
# policy sharp enough to measure 1.9e-3 at all -- disagrees by far more than
# that, which is also where the tail statistic starts seeing it. What fits
# inside the window is the mild, uniform perturbation, which is a description
# of numerics rather than of staging. And nothing in the band is silent: a
# warned breach prints and lands in telemetry next to the abort line it was
# judged against, so the run announces it within one interval.
#
# The band widens as the baseline drifts up, since the abort line is a
# multiple of it -- at a baseline of 4e-3 a defect may add 16e-3 and still only
# warn -- and that widening is what the ceiling below exists to stop. Up to the
# ceiling it is the price of the drift tolerance this rule buys, and the two
# genuinely cannot both be had: a gate aborts on every breach only where its
# bound sits at least this factor above its own baseline. That is false of the
# KL gate at 2.63 and true of the tail gate on every head -- 6.3x on the unit
# head's measured 3.2e-5, and far wider on the two small heads, whose floors
# are 3e-6 and 9e-6 -- so the unit head loses the property first, if its tail
# floor ever passes 4e-5. Which head is soft, and by how much, is exactly what
# the `_fatal_at` telemetry reports, so it is legible while a run is in
# progress rather than afterwards.
REPLAY_PARITY_STEP_CHANGE_FACTOR = 5.0
# The audited heads, and the two statistics gated on each of them.
#
# Per head, not on the max over heads, and the difference is load-bearing. The
# aggregate exists so a single-head defect cannot be diluted by the unit head,
# which carries 1.45M of the 1.9M active components; baselining the aggregate
# would hand that dilution straight back, because every head would then be
# judged against the largest head's level. Measured healthy KL is 1.9e-3 on
# the unit head against 8.0e-4 on the kind head, so a kind-head defect judged
# against the aggregate is excused to 11.9x its own healthy level rather than
# the 5x intended -- and a defect lifting kind to 8e-3 would warn, where
# against its own head it is a 10x step and aborts.
PARITY_COMPONENTS = ("unit", "kind", "quantity")
PARITY_STATISTICS: tuple[tuple[str, float, str], ...] = (
    ("kl", MAX_UPDATE_REPLAY_KL, "sampling-vs-update policy divergence exceeded"),
    # The KL is a mean and a localized defect dilutes into it, so the share of
    # materially disagreeing components is bounded separately.
    (
        "tail_fraction",
        MAX_UPDATE_REPLAY_TAIL_FRACTION,
        "sampling-vs-update materially divergent component share exceeded",
    ),
)
PARITY_STAGING_KEYS = ("self-play", "league", "population")


def _parity_metric_key(component: str, statistic: str) -> str:
    return f"update_replay_{component}_{statistic}"


def _parity_staging_key(league_games: int, population: int = 1) -> str:
    """Name the staging configuration a wave exercises.

    League play and pure self-play stage the rollout arena differently -- the
    full arena, or the self-play prefix view of it -- so the audit *cadence*
    tracks them separately. A cadence blind to the difference could audit
    whichever one the interval landed on while never examining the other,
    including the iteration where the league rows past the prefix are written
    for the first time.

    A population wave is a third configuration rather than the self-play one at
    a different width: its behaviour policy is one vmapped ensemble forward over
    N lanes where self-play is a plain module forward, which is precisely the
    staging difference the audit exists to measure.
    """
    if population > 1:
        return PARITY_STAGING_KEYS[2]
    return PARITY_STAGING_KEYS[1] if league_games else PARITY_STAGING_KEYS[0]


def _parity_audit_due(
    last_audits: Mapping[str, int],
    iteration: int,
    staging: str,
) -> bool:
    """Decide whether this iteration re-runs the sampling-vs-update audit.

    A configuration this process has not yet audited is always due, which is
    what makes a resumed process audit its first iteration: a resume rebuilds
    the compiled callables and restages every buffer, so it is exactly when a
    staging bug appears.
    """
    previous = last_audits.get(staging)
    return previous is None or iteration - previous >= REPLAY_PARITY_AUDIT_INTERVAL


def _parity_headroom(measured: float, previous: float, ceiling: float) -> str:
    """Describe how much room is left before this statistic becomes fatal.

    Warned drift otherwise reads as a stream of identical lines, which says
    that something is off but not whether it is settling or converging on the
    ceiling. The distance is only meaningful in units of the growth producing
    it, so it is reported as audits remaining at the rate the last two
    measured: with r = measured / previous, r^n reaches the ceiling at
    n = log(ceiling / measured) / log(r).

    Two points make a noisy slope, so this is labelled as an observed rate
    rather than offered as a prediction. Its purpose is a stop-at-a-checkpoint
    decision taken while the run is still healthy, instead of the same decision
    taken after the ceiling has already ended it. Recovery means a fresh run
    warm-started from the last actor either way, so the choice worth informing
    is when to take it, not whether.

    Only ever called for a warned breach, which bounds the arithmetic: warning
    requires measured <= factor * previous and measured > bound > 0, so
    previous >= measured / factor > 0 and the ratio is well defined.
    """
    rate = measured / previous
    if rate <= 1.0:
        return "not climbing toward the ceiling at the last observed rate"
    audits = math.log(ceiling / measured) / math.log(rate)
    return f"about {audits:.1f} audits of headroom at the last observed rate"


def _parity_ceilings() -> dict[str, float]:
    """Return the absolute level at which each statistic is a defect regardless.

    The step-change test alone cannot supply this. That test is a derivative,
    so it says nothing about level: a value growing by less than the factor at
    every audit is never fatal at any magnitude, and since the baseline
    advances after every warned audit the accepted level ratchets upward. Nine
    consecutive warned audits, 225 iterations at this cadence, take 1.9e-3 past
    1e3. Without a ceiling MAX_UPDATE_REPLAY_KL stops being a bound on accepted
    bias and becomes only the level at which warnings start.

    The ceiling is one full step change past the calibrated bound, stated
    purely in this gate's own terms: drift is tolerated up to the point where
    the accumulated excess equals what a single audit would have had to jump to
    be called a defect outright. Past that the drift story is no longer
    credible however gradually it arrived, because the run has quietly
    travelled the whole distance the step test exists to catch.

    It deliberately imports nothing. An earlier version derived this from the
    update's trust region -- std[d] = sqrt(2 * KL), so a parity KL of target_kl
    would mean the uncorrected divergence had the same width as the divergence
    the update allows and then corrects. The run telemetry falsifies that:
    across four runs, including a full 500-iteration one at target_kl = 0.03,
    realized approx_kl has a median of 2.3e-3 and a maximum of 5.4e-3, and the
    worst single minibatch ever recorded is 2.2e-2. target_kl is a safety valve
    the update never reaches, so a ceiling placed there would permit a std[d]
    of 0.245 against a realized 0.068. Worse, the same argument applied
    honestly to the realized movement puts the ceiling near 2.3e-3, *below*
    MAX_UPDATE_REPLAY_KL -- so the coherence argument cannot select a level at
    all. Deriving it from the step factor also keeps ceiling >= bound true by
    construction, which is what makes _validate_parity_baseline's invariant
    sound rather than merely true today: a ceiling under its own bound would
    let a never-breaching measurement be persisted and then rejected on the
    next resume, wedging the run at its first restart.

    What that telemetry does establish, and what this file should not imply
    otherwise: the healthy parity divergence is not negligible next to the
    update's real movement. The clone measures 1.9e-3 against a realized 2.3e-3
    per iteration -- the same number. The uncorrected divergence is already
    about as wide as the corrected one at the healthy operating point, which is
    why the bound is 5e-3 rather than anything looser, and why the drift
    tolerance above is a tolerance for measurement spread, not for growth.

    Unlike a step change, a ceiling breach is deliberately unrecoverable: a
    resume re-measures the same level and dies again, and there is no flag to
    raise the ceiling, because a gate whose last line can be waved through is a
    warning. A step change is a staging defect, which a resume genuinely might
    not reproduce; a run that has drifted a full step past its bound has
    instead reached a numerics regime nothing here can correct for, and the
    answer is a decision about the update forward -- fp32 rather than bf16 --
    not another attempt at the same configuration. That decision edits the
    tree, so the repaired run cannot resume these checkpoints; it warm-starts
    from the last actor instead, which is what the abort tells the operator.
    """
    return {
        statistic: REPLAY_PARITY_STEP_CHANGE_FACTOR * bound
        for statistic, bound, _description in PARITY_STATISTICS
    }


def _parity_fatal_thresholds(
    baseline: Mapping[str, float] | None,
    ceilings: Mapping[str, float],
) -> dict[str, float]:
    """Report the value at which each gated statistic actually aborts.

    Not the same number as the bound, which is why it belongs in telemetry
    rather than in someone's head. With no baseline the bound is the abort
    line; once a head has been measured the abort line floats up to the step
    change above it, capped by the ceiling. A head whose bound sits less than
    REPLAY_PARITY_STEP_CHANGE_FACTOR above its own baseline therefore has a
    band in which a breach only warns -- the KL gate's normal condition -- and
    plotting the measurement against this shows both that band and the rate it
    is opening.
    """
    thresholds: dict[str, float] = {}
    for component in PARITY_COMPONENTS:
        for statistic, bound, _description in PARITY_STATISTICS:
            key = _parity_metric_key(component, statistic)
            previous = None if baseline is None else baseline.get(key)
            step = (
                bound
                if previous is None
                else max(bound, REPLAY_PARITY_STEP_CHANGE_FACTOR * previous)
            )
            thresholds[f"{key}_fatal_at"] = min(step, ceilings[statistic])
    return thresholds


def _parity_breaches(
    metrics: Mapping[str, float | int],
    baseline: Mapping[str, float] | None,
    ceilings: Mapping[str, float],
) -> list[tuple[str, bool]]:
    """Report each breached parity bound, and whether it reads as a defect.

    A bound that is not breached is not reported at all: the step-change test
    only ever escalates a value that has already left the budget, so a jump
    inside the budget is a number for telemetry rather than a failure.

    A breach is a defect when it steps away from what this head last measured,
    when it passes the absolute ceiling however gradually it got there, or when
    there is no previous measurement to compare against -- the last covering a
    fresh run's first audit, where the measurement is the launch check itself
    and there is nothing yet to have drifted from. A non-finite measurement is
    a defect too, and explicitly so: it fails every ordered comparison, so
    without naming it the step-change test would silently read NaN as drift and
    write it into the baseline.
    """
    breaches: list[tuple[str, bool]] = []
    for component in PARITY_COMPONENTS:
        for statistic, bound, description in PARITY_STATISTICS:
            key = _parity_metric_key(component, statistic)
            measured = float(metrics[key])
            if math.isfinite(measured) and measured <= bound:
                continue
            message = f"{component} {description} {bound}: {measured}"
            if not math.isfinite(measured):
                breaches.append((message, True))
                continue
            previous = None if baseline is None else baseline.get(key)
            if previous is None:
                breaches.append((message, True))
                continue
            is_defect = (
                measured > REPLAY_PARITY_STEP_CHANGE_FACTOR * previous
                or measured > ceilings[statistic]
            )
            detail = f"previous audit {previous}, trend in {key}"
            if not is_defect:
                detail += f", {_parity_headroom(measured, previous, ceilings[statistic])}"
            breaches.append((f"{message} ({detail})", is_defect))
    return breaches


def _parity_measurements(metrics: Mapping[str, float | int]) -> dict[str, float]:
    """Extract the per-head values a later audit will be compared against."""
    return {
        _parity_metric_key(component, statistic): float(
            metrics[_parity_metric_key(component, statistic)]
        )
        for component in PARITY_COMPONENTS
        for statistic, _bound, _description in PARITY_STATISTICS
    }


def _validate_parity_baseline(
    baseline: object,
    ceilings: Mapping[str, float],
) -> dict[str, float]:
    """Validate the persisted previous audit, or return an empty baseline.

    Bounded by the same ceilings the live gate enforces, because a measurement
    above one can never have been persisted: the audit that produced it would
    have aborted before any checkpoint was written. A value above the ceiling
    in a checkpoint is therefore corruption, and accepting it would excuse
    every breach below five times it.

    That reasoning needs ceiling >= bound to hold, and it does only because
    _parity_ceilings derives the ceiling as a multiple of the bound. A ceiling
    below its bound would leave a window in which a measurement never breaches
    -- _parity_breaches gates on the bound, and the ceiling only escalates an
    already-breaching value -- yet is rejected here on the next resume, wedging
    the run at its first restart over a value the gate itself called healthy.
    """
    if not isinstance(baseline, dict):
        raise ValueError("resume checkpoint has no valid replay-parity baseline state")
    if not baseline:
        return {}
    expected = {
        _parity_metric_key(component, statistic)
        for component in PARITY_COMPONENTS
        for statistic, _bound, _description in PARITY_STATISTICS
    }
    if set(baseline) != expected:
        raise ValueError("resume checkpoint replay-parity baseline is incomplete")
    validated: dict[str, float] = {}
    for key, measurement in baseline.items():
        statistic = key.rsplit("update_replay_", 1)[1].split("_", 1)[1]
        if type(measurement) is not float or not 0.0 <= measurement <= ceilings[statistic]:
            raise ValueError("resume checkpoint has an invalid replay-parity measurement")
        validated[key] = measurement
    return validated


def _parity_baseline_record(
    baselines: Sequence[Mapping[str, float]], population: int
) -> dict[str, float] | list[dict[str, float]]:
    """The persisted form of the per-member baselines.

    A single learner's record stays the flat dict every checkpoint has carried.
    A population's is one entry per member, because each member's audit is judged
    against its own previous audit: the members are separate policies sharpening
    at their own rates, so pooling them would judge every member against whichever
    one had sharpened furthest.
    """
    if population == 1:
        return dict(baselines[0])
    return [dict(baseline) for baseline in baselines]


def _validate_parity_baselines(
    record: object,
    ceilings: Mapping[str, float],
    *,
    population: int,
) -> list[dict[str, float]]:
    """Validate one persisted baseline per population member."""
    if population == 1:
        return [_validate_parity_baseline(record, ceilings)]
    if not isinstance(record, list) or len(record) != population:
        raise ValueError(
            "resume checkpoint replay-parity baseline does not cover every population member"
        )
    return [_validate_parity_baseline(entry, ceilings) for entry in record]


def _validate_critic_warmup_state(
    initial_actor: object,
    *,
    population: int,
) -> tuple[int, bool, list[float | None]]:
    """Restore the adaptive actor-release gate from warm-start provenance."""
    if initial_actor is None:
        return 0, True, []
    if not isinstance(initial_actor, dict):
        raise ValueError("resume checkpoint has invalid initial-actor provenance")
    minimum = initial_actor.get("critic_warmup_iterations")
    state = initial_actor.get("critic_warmup_state")
    if type(minimum) is not int or minimum < 0 or minimum > MAX_CRITIC_WARMUP_ITERATIONS:
        raise ValueError("resume checkpoint has an invalid critic warmup minimum")
    if not isinstance(state, dict) or set(state) != {
        "complete",
        "last_monte_carlo_explained_variance",
    }:
        raise ValueError("resume checkpoint has no valid adaptive critic warmup state")
    complete = state["complete"]
    values = state["last_monte_carlo_explained_variance"]
    if type(complete) is not bool or not isinstance(values, list) or len(values) != population:
        raise ValueError("resume checkpoint has no valid adaptive critic warmup state")
    validated: list[float | None] = []
    for value in values:
        if value is None:
            validated.append(None)
        elif type(value) is float and math.isfinite(value):
            validated.append(value)
        else:
            raise ValueError("resume checkpoint has an invalid critic warmup EV")
    return minimum, complete, validated


def _critic_warmup_decision(
    *,
    iteration: int,
    minimum: int,
    complete: bool,
    previous_evs: Sequence[float | None],
) -> tuple[bool, str]:
    """Decide actor release from only prior-wave evidence."""
    if complete:
        return False, "complete"
    if iteration < minimum:
        return True, "minimum_iterations"
    ready = bool(previous_evs) and all(
        value is not None and math.isfinite(value) and value >= CRITIC_WARMUP_READY_MONTE_CARLO_EV
        for value in previous_evs
    )
    if ready:
        return False, "monte_carlo_ev_ready"
    if iteration >= MAX_CRITIC_WARMUP_ITERATIONS:
        raise RuntimeError(
            "critic warmup failed to reach Monte Carlo-return explained variance "
            f"{CRITIC_WARMUP_READY_MONTE_CARLO_EV:.2f} within "
            f"{MAX_CRITIC_WARMUP_ITERATIONS} iterations; previous member EVs={list(previous_evs)}"
        )
    return True, "waiting_for_monte_carlo_ev"


def _validate_league_score_rates(rates: object) -> dict[str, float]:
    if not isinstance(rates, dict):
        raise ValueError("resume checkpoint has no valid league score-rate state")
    validated: dict[str, float] = {}
    for key, rate in rates.items():
        valid_snapshot = type(key) is str and len(key) == 8 and key.isascii() and key.isdigit()
        valid_builtin = (
            type(key) is str
            and key.startswith("builtin_")
            and key.removeprefix("builtin_") in BUILTIN_OPPONENTS
        )
        if not valid_snapshot and not valid_builtin:
            raise ValueError("resume checkpoint has an invalid league score-rate opponent")
        if type(rate) is not float or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("resume checkpoint has an invalid league score rate")
        validated[key] = rate
    return validated


def _blend_league_score_rates(
    score_rates: dict[str, float],
    measured: dict[str, float],
) -> None:
    """Fold this iteration's measurements into the persistent PFSP estimates."""
    for key, rate in measured.items():
        previous = score_rates.get(key, PFSP_UNMEASURED_SCORE_RATE)
        score_rates[key] = LEAGUE_SCORE_RATE_EMA * rate + (1.0 - LEAGUE_SCORE_RATE_EMA) * previous
    for key in score_rates.keys() - measured.keys():
        previous = score_rates[key]
        score_rates[key] = previous + LEAGUE_SCORE_RATE_DECAY * (
            PFSP_UNMEASURED_SCORE_RATE - previous
        )


def _league_opponent_diagnostics(
    league: RolloutBatch,
    assignments: np.ndarray,
    selections: list[LeagueSelection],
) -> tuple[dict[str, float | int | str], dict[str, float]]:
    diagnostics: dict[str, float | int | str] = {}
    score_rates: dict[str, float] = {}
    margins = league.final_money - league.opponent_money
    outcomes = (margins > 0).astype(np.float32) - (margins < 0).astype(np.float32)
    for index, selection in enumerate(selections):
        selected = assignments == index
        games = int(selected.sum())
        if not games:
            # An unplayed opponent has no measurement; emitting one would put
            # a NaN into the journal and poison the PFSP estimates.
            continue
        prefix = f"league_opponent_{selection.key}"
        score_rate = float(((outcomes[selected] + 1.0) / 2.0).mean())
        diagnostics[f"{prefix}_category"] = selection.category
        diagnostics[f"{prefix}_games"] = games
        diagnostics[f"{prefix}_score_rate"] = score_rate
        diagnostics[f"{prefix}_mean_margin"] = float(margins[selected].mean())
        score_rates[selection.key] = score_rate
    return diagnostics, score_rates


def _resolve_external_eval_opponents(args: argparse.Namespace) -> None:
    """Drop unavailable external-eval opponents instead of blocking training."""
    if not args.external_eval:
        return
    resolved = []
    for spec in filter(None, (spec.strip() for spec in args.external_eval_opponents.split(","))):
        try:
            normalize_opponent(spec)
        except FileNotFoundError as error:
            print(f"external eval opponent dropped: {error}", file=sys.stderr, flush=True)
        else:
            resolved.append(spec)
    if not resolved:
        print(
            "external evaluation disabled: no configured opponent is available",
            file=sys.stderr,
            flush=True,
        )
        args.external_eval = False
    args.external_eval_opponents = ",".join(resolved)


_EXTERNAL_EVAL_PENDING = "external-eval-pending.json"


def _external_eval_pending(args: argparse.Namespace) -> list[tuple[Path, int]]:
    cached = getattr(args, "_kaggriculture_pending_evals", None)
    if cached is not None:
        return list(cached)
    path = args.run_dir / _EXTERNAL_EVAL_PENDING
    pending: list[tuple[Path, int]] = []
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"invalid external evaluation queue: {path}")
        for event in payload:
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("checkpoint"), str)
                or type(event.get("iteration")) is not int
                or event["iteration"] < 1
            ):
                raise ValueError(f"invalid external evaluation queue event: {event!r}")
            pending.append((Path(event["checkpoint"]), event["iteration"]))
    args._kaggriculture_pending_evals = tuple(pending)
    return pending


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry change before acknowledging queue mutation."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _persist_external_eval_pending(
    args: argparse.Namespace,
    pending: Sequence[tuple[Path, int]],
) -> None:
    events = tuple(pending)
    path = args.run_dir / _EXTERNAL_EVAL_PENDING
    if not events:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        args._kaggriculture_pending_evals = events
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                [
                    {"checkpoint": str(checkpoint), "iteration": iteration}
                    for checkpoint, iteration in events
                ],
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        args._kaggriculture_pending_evals = events
    finally:
        temporary.unlink(missing_ok=True)


def _external_eval_complete(
    args: argparse.Namespace,
    checkpoint: Path,
    iteration: int,
) -> bool:
    """Return whether the journal durably commits this event's full probe matrix."""
    if not checkpoint.is_file():
        return False
    members: list[int | None] = [None] if args.population < 2 else list(range(args.population))
    opponents = [
        normalize_opponent(spec)[0] for spec in args.external_eval_opponents.split(",") if spec
    ]
    if len(set(opponents)) != len(opponents):
        return False
    digest = file_sha256(checkpoint)
    records = read_jsonl_snapshot(args.run_dir / "metrics-external.jsonl").records
    for marker_index in range(len(records) - 1, -1, -1):
        marker = records[marker_index]
        if not (
            marker.get("event") == "external_eval_complete"
            and marker.get("iteration") == iteration
            and marker.get("artifact") == checkpoint.name
            and marker.get("artifact_sha256") == digest
            and marker.get("members") == members
            and marker.get("opponents") == opponents
            and marker.get("records") == len(members) * len(opponents)
        ):
            continue
        completed_rows: dict[tuple[int | None, str], Mapping[str, Any]] = {}
        for record in records[:marker_index]:
            key = (record.get("agent"), record.get("opponent"))
            if (
                record.get("event") == "external_eval"
                and record.get("iteration") == iteration
                and record.get("artifact") == checkpoint.name
                and record.get("artifact_sha256") == digest
                and key[0] in members
                and key[1] in opponents
            ):
                completed_rows[key] = record
        return all(
            (row := completed_rows.get((member, opponent))) is not None
            and type(row.get("games")) is int
            and row["games"] == args.external_eval_seeds * 2
            and row.get("seed_start") == args.external_eval_seed_start
            and row.get("seed_count") == args.external_eval_seeds
            and row.get("completed_games") == row["games"]
            for member in members
            for opponent in opponents
        )
    return False


def _maybe_launch_external_eval(
    args: argparse.Namespace,
    checkpoint: Path | None,
    committed_iteration: int,
    process: subprocess.Popen | None,
    *,
    wait_for_slot: bool = False,
) -> subprocess.Popen | None:
    """Run the durable FIFO, acknowledging only exited and journaled workers."""
    if not args.external_eval:
        return process
    pending = _external_eval_pending(args)
    event = (checkpoint, committed_iteration)
    if checkpoint is not None and committed_iteration >= 1 and event not in pending:
        pending.append(event)
        _persist_external_eval_pending(args, pending)
    if not pending:
        return process

    members = (
        "" if args.population < 2 else ",".join(str(member) for member in range(args.population))
    )
    log_path = args.run_dir / "external-eval.log"
    while pending:
        head_checkpoint, head_iteration = pending[0]
        if process is not None:
            returncode = process.poll()
            if returncode is None:
                if not wait_for_slot:
                    return process
                returncode = process.wait()
            if returncode == 0 and _external_eval_complete(args, head_checkpoint, head_iteration):
                pending.pop(0)
                _persist_external_eval_pending(args, pending)
                process = None
                continue

            reason = (
                f"exited with code {returncode}"
                if returncode != 0
                else "exited without a matching completion record"
            )
            print(
                f"external eval worker {reason}; see external-eval.log",
                file=sys.stderr,
                flush=True,
            )
            process = None
            if wait_for_slot:
                raise RuntimeError(f"final external evaluation {reason}")
        elif _external_eval_complete(args, head_checkpoint, head_iteration):
            # The prior process may have been interrupted after its fsynced
            # completion row but before the queue file was acknowledged.
            pending.pop(0)
            _persist_external_eval_pending(args, pending)
            continue

        try:
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve().parent / "external_eval_worker.py"),
                        "--artifact",
                        str(head_checkpoint),
                        "--agents",
                        members,
                        "--iteration",
                        str(head_iteration),
                        "--output",
                        str(args.run_dir / "metrics-external.jsonl"),
                        "--opponents",
                        args.external_eval_opponents,
                        "--seeds",
                        str(args.external_eval_seeds),
                        "--seed-start",
                        str(args.external_eval_seed_start),
                        "--episode-steps",
                        str(args.episode_steps),
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
        except OSError as error:
            print(f"external eval launch failed: {error}", file=sys.stderr, flush=True)
            if wait_for_slot:
                raise RuntimeError("final external evaluation could not launch") from error
            return None
        if not wait_for_slot:
            return process
    return None


def _league_builtin_opponents(args: argparse.Namespace) -> list[str]:
    """Names admitted to the training league, in launch-command order."""
    return [
        name for name in (name.strip() for name in args.league_builtin_opponents.split(",")) if name
    ]


def _select_league_opponents(
    args: argparse.Namespace,
    refs: Sequence[SnapshotRef],
    iteration: int,
    generator: np.random.Generator,
    score_rates: dict[str, float],
    *,
    pretrained_start: bool,
) -> list[LeagueSelection]:
    """Select a bounded opponent mix without consuming RNG when league play is off."""
    if not args.league_games:
        return []
    selections = select_league_mix(
        refs,
        current_iteration=max(1, iteration),
        active_count=args.league_active_opponents,
        historical_count=args.league_historical_opponents,
        active_pool_size=args.league_active_pool_size,
        generator=generator,
        builtins=_league_builtin_opponents(args),
        builtin_lanes=args.league_builtin_lanes,
        score_rates=score_rates,
        pretrained_start=pretrained_start,
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
        "deterministic_training": args.deterministic_training,
        "league_historical_opponents": args.league_historical_opponents,
        "league_active_pool_size": args.league_active_pool_size,
        # Which reference agents share the wave, and how many lanes they may
        # hold, decide what the learner plays against; a resume that changed
        # either would be generating different data under the same run.
        "league_builtin_opponents": ",".join(_league_builtin_opponents(args)),
        "league_builtin_lanes": args.league_builtin_lanes,
        "episode_steps": args.episode_steps,
        "temperature": args.temperature,
        # Both calibrated knobs are modes, and both are cross-checked against the
        # calibration decision by name (`provenance.CALIBRATION_KNOBS`), so the
        # record stores the mode itself rather than any boolean projection of it.
        "update_compile_mode": args.update_compile_mode,
        # The collection backend and precision both change the sampled behavior
        # policy, so a resume that changes either is a different data generator
        # and this record is what refuses it. The mode is also the calibrated
        # rollout knob, cross-checked against the decision in
        # `training.load_checkpoint`; the precision is fixed configuration.
        "rollout_forward_mode": args.rollout_forward_mode,
        "rollout_bfloat16": args.rollout_bfloat16,
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


def _checkpoint_recovery_values_equal(left: object, right: object) -> bool:
    """Compare exact resumable state while ignoring diagnostic timing metrics."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if left.keys() != right.keys():
        return False
    return all(key == "metrics" or _checkpoint_values_equal(left[key], right[key]) for key in left)


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
    if 0 not in validated:
        raise ValueError("resume checkpoint league manifest must retain iteration zero")
    return validated


def _rollback_metrics_journal(
    path: Path,
    checkpoint_iteration: int,
    checkpoint_metrics: Mapping[str, Any],
) -> None:
    """Atomically discard journal rows beyond a verified recovery boundary."""
    snapshot = read_jsonl_snapshot(path)
    iterations: list[int] = []
    kept: list[dict[str, Any]] = []
    boundary: dict[str, Any] | None = None
    for record in snapshot.records:
        iteration = record.get("iteration")
        if type(iteration) is not int or iteration < 1:
            raise ValueError("metrics journal has an invalid iteration")
        if iterations and iteration <= iterations[-1]:
            raise ValueError("metrics journal iterations are not strictly increasing")
        iterations.append(iteration)
        if iteration <= checkpoint_iteration:
            kept.append(record)
        if iteration == checkpoint_iteration:
            boundary = record
    if checkpoint_iteration > 0 and (
        boundary is None or not _checkpoint_values_equal(boundary, dict(checkpoint_metrics))
    ):
        raise ValueError("metrics journal checkpoint boundary does not match recovery state")
    if not iterations or iterations[-1] <= checkpoint_iteration:
        return

    temporary = path.with_name(f".{path.name}.{os.getpid()}.rollback")
    try:
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in kept),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_league_archive(
    *,
    checkpoint: Path,
    destination: Path,
    manifest: dict[int, str],
    current_iteration: int,
    model_config: ModelConfig | StructuredConfig,
) -> None:
    """Restore exactly the immutable archive bound to a training checkpoint."""
    source_directory = checkpoint.resolve().parent / "league"
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {f"league-actor-{iteration:08d}.pt" for iteration in manifest}
    existing_refs = list_actor_snapshots(destination)
    existing_by_name = {ref.path.name: ref for ref in existing_refs}
    unexpected = set(existing_by_name) - expected_names
    rollback: list[SnapshotRef] = []
    for name in sorted(unexpected):
        ref = existing_by_name[name]
        if ref.iteration <= current_iteration:
            raise ValueError(
                "resume destination contains a league snapshot missing from the "
                f"checkpoint manifest: {ref.path}"
            )
        load_actor_snapshot(
            ref.path,
            expected_model_config=model_config,
            device="cpu",
        )
        rollback.append(ref)
    missing: list[Path] = []
    for iteration, expected_digest in sorted(manifest.items()):
        name = f"league-actor-{iteration:08d}.pt"
        target = destination / name
        source = target if target.exists() else source_directory / name
        if not source.is_file():
            raise FileNotFoundError(
                f"checkpoint league sidecar is incomplete; missing snapshot: {source}"
            )
        load_actor_snapshot(
            source,
            expected_model_config=model_config,
            device="cpu",
        )
        if snapshot_sha256(source) != expected_digest:
            raise ValueError(f"league snapshot digest mismatch: {source}")
        if source != target:
            missing.append(source)

    for ref in rollback:
        ref.path.unlink()
    for source in missing:
        copy_actor_snapshot(
            source,
            destination,
            expected_model_config=model_config,
        )


def _agent_fields(
    measured: Mapping[str, float | int], agent: int, population: int
) -> dict[str, float | int]:
    """Journal field names for one member's readings.

    A single learner writes the bare names every existing journal, mirror layout
    and downstream reader already uses; a population prefixes each member so one
    member's curve can be read without the other three drawn over it.
    """
    if population == 1:
        return dict(measured)
    return {population_agent_field(agent, name): value for name, value in measured.items()}


def _audit_replay_parity(
    actor: Actor,
    rollout: RolloutBatch,
    *,
    rows: np.ndarray | None,
    ppo_config: PpoConfig,
    device: torch.device,
    baseline: Mapping[str, float],
    ceilings: Mapping[str, float],
    iteration: int,
    agent: int | None,
) -> tuple[dict[str, float | int], dict[str, float]]:
    """Audit one policy's sampling-vs-update parity; return metrics and baseline.

    The update pins its importance ratio to one by replaying behavior likelihoods
    through its own forward, so a staging bug applied identically to both
    update-path sides would never move the KL guard. Comparing that replay against
    the rollout's stored sampling likelihoods catches exactly that class of bug.
    The bound is a KL against the trust region the update already accepts, not a
    worst component: see MAX_UPDATE_REPLAY_KL for why the extreme value is
    reported but not gated.

    A breach aborts only when it reads as a defect rather than as drift, and
    REPLAY_PARITY_STEP_CHANGE_FACTOR is where that distinction is argued. The
    asymmetry matters because aborting is unrecoverable: the audit precedes the
    update, so no checkpoint covers the iteration, and a resume re-audits the same
    actor through the same code and dies again. That is the correct outcome for a
    defect, which a human has to go fix, and the wrong one for numerics drifting
    past a calibrated bound in an otherwise healthy multi-day run -- which warns
    instead, and leaves the trend in telemetry where it is the useful artifact.

    `rows` restricts the audit to one member's trajectories, which is required
    rather than an optimization: in a population wave every row was sampled by its
    own member, so replaying the whole wave through one of them would measure a
    difference between policies and report it as a staging defect.
    """
    where = "" if agent is None else f"agent {agent} "
    metrics: dict[str, float | int] = update_replay_parity(
        actor,
        rollout,
        minibatch_size=ppo_config.minibatch_size,
        compile_mode=ppo_config.update_compile_mode,
        autocast_enabled=ppo_config.use_bfloat16 and device.type == "cuda",
        rows=rows,
    )
    # A head with no active components reports zero divergence, which would pass
    # the bound without having audited anything. The calibration benchmark already
    # refuses that; training must too, or an audit can pass while having examined
    # nothing. This one is always fatal: it means the audit examined nothing, at
    # any point in the run, which is never expected drift.
    for component in PARITY_COMPONENTS:
        if metrics[f"update_replay_{component}_active_count"] < 1:
            raise RuntimeError(f"{where}update replay parity saw no active {component} components")
    breaches = _parity_breaches(metrics, dict(baseline) or None, ceilings)
    metrics["replay_parity_breached"] = len(breaches)
    metrics.update(_parity_fatal_thresholds(dict(baseline) or None, ceilings))
    for message, is_defect in breaches:
        if is_defect:
            continue
        # The journalled metrics carry iteration + 1, since the counter advances
        # before the record is written, so the warning names the row it will
        # appear in rather than the loop variable.
        print(
            f"warning: iteration {iteration + 1} {where}{message} — this is "
            f"within {REPLAY_PARITY_STEP_CHANGE_FACTOR}x of the previous "
            "audit and under the absolute ceiling, so it reads as drift "
            "rather than a defect and the run continues",
            file=sys.stderr,
            flush=True,
        )
    defects = [message for message, is_defect in breaches if is_defect]
    if defects:
        raise RuntimeError(
            where
            + "; ".join(defects)
            + " — a step change away from the previous audit, or past the "
            "absolute ceiling, rather than drift; resuming reproduces it. "
            "A step change is a staging defect worth diagnosing directly; "
            "a ceiling breach means the update forward's numerics no "
            "longer support this bound, and the run continues as a fresh "
            "one warm-started from the last actor under the repaired tree, "
            "since repairing it changes the source identity these "
            "checkpoints are bound to"
        )
    # The baseline advances after every audit, warned breaches included, so drift
    # is always compared against recent drift and can never accumulate into a
    # false step change. The ceiling is what stops that from ratcheting without
    # limit. It advances even when this audit aborted for a different member: the
    # process is ending either way, and nothing reads it after that.
    return metrics, _parity_measurements(metrics)


def _validate_policy_entropy_reference(value: object) -> float:
    """Validate a telemetry-only first-actor entropy reference.

    Entropy collapse is deliberately not a stop condition: a sharp policy that
    still beats its history is useful, and a strong BC warm start can begin
    below the from-scratch range. The reference remains checkpointed for
    continuity and diagnostics, so it must be numeric, finite, and nonnegative,
    but it must not reject that valid warm start.
    """
    if (
        not isinstance(value, float | int)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("policy entropy reference must be finite and nonnegative")
    return float(value)


_STRUCTURED_ACTOR_PERSISTENCE_FIELDS = (
    "combined",
    "decision",
    "patch",
    "economy",
    "opponent_summary",
    "opponent_patches",
)
_STRUCTURED_CRITIC_PERSISTENCE_FIELDS = ("combined", "latent", "value")


def _structured_persistence_diagnostics(
    update_metrics: Mapping[str, float | int],
    *,
    kind: str,
) -> dict[str, float | int]:
    """Compare predictors with fresh-wave persistence without controlling learning."""
    if kind == "actor":
        fields = _STRUCTURED_ACTOR_PERSISTENCE_FIELDS
        metric_prefix = "structured_preupdate_"
        telemetry_prefix = "structured_persistence_"
    elif kind == "critic":
        fields = _STRUCTURED_CRITIC_PERSISTENCE_FIELDS
        metric_prefix = "structured_critic_preupdate_"
        telemetry_prefix = "structured_critic_persistence_"
    else:
        raise ValueError(f"unknown structured predictor diagnostic kind: {kind}")
    measured = {name: float(update_metrics[f"{metric_prefix}{name}"]) for name in fields}
    # The predictor and persistence use the same fresh wave, source encoder and
    # decoder. Historical losses are incomparable as representations evolve.
    reference = {
        name: float(update_metrics[f"{metric_prefix}persistence_{name}"]) for name in fields
    }
    if any(not math.isfinite(value) for value in (*measured.values(), *reference.values())):
        raise FloatingPointError(f"non-finite structured {kind} persistence loss")
    # A zero persistence loss provides no prediction task for that decoder.
    # Keep the ratio finite and label it uninformative, reconsidering each wave.
    informative = {name: reference[name] > 0.0 for name in fields}
    ratios = {
        name: max(0.0, measured[name]) / reference[name] if informative[name] else 1.0
        for name in fields
    }
    if any(not math.isfinite(value) for value in ratios.values()):
        raise FloatingPointError(f"non-finite structured {kind} persistence ratio")
    telemetry: dict[str, float | int] = {
        f"{telemetry_prefix}{name}_ratio": ratio for name, ratio in ratios.items()
    }
    telemetry.update(
        {
            f"{telemetry_prefix}{name}_informative": int(active)
            for name, active in informative.items()
        }
    )
    return telemetry


def _entropy_reference_record(
    references: Sequence[float | None], population: int
) -> float | list[float | None] | None:
    """The persisted form of the per-member entropy references.

    `None` until the warmup ends and the first real actor update measures it, so
    a checkpoint written inside the warmup window carries no reference and a
    resume from it measures its own -- which is correct, because nothing has
    stepped yet and there is nothing to preserve.
    """
    if population == 1:
        return references[0]
    return list(references)


def _validate_entropy_references(
    record: object,
    *,
    population: int,
) -> list[float | None]:
    """Validate one persisted entropy reference per population member."""
    if population == 1:
        record = [record]
    if not isinstance(record, list) or len(record) != population:
        raise ValueError(
            "resume checkpoint policy entropy reference does not cover every population member"
        )
    return [
        None if entry is None else _validate_policy_entropy_reference(entry) for entry in record
    ]


def _gate_update_metrics(
    update_metrics: Mapping[str, float],
    *,
    warmup_active: bool,
    agent: int | None = None,
) -> None:
    """Stop the run on an update whose numbers say the next one is wasted.

    Ordered by cause, not by severity. An inflated first-minibatch KL at
    unchanged weights trips the trust region on minibatch zero, so checking the
    update count first would report the symptom; and a saturated value target
    explains a missing actor update rather than the other way round.

    `agent` names the population member the numbers belong to, and is why these
    gates are applied per member: one collapsed member has to stop the run as
    itself rather than be averaged into three healthy ones. `None` is the single
    learner, whose messages are unprefixed.
    """
    where = "" if agent is None else f"agent {agent} "
    first_minibatch_kl = float(update_metrics["first_minibatch_approx_kl"])
    if first_minibatch_kl > MAX_FIRST_MINIBATCH_KL:
        raise RuntimeError(
            f"{where}first-minibatch KL at unchanged weights exceeded "
            f"{MAX_FIRST_MINIBATCH_KL}: {first_minibatch_kl}"
        )
    # A target the support cannot hold is regressed onto a constant edge label,
    # which then holds the critic where it is. A few are ordinary critic error.
    saturated_fraction = float(update_metrics["value_target_saturated_fraction"])
    if saturated_fraction > MAX_VALUE_TARGET_SATURATED_FRACTION:
        raise RuntimeError(
            f"{where}value targets saturated the critic support beyond "
            f"{MAX_VALUE_TARGET_SATURATED_FRACTION}: {saturated_fraction}"
        )
    if int(update_metrics["actor_updates"]) < 1 and not warmup_active:
        raise RuntimeError(f"{where}PPO iteration completed without an actor update")
    # Entropy collapse and a short KL-clipped epoch are not stop conditions.
    # Intra-league win rate is the ranking; a sharp policy that still wins
    # more of its own history is an improvement, not a dead run.


# Fraction of this population's own iteration-0 pairwise disagreement below which
# the members have converged into each other and the wave is mirror play under
# another name. A share rather than an absolute level, because the level is a
# property of the initializations and nothing inside the loop knows it: two
# identical policies measure exactly 0.0, four independent ones about 0.894 --
# the 1 - 1/9 that nine legal unit actions give -- and four BC clones of one
# corpus land wherever their seeds put them. So the reference is recorded at
# iteration 0 and this is the share of it a run may lose, which leaves a factor
# of four of genuine convergence before the gate reads collapse.
POPULATION_DISAGREEMENT_FLOOR_FRACTION = 0.25
# States drawn from the iteration's own wave to score every member on. Sampled
# from the wave rather than held fixed for the run's life: a genuinely fixed
# batch would have to be checkpointed, and the reading is a mean over this many
# active unit decisions, whose sampling noise sits orders of magnitude below the
# factor of four the floor above allows.
POPULATION_DISAGREEMENT_STATES = 2048


def _population_row_partition(agents: np.ndarray, population: int) -> list[np.ndarray]:
    """Each member's trajectory rows, in agent order, partitioning the whole wave.

    Deliberately not contiguous. A game's two rows belong to two different
    members, so no storage order makes one member's rows a block; the update
    therefore takes row indices rather than a copied sub-batch, which would
    duplicate the wave's state arrays -- the largest allocation in the process.
    """
    rows = [np.flatnonzero(agents == agent) for agent in range(population)]
    covered = sum(index.size for index in rows)
    if covered != agents.size:
        raise ValueError(
            f"population wave rows are not covered by agents 0..{population - 1}: "
            f"{covered} of {agents.size} rows"
        )
    return rows


def _population_state_sample(
    rollout: RolloutBatch,
    generator: np.random.Generator,
    device: torch.device,
) -> tuple[tuple[Any, ...], torch.Tensor, torch.Tensor]:
    """One shared batch of states, as forward arguments plus unit mask and active.

    Only slots the collector marked valid hold a decision -- an episode that ended
    early leaves the rest of its lane untouched -- so sampling the raw block would
    score every member on stale padding and pull the reading toward whatever that
    padding happens to contain.

    Stored features and masks already live in the game's oriented frame. Every
    member is forwarded on that same frame, so the disagreement comparison
    stays in that shared label space.
    """
    valid = np.flatnonzero(np.asarray(rollout.valid).reshape(-1))
    index = generator.permutation(valid)[:POPULATION_DISAGREEMENT_STATES]

    def sampled(array: np.ndarray) -> np.ndarray:
        return array.reshape((-1, *array.shape[2:]))[index]

    # `unit_active` is not in any architecture's state dict but the structured
    # actor's forward reads it, so the update stages it alongside them and this
    # has to as well; the convolutional family ignores the extra entry.
    fields = {**rollout.states, "unit_active": rollout.unit_active}
    states = {name: sampled(value) for name, value in fields.items()}
    masks = torch.as_tensor(sampled(rollout.unit_masks)).to(device=device, dtype=torch.bool)
    active = torch.as_tensor(sampled(rollout.unit_active)).to(device=device, dtype=torch.bool)
    return actor_forward_args(rollout.architecture, states, device), masks, active


def _population_disagreement(
    actors: Sequence[torch.nn.Module],
    forward_args: tuple[Any, ...],
    masks: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """The N x N greedy-disagreement matrix over one shared batch of states.

    Scored in evaluation mode and restored afterwards: the measurement is about
    which program each member runs, and a train-mode forward would fold whatever
    stochastic layers the family has into a number the gate treats as structural.
    Every member sees the same oriented features, so their logits already share
    a label space and need no per-member remapping.
    """
    training = [actor.training for actor in actors]
    try:
        for actor in actors:
            actor.eval()
        with torch.inference_mode():
            logits = [actor(*forward_args).unit_logits for actor in actors]
    finally:
        for actor, mode in zip(actors, training, strict=True):
            actor.train(mode)
    return population_disagreement(logits, masks, active)


def _validate_population_reference(value: object) -> float:
    """Validate the checkpointed iteration-0 disagreement the floor is a share of.

    Persisted rather than re-measured, because re-measuring after a resume would
    recalibrate the floor against however far the members had already converged --
    which is the state the gate exists to refuse. `--max-hours` makes chunked
    restarts the designed operating mode, so a process-scoped reference would let
    a converging population reset its own floor every few hours.
    """
    if (
        not isinstance(value, float | int)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= 1.0
    ):
        raise ValueError("resume checkpoint has no valid population disagreement reference")
    return float(value)


def _gate_population_disagreement(measured: float, reference: float) -> None:
    """Stop a population whose members have converged into a single policy.

    The one failure this catches is invisible everywhere else. Identical members
    score exactly 0.5 against each other because `_relative_score` is zero at
    equal banks, so the wave carries no gradient at all -- while entropy, the
    first-minibatch KL, the epoch fraction and the money curve every one stay
    inside their bounds and the journal reads like a healthy run.
    """
    if not math.isfinite(measured):
        raise RuntimeError(
            "population pairwise disagreement is not finite: no active unit decision "
            "was sampled to compare the members on"
        )
    # A floor that is a share of zero admits everything, so a population whose
    # members already agree everywhere would run with this gate switched off. It
    # is also the same defect the gate exists for, arriving before iteration 1:
    # this architecture's unit logits carry a large fixed action prior that
    # dominates a fresh head, so cold-started members are one program in practice
    # and the members have to come from N differently trained artifacts.
    if reference <= 0.0:
        raise RuntimeError(
            "this population's members agree on every sampled decision at iteration 0, "
            "so they are one policy and every game is mirror play scoring 0.5 by "
            "symmetry; warm-start the members from as many differently trained "
            "artifacts as there are agents"
        )
    floor = POPULATION_DISAGREEMENT_FLOOR_FRACTION * reference
    if measured < floor:
        raise RuntimeError(
            f"population pairwise disagreement {measured:.4g} is below {floor:.4g}, "
            f"{POPULATION_DISAGREEMENT_FLOOR_FRACTION:.0%} of the {reference:.4g} this "
            "population started at; the members have converged into each other and "
            "every game is mirror play scoring 0.5 by symmetry"
        )


def _minimum_off_diagonal(matrix: torch.Tensor) -> float:
    """Smallest pairwise entry of a symmetric matrix whose diagonal is zero.

    Reported beside the mean because convergence starts as one pair, and a mean
    over six pairs stays comfortably inside the floor while one of them has
    already collapsed to zero.
    """
    off_diagonal = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return float(matrix[off_diagonal].min())


def _population_outcomes(rollout: RolloutBatch, population: int) -> dict[str, float]:
    """Per-member score rate and the score rate of every ordered pairing.

    Rows are game-major and seat-minor, so a row's opponent is its sibling row:
    2g and 2g + 1 are the two seats of one game. A pairing with no games is
    absent rather than zero, since zero is a score and not a missing measurement.
    """
    agents = np.asarray(rollout.agents)
    if agents.size % 2:
        raise ValueError("a population wave stores both seats of every game")
    money = np.asarray(rollout.final_money, dtype=np.float64)
    margins = money - np.asarray(rollout.opponent_money, dtype=np.float64)
    outcomes = (margins > 0).astype(np.float64) - (margins < 0).astype(np.float64)

    scores = (outcomes + 1.0) / 2.0
    opponents = agents[np.arange(agents.size) ^ 1]
    fields: dict[str, float] = {}
    for agent in range(population):
        own = agents == agent
        if own.any():
            fields[population_agent_field(agent, "score_rate")] = float(scores[own].mean())
            fields[population_agent_field(agent, "money_mean")] = float(money[own].mean())
        for opponent in range(population):
            if agent == opponent:
                continue
            rows = own & (opponents == opponent)
            if rows.any():
                fields[population_head_to_head_field(agent, opponent)] = float(scores[rows].mean())
    return fields


def _configure_training_determinism(enabled: bool) -> None:
    """Apply the resume-bound deterministic execution contract."""
    if enabled:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError(
                "deterministic training requires CUBLAS_WORKSPACE_CONFIG to be ':4096:8' or ':16:8'"
            )
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    if enabled:
        torch.backends.cudnn.benchmark = False


def main() -> None:
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    args = parse_args()
    _validate_args(args)
    _configure_training_determinism(args.deterministic_training)
    _resolve_external_eval_opponents(args)
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
    # Exactly the knobs the decision attributes a speedup to, taken from the
    # validator that re-derives them rather than restated here. The rollout knob
    # `rollout_bfloat16` is deliberately not among them: collection precision is
    # fixed configuration, pinned identical on every chain node instead of
    # attributed, so the decision carries nothing to compare it against. It is
    # recorded in `_training_data_config`, where a resume must match it exactly.
    if run_provenance is not None:
        calibration = run_provenance["calibration"]
        if not isinstance(calibration, Mapping):
            raise ValueError("run provenance calibration must be a mapping")
        if any(calibration[knob] != getattr(args, knob) for knob in CALIBRATION_KNOBS):
            raise ValueError("training compile mode does not match calibration run provenance")
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = not args.deterministic_training

    architecture = resolve_architecture(args.architecture)
    model_config: ModelConfig | StructuredConfig = model_config_from_args(architecture, args)
    ppo_config = PpoConfig(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        epochs=args.epochs,
        critic_epochs=args.critic_epochs,
        minibatch_size=args.minibatch_size,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        gamma=args.gamma,
        actor_gae_lambda=args.actor_gae_lambda,
        critic_gae_lambda=args.critic_gae_lambda,
        nextlat_max_gradient_norm=args.nextlat_max_gradient_norm,
        target_kl=args.target_kl,
        optimizer=args.optimizer,
        use_bfloat16=not args.no_bfloat16,
        update_compile_mode=args.update_compile_mode,
        structured_decision_coefficient=args.structured_decision_coefficient,
        structured_patch_coefficient=args.structured_patch_coefficient,
        structured_economy_coefficient=args.structured_economy_coefficient,
        structured_opponent_summary_coefficient=(args.structured_opponent_summary_coefficient),
        structured_opponent_patch_coefficient=(args.structured_opponent_patch_coefficient),
        structured_decision_horizon=args.structured_decision_horizon,
        structured_patch_horizon=args.structured_patch_horizon,
        structured_critic_latent_coefficient=args.structured_critic_latent_coefficient,
        structured_critic_value_coefficient=args.structured_critic_value_coefficient,
        structured_critic_horizon=args.structured_critic_horizon,
        structured_learning_rate=args.structured_learning_rate,
        structured_critic_learning_rate=args.structured_critic_learning_rate,
    )
    training_data_config = _training_data_config(args, device)
    # Derived from the configured trust region rather than fixed, because that
    # is what the ceiling means: the level at which the uncorrected parity
    # divergence matches the divergence this run's update deliberately allows.
    parity_ceilings = _parity_ceilings()
    # A single learner is a population of one, so members are built one way. The
    # four names below alias member zero because everything a single-learner run
    # does with them is unchanged; a population run addresses `members` instead.
    # Cold-started members are genuinely different weights already: the global
    # torch seed was set once above, so each construction advances it.
    population = args.population
    members: list[TrainingAgent] = []
    for _ in range(population):
        member_actor = architecture.actor_class(model_config).to(device)
        # Preserve the historical actor/critic initialization stream. The
        # training-only predictors are constructed afterward, so enabling
        # NextLat cannot silently change the critic seed it is compared against.
        member_critic = architecture.critic_class(model_config).to(device)
        member_dynamics = (
            StructuredDynamics(model_config).to(device)
            if ppo_config.structured_actor_auxiliary_active
            and isinstance(model_config, StructuredConfig)
            else None
        )
        member_critic_dynamics = (
            StructuredCriticDynamics(model_config).to(device)
            if ppo_config.structured_critic_auxiliary_active
            and isinstance(model_config, StructuredConfig)
            else None
        )
        actor_optimizer, critic_optimizer = make_optimizers(
            member_actor,
            member_critic,
            ppo_config,
        )
        dynamics_optimizer = (
            make_structured_dynamics_optimizer(member_dynamics, ppo_config)
            if member_dynamics is not None
            else None
        )
        critic_dynamics_optimizer = (
            make_structured_dynamics_optimizer(member_critic_dynamics, ppo_config)
            if member_critic_dynamics is not None
            else None
        )
        members.append(
            TrainingAgent(
                actor=member_actor,
                critic=member_critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                structured_dynamics=member_dynamics,
                structured_dynamics_optimizer=dynamics_optimizer,
                structured_critic_dynamics=member_critic_dynamics,
                structured_critic_dynamics_optimizer=critic_dynamics_optimizer,
            )
        )
    # Member 0's networks are aliased for the single-learner code below; its
    # optimizers are not, because every step site reaches them through the
    # member itself, and a second name for one of N would read as though it
    # were the run's optimizer.
    actor = members[0].actor
    critic = members[0].critic
    initial_actor_provenance: dict[str, Any] | None = None
    critic_warmup_iterations = 0
    critic_warmup_complete = True
    critic_warmup_previous_evs: list[float | None] = []
    initial_actors = _initial_actor_paths(args)
    if initial_actors:
        records = [
            _load_initial_actor(path, member.actor, architecture.name, model_config, device)
            for path, member in zip(initial_actors, members, strict=True)
        ]
        # Two copies of one artifact under different names are the same policy.
        # Four learners require four independently trained initial programs;
        # otherwise the nominal population begins as mirror play under aliases.
        digests = {record["sha256"] for record in records}
        if len(digests) != len(records):
            raise ValueError(
                "initial actor artifacts must hold different weights per agent: "
                f"{len(records)} paths carry {len(digests)} distinct digests"
            )
        initial_actor_provenance = records[0] if population == 1 else {"agents": records}
        critic_warmup_iterations = args.critic_warmup_iterations or 0
        critic_warmup_complete = False
        critic_warmup_previous_evs = [None for _ in members]
        # The release gate travels inside the warm-start record so a crash
        # resumes with the exact prior-wave evidence and latch state.
        initial_actor_provenance["critic_warmup_iterations"] = critic_warmup_iterations
        initial_actor_provenance["critic_warmup_state"] = {
            "complete": critic_warmup_complete,
            "last_monte_carlo_explained_variance": list(critic_warmup_previous_evs),
        }
    generator = np.random.default_rng(args.seed + 1)
    auxiliary_generator = (
        np.random.default_rng(args.seed + 2) if ppo_config.structured_auxiliary_active else None
    )
    iteration = 0
    next_seed = args.seed
    resume_payload = None
    if args.resume:
        # Architecture and model-config identity are validated inside
        # load_checkpoint before it mutates the freshly constructed models.
        resume_payload = load_checkpoint(args.resume, members, device=device)
        if resume_payload["ppo_config"] != asdict(ppo_config):
            raise ValueError("resume checkpoint PPO configuration does not match arguments")
        if resume_payload.get("training_data_config") != training_data_config:
            raise ValueError(
                "resume checkpoint data-generation configuration does not match arguments"
            )
        require_source_identity(resume_payload.get("source_identity"))
        checkpoint_run_provenance = validate_run_provenance(resume_payload.get("run_provenance"))
        # The same attributed knobs as the launch-time check above, compared the
        # same way. Collection precision is again left out; the
        # `training_data_config` equality a few lines above already refuses a
        # resume that changes it.
        if checkpoint_run_provenance is not None and any(
            checkpoint_run_provenance["calibration"][knob] != getattr(args, knob)
            for knob in CALIBRATION_KNOBS
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
        if auxiliary_generator is not None:
            auxiliary_generator.bit_generator.state = resume_payload["structured_auxiliary_rng"]
        # Restore warm-start provenance: the resumed run must keep treating
        # the iteration-0 league snapshot as a pretrained baseline, and must
        # keep freezing the actor for whatever remains of the critic warmup.
        initial_actor_provenance = resume_payload.get("initial_actor")
        (
            critic_warmup_iterations,
            critic_warmup_complete,
            critic_warmup_previous_evs,
        ) = _validate_critic_warmup_state(
            initial_actor_provenance,
            population=population,
        )

    seed_usage = (
        artifact_seed_usage(resume_payload)
        if resume_payload is not None
        else [
            row
            for record in (
                initial_actor_provenance.get("agents", [initial_actor_provenance])
                if initial_actor_provenance is not None
                else []
            )
            for row in record["seed_usage"]
        ]
    )
    planned_games = (args.games + (args.league_games if population == 1 else 0)) * max(
        0, args.iterations - iteration
    )
    if planned_games:
        online_interval = validate_seed_interval(
            "online_rl",
            next_seed,
            planned_games,
            usage=seed_usage,
        )
        if not any(
            row["domain"] == "online_rl"
            and row["start"] <= next_seed
            and row["start"] + row["count"] >= next_seed + planned_games
            for row in seed_usage
        ):
            seed_usage.append(online_interval)
    if args.external_eval:
        development_interval = validate_seed_interval(
            "development",
            args.external_eval_seed_start,
            args.external_eval_seeds,
            usage=seed_usage,
        )
        if development_interval not in seed_usage:
            seed_usage.append(development_interval)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    journal_path = args.run_dir / "metrics.jsonl"
    journal_iteration = metrics_journal_iteration(journal_path)
    if resume_payload is not None and journal_iteration > iteration:
        _rollback_metrics_journal(journal_path, iteration, resume_payload["metrics"])
        journal_iteration = metrics_journal_iteration(journal_path)
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
    # PFSP score-rate estimates per snapshot iteration. Persisted in every
    # checkpoint: opponent selection consumes RNG as a function of these
    # estimates, so a resume that reset them would diverge from the
    # uninterrupted run's entire downstream RNG stream.
    league_score_rates: dict[str, float] = {}
    # The most recent audit's per-head measurements, which is what a later
    # breach is judged a defect or drift against. Persisted because the
    # judgement is a comparison across audits, and a run long enough to drift
    # is a run long enough to be restarted -- --max-hours makes chunked
    # restarts the designed operating mode -- so a process-scoped baseline
    # would make the first audit after every restart a fresh launch check and
    # abort a run that had merely drifted. It carries no version tag because it
    # does not need one: a resume already requires the checkpoint's source
    # identity to equal the tree exactly, so a baseline can never be read back
    # under constants that changed what it measured.
    #
    # One baseline, not one per staging configuration, even though the cadence
    # is per configuration. What it tracks is a property of how sharp the heads
    # have become, which the two configurations share -- they audit the same
    # actor over largely the same states, differing only in which opponents
    # generated them. Keying it per configuration would mean a configuration
    # that lapses for hundreds of iterations gets compared against its own
    # stale measurement from a much less sharpened policy, which is a step
    # change of thousands and a dead run. A level difference between the two
    # configurations that is large enough to trip the factor is not noise to be
    # tolerated anyway; it is one of them staging differently from the other,
    # which is the defect this whole audit exists to find.
    parity_baselines: list[dict[str, float]] = [{} for _ in members]
    # The population's iteration-0 pairwise disagreement, once measured. `None`
    # until then, which is also what says "this iteration records it".
    population_reference: float | None = None
    # Each member's own first actor-active entropy, once its actor has stepped.
    # `None` until then, which is also what says "this update records it".
    entropy_references: list[float | None] = [None for _ in members]
    if resume_payload is not None:
        league_score_rates = _validate_league_score_rates(resume_payload.get("league_score_rates"))
        parity_baselines = _validate_parity_baselines(
            resume_payload.get("replay_parity_baseline"), parity_ceilings, population=population
        )
        entropy_references = _validate_entropy_references(
            resume_payload.get("policy_entropy_reference"), population=population
        )
        if population > 1:
            population_reference = _validate_population_reference(
                resume_payload.get("population_disagreement_reference")
            )
            # A population wave has no frozen lanes, so it writes no snapshot
            # archive and there is none to restore. A non-empty manifest here is
            # a checkpoint from the single-learner scheme being resumed as a
            # population, which would silently change what the run plays.
            if resume_payload.get("league_snapshot_manifest"):
                raise ValueError(
                    "resume checkpoint carries a league snapshot archive; a population "
                    "run has no frozen lanes and cannot continue that run"
                )
        else:
            league_snapshot_manifest = _validate_league_manifest(
                resume_payload.get("league_snapshot_manifest"),
                current_iteration=iteration,
            )
            _restore_league_archive(
                checkpoint=args.resume,
                destination=league_directory,
                manifest=league_snapshot_manifest,
                current_iteration=iteration,
                model_config=model_config,
            )
    serialized_arguments = {name: _recorded_argument(value) for name, value in vars(args).items()}
    serialized_arguments["resume"] = str(args.resume or "")
    configuration = {
        "arguments": serialized_arguments,
        "architecture": architecture.name,
        "model": model_config.to_dict(),
        "ppo": asdict(ppo_config),
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "device": str(device),
        "torch_version": torch.__version__,
        # torch is pinned by uv.lock, which source_identity hashes; the Rust
        # toolchain is pinned by nothing, so it is recorded to be comparable.
        "native_toolchain": toolchain_identity(),
        "source_identity": current_source_identity,
        "run_provenance": run_provenance,
        "initial_actor": initial_actor_provenance,
        "seed_usage": seed_usage,
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if resume_payload is not None and iteration > 0:
        append_iteration_jsonl(args.run_dir / "metrics.jsonl", resume_payload["metrics"])
    writer = TensorboardMirror(
        args.run_dir / "metrics.jsonl",
        args.run_dir / "tensorboard",
        writer_factory=lambda path: SummaryWriter(path),
    )
    opponent_pool = FrozenActorPool(model_config, device)
    # One reusable trajectory-major pinned arena receives the whole mixed
    # wave (self-play rows first, league rows after) directly from the
    # collector, so the replay stages to the accelerator without any host
    # concatenation. The self-play prefix serves iterations without league
    # play, which produce fewer trajectories. A population wave has no league
    # rows at all and both of its seats are stored, so it fills exactly the
    # self-play prefix, which is the whole arena.
    self_play_rows = args.games * 2
    rollout_arena = allocate_rollout_storage(
        architecture.name,
        self_play_rows + args.league_games,
        args.episode_steps - 1,
        pin_memory=device.type == "cuda",
    )
    self_play_storage = {name: array[:self_play_rows] for name, array in rollout_arena.items()}

    commit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="commit")
    pending_commit: Future[tuple[tuple[Path, int] | None, SnapshotRef | None]] | None = None
    # The archive is immutable and this process is its only steady-state
    # publisher. Keep its ordered refs in memory instead of rescanning and
    # stat'ing every historical snapshot before every rollout; resume already
    # validates/restores the complete directory before this cache is built.

    def build_recovery_payload(metrics: dict[str, Any]) -> dict[str, Any]:
        """Capture the complete CPU recovery state only for a checkpoint event."""
        agent_states = [cpu_state_copy(member.state()) for member in members]
        return checkpoint_payload(
            agents=agent_states,
            model_config=model_config,
            ppo_config=ppo_config,
            iteration=iteration,
            next_seed=next_seed,
            metrics=metrics,
            source_identity=current_source_identity,
            run_provenance=run_provenance,
            rng_states=training_rng_states(),
            training_rng_state=dict(generator.bit_generator.state),
            training_data_config=training_data_config,
            auxiliary_rng_state=(
                dict(auxiliary_generator.bit_generator.state)
                if auxiliary_generator is not None
                else None
            ),
            league_snapshot_manifest=league_snapshot_manifest,
            league_score_rates=dict(league_score_rates),
            replay_parity_baseline=_parity_baseline_record(parity_baselines, population),
            population_disagreement_reference=population_reference,
            policy_entropy_reference=_entropy_reference_record(entropy_references, population),
            initial_actor=copy.deepcopy(initial_actor_provenance),
            seed_usage=seed_usage,
        )

    def publish_checkpoint(payload: dict[str, Any]) -> Path:
        committed = int(payload["iteration"])
        payload["league_snapshot_manifest"] = dict(league_snapshot_manifest)
        checkpoint = args.run_dir / f"checkpoint-{committed:06d}.pt"
        if checkpoint.exists():
            existing = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not _checkpoint_recovery_values_equal(existing, payload):
                raise FileExistsError(
                    f"numbered checkpoint conflicts with committed state: {checkpoint}"
                )
            payload["metrics"] = existing["metrics"]
        else:
            write_immutable_checkpoint(checkpoint, payload)
        replace_checkpoint_alias(checkpoint, args.run_dir / "latest.pt")
        return checkpoint

    def commit_iteration(
        metrics: dict[str, Any],
        actor_state: dict[str, Any] | None,
        recovery_payload: dict[str, Any] | None,
    ) -> tuple[tuple[Path, int] | None, SnapshotRef | None]:
        """Commit one completed update, optionally including a recovery event."""
        committed = int(metrics["iteration"])
        snapshot = None
        if actor_state is not None:
            snapshot = save_actor_state_snapshot(
                league_directory, model_config, actor_state, committed
            )
            league_snapshot_manifest[committed] = snapshot_sha256(snapshot.path)
        checkpoint = publish_checkpoint(recovery_payload) if recovery_payload is not None else None
        committed_metrics = recovery_payload["metrics"] if recovery_payload is not None else metrics
        append_iteration_jsonl(args.run_dir / "metrics.jsonl", committed_metrics)
        writer.record(committed_metrics)
        checkpoint_event = None if checkpoint is None else (checkpoint, committed)
        return checkpoint_event, snapshot

    started = time.monotonic()

    if population == 1 and (iteration == 0 or critic_warmup_complete):
        # Warmup checkpoints intentionally carry a sparse league manifest: the
        # frozen actor already exists at iteration zero, so serializing it under
        # every warmup iteration would create byte-identical archive churn.
        current_snapshot = save_actor_snapshot(league_directory, actor, iteration)
        current_digest = snapshot_sha256(current_snapshot.path)
        previous_digest = league_snapshot_manifest.get(iteration)
        if previous_digest is not None and previous_digest != current_digest:
            raise ValueError("resume checkpoint actor does not match its current league snapshot")
        league_snapshot_manifest[iteration] = current_digest
    league_snapshot_refs = list_actor_snapshots(league_directory)

    numbered_checkpoint = args.run_dir / f"checkpoint-{iteration:06d}.pt"
    destination_latest = args.run_dir / "latest.pt"
    if resume_payload is not None:
        in_place_resume = args.resume.parent.resolve() == args.run_dir.resolve()
        for existing in (numbered_checkpoint, destination_latest):
            if not existing.exists():
                continue
            existing_payload = torch.load(existing, map_location="cpu", weights_only=False)
            if _checkpoint_values_equal(existing_payload, resume_payload):
                continue
            if existing == destination_latest and in_place_resume:
                replace_checkpoint_alias(args.resume, destination_latest)
                continue
            raise FileExistsError(f"checkpoint conflicts with resume state: {existing}")
        if not numbered_checkpoint.exists():
            install_immutable_checkpoint(args.resume, numbered_checkpoint)
    else:
        save_checkpoint(
            numbered_checkpoint,
            agents=members,
            model_config=model_config,
            ppo_config=ppo_config,
            iteration=0,
            next_seed=next_seed,
            metrics={"iteration": 0},
            training_rng_state=dict(generator.bit_generator.state),
            training_data_config=training_data_config,
            auxiliary_rng_state=(
                dict(auxiliary_generator.bit_generator.state)
                if auxiliary_generator is not None
                else None
            ),
            league_snapshot_manifest=league_snapshot_manifest,
            league_score_rates=league_score_rates,
            replay_parity_baseline=_parity_baseline_record(parity_baselines, population),
            population_disagreement_reference=population_reference,
            policy_entropy_reference=_entropy_reference_record(entropy_references, population),
            source_identity=current_source_identity,
            run_provenance=run_provenance,
            initial_actor=initial_actor_provenance,
            seed_usage=seed_usage,
        )
    replace_checkpoint_alias(numbered_checkpoint, destination_latest)
    last_checkpoint_iteration = iteration
    checkpoint_timer = RecoveryCheckpointTimer(args.checkpoint_seconds, clock=time.monotonic)
    last_metrics = resume_payload["metrics"] if resume_payload is not None else {"iteration": 0}

    # Iteration of the most recent audit per staging configuration. Deliberately
    # process-scoped rather than checkpointed: a resume rebuilds the compiled
    # callables and restages every buffer, so it should audit its first
    # iteration rather than wait out the cadence.
    last_parity_audit: dict[str, int] = {}
    external_eval_process: subprocess.Popen | None = None
    while iteration < args.iterations:
        # Opponent discovery below must observe the previous iteration's
        # immutable actor snapshot. The same barrier publishes its metrics and
        # recovery checkpoint before this iteration consumes league RNG.
        if pending_commit is not None:
            completed_checkpoint, completed_snapshot = pending_commit.result()
            pending_commit = None
            if completed_snapshot is not None:
                league_snapshot_refs.append(completed_snapshot)
            if completed_checkpoint is not None:
                external_eval_process = _maybe_launch_external_eval(
                    args,
                    completed_checkpoint[0],
                    completed_checkpoint[1],
                    external_eval_process,
                )
        if args.max_hours and (time.monotonic() - started) / 3600.0 >= args.max_hours:
            break
        warmup_active, warmup_reason = _critic_warmup_decision(
            iteration=iteration,
            minimum=critic_warmup_iterations,
            complete=critic_warmup_complete,
            previous_evs=critic_warmup_previous_evs,
        )
        if initial_actor_provenance is not None and not warmup_active:
            critic_warmup_complete = True
            initial_actor_provenance["critic_warmup_state"]["complete"] = True
        warmup_diagnostics: dict[str, Any] = {}
        if initial_actor_provenance is not None:
            ready_members = sum(
                value is not None
                and math.isfinite(value)
                and value >= CRITIC_WARMUP_READY_MONTE_CARLO_EV
                for value in critic_warmup_previous_evs
            )
            warmup_diagnostics = {
                "critic_warmup_active": int(warmup_active),
                "critic_warmup_reason": warmup_reason,
                "critic_warmup_minimum_iterations": critic_warmup_iterations,
                "critic_warmup_max_iterations": MAX_CRITIC_WARMUP_ITERATIONS,
                "critic_warmup_readiness_threshold": CRITIC_WARMUP_READY_MONTE_CARLO_EV,
                "critic_warmup_ready_members": ready_members,
                "critic_warmup_required_members": population,
            }
            for agent, value in enumerate(critic_warmup_previous_evs):
                name = "critic_warmup_previous_monte_carlo_explained_variance"
                if population > 1:
                    name = population_agent_field(agent, name)
                warmup_diagnostics[name] = value
        iteration_started = time.monotonic()
        sampling_seed = int(generator.integers(0, np.iinfo(np.int64).max))
        opponent_checkpoint = ""
        league_diagnostics: dict[str, float | int | str] = {}
        self_play_diagnostics: dict[str, float | int] = {}
        population_metrics: dict[str, float] = {}
        # The wave's wall-clock is indivisible; per-part timing keys would
        # merely repeat it, so slice diagnostics keep only outcome metrics.
        indivisible_timings = ("rollout_seconds", "rollout_states_per_second")
        if population > 1:
            # One ensemble forward over N lanes covering every row, lane index =
            # agent index. Both seats belong to learners and both are stored, so
            # the wave is 2G trajectories and there is no frozen or built-in lane
            # for any schedule to reserve.
            league_games = 0
            rollout = collect_population_play_rust(
                [member.actor for member in members],
                games=args.games,
                seed_start=next_seed,
                episode_steps=args.episode_steps,
                temperature=args.temperature,
                gamma=args.gamma,
                sampling_seed=sampling_seed,
                forward_mode=args.rollout_forward_mode,
                forward_autocast=args.rollout_bfloat16,
                storage=rollout_arena,
            )
            diagnostic_groups = {"self_play": np.ones(rollout.trajectories, dtype=bool)}
            next_seed += args.games
            agent_rows: list[np.ndarray | None] = list(
                _population_row_partition(np.asarray(rollout.agents), population)
            )
            # Measured before the update, so iteration 0 records what these
            # initializations were rather than what one update already left.
            forward_args, unit_masks, unit_active = _population_state_sample(
                rollout, generator, device
            )
            disagreement = _population_disagreement(
                [member.actor for member in members],
                forward_args,
                unit_masks,
                unit_active,
            )

            mean_disagreement = mean_off_diagonal(disagreement)
            if population_reference is None:
                population_reference = mean_disagreement
            _gate_population_disagreement(mean_disagreement, population_reference)
            population_metrics = {
                population_disagreement_field("mean"): mean_disagreement,
                population_disagreement_field("min"): _minimum_off_diagonal(disagreement),
                population_disagreement_field("floor"): population_reference,
                **{
                    population_disagreement_pair_field(first, second): float(
                        disagreement[first, second]
                    )
                    for first in range(population)
                    for second in range(first + 1, population)
                },
                **_population_outcomes(rollout, population),
            }
        else:
            agent_rows = [None]
            selections = _select_league_opponents(
                args,
                league_snapshot_refs,
                iteration,
                generator,
                league_score_rates,
                pretrained_start=initial_actor_provenance is not None,
            )
            league_games = args.league_games if selections else 0
            opponents = []
            assignments = None
            builtin_lanes: list[str] = []
            if league_games:
                # The selection's contract puts every snapshot lane before every
                # built-in lane, which is exactly the lane index space the wave
                # addresses: frozen modules first, built-ins after them.
                snapshots = [row for row in selections if isinstance(row, SnapshotSelection)]
                builtin_lanes = [
                    row.ref.name for row in selections if isinstance(row, BuiltinSelection)
                ]
                opponents = opponent_pool.acquire([row.ref.path for row in snapshots])
                assignments = _balanced_assignments(
                    league_games,
                    len(selections),
                    generator,
                    seed_start=next_seed + args.games,
                )
                opponent_checkpoint = ",".join(row.label for row in selections)
            # Self-play and league games advance in one native wave, so the
            # learner forward covers every current-policy row at once and the
            # collector writes straight into the shared arena.
            rollout = collect_mixed_play_rust(
                actor,
                opponents,
                self_play_games=args.games,
                league_games=league_games,
                opponent_indices=assignments,
                builtin_lanes=builtin_lanes,
                seed_start=next_seed,
                episode_steps=args.episode_steps,
                temperature=args.temperature,
                gamma=args.gamma,
                # Every league seat decodes exactly as the learner does.
                # Sharpening them instead -- active lanes at 0.8, historical ones
                # at argmax -- handed the learner an opponent that was a strictly
                # better executor of its own policy, so an identical snapshot beat
                # it: iteration 40 scored 0.302 in league lanes against copies of
                # itself, and a temperature-1.0 seat loses to a temperature-0.8
                # one of the same weights at a 0.4375 win rate. The learner cannot
                # answer that by playing better, only by playing something whose
                # payoff survives its own sampling noise, and it found one: farming
                # 143,000 needs roughly 8,000 correct decisions in a row while
                # denying an opponent needs far fewer, so the gradient preferred
                # the noise-robust strategy and the economy went with it.
                opponent_temperature=args.temperature,
                sampling_seed=sampling_seed,
                # One decision, stated once. `forward_mode` drives the learner
                # forward, and `compile_models` -- which now governs only the
                # frozen-league ensemble -- follows it, because the pairing the
                # 1.66x speedup and the 4-wave parity gate were measured under had
                # both compiled together. A non-eager mode must not leave the
                # ensemble eager.
                forward_mode=args.rollout_forward_mode,
                forward_autocast=args.rollout_bfloat16,
                storage=rollout_arena if league_games else self_play_storage,
            )
            diagnostic_groups = {
                "self_play": np.arange(rollout.trajectories) < self_play_rows,
                "league": np.arange(rollout.trajectories) >= self_play_rows,
            }
            if assignments is not None:
                for selection_index, selection in enumerate(selections):
                    diagnostic_groups[f"opponent_{selection.key}"] = np.concatenate(
                        (
                            np.zeros(self_play_rows, dtype=bool),
                            assignments == selection_index,
                        )
                    )
            next_seed += args.games + league_games
            self_play_diagnostics = {
                f"self_play_{name}": value
                for name, value in rollout_diagnostics(
                    slice_trajectories(rollout, 0, self_play_rows)
                ).items()
                if name not in indivisible_timings
            }
            measured_rates: dict[str, float] = {}
            if league_games:
                league_part = slice_trajectories(rollout, self_play_rows, rollout.trajectories)
                league_diagnostics = {
                    f"league_{name}": value
                    for name, value in rollout_diagnostics(league_part).items()
                    if name not in indivisible_timings
                }
                assert assignments is not None
                opponent_diagnostics, measured_rates = _league_opponent_diagnostics(
                    league_part, assignments, selections
                )
                league_diagnostics.update(opponent_diagnostics)
            _blend_league_score_rates(league_score_rates, measured_rates)
        # The audit is per member, on that member's own rows: in a population wave
        # every row was sampled by its own policy, so a whole-wave replay through
        # one of them measures a policy difference and calls it a staging defect.
        # `_audit_replay_parity` carries the rest of the reasoning.
        replay_parity_metrics: dict[str, float | int] = {}
        audited_staging = _parity_staging_key(league_games, population)
        if _parity_audit_due(last_parity_audit, iteration, audited_staging):
            for agent, (member, rows) in enumerate(zip(members, agent_rows, strict=True)):
                measured, parity_baselines[agent] = _audit_replay_parity(
                    member.actor,
                    rollout,
                    rows=rows,
                    ppo_config=ppo_config,
                    device=device,
                    baseline=parity_baselines[agent],
                    ceilings=parity_ceilings,
                    iteration=iteration,
                    agent=None if population == 1 else agent,
                )
                replay_parity_metrics.update(_agent_fields(measured, agent, population))
            last_parity_audit[audited_staging] = iteration
        update_started = time.monotonic()
        # Actor release is decided above from the previous fresh wave. The
        # current rollout's pre-update EV becomes evidence for the next wave.
        # Once per member, over that member's rows. A game's two seats belong to
        # two members, so the partition is a row index, not a slice.
        update_metrics: dict[str, float | int] = {}
        current_warmup_evs: list[float] = []
        for agent, (member, rows) in enumerate(zip(members, agent_rows, strict=True)):
            if member.actor_optimizer is None or member.critic_optimizer is None:
                raise RuntimeError("training member has no optimizer")
            for name, predictor, optimizer in (
                (
                    "actor",
                    member.structured_dynamics,
                    member.structured_dynamics_optimizer,
                ),
                (
                    "critic",
                    member.structured_critic_dynamics,
                    member.structured_critic_dynamics_optimizer,
                ),
            ):
                if (predictor is None) != (optimizer is None):
                    raise RuntimeError(
                        f"training member has incomplete structured {name} predictor state"
                    )
            measured = update_ppo(
                member.actor,
                member.critic,
                member.actor_optimizer,
                member.critic_optimizer,
                rollout,
                ppo_config,
                generator=generator,
                actor_epochs=0 if warmup_active else None,
                rows=rows,
                structured_dynamics=member.structured_dynamics,
                structured_dynamics_optimizer=member.structured_dynamics_optimizer,
                structured_actor_auxiliary=(
                    ppo_config.structured_actor_auxiliary_active and not warmup_active
                ),
                structured_critic_dynamics=member.structured_critic_dynamics,
                structured_critic_dynamics_optimizer=(member.structured_critic_dynamics_optimizer),
                structured_critic_auxiliary=ppo_config.structured_critic_auxiliary_active,
                auxiliary_generator=auxiliary_generator,
                diagnostic_groups=diagnostic_groups,
                diagnostic_gradients=iteration % 25 == 0,
            )
            for kind, active in (
                ("actor", ppo_config.structured_actor_auxiliary_active),
                ("critic", ppo_config.structured_critic_auxiliary_active),
            ):
                if active:
                    measured.update(_structured_persistence_diagnostics(measured, kind=kind))
            # Per member, so one collapsed member stops the run as itself rather
            # than being averaged into three healthy ones.
            if not warmup_active and entropy_references[agent] is None:
                # The first update this member's actor actually applied, which is
                # the only iteration whose entropy describes where it started
                # rather than where the objective has moved it.
                entropy_references[agent] = _validate_policy_entropy_reference(
                    float(measured["entropy"])
                )
            _gate_update_metrics(
                measured,
                warmup_active=warmup_active,
                agent=None if population == 1 else agent,
            )
            update_metrics.update(_agent_fields(measured, agent, population))
            if initial_actor_provenance is not None:
                current_warmup_evs.append(float(measured["monte_carlo_explained_variance"]))
        if initial_actor_provenance is not None:
            critic_warmup_previous_evs = current_warmup_evs
            initial_actor_provenance["critic_warmup_state"][
                "last_monte_carlo_explained_variance"
            ] = list(critic_warmup_previous_evs)
        update_seconds = time.monotonic() - update_started
        iteration += 1
        metrics = {
            "iteration": iteration,
            "next_seed": next_seed,
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            "iteration_seconds": time.monotonic() - iteration_started,
            "update_seconds": update_seconds,
            "critic_replayed_states_per_second": (
                rollout.state_count
                * (
                    ppo_config.epochs
                    if ppo_config.critic_epochs is None
                    else ppo_config.critic_epochs
                )
                / max(update_seconds, 1e-9)
            ),
            "league_checkpoint": opponent_checkpoint,
            **rollout_diagnostics(rollout),
            **self_play_diagnostics,
            **league_diagnostics,
            **population_metrics,
            **replay_parity_metrics,
            **update_metrics,
            **warmup_diagnostics,
        }
        if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError(f"non-finite training metric: {metrics}")

        checkpoint_now = time.monotonic()
        clean_final = iteration >= args.iterations or bool(
            args.max_hours and (checkpoint_now - started) / 3600.0 >= args.max_hours
        )
        recovery_due = clean_final or checkpoint_timer.due(checkpoint_now)
        recovery_payload = build_recovery_payload(metrics) if recovery_due else None
        if recovery_due:
            checkpoint_timer.committed(checkpoint_now)
            last_checkpoint_iteration = iteration

        # League snapshots remain semantic PFSP history. They only need the
        # actor, while the much larger critic/optimizer/RNG recovery state above
        # is copied only for a due checkpoint. On checkpoint iterations, reuse
        # that payload's already-detached actor tensors instead of synchronously
        # copying the same GPU parameters to CPU a second time.
        actor_state = (
            (
                recovery_payload["actor"]
                if recovery_payload is not None
                else cpu_state_copy(members[0].actor.state_dict())
            )
            if population == 1 and not warmup_active
            else None
        )
        pending_commit = commit_executor.submit(
            commit_iteration, metrics, actor_state, recovery_payload
        )
        last_metrics = metrics
        print(json.dumps(metrics, sort_keys=True), flush=True)
        del rollout

    completed_commit = pending_commit.result() if pending_commit is not None else (None, None)
    completed_checkpoint, completed_snapshot = completed_commit
    if completed_snapshot is not None:
        league_snapshot_refs.append(completed_snapshot)
    if (
        initial_actor_provenance is not None
        and iteration >= args.iterations
        and not critic_warmup_complete
    ):
        commit_executor.shutdown(wait=True)
        writer.close()
        raise RuntimeError(
            "training reached its final iteration before the critic satisfied the "
            "Monte Carlo-return explained-variance actor-release gate"
        )
    # The max-hours boundary can become true while the last asynchronous
    # journal/snapshot commit finishes. Force that clean terminal state once,
    # but never rewrite an iteration already committed as periodic or final.
    if last_checkpoint_iteration != iteration:
        final_checkpoint = publish_checkpoint(build_recovery_payload(last_metrics))
        external_eval_process = _maybe_launch_external_eval(
            args,
            final_checkpoint,
            iteration,
            external_eval_process,
            wait_for_slot=True,
        )
    elif completed_checkpoint is not None:
        external_eval_process = _maybe_launch_external_eval(
            args,
            completed_checkpoint[0],
            completed_checkpoint[1],
            external_eval_process,
            wait_for_slot=True,
        )
    else:
        external_eval_process = _maybe_launch_external_eval(
            args, None, iteration, external_eval_process, wait_for_slot=True
        )
    commit_executor.shutdown(wait=True)
    writer.close()


if __name__ == "__main__":
    main()
