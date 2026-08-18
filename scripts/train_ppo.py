#!/usr/bin/env python3
"""Train a from-scratch Kaggriculture policy with self-play PPO."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from kaggriculture.inference import load_actor_artifact
from kaggriculture.league import (
    LEAGUE_OPPONENT_KEY,
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
from kaggriculture.ppo import (
    DEFAULT_ACTOR_GAE_LAMBDA,
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    MAX_VALUE_TARGET_SATURATED_FRACTION,
    MINIMUM_ACTOR_EPOCH_FRACTION,
    UPDATE_COMPILE_MODES,
    PpoConfig,
    make_optimizers,
    update_ppo,
    update_replay_parity,
)
from kaggriculture.provenance import (
    CALIBRATION_KNOBS,
    file_sha256,
    require_source_identity,
    run_provenance_from_decision,
    source_identity,
    validate_run_provenance,
)
from kaggriculture.registry import ARCHITECTURES, CONV_ENTITY, resolve_architecture
from kaggriculture.rollout import (
    ROLLOUT_FORWARD_MODES,
    RolloutBatch,
    allocate_rollout_storage,
    collect_mixed_play_rust,
    slice_trajectories,
)
from kaggriculture.rust_env import toolchain_identity
from kaggriculture.structured import StructuredConfig
from kaggriculture.telemetry import TensorboardMirror
from kaggriculture.training import (
    append_iteration_jsonl,
    checkpoint_payload,
    cpu_state_copy,
    load_checkpoint,
    metrics_journal_iteration,
    rollout_diagnostics,
    save_checkpoint,
    training_rng_states,
    write_checkpoint,
)


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
    parser.add_argument("--opponent-temperature", type=float, default=0.8)
    parser.add_argument(
        "--external-eval-every",
        type=int,
        default=0,
        help="iterations between diagnostic CPU evaluations vs external agents; 0 disables",
    )
    parser.add_argument(
        "--external-eval-opponents",
        default="starter,public-v27",
        help="comma-separated opponents forwarded to external_eval_worker.py",
    )
    parser.add_argument("--external-eval-seeds", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=CONV_ENTITY,
        help="actor/critic family; each family's structural flags default to that "
        "family's model configuration and a flag from another family is rejected",
    )
    add_model_config_arguments(parser)
    # These two read the dataclass rather than restating it. `actor_learning_rate`
    # is a measurement -- the largest rate whose full epoch fits inside
    # `target_kl` -- and a second copy here would silently outrank it whenever
    # this script is driven by hand. The neighbours below deliberately do not
    # follow: `--epochs` defaults to 1 against the dataclass's 4, because the
    # shipped schedule reaches its critic epochs through `--critic-epochs`.
    parser.add_argument("--actor-lr", type=float, default=PpoConfig.actor_learning_rate)
    parser.add_argument("--critic-lr", type=float, default=PpoConfig.critic_learning_rate)
    parser.add_argument("--lr-warmup-steps", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--critic-epochs",
        type=int,
        default=None,
        help="total critic epochs (>= --epochs; the excess are critic-only refits); "
        "defaults to --epochs",
    )
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--clip-low", type=float, default=0.80)
    parser.add_argument("--clip-high", type=float, default=1.28)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="reward discount; 1.0 preserves the exact final relative-bank objective",
    )
    parser.add_argument(
        "--actor-gae-lambda",
        type=float,
        default=DEFAULT_ACTOR_GAE_LAMBDA,
        help=(
            "GAE lambda; defaults to VAPO's alpha=0.05 value for the fixed "
            "719-action competition horizon. It sets the critic too: the target "
            "is the lambda-return the advantage came from"
        ),
    )
    parser.add_argument("--target-kl", type=float, default=0.03)
    # Sourced from the dataclass rather than restated, so the justification
    # recorded there cannot drift out of agreement with what the CLI ships.
    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=PpoConfig.entropy_coefficient,
        help=(
            "weight on the policy entropy bonus; the only exploration available, "
            "since replay parity pins the sampling temperature to 1.0"
        ),
    )
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    # Two phases, two knobs, decided separately by calibration: the collection
    # forward and the update compile different graphs, and their measured
    # speedups on the conv model fall on opposite sides of the threshold.
    # Neither knob is a boolean, and for the same reason on both sides -- the
    # decision is which execution mode, and the modes are not one measurement.
    # On the collection side that is measured: on production 112-game waves with
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
    # The collection forward is ~64% of a wave's wall clock, and these two
    # defaults are where the measurement landed on both axes rather than a
    # preference. Production 112-game waves, real BC actor: the rollout sweep
    # moves 8.91 s (eager/fp32) -> 5.36 s (inductor/bf16), 1.66x, and the
    # shipped 4-wave `scripts/audit_replay_parity.py` gate on the league-mixed
    # path moves worst max_kl 1.9089e-03 -> 2.2786e-04, 8.4x lower drift.
    # Faster and closer to parity at once: the update path is already Inductor
    # + bf16, so most of the collect/update gap is a systematic backend and
    # precision difference, and matching the update path's backend and
    # precision cancels it instead of adding to it. `--rollout-bfloat16` is the
    # collection precision; `--no-bfloat16` above is the update's, and they are
    # decided separately.
    parser.add_argument(
        "--rollout-forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        default="inductor",
        help="collection forward backend, and the whole compile decision for "
        "collection: eager does not compile; default inductor",
    )
    parser.add_argument(
        "--rollout-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the collection forward under bf16 autocast; default enabled",
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
        help="actor artifact (e.g. a BC clone) whose weights initialize a fresh run's actor",
    )
    parser.add_argument(
        "--critic-warmup-iterations",
        type=int,
        help=(
            "iterations of critic-only updates before the actor participates; "
            "belongs to the warm start, and a resumed run restores it from its checkpoint"
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    # Model-configuration flags are validated by the config dataclass itself,
    # so every entry point that builds one gets the same rules.
    positive = {
        "iterations": args.iterations,
        "games": args.games,
        "league_active_pool_size": args.league_active_pool_size,
        "episode_steps": args.episode_steps,
        "checkpoint_every": args.checkpoint_every,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
    }
    if args.critic_epochs is not None:
        positive["critic_epochs"] = args.critic_epochs
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {', '.join(invalid)}")
    if args.episode_steps != 720:
        raise ValueError("training requires the competition horizon: --episode-steps 720")
    # The k3 estimator is non-negative, and the trust region stops on
    # `batch_kl > target_kl`, so zero admits only the exactly-parity first
    # minibatch and anything negative admits nothing at all. Either collapses
    # the update to near-zero optimizer steps silently rather than erroring.
    if not math.isfinite(args.target_kl) or args.target_kl <= 0.0:
        raise ValueError("target KL must be finite and positive")
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
    if args.external_eval_every < 0 or (args.external_eval_every and args.external_eval_seeds < 1):
        raise ValueError("external evaluation needs a non-negative cadence and positive seeds")
    if args.league_games and args.league_games < configured_opponents:
        raise ValueError(
            "league games must cover the initial anchor and every configured "
            f"active/historical/built-in lane ({configured_opponents})"
        )
    if not math.isfinite(args.opponent_temperature) or args.opponent_temperature <= 0.0:
        raise ValueError("opponent temperature must be finite and positive")
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
    if args.critic_warmup_iterations is not None and args.init_actor_from is None:
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
    if not 0 < args.clip_low < 1 < args.clip_high:
        raise ValueError("clip interval must straddle one")
    if args.lr_warmup_steps < 0:
        raise ValueError("LR warmup steps cannot be negative")
    if args.gamma != 1.0:
        raise ValueError("Kaggriculture bank-delta rewards require --gamma 1.0")
    if not math.isfinite(args.actor_gae_lambda) or not 0.0 <= args.actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")
    if not math.isfinite(args.max_hours) or args.max_hours < 0.0:
        raise ValueError("max hours must be finite and non-negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
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
    if resolve_architecture(payload).name != architecture_name:
        raise ValueError("initial actor artifact architecture does not match arguments")
    if payload["model_config"] != model_config.to_dict():
        raise ValueError("initial actor artifact model configuration does not match arguments")
    actor.load_state_dict(pretrained.state_dict())
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "format_version": payload["format_version"],
        "iteration": int(payload.get("iteration", 0)),
        "bc_provenance": payload.get("bc_provenance"),
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


def _balanced_assignments(games: int, opponents: int, generator: np.random.Generator) -> np.ndarray:
    """Assign nearly equal game counts to each selected opponent."""
    if games < 1 or opponents < 1:
        raise ValueError("balanced assignments require positive games and opponents")
    assignments = np.arange(games, dtype=np.int64) % opponents
    generator.shuffle(assignments)
    return assignments


# Blend of the previous estimate and this iteration's measured score rate for
# PFSP opponent weighting; each measurement covers only ~20-50 games, so the
# estimate keeps some memory while still down-weighting a beaten opponent
# quickly. The prior seeds the blend, so a single clean sweep can never pin
# an estimate at exactly 1.0 and permanently retire an opponent.
LEAGUE_SCORE_RATE_EMA = 0.5
# Unsampled estimates decay toward the unmeasured prior each iteration.
# A stale estimate is most wrong exactly when the learner has changed the
# most, and the decay guarantees every opponent is eventually re-measured.
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
PARITY_STAGING_KEYS = ("self-play", "league")


def _parity_metric_key(component: str, statistic: str) -> str:
    return f"update_replay_{component}_{statistic}"


def _parity_staging_key(league_games: int) -> str:
    """Name the staging configuration a wave exercises.

    League play and pure self-play stage the rollout arena differently -- the
    full arena, or the self-play prefix view of it -- so the audit *cadence*
    tracks them separately. A cadence blind to the difference could audit
    whichever one the interval landed on while never examining the other,
    including the iteration where the league rows past the prefix are written
    for the first time.
    """
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


def _validate_league_score_rates(rates: object) -> dict[str, float]:
    if not isinstance(rates, dict):
        raise ValueError("resume checkpoint has no valid league score-rate state")
    validated: dict[str, float] = {}
    for key, rate in rates.items():
        if type(key) is not str or LEAGUE_OPPONENT_KEY.fullmatch(key) is None:
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
    """Drop unavailable external-eval opponents instead of blocking training.

    The probes are diagnostics: a public agent file cleaned out of /var/tmp
    must not make a multi-day production run unlaunchable. Every dropped spec
    is reported at launch, and losing all of them disables the cadence.
    """
    if not args.external_eval_every:
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
        args.external_eval_every = 0
    args.external_eval_opponents = ",".join(resolved)


def _maybe_launch_external_eval(
    args: argparse.Namespace,
    committed_iteration: int,
    league_directory: Path,
    process: subprocess.Popen | None,
) -> subprocess.Popen | None:
    """Launch at most one CPU worker evaluating the latest durable snapshot.

    The worker is diagnostics only: it appends to metrics-external.jsonl and
    its absence never blocks training. A still-running worker simply skips
    the tick, so cadence degrades gracefully when episodes run long.
    """
    if (
        not args.external_eval_every
        or committed_iteration < 1
        or committed_iteration % args.external_eval_every
    ):
        return process
    if process is not None and process.poll() is None:
        return process
    if process is not None and process.returncode:
        print(
            f"external eval worker exited with code {process.returncode}; see external-eval.log",
            file=sys.stderr,
            flush=True,
        )
    snapshot = league_directory / f"league-actor-{committed_iteration:08d}.pt"
    log_path = args.run_dir / "external-eval.log"
    try:
        with log_path.open("ab") as log:
            return subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "external_eval_worker.py"),
                    "--snapshot",
                    str(snapshot),
                    "--iteration",
                    str(committed_iteration),
                    "--output",
                    str(args.run_dir / "metrics-external.jsonl"),
                    "--opponents",
                    args.external_eval_opponents,
                    "--seeds",
                    str(args.external_eval_seeds),
                    "--episode-steps",
                    str(args.episode_steps),
                ],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
    except OSError as error:
        # Diagnostics must never kill training: ENOSPC on the log, EMFILE, or
        # a failed fork under memory pressure only skips this probe.
        print(f"external eval launch failed: {error}", file=sys.stderr, flush=True)
        return process


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
        "league_historical_opponents": args.league_historical_opponents,
        "league_active_pool_size": args.league_active_pool_size,
        # Which reference agents share the wave, and how many lanes they may
        # hold, decide what the learner plays against; a resume that changed
        # either would be generating different data under the same run.
        "league_builtin_opponents": ",".join(_league_builtin_opponents(args)),
        "league_builtin_lanes": args.league_builtin_lanes,
        "opponent_temperature": args.opponent_temperature,
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
    model_config: ModelConfig | StructuredConfig,
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


def _gate_update_metrics(update_metrics: Mapping[str, float], *, warmup_active: bool) -> None:
    """Stop the run on an update whose numbers say the next one is wasted.

    Ordered by cause, not by severity. An inflated first-minibatch KL at
    unchanged weights trips the trust region on minibatch zero, so checking the
    update count first would report the symptom; and a saturated value target
    explains a missing actor update rather than the other way round.
    """
    first_minibatch_kl = float(update_metrics["first_minibatch_approx_kl"])
    if first_minibatch_kl > MAX_FIRST_MINIBATCH_KL:
        raise RuntimeError(
            "first-minibatch KL at unchanged weights exceeded "
            f"{MAX_FIRST_MINIBATCH_KL}: {first_minibatch_kl}"
        )
    # A target the support cannot hold is regressed onto a constant edge label,
    # which then holds the critic where it is. A few are ordinary critic error.
    saturated_fraction = float(update_metrics["value_target_saturated_fraction"])
    if saturated_fraction > MAX_VALUE_TARGET_SATURATED_FRACTION:
        raise RuntimeError(
            "value targets saturated the critic support beyond "
            f"{MAX_VALUE_TARGET_SATURATED_FRACTION}: {saturated_fraction}"
        )
    if int(update_metrics["actor_updates"]) < 1 and not warmup_active:
        raise RuntimeError("PPO iteration completed without an actor update")
    # A trust region set against the wrong policy sharpness stops the epoch after
    # its first minibatch rather than before it, so the count above is 1 and
    # passes while the iteration trains on under 1% of the wave. Nothing else
    # reports it: `approx_kl` averages over the minibatches that stepped.
    intended = int(update_metrics["actor_minibatches_intended"])
    applied = int(update_metrics["actor_updates"])
    if not warmup_active and intended > 0 and applied < MINIMUM_ACTOR_EPOCH_FRACTION * intended:
        raise RuntimeError(
            f"actor applied {applied} of {intended} minibatches, below "
            f"{MINIMUM_ACTOR_EPOCH_FRACTION:.0%} of the epoch; the trust region "
            f"{'stopped it early' if update_metrics.get('kl_early_stop') else 'is not the cause'} "
            f"at max_approx_kl {float(update_metrics['max_approx_kl']):.4g}"
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)
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
    # is mode-valued, so `!=` rather than the `is not` a boolean allowed.
    # `rollout_bfloat16` is deliberately not among them: collection precision is
    # fixed configuration, pinned identical on every chain node instead of
    # attributed, so the decision carries nothing to compare it against. It is
    # recorded in `_training_data_config`, where a resume must match it exactly.
    if run_provenance is not None and any(
        run_provenance["calibration"][knob] != getattr(args, knob) for knob in CALIBRATION_KNOBS
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

    architecture = resolve_architecture(args.architecture)
    model_config: ModelConfig | StructuredConfig = model_config_from_args(architecture, args)
    ppo_config = PpoConfig(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        critic_epochs=args.critic_epochs,
        minibatch_size=args.minibatch_size,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        gamma=args.gamma,
        actor_gae_lambda=args.actor_gae_lambda,
        max_gradient_norm=args.max_gradient_norm,
        target_kl=args.target_kl,
        entropy_coefficient=args.entropy_coefficient,
        use_bfloat16=not args.no_bfloat16,
        update_compile_mode=args.update_compile_mode,
    )
    training_data_config = _training_data_config(args, device)
    # Derived from the configured trust region rather than fixed, because that
    # is what the ceiling means: the level at which the uncorrected parity
    # divergence matches the divergence this run's update deliberately allows.
    parity_ceilings = _parity_ceilings()
    actor = architecture.actor_class(model_config).to(device)
    critic = architecture.critic_class(model_config).to(device)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
    initial_actor_provenance = None
    critic_warmup_iterations = 0
    if args.init_actor_from is not None:
        initial_actor_provenance = _load_initial_actor(
            args.init_actor_from, actor, architecture.name, model_config, device
        )
        critic_warmup_iterations = args.critic_warmup_iterations or 0
        # The count travels inside the warm-start record so it survives a
        # resume; validation already guarantees the two arrive together.
        initial_actor_provenance["critic_warmup_iterations"] = critic_warmup_iterations
    generator = np.random.default_rng(args.seed + 1)
    iteration = 0
    next_seed = args.seed
    resume_payload = None
    if args.resume:
        # Architecture and model-config identity are validated inside
        # load_checkpoint before it mutates the freshly constructed models.
        resume_payload = load_checkpoint(
            args.resume,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            device=device,
        )
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
        # Restore warm-start provenance: the resumed run must keep treating
        # the iteration-0 league snapshot as a pretrained baseline, and must
        # keep freezing the actor for whatever remains of the critic warmup.
        initial_actor_provenance = resume_payload.get("initial_actor")
        critic_warmup_iterations = int(
            (initial_actor_provenance or {}).get("critic_warmup_iterations", 0)
        )

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
    parity_baseline: dict[str, float] = {}
    if resume_payload is not None:
        league_snapshot_manifest = _validate_league_manifest(
            resume_payload.get("league_snapshot_manifest"),
            current_iteration=iteration,
        )
        league_score_rates = _validate_league_score_rates(resume_payload.get("league_score_rates"))
        parity_baseline = _validate_parity_baseline(
            resume_payload.get("replay_parity_baseline"), parity_ceilings
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
                ppo_config=ppo_config,
                iteration=iteration,
                next_seed=next_seed,
                metrics=resume_payload["metrics"],
                training_rng_state=generator.bit_generator.state,
                training_data_config=training_data_config,
                league_snapshot_manifest=league_snapshot_manifest,
                league_score_rates=league_score_rates,
                replay_parity_baseline=dict(parity_baseline),
                source_identity=current_source_identity,
                run_provenance=run_provenance,
                initial_actor=initial_actor_provenance,
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
                    ppo_config=ppo_config,
                    iteration=iteration,
                    next_seed=next_seed,
                    metrics=resume_payload["metrics"],
                    training_rng_state=generator.bit_generator.state,
                    training_data_config=training_data_config,
                    league_snapshot_manifest=league_snapshot_manifest,
                    league_score_rates=league_score_rates,
                    replay_parity_baseline=dict(parity_baseline),
                    source_identity=current_source_identity,
                    run_provenance=run_provenance,
                    initial_actor=initial_actor_provenance,
                )
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
    # play, which produce fewer trajectories.
    self_play_rows = args.games * 2
    rollout_arena = allocate_rollout_storage(
        architecture.name,
        self_play_rows + args.league_games,
        args.episode_steps - 1,
        pin_memory=device.type == "cuda",
    )
    self_play_storage = {name: array[:self_play_rows] for name, array in rollout_arena.items()}

    commit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="commit")
    pending_commit: Future[None] | None = None

    def commit_iteration(payload: dict, actor_state: dict) -> None:
        """Durably commit one iteration's artifacts in canonical order.

        Runs on the single commit worker, so commits execute in submission
        order: league snapshot before the checkpoint that references it, and
        the checkpoint before its journal record, exactly as the recovery
        logic expects.
        """
        committed = int(payload["iteration"])
        snapshot = save_actor_state_snapshot(league_directory, model_config, actor_state, committed)
        league_snapshot_manifest[committed] = snapshot_sha256(snapshot.path)
        write_checkpoint(args.run_dir / "latest.pt", payload)
        if committed % args.checkpoint_every == 0:
            write_checkpoint(args.run_dir / f"checkpoint-{committed:06d}.pt", payload)
        append_iteration_jsonl(args.run_dir / "metrics.jsonl", payload["metrics"])
        writer.record(payload["metrics"])

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
            ppo_config=ppo_config,
            iteration=0,
            next_seed=next_seed,
            metrics={"iteration": 0},
            training_rng_state=generator.bit_generator.state,
            training_data_config=training_data_config,
            league_snapshot_manifest=league_snapshot_manifest,
            league_score_rates=league_score_rates,
            replay_parity_baseline=dict(parity_baseline),
            source_identity=current_source_identity,
            run_provenance=run_provenance,
            initial_actor=initial_actor_provenance,
        )

    # Iteration of the most recent audit per staging configuration. Deliberately
    # process-scoped rather than checkpointed: a resume rebuilds the compiled
    # callables and restages every buffer, so it should audit its first
    # iteration rather than wait out the cadence.
    last_parity_audit: dict[str, int] = {}
    external_eval_process: subprocess.Popen | None = None
    while iteration < args.iterations:
        if args.max_hours and (time.monotonic() - started) / 3600.0 >= args.max_hours:
            break
        iteration_started = time.monotonic()
        sampling_seed = int(generator.integers(0, np.iinfo(np.int64).max))
        opponent_checkpoint = ""
        league_diagnostics = {}
        selections = _select_league_opponents(
            args,
            list_actor_snapshots(league_directory),
            iteration,
            generator,
            league_score_rates,
            pretrained_start=initial_actor_provenance is not None,
        )
        league_games = args.league_games if selections else 0
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
            )
            opponent_temperatures = np.asarray(
                [
                    args.opponent_temperature if row.category == "active" else 1.0
                    for row in snapshots
                ],
                dtype=np.float32,
            )
            deterministic_opponents = np.asarray(
                [row.category != "active" for row in snapshots],
                dtype=np.bool_,
            )
            opponent_checkpoint = ",".join(row.label for row in selections)
        # Self-play and league games advance in one native wave, so the
        # learner forward covers every current-policy row at once and the
        # collector writes straight into the shared arena.
        rollout = collect_mixed_play_rust(
            actor,
            opponents if league_games else (),
            self_play_games=args.games,
            league_games=league_games,
            opponent_indices=assignments if league_games else None,
            builtin_lanes=builtin_lanes if league_games else (),
            seed_start=next_seed,
            episode_steps=args.episode_steps,
            temperature=args.temperature,
            opponent_temperature=args.opponent_temperature,
            opponent_temperatures=opponent_temperatures if league_games else None,
            deterministic_opponents=deterministic_opponents if league_games else None,
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
        next_seed += args.games + league_games
        # The wave's wall-clock is indivisible; per-part timing keys would
        # merely repeat it, so slice diagnostics keep only outcome metrics.
        indivisible_timings = ("rollout_seconds", "rollout_states_per_second")
        self_play_diagnostics = {
            f"self_play_{name}": value
            for name, value in rollout_diagnostics(
                slice_trajectories(rollout, 0, self_play_rows)
            ).items()
            if name not in indivisible_timings
        }
        if league_games:
            league_part = slice_trajectories(rollout, self_play_rows, rollout.trajectories)
            league_diagnostics = {
                f"league_{name}": value
                for name, value in rollout_diagnostics(league_part).items()
                if name not in indivisible_timings
            }
            opponent_diagnostics, measured_rates = _league_opponent_diagnostics(
                league_part, assignments, selections
            )
            league_diagnostics.update(opponent_diagnostics)
            _blend_league_score_rates(league_score_rates, measured_rates)
        # The update pins its importance ratio to one by replaying behavior
        # likelihoods through its own forward, so a staging bug applied
        # identically to both update-path sides would never move the KL guard.
        # Comparing that replay against the rollout's stored sampling
        # likelihoods catches exactly that class of bug. The bound is a KL
        # against the trust region the update already accepts, not a worst
        # component: see MAX_UPDATE_REPLAY_KL for why the extreme value is
        # reported but not gated.
        #
        # A breach aborts only when it reads as a defect rather than as drift,
        # and REPLAY_PARITY_STEP_CHANGE_FACTOR is where that distinction is
        # argued. The asymmetry matters because aborting is unrecoverable: the
        # audit precedes the update, so no checkpoint covers the iteration, and
        # a resume re-audits the same actor through the same code and dies
        # again. That is the correct outcome for a defect, which a human has to
        # go fix, and the wrong one for numerics drifting past a calibrated
        # bound in an otherwise healthy multi-day run -- which warns instead,
        # and leaves the trend in telemetry where it is the useful artifact.
        replay_parity_metrics: dict[str, float | int] = {}
        audited_staging = _parity_staging_key(league_games)
        if _parity_audit_due(last_parity_audit, iteration, audited_staging):
            replay_parity_metrics = update_replay_parity(
                actor,
                rollout,
                minibatch_size=ppo_config.minibatch_size,
                compile_mode=ppo_config.update_compile_mode,
                autocast_enabled=ppo_config.use_bfloat16 and device.type == "cuda",
            )
            # A head with no active components reports zero divergence, which
            # would pass the bound without having audited anything. The
            # calibration benchmark already refuses that; training must too,
            # or an audit can pass while having examined nothing. This one is
            # always fatal: it means the audit examined nothing, at any point
            # in the run, which is never expected drift.
            for component in PARITY_COMPONENTS:
                if replay_parity_metrics[f"update_replay_{component}_active_count"] < 1:
                    raise RuntimeError(f"update replay parity saw no active {component} components")
            breaches = _parity_breaches(
                replay_parity_metrics, parity_baseline or None, parity_ceilings
            )
            replay_parity_metrics["replay_parity_breached"] = len(breaches)
            replay_parity_metrics.update(
                _parity_fatal_thresholds(parity_baseline or None, parity_ceilings)
            )
            for message, is_defect in breaches:
                if is_defect:
                    continue
                # The journalled metrics carry iteration + 1, since the counter
                # advances before the record is written, so the warning names
                # the row it will appear in rather than the loop variable.
                print(
                    f"warning: iteration {iteration + 1} {message} — this is "
                    f"within {REPLAY_PARITY_STEP_CHANGE_FACTOR}x of the previous "
                    "audit and under the absolute ceiling, so it reads as drift "
                    "rather than a defect and the run continues",
                    file=sys.stderr,
                    flush=True,
                )
            # The baseline advances after every audit, warned breaches
            # included, so drift is always compared against recent drift and
            # can never accumulate into a false step change. The ceiling is
            # what stops that from ratcheting without limit.
            parity_baseline = _parity_measurements(replay_parity_metrics)
            last_parity_audit[audited_staging] = iteration
            defects = [message for message, is_defect in breaches if is_defect]
            if defects:
                raise RuntimeError(
                    "; ".join(defects)
                    + " — a step change away from the previous audit, or past the "
                    "absolute ceiling, rather than drift; resuming reproduces it. "
                    "A step change is a staging defect worth diagnosing directly; "
                    "a ceiling breach means the update forward's numerics no "
                    "longer support this bound, and the run continues as a fresh "
                    "one warm-started from the last actor under the repaired tree, "
                    "since repairing it changes the source identity these "
                    "checkpoints are bound to"
                )
        update_started = time.monotonic()
        # Critic-first warm start: a freshly initialized critic must fit
        # before its advantages may push a pretrained actor.
        warmup_active = iteration < critic_warmup_iterations
        update_metrics = update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            ppo_config,
            generator=generator,
            actor_epochs=0 if warmup_active else None,
        )
        _gate_update_metrics(update_metrics, warmup_active=warmup_active)
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
            **replay_parity_metrics,
            **update_metrics,
        }
        if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError(f"non-finite training metric: {metrics}")
        # Capture every mutable input on this thread, then commit the durable
        # artifacts (league snapshot, checkpoints, journal, mirror) in the
        # background so serialization and fsync overlap the next rollout. The
        # snapshot digest lands in the shared manifest inside the worker,
        # before the payload referencing that manifest is serialized.
        actor_state = cpu_state_copy(actor.state_dict())
        payload = checkpoint_payload(
            actor_state=actor_state,
            critic_state=cpu_state_copy(critic.state_dict()),
            actor_optimizer_state=cpu_state_copy(actor_optimizer.state_dict()),
            critic_optimizer_state=cpu_state_copy(critic_optimizer.state_dict()),
            model_config=model_config,
            ppo_config=ppo_config,
            iteration=iteration,
            next_seed=next_seed,
            metrics=metrics,
            source_identity=current_source_identity,
            rng_states=training_rng_states(),
            run_provenance=run_provenance,
            training_rng_state=generator.bit_generator.state,
            training_data_config=training_data_config,
            league_snapshot_manifest=league_snapshot_manifest,
            # Snapshot the estimates: the commit serializes on a worker
            # thread while the next iteration's blend mutates the live dict.
            league_score_rates=dict(league_score_rates),
            # Snapshot for the same reason: the audit at the next interval
            # replaces this configuration's entry while the commit serializes.
            replay_parity_baseline=dict(parity_baseline),
            initial_actor=initial_actor_provenance,
        )
        if pending_commit is not None:
            pending_commit.result()
        # The awaited commit belongs to the previous pass, so the newest
        # durable league snapshot is ``iteration - 1``; the current payload is
        # only being submitted now.
        external_eval_process = _maybe_launch_external_eval(
            args, iteration - 1, league_directory, external_eval_process
        )
        pending_commit = commit_executor.submit(commit_iteration, payload, actor_state)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        del rollout
    if pending_commit is not None:
        pending_commit.result()
        # The final snapshot is the one an operator most wants an external
        # number for; probe it if the cadence lands on it.
        _maybe_launch_external_eval(args, iteration, league_directory, external_eval_process)
    commit_executor.shutdown(wait=True)
    writer.close()


if __name__ == "__main__":
    main()
