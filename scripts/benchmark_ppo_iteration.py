#!/usr/bin/env python3
"""Benchmark one complete native rollout plus PPO replay at realistic batch sizes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import statistics
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

from kaggriculture.actor_dynamics import ActorDynamics
from kaggriculture.constants import DEFAULT_REWARD_MODE
from kaggriculture.entity import EntityConfig
from kaggriculture.inference import load_actor_artifact
from kaggriculture.lejepa import JepaObjective, load_artifact_objective
from kaggriculture.lejepa_model import LejepaCritic
from kaggriculture.model import ModelConfig, parameter_count
from kaggriculture.modelargs import (
    actor_model_config,
    add_model_config_arguments,
    model_config_from_args,
)
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    MAX_VALUE_TARGET_SATURATED_FRACTION,
    UPDATE_COMPILE_MODES,
    UPDATE_REPLAY_TAIL_LOGPROB,
    PpoConfig,
    family_ppo_defaults,
    make_optimizers,
    make_structured_dynamics_optimizer,
    update_ppo,
    update_replay_parity,
)
from kaggriculture.production import (
    PRODUCTION_ARCHITECTURE,
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_TEMPERATURE,
    production_model_config,
    production_ppo_config,
)
from kaggriculture.provenance import UNCOMPILED_UPDATE_COMPILE_MODE, file_sha256, source_identity
from kaggriculture.registry import (
    ARCHITECTURES,
    CONV_ENTITY,
    pair_towers,
    resolve_architecture,
)
from kaggriculture.rollout import (
    _ANIMAL_STOCK_COLUMNS,
    _CROP_SEED_COLUMNS,
    _PRODUCT_STOCK_COLUMNS,
    REWARD_MODES,
    ROLLOUT_FORWARD_MODES,
    allocate_rollout_storage,
    collect_mixed_play_rust,
)
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredConfig
from kaggriculture.structured_dynamics import StructuredCriticDynamics
from kaggriculture.telemetry import TensorboardMirror
from kaggriculture.training import rollout_diagnostics

_REPORT_PATH: Path | None = None
_REPORT_LINES: list[str] = []
_REPORT_MIRROR: TensorboardMirror | None = None
_REPORT_TENSORBOARD_DIR: Path | None = None

# Calibration reports are only valid launch evidence when their configuration
# matches production exactly. The benchmark therefore defaults to the shared
# structured architecture and fills that family's flags from the complete
# production contract rather than from the research dataclass defaults.
_PRODUCTION_PPO = production_ppo_config(update_compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE)
_PRODUCTION_LEAGUE_OPPONENTS = (
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS
)


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
    """Write one finite, standards-compliant JSONL record."""

    global _REPORT_MIRROR

    def normalize(value, path: str):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite benchmark metric at {path}: {value}")
        if isinstance(value, dict):
            return {key: normalize(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [normalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    rendered = json.dumps(normalize(payload, "payload"), sort_keys=True, allow_nan=False)
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
    parser.add_argument(
        "--games",
        default=str(PRODUCTION_SELF_PLAY_GAMES),
        help=(
            "comma-separated self-play games per iteration; defaults to the production "
            "batch (each game yields two trajectories). Pass multiple sizes only for a "
            "capacity sweep"
        ),
    )
    parser.add_argument(
        "--league-games",
        type=int,
        default=PRODUCTION_LEAGUE_GAMES,
        help="frozen-opponent games per iteration (each yields one learner trajectory)",
    )
    parser.add_argument(
        "--league-opponents",
        type=int,
        default=_PRODUCTION_LEAGUE_OPPONENTS,
        help="separate frozen actor forwards, matching active + historical production",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        # Six, not two, because the calibration launcher requires six and the
        # default should produce a report it accepts. Two drops the cold start
        # and leaves a single steady iteration, so every "median" downstream is
        # one sample, and one clock-boost dip can decide a knob for a
        # 500-iteration run.
        help="full rollout+update iterations per size; first is cold, later repeats are steady",
    )
    parser.add_argument(
        "--profile-repeat",
        type=int,
        help="zero-based warm iteration to profile in full; requires --trace-path",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        help="Chrome CPU/CUDA trace for --profile-repeat (single --games size only)",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--init-actor-from",
        type=Path,
        help="benchmark the action distribution of this BC actor instead of random initialization",
    )
    parser.add_argument(
        "--auxiliary-mode",
        choices=("off", "predictor", "enabled"),
        default="enabled",
        help="isolate ordinary PPO, predictor fitting, or predictor plus source updates; "
        "enabled uses the configured auxiliary coefficients (production defaults are zero)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=PRODUCTION_ARCHITECTURE,
        help="actor/critic family; the production family receives the exact production "
        "configuration, while other families retain their dataclass defaults",
    )
    add_model_config_arguments(parser)
    parser.add_argument("--epochs", type=int, default=_PRODUCTION_PPO["epochs"])
    parser.add_argument(
        "--critic-epochs",
        type=int,
        default=_PRODUCTION_PPO["critic_epochs"],
        help="total critic epochs (>= --epochs); the actor trains only in the first --epochs",
    )
    parser.add_argument("--minibatch-size", type=int, default=_PRODUCTION_PPO["minibatch_size"])
    parser.add_argument(
        "--actor-gae-lambda", type=float, default=_PRODUCTION_PPO["actor_gae_lambda"]
    )
    parser.add_argument("--temperature", type=float, default=PRODUCTION_TEMPERATURE)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default=DEFAULT_REWARD_MODE)
    parser.add_argument("--target-kl", type=float, default=_PRODUCTION_PPO["target_kl"])
    parser.add_argument(
        "--policy-loss-reduction",
        choices=("components", "states"),
        default=_PRODUCTION_PPO["policy_loss_reduction"],
    )
    parser.add_argument(
        "--policy-ratio-scope",
        choices=("components", "joint"),
        default=_PRODUCTION_PPO["policy_ratio_scope"],
    )
    parser.add_argument(
        "--structured-latent-coefficient",
        type=float,
        default=_PRODUCTION_PPO["structured_latent_coefficient"],
    )
    parser.add_argument(
        "--structured-decision-coefficient",
        type=float,
        default=_PRODUCTION_PPO["structured_decision_coefficient"],
    )
    for term in ("latent", "value"):
        parser.add_argument(
            f"--structured-critic-{term}-coefficient",
            type=float,
            default=_PRODUCTION_PPO[f"structured_critic_{term}_coefficient"],
        )
    parser.add_argument(
        "--economic-forecast-coefficient",
        type=float,
        default=None,
        help="forecast critic delta-supervision weight; defaults to the chosen family's, "
        "as in train_ppo",
    )
    # The `lejepa` family refuses to train without its world-model objective and
    # every other family refuses it, so its weights default to the family's.
    for term in ("prediction", "sigreg"):
        parser.add_argument(f"--jepa-{term}-coefficient", type=float, default=None)
    parser.add_argument(
        "--jepa-reward-coefficient",
        type=float,
        default=_PRODUCTION_PPO["jepa_reward_coefficient"],
    )
    parser.add_argument("--jepa-horizon", type=int, default=_PRODUCTION_PPO["jepa_horizon"])
    parser.add_argument(
        "--jepa-detach-target",
        action=argparse.BooleanOptionalAction,
        default=_PRODUCTION_PPO["jepa_detach_target"],
    )
    parser.add_argument(
        "--max-update-replay-kl",
        type=float,
        default=MAX_UPDATE_REPLAY_KL,
        help=(
            "maximum KL divergence between the rollout sampler likelihoods "
            "stored in the batch and an update-path replay at unchanged weights; "
            "the PPO denominator remains the actual sampler likelihood, while "
            "this gate detects execution-path drift before optimization"
        ),
    )
    parser.add_argument(
        "--max-update-replay-tail-fraction",
        type=float,
        default=MAX_UPDATE_REPLAY_TAIL_FRACTION,
        help=(
            "maximum share of sampled action components whose sampling-path and "
            f"update-path likelihoods disagree by more than {UPDATE_REPLAY_TAIL_LOGPROB} "
            "nats; this extends the mean KL's reach to localized staging defects "
            "too small in extent for it to notice"
        ),
    )
    parser.add_argument(
        "--max-first-minibatch-kl",
        type=float,
        default=MAX_FIRST_MINIBATCH_KL,
        help=(
            "maximum approximate KL of the first actor minibatch at unchanged "
            "weights, measured from the rollout sampler likelihood to the "
            "grad-mode minibatch likelihood; this gates collection-versus-update "
            "numeric drift before the first optimizer step"
        ),
    )
    parser.add_argument(
        "--max-value-target-saturated-fraction",
        type=float,
        default=MAX_VALUE_TARGET_SATURATED_FRACTION,
        help=(
            "maximum share of value targets the critic support may saturate; "
            "the lambda-return adds the critic's own prediction to the reward, "
            "so no support width contains it and this gates the degenerate end "
            "where the critic has collapsed onto the outermost atom"
        ),
    )
    # Collection and update are separate execution decisions. The collector
    # owns a whole-wave CUDA graph over fixed-address inputs; the update uses
    # its own compiled forward/backward. Synchronization at each phase boundary
    # keeps their timings attributable.
    #
    # The rollout knob is mode-valued because eager, compiled, and explicit
    # graph execution have different launch, warmup, and replay-parity behavior.
    # The default comes from the shared production constant rather than from an
    # old isolated-forward ranking. Precision is held separately: structured
    # collection uses a native-BF16 inference replica, matching the BF16 update
    # while retaining FP32 trainable weights and quantity heads.
    parser.add_argument(
        "--rollout-forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        default=PRODUCTION_ROLLOUT_FORWARD_MODE,
        help="execution mode of the collection forward; defaults to the production "
        "explicit CUDA graph path, while `eager` provides the uncompiled control",
    )
    parser.add_argument(
        "--rollout-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run structured collection with a native bf16 inference replica; the update "
        "path is bf16 regardless, so fp32 collection is a second precision rather than "
        "a safer one",
    )
    # `--compile-rollout` and `--compile-update` are both gone rather than kept
    # as their modes' projections: the calibration chain identifies each phase's
    # knob by the mode now, so nothing read either boolean, and a report carrying
    # one would break the chain outright -- the launcher requires every non-knob
    # configuration key to be identical across nodes, and a boolean derived from
    # a mode differs exactly where the chain varies it.
    parser.add_argument(
        "--update-compile-mode",
        choices=UPDATE_COMPILE_MODES,
        default="default",
        help="execution mode of the update-path forward/backward; `eager` is one of the modes, "
        "so this alone decides whether the update compiles",
    )
    parser.add_argument(
        "--rematerialize-actor-update",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replay the actor's update-forward activations in backward when an auxiliary "
        "reads its belief, trading a second forward for memory; the same function either way. "
        "Defaults to the chosen family's: lejepa retains them, every other family replays",
    )
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument(
        "--deterministic-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="match train_ppo's deterministic CUDA algorithm contract",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    args = parser.parse_args()
    if args.architecture == PRODUCTION_ARCHITECTURE:
        for name, value in production_model_config().items():
            if hasattr(args, name) and getattr(args, name) is None:
                setattr(args, name, value)
    # Resolved as train_ppo resolves them, so a benchmark and the launch it
    # gates agree on the family's objective and backbone rate.
    family = family_ppo_defaults(args.architecture, args.critic_architecture)
    for name in (
        "economic_forecast_coefficient",
        "jepa_prediction_coefficient",
        "jepa_sigreg_coefficient",
        "rematerialize_actor_update",
    ):
        if getattr(args, name) is None:
            setattr(args, name, family.get(name, getattr(PpoConfig, name)))
    args.structured_learning_rate = family.get(
        "structured_learning_rate", PpoConfig.structured_learning_rate
    )
    return args


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _verify_first_step_critic_state(rollout, self_play_games: int, seed_start: int) -> None:
    """Cross-validate stored critic-only state against a fresh native encode.

    Behavior values are replayed solely from stored critic inputs that nothing
    else consumes during collection: `critic_features` for the conv family, and
    the paired seat's critic extras for the structured family. Recomputing the
    opening step from a fresh BatchEnv and demanding bitwise staged-dtype
    equality catches arena-layout or staging corruption across the whole merged
    wave at production scale.
    """
    league_games = rollout.trajectories - self_play_games * 2
    seeds = np.arange(seed_start, seed_start + self_play_games + league_games, dtype=np.uint64)
    self_rows = np.arange(self_play_games * 2, dtype=np.int64)
    league_seats = rollout.seats[self_play_games * 2 :].astype(np.int64)
    league_rows = 2 * (self_play_games + np.arange(league_games, dtype=np.int64)) + league_seats
    stored_rows = np.concatenate([self_rows, league_rows])
    if rollout.architecture == CONV_ENTITY:
        fresh = np.asarray(load_native().BatchEnv(seeds).encoded()["critic_features"])
        if not np.array_equal(
            rollout.states["critic_features"][:, 0], fresh[stored_rows].astype(np.float16)
        ):
            raise RuntimeError(
                "stored first-step critic features do not match a fresh native encode"
            )
        return
    fresh = load_native().BatchEnv(seeds).structured()
    pair_rows = stored_rows ^ 1
    expected = {
        "critic_products": np.asarray(fresh["products"])[pair_rows][:, :, _PRODUCT_STOCK_COLUMNS],
        "critic_crops": np.asarray(fresh["crops"])[pair_rows][:, :, _CROP_SEED_COLUMNS],
        "critic_animals": np.asarray(fresh["animals"])[pair_rows][:, :, _ANIMAL_STOCK_COLUMNS],
        "opponent_unit_categorical": np.asarray(fresh["unit_categorical"])[pair_rows],
        "opponent_unit_continuous": np.asarray(fresh["unit_continuous"])[pair_rows],
        "opponent_unit_active": np.asarray(fresh["unit_active"])[pair_rows],
    }
    for name, value in expected.items():
        if not np.array_equal(rollout.states[name][:, 0], value):
            raise RuntimeError(f"stored first-step {name} does not match a fresh native encode")


def _hardware_identity(device: torch.device) -> dict[str, object]:
    """Return stable runtime hardware metadata used to match benchmark modes."""
    identity: dict[str, object] = {
        "device_type": device.type,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type != "cuda":
        return identity
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    identity.update(
        {
            "device_index": index,
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    )
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is not None:
        identity["device_uuid"] = str(device_uuid)
    return identity


def _completion_record(game_counts: list[int], repeats: int) -> dict[str, object]:
    """Describe the exact batch/repeat Cartesian product completed by this run."""
    return {
        "event": "benchmark_complete",
        "completed": True,
        "self_play_game_counts": game_counts,
        "repeats": repeats,
        "completed_batches": [
            {
                "self_play_games": games,
                "completed_repeats": list(range(repeats)),
            }
            for games in game_counts
        ],
        "iteration_records": len(game_counts) * repeats,
        "batch_summaries": len(game_counts),
    }


#: Each gate override, paired with the shipped constant that ceilings it.
_NUMERICS_GATES: tuple[tuple[str, str, float], ...] = (
    ("--max-update-replay-kl", "max_update_replay_kl", MAX_UPDATE_REPLAY_KL),
    (
        "--max-update-replay-tail-fraction",
        "max_update_replay_tail_fraction",
        MAX_UPDATE_REPLAY_TAIL_FRACTION,
    ),
    ("--max-first-minibatch-kl", "max_first_minibatch_kl", MAX_FIRST_MINIBATCH_KL),
    (
        "--max-value-target-saturated-fraction",
        "max_value_target_saturated_fraction",
        MAX_VALUE_TARGET_SATURATED_FRACTION,
    ),
)


def _validate_numerics_gates(args: argparse.Namespace) -> None:
    """Allow an override to tighten a gate, never to loosen it past training.

    The ceiling is the shipped constant itself rather than a number written
    beside it. A hardcoded ceiling goes stale silently and in the worse
    direction: this one sat at 1e-3 against a 1e-4 constant, so recalibrating
    the constant against a behavior-cloned actor -- which diverges four orders
    of magnitude further than the random init the original came from -- turned
    every calibration benchmark into an argument-parsing failure citing a limit
    nothing in the tree enforced any more.
    """
    for flag, attribute, ceiling in _NUMERICS_GATES:
        value = getattr(args, attribute)
        if not math.isfinite(value) or value <= 0.0 or value > ceiling:
            raise ValueError(f"{flag} must be finite, positive, and at most {ceiling}")


def main() -> None:
    args = parse_args()
    _configure_report(args.output, args.tensorboard_dir)
    game_counts = [int(value) for value in args.games.split(",")]
    if not game_counts or any(value < 1 for value in game_counts):
        raise ValueError("--games must be a comma-separated list of positive integers")
    if len(set(game_counts)) != len(game_counts):
        raise ValueError("--games cannot contain duplicate batch sizes")
    if args.league_games < 0:
        raise ValueError("--league-games cannot be negative")
    if args.league_opponents < 1:
        raise ValueError("--league-opponents must be positive")
    if args.league_games and args.league_opponents > args.league_games:
        raise ValueError("--league-games must cover every --league-opponents policy")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least two to separate cold and steady iterations")
    if args.epochs < 1 or args.minibatch_size < 1:
        raise ValueError("epochs and minibatch size must be positive")
    if not math.isfinite(args.actor_gae_lambda) or not 0.0 <= args.actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")
    if args.critic_epochs < args.epochs:
        raise ValueError("--critic-epochs cannot be fewer than --epochs")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if args.temperature != 1.0:
        raise ValueError("on-policy PPO benchmarking requires --temperature 1.0")
    _validate_numerics_gates(args)
    device = torch.device(args.device)
    if (args.profile_repeat is None) != (args.trace_path is None):
        raise ValueError("--profile-repeat and --trace-path must be supplied together")
    if args.profile_repeat is not None:
        if device.type != "cuda":
            raise ValueError("production iteration profiling requires CUDA")
        if not 1 <= args.profile_repeat < args.repeats:
            raise ValueError("--profile-repeat must be at least one and less than --repeats")
        if args.repeats - 2 < 2:
            raise ValueError("profiling requires at least two unprofiled steady iterations")
        if len(game_counts) != 1:
            raise ValueError("profiling requires one --games batch size for one complete trace")
        args.trace_path = args.trace_path.expanduser().resolve()
        if args.output is not None and args.trace_path == args.output.expanduser().resolve():
            raise ValueError("--trace-path must differ from the JSONL --output path")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.deterministic_training:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError(
                "deterministic training requires CUBLAS_WORKSPACE_CONFIG to be ':4096:8' or ':16:8'"
            )
    torch.use_deterministic_algorithms(args.deterministic_training)
    torch.backends.cudnn.deterministic = args.deterministic_training
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = not args.deterministic_training

    architecture = resolve_architecture(args.architecture)
    model_config: ModelConfig | StructuredConfig | EntityConfig = model_config_from_args(
        architecture, args
    )
    # The shipped structured case must inherit every production PPO field,
    # including both NextLat objectives. Other architecture sweeps cannot admit
    # typed structured predictors and retain their ordinary PPO schedule.
    ppo_schedule = _PRODUCTION_PPO if architecture.structured_inputs else asdict(PpoConfig())
    ppo_config = PpoConfig(
        **{
            **ppo_schedule,
            "epochs": args.epochs,
            "critic_epochs": args.critic_epochs,
            "minibatch_size": args.minibatch_size,
            "target_kl": args.target_kl,
            "actor_gae_lambda": args.actor_gae_lambda,
            "policy_loss_reduction": args.policy_loss_reduction,
            "policy_ratio_scope": args.policy_ratio_scope,
            "structured_latent_coefficient": args.structured_latent_coefficient,
            "structured_decision_coefficient": args.structured_decision_coefficient,
            "structured_critic_latent_coefficient": args.structured_critic_latent_coefficient,
            "structured_critic_value_coefficient": args.structured_critic_value_coefficient,
            "economic_forecast_coefficient": args.economic_forecast_coefficient,
            "jepa_prediction_coefficient": args.jepa_prediction_coefficient,
            "jepa_sigreg_coefficient": args.jepa_sigreg_coefficient,
            "jepa_reward_coefficient": args.jepa_reward_coefficient,
            "jepa_horizon": args.jepa_horizon,
            "jepa_detach_target": args.jepa_detach_target,
            "structured_learning_rate": args.structured_learning_rate,
            "use_bfloat16": not args.no_bfloat16,
            "update_compile_mode": args.update_compile_mode,
            "rematerialize_actor_update": args.rematerialize_actor_update,
        }
    )
    if args.auxiliary_mode == "off":
        ppo_config = PpoConfig(
            **{
                **asdict(ppo_config),
                "structured_latent_coefficient": 0.0,
                "structured_decision_coefficient": 0.0,
                "structured_critic_latent_coefficient": 0.0,
                "structured_critic_value_coefficient": 0.0,
                "jepa_prediction_coefficient": 0.0,
                "jepa_sigreg_coefficient": 0.0,
                "jepa_reward_coefficient": 0.0,
                "jepa_detach_target": False,
            }
        )
    initial_actor_digest = (
        file_sha256(args.init_actor_from) if args.init_actor_from is not None else None
    )
    identity = source_identity()
    emit(
        {
            "event": "configuration",
            "architecture": architecture.name,
            "device": str(device),
            "deterministic_training": args.deterministic_training,
            "hardware": _hardware_identity(device),
            "self_play_game_counts": game_counts,
            "league_games_per_iteration": args.league_games,
            "league_opponents": args.league_opponents,
            "league_active_opponents": min(
                PRODUCTION_LEAGUE_ACTIVE_OPPONENTS, args.league_opponents
            ),
            "league_historical_opponents": max(
                args.league_opponents - PRODUCTION_LEAGUE_ACTIVE_OPPONENTS, 0
            ),
            "episode_steps": PRODUCTION_EPISODE_STEPS,
            "physical_games_per_iteration": [games + args.league_games for games in game_counts],
            "repeats": args.repeats,
            "profile": {
                "repeat": args.profile_repeat,
                "trace_path": None if args.trace_path is None else str(args.trace_path),
                "activities": [] if args.profile_repeat is None else ["CPU", "CUDA"],
                "record_shapes": args.profile_repeat is not None,
                "profile_memory": args.profile_repeat is not None,
                "excluded_from_steady_summary": args.profile_repeat is not None,
            },
            "seed": args.seed,
            "initial_actor_sha256": initial_actor_digest,
            "auxiliary_mode": args.auxiliary_mode,
            "source_digest": identity["sha256"],
            "source_identity": identity,
            "temperature": args.temperature,
            "max_update_replay_kl": args.max_update_replay_kl,
            "max_update_replay_tail_fraction": args.max_update_replay_tail_fraction,
            "max_first_minibatch_kl": args.max_first_minibatch_kl,
            "max_value_target_saturated_fraction": args.max_value_target_saturated_fraction,
            "precision": {
                "use_bfloat16": ppo_config.use_bfloat16,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "model": model_config.to_dict(),
            # `asdict` carries `update_compile_mode`, which is the update knob
            # whole: the launcher reads it from here (`_declared_knobs`), so the
            # only record of what the update phase median was measured under is
            # the mode itself rather than any boolean beside it.
            "ppo": asdict(ppo_config),
            # The collection configuration the rollout phase median was measured
            # under; a benchmark that cannot say which mode it timed is not
            # evidence for one, and this median is what the launcher attributes
            # the rollout knob's speedup from. `rollout_forward_mode` is that
            # knob, whole -- no boolean projection of either knob is recorded,
            # because a projection would be a second name for one decision and
            # the chain would have two places to disagree. `precision.use_bfloat16`
            # above is the update path; `rollout_bfloat16` is this one.
            "rollout_forward_mode": args.rollout_forward_mode,
            "rollout_bfloat16": args.rollout_bfloat16,
            "torch": torch.__version__,
        }
    )

    for self_play_games in game_counts:
        # Keep initial parameters identical across batch sizes so throughput
        # comparisons are not confounded by a different policy/action mix.
        torch.manual_seed(args.seed)
        # Compilation state is part of that independence. The update path
        # specializes each static minibatch shape, and grad-tracking training
        # graphs specialize separately from the no-grad parity audit. Resetting
        # stops a later batch size from inheriting artifacts or cache pressure
        # that an earlier case paid to compile.
        torch._dynamo.reset()
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.empty_cache()
        actor = architecture.actor_class(model_config).to(device)
        if architecture.name == "causal-execution":
            from kaggriculture.device_ledger import get_device_ledger

            actor.set_device_ledger(get_device_ledger(device))
        initial_payload = None
        if args.init_actor_from is not None:
            pretrained, initial_payload = load_actor_artifact(args.init_actor_from, device=device)
            artifact_config = actor_model_config(pretrained.config)
            expected_config = actor_model_config(model_config)
            if (
                not isinstance(pretrained, architecture.actor_class)
                or artifact_config != expected_config
            ):
                raise ValueError("initial actor model configuration does not match benchmark")
            actor.load_state_dict(pretrained.state_dict())
            del pretrained
        critic = architecture.critic_class(model_config).to(device)
        pair_towers(actor, critic)
        # Keep critic and critic-predictor initialization independent of whether
        # the optional actor predictor is enabled.
        structured_critic_dynamics = (
            StructuredCriticDynamics(model_config).to(device)
            if architecture.structured_inputs and ppo_config.structured_critic_auxiliary_active
            else None
        )
        structured_dynamics = (
            (
                JepaObjective(model_config)
                if ppo_config.jepa_active
                else ActorDynamics(model_config)
            ).to(device)
            if architecture.structured_inputs and ppo_config.structured_actor_auxiliary_active
            else None
        )
        if initial_payload is not None:
            # As `train_ppo` warm-starts: a gate measuring a cloned backbone under
            # a fresh projector would not be measuring the run it gates.
            load_artifact_objective(initial_payload, structured_dynamics)
            del initial_payload
        frozen_opponent_state = {
            name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
        }
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
        structured_dynamics_optimizer = (
            make_structured_dynamics_optimizer(
                structured_dynamics,
                ppo_config,
                # The LeJEPA optimizer owns the shared backbone, as in train_ppo.
                actor=actor if ppo_config.jepa_active else None,
            )
            if structured_dynamics is not None
            else None
        )
        structured_critic_dynamics_optimizer = (
            make_structured_dynamics_optimizer(structured_critic_dynamics, ppo_config)
            if structured_critic_dynamics is not None
            else None
        )
        generator = np.random.default_rng(args.seed)
        # Predictor window sampling is an independent production RNG stream.
        # Sharing rollout assignment RNG would make later repeats collect
        # different games merely because NextLat is enabled.
        auxiliary_generator = (
            np.random.default_rng(args.seed + 2) if ppo_config.structured_auxiliary_active else None
        )
        seed_cursor = args.seed
        physical_games = self_play_games + args.league_games
        # Production collects the whole mixed wave into one reusable pinned
        # arena; mirror that here so staging behavior matches training.
        arena = allocate_rollout_storage(
            architecture.name,
            self_play_games * 2 + args.league_games,
            PRODUCTION_EPISODE_STEPS - 1,
            pin_memory=device.type == "cuda",
            action_interface=actor.config.action_interface,
        )
        repeat_payloads = []

        for repeat in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            profiled = repeat == args.profile_repeat
            # Only the selected full wave installs Kineto or record-function
            # ranges. Normal timing runs never invoke profiler instrumentation.
            with (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )
                if profiled
                else nullcontext()
            ) as profiler:
                with (
                    torch.profiler.record_function("opponent_setup") if profiled else nullcontext()
                ):
                    setup_started = time.perf_counter()

                    opponents = []
                    assignments = None
                    if args.league_games:
                        # Production reconstructs selected frozen actors from archive
                        # snapshots on every iteration. Keep that object lifecycle in
                        # the benchmark: compiled current actor/critic graphs persist,
                        # while frozen-policy wrappers are fresh each repeat.
                        opponents = [
                            architecture.actor_class(model_config).to(device)
                            for _ in range(args.league_opponents)
                        ]
                        for opponent in opponents:
                            opponent.load_state_dict(frozen_opponent_state)
                            opponent.requires_grad_(False)
                        assignments = np.arange(args.league_games, dtype=np.int64) % len(opponents)
                        generator.shuffle(assignments)
                        # No per-lane temperature or determinism arrays: they existed to
                        # reproduce production's split between stochastic active lanes
                        # and argmax historical ones, and production is now uniform at
                        # the learner's temperature. Keeping them would benchmark a wave
                        # production no longer runs.
                    # Opponent reconstruction is a real per-iteration cost and belongs
                    # in the iteration total, but it is compile-invariant: the same
                    # module construction and state-dict load happens whichever way the
                    # knobs are set. Leaving it inside the rollout span made the
                    # measured rollout ratio (c + r_eager) / (c + r_compiled), which is
                    # biased toward 1.0 in both directions -- it shrinks a loss and a
                    # win alike. That was harmless while the blended total decided one
                    # knob; now that the rollout median decides its own knob against a
                    # 1.05 gate, a constant added to both sides is a thumb on the
                    # scale. Timed separately so the phase the knob turns on contains
                    # only what the knob changes.
                    _synchronize(device)
                    opponent_setup_seconds = time.perf_counter() - setup_started

                wave_seed_start = seed_cursor
                with (
                    torch.profiler.record_function("production_rollout")
                    if profiled
                    else nullcontext()
                ):
                    rollout_started = time.perf_counter()
                    rollout = collect_mixed_play_rust(
                        actor,
                        opponents,
                        self_play_games=self_play_games,
                        league_games=args.league_games,
                        opponent_indices=assignments,
                        seed_start=wave_seed_start,
                        episode_steps=PRODUCTION_EPISODE_STEPS,
                        temperature=args.temperature,
                        reward_mode=args.reward_mode,
                        opponent_temperature=args.temperature,
                        sampling_seed=int(generator.integers(0, np.iinfo(np.int64).max)),
                        # The frozen league ensemble follows the learner. Since the
                        # mode became authoritative, `compile_models` decides only
                        # whether that stacked forward compiles, and the configuration
                        # measurement selected compiled both -- a compiled learner
                        # beside an eager ensemble is a mix nothing trains in.
                        forward_mode=args.rollout_forward_mode,
                        forward_autocast=args.rollout_bfloat16,
                        storage=arena,
                        # As training does: the `lejepa` critic's values come from the wave.
                        critic=critic if isinstance(critic, LejepaCritic) else None,
                    )
                    seed_cursor += physical_games
                    _synchronize(device)
                    rollout_seconds = time.perf_counter() - rollout_started
                del opponents

                # Compare the actual sampler likelihoods against update replay before
                # weights change. PPO keeps the stored sampler as its denominator;
                # this separate audit measures numerical execution-path divergence.
                with (
                    torch.profiler.record_function("update_replay_parity")
                    if profiled
                    else nullcontext()
                ):
                    parity_started = time.perf_counter()
                    parity = update_replay_parity(
                        actor,
                        rollout,
                        minibatch_size=ppo_config.minibatch_size,
                        compile_mode=ppo_config.update_compile_mode,
                        autocast_enabled=ppo_config.use_bfloat16 and device.type == "cuda",
                    )
                    _synchronize(device)
                    parity_seconds = time.perf_counter() - parity_started
                    for component in ("unit", "kind", "quantity"):
                        if parity[f"update_replay_{component}_active_count"] < 1:
                            raise RuntimeError(
                                f"update replay parity saw no active {component} components"
                            )
                    if parity["update_replay_max_kl"] > args.max_update_replay_kl:
                        raise RuntimeError(
                            "sampling-vs-update policy divergence exceeded "
                            f"{args.max_update_replay_kl}: {parity['update_replay_max_kl']}"
                        )
                    if (
                        parity["update_replay_max_tail_fraction"]
                        > args.max_update_replay_tail_fraction
                    ):
                        raise RuntimeError(
                            "sampling-vs-update materially divergent component share exceeded "
                            f"{args.max_update_replay_tail_fraction}: "
                            f"{parity['update_replay_max_tail_fraction']}"
                        )
                    _verify_first_step_critic_state(rollout, self_play_games, wave_seed_start)

                with (
                    torch.profiler.record_function("production_ppo_update")
                    if profiled
                    else nullcontext()
                ):
                    update_started = time.perf_counter()
                    update_metrics = update_ppo(
                        actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        rollout,
                        ppo_config,
                        generator=generator,
                        structured_dynamics=structured_dynamics,
                        structured_dynamics_optimizer=structured_dynamics_optimizer,
                        structured_actor_auxiliary=(
                            structured_dynamics is not None and args.auxiliary_mode == "enabled"
                        ),
                        structured_critic_dynamics=structured_critic_dynamics,
                        structured_critic_dynamics_optimizer=structured_critic_dynamics_optimizer,
                        structured_critic_auxiliary=(
                            structured_critic_dynamics is not None
                            and args.auxiliary_mode == "enabled"
                        ),
                        auxiliary_generator=auxiliary_generator,
                        diagnostic_groups={
                            "self_play": np.arange(rollout.trajectories) < self_play_games * 2,
                            "league": np.arange(rollout.trajectories) >= self_play_games * 2,
                        },
                        diagnostic_gradients=repeat == 1,
                    )
                    _synchronize(device)
                    update_seconds = time.perf_counter() - update_started
                    # Gate the first-minibatch KL before the actor-update count: an
                    # inflated KL at unchanged weights trips the trust region on
                    # minibatch zero, so checking update counts first would report the
                    # symptom instead of the cause.
                    first_minibatch_kl = float(update_metrics["first_minibatch_component_kl"])
                    if first_minibatch_kl > args.max_first_minibatch_kl:
                        raise RuntimeError(
                            "first-minibatch KL at unchanged weights exceeded "
                            f"{args.max_first_minibatch_kl}: {first_minibatch_kl}"
                        )
                    saturated_fraction = float(update_metrics["value_target_saturated_fraction"])
                    if saturated_fraction > args.max_value_target_saturated_fraction:
                        raise RuntimeError(
                            "value targets saturated the critic support beyond "
                            f"{args.max_value_target_saturated_fraction}: {saturated_fraction}"
                        )
                    if int(update_metrics["actor_updates"]) < 1:
                        raise RuntimeError("benchmark iteration completed without an actor update")
            if profiled:
                args.trace_path.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(args.trace_path))
            # Release captured events before subsequent steady waves.
            del profiler
            # The parity gate is benchmark-only instrumentation; production
            # iterations are opponent reconstruction plus rollout plus update,
            # so the calibration decision must be based on exactly that. Setup
            # is a term of the total but a term neither knob moves, which is
            # why it is added here rather than folded into a phase.
            total_seconds = opponent_setup_seconds + rollout_seconds + update_seconds
            diagnostics = rollout_diagnostics(rollout)
            payload = {
                "event": "iteration",
                "phase": "profiled"
                if profiled
                else ("cold_start" if repeat == 0 else "steady_state"),
                "profiled": profiled,
                "trace_path": str(args.trace_path) if profiled else None,
                "repeat": repeat,
                "self_play_games": self_play_games,
                "league_games": args.league_games,
                "physical_games": physical_games,
                "learner_trajectories": rollout.trajectories,
                "learner_states": rollout.state_count,
                "opponent_setup_seconds": opponent_setup_seconds,
                "rollout_seconds": rollout_seconds,
                "update_replay_parity_seconds": parity_seconds,
                "update_seconds": update_seconds,
                "total_seconds": total_seconds,
                "physical_games_per_rollout_second": physical_games / rollout_seconds,
                "learner_states_per_rollout_second": rollout.state_count / rollout_seconds,
                "critic_replayed_states_per_second": (
                    rollout.state_count
                    * (
                        ppo_config.epochs
                        if ppo_config.critic_epochs is None
                        else ppo_config.critic_epochs
                    )
                    / update_seconds
                ),
                "iterations_per_hour": 3600.0 / total_seconds,
                "peak_cuda_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
                "peak_cuda_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                ),
                "current_cuda_allocated_bytes": (
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                ),
                "current_cuda_reserved_bytes": (
                    torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
                ),
                "process_lifetime_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "actor_parameters": parameter_count(actor),
                "critic_parameters": parameter_count(critic),
                "money_mean": diagnostics["money_mean"],
                "score_rate": diagnostics["score_rate"],
                "tie_fraction": diagnostics["tie_fraction"],
                **parity,
                **update_metrics,
            }
            emit(payload)
            repeat_payloads.append(payload)
            del rollout
            gc.collect()

        steady = [item for item in repeat_payloads[1:] if not item["profiled"]]
        emit(
            {
                "event": "batch_summary",
                "self_play_games": self_play_games,
                "league_games": args.league_games,
                "physical_games": physical_games,
                "steady_iterations": len(steady),
                "cold_total_seconds": repeat_payloads[0]["total_seconds"],
                "cold_iterations_per_hour": repeat_payloads[0]["iterations_per_hour"],
                "cold_physical_games_per_rollout_second": repeat_payloads[0][
                    "physical_games_per_rollout_second"
                ],
                "steady_total_seconds_median": statistics.median(
                    item["total_seconds"] for item in steady
                ),
                # The phases are summarized separately because compilation is
                # decided separately for each. `total_seconds` is exactly these
                # three summed -- a device synchronization ends each -- so a
                # per phase median is a measurement rather than an attribution.
                # Setup is reported alongside them because it belongs to the
                # total the projection has to reconstruct, even though no knob
                # moves it.
                "steady_opponent_setup_seconds_median": statistics.median(
                    item["opponent_setup_seconds"] for item in steady
                ),
                "steady_rollout_seconds_median": statistics.median(
                    item["rollout_seconds"] for item in steady
                ),
                "steady_update_seconds_median": statistics.median(
                    item["update_seconds"] for item in steady
                ),
                "steady_iterations_per_hour_median": statistics.median(
                    item["iterations_per_hour"] for item in steady
                ),
                "steady_physical_games_per_rollout_second_median": statistics.median(
                    item["physical_games_per_rollout_second"] for item in steady
                ),
                "steady_critic_replayed_states_per_second_median": statistics.median(
                    item["critic_replayed_states_per_second"] for item in steady
                ),
            }
        )
        del (
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            structured_dynamics,
            structured_dynamics_optimizer,
            structured_critic_dynamics,
            structured_critic_dynamics_optimizer,
            frozen_opponent_state,
            arena,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    emit(_completion_record(game_counts, args.repeats))
    _configure_report(None)


if __name__ == "__main__":
    main()
