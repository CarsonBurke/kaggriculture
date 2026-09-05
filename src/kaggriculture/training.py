"""Checkpointing and diagnostics for long-running PPO jobs."""

from __future__ import annotations

import errno
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.constants import QUANTITY_BINS
from kaggriculture.inference import CHECKPOINT_FORMAT_VERSION, POPULATION_CHECKPOINT_KEY
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.orientation import Orientation
from kaggriculture.ppo import PpoConfig
from kaggriculture.provenance import (
    CALIBRATION_KNOBS,
    UPDATE_COMPILE_MODE_KNOB,
    validate_run_provenance,
    validate_source_identity,
)
from kaggriculture.registry import architecture_of, architecture_of_config, resolve_architecture
from kaggriculture.rollout import RolloutBatch
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic
from kaggriculture.structured_dynamics import StructuredCriticDynamics, StructuredDynamics

AnyActor = FarmActor | StructuredActor
AnyCritic = DistributionalCritic | StructuredCritic
AnyModelConfig = ModelConfig | StructuredConfig

#: The base states every member owns. Structured learners may additionally own
#: independent actor- and critic-side training-only predictors and optimizers.
AGENT_STATE_KEYS = ("actor", "critic", "actor_optimizer", "critic_optimizer")
PREDICTOR_STATE_KEY_PAIRS = (
    ("structured_dynamics", "structured_dynamics_optimizer"),
    ("structured_critic_dynamics", "structured_critic_dynamics_optimizer"),
)
OPTIONAL_AGENT_STATE_KEYS = tuple(
    key for predictor_pair in PREDICTOR_STATE_KEY_PAIRS for key in predictor_pair
)


def _complete_agent_state(state: object) -> bool:
    if not isinstance(state, Mapping):
        return False
    keys = set(state)
    allowed = {
        *AGENT_STATE_KEYS,
        *OPTIONAL_AGENT_STATE_KEYS,
        "orientation",
    }
    return set(AGENT_STATE_KEYS) <= keys <= allowed and all(
        (predictor in keys) == (optimizer in keys)
        for predictor, optimizer in PREDICTOR_STATE_KEY_PAIRS
    )


@dataclass(frozen=True, slots=True)
class TrainingAgent:
    """One population member's live modules and optimizers.

    A single learner is a population of one, so the checkpoint API takes a
    sequence of these rather than a bare actor/critic pair and there is exactly
    one code path to keep correct. The base optimizers are optional only because
    evaluation-side readers restore weights without ever stepping them.
    """

    actor: AnyActor
    critic: AnyCritic
    actor_optimizer: torch.optim.Optimizer | None = None
    critic_optimizer: torch.optim.Optimizer | None = None
    structured_dynamics: StructuredDynamics | None = None
    structured_dynamics_optimizer: torch.optim.Optimizer | None = None
    structured_critic_dynamics: StructuredCriticDynamics | None = None
    structured_critic_dynamics_optimizer: torch.optim.Optimizer | None = None

    def state(self) -> dict[str, Any]:
        """This member's complete recovery state."""
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise ValueError("a checkpointed agent needs both of its optimizers")
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        for predictor_key, optimizer_key in PREDICTOR_STATE_KEY_PAIRS:
            predictor = getattr(self, predictor_key)
            optimizer = getattr(self, optimizer_key)
            if (predictor is None) != (optimizer is None):
                label = predictor_key.replace("_", " ")
                raise ValueError(f"{label} and its optimizer must be checkpointed together")
            if predictor is not None:
                assert optimizer is not None
                state[predictor_key] = predictor.state_dict()
                state[optimizer_key] = optimizer.state_dict()
        return state


def checkpoint_agent_states(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every member's states, in agent order, from either payload shape."""
    members = payload.get(POPULATION_CHECKPOINT_KEY)
    if isinstance(members, list):
        return list(members)
    # A single learner's orientation lives at the top level beside its states;
    # flattening it in here keeps every orientation reader shape-agnostic.
    single = {
        name: payload[name]
        for name in (*AGENT_STATE_KEYS, *OPTIONAL_AGENT_STATE_KEYS)
        if name in payload
    }
    if "orientation" in payload:
        single["orientation"] = payload["orientation"]
    return [single]


def checkpoint_member_orientations(payload: Mapping[str, Any]) -> list[Orientation]:
    """Each member's recorded orientation, in agent order.

    Absent reads as identity: payloads that predate the field were rendered
    without one, so resuming them under identity is the truth, not a fallback.
    """
    return [
        Orientation(int(state.get("orientation", int(Orientation.IDENTITY))))
        for state in checkpoint_agent_states(payload)
    ]


def require_checkpoint_format(payload: dict[str, Any]) -> None:
    """Reject checkpoints from incompatible model and action schemas.

    Resume demands the current format exactly — a stale checkpoint must fail
    with a format error, not a downstream schema error.
    """
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if not isinstance(payload.get("seed_usage"), list):
        raise ValueError("checkpoint lacks seed_usage provenance; fresh training is required")
    members = payload.get(POPULATION_CHECKPOINT_KEY)
    if members is not None and (
        not isinstance(members, list)
        or len(members) < 2
        or any(not _complete_agent_state(entry) for entry in members)
    ):
        raise ValueError("checkpoint population is not a list of complete agent states")
    # A single learner keeps its states at the top level, so validate the same
    # required keys and optional-pair integrity as a population member.
    if members is None and not _complete_agent_state(checkpoint_agent_states(payload)[0]):
        raise ValueError("checkpoint is missing or has incomplete single-learner training state")
    if isinstance(members, list):
        for predictor_key, _ in PREDICTOR_STATE_KEY_PAIRS:
            presence = [predictor_key in entry for entry in members]
            if any(presence) != all(presence):
                label = predictor_key.replace("_", " ")
                raise ValueError(f"checkpoint population cannot mix {label} presence")
    states = members if isinstance(members, list) else checkpoint_agent_states(payload)
    has_predictor = any(
        any(predictor_key in state for predictor_key, _ in PREDICTOR_STATE_KEY_PAIRS)
        for state in states
    )
    if ("structured_auxiliary_rng" in payload) != has_predictor:
        raise ValueError("checkpoint structured predictor recovery RNG is incomplete")
    if (payload.get("structured_gate_state") is not None) != has_predictor:
        raise ValueError("checkpoint structured predictor gate state is incomplete")
    identity = validate_source_identity(payload.get("source_identity"))
    run_provenance = validate_run_provenance(payload.get("run_provenance"))
    if run_provenance is not None and run_provenance["source_identity"] != identity:
        raise ValueError("checkpoint run provenance source does not match source identity")
    if run_provenance is not None and (
        not isinstance(payload.get("training_data_config"), dict)
        or any(
            payload["training_data_config"].get(knob) != run_provenance["calibration"][knob]
            for knob in CALIBRATION_KNOBS
        )
    ):
        raise ValueError("checkpoint compile mode does not match run provenance")


def rollout_diagnostics(rollout: RolloutBatch) -> dict[str, float | int]:
    margins = rollout.final_money - rollout.opponent_money
    outcomes = (margins > 0).astype(np.float32) - (margins < 0).astype(np.float32)
    active_units = rollout.unit_active
    active_markets = rollout.market_active
    active_quantities = rollout.market_quantity_active
    unit_actions = rollout.unit_actions[active_units]
    market_kinds = rollout.market_kinds[active_markets]
    market_quantities = rollout.market_quantities[active_quantities]
    return {
        "rollout_trajectories": rollout.trajectories,
        "rollout_states": rollout.state_count,
        "rollout_horizon": rollout.horizon,
        "rollout_seconds": rollout.elapsed_seconds,
        "rollout_states_per_second": rollout.state_count / max(rollout.elapsed_seconds, 1e-9),
        "rollout_entropy": rollout.mean_entropy,
        # Quantiles rather than extremes. Final money floors at zero and most
        # games end near it -- a production wave measured a median of 63 against
        # a mean of 27,903 and a maximum of 126,405 -- so the minimum is a flat
        # zero line the moment any one trajectory goes broke, and the maximum is
        # a single lucky game that grows with the sample count rather than
        # describing the policy. The mean is kept because the total economy is
        # the thing being maximized, but under that skew it moves with the tail
        # and the quantiles are what show the distribution.
        "money_mean": float(rollout.final_money.mean()),
        "money_p10": float(np.quantile(rollout.final_money, 0.10)),
        "money_median": float(np.median(rollout.final_money)),
        "money_p90": float(np.quantile(rollout.final_money, 0.90)),
        "margin_abs_mean": float(np.abs(margins).mean()),
        "score_rate": float(((outcomes + 1.0) / 2.0).mean()),
        "seat_zero_score_rate": float(
            ((outcomes[rollout.seats == 0] + 1.0) / 2.0).mean()
            if (rollout.seats == 0).any()
            else 0.0
        ),
        "seat_one_score_rate": float(
            ((outcomes[rollout.seats == 1] + 1.0) / 2.0).mean()
            if (rollout.seats == 1).any()
            else 0.0
        ),
        "tie_fraction": float((outcomes == 0).mean()),
        "unit_pass_fraction": float((unit_actions == UnitAction.PASS).mean()),
        "unit_move_fraction": float(
            ((unit_actions >= UnitAction.NORTH) & (unit_actions <= UnitAction.WEST)).mean()
        ),
        "unit_shed_fraction": float(
            ((unit_actions >= UnitAction.DROP) & (unit_actions <= UnitAction.PICKUP_SHEEP)).mean()
        ),
        "unit_place_fraction": float(
            (
                (unit_actions >= UnitAction.PLACE_GOOSE) & (unit_actions <= UnitAction.PLACE_SHEEP)
            ).mean()
        ),
        "unit_plant_fraction": float(
            (
                (unit_actions >= UnitAction.PLANT_WHEAT) & (unit_actions <= UnitAction.PLANT_MELON)
            ).mean()
        ),
        "unit_water_fraction": float((unit_actions == UnitAction.WATER).mean()),
        "unit_harvest_fraction": float((unit_actions == UnitAction.HARVEST).mean()),
        "unit_fertilize_fraction": float((unit_actions == UnitAction.FERTILIZE).mean()),
        "unit_dig_fraction": float((unit_actions == UnitAction.DIG).mean()),
        "unit_build_fraction": float(
            (
                (unit_actions == UnitAction.BUILD_COOP) | (unit_actions == UnitAction.BUILD_PASTURE)
            ).mean()
        ),
        "unit_feed_fraction": float((unit_actions == UnitAction.FEED).mean()),
        "unit_collect_fertilizer_fraction": float(
            (unit_actions == UnitAction.COLLECT_FERTILIZER).mean()
        ),
        "unit_care_fraction": float((unit_actions == UnitAction.CARE).mean()),
        "market_stop_fraction": float((market_kinds == MarketKind.STOP).mean()),
        "market_hire_fraction": float((market_kinds == MarketKind.HIRE).mean()),
        "market_buy_land_fraction": float((market_kinds == MarketKind.BUY_LAND).mean()),
        "market_buy_seed_fraction": float(
            (
                (market_kinds >= MarketKind.BUY_SEED_WHEAT)
                & (market_kinds <= MarketKind.BUY_SEED_MELON)
            ).mean()
        ),
        "market_buy_product_fraction": float(
            (
                (market_kinds >= MarketKind.BUY_PRODUCT_WHEAT)
                & (market_kinds <= MarketKind.BUY_PRODUCT_FERTILIZER)
            ).mean()
        ),
        "market_buy_animal_fraction": float(
            (
                (market_kinds >= MarketKind.BUY_ANIMAL_GOOSE)
                & (market_kinds <= MarketKind.BUY_ANIMAL_SHEEP)
            ).mean()
        ),
        "market_sell_fraction": float(
            (
                (market_kinds >= MarketKind.SELL_WHEAT)
                & (market_kinds <= MarketKind.SELL_FERTILIZER)
            ).mean()
        ),
        "market_quantity_fraction": float(active_quantities.sum() / max(1, active_markets.sum())),
        "market_quantity_mean": float(
            np.asarray(QUANTITY_BINS)[market_quantities].mean() if market_quantities.size else 0.0
        ),
    }


def cpu_state_copy(state: Any) -> Any:
    """Deep-copy a (possibly nested) state container with tensors moved to CPU.

    Snapshots the live training state so serialization can proceed off the
    critical path while the optimizer keeps mutating the originals.
    """
    if isinstance(state, torch.Tensor):
        return state.detach().to("cpu", copy=True)
    if isinstance(state, dict):
        return {name: cpu_state_copy(value) for name, value in state.items()}
    if isinstance(state, list | tuple):
        return type(state)(cpu_state_copy(value) for value in state)
    return state


def training_rng_states() -> dict[str, Any]:
    """Capture every process-global RNG stream a checkpoint must restore."""
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def checkpoint_payload(
    *,
    agents: Sequence[Mapping[str, Any]],
    model_config: AnyModelConfig,
    ppo_config: PpoConfig,
    iteration: int,
    next_seed: int,
    metrics: dict[str, Any],
    source_identity: dict[str, Any],
    rng_states: dict[str, Any],
    run_provenance: dict[str, Any] | None = None,
    training_rng_state: Mapping[str, Any] | None = None,
    training_data_config: dict[str, Any] | None = None,
    auxiliary_rng_state: Mapping[str, Any] | None = None,
    league_snapshot_manifest: dict[int, str] | None = None,
    league_score_rates: dict[str, float] | None = None,
    replay_parity_baseline: dict[str, float] | list[dict[str, float]] | None = None,
    population_disagreement_reference: float | None = None,
    policy_entropy_reference: float | list[float | None] | None = None,
    initial_actor: dict[str, Any] | None = None,
    structured_gate_state: Mapping[str, Any] | None = None,
    seed_usage: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble a validated checkpoint payload from already-captured state."""
    normalized_source_identity = validate_source_identity(source_identity)
    normalized_run_provenance = validate_run_provenance(run_provenance)
    if (
        normalized_run_provenance is not None
        and normalized_run_provenance["source_identity"] != normalized_source_identity
    ):
        raise ValueError("checkpoint run provenance source does not match source identity")
    if normalized_run_provenance is not None and (
        not isinstance(training_data_config, dict)
        or any(
            training_data_config.get(knob) != normalized_run_provenance["calibration"][knob]
            for knob in CALIBRATION_KNOBS
        )
    ):
        raise ValueError("checkpoint compile mode does not match run provenance")
    # `training_data_config` is a record; `ppo_config.update_compile_mode` is what
    # actually drives compilation in the update (see ppo.py). They arrive here
    # as independent parameters, so nothing but this makes them agree, and a
    # checkpoint whose record names one mode while its config names another would
    # carry provenance for a run that did not happen. Compared by value rather
    # than identity now that the knob is a string: `is not` on two equal strings
    # is true whenever they are not the same interned object, which for a mode
    # read back out of JSON is exactly the case. `rollout_forward_mode` needs no
    # equivalent: the record is the only place it is stored.
    if (
        isinstance(training_data_config, dict)
        and UPDATE_COMPILE_MODE_KNOB in training_data_config
        and training_data_config[UPDATE_COMPILE_MODE_KNOB] != ppo_config.update_compile_mode
    ):
        raise ValueError(
            "checkpoint training data config update_compile_mode does not match its ppo config"
        )
    if set(rng_states) != {"torch_rng", "cuda_rng", "numpy_rng", "python_rng"}:
        raise ValueError("checkpoint RNG capture is incomplete")
    if not agents or any(not _complete_agent_state(agent) for agent in agents):
        raise ValueError(f"every checkpointed agent needs {AGENT_STATE_KEYS}")
    for predictor_key, _ in PREDICTOR_STATE_KEY_PAIRS:
        presence = [predictor_key in agent for agent in agents]
        if any(presence) != all(presence):
            label = predictor_key.replace("_", " ")
            raise ValueError(f"checkpoint population cannot mix {label} presence")
    auxiliary_members = [
        any(predictor_key in agent for predictor_key, _ in PREDICTOR_STATE_KEY_PAIRS)
        for agent in agents
    ]
    if any(auxiliary_members) != (auxiliary_rng_state is not None):
        raise ValueError(
            "structured predictors, optimizers, and auxiliary RNG must be checkpointed together"
        )
    if any(auxiliary_members) != (structured_gate_state is not None):
        raise ValueError(
            "structured predictors and predictor gate state must be checkpointed together"
        )
    # Evaluation and submission play the real board. Training cycles
    # symmetries per game, so a member has no private frame to record.
    # Identity is the code inference applies when it is asked to play.
    oriented = []
    for agent in agents:
        entry = dict(agent)
        entry.setdefault("orientation", int(Orientation.IDENTITY))
        oriented.append(entry)

    if len(oriented) == 1:
        # A single learner's payload is byte-for-byte what it has always been:
        # the four states at the top level and no population list, so every
        # reader of "the actor" keeps working without being told which member.
        members: dict[str, Any] = dict(oriented[0])
    else:
        # A population payload deliberately carries no top-level actor. See
        # POPULATION_CHECKPOINT_KEY: a member-zero alias there is how a reader
        # ends up reporting one arbitrary member as the whole run's strength.
        members = {POPULATION_CHECKPOINT_KEY: oriented}
    auxiliary_recovery = (
        {} if auxiliary_rng_state is None else {"structured_auxiliary_rng": auxiliary_rng_state}
    )
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "iteration": iteration,
        "next_seed": next_seed,
        "seed_usage": [dict(row) for row in seed_usage],
        "architecture": architecture_of_config(model_config).name,
        "model_config": model_config.to_dict(),
        "ppo_config": asdict(ppo_config),
        **members,
        "metrics": metrics,
        **rng_states,
        "training_rng": training_rng_state,
        **auxiliary_recovery,
        "structured_gate_state": (
            None if structured_gate_state is None else dict(structured_gate_state)
        ),
        "training_data_config": training_data_config,
        "league_snapshot_manifest": league_snapshot_manifest,
        # PFSP opponent estimates are part of the training state: without
        # them a resume replays retired opponents and perturbs the RNG
        # stream that opponent selection consumes.
        "league_score_rates": league_score_rates,
        # The most recent sampling-vs-update parity measurement, per audited
        # head. Persisted because the training gate decides defect versus
        # drift by comparing an audit against the previous one, and a run
        # restarted under --max-hours would otherwise judge its first audit
        # with no history and abort a merely drifted run.
        "replay_parity_baseline": replay_parity_baseline,
        # The population's pairwise disagreement at iteration 0, which the
        # convergence gate takes a fixed share of. Persisted for the same reason
        # as the baseline above and one more: re-measuring it after a resume
        # would recalibrate the floor against however far the members had
        # already converged, which is the state the gate exists to refuse.
        "population_disagreement_reference": population_disagreement_reference,
        # Each member's entropy at the first iteration its actor stepped, which
        # the entropy floor takes a share of when that is lower than the absolute
        # level. Persisted rather than re-measured for the same reason: a resume
        # would recalibrate the floor against however far the policy had already
        # sharpened, and a chunked run under --max-hours would ratchet its own
        # floor down every restart until the gate admitted a collapsed policy.
        "policy_entropy_reference": policy_entropy_reference,
        # Warm-start provenance travels with the run: opponent selection
        # keeps the iteration-0 league snapshot eligible only when it is a
        # pretrained baseline, and a resume must preserve that decision.
        "initial_actor": initial_actor,
        "source_identity": normalized_source_identity,
        "run_provenance": normalized_run_provenance,
    }


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically serialize one checkpoint payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically serialize a checkpoint without replacing an existing event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_checkpoint_alias(source: Path, alias: Path) -> bool:
    """Atomically point a regular-file alias at an immutable checkpoint.

    The temporary hard link lives beside the alias, so replacing ``latest.pt``
    never exposes a missing or partial file and never serializes the payload a
    second time. ``False`` means the alias already named the same inode.
    """
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"checkpoint alias source is not a regular file: {source}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() and not alias.is_symlink() and os.path.samefile(source, alias):
        return False
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{alias.name}.", suffix=".tmp", dir=alias.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.link(source, temporary)
        os.replace(temporary, alias)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def install_immutable_checkpoint(source: Path, destination: Path) -> bool:
    """Install an existing checkpoint once, preferring a byte-free hard link."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"checkpoint source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            not destination.is_symlink()
            and destination.is_file()
            and os.path.samefile(source, destination)
        ):
            return False
        raise FileExistsError(f"immutable checkpoint already exists: {destination}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, temporary)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def save_checkpoint(
    path: Path,
    *,
    agents: Sequence[TrainingAgent],
    model_config: AnyModelConfig,
    ppo_config: PpoConfig,
    iteration: int,
    next_seed: int,
    metrics: dict[str, Any],
    source_identity: dict[str, Any],
    run_provenance: dict[str, Any] | None = None,
    training_rng_state: Mapping[str, Any] | None = None,
    training_data_config: dict[str, Any] | None = None,
    auxiliary_rng_state: Mapping[str, Any] | None = None,
    league_snapshot_manifest: dict[int, str] | None = None,
    league_score_rates: dict[str, float] | None = None,
    replay_parity_baseline: dict[str, float] | list[dict[str, float]] | None = None,
    population_disagreement_reference: float | None = None,
    policy_entropy_reference: float | list[float | None] | None = None,
    initial_actor: dict[str, Any] | None = None,
    structured_gate_state: Mapping[str, Any] | None = None,
    seed_usage: Sequence[Mapping[str, Any]] = (),
) -> None:
    payload = checkpoint_payload(
        agents=[agent.state() for agent in agents],
        model_config=model_config,
        ppo_config=ppo_config,
        iteration=iteration,
        next_seed=next_seed,
        metrics=metrics,
        source_identity=source_identity,
        rng_states=training_rng_states(),
        run_provenance=run_provenance,
        training_rng_state=training_rng_state,
        training_data_config=training_data_config,
        auxiliary_rng_state=auxiliary_rng_state,
        league_snapshot_manifest=league_snapshot_manifest,
        league_score_rates=league_score_rates,
        replay_parity_baseline=replay_parity_baseline,
        population_disagreement_reference=population_disagreement_reference,
        policy_entropy_reference=policy_entropy_reference,
        initial_actor=initial_actor,
        structured_gate_state=structured_gate_state,
        seed_usage=seed_usage,
    )
    write_checkpoint(path, payload)


def load_checkpoint(
    path: Path,
    agents: Sequence[TrainingAgent],
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    require_checkpoint_format(payload)
    for agent in agents:
        for predictor_key, optimizer_key in PREDICTOR_STATE_KEY_PAIRS:
            predictor = getattr(agent, predictor_key)
            optimizer = getattr(agent, optimizer_key)
            if (predictor is None) != (optimizer is None):
                label = predictor_key.replace("_", " ")
                raise ValueError(
                    f"constructed learner {label} and optimizer must be provided together"
                )
    # Validate the model identity before mutating anything: a mismatched
    # checkpoint must fail with these messages, not with a strict-loading
    # key dump halfway through restoring the actor.
    if resolve_architecture(payload).name != architecture_of(agents[0].actor).name:
        raise ValueError("checkpoint architecture does not match the constructed models")
    if payload["model_config"] != agents[0].actor.config.to_dict():
        raise ValueError("checkpoint model configuration does not match the constructed models")
    expects_auxiliary_rng = any(
        agent.structured_dynamics is not None or agent.structured_critic_dynamics is not None
        for agent in agents
    )
    if ("structured_auxiliary_rng" in payload) != expects_auxiliary_rng:
        raise ValueError(
            "checkpoint structured auxiliary RNG presence does not match the constructed learner"
        )
    states = checkpoint_agent_states(payload)
    # The population size is part of the run's identity, not something to pad or
    # truncate: restoring three members into four would leave the fourth training
    # from its fresh initialization while the run reported itself as resumed.
    if len(states) != len(agents):
        raise ValueError(
            f"checkpoint carries {len(states)} agents; this run configures {len(agents)}"
        )
    for agent, state in zip(agents, states, strict=True):
        for predictor_key, _optimizer_key in PREDICTOR_STATE_KEY_PAIRS:
            has_predictor = predictor_key in state
            expects_predictor = getattr(agent, predictor_key) is not None
            if has_predictor != expects_predictor:
                label = predictor_key.replace("_", " ")
                raise ValueError(
                    f"checkpoint {label} optimizer presence does not match the constructed learner"
                )
    for agent, state in zip(agents, states, strict=True):
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
        for predictor_key, optimizer_key in PREDICTOR_STATE_KEY_PAIRS:
            predictor = getattr(agent, predictor_key)
            optimizer = getattr(agent, optimizer_key)
            if predictor is not None:
                assert optimizer is not None
                predictor.load_state_dict(state[predictor_key])
                optimizer.load_state_dict(state[optimizer_key])
        if agent.actor_optimizer is not None:
            agent.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if agent.critic_optimizer is not None:
            agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        cuda_rng = payload["cuda_rng"]
        if len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state count does not match visible CUDA devices")
        # `map_location` above moved every tensor in the payload onto the training
        # device, and a generator state is only accepted as a CPU ByteTensor -- the
        # same reason the CPU generator above is restored through `.cpu()`.
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return payload


_JOURNAL_TAIL_WINDOW = 1 << 20


def _journal_last_line(stream: Any, path: Path) -> str | None:
    """Truncate a torn final suffix and return the last complete record."""
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if not size:
        return None
    window = min(size, _JOURNAL_TAIL_WINDOW)
    stream.seek(size - window)
    tail = stream.read(window)
    if not tail.endswith(b"\n"):
        # A torn suffix can only follow a crash mid-append. Recover the last
        # complete line while retaining strict validation of that record.
        cut = tail.rfind(b"\n")
        if cut < 0 and window < size:
            raise ValueError(f"metrics journal has an oversized torn record: {path}")
        size = size - (len(tail) - cut - 1) if cut >= 0 else 0
        stream.truncate(size)
        if not size:
            return None
        window = min(size, _JOURNAL_TAIL_WINDOW)
        stream.seek(size - window)
        tail = stream.read(window)
    body = tail[:-1]
    cut = body.rfind(b"\n")
    if cut < 0 and window < size:
        raise ValueError(f"metrics journal has an oversized record: {path}")
    return body[cut + 1 :].decode("utf-8")


def append_iteration_jsonl(path: Path, payload: dict[str, Any]) -> bool:
    """Append one canonical iteration record idempotently in constant time.

    Checkpoints commit before telemetry. On recovery this fills a missing final
    record without duplicating one that was already durably appended. Readers
    recover a torn final suffix, so a plain fsynced append preserves the
    journal's crash-safety contract without rewriting the complete file.
    """
    iteration = payload.get("iteration")
    if type(iteration) is not int or iteration < 1:
        raise ValueError("iteration metrics require a positive integer iteration")
    rendered = json.dumps(payload, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(descriptor, "r+b") as stream:
        last_line = _journal_last_line(stream, path)
        if last_line is not None:
            try:
                previous = json.loads(last_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"metrics journal has an invalid final record: {path}") from error
            previous_iteration = previous.get("iteration") if isinstance(previous, dict) else None
            if type(previous_iteration) is not int:
                raise ValueError(f"metrics journal has an invalid final record: {path}")
            if previous_iteration > iteration:
                raise ValueError(
                    f"metrics journal is ahead of checkpoint iteration {iteration}: {path}"
                )
            if previous_iteration == iteration:
                if last_line != rendered:
                    raise ValueError(
                        f"metrics journal conflicts with checkpoint iteration {iteration}: {path}"
                    )
                return False
            if previous_iteration + 1 != iteration:
                raise ValueError(
                    f"metrics journal is missing iterations before {iteration}: {path}"
                )
        stream.seek(0, os.SEEK_END)
        stream.write(rendered.encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return True


def metrics_journal_iteration(path: Path) -> int:
    """Return the last iteration of a contiguous, possibly portable journal suffix."""
    path = Path(path)
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text.rpartition("\n")[0]
    last_iteration: int | None = None
    for line in (line for line in text.splitlines() if line):
        payload = json.loads(line)
        iteration = payload.get("iteration")
        if (
            type(iteration) is not int
            or iteration < 1
            or (last_iteration is not None and iteration != last_iteration + 1)
        ):
            raise ValueError(f"metrics journal iterations are not contiguous: {path}")
        last_iteration = iteration
    return 0 if last_iteration is None else last_iteration
