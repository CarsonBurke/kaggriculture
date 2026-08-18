"""Checkpointing and diagnostics for long-running PPO jobs."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.constants import QUANTITY_BINS
from kaggriculture.inference import CHECKPOINT_FORMAT_VERSION
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
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

AnyActor = FarmActor | StructuredActor
AnyCritic = DistributionalCritic | StructuredCritic
AnyModelConfig = ModelConfig | StructuredConfig


def require_checkpoint_format(payload: dict[str, Any]) -> None:
    """Reject checkpoints from incompatible model and action schemas.

    Resume demands the current format exactly — a stale checkpoint must fail
    with a format error, not a downstream schema error. Actor-only export in
    inference.py separately accepts the legacy read-compatible versions.
    """
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
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
    actor_state: dict[str, Any],
    critic_state: dict[str, Any],
    actor_optimizer_state: dict[str, Any],
    critic_optimizer_state: dict[str, Any],
    model_config: AnyModelConfig,
    ppo_config: PpoConfig,
    iteration: int,
    next_seed: int,
    metrics: dict[str, Any],
    source_identity: dict[str, Any],
    rng_states: dict[str, Any],
    run_provenance: dict[str, Any] | None = None,
    training_rng_state: dict[str, Any] | None = None,
    training_data_config: dict[str, Any] | None = None,
    league_snapshot_manifest: dict[int, str] | None = None,
    league_score_rates: dict[int, float] | None = None,
    replay_parity_baseline: dict[str, float] | None = None,
    initial_actor: dict[str, Any] | None = None,
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
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "iteration": iteration,
        "next_seed": next_seed,
        "architecture": architecture_of_config(model_config).name,
        "model_config": model_config.to_dict(),
        "ppo_config": asdict(ppo_config),
        "actor": actor_state,
        "critic": critic_state,
        "actor_optimizer": actor_optimizer_state,
        "critic_optimizer": critic_optimizer_state,
        "metrics": metrics,
        **rng_states,
        "training_rng": training_rng_state,
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


def save_checkpoint(
    path: Path,
    *,
    actor: AnyActor,
    critic: AnyCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    model_config: AnyModelConfig,
    ppo_config: PpoConfig,
    iteration: int,
    next_seed: int,
    metrics: dict[str, Any],
    source_identity: dict[str, Any],
    run_provenance: dict[str, Any] | None = None,
    training_rng_state: dict[str, Any] | None = None,
    training_data_config: dict[str, Any] | None = None,
    league_snapshot_manifest: dict[int, str] | None = None,
    league_score_rates: dict[int, float] | None = None,
    replay_parity_baseline: dict[str, float] | None = None,
    initial_actor: dict[str, Any] | None = None,
) -> None:
    payload = checkpoint_payload(
        actor_state=actor.state_dict(),
        critic_state=critic.state_dict(),
        actor_optimizer_state=actor_optimizer.state_dict(),
        critic_optimizer_state=critic_optimizer.state_dict(),
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
        league_snapshot_manifest=league_snapshot_manifest,
        league_score_rates=league_score_rates,
        replay_parity_baseline=replay_parity_baseline,
        initial_actor=initial_actor,
    )
    write_checkpoint(path, payload)


def load_checkpoint(
    path: Path,
    actor: AnyActor,
    critic: AnyCritic,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    require_checkpoint_format(payload)
    # Validate the model identity before mutating anything: a mismatched
    # checkpoint must fail with these messages, not with a strict-loading
    # key dump halfway through restoring the actor.
    if resolve_architecture(payload).name != architecture_of(actor).name:
        raise ValueError("checkpoint architecture does not match the constructed models")
    if payload["model_config"] != actor.config.to_dict():
        raise ValueError("checkpoint model configuration does not match the constructed models")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    if actor_optimizer is not None:
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
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
