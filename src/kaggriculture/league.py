"""Immutable actor snapshots and reproducible league selection over snapshots and built-ins."""

from __future__ import annotations

import errno
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.opponents import BUILTIN_OPPONENTS
from kaggriculture.registry import (
    Architecture,
    architecture_of_config,
    resolve_architecture,
)
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic

AnyActor = FarmActor | StructuredActor
AnyCritic = DistributionalCritic | StructuredCritic
AnyModelConfig = ModelConfig | StructuredConfig

LEAGUE_SNAPSHOT_FORMAT_VERSION = 2
_SNAPSHOT_NAME = re.compile(r"league-actor-(\d{8})\.pt")
_MAX_CANONICAL_ITERATION = 99_999_999
# Identity of a league opponent in the PFSP score-rate state and in the journal.
# Snapshots key on their zero-padded iteration; built-ins on their engine name.
# One keyspace, because one weighting decides which of them plays.
LEAGUE_OPPONENT_KEY = re.compile(r"\d{8}|builtin_[a-z]+")


@dataclass(frozen=True, order=True)
class SnapshotRef:
    iteration: int
    path: Path


@dataclass(frozen=True, order=True)
class BuiltinRef:
    """A built-in reference agent, played natively inside the batched wave."""

    name: str


def _opponent_key(ref: SnapshotRef | BuiltinRef) -> str:
    return f"builtin_{ref.name}" if isinstance(ref, BuiltinRef) else f"{ref.iteration:08d}"


@dataclass(frozen=True)
class SnapshotSelection:
    ref: SnapshotRef
    category: Literal["active", "historical"]

    @property
    def key(self) -> str:
        return _opponent_key(self.ref)

    @property
    def label(self) -> str:
        return self.ref.path.name


@dataclass(frozen=True)
class BuiltinSelection:
    ref: BuiltinRef
    category: Literal["builtin"] = "builtin"

    @property
    def key(self) -> str:
        return _opponent_key(self.ref)

    @property
    def label(self) -> str:
        return self.ref.name


LeagueSelection = SnapshotSelection | BuiltinSelection


def _snapshot_path(directory: Path, iteration: int) -> Path:
    if not 0 <= iteration <= _MAX_CANONICAL_ITERATION:
        raise ValueError(
            f"snapshot iteration must be in [0, {_MAX_CANONICAL_ITERATION}], got {iteration}"
        )
    return directory / f"league-actor-{iteration:08d}.pt"


def _model_config_dict(config: AnyModelConfig | dict[str, Any]) -> dict[str, Any]:
    return dict(config) if isinstance(config, dict) else config.to_dict()


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"league snapshot is not a dictionary: {path}")
    # Snapshots written before the architecture registry carry no tag and are
    # all convolutional entity transformers, matching the registry's default.
    required_keys = {"format_version", "iteration", "model_config", "actor"}
    if not required_keys <= set(payload) or set(payload) - required_keys - {"architecture"}:
        missing = sorted(required_keys - set(payload))
        unexpected = sorted(set(payload) - required_keys - {"architecture"})
        raise ValueError(
            f"invalid league snapshot schema at {path}: missing={missing}, unexpected={unexpected}"
        )
    if "architecture" in payload and type(payload["architecture"]) is not str:
        raise ValueError(f"league snapshot has an invalid architecture tag: {path}")
    if (
        type(payload["format_version"]) is not int
        or payload["format_version"] != LEAGUE_SNAPSHOT_FORMAT_VERSION
    ):
        raise ValueError(
            f"unsupported league snapshot format at {path}: {payload['format_version']}; "
            f"expected {LEAGUE_SNAPSHOT_FORMAT_VERSION}"
        )
    iteration = payload["iteration"]
    if type(iteration) is not int or not 0 <= iteration <= _MAX_CANONICAL_ITERATION:
        raise ValueError(f"league snapshot has invalid iteration: {path}")
    if not isinstance(payload["model_config"], dict) or not isinstance(payload["actor"], dict):
        raise ValueError(f"league snapshot has invalid model metadata or actor state: {path}")
    try:
        architecture = resolve_architecture(payload)
    except ValueError as error:
        raise ValueError(f"league snapshot has an unknown architecture: {path}") from error
    expected_config = architecture.config_class().to_dict()
    if set(payload["model_config"]) != set(expected_config) or any(
        type(payload["model_config"][name]) is not type(default)
        for name, default in expected_config.items()
    ):
        raise ValueError(f"league snapshot has invalid model configuration schema: {path}")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload["actor"].items()
    ):
        raise ValueError(f"league snapshot has invalid actor state schema: {path}")
    return payload


def _validate_canonical_filename(path: Path, iteration: int) -> None:
    match = _SNAPSHOT_NAME.fullmatch(path.name)
    if match is None or int(match.group(1)) != iteration:
        raise ValueError(f"league snapshot filename/iteration mismatch: {path}")


def _matches_actor_state(
    payload: dict[str, Any],
    architecture: Architecture,
    config: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> bool:
    return (
        resolve_architecture(payload).name == architecture.name
        and payload["model_config"] == config
        and payload["actor"].keys() == state.keys()
        and all(torch.equal(payload["actor"][name], value) for name, value in state.items())
    )


def save_actor_snapshot(directory: Path, actor: AnyActor, iteration: int) -> SnapshotRef:
    """Atomically save one immutable CPU actor snapshot.

    Repeating the exact same save is idempotent. Reusing an iteration for
    different weights fails instead of silently changing the frozen league.
    """
    state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    return save_actor_state_snapshot(directory, actor.config, state, iteration)


def save_actor_state_snapshot(
    directory: Path,
    model_config: AnyModelConfig,
    state: dict[str, torch.Tensor],
    iteration: int,
) -> SnapshotRef:
    """Atomically save one immutable snapshot from an already-captured CPU state."""
    directory = Path(directory)
    path = _snapshot_path(directory, iteration)
    state = {name: value.detach().cpu() for name, value in state.items()}
    architecture = architecture_of_config(model_config)
    config = _model_config_dict(model_config)
    if path.exists():
        existing = _load_payload(path)
        _validate_canonical_filename(path, existing["iteration"])
        if not _matches_actor_state(existing, architecture, config, state):
            raise FileExistsError(f"refusing to replace immutable league snapshot: {path}")
        return SnapshotRef(iteration, path)

    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": LEAGUE_SNAPSHOT_FORMAT_VERSION,
        "iteration": iteration,
        "architecture": architecture.name,
        "model_config": config,
        "actor": state,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        try:
            # Same-directory hard-link installation is atomic and cannot
            # replace an existing immutable snapshot. The temporary file is
            # already complete before the canonical name becomes visible.
            os.link(temporary, path)
        except FileExistsError:
            existing = _load_payload(path)
            _validate_canonical_filename(path, existing["iteration"])
            if not _matches_actor_state(existing, architecture, config, state):
                raise FileExistsError(
                    f"refusing to replace immutable league snapshot: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return SnapshotRef(iteration, path)


def _validated_snapshot_state(
    path: Path,
    expected_model_config: AnyModelConfig | dict[str, Any] | None,
) -> tuple[Architecture, AnyModelConfig, dict[str, torch.Tensor]]:
    """Validate one snapshot file and return its family, configuration, and state."""
    path = Path(path)
    payload = _load_payload(path)
    _validate_canonical_filename(path, payload["iteration"])
    architecture = resolve_architecture(payload)
    if expected_model_config is not None:
        if not isinstance(expected_model_config, dict) and (
            architecture_of_config(expected_model_config).name != architecture.name
        ):
            raise ValueError(f"league snapshot architecture mismatch: {path}")
        expected = _model_config_dict(expected_model_config)
        if payload["model_config"] != expected:
            raise ValueError(f"league snapshot model configuration mismatch: {path}")
    try:
        config = architecture.config_class(**payload["model_config"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid league snapshot model configuration: {path}") from error
    return architecture, config, payload["actor"]


def load_actor_snapshot(
    path: Path,
    *,
    expected_model_config: AnyModelConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> AnyActor:
    """Validate and strictly load a frozen actor snapshot."""
    path = Path(path)
    architecture, config, state = _validated_snapshot_state(path, expected_model_config)
    # Parameter initialization is discarded immediately by strict loading, so
    # frozen-policy I/O must not perturb training's checkpointed RNG stream.
    with torch.random.fork_rng(devices=[]):
        actor = architecture.actor_class(config).to(device)
    try:
        actor.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"invalid league snapshot actor state: {path}") from error
    actor.eval().requires_grad_(False)
    return actor


class FrozenActorPool:
    """Persistent frozen-actor slots that load snapshot weights in place.

    Rollout compilation caches its wrapper per module instance, so constructing
    a fresh ``FarmActor`` for every selected opponent forces a Dynamo retrace
    and CUDA graph recapture every iteration. Slot ``i`` always serves the
    ``i``-th selection of an iteration; reloading weights into the same module
    keeps every captured graph valid because parameter storages are reused.
    """

    def __init__(self, model_config: AnyModelConfig, device: torch.device | str) -> None:
        self._architecture = architecture_of_config(model_config)
        self._model_config = model_config
        self._device = device
        self._slots: list[AnyActor] = []
        self._loaded: list[Path | None] = []

    def acquire(self, snapshot_paths: Sequence[Path]) -> list[AnyActor]:
        """Return one validated frozen actor per snapshot path, reusing slots."""
        while len(self._slots) < len(snapshot_paths):
            with torch.random.fork_rng(devices=[]):
                slot = self._architecture.actor_class(self._model_config).to(self._device)
            slot.eval().requires_grad_(False)
            self._slots.append(slot)
            self._loaded.append(None)
        for index, path in enumerate(snapshot_paths):
            resolved = Path(path).resolve()
            if self._loaded[index] == resolved:
                continue
            _, _, state = _validated_snapshot_state(resolved, self._model_config)
            self._loaded[index] = None
            try:
                self._slots[index].load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise ValueError(f"invalid league snapshot actor state: {resolved}") from error
            self._loaded[index] = resolved
        return self._slots[: len(snapshot_paths)]


def snapshot_sha256(path: Path) -> str:
    """Return the content digest used to bind a checkpoint to its league archive."""
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def copy_actor_snapshot(
    source: Path,
    directory: Path,
    *,
    expected_model_config: AnyModelConfig | dict[str, Any],
) -> SnapshotRef:
    """Validate and atomically install an immutable snapshot into another archive."""
    source = Path(source)
    payload = _load_payload(source)
    iteration = payload["iteration"]
    _validate_canonical_filename(source, iteration)
    source_digest = snapshot_sha256(source)
    # Strict state loading catches malformed tensor names or shapes before the
    # copied snapshot can become visible in the destination archive.
    load_actor_snapshot(source, expected_model_config=expected_model_config)

    directory = Path(directory)
    destination = _snapshot_path(directory, iteration)
    if destination.exists():
        load_actor_snapshot(destination, expected_model_config=expected_model_config)
        if snapshot_sha256(destination) != source_digest:
            raise FileExistsError(f"refusing to replace immutable league snapshot: {destination}")
        return SnapshotRef(iteration, destination)

    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        load_actor_snapshot(destination, expected_model_config=expected_model_config)
        if snapshot_sha256(destination) != source_digest:
            raise FileExistsError(f"conflicting concurrent league snapshot: {destination}")
        return SnapshotRef(iteration, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    else:
        return SnapshotRef(iteration, destination)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        with suppress(FileExistsError):
            os.link(temporary, destination)
        # A concurrent restore may have won the immutable-name race. Validation
        # and the caller's digest check determine whether it installed the
        # exact same archive member.
    except BaseException:
        # os.fdopen owns the descriptor once entered. If opening the source
        # fails first, close the still-live descriptor explicitly.
        with suppress(OSError):
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)

    load_actor_snapshot(destination, expected_model_config=expected_model_config)
    if snapshot_sha256(destination) != source_digest:
        raise FileExistsError(f"conflicting concurrent league snapshot: {destination}")
    return SnapshotRef(iteration, destination)


def list_actor_snapshots(directory: Path) -> list[SnapshotRef]:
    """List canonical snapshot filenames in numeric iteration order."""
    directory = Path(directory)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    refs = []
    for path in directory.iterdir():
        match = _SNAPSHOT_NAME.fullmatch(path.name)
        if match is not None and path.is_file():
            refs.append(SnapshotRef(int(match.group(1)), path))
    return sorted(refs)


# Prioritized fictitious self-play weighting: an opponent's sampling weight is
# (1 - score_rate)^2, so competitive opponents dominate and a fully beaten one
# (score rate 1.0) retires from sampling entirely. Unmeasured opponents count
# as even (0.5) so new snapshots and built-ins enter the rotation at moderate
# priority.
PFSP_UNMEASURED_SCORE_RATE = 0.5


def _pfsp_weights(
    values: Sequence[SnapshotRef | BuiltinRef],
    score_rates: Mapping[str, float] | None,
) -> np.ndarray:
    rates = np.asarray(
        [
            (
                PFSP_UNMEASURED_SCORE_RATE
                if score_rates is None
                else float(score_rates.get(_opponent_key(ref), PFSP_UNMEASURED_SCORE_RATE))
            )
            for ref in values
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(rates)) or np.any(rates < 0.0) or np.any(rates > 1.0):
        raise ValueError("opponent score rates must be finite and within [0, 1]")
    return np.square(1.0 - rates)


def _weighted_sample_without_replacement(
    values: Sequence[SnapshotRef | BuiltinRef],
    count: int,
    generator: np.random.Generator,
    weights: np.ndarray,
) -> list[SnapshotRef | BuiltinRef]:
    if count <= 0 or not values:
        return []
    total = float(weights.sum())
    if total <= 0.0:
        # Every candidate is fully beaten; nothing here is worth games.
        return []
    size = min(count, int(np.count_nonzero(weights)))
    indices = np.atleast_1d(
        generator.choice(len(values), size=size, replace=False, p=weights / total)
    )
    return [values[int(index)] for index in indices]


def _contest_builtin_lanes(
    builtins: Sequence[BuiltinRef],
    budget: int,
    snapshot_weight: float,
    generator: np.random.Generator,
    weights: np.ndarray,
) -> list[BuiltinRef]:
    """Fill the reserved built-in lanes, contesting each against a snapshot.

    Every reserved lane is decided between the built-ins that have not taken
    one yet and the alternative of one more frozen snapshot, all on the same
    PFSP scale. This is where a built-in retires: a fixed lane per admitted
    agent would leave the weight nothing to allocate, whereas here a beaten
    built-in loses its lane to a still-competitive one, and once none of them
    is worth games the whole budget goes to snapshots. Lanes the built-ins do
    not win are returned to the caller as the shortfall.
    """
    remaining = list(builtins)
    remaining_weights = [float(weight) for weight in weights]
    chosen: list[BuiltinRef] = []
    for _ in range(budget):
        total = sum(remaining_weights) + snapshot_weight
        if not remaining or total <= 0.0:
            break
        probabilities = np.asarray([*remaining_weights, snapshot_weight], dtype=np.float64)
        index = int(generator.choice(len(probabilities), p=probabilities / total))
        if index == len(remaining):
            # The snapshot alternative took this lane; the rest are contested
            # on their own, so one loss does not close the stratum.
            continue
        chosen.append(remaining.pop(index))
        remaining_weights.pop(index)
    return chosen


def _sample_log_age_strata(
    values: Sequence[SnapshotRef],
    count: int,
    current_iteration: int,
    generator: np.random.Generator,
    weights: np.ndarray,
) -> list[SnapshotRef]:
    """Sample across log2 age buckets, PFSP-weighted at both levels.

    Buckets are drawn without replacement in proportion to their total PFSP
    weight, then one member is drawn within the chosen bucket by weight.
    Weighting the bucket draw keeps age diversity while denying a full
    historical slot to an age stratum whose only members are nearly beaten.
    """
    buckets: dict[int, list[tuple[SnapshotRef, float]]] = {}
    for ref, weight in zip(values, weights, strict=True):
        if weight <= 0.0:
            continue
        age = max(1, current_iteration - ref.iteration)
        buckets.setdefault(age.bit_length() - 1, []).append((ref, float(weight)))
    selected: list[SnapshotRef] = []
    remaining = list(buckets)
    while remaining and len(selected) < count:
        totals = np.asarray(
            [sum(weight for _, weight in buckets[bucket]) for bucket in remaining],
            dtype=np.float64,
        )
        drawn = remaining[int(generator.choice(len(remaining), p=totals / totals.sum()))]
        remaining.remove(drawn)
        candidates = buckets[drawn]
        bucket_weights = np.asarray([weight for _, weight in candidates], dtype=np.float64)
        index = int(generator.choice(len(candidates), p=bucket_weights / bucket_weights.sum()))
        selected.append(candidates.pop(index)[0])
        if not candidates:
            del buckets[drawn]
        if not remaining and len(selected) < count:
            remaining = [bucket for bucket in buckets if buckets[bucket]]
    return selected


def select_league_mix(
    refs: Sequence[SnapshotRef],
    *,
    current_iteration: int,
    active_count: int,
    historical_count: int,
    active_pool_size: int,
    generator: np.random.Generator,
    builtins: Sequence[str] = (),
    builtin_lanes: int = 0,
    score_rates: Mapping[str, float] | None = None,
    pretrained_start: bool = False,
) -> list[LeagueSelection]:
    """Select distinct recent-active, log-age historical, and built-in opponents.

    Active candidates are the newest ``active_pool_size`` frozen iterations.
    Historical candidates must be strictly older than that complete active
    window. The iteration-0 snapshot is excluded by default: games against a
    randomly initialized policy teach nothing a trained snapshot cannot. A
    ``pretrained_start`` run keeps it eligible — there iteration 0 is the
    warm-start baseline, and playing it holds anti-regression pressure
    against the learner's own starting point.
    Undersized pools return fewer selections without duplicating a policy.

    ``builtins`` names engine reference agents and ``builtin_lanes`` reserves
    that many lanes for them. Each reserved lane is contested between the
    admitted built-ins that have not taken one and one more frozen snapshot,
    decided by the same PFSP weight, so the budget is a ceiling rather than a
    floor: while a built-in is unbeaten it outweighs the snapshot alternative
    and holds its lane, and as the learner beats them the reserved lanes drain
    back into the active stratum with no threshold anywhere. Reserving lanes
    rather than letting built-ins contest the active slots matters because the
    active window holds sixteen candidates — inside it an unbeaten built-in
    would win well under half a lane per wave, and the learner has to actually
    learn a farming loop against these agents, not be exposed to one
    occasionally. Built-in lanes need no snapshot pool, so a run with nothing
    frozen yet still plays them from its first iteration. The total lane count
    stays ``active_count + historical_count + builtin_lanes`` however the
    contest goes, which is what keeps the wave's stacked frozen forward on one
    captured shape.

    ``score_rates`` maps opponent key to the learner's recent score rate
    against that opponent; sampling is prioritized fictitious self-play with
    weight (1 - score_rate)^2, so fully beaten opponents retire and their
    games return to competitive opponents instead of 100%-win blowouts.

    Selections list every active snapshot (sorted by iteration), then every
    historical one (also sorted), then every built-in (sorted by name). The
    wave numbers its frozen-module lanes before its built-in lanes, so that
    ordering is part of the contract. The active/historical boundary decides
    only which snapshots are eligible for a lane, not how they decode: every
    seat in a wave samples at the learner's own temperature.
    """
    if current_iteration < 0:
        raise ValueError("current iteration cannot be negative")
    if active_count < 0 or historical_count < 0:
        raise ValueError("snapshot selection counts cannot be negative")
    if active_pool_size < 1:
        raise ValueError("active pool size must be positive")
    if builtin_lanes < 0:
        raise ValueError("built-in lane budget cannot be negative")
    unknown = sorted(set(builtins) - BUILTIN_OPPONENTS)
    if unknown:
        raise ValueError(f"unknown built-in league opponents: {', '.join(unknown)}")
    if len(set(builtins)) != len(builtins):
        raise ValueError("built-in league opponents must be distinct")
    eligible_by_iteration: dict[int, SnapshotRef] = {}
    iteration_by_path: dict[Path, int] = {}
    for ref in refs:
        if type(ref.iteration) is not int or ref.iteration < 0:
            raise ValueError(f"invalid snapshot iteration: {ref.iteration!r}")
        if ref.iteration >= current_iteration:
            continue
        normalized_path = ref.path.resolve()
        previous = eligible_by_iteration.get(ref.iteration)
        if previous is not None and previous.path.resolve() != normalized_path:
            raise ValueError(f"conflicting paths for snapshot iteration {ref.iteration}")
        previous_iteration = iteration_by_path.get(normalized_path)
        if previous_iteration is not None and previous_iteration != ref.iteration:
            raise ValueError(
                f"snapshot path is reused for iterations {previous_iteration} and {ref.iteration}"
            )
        eligible_by_iteration[ref.iteration] = ref
        iteration_by_path[normalized_path] = ref.iteration
    eligible = sorted(eligible_by_iteration.values())

    trained = eligible if pretrained_start else [ref for ref in eligible if ref.iteration != 0]
    active_window = trained[-active_pool_size:]
    active_weights = _pfsp_weights(active_window, score_rates)
    builtin_refs = [BuiltinRef(name) for name in builtins]
    drawn_builtins = _contest_builtin_lanes(
        builtin_refs,
        min(builtin_lanes, len(builtin_refs)),
        # What one more snapshot lane is worth, on the same scale, so the
        # contest compares like with like instead of against the whole window.
        float(active_weights.mean()) if active_weights.size else 0.0,
        generator,
        _pfsp_weights(builtin_refs, score_rates),
    )
    released = min(builtin_lanes, len(builtin_refs)) - len(drawn_builtins)
    active = _weighted_sample_without_replacement(
        active_window,
        active_count + released,
        generator,
        active_weights,
    )
    active_iterations = {ref.iteration for ref in active_window}
    historical_candidates = [ref for ref in trained if ref.iteration not in active_iterations]
    historical = _sample_log_age_strata(
        historical_candidates,
        min(historical_count, len(historical_candidates)),
        current_iteration,
        generator,
        _pfsp_weights(historical_candidates, score_rates),
    )

    selections: list[LeagueSelection] = []
    selections.extend(SnapshotSelection(ref, "active") for ref in sorted(active))
    selections.extend(SnapshotSelection(ref, "historical") for ref in sorted(historical))
    selections.extend(BuiltinSelection(ref) for ref in sorted(drawn_builtins))
    return selections
