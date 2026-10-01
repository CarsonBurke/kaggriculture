"""Immutable actor snapshots and reproducible league selection over snapshots and built-ins."""

from __future__ import annotations

import errno
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import torch

from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.lejepa_model import LejepaCritic
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.modelargs import actor_model_config
from kaggriculture.opponents import BUILTIN_OPPONENTS
from kaggriculture.registry import (
    Architecture,
    architecture_of_config,
    resolve_architecture,
)
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic

AnyActor = FarmActor | StructuredActor | EntityActor
AnyCritic = DistributionalCritic | StructuredCritic | EntityCritic | LejepaCritic
AnyModelConfig = ModelConfig | StructuredConfig | EntityConfig

LEAGUE_SNAPSHOT_FORMAT_VERSION = 2
_SNAPSHOT_NAME = re.compile(r"league-actor-(\d{8})\.pt")
_MAX_CANONICAL_ITERATION = 99_999_999


@dataclass(frozen=True, order=True)
class SnapshotRef:
    iteration: int
    path: Path


@dataclass(frozen=True, order=True)
class BuiltinRef:
    """A built-in reference agent, played natively inside the batched wave."""

    name: str


def opponent_key(ref: SnapshotRef | BuiltinRef) -> str:
    return f"builtin_{ref.name}" if isinstance(ref, BuiltinRef) else f"{ref.iteration:08d}"


@dataclass(frozen=True)
class SnapshotSelection:
    ref: SnapshotRef
    category: Literal["active", "historical"]
    role: Literal["stratified", "hardness", "discovery", "probe", "screen"] = "stratified"

    @property
    def key(self) -> str:
        return opponent_key(self.ref)

    @property
    def label(self) -> str:
        return self.ref.path.name


@dataclass(frozen=True)
class BuiltinSelection:
    ref: BuiltinRef
    category: Literal["builtin"] = "builtin"
    role: Literal["stratified", "hardness", "discovery", "probe", "screen"] = "stratified"

    @property
    def key(self) -> str:
        return opponent_key(self.ref)

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
    if payload["model_config"].keys() != expected_config.keys() or any(
        type(value) is not type(expected_config[name])
        for name, value in payload["model_config"].items()
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


def _validated_snapshot_payload(
    path: Path,
    expected_model_config: AnyModelConfig | dict[str, Any] | None,
) -> tuple[dict[str, Any], Architecture, AnyModelConfig]:
    """Validate one snapshot file and return its payload, family, and configuration."""
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
        if actor_model_config(payload["model_config"]) != actor_model_config(expected):
            raise ValueError(f"league snapshot model configuration mismatch: {path}")
    try:
        config = architecture.build_config(payload["model_config"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid league snapshot model configuration: {path}") from error
    return payload, architecture, config


def load_actor_snapshot_payload(
    path: Path,
    *,
    expected_model_config: AnyModelConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a frozen actor snapshot without constructing its device-specific actor."""
    payload, _, _ = _validated_snapshot_payload(Path(path), expected_model_config)
    return payload


def _validated_snapshot_state(
    path: Path,
    expected_model_config: AnyModelConfig | dict[str, Any] | None,
) -> tuple[Architecture, AnyModelConfig, dict[str, torch.Tensor]]:
    """Validate one snapshot file and return its family, configuration, and state."""
    payload, architecture, config = _validated_snapshot_payload(path, expected_model_config)
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
    except FileExistsError as error:
        load_actor_snapshot(destination, expected_model_config=expected_model_config)
        if snapshot_sha256(destination) != source_digest:
            raise FileExistsError(
                f"conflicting concurrent league snapshot: {destination}"
            ) from error
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

# Discount old games, rather than giving a five-game wave the same weight as
# fifty games. A roughly 34-wave half-life follows a changing current learner.
MATCHUP_EVIDENCE_RETENTION = 0.98


class MatchupEvidence(TypedDict):
    score_sum: float
    games: float
    last_iteration: int


def matchup_estimate(record: MatchupEvidence | None) -> float:
    """The learner's expected score against an opponent, from its evidence.

    Beta(1, 1) over the evidence as it stood when last measured: few games
    shrink the estimate toward even odds, but staleness does not. An opponent
    beaten 90% of the time is still expected to be beaten until it is played
    again; how far to trust that is its uncertainty, which is what refreshing
    ranks by. Shrinking stale estimates to even odds instead would, in a long
    run, rank every long-unplayed easy snapshot above the genuinely hard ones.
    """
    if record is None:
        return 0.5
    return (1.0 + record["score_sum"]) / (2.0 + record["games"])


def validate_matchup_evidence(
    evidence: object, *, current_iteration: int
) -> dict[str, MatchupEvidence]:
    """Validate all selection state before consuming the restored RNG stream."""
    if not isinstance(evidence, dict):
        raise ValueError("resume checkpoint has no valid league matchup evidence")
    validated: dict[str, MatchupEvidence] = {}
    for key, value in evidence.items():
        valid_key = isinstance(key, str) and (
            re.fullmatch(r"[0-9]{8}", key) is not None
            or (key.removeprefix("builtin_") in BUILTIN_OPPONENTS and key.startswith("builtin_"))
            or re.fullmatch(r"script_[A-Za-z0-9_-]+", key) is not None
        )
        if (
            not valid_key
            or not isinstance(value, dict)
            or value.keys() != {"score_sum", "games", "last_iteration"}
        ):
            raise ValueError("resume checkpoint has invalid league matchup evidence")
        score, games, last = value["score_sum"], value["games"], value["last_iteration"]
        if (
            type(score) is not float
            or type(games) is not float
            or not math.isfinite(score)
            or not math.isfinite(games)
            or not 0.0 <= score <= games
            or games <= 0.0
            or type(last) is not int
            or not 0 <= last <= current_iteration
        ):
            raise ValueError("resume checkpoint has invalid league matchup evidence values")
        validated[key] = MatchupEvidence(score_sum=score, games=games, last_iteration=last)
    return validated


def update_matchup_evidence(
    evidence: dict[str, MatchupEvidence],
    measured: Mapping[str, tuple[int, float]],
    *,
    iteration: int,
) -> None:
    """Accumulate fractional wins (ties count half) from current-learner games."""
    for key, (games, rate) in measured.items():
        previous = evidence.get(key)
        if games <= 0 or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("league matchup measurements require games and a valid score rate")
        if previous is not None and iteration < previous["last_iteration"]:
            raise ValueError("league matchup measurements cannot move backward in time")
        discount = (
            MATCHUP_EVIDENCE_RETENTION ** (iteration - previous["last_iteration"])
            if previous is not None
            else 0.0
        )
        evidence[key] = MatchupEvidence(
            score_sum=float(games * rate + (previous["score_sum"] * discount if previous else 0)),
            games=float(games + (previous["games"] * discount if previous else 0)),
            last_iteration=iteration,
        )


def _hardness_selections(
    snapshots: Sequence[SnapshotRef],
    builtins: Sequence[BuiltinRef],
    *,
    count: int,
    current_iteration: int,
    active_pool_size: int,
    generator: np.random.Generator,
    evidence: Mapping[str, MatchupEvidence],
    screen_lanes: int = 0,
    screen_opponents: int = 0,
) -> list[LeagueSelection]:
    candidates = [*snapshots, *builtins]
    count = min(count, len(candidates))
    if count <= 0:
        return []
    score_estimates, probe_priorities = [], []
    for ref in candidates:
        record = evidence.get(opponent_key(ref))
        last = (
            record["last_iteration"]
            if record
            else (ref.iteration if isinstance(ref, SnapshotRef) else 0)
        )
        age = max(0, current_iteration - last)
        discount = MATCHUP_EVIDENCE_RETENTION**age
        games = record["games"] * discount if record else 0.0
        score = record["score_sum"] * discount if record else 0.0
        # Hardness ranks by the estimate as last measured (`matchup_estimate`);
        # refreshing ranks by how uncertain that has become: Beta(1, 1) over the
        # evidence discounted for its age, scaled by the age itself.
        alpha, beta = 1.0 + score, 1.0 + games - score
        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        score_estimates.append(matchup_estimate(record))
        probe_priorities.append((age + 1.0) * math.sqrt(variance))
    # Discovery and forgetting are different problems. A stale archive can
    # monopolize an age-weighted probe forever while new snapshots sit at the
    # prior mean, behind every established hard opponent. Admit one unseen
    # policy separately; prioritize the finite built-in set, then the newest
    # snapshot so archive growth cannot delay testing current strategies.
    unseen = [index for index, ref in enumerate(candidates) if opponent_key(ref) not in evidence]
    discovery = bool(unseen) and (
        count >= 3
        or (count == 2 and current_iteration % 2 == 1)
        or (count == 1 and current_iteration % 3 == 1)
    )
    # Screening spends whole lanes' games two apiece (one per seat) across
    # many stale opponents instead of one lane on the single stalest: a loss
    # in two games is signal enough, and the estimate then carries the
    # opponent into a hardness lane. It needs room for discovery and at least
    # one hardness lane beside it; smaller budgets refresh one opponent.
    screening = screen_opponents > 0 and count >= int(discovery) + screen_lanes + 1
    # Tiny lane budgets rotate purposes, rather than silently disabling
    # exploration. The schedule depends only on checkpointed iteration.
    refresh = not screening and (
        count >= 3
        or (count == 2 and not discovery)
        or (count == 1 and current_iteration % 3 != 0 and not discovery)
    )
    chosen_indices: list[int] = []
    roles: dict[str, Literal["hardness", "discovery", "probe", "screen"]] = {}
    if discovery:
        unseen_builtins = [index for index in unseen if isinstance(candidates[index], BuiltinRef)]
        index = (
            int(generator.choice(unseen_builtins))
            if unseen_builtins
            else unseen[-1]  # snapshots arrive sorted by iteration
        )
        chosen_indices.append(index)
        roles[opponent_key(candidates[index])] = "discovery"
    main_count = count - len(chosen_indices) - (screen_lanes if screening else int(refresh))
    # Rank individual matchups, not their aggregate archive mass: hundreds of
    # easy snapshots must not crowd out a few established hard opponents.
    # Shuffle before a stable sort so exact posterior ties have no age bias.
    order = generator.permutation(len(candidates))
    order = order[np.argsort(np.asarray(score_estimates)[order], kind="stable")]
    for index in order:
        if main_count == 0:
            break
        if int(index) not in chosen_indices:
            chosen_indices.append(int(index))
            roles[opponent_key(candidates[int(index)])] = "hardness"
            main_count -= 1
    if screening:
        remaining = np.asarray(
            [index for index in range(len(candidates)) if index not in chosen_indices],
            dtype=np.int64,
        )
        # Most uncertain first, exact ties in random order.
        remaining = remaining[generator.permutation(remaining.size)]
        priorities = np.asarray([probe_priorities[index] for index in remaining])
        for index in remaining[np.argsort(-priorities, kind="stable")][:screen_opponents]:
            chosen_indices.append(int(index))
            roles[opponent_key(candidates[int(index)])] = "screen"
    if refresh:
        remaining = [index for index in range(len(candidates)) if index not in chosen_indices]
        priorities = np.asarray([probe_priorities[index] for index in remaining])
        tied = np.flatnonzero(priorities == priorities.max())
        index = remaining[int(generator.choice(tied))]
        chosen_indices.append(index)
        roles[opponent_key(candidates[index])] = "probe"
    chosen = [candidates[index] for index in chosen_indices]
    active = {ref.iteration for ref in snapshots[-active_pool_size:]}
    selections: list[LeagueSelection] = []
    for ref in sorted(ref for ref in chosen if isinstance(ref, SnapshotRef)):
        selections.append(
            SnapshotSelection(
                ref,
                "active" if ref.iteration in active else "historical",
                roles[opponent_key(ref)],
            )
        )
    selections.extend(
        BuiltinSelection(ref, role=roles[opponent_key(ref)])
        for ref in sorted(ref for ref in chosen if isinstance(ref, BuiltinRef))
    )
    return selections


def _pfsp_weights(
    values: Sequence[SnapshotRef | BuiltinRef],
    score_rates: Mapping[str, float] | None,
) -> np.ndarray:
    rates = np.asarray(
        [
            (
                PFSP_UNMEASURED_SCORE_RATE
                if score_rates is None
                else float(score_rates.get(opponent_key(ref), PFSP_UNMEASURED_SCORE_RATE))
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
    selection_mode: Literal["stratified", "hardness"] = "hardness",
    matchup_evidence: Mapping[str, MatchupEvidence] | None = None,
    screen_lanes: int = 0,
    screen_opponents: int = 0,
) -> list[LeagueSelection]:
    """Select distinct recent-active, log-age historical, and built-in opponents.

    The default ``hardness`` mode ranks all eligible candidates by their
    game-count-weighted posterior learner score, randomizing exact ties. The
    configured lane counts specify only the total budget. One lane refreshes
    stale/uncertain evidence, another admits an unseen policy when available,
    and the rest take the hardest distinct matchups. With ``screen_lanes``,
    that many lanes instead screen ``screen_opponents`` of the most uncertain
    remaining matchups, which the caller plays two games apiece. One- and
    two-lane budgets rotate exploration purposes across iterations. With builtin_lanes=0,
    built-ins are disabled. Labels still identify each policy's
    age stratum, while ``role`` records why it was selected.

    In the optional ``stratified`` mode, active candidates are the newest
    ``active_pool_size`` frozen iterations.
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
    frozen yet still plays them from its first iteration. With sufficient
    eligible snapshots the total lane count stays
    ``active_count + historical_count + builtin_lanes`` however the contest
    goes; the neural ensemble contains only the selected snapshot lanes.

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
    if selection_mode not in {"stratified", "hardness"}:
        raise ValueError("unknown league selection mode")
    if active_count < 0 or historical_count < 0:
        raise ValueError("snapshot selection counts cannot be negative")
    if active_pool_size < 1:
        raise ValueError("active pool size must be positive")
    if builtin_lanes < 0:
        raise ValueError("built-in lane budget cannot be negative")
    if screen_lanes < 0 or screen_opponents < 0 or bool(screen_lanes) != bool(screen_opponents):
        raise ValueError("screen lanes and screen opponents must both be positive or both zero")
    if screen_lanes and selection_mode != "hardness":
        raise ValueError("screening is part of hardness selection")
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
    if selection_mode == "hardness":
        return _hardness_selections(
            trained,
            [BuiltinRef(name) for name in builtins] if builtin_lanes else [],
            count=active_count + historical_count + min(builtin_lanes, len(builtins)),
            current_iteration=current_iteration,
            active_pool_size=active_pool_size,
            generator=generator,
            evidence=matchup_evidence or {},
            screen_lanes=screen_lanes,
            screen_opponents=screen_opponents,
        )
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


# Thinned archives keep a snapshot the learner is not clearly beating
# whatever its age: forgetting shows first against those.
LEAGUE_ARCHIVE_PROTECTED_ESTIMATE = 0.6


def retired_snapshots(
    refs: Sequence[SnapshotRef],
    *,
    current_iteration: int,
    recent: int,
    evidence: Mapping[str, MatchupEvidence],
) -> list[SnapshotRef]:
    """Snapshots a thinned archive no longer holds, oldest first.

    Every snapshot younger than ``recent`` iterations is kept; an older one is
    kept while its iteration is a multiple of 2^floor(log2(age / recent)), so
    spacing doubles with age and the archive grows only logarithmically, yet
    every stage of the run keeps a representative. Iteration zero (the
    initial actor), and any snapshot whose estimate is below
    ``LEAGUE_ARCHIVE_PROTECTED_ESTIMATE``, are always kept. Granularity only
    grows with age, so a snapshot once retired never becomes due again, and
    whether one is off the grid depends on nothing but the two iterations.
    """
    if recent < 1:
        raise ValueError("a thinned archive keeps at least one recent snapshot")
    retired = []
    for ref in refs:
        if ref.iteration == 0 or snapshot_on_archive_grid(ref.iteration, current_iteration, recent):
            continue
        record = evidence.get(opponent_key(ref))
        if record is not None and matchup_estimate(record) < LEAGUE_ARCHIVE_PROTECTED_ESTIMATE:
            continue
        retired.append(ref)
    return retired


def snapshot_on_archive_grid(iteration: int, current_iteration: int, recent: int) -> bool:
    """Whether a thinned archive keeps this iteration by its age alone."""
    age = current_iteration - iteration
    if age < recent:
        return True
    level = (age // recent).bit_length() - 1
    return iteration % (1 << level) == 0


def script_game_counts(
    keys: Sequence[str],
    games: int,
    evidence: Mapping[str, MatchupEvidence],
) -> np.ndarray:
    """Games per script opponent: two each, the rest by (1 - estimate)^2.

    Games move in pairs so every opponent keeps both seats balanced. The floor
    keeps a beaten opponent measured, so a regression against it still shows;
    everything above it goes where the learner still loses or draws. With
    every opponent fully beaten the surplus is spread evenly.
    """
    count = len(keys)
    if count < 1 or games < 2 * count or games % 2:
        raise ValueError("weighted script games must be even and at least two per opponent")
    weights = np.asarray(
        [(1.0 - matchup_estimate(evidence.get(key))) ** 2 for key in keys], dtype=np.float64
    )
    pairs = games // 2 - count
    if weights.sum() <= 0.0:
        weights = np.ones(count)
    # Largest remainder, ties to the earlier opponent: deterministic and exact.
    quota = pairs * weights / weights.sum()
    extra = np.floor(quota).astype(np.int64)
    order = np.argsort(-(quota - extra), kind="stable")
    extra[order[: pairs - int(extra.sum())]] += 1
    return 2 * (1 + extra)
