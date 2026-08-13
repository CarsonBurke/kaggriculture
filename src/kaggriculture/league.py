"""Immutable actor snapshots and reproducible active/historical league selection."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from kaggriculture.model import FarmActor, ModelConfig

LEAGUE_SNAPSHOT_FORMAT_VERSION = 2
_SNAPSHOT_NAME = re.compile(r"league-actor-(\d{8})\.pt")
_MAX_CANONICAL_ITERATION = 99_999_999


@dataclass(frozen=True, order=True)
class SnapshotRef:
    iteration: int
    path: Path


@dataclass(frozen=True)
class SnapshotSelection:
    ref: SnapshotRef
    category: Literal["initial", "active", "historical"]


def _snapshot_path(directory: Path, iteration: int) -> Path:
    if not 0 <= iteration <= _MAX_CANONICAL_ITERATION:
        raise ValueError(
            f"snapshot iteration must be in [0, {_MAX_CANONICAL_ITERATION}], got {iteration}"
        )
    return directory / f"league-actor-{iteration:08d}.pt"


def _model_config_dict(config: ModelConfig | dict[str, Any]) -> dict[str, Any]:
    return config.to_dict() if isinstance(config, ModelConfig) else dict(config)


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"league snapshot is not a dictionary: {path}")
    expected_keys = {"format_version", "iteration", "model_config", "actor"}
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        unexpected = sorted(set(payload) - expected_keys)
        raise ValueError(
            f"invalid league snapshot schema at {path}: missing={missing}, unexpected={unexpected}"
        )
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
    expected_config = ModelConfig().to_dict()
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
    payload: dict[str, Any], config: dict[str, Any], state: dict[str, torch.Tensor]
) -> bool:
    return (
        payload["model_config"] == config
        and payload["actor"].keys() == state.keys()
        and all(torch.equal(payload["actor"][name], value) for name, value in state.items())
    )


def save_actor_snapshot(directory: Path, actor: FarmActor, iteration: int) -> SnapshotRef:
    """Atomically save one immutable CPU actor snapshot.

    Repeating the exact same save is idempotent. Reusing an iteration for
    different weights fails instead of silently changing the frozen league.
    """
    directory = Path(directory)
    path = _snapshot_path(directory, iteration)
    state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    config = actor.config.to_dict()
    if path.exists():
        existing = _load_payload(path)
        _validate_canonical_filename(path, existing["iteration"])
        if not _matches_actor_state(existing, config, state):
            raise FileExistsError(f"refusing to replace immutable league snapshot: {path}")
        return SnapshotRef(iteration, path)

    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": LEAGUE_SNAPSHOT_FORMAT_VERSION,
        "iteration": iteration,
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
            if not _matches_actor_state(existing, config, state):
                raise FileExistsError(
                    f"refusing to replace immutable league snapshot: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return SnapshotRef(iteration, path)


def load_actor_snapshot(
    path: Path,
    *,
    expected_model_config: ModelConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> FarmActor:
    """Validate and strictly load a frozen actor snapshot."""
    path = Path(path)
    payload = _load_payload(path)
    _validate_canonical_filename(path, payload["iteration"])
    if expected_model_config is not None:
        expected = _model_config_dict(expected_model_config)
        if payload["model_config"] != expected:
            raise ValueError(f"league snapshot model configuration mismatch: {path}")
    try:
        config = ModelConfig(**payload["model_config"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid league snapshot model configuration: {path}") from error
    # Parameter initialization is discarded immediately by strict loading, so
    # frozen-policy I/O must not perturb training's checkpointed RNG stream.
    with torch.random.fork_rng(devices=[]):
        actor = FarmActor(config).to(device)
    try:
        actor.load_state_dict(payload["actor"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"invalid league snapshot actor state: {path}") from error
    actor.eval().requires_grad_(False)
    return actor


def snapshot_sha256(path: Path) -> str:
    """Return the content digest used to bind a checkpoint to its league archive."""
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def copy_actor_snapshot(
    source: Path,
    directory: Path,
    *,
    expected_model_config: ModelConfig | dict[str, Any],
) -> SnapshotRef:
    """Validate and atomically copy an immutable snapshot into another archive."""
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


def _sample_without_replacement(
    values: Sequence[SnapshotRef], count: int, generator: np.random.Generator
) -> list[SnapshotRef]:
    if count <= 0 or not values:
        return []
    size = min(count, len(values))
    indices = np.atleast_1d(generator.choice(len(values), size=size, replace=False))
    return [values[int(index)] for index in indices]


def _sample_log_age_strata(
    values: Sequence[SnapshotRef],
    count: int,
    current_iteration: int,
    generator: np.random.Generator,
) -> list[SnapshotRef]:
    """Round-robin across log2 age buckets, sampling within each bucket."""
    buckets: dict[int, list[SnapshotRef]] = {}
    for ref in values:
        age = max(1, current_iteration - ref.iteration)
        buckets.setdefault(age.bit_length() - 1, []).append(ref)
    selected: list[SnapshotRef] = []
    while buckets and len(selected) < count:
        bucket_ids = list(buckets)
        generator.shuffle(bucket_ids)
        for bucket in bucket_ids:
            candidates = buckets[bucket]
            index = int(generator.integers(0, len(candidates)))
            selected.append(candidates.pop(index))
            if not candidates:
                del buckets[bucket]
            if len(selected) == count:
                break
    return selected


def select_snapshot_mix(
    refs: Sequence[SnapshotRef],
    *,
    current_iteration: int,
    active_count: int,
    historical_count: int,
    active_pool_size: int,
    generator: np.random.Generator,
    include_initial: bool = True,
) -> list[SnapshotSelection]:
    """Select distinct initial, recent-active, and log-age historical opponents.

    The optional initial anchor is additional to ``active_count`` and
    ``historical_count``. Active candidates are the newest
    ``active_pool_size`` frozen iterations. Historical candidates must be
    strictly older than that complete active window. Undersized pools return
    fewer selections without duplicating a policy.
    """
    if current_iteration < 0:
        raise ValueError("current iteration cannot be negative")
    if active_count < 0 or historical_count < 0:
        raise ValueError("snapshot selection counts cannot be negative")
    if active_pool_size < 1:
        raise ValueError("active pool size must be positive")
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
    if not eligible:
        return []

    initial = eligible_by_iteration.get(0)
    non_initial = [ref for ref in eligible if ref.iteration != 0]
    active_window = non_initial[-active_pool_size:]
    active = _sample_without_replacement(active_window, active_count, generator)
    active_iterations = {ref.iteration for ref in active_window}
    historical_candidates = [ref for ref in non_initial if ref.iteration not in active_iterations]
    historical = _sample_log_age_strata(
        historical_candidates,
        min(historical_count, len(historical_candidates)),
        current_iteration,
        generator,
    )

    selections = []
    if include_initial and initial is not None:
        selections.append(SnapshotSelection(initial, "initial"))
    selections.extend(SnapshotSelection(ref, "active") for ref in sorted(active))
    selections.extend(SnapshotSelection(ref, "historical") for ref in sorted(historical))
    return selections
