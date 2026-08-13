from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.league import (
    SnapshotRef,
    copy_actor_snapshot,
    list_actor_snapshots,
    load_actor_snapshot,
    save_actor_snapshot,
    select_snapshot_mix,
    snapshot_sha256,
)
from kaggriculture.model import FarmActor, ModelConfig


def _actor() -> FarmActor:
    return FarmActor(
        ModelConfig(
            cnn_width=8,
            cnn_blocks=1,
            model_dim=16,
            transformer_layers=3,
            attention_heads=2,
        )
    )


def test_actor_snapshot_is_small_strict_cpu_only_and_idempotent(tmp_path) -> None:
    actor = _actor()
    ref = save_actor_snapshot(tmp_path, actor, 7)
    repeated = save_actor_snapshot(tmp_path, actor, 7)

    assert repeated == ref
    assert ref.path.name == "league-actor-00000007.pt"
    payload = torch.load(ref.path, map_location="cpu", weights_only=False)
    assert set(payload) == {"format_version", "iteration", "model_config", "actor"}
    assert all(value.device.type == "cpu" for value in payload["actor"].values())
    loaded = load_actor_snapshot(ref.path, expected_model_config=actor.config)
    for expected, actual in zip(actor.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_actor_snapshot_refuses_to_mutate_an_existing_iteration(tmp_path) -> None:
    actor = _actor()
    save_actor_snapshot(tmp_path, actor, 3)
    with torch.no_grad():
        next(actor.parameters()).add_(1.0)

    with pytest.raises(FileExistsError, match="immutable"):
        save_actor_snapshot(tmp_path, actor, 3)


def test_actor_snapshot_copy_is_byte_exact_validated_and_immutable(tmp_path) -> None:
    actor = _actor()
    source = save_actor_snapshot(tmp_path / "source", actor, 3)

    copied = copy_actor_snapshot(
        source.path,
        tmp_path / "destination",
        expected_model_config=actor.config,
    )

    assert copied.path.read_bytes() == source.path.read_bytes()
    assert snapshot_sha256(copied.path) == snapshot_sha256(source.path)
    assert (
        copy_actor_snapshot(
            source.path,
            tmp_path / "destination",
            expected_model_config=actor.config,
        )
        == copied
    )

    conflicting_actor = _actor()
    with torch.no_grad():
        next(conflicting_actor.parameters()).add_(1.0)
    conflicting = save_actor_snapshot(tmp_path / "conflict", conflicting_actor, 3)
    with pytest.raises(FileExistsError, match="immutable"):
        copy_actor_snapshot(
            conflicting.path,
            tmp_path / "destination",
            expected_model_config=actor.config,
        )


def test_loading_frozen_actor_does_not_advance_torch_rng(tmp_path) -> None:
    actor = _actor()
    ref = save_actor_snapshot(tmp_path, actor, 4)
    torch.manual_seed(917)
    before = torch.get_rng_state().clone()

    load_actor_snapshot(ref.path, expected_model_config=actor.config)

    assert torch.equal(torch.get_rng_state(), before)


def test_snapshot_loading_rejects_schema_filename_and_model_mismatches(tmp_path) -> None:
    actor = _actor()
    ref = save_actor_snapshot(tmp_path, actor, 2)
    payload = torch.load(ref.path, map_location="cpu", weights_only=False)

    bad_schema = tmp_path / "league-actor-00000003.pt"
    torch.save(payload | {"extra": True, "iteration": 3}, bad_schema)
    with pytest.raises(ValueError, match="schema"):
        load_actor_snapshot(bad_schema)

    mismatched_name = tmp_path / "league-actor-00000004.pt"
    torch.save(payload, mismatched_name)
    with pytest.raises(ValueError, match="filename/iteration"):
        load_actor_snapshot(mismatched_name)

    other = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_actor_snapshot(ref.path, expected_model_config=other)


@pytest.mark.parametrize("iteration", ["2", 2.9, True])
def test_snapshot_loading_rejects_non_integer_iterations(tmp_path, iteration) -> None:
    actor = _actor()
    ref = save_actor_snapshot(tmp_path, actor, 2)
    payload = torch.load(ref.path, map_location="cpu", weights_only=True)
    payload["iteration"] = iteration
    torch.save(payload, ref.path)

    with pytest.raises(ValueError, match="invalid iteration"):
        load_actor_snapshot(ref.path)


def test_snapshot_loading_rejects_incomplete_model_config(tmp_path) -> None:
    actor = _actor()
    ref = save_actor_snapshot(tmp_path, actor, 2)
    payload = torch.load(ref.path, map_location="cpu", weights_only=True)
    payload["model_config"].pop("quantity_rank")
    torch.save(payload, ref.path)

    with pytest.raises(ValueError, match="configuration schema"):
        load_actor_snapshot(ref.path)


def test_listing_ignores_noncanonical_files_and_sorts_numerically(tmp_path) -> None:
    actor = _actor()
    for iteration in (12, 0, 3):
        save_actor_snapshot(tmp_path, actor, iteration)
    (tmp_path / "latest.pt").touch()
    (tmp_path / "league-actor-3.pt").touch()

    assert [ref.iteration for ref in list_actor_snapshots(tmp_path)] == [0, 3, 12]


def test_snapshot_mix_is_distinct_reproducible_and_separates_age_windows(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in range(33)
    ]
    first = select_snapshot_mix(
        refs,
        current_iteration=33,
        active_count=2,
        historical_count=4,
        active_pool_size=8,
        generator=np.random.default_rng(91),
    )
    second = select_snapshot_mix(
        refs,
        current_iteration=33,
        active_count=2,
        historical_count=4,
        active_pool_size=8,
        generator=np.random.default_rng(91),
    )

    assert first == second
    assert first[0].category == "initial"
    assert first[0].ref.iteration == 0
    assert len({selection.ref.iteration for selection in first}) == len(first) == 7
    active = [row.ref.iteration for row in first if row.category == "active"]
    historical = [row.ref.iteration for row in first if row.category == "historical"]
    assert all(iteration >= 25 for iteration in active)
    assert all(0 < iteration < 25 for iteration in historical)
    assert len({(33 - iteration).bit_length() - 1 for iteration in historical}) >= 3


def test_snapshot_mix_handles_tiny_pools_without_duplicates(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in (0, 1)
    ]

    selected = select_snapshot_mix(
        refs,
        current_iteration=2,
        active_count=4,
        historical_count=4,
        active_pool_size=16,
        generator=np.random.default_rng(2),
    )

    assert [(row.ref.iteration, row.category) for row in selected] == [
        (0, "initial"),
        (1, "active"),
    ]


def test_snapshot_mix_never_relabels_a_resume_snapshot_as_initial(tmp_path) -> None:
    resumed = SnapshotRef(12, tmp_path / "league-actor-00000012.pt")

    selected = select_snapshot_mix(
        [resumed],
        current_iteration=13,
        active_count=1,
        historical_count=0,
        active_pool_size=16,
        generator=np.random.default_rng(4),
    )

    assert [(row.ref.iteration, row.category) for row in selected] == [(12, "active")]


def test_snapshot_mix_excludes_iteration_zero_when_initial_anchor_is_disabled(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in (0, 1)
    ]

    selected = select_snapshot_mix(
        refs,
        current_iteration=2,
        active_count=2,
        historical_count=2,
        active_pool_size=16,
        generator=np.random.default_rng(4),
        include_initial=False,
    )

    assert [(row.ref.iteration, row.category) for row in selected] == [(1, "active")]


def test_snapshot_mix_rejects_conflicting_duplicate_refs(tmp_path) -> None:
    first = SnapshotRef(1, tmp_path / "first.pt")
    second = SnapshotRef(1, tmp_path / "second.pt")
    reused = SnapshotRef(2, tmp_path / "first.pt")
    arguments = {
        "current_iteration": 3,
        "active_count": 1,
        "historical_count": 1,
        "active_pool_size": 1,
        "generator": np.random.default_rng(1),
    }

    with pytest.raises(ValueError, match="conflicting paths"):
        select_snapshot_mix([first, second], **arguments)
    with pytest.raises(ValueError, match="reused"):
        select_snapshot_mix([first, reused], **arguments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_iteration": -1},
        {"active_count": -1},
        {"historical_count": -1},
        {"active_pool_size": 0},
    ],
)
def test_snapshot_mix_rejects_invalid_configuration(tmp_path, kwargs) -> None:
    arguments = {
        "current_iteration": 1,
        "active_count": 1,
        "historical_count": 1,
        "active_pool_size": 1,
        "generator": np.random.default_rng(1),
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        select_snapshot_mix([SnapshotRef(0, tmp_path / "unused")], **arguments)
