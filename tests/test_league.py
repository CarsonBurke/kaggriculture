from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.league import (
    FrozenActorPool,
    SnapshotRef,
    copy_actor_snapshot,
    list_actor_snapshots,
    load_actor_snapshot,
    save_actor_snapshot,
    select_league_mix,
    snapshot_sha256,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.structured import StructuredActor, StructuredConfig


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
    assert set(payload) == {"format_version", "iteration", "model_config", "actor", "architecture"}
    assert payload["architecture"] == CONV_ENTITY
    assert all(value.device.type == "cpu" for value in payload["actor"].values())
    loaded = load_actor_snapshot(ref.path, expected_model_config=actor.config)
    for expected, actual in zip(actor.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_structured_actor_snapshot_round_trips_through_the_registry(tmp_path) -> None:
    config = StructuredConfig(
        model_dim=16,
        attention_heads=2,
        ffn_multiplier=1,
        farm_blocks=1,
        opponent_latents=2,
        latents=4,
        core_layers=1,
        quantity_rank=4,
    )
    actor = StructuredActor(config)
    ref = save_actor_snapshot(tmp_path, actor, 4)
    payload = torch.load(ref.path, map_location="cpu", weights_only=False)
    assert payload["architecture"] == STRUCTURED

    loaded = load_actor_snapshot(ref.path, expected_model_config=config)

    assert type(loaded) is StructuredActor
    for expected, actual in zip(actor.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    with pytest.raises(ValueError):
        load_actor_snapshot(ref.path, expected_model_config=_actor().config)


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
    assert copied.path.stat().st_ino == source.path.stat().st_ino
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


def test_league_mix_is_distinct_reproducible_and_separates_age_windows(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in range(33)
    ]
    first = select_league_mix(
        refs,
        current_iteration=33,
        active_count=2,
        historical_count=4,
        active_pool_size=8,
        generator=np.random.default_rng(91),
    )
    second = select_league_mix(
        refs,
        current_iteration=33,
        active_count=2,
        historical_count=4,
        active_pool_size=8,
        generator=np.random.default_rng(91),
    )

    assert first == second
    assert len({selection.ref.iteration for selection in first}) == len(first) == 6
    active = [row.ref.iteration for row in first if row.category == "active"]
    historical = [row.ref.iteration for row in first if row.category == "historical"]
    assert all(iteration >= 25 for iteration in active)
    assert all(0 < iteration < 25 for iteration in historical)
    assert len({(33 - iteration).bit_length() - 1 for iteration in historical}) >= 3
    # Positional contract: sorted actives lead, sorted historicals follow. The
    # iteration benchmark reconstructs the temperature/deterministic decode
    # from this ordering.
    assert [row.category for row in first] == ["active"] * 2 + ["historical"] * 4
    assert active == sorted(active)
    assert historical == sorted(historical)


def test_league_mix_fills_every_log_age_rung_of_a_deep_archive(tmp_path) -> None:
    """Six historical seats are the production count: one per occupied log2 rung.

    At iteration 500 the active window is 484-499. Historical ages 17-499 occupy
    five rungs; six seats take all five plus one PFSP refill. Two seats would
    leave the older rungs unused.
    """
    refs = _snapshot_refs(tmp_path, range(1, 500))
    selected = select_league_mix(
        refs,
        current_iteration=500,
        active_count=2,
        historical_count=6,
        active_pool_size=16,
        generator=np.random.default_rng(7),
    )
    historical = [row.ref.iteration for row in selected if row.category == "historical"]
    ages = [500 - iteration for iteration in historical]
    buckets = {(age.bit_length() - 1) for age in ages}

    assert len(historical) == 6
    assert buckets == {4, 5, 6, 7, 8}
    assert max(ages) >= 256
    assert min(ages) >= 17


def _snapshot_refs(tmp_path, iterations) -> list[SnapshotRef]:
    return [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in iterations
    ]


def test_league_mix_reserves_lanes_for_built_ins_after_every_snapshot(tmp_path) -> None:
    refs = _snapshot_refs(tmp_path, range(1, 20))

    selected = select_league_mix(
        refs,
        current_iteration=20,
        active_count=2,
        historical_count=2,
        active_pool_size=16,
        generator=np.random.default_rng(0),
        builtins=["pass", "random", "starter"],
        builtin_lanes=3,
        score_rates={"builtin_pass": 0.0, "builtin_random": 0.0, "builtin_starter": 0.0},
    )

    # Positional contract: the wave numbers its frozen-module lanes first, so
    # every built-in has to trail every snapshot.
    assert [row.category for row in selected] == ["active"] * 2 + ["historical"] * 2 + [
        "builtin"
    ] * 3
    assert [row.key for row in selected[-3:]] == [
        "builtin_pass",
        "builtin_random",
        "builtin_starter",
    ]
    assert [row.label for row in selected[-3:]] == ["pass", "random", "starter"]


def test_league_mix_drains_beaten_built_in_lanes_back_to_snapshots(tmp_path) -> None:
    """A beaten built-in must lose its lane, not merely lose weight inside it."""
    refs = _snapshot_refs(tmp_path, range(1, 20))
    arguments = {
        "current_iteration": 20,
        "active_count": 2,
        "historical_count": 2,
        "active_pool_size": 16,
        "builtins": ["pass", "random", "starter"],
        "builtin_lanes": 3,
    }
    beaten = {"builtin_pass": 1.0, "builtin_random": 1.0, "builtin_starter": 1.0}

    for seed in range(16):
        selected = select_league_mix(
            refs, generator=np.random.default_rng(seed), score_rates=beaten, **arguments
        )
        # The reserved lanes are released, not dropped: the lane count that
        # the wave's stacked frozen forward is captured for stays put.
        assert len(selected) == 7
        assert not [row for row in selected if row.category == "builtin"]


def test_league_mix_gives_reserved_lanes_to_the_unbeaten_built_in(tmp_path) -> None:
    refs = _snapshot_refs(tmp_path, range(1, 20))
    counts = {"pass": 0, "random": 0, "starter": 0}

    for seed in range(64):
        selected = select_league_mix(
            refs,
            current_iteration=20,
            active_count=2,
            historical_count=2,
            active_pool_size=16,
            generator=np.random.default_rng(seed),
            builtins=["pass", "random", "starter"],
            builtin_lanes=1,
            # Only `starter` still beats the learner; the other two are done.
            score_rates={"builtin_pass": 1.0, "builtin_random": 1.0, "builtin_starter": 0.0},
        )
        for row in selected:
            if row.category == "builtin":
                counts[row.label] += 1

    # The single reserved lane is contested against one more snapshot at
    # weight (1 - 0.5)^2 = 0.25 against starter's 1.0, so starter takes about
    # four fifths of them and the retired pair take none.
    assert counts["pass"] == counts["random"] == 0
    assert counts["starter"] > 40


def test_league_mix_plays_built_ins_before_any_snapshot_exists(tmp_path) -> None:
    selected = select_league_mix(
        [],
        current_iteration=1,
        active_count=2,
        historical_count=2,
        active_pool_size=16,
        generator=np.random.default_rng(3),
        builtins=["starter"],
        builtin_lanes=1,
    )

    assert [(row.label, row.category) for row in selected] == [("starter", "builtin")]


def test_league_mix_rejects_unknown_and_duplicated_built_ins(tmp_path) -> None:
    arguments = {
        "current_iteration": 2,
        "active_count": 1,
        "historical_count": 0,
        "active_pool_size": 16,
        "builtin_lanes": 2,
    }
    refs = _snapshot_refs(tmp_path, (1,))

    with pytest.raises(ValueError, match="unknown built-in"):
        select_league_mix(
            refs, generator=np.random.default_rng(0), builtins=["starter", "v27"], **arguments
        )
    with pytest.raises(ValueError, match="distinct"):
        select_league_mix(
            refs, generator=np.random.default_rng(0), builtins=["starter", "starter"], **arguments
        )


def test_league_mix_excludes_the_random_init_snapshot_from_tiny_pools(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in (0, 1)
    ]

    selected = select_league_mix(
        refs,
        current_iteration=2,
        active_count=4,
        historical_count=4,
        active_pool_size=16,
        generator=np.random.default_rng(2),
    )

    assert [(row.ref.iteration, row.category) for row in selected] == [(1, "active")]


def test_league_mix_keeps_a_pretrained_start_as_a_baseline_opponent(tmp_path) -> None:
    """A warm-started run's iteration-0 snapshot stays a league candidate.

    The default exclusion targets the random-init snapshot; under a
    warm start iteration 0 is the pretrained baseline, and playing it holds
    anti-regression pressure against the learner's own starting point.
    """
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in (0, 1)
    ]

    selected = select_league_mix(
        refs,
        current_iteration=2,
        active_count=4,
        historical_count=4,
        active_pool_size=16,
        generator=np.random.default_rng(2),
        pretrained_start=True,
    )

    assert [(row.ref.iteration, row.category) for row in selected] == [
        (0, "active"),
        (1, "active"),
    ]


def test_league_mix_treats_a_lone_resume_snapshot_as_active(tmp_path) -> None:
    resumed = SnapshotRef(12, tmp_path / "league-actor-00000012.pt")

    selected = select_league_mix(
        [resumed],
        current_iteration=13,
        active_count=1,
        historical_count=0,
        active_pool_size=16,
        generator=np.random.default_rng(4),
    )

    assert [(row.ref.iteration, row.category) for row in selected] == [(12, "active")]


def test_league_mix_retires_fully_beaten_opponents(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in range(1, 9)
    ]
    score_rates = {f"{iteration:08d}": 1.0 for iteration in range(1, 9)}
    score_rates["00000006"] = 0.4

    for seed in range(32):
        selected = select_league_mix(
            refs,
            current_iteration=9,
            active_count=2,
            historical_count=2,
            active_pool_size=4,
            generator=np.random.default_rng(seed),
            score_rates=score_rates,
        )
        # Iterations 5-8 form the active window; every candidate but 6 is
        # fully beaten, and the entire historical stratum (1-4) is beaten too.
        assert [(row.ref.iteration, row.category) for row in selected] == [(6, "active")]


def test_league_mix_returns_empty_when_every_opponent_is_beaten(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in range(1, 9)
    ]

    selected = select_league_mix(
        refs,
        current_iteration=9,
        active_count=2,
        historical_count=2,
        active_pool_size=4,
        generator=np.random.default_rng(11),
        score_rates={f"{iteration:08d}": 1.0 for iteration in range(1, 9)},
    )

    assert selected == []


def test_league_mix_prioritizes_competitive_over_unmeasured_opponents(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in (1, 2)
    ]
    counts = {1: 0, 2: 0}

    for seed in range(400):
        selected = select_league_mix(
            refs,
            current_iteration=3,
            active_count=1,
            historical_count=0,
            active_pool_size=16,
            generator=np.random.default_rng(seed),
            score_rates={"00000001": 0.0},
        )
        counts[selected[0].ref.iteration] += 1

    # Weights are (1 - 0.0)^2 = 1.0 against the unmeasured (1 - 0.5)^2 = 0.25,
    # so the never-beaten opponent should win roughly 80% of draws.
    assert counts[1] + counts[2] == 400
    assert counts[1] > 280
    assert counts[2] > 20


def test_league_mix_rejects_invalid_score_rates(tmp_path) -> None:
    refs = [SnapshotRef(1, tmp_path / "league-actor-00000001.pt")]

    for invalid in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="score rates"):
            select_league_mix(
                refs,
                current_iteration=2,
                active_count=1,
                historical_count=0,
                active_pool_size=16,
                generator=np.random.default_rng(0),
                score_rates={"00000001": invalid},
            )


def test_league_mix_rejects_conflicting_duplicate_refs(tmp_path) -> None:
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
        select_league_mix([first, second], **arguments)
    with pytest.raises(ValueError, match="reused"):
        select_league_mix([first, reused], **arguments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_iteration": -1},
        {"active_count": -1},
        {"historical_count": -1},
        {"active_pool_size": 0},
    ],
)
def test_league_mix_rejects_invalid_configuration(tmp_path, kwargs) -> None:
    arguments = {
        "current_iteration": 1,
        "active_count": 1,
        "historical_count": 1,
        "active_pool_size": 1,
        "generator": np.random.default_rng(1),
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        select_league_mix([SnapshotRef(0, tmp_path / "unused")], **arguments)


def test_frozen_actor_pool_reuses_slots_and_reloads_in_place(tmp_path) -> None:
    actor = _actor()
    first = save_actor_snapshot(tmp_path, actor, 1)
    with torch.no_grad():
        next(actor.parameters()).add_(0.5)
    second = save_actor_snapshot(tmp_path, actor, 2)
    pool = FrozenActorPool(actor.config, torch.device("cpu"))

    [loaded] = pool.acquire([first.path])
    pointer = next(loaded.parameters()).data_ptr()
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())

    [reacquired] = pool.acquire([first.path])
    assert reacquired is loaded
    assert next(reacquired.parameters()).data_ptr() == pointer

    # Selecting a different snapshot reuses the slot and its parameter
    # storage, which is what keeps captured compiled forwards valid.
    [reloaded] = pool.acquire([second.path])
    assert reloaded is loaded
    assert next(reloaded.parameters()).data_ptr() == pointer
    reference = load_actor_snapshot(second.path, expected_model_config=actor.config)
    for actual, expected in zip(reloaded.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
