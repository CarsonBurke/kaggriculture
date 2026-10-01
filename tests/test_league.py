from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.league import (
    LEAGUE_ARCHIVE_PROTECTED_ESTIMATE,
    MATCHUP_EVIDENCE_RETENTION,
    FrozenActorPool,
    SnapshotRef,
    copy_actor_snapshot,
    list_actor_snapshots,
    load_actor_snapshot,
    matchup_estimate,
    retired_snapshots,
    save_actor_snapshot,
    script_game_counts,
    select_league_mix,
    snapshot_on_archive_grid,
    snapshot_sha256,
    update_matchup_evidence,
    validate_matchup_evidence,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.structured import StructuredActor, StructuredConfig


def test_hardness_pool_prefers_hard_history_without_mandatory_age_slots(tmp_path) -> None:
    refs = [SnapshotRef(i, tmp_path / f"league-actor-{i:08d}.pt") for i in range(1, 41)]
    evidence = {
        f"{i:08d}": {"score_sum": 0.0 if i <= 10 else 100.0, "games": 100.0, "last_iteration": 199}
        for i in range(1, 41)
    }
    hard_slots = 0
    for seed in range(100):
        selected = select_league_mix(
            refs,
            current_iteration=200,
            active_count=2,
            historical_count=9,
            active_pool_size=16,
            generator=np.random.default_rng(seed),
            selection_mode="hardness",
            matchup_evidence=evidence,
        )
        assert len(selected) == len({row.key for row in selected}) == 11
        assert sum(row.role == "probe" for row in selected) == 1
        hard_slots += sum(row.role == "hardness" and row.ref.iteration <= 10 for row in selected)
    # All ten hard candidates are outside the active window, but still compete
    # for all ten main slots. Uniform or age-stratified sampling fails badly.
    assert hard_slots == 1000


def test_hardness_archive_size_cannot_crowd_out_established_hard_opponents(tmp_path) -> None:
    refs = [SnapshotRef(i, tmp_path / f"league-actor-{i:08d}.pt") for i in range(1, 511)]
    evidence = {
        f"{i:08d}": {"score_sum": 30.0 if i <= 10 else 70.0, "games": 100.0, "last_iteration": 599}
        for i in range(1, 511)
    }
    for seed in range(10):
        selected = select_league_mix(
            refs,
            current_iteration=600,
            active_count=2,
            historical_count=9,
            active_pool_size=16,
            generator=np.random.default_rng(seed),
            selection_mode="hardness",
            matchup_evidence=evidence,
        )
        assert {row.key for row in selected if row.role == "hardness"} == {
            f"{i:08d}" for i in range(1, 11)
        }
        assert len({row.key for row in selected}) == 11


def test_hardness_equal_posteriors_randomize_without_age_preference(tmp_path) -> None:
    refs = [SnapshotRef(i, tmp_path / f"league-actor-{i:08d}.pt") for i in range(1, 21)]
    selections = []
    for seed in range(10):
        selected = select_league_mix(
            refs,
            current_iteration=30,
            active_count=2,
            historical_count=1,
            active_pool_size=16,
            generator=np.random.default_rng(seed),
            selection_mode="hardness",
        )
        selections.append(tuple(row.key for row in selected if row.role == "hardness"))
    assert len(set(selections)) > 1


def test_hardness_probe_refreshes_stale_easy_opponent_and_keeps_builtins_last(tmp_path) -> None:
    refs = [SnapshotRef(i, tmp_path / f"league-actor-{i:08d}.pt") for i in range(1, 21)]
    evidence = {
        f"{i:08d}": {"score_sum": 0.0, "games": 100.0, "last_iteration": 199} for i in range(1, 21)
    }
    evidence["builtin_pass"] = {"score_sum": 1000.0, "games": 1000.0, "last_iteration": 0}
    selected = select_league_mix(
        refs,
        current_iteration=200,
        active_count=2,
        historical_count=6,
        active_pool_size=16,
        builtins=["pass"],
        builtin_lanes=3,
        generator=np.random.default_rng(11),
        selection_mode="hardness",
        matchup_evidence=evidence,
    )
    assert len(selected) == 9
    assert selected[-1].key == "builtin_pass"
    assert selected[-1].role == "probe"


def test_matchup_evidence_accumulates_game_counts_and_ages_only_once() -> None:
    evidence = {}
    update_matchup_evidence(evidence, {"00000001": (40, 0.75)}, iteration=10)
    update_matchup_evidence(evidence, {"00000001": (5, 0.0)}, iteration=12)
    discount = MATCHUP_EVIDENCE_RETENTION**2
    assert evidence["00000001"] == {
        "score_sum": 30.0 * discount,
        "games": 40.0 * discount + 5.0,
        "last_iteration": 12,
    }
    update_matchup_evidence(evidence, {}, iteration=15)
    assert evidence["00000001"]["last_iteration"] == 12
    with pytest.raises(ValueError, match="backward"):
        update_matchup_evidence(evidence, {"00000001": (2, 0.5)}, iteration=11)


def test_hardness_restored_evidence_and_rng_continue_identically(tmp_path) -> None:
    import copy
    import json

    refs = [SnapshotRef(i, tmp_path / f"league-actor-{i:08d}.pt") for i in range(1, 30)]
    evidence = {}
    update_matchup_evidence(
        evidence, {"00000003": (5, 0.0), "builtin_pass": (40, 1.0)}, iteration=29
    )
    generator = np.random.default_rng(51)
    restored_generator = np.random.default_rng()
    restored_generator.bit_generator.state = copy.deepcopy(generator.bit_generator.state)
    restored_evidence = validate_matchup_evidence(
        json.loads(json.dumps(evidence)), current_iteration=30
    )
    arguments = dict(
        current_iteration=30,
        active_count=2,
        historical_count=6,
        active_pool_size=16,
        builtins=["pass"],
        builtin_lanes=1,
        selection_mode="hardness",
    )
    assert select_league_mix(
        refs, generator=generator, matchup_evidence=evidence, **arguments
    ) == select_league_mix(
        refs, generator=restored_generator, matchup_evidence=restored_evidence, **arguments
    )
    assert generator.random() == restored_generator.random()


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {"bad": {"score_sum": 0.0, "games": 1.0, "last_iteration": 0}},
        {"00000001": {"score_sum": float("nan"), "games": 1.0, "last_iteration": 0}},
        {"00000001": {"score_sum": 2.0, "games": 1.0, "last_iteration": 0}},
        {"00000001": {"score_sum": 0.0, "games": 0.0, "last_iteration": 0}},
        {"00000001": {"score_sum": 0.0, "games": 1.0, "last_iteration": 31}},
    ],
)
def test_matchup_evidence_rejects_corrupt_or_future_recovery_state(evidence) -> None:
    with pytest.raises(ValueError, match="matchup evidence"):
        validate_matchup_evidence(evidence, current_iteration=30)


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
        selection_mode="stratified",
        current_iteration=33,
        active_count=2,
        historical_count=4,
        active_pool_size=8,
        generator=np.random.default_rng(91),
    )
    second = select_league_mix(
        refs,
        selection_mode="stratified",
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
        selection_mode="stratified",
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
        selection_mode="stratified",
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
            refs,
            selection_mode="stratified",
            generator=np.random.default_rng(seed),
            score_rates=beaten,
            **arguments,
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
            selection_mode="stratified",
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
        selection_mode="stratified",
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
            refs,
            selection_mode="stratified",
            generator=np.random.default_rng(0),
            builtins=["starter", "v27"],
            **arguments,
        )
    with pytest.raises(ValueError, match="distinct"):
        select_league_mix(
            refs,
            selection_mode="stratified",
            generator=np.random.default_rng(0),
            builtins=["starter", "starter"],
            **arguments,
        )


def test_league_mix_excludes_the_random_init_snapshot_from_tiny_pools(tmp_path) -> None:
    refs = [
        SnapshotRef(iteration, tmp_path / f"league-actor-{iteration:08d}.pt")
        for iteration in (0, 1)
    ]

    selected = select_league_mix(
        refs,
        selection_mode="stratified",
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
        selection_mode="stratified",
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
        selection_mode="stratified",
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
            selection_mode="stratified",
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
        selection_mode="stratified",
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
            selection_mode="stratified",
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
                selection_mode="stratified",
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
        select_league_mix([first, second], selection_mode="stratified", **arguments)
    with pytest.raises(ValueError, match="reused"):
        select_league_mix([first, reused], selection_mode="stratified", **arguments)


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


def test_default_hardness_discovers_new_policies_without_losing_stale_refresh(tmp_path):
    refs = _snapshot_refs(tmp_path, range(1, 501))
    evidence = {
        f"{i:08d}": {
            "score_sum": 0.0 if i <= 9 else 100.0,
            "games": 100.0,
            "last_iteration": 500 if i <= 9 else 300,
        }
        for i in range(1, 500)
    }
    for iteration in range(501, 521):
        selected = select_league_mix(
            refs,
            current_iteration=iteration,
            active_count=2,
            historical_count=9,
            active_pool_size=16,
            generator=np.random.default_rng(iteration),
            matchup_evidence=evidence,
        )
        assert len(selected) == len({row.key for row in selected}) == 11
        assert {row.ref.iteration for row in selected if row.role == "hardness"} == set(
            range(1, 10)
        )
        assert [row.ref.iteration for row in selected if row.role == "discovery"] == [iteration - 1]
        assert len([row for row in selected if row.role == "probe"]) == 1
        update_matchup_evidence(
            evidence,
            {row.key: (6, 0.0 if row.ref.iteration <= 9 else 1.0) for row in selected},
            iteration=iteration,
        )
        refs.extend(_snapshot_refs(tmp_path, [iteration]))


@pytest.mark.parametrize("count", [1, 2])
def test_small_hardness_budgets_rotate_discovery_refresh_and_hard_opponents(tmp_path, count):
    refs = _snapshot_refs(tmp_path, [1, 2, 3])
    evidence = {
        "00000001": {"score_sum": 0.0, "games": 100.0, "last_iteration": 9},
        "00000002": {"score_sum": 100.0, "games": 100.0, "last_iteration": 0},
    }
    observed = set()
    for iteration in range(13, 19):
        selected = select_league_mix(
            refs,
            current_iteration=iteration,
            active_count=count,
            historical_count=0,
            active_pool_size=16,
            generator=np.random.default_rng(3),
            matchup_evidence=evidence,
        )
        assert len(selected) == len({row.key for row in selected}) == count
        observed.update((row.ref.iteration, row.role) for row in selected)
        update_matchup_evidence(
            evidence,
            {row.key: (100, 0.0 if row.ref.iteration == 1 else 1.0) for row in selected},
            iteration=iteration,
        )
    assert {(1, "hardness"), (2, "probe"), (3, "discovery")} <= observed


def test_hardness_zero_builtin_budget_disables_builtin_candidates(tmp_path):
    selected = select_league_mix(
        _snapshot_refs(tmp_path, [1, 2]),
        current_iteration=3,
        active_count=2,
        historical_count=0,
        active_pool_size=16,
        generator=np.random.default_rng(0),
        builtins=["pass", "starter"],
        builtin_lanes=0,
    )
    assert {row.key for row in selected} == {"00000001", "00000002"}


def test_hardness_refresh_can_reach_older_unseen_backlog(tmp_path):
    refs = _snapshot_refs(tmp_path, range(1, 11))
    evidence = {"00000002": {"score_sum": 0.0, "games": 100.0, "last_iteration": 10}}
    selected = select_league_mix(
        refs,
        current_iteration=11,
        active_count=3,
        historical_count=0,
        active_pool_size=16,
        generator=np.random.default_rng(0),
        matchup_evidence=evidence,
    )
    assert {(row.ref.iteration, row.role) for row in selected} == {
        (1, "probe"),
        (2, "hardness"),
        (10, "discovery"),
    }


def test_hardness_discovers_unseen_builtins_before_snapshots_and_sorts_lanes(tmp_path):
    selected = select_league_mix(
        _snapshot_refs(tmp_path, [1, 2, 3]),
        current_iteration=4,
        active_count=2,
        historical_count=0,
        active_pool_size=16,
        generator=np.random.default_rng(0),
        builtins=["starter"],
        builtin_lanes=1,
    )
    assert selected[-1].key == "builtin_starter"
    assert selected[-1].role == "discovery"
    assert all(row.category != "builtin" for row in selected[:-1])


def test_staleness_does_not_make_a_beaten_snapshot_look_hard(tmp_path) -> None:
    # Snapshot 1 was beaten 90% of the time long ago; snapshots 2-4 are beaten
    # 70% of the time now. Shrinking stale evidence to even odds would rank 1
    # as the hardest; its estimate as last measured ranks it the easiest.
    refs = _snapshot_refs(tmp_path, [1, 2, 3, 4, 5])
    evidence = {
        "00000001": {"score_sum": 90.0, "games": 100.0, "last_iteration": 10},
        **{
            f"{i:08d}": {"score_sum": 70.0, "games": 100.0, "last_iteration": 499}
            for i in (2, 3, 4, 5)
        },
    }
    assert matchup_estimate(evidence["00000001"]) == pytest.approx(91 / 102)
    for seed in range(20):
        selected = select_league_mix(
            refs,
            current_iteration=500,
            active_count=3,
            historical_count=0,
            active_pool_size=16,
            generator=np.random.default_rng(seed),
            matchup_evidence=evidence,
        )
        hardness = {row.ref.iteration for row in selected if row.role == "hardness"}
        assert len(hardness) == 2 and 1 not in hardness
        # The stale snapshot is instead the one refreshed.
        assert [row.ref.iteration for row in selected if row.role == "probe"] == [1]


def test_screening_spreads_lanes_over_the_most_uncertain_stale_snapshots(tmp_path) -> None:
    refs = _snapshot_refs(tmp_path, range(1, 41))
    # 1-5 are hard and fresh, 6-39 beaten, older ones longer ago; 40 is unseen.
    evidence = {
        f"{i:08d}": {
            "score_sum": 30.0 if i <= 5 else 90.0,
            "games": 100.0,
            "last_iteration": 199 if i <= 5 else 100 + i,
        }
        for i in range(1, 40)
    }
    selected = select_league_mix(
        refs,
        current_iteration=200,
        active_count=2,
        historical_count=6,
        active_pool_size=16,
        generator=np.random.default_rng(0),
        matchup_evidence=evidence,
        screen_lanes=2,
        screen_opponents=8,
    )
    roles = {row.ref.iteration: row.role for row in selected}
    assert len(selected) == len(roles) == 14
    assert [i for i, role in roles.items() if role == "discovery"] == [40]
    assert {i for i, role in roles.items() if role == "hardness"} == {1, 2, 3, 4, 5}
    # Equal evidence, so the longest unplayed are the most uncertain.
    assert {i for i, role in roles.items() if role == "screen"} == set(range(6, 14))
    assert "probe" not in roles.values()


def test_screening_needs_room_for_discovery_and_a_hardness_lane(tmp_path) -> None:
    refs = _snapshot_refs(tmp_path, range(1, 11))
    selected = select_league_mix(
        refs,
        current_iteration=11,
        active_count=3,
        historical_count=0,
        active_pool_size=16,
        generator=np.random.default_rng(0),
        screen_lanes=2,
        screen_opponents=8,
    )
    assert sorted(row.role for row in selected) == ["discovery", "hardness", "probe"]
    with pytest.raises(ValueError, match="both be positive"):
        select_league_mix(
            refs,
            current_iteration=11,
            active_count=3,
            historical_count=0,
            active_pool_size=16,
            generator=np.random.default_rng(0),
            screen_lanes=2,
        )


def test_thinned_archive_is_logarithmic_stable_and_keeps_protected_snapshots(tmp_path) -> None:
    refs = _snapshot_refs(tmp_path, range(1300))
    retired = {
        ref.iteration
        for ref in retired_snapshots(refs, current_iteration=1300, recent=16, evidence={})
    }
    kept = set(range(1300)) - retired
    assert len(kept) == 116
    assert set(range(1284, 1300)) <= kept and 0 in kept
    # Spacing only grows with age: nothing retired earlier is due again later.
    for later in (1301, 1400, 2600):
        assert not any(snapshot_on_archive_grid(i, later, 16) for i in retired)
    # A snapshot the learner is not clearly beating is kept whatever its age.
    hard, easy = sorted(retired)[:2]
    evidence = {
        f"{hard:08d}": {"score_sum": 50.0, "games": 100.0, "last_iteration": 3},
        f"{easy:08d}": {"score_sum": 95.0, "games": 100.0, "last_iteration": 3},
    }
    assert matchup_estimate(evidence[f"{hard:08d}"]) < LEAGUE_ARCHIVE_PROTECTED_ESTIMATE
    again = {
        ref.iteration
        for ref in retired_snapshots(refs, current_iteration=1300, recent=16, evidence=evidence)
    }
    assert again == retired - {hard}


def test_script_games_go_two_each_then_to_opponents_not_yet_beaten() -> None:
    keys = ["script_a", "script_b", "script_c", "script_d", "script_e"]

    def beaten(rate: float) -> dict:
        return {"score_sum": 100.0 * rate, "games": 100.0, "last_iteration": 3}

    # Unmeasured opponents are even odds, so the first wave splits evenly.
    assert script_game_counts(keys, 40, {}).tolist() == [8] * 5
    counts = script_game_counts(
        keys,
        40,
        {"script_a": beaten(1.0), "script_b": beaten(0.9), "script_c": beaten(0.5)},
    )
    assert counts.sum() == 40 and np.all(counts % 2 == 0) and np.all(counts >= 2)
    assert counts[0] == 2 and counts[0] <= counts[1] < counts[2]
    everything_beaten = {key: beaten(1.0) for key in keys}
    assert np.ptp(script_game_counts(keys, 40, everything_beaten)) <= 2
    with pytest.raises(ValueError, match="at least two"):
        script_game_counts(keys, 8, {})
    with pytest.raises(ValueError, match="even"):
        script_game_counts(keys, 41, {})


def test_matchup_evidence_accepts_script_opponent_keys() -> None:
    record = {"score_sum": 3.0, "games": 8.0, "last_iteration": 4}
    assert validate_matchup_evidence({"script_bronze-v31": record}, current_iteration=4)
    with pytest.raises(ValueError, match="invalid league matchup evidence"):
        validate_matchup_evidence({"script_bad name": record}, current_iteration=4)
