from __future__ import annotations

import math

import pytest

from kaggriculture.evaluation import (
    artifact_seed_usage,
    bounded_mean_interval,
    paired_score_comparison,
    seed_protocol,
    validate_seed_interval,
    validate_seed_protocol,
)


def test_actual_bc_seed_sets_include_every_population_initializer() -> None:
    first = {
        "seed_usage": [],
        "bc_provenance": {
            "datasets": [
                {"train_seeds": list(range(56)), "holdout_seeds": list(range(56, 64))},
                {"train_seeds": [1_000_000, 1_000_001], "holdout_seeds": [1_000_002]},
            ]
        },
    }
    second = {
        "seed_usage": [],
        "bc_provenance": {
            "datasets": [
                {"train_seeds": [4_000_001], "holdout_seeds": [4_000_002]},
            ]
        },
    }
    usage = artifact_seed_usage({"seed_usage": [], "initial_actor": {"agents": [first, second]}})
    assert {row["start"] for row in usage} == set(range(64)) | {
        1_000_000,
        1_000_001,
        1_000_002,
        4_000_001,
        4_000_002,
    }
    with pytest.raises(ValueError, match="recorded bc"):
        validate_seed_interval("development", 4_000_000, 32, usage=usage)
    for seen in (0, 55, 1_000_000):
        with pytest.raises(ValueError, match="reserved domain"):
            validate_seed_interval("finalist", seen, 32, usage=usage)


@pytest.mark.parametrize(
    "domain,start,previous",
    [
        ("development", 4_000_000, "online_rl"),
        ("screening", 10_000_000, "development"),
        ("finalist", 12_000_000, "screening"),
        ("finalist", 12_000_000, "online_rl"),
    ],
)
def test_actual_usage_overrides_cannot_bypass_disjointness(domain, start, previous) -> None:
    with pytest.raises(ValueError, match=f"recorded {previous}"):
        validate_seed_interval(
            domain, start, 32, usage=[{"domain": previous, "start": start + 31, "count": 1}]
        )
    with pytest.raises(ValueError, match="reserved domain"):
        validate_seed_interval(
            domain, start, 32, usage=[{"domain": previous, "start": start + 32, "count": 1}]
        )


def test_protocol_rejects_stale_provenance_and_changed_intervals() -> None:
    with pytest.raises(ValueError, match="fresh training"):
        artifact_seed_usage({"initial_actor": None})
    with pytest.raises(ValueError, match="actual train/holdout"):
        artifact_seed_usage(
            {"seed_usage": [], "bc_provenance": {"datasets": [{"path": "old-corpus"}]}}
        )
    protocol = seed_protocol("finalist", 12_000_000, 32, usage=[])
    with pytest.raises(ValueError, match="reported interval"):
        validate_seed_protocol(protocol, domain="finalist", start=12_000_001, count=32)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_unanimous_32_cluster_sample_never_has_zero_uncertainty(value) -> None:
    low, high = bounded_mean_interval([value] * 32)
    assert low < high
    assert high - low == pytest.approx(math.sqrt(math.log(40) / 64))


def test_paired_comparison_counts_maps_not_seats_or_opponents() -> None:
    result = paired_score_comparison(
        {seed: 1.0 for seed in range(32)}, {seed: 0.0 for seed in range(32)}
    )
    assert result["seed_clusters"] == 32
    assert result["mean_score_difference"] == 1
    assert result["score_difference_95ci"] == pytest.approx(
        [1 - 2 * math.sqrt(math.log(40) / 64), 1]
    )
    with pytest.raises(ValueError, match="identical"):
        paired_score_comparison({0: 1.0}, {1: 0.0})


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [1.01], []])
def test_finite_bounded_panel_required(values) -> None:
    with pytest.raises(ValueError):
        bounded_mean_interval(values)


def test_recorded_finalist_maps_cannot_be_reused_but_development_panels_can() -> None:
    seen = {"domain": "finalist", "start": 12_000_000, "count": 32}
    with pytest.raises(ValueError, match="recorded finalist"):
        validate_seed_interval("finalist", 12_000_000, 32, usage=[seen])
    validate_seed_interval("finalist", 12_000_032, 32, usage=[seen])
    development = {"domain": "development", "start": 4_000_000, "count": 32}
    validate_seed_interval("development", 4_000_000, 32, usage=[development])
