from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def evaluator():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_architecture_campaign.py"
    spec = importlib.util.spec_from_file_location("architecture_campaign_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rollout():
    rewards = np.zeros((4, 719), dtype=np.float32)
    rewards[:, -1] = [1, 0, -1, -1]
    return SimpleNamespace(
        state_count=4 * 719,
        valid=np.ones((4, 719), dtype=bool),
        episode_seeds=np.arange(4_501_000, 4_501_004),
        seats=np.asarray([0, 1, 0, 1]),
        final_money=np.asarray([10.0, 10.0, 5.0, 1.0]),
        opponent_money=np.asarray([5.0, 10.0, 10.0, 10.0]),
        rewards=rewards,
        reward_mode="terminal-outcome",
    )


@pytest.mark.parametrize(
    "decoding",
    [
        "argmax",
        "sampled",
        "units",
        "kinds",
        "unit_kind",
        "unit_quantity",
        "kind_quantity",
        "quantities",
    ],
)
def test_panel_accepts_independent_head_decoding(evaluator, monkeypatch, decoding, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_architecture_campaign.py",
            "--artifact",
            "actor=actor.pt",
            "--output",
            str(tmp_path / "panel.json"),
            "--decoding",
            decoding,
        ],
    )
    assert evaluator.parse_args().decoding == decoding
    assert (
        evaluator.HEAD_DECODINGS[decoding]
        == {
            "argmax": (),
            "sampled": ("units", "kinds", "quantities"),
            "units": ("units",),
            "kinds": ("kinds",),
            "unit_kind": ("units", "kinds"),
            "unit_quantity": ("units", "quantities"),
            "kind_quantity": ("kinds", "quantities"),
            "quantities": ("quantities",),
        }[decoding]
    )


def test_panel_counts_low_money_with_strict_admission_boundary(evaluator):
    rows = evaluator.panel_rows(_rollout(), seed_start=4_501_000, games=4)
    for row, money in zip(rows, [999.0, 1_000.0, 1_001.0, 0.0], strict=True):
        row["money"] = money
    summary = evaluator.summarize(rows)
    assert summary["low_money_threshold"] == 1_000
    assert summary["low_money_count"] == 2
    assert summary["low_money_fraction"] == 0.5


def test_panel_scores_ties_and_preserves_matched_keys(evaluator):
    rows = evaluator.panel_rows(_rollout(), seed_start=4_501_000, games=4)
    assert [row["score"] for row in rows] == [1.0, 0.5, 0.0, 0.0]
    assert evaluator.summarize(rows)["score_rate"] == 0.375
    assert [(row["seed"], row["seat"]) for row in rows] == [
        (4_501_000, 0),
        (4_501_001, 1),
        (4_501_002, 0),
        (4_501_003, 1),
    ]


def test_panel_uses_native_outcome_when_float32_banks_appear_tied(evaluator):
    rollout = _rollout()
    rollout.final_money = np.asarray([100_000_001.0] * 4, dtype=np.float32)
    rollout.opponent_money = np.asarray([100_000_000.0] * 4, dtype=np.float32)
    assert np.array_equal(rollout.final_money, rollout.opponent_money)
    rows = evaluator.panel_rows(rollout, seed_start=4_501_000, games=4)
    assert rows[0]["money"] == rows[0]["opponent_money"]
    assert rows[0]["terminal_outcome"] == 1.0
    assert [row["score"] for row in rows] == [1.0, 0.5, 0.0, 0.0]


@pytest.mark.parametrize("corruption", ["incomplete", "wrong_seat", "nonfinite", "wrong_seed"])
def test_panel_rejects_invalid_evidence(evaluator, corruption):
    rollout = _rollout()
    if corruption == "incomplete":
        rollout.valid[0, -1] = False
    elif corruption == "wrong_seat":
        rollout.seats[0] = 1
    elif corruption == "wrong_seed":
        rollout.episode_seeds[0] += 1
    else:
        rollout.final_money[0] = np.nan
    with pytest.raises(ValueError, match="panel"):
        evaluator.panel_rows(rollout, seed_start=4_501_000, games=4)


def test_bootstrap_keeps_opponents_in_the_same_seed_cluster(evaluator):
    rows = evaluator.panel_rows(_rollout(), seed_start=4_501_000, games=4)
    reference = {
        opponent: [dict(row, score=0.5) for row in rows] for opponent in evaluator.OPPONENTS
    }
    candidate = {
        "starter": [dict(row, score=float(index % 2)) for index, row in enumerate(rows)],
        "scripted-v27": [dict(row, score=float(1 - index % 2)) for index, row in enumerate(rows)],
    }
    comparison = evaluator.paired_comparison(candidate, reference)
    assert comparison["seed_clusters"] == 4
    assert comparison["panels"]["overall"]["score"] == {"difference": 0.0, "ci95": [0.0, 0.0]}
    candidate["starter"] = candidate["starter"][::-1]
    with pytest.raises(ValueError, match="identical seeds and seats"):
        evaluator.paired_comparison(candidate, reference)


def test_bootstrap_pools_neural_reference_opponents_into_overall(evaluator):
    rows = evaluator.panel_rows(_rollout(), seed_start=4_501_000, games=4)
    opponents = (*evaluator.OPPONENTS, "bc")
    reference = {opponent: [dict(row, score=0.5) for row in rows] for opponent in opponents}
    candidate = {opponent: [dict(row, score=0.5) for row in rows] for opponent in opponents}
    candidate["bc"] = [dict(row, score=1.0) for row in rows]

    comparison = evaluator.paired_comparison(candidate, reference, opponents)

    assert comparison["panels"]["bc"]["score"]["difference"] == 0.5
    assert comparison["panels"]["starter"]["score"]["difference"] == 0.0
    assert comparison["panels"]["overall"]["score"]["difference"] == pytest.approx(0.5 / 3)
    with pytest.raises(ValueError, match="complete opponent panel"):
        evaluator.paired_comparison(candidate, reference)


def _reference_args(evaluator, monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_architecture_campaign.py",
            "--artifact",
            "actor=actor.pt",
            "--output",
            str(tmp_path / "panel.json"),
            *extra,
        ],
    )
    return evaluator.parse_args()


def test_reference_opponents_are_labelled_and_decoded_like_the_candidate(
    evaluator, monkeypatch, tmp_path
):
    args = _reference_args(
        evaluator,
        monkeypatch,
        tmp_path,
        "--reference-opponent",
        "bc=bc.pt",
        "--decoding",
        "sampled",
    )
    assert args.reference_opponent == [("bc", Path("bc.pt"))]
    assert _reference_args(evaluator, monkeypatch, tmp_path).reference_opponent == []
    for extra in (
        ("--reference-opponent", "starter=bc.pt"),
        ("--reference-opponent", "bc=a.pt", "--reference-opponent", "bc=b.pt"),
        ("--reference-opponent", "bc=bc.pt", "--decoding", "units"),
    ):
        with pytest.raises(SystemExit):
            _reference_args(evaluator, monkeypatch, tmp_path, *extra)
