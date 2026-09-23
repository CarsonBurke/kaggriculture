from __future__ import annotations

import importlib.util
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
