from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic():
    path = Path(__file__).parents[1] / "scripts" / "fit_common_policy_critics.py"
    spec = importlib.util.spec_from_file_location("common_policy_critic_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_keeps_both_seats_of_each_physical_game_together(diagnostic):
    seeds = np.asarray([4502000, 4502000, 4502001, 4502001, 4502005])
    train, holdout = diagnostic.split_seed_rows(seeds)
    assert train.tolist() == [2, 3]
    assert holdout.tolist() == [0, 1, 4]
    assert set(seeds[train]).isdisjoint(seeds[holdout])


def test_r_squared_penalizes_constant_bias_and_nulls_constant_outcomes(diagnostic):
    outcomes = np.asarray([-1.0, 1.0, 1.0])
    predictions = np.repeat((outcomes + 0.5)[:, None], 719, axis=1)
    result = diagnostic.fit_statistics(
        predictions, outcomes, np.arange(3), np.asarray(["self_play", "self_play", "starter"])
    )
    assert result["self_play"]["all"]["r_squared"] == 0.75
    assert result["self_play"]["all"]["mse"] == 0.25
    assert result["starter"]["all"]["r_squared"] is None
    assert result["all"]["ttg_1_32"]["states"] == 3 * 32
    assert result["all"]["ttg_513_plus"]["states"] == 3 * 207


def test_holdout_metrics_exclude_training_predictions(diagnostic):
    predictions = np.zeros((3, 719))
    predictions[0] = 100
    result = diagnostic.fit_statistics(
        predictions, np.asarray([1, -1, 1]), np.asarray([1, 2]), np.asarray(["same"] * 3)
    )
    assert result["all"]["all"]["mse"] == 1.0
    assert result["all"]["all"]["r_squared"] == 0.0
