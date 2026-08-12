from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_evaluate_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_counts_ties_and_seats() -> None:
    evaluate = _load_evaluate_module()
    rows = [
        evaluate.GameResult(1, 0, 10.0, 8.0, "DONE", "DONE", 0.1),
        evaluate.GameResult(1, 1, 7.0, 7.0, "DONE", "DONE", 0.1),
        evaluate.GameResult(2, 0, 4.0, 9.0, "DONE", "DONE", 0.1),
    ]

    summary = evaluate.summarize(rows)

    assert summary["wins"] == 1
    assert summary["ties"] == 1
    assert summary["losses"] == 1
    assert summary["score_rate"] == 0.5
    assert summary["seats"]["1"]["score_rate"] == 0.5
    assert summary["seed_clusters"] == 2
    assert summary["complete_seat_pairs"] == 1


def test_error_result_is_excluded_from_metrics() -> None:
    evaluate = _load_evaluate_module()
    rows = [
        evaluate.GameResult(1, 0, None, None, "ERROR", "UNKNOWN", 0.1, "boom"),
        evaluate.GameResult(1, 1, 12.0, 10.0, "DONE", "DONE", 0.1),
    ]

    summary = evaluate.summarize(rows)

    assert summary["games_requested"] == 2
    assert summary["games_completed"] == 1
    assert summary["complete_seat_pairs"] == 0
    assert summary["incomplete_seed_clusters"] == 1
    assert summary["score_rate_95ci"] == [0.0, 1.0]
    assert summary["errors"] == 1
    assert summary["score_rate"] == 1.0


def test_confidence_interval_clusters_the_two_seats_by_seed() -> None:
    evaluate = _load_evaluate_module()
    rows = [
        evaluate.GameResult(1, 0, 10.0, 0.0, "DONE", "DONE", 0.1),
        evaluate.GameResult(1, 1, 10.0, 0.0, "DONE", "DONE", 0.1),
        evaluate.GameResult(2, 0, 0.0, 10.0, "DONE", "DONE", 0.1),
        evaluate.GameResult(2, 1, 0.0, 10.0, "DONE", "DONE", 0.1),
    ]

    summary = evaluate.summarize(rows)

    assert summary["complete_seat_pairs"] == 2
    assert summary["margin_95ci"] == pytest.approx([-19.6, 19.6])
    assert summary["score_rate_95ci"] == [0.0, 1.0]
