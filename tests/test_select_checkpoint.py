from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _selection_script(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "select_checkpoint.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_select_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation(label: str, scores: list[float], margins: list[float], *, valid: bool = True):
    return {
        "valid_for_selection": valid,
        "opponent": label,
        "opponent_label": label,
        "summary": {
            "score_rate": sum(scores) / len(scores),
            "score_rate_95ci": [0.0, 1.0],
            "mean_margin": sum(margins) / len(margins),
            "margin_95ci": [-1.0, 1.0],
            "seed_cluster_statistics": [
                {"seed": index, "score_rate": score, "mean_margin": margin}
                for index, (score, margin) in enumerate(zip(scores, margins, strict=True))
            ],
        },
    }


def test_panel_summary_clusters_seeds_across_fixed_opponents(monkeypatch) -> None:
    module = _selection_script(monkeypatch)

    panel = module.summarize_panel(
        [
            _evaluation("strong", [1.0, 0.0, 1.0, 0.0], [10.0, -2.0, 8.0, -4.0]),
            _evaluation("style", [0.5, 0.5, 1.0, 0.0], [2.0, 2.0, 6.0, -2.0]),
        ]
    )

    assert panel["paired_seed_clusters"] == 4
    assert panel["panel_score_rate"] == pytest.approx(0.5)
    assert panel["panel_mean_margin"] == pytest.approx(2.5)
    assert panel["worst_opponent_score_rate"] == pytest.approx(0.5)
    assert len(panel["opponent_summaries"]) == 2


def test_panel_summary_rejects_invalid_or_misaligned_evidence(monkeypatch) -> None:
    module = _selection_script(monkeypatch)
    with pytest.raises(ValueError, match="invalid"):
        module.summarize_panel([_evaluation("broken", [0.5], [0.0], valid=False)])

    first = _evaluation("first", [0.5, 1.0], [0.0, 1.0])
    second = _evaluation("second", [0.5, 1.0], [0.0, 1.0])
    second["summary"]["seed_cluster_statistics"][1]["seed"] = 7
    with pytest.raises(ValueError, match="identical"):
        module.summarize_panel([first, second])


def test_ranking_prefers_confident_panel_strength_before_iteration(monkeypatch) -> None:
    module = _selection_script(monkeypatch)
    stronger = {
        "iteration": 10,
        "panel": {
            "panel_score_rate_95ci": [0.61, 0.80],
            "worst_opponent_score_rate": 0.55,
            "panel_margin_95ci": [2.0, 8.0],
        },
    }
    newer_but_weaker = {
        "iteration": 100,
        "panel": {
            "panel_score_rate_95ci": [0.60, 0.90],
            "worst_opponent_score_rate": 0.90,
            "panel_margin_95ci": [100.0, 200.0],
        },
    }

    assert module._ranking_key(stronger) > module._ranking_key(newer_but_weaker)


def test_atomic_promotion_requires_the_evaluated_checkpoint_digest(monkeypatch, tmp_path) -> None:
    module = _selection_script(monkeypatch)
    source = tmp_path / "checkpoint.pt"
    destination = tmp_path / "best.pt"
    source.write_bytes(b"evaluated")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    assert module._copy_atomic(source, destination, expected) == expected
    assert destination.read_bytes() == b"evaluated"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after evaluation"):
        module._copy_atomic(source, destination, expected)
    assert destination.read_bytes() == b"evaluated"
