from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _script():
    path = Path(__file__).parents[1] / "scripts" / "summarize_ppo_run.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_summarize_ppo_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"arguments": {"iterations": 100}}), encoding="utf-8"
    )
    rows = [
        {
            "iteration": 1,
            "actor_updates": 0,
            "actor_minibatches_intended": 0,
            "money_mean": 10.0,
            "rollout_entropy": 0.3,
            "iteration_seconds": 8.0,
            "rollout_seconds": 3.0,
            "update_seconds": 5.0,
        },
        {
            "iteration": 2,
            "actor_updates": 4,
            "actor_minibatches_intended": 4,
            "money_mean": 20.0,
            "score_rate": 0.5,
            "rollout_entropy": 0.2,
            "first_minibatch_approx_kl": 0.001,
            "max_approx_kl": 0.002,
            "kl_early_stop": 0,
            "iteration_seconds": 12.0,
            "rollout_seconds": 4.0,
            "update_seconds": 8.0,
        },
        {
            "iteration": 3,
            "actor_updates": 3,
            "actor_minibatches_intended": 4,
            "money_mean": 30.0,
            "score_rate": 0.75,
            "rollout_entropy": 0.1,
            "first_minibatch_approx_kl": 0.003,
            "max_approx_kl": 0.04,
            "kl_early_stop": 1,
            "iteration_seconds": 16.0,
            "rollout_seconds": 6.0,
            "update_seconds": 10.0,
        },
    ]
    (path / "metrics.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
    )


def test_summary_reports_learning_progress_and_recent_timings(tmp_path: Path) -> None:
    module = _script()
    run = tmp_path / "run"
    _write_run(run)

    summary = module.summarize_run(run, recent=2)

    assert summary == {
        "run": str(run),
        "iteration": 3,
        "iterations_target": 100,
        "actor_iterations": 2,
        "actor_updates": 7,
        "warmup_iterations": 1,
        "warmup_money_mean": 10.0,
        "warmup_entropy_mean": 0.3,
        "best_actor_money": 30.0,
        "best_actor_money_iteration": 3,
        "latest_money": 30.0,
        "latest_score": 0.75,
        "latest_entropy": 0.1,
        "latest_actor_updates": 3,
        "latest_actor_minibatches": 4,
        "latest_first_kl": 0.003,
        "latest_max_kl": 0.04,
        "latest_kl_early_stop": 1,
        "latest_iteration_seconds": 16.0,
        "latest_rollout_seconds": 6.0,
        "latest_update_seconds": 10.0,
        "recent_actor_window": 2,
        "recent_iteration_seconds_median": 14.0,
        "recent_rollout_seconds_median": 5.0,
        "recent_update_seconds_median": 9.0,
    }


def test_summary_accepts_metrics_path_and_formats_one_line(tmp_path: Path) -> None:
    module = _script()
    run = tmp_path / "run"
    _write_run(run)

    summary = module.summarize_run(run / "metrics.jsonl")

    assert module.format_summary(summary) == (
        "run=run iter=3/100 actor_iters=2 actor_updates=7 money=30 "
        "best_money=30@3 entropy=0.1 kl=0.003/0.04 minibatches=3/4 "
        "seconds=16 recent_update_median=9"
    )


def test_summary_rejects_empty_journal_and_nonpositive_window(tmp_path: Path) -> None:
    module = _script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no metrics"):
        module.summarize_run(run)
    with pytest.raises(ValueError, match="recent window"):
        module.summarize_run(run, recent=0)
