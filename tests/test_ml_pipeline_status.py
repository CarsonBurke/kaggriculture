from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _script():
    path = Path(__file__).parents[1] / "scripts" / "ml_pipeline_status.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_ml_pipeline_status", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_infrastructure_failures_are_self_healed() -> None:
    module = _script()

    assert module._is_transient_failure(
        {"state": "failed", "stateReason": "launch failed: Transport endpoint is not connected"}
    )
    assert not module._is_transient_failure(
        {"state": "failed", "stateReason": "command exited with code 1"}
    )
    assert not module._is_transient_failure(
        {"state": "failed", "stateReason": "launch failed: executable does not exist"}
    )
    assert not module._is_transient_failure({"state": "running"})


def test_compact_report_summary_uses_only_completed_batches_and_latest_iteration(
    tmp_path: Path,
) -> None:
    module = _script()
    report = tmp_path / "benchmark.jsonl"
    records = [
        {"event": "configuration"},
        {
            "event": "iteration",
            "phase": "steady_state",
            "self_play_games": 112,
            "total_seconds": 80.0,
            "rollout_seconds": 32.0,
            "update_seconds": 48.0,
            "peak_cuda_bytes": 2**30,
        },
        {
            "event": "batch_summary",
            "self_play_games": 112,
            "steady_total_seconds_median": 80.0,
            "steady_iterations_per_hour_median": 45.0,
        },
    ]
    report.write_text("".join(json.dumps(record) + "\n" for record in records))

    rendered = "\n".join(module._render_report(report))

    assert "3 records, last=batch_summary" in rendered
    assert "games=112: 80.00s/iteration, 45.00/hour" in rendered
    assert "latest: games=112 steady_state" in rendered


def test_empty_report_is_rendered_without_crashing(tmp_path: Path) -> None:
    module = _script()
    report = tmp_path / "benchmark.jsonl"
    report.touch()

    assert module._render_report(report) == [f"report {report}: empty"]


def test_status_queries_each_unchanged_job_once(monkeypatch) -> None:
    module = _script()
    calls = 0

    def fake_job(job_id: int) -> dict:
        nonlocal calls
        calls += 1
        return {"id": job_id, "name": "done", "state": "succeeded"}

    monkeypatch.setattr(module, "_job", fake_job)
    monkeypatch.setattr(sys, "argv", ["ml_pipeline_status.py", "7"])

    module.main()

    assert calls == 1


def test_watch_succeeds_after_rendering_success(monkeypatch, capsys) -> None:
    module = _script()
    monkeypatch.setattr(
        module,
        "_job",
        lambda job_id: {"id": job_id, "name": "pipeline", "state": "succeeded"},
    )
    monkeypatch.setattr(sys, "argv", ["ml_pipeline_status.py", "--watch", "7"])

    assert module.main() == 0
    assert capsys.readouterr().out == "job 7 pipeline: succeeded\n"


def test_watch_fails_after_rendering_failure(monkeypatch, capsys) -> None:
    module = _script()
    monkeypatch.setattr(
        module,
        "_job",
        lambda job_id: {
            "id": job_id,
            "name": "pipeline",
            "state": "failed",
            "stateReason": "command exited with code 1",
        },
    )
    monkeypatch.setattr(sys, "argv", ["ml_pipeline_status.py", "--watch", "7"])

    assert module.main() == 1
    assert capsys.readouterr().out == "job 7 pipeline: failed (command exited with code 1)\n"


def test_watch_fails_after_rendering_cancellation(monkeypatch, capsys) -> None:
    module = _script()
    monkeypatch.setattr(
        module,
        "_job",
        lambda job_id: {"id": job_id, "name": "pipeline", "state": "cancelled"},
    )
    monkeypatch.setattr(sys, "argv", ["ml_pipeline_status.py", "--watch", "7"])

    assert module.main() == 1
    assert capsys.readouterr().out == "job 7 pipeline: cancelled\n"


def test_watch_retries_until_healing_is_exhausted(monkeypatch, capsys) -> None:
    module = _script()
    retry_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        module,
        "_job",
        lambda job_id: {
            "id": job_id,
            "name": "pipeline",
            "state": "failed",
            "stateReason": "runner lost",
        },
    )
    monkeypatch.setattr(
        module,
        "_command",
        lambda *arguments: retry_calls.append(arguments) or {},
    )
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ml_pipeline_status.py",
            "--watch",
            "--heal",
            "--max-heals",
            "2",
            "--interval",
            "0.01",
            "7",
        ],
    )

    assert module.main() == 1
    assert retry_calls == [
        ("mlq", "retry", "7", "--json"),
        ("mlq", "retry", "7", "--json"),
    ]
    assert capsys.readouterr().out == "job 7 pipeline: failed (runner lost)\n"
