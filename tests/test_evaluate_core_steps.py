"""Boundary selection must never substitute a later, better-trained endpoint."""

import json

import pytest

from scripts.evaluate_core_steps import select_checkpoints


def write_run(tmp_path, rows, checkpoints):
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for iteration in checkpoints:
        (tmp_path / f"checkpoint-{iteration:06d}.pt").write_bytes(b"fake checkpoint")
    return tmp_path


def row(iteration, wave, updates=2, warmup=False):
    return {
        "iteration": iteration,
        "states": 100,
        "actor_updates": updates,
        "critic_warmup_active": warmup,
        "architecture_panel_state": {"actor_waves": wave},
    }


def test_exact_boundary_and_separate_warmup_accounting(tmp_path):
    run = write_run(tmp_path, [row(1, 0, 0, True), row(2, 1), row(3, 2)], [2, 3])
    result = select_checkpoints(run, (1, 2))["milestones"]["2"]
    assert result["iteration"] == 3
    assert result["cumulative"] == {
        "actor_updates": 4,
        "states": 300,
        "warmup_iterations": 1,
        "warmup_states": 100,
    }


def test_missing_boundary_does_not_use_later_checkpoint(tmp_path):
    run = write_run(tmp_path, [row(1, 25), row(2, 26)], [2])
    assert select_checkpoints(run, (25,))["milestones"]["25"] is None


def test_partial_journal_does_not_claim_cumulative_accounting(tmp_path):
    run = write_run(tmp_path, [row(26, 25)], [26])
    result = select_checkpoints(run, (25,))["milestones"]["25"]
    assert result["cumulative"] is None
    assert not result["accounting_complete"]


def test_duplicate_iterations_rejected(tmp_path):
    run = write_run(tmp_path, [row(1, 1), row(1, 1)], [1])
    with pytest.raises(ValueError, match="duplicate"):
        select_checkpoints(run)


def test_absent_run_records_all_missing_milestones(tmp_path):
    result = select_checkpoints(tmp_path, (25, 50))
    assert result["milestones"] == {"25": None, "50": None}
