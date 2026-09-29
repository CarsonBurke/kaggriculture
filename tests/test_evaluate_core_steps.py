"""Boundary selection must never substitute a later, better-trained endpoint."""

import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/evaluate_core_steps.py"
_spec = importlib.util.spec_from_file_location("evaluate_core_steps", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
select_checkpoints = _module.select_checkpoints


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


def test_final_selects_only_the_clean_endpoint_under_its_own_label(tmp_path):
    run = write_run(tmp_path, [row(1, 0, 0, True), row(2, 1), row(3, 2)], [2, 3])
    milestones = select_checkpoints(run, (1,), final=True)["milestones"]
    assert milestones["1"]["iteration"] == 2
    assert milestones["final"]["iteration"] == 3
    assert milestones["final"]["actor_waves"] == 2
    assert milestones["final"]["cumulative"]["actor_updates"] == 4
    assert "final" not in select_checkpoints(run, (1,))["milestones"]


def test_final_is_missing_rather_than_an_earlier_checkpoint(tmp_path):
    run = write_run(tmp_path, [row(1, 1), row(2, 2)], [1])
    assert select_checkpoints(run, (1,), final=True)["milestones"]["final"] is None


def test_an_endpoint_on_a_milestone_is_both(tmp_path):
    run = write_run(tmp_path, [row(1, 1), row(2, 2)], [1, 2])
    milestones = select_checkpoints(run, (2,), final=True)["milestones"]
    assert milestones["2"] == milestones["final"]


def test_each_arm_is_verified_against_the_source_it_trained_under(tmp_path):
    def snapshot(name, sha):
        root = tmp_path / name
        root.mkdir()
        (root / ".source-identity.json").write_text(json.dumps({"sha256": sha}))
        return str(root)

    manifest = {
        "source": snapshot("current", "c" * 64),
        "variants": {
            "new-arm": {"ppo": []},
            "reference": {"ppo": [], "source": snapshot("earlier", "e" * 64)},
        },
    }
    assert _module.training_source_digests(manifest) == {
        "new-arm": "c" * 64,
        "reference": "e" * 64,
    }
