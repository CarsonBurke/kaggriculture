"""Cross-run reporting must bind compute counts to the policy actually evaluated."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def module():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "structural_summary_test", scripts / "summarize_structural_campaign.py"
        )
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.remove(str(scripts))


def test_culled_policy_and_older_checkpoint_keep_their_actual_update_counts(tmp_path):
    rows = [
        {"iteration": 1, "iteration_seconds": 10, "elapsed_hours": 0.1, "actor_updates": 0},
        {"iteration": 2, "iteration_seconds": 10, "elapsed_hours": 0.2, "actor_updates": 29},
        {
            "iteration": 3,
            "iteration_seconds": 10,
            "elapsed_hours": 0.3,
            "actor_updates": 17,
            "architecture_panel_state": {"culled": True, "best_checkpoint": "checkpoint-000002.pt"},
            "architecture_panel": {"score_rate": 0.5},
        },
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    summary = module().read_training(tmp_path)
    assert summary["status"] == "culled"
    assert summary["actor_updates"] == 46
    assert summary["milestones"][2] == {
        "actor_updates": 29,
        "measured_hours": 20 / 3600,
        "complete_prefix": True,
    }
    assert summary["panels"][0]["actor_updates"] == 46
    report = {
        "arms": {
            "candidate": {
                "training": summary,
                "evaluated_budget": summary["milestones"][2],
                "evaluations": {},
            }
        }
    }
    text = module().markdown(report)
    assert "| candidate | 29 | 0.01 | pending | pending |" in text
    assert "| candidate | 46 |" not in text


def test_resumed_session_clock_cannot_erase_measured_compute(tmp_path):
    rows = [
        {"iteration": 1, "iteration_seconds": 360, "elapsed_hours": 0.1, "actor_updates": 29},
        {
            "iteration": 2,
            "iteration_seconds": 180,
            "elapsed_hours": 0.05,
            "actor_updates": 20,
            "architecture_panel_seconds": 36,
        },
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    training = module().read_training(tmp_path)
    assert training["measured_hours"] == pytest.approx(0.16)
    assert training["actor_updates"] == 49
    assert training["last_session_elapsed_hours"] == 0.05


def test_partial_journal_does_not_claim_complete_update_budget(tmp_path):
    row = {"iteration": 20, "iteration_seconds": 36, "elapsed_hours": 0.01, "actor_updates": 10}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row) + '\n{"iteration":21')
    training = module().read_training(tmp_path)
    assert training["actor_updates"] is None and training["measured_hours"] is None
    assert training["observed_actor_updates"] == 10
    assert training["partial_tail"]
    text = module().markdown({"arms": {"partial": {"training": training, "evaluations": {}}}})
    assert "| partial | unknown |" in text
    assert "partial journal" in text and "incomplete final row" in text


@pytest.mark.parametrize("iterations", [[1, 1], [2, 1]])
def test_duplicate_or_regressing_journals_are_rejected(tmp_path, iterations):
    rows = [{"iteration": i, "iteration_seconds": 10, "actor_updates": 29} for i in iterations]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="duplicate or out-of-order"):
        module().read_training(tmp_path)


def test_complete_corrupted_journal_record_is_not_an_in_progress_tail(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"iteration":1\n')
    with pytest.raises(ValueError, match="invalid training journal row"):
        module().read_training(tmp_path)


def _entry(tmp_path):
    report = module()
    checkpoint = tmp_path / "checkpoint-000002.pt"
    checkpoint.write_bytes(b"actual trained checkpoint")
    clone = tmp_path / "bc.pt"
    clone.write_bytes(b"matched clone")

    def artifact(path, iteration):
        return {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "iteration": iteration,
            "architecture": "entity-attention",
            "model_config": {"model_dim": 96},
            "source_identity": {"sha256": "abc"},
        }

    evaluation = {
        "seed_start": 1,
        "games_per_opponent": 2,
        "sampling_seed": 10,
        "precision": "CUDA BF16",
        "seat_protocol": "seed modulo 2",
        "artifacts": {"ppo": artifact(checkpoint, 2), "bc": artifact(clone, 0)},
    }
    entry = {
        "run": str(tmp_path),
        "evaluations": {"sampled": evaluation, "argmax": copy.deepcopy(evaluation)},
        "training": {
            "milestones": {2: {"actor_updates": 29, "measured_hours": 0.1, "complete_prefix": True}}
        },
    }
    command = ["--run-dir", str(tmp_path), "--init-actor-from", str(clone)]
    return report, entry, command


def test_both_modes_bind_to_declared_immutable_checkpoint_and_clone(tmp_path):
    report, entry, command = _entry(tmp_path)
    report.bind_evaluations(entry, command, arm="candidate")
    assert entry["evaluated_budget"]["iteration"] == 2
    assert entry["evaluated_budget"]["actor_updates"] == 29
    assert entry["evaluated_checkpoint"].endswith("checkpoint-000002.pt")


@pytest.mark.parametrize(
    "artifact,field,value",
    [
        ("bc", "sha256", "foreign clone"),
        ("ppo", "iteration", 3),
        ("ppo", "architecture", "other"),
        ("ppo", "model_config", {"model_dim": 48}),
    ],
)
def test_decoding_metadata_must_describe_same_artifact(tmp_path, artifact, field, value):
    report, entry, command = _entry(tmp_path)
    entry["evaluations"]["argmax"]["artifacts"][artifact][field] = value
    with pytest.raises(ValueError, match="decoding panels disagree"):
        report.bind_evaluations(entry, command, arm="candidate")


@pytest.mark.parametrize("artifact", ["ppo", "bc"])
def test_foreign_run_or_clone_rejected_even_if_all_modes_agree(tmp_path, artifact):
    report, entry, command = _entry(tmp_path)
    for evaluation in entry["evaluations"].values():
        evaluation["artifacts"][artifact]["sha256"] = "foreign bytes"
    with pytest.raises(ValueError, match="does not match"):
        report.bind_evaluations(entry, command, arm="candidate")


def test_decoding_modes_must_use_the_same_evaluation_protocol(tmp_path):
    report, entry, command = _entry(tmp_path)
    entry["evaluations"]["argmax"]["seed_start"] = 100
    with pytest.raises(ValueError, match="evaluation seed_start"):
        report.bind_evaluations(entry, command, arm="candidate")


def test_game_rows_must_match_summary_and_bc_seed_pairs(tmp_path):
    report = module()
    panel = {
        "summary": {"score_rate": 0.5},
        "games": [
            {"seed": 1, "seat": 1, "score": 0},
            {"seed": 2, "seat": 0, "score": 1},
        ],
    }
    artifact = {"panels": {op: copy.deepcopy(panel) for op in report.OPPONENTS}}
    evaluation = {
        "complete": True,
        "decoding": "sampled",
        "games_per_opponent": 2,
        "seed_start": 1,
        "artifacts": {
            "bc": artifact,
            "ppo": copy.deepcopy(artifact),
        },
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(evaluation))
    report.read_evaluation(path, "sampled")
    wrong = copy.deepcopy(evaluation)
    wrong["artifacts"]["ppo"]["panels"][report.OPPONENTS[0]]["summary"]["score_rate"] = 0.9
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="summary disagrees"):
        report.read_evaluation(path, "sampled")
    wrong = copy.deepcopy(evaluation)
    wrong["artifacts"]["ppo"]["panels"][report.OPPONENTS[0]]["games"][0]["seed"] = 3
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="different seed pairs"):
        report.read_evaluation(path, "sampled")
    evaluation["games_per_opponent"] = 4
    path.write_text(json.dumps(evaluation))
    with pytest.raises(ValueError, match="different seed pairs"):
        report.read_evaluation(path, "sampled")
