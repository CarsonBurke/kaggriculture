"""Model-free aggregation checks for the matched architecture report."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ARMS = ("control", "critic-source", "writeback", "tile-bias", "hardness-league")


@pytest.fixture
def summarizer(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "summarize_architecture_campaign", scripts / "summarize_architecture_campaign.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _league_row(games, score, *, role="hardness"):
    return {
        "league_opponent_builtin_scripted-v27_games": games,
        "league_opponent_builtin_scripted-v27_score_rate": score,
        "league_opponent_builtin_scripted-v27_category": "builtin",
        "league_opponent_builtin_scripted-v27_role": role,
        "league_score_rate": 0.99,  # An aggregate must not be counted as another opponent.
    }


def test_realized_league_is_game_weighted_without_independence_intervals(summarizer):
    result = summarizer.league_summary([_league_row(5, 1), _league_row(45, 0)])
    expected = {"games": 50, "score_rate": 0.1}
    assert result == {
        "all": expected,
        "category:builtin": expected,
        "role:hardness": expected,
        "opponent:builtin_scripted-v27": expected,
    }
    old_row = _league_row(8, 0.5)
    del old_row["league_opponent_builtin_scripted-v27_role"]
    assert summarizer.league_summary([old_row])["role:stratified"] == {
        "games": 8,
        "score_rate": 0.5,
    }


@pytest.fixture
def campaign(summarizer, tmp_path, monkeypatch):
    manifest = {"jobs": {}}
    for index, arm in enumerate(ARMS):
        run = tmp_path / "runs" / arm
        run.mkdir(parents=True)
        manifest["jobs"][f"learn-{arm}"] = {
            "command": ["python", "train_ppo.py", "--run-dir", str(run)],
            "submission": {"id": index + 1},
        }
        manifest["jobs"][f"evaluate-{arm}"] = {"submission": {"id": index + 101}}
        rows = [{"event": "configuration", "iteration": 0}]
        for iteration in range(1, 53):
            rows.append(
                {
                    "iteration": iteration,
                    "elapsed_hours": iteration / 100,
                    "iteration_seconds": 100 if iteration <= 2 else iteration,
                    "actor_updates": 0 if iteration <= 2 else 29,
                    **_league_row(64, float(iteration <= 2)),
                }
            )
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        panels = {}
        for opponent_index, opponent in enumerate(summarizer.OPPONENTS):
            games = []
            for seed_index in range(256):
                # Candidate improvements on one opponent are exactly cancelled
                # by regressions on the other opponent on the SAME map seed.
                score = 0.5 if arm == "control" else float((seed_index + opponent_index) % 2)
                games.append(
                    {
                        "seed": 4_501_000 + seed_index,
                        "seat": seed_index % 2,
                        "score": score,
                        "money": 2 * score,
                        "opponent_money": 1.0,
                    }
                )
            panels[opponent] = {"summary": {"score_rate": 0.5, "games": 256}, "games": games}
        result = {
            "complete": True,
            "decoding": "argmax",
            "artifacts": {
                label: {
                    "iteration": 0 if label == "bc" else 52,
                    "sha256": f"{index:064d}",
                    "overall_score_rate": 0.5,
                    "panels": panels,
                }
                for label in ("bc", "ppo")
            },
            "comparisons_to_first": {"ppo": {"source": f"{arm}-paired-bc-comparison"}},
        }
        (tmp_path / f"{arm}-evaluation.json").write_text(json.dumps(result))
    benchmark = [
        {"event": "iteration", "total_seconds": 100},
        {"event": "batch_summary", "steady_total_seconds_median": 12.0},
        {"event": "benchmark_complete", "completed": True},
    ]
    (tmp_path / "control-benchmark.jsonl").write_text(
        "\n".join(json.dumps(row) for row in benchmark) + "\n"
    )
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(json.dumps(manifest))

    def status(command, *, text):
        assert command[:2] == ["mlq", "show"] and command[-1] == "--json"
        assert text
        return json.dumps({"id": int(command[2]), "state": "succeeded"})

    monkeypatch.setattr(summarizer.subprocess, "check_output", status)
    monkeypatch.setattr(sys, "argv", ["summarize_architecture_campaign.py", str(manifest_path)])
    return tmp_path


def test_summary_retains_seed_clusters_and_separates_training_from_panel_skill(
    summarizer, campaign
):
    summarizer.main()
    report = json.loads((campaign / "outcome-summary.json").read_text())
    assert report["campaign_sha256"] == hashlib.sha256(
        (campaign / "campaign.json").read_bytes()
    ).hexdigest()
    assert report["analysis_script_sha256"] == hashlib.sha256(
        Path(summarizer.__file__).read_bytes()
    ).hexdigest()
    for arm in ARMS:
        entry = report["arms"][arm]
        training = entry["training"]
        assert training["last_iteration"] == 52
        assert training["actor_active_waves"] == 50
        assert training["actor_updates"] == 50 * 29
        assert training["actor_active_iteration_seconds_median"] == 27.5
        assert training["league_all"]["all"] == {"games": 52 * 64, "score_rate": 2 / 52}
        assert training["league_actor_active"]["all"] == {"games": 50 * 64, "score_rate": 0}
        assert training["league_last_50"]["all"] == {"games": 50 * 64, "score_rate": 0}
        assert entry["evaluation"]["ppo"]["overall_score_rate"] == 0.5
        assert entry["paired_to_bc"] == {"source": f"{arm}-paired-bc-comparison"}
        assert entry["learning_evidence"]["status"] == "evaluated_actor_learning"
        assert entry["learning_evidence"]["eligible_for_learning_comparison"] is True
        if arm == "control":
            assert "paired_to_control" not in entry
            continue
        comparison = entry["paired_to_control"]
        assert entry["paired_to_control_qualification"]["both_arms_learning_eligible"] is True
        assert comparison["seed_clusters"] == 256
        assert "training-seed" in comparison["scope"]
        assert comparison["panels"]["overall"]["score"] == {
            "difference": 0,
            "ci95": [0, 0],
        }
        assert comparison["panels"]["starter"]["score"]["ci95"][0] < 0
        assert comparison["panels"]["starter"]["score"]["ci95"][1] > 0
    control = report["arms"]["control"]
    hardness = report["arms"]["hardness-league"]
    assert control["benchmark_complete"] is True
    assert (
        hardness["benchmark"]
        == control["benchmark"]
        == [{"event": "batch_summary", "steady_total_seconds_median": 12.0}]
    )


@pytest.mark.parametrize("invalid", [{"complete": False}, {"decoding": "sampled"}])
def test_primary_summary_rejects_unfinished_or_sampled_evaluation(summarizer, campaign, invalid):
    path = campaign / "control-evaluation.json"
    evaluation = json.loads(path.read_text())
    path.write_text(json.dumps(evaluation | invalid))
    with pytest.raises(ValueError, match="invalid primary evaluation"):
        summarizer.main()
    assert not (campaign / "outcome-summary.json").exists()


def test_primary_comparison_rejects_a_candidate_with_different_seed_order(summarizer, campaign):
    path = campaign / "critic-source-evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["artifacts"]["ppo"]["panels"]["starter"]["games"].reverse()
    path.write_text(json.dumps(evaluation))
    with pytest.raises(ValueError, match="identical seeds and seats"):
        summarizer.main()
    assert not (campaign / "outcome-summary.json").exists()


def test_summary_retains_supplemental_diagnostic_results_and_job_identity(summarizer, campaign):
    manifest_path = campaign / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["supplemental_jobs"] = {
        "common-policy-fit-fixed": {"submission": {"id": 7680}},
    }
    manifest_path.write_text(json.dumps(manifest))
    fit = {"complete": True, "source_identity": {"sha256": "diagnostic-source"}, "fits": {}}
    attention = {"complete": True, "source_identity": {"sha256": "attention-source"}}
    (campaign / "common-policy-fit.json").write_text(json.dumps(fit))
    (campaign / "critic-source-attention.json").write_text(json.dumps(attention))
    summarizer.main()
    report = json.loads((campaign / "outcome-summary.json").read_text())
    assert report["jobs"]["common-policy-fit-fixed"]["id"] == 7680
    assert report["common_policy_fit"] == fit
    assert report["source_attention"] == attention


def test_zero_update_time_budget_exit_retains_scores_without_claiming_actor_learning(
    summarizer, campaign
):
    path = campaign / "runs" / "control" / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if "actor_updates" in row:
            row.update(actor_updates=0, critic_warmup_active=1)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summarizer.main()
    report = json.loads((campaign / "outcome-summary.json").read_text())
    control = report["arms"]["control"]
    evidence = control["learning_evidence"]
    assert evidence["status"] == "no_actor_learning"
    assert evidence["eligible_for_learning_comparison"] is False
    assert evidence["actor_updates_observed"] == 0
    assert evidence["reasons"] == [
        "no_actor_updates",
        "actor_not_released_from_critic_warmup_before_stop",
    ]
    assert control["evaluation"]["ppo"]["overall_score_rate"] == 0.5
    assert control["paired_to_bc"] == {"source": "control-paired-bc-comparison"}
    assert evidence["completed_gameplay_evaluation_available"] is True
    comparison = report["arms"]["critic-source"]["paired_to_control_qualification"]
    assert comparison["candidate_learning_eligible"] is True
    assert comparison["reference_learning_eligible"] is False
    assert comparison["both_arms_learning_eligible"] is False
    assert comparison["raw_gameplay_comparison"] == "valid_matched_panel_comparison"


@pytest.mark.parametrize(
    ("training_state", "evaluation_state", "expected"),
    [
        ("failed", "cancelled", "unsuccessful_queue"),
        ("timed_out", "queued", "unsuccessful_queue"),
        ("running", "queued", "incomplete_queue"),
        ("succeeded", "failed", "unsuccessful_queue"),
        ("succeeded", None, "incomplete_queue"),
    ],
)
def test_non_successful_jobs_cannot_qualify_existing_evaluation_as_learning_evidence(
    summarizer, training_state, evaluation_state, expected
):
    entry = {
        "training": {"actor_updates": 29, "last_iteration": 3},
        "evaluation": {"ppo": {"iteration": 3}},
    }
    evidence = summarizer.learning_evidence(
        entry,
        training_job={"state": training_state},
        evaluation_job={"state": evaluation_state},
    )
    assert evidence["status"] == expected
    assert evidence["actor_updates_observed"] == 29
    assert evidence["eligible_for_learning_comparison"] is False
    assert evidence["reasons"]


@pytest.mark.parametrize(
    ("entry", "expected_reason"),
    [
        ({"evaluation": {"ppo": {"iteration": 3}}}, "missing_training_metrics"),
        (
            {"training": {"actor_updates": 29, "last_iteration": 3}},
            "missing_completed_evaluation",
        ),
        (
            {
                "training": {"actor_updates": 29, "last_iteration": 3},
                "evaluation": {"ppo": {"iteration": 2}},
            },
            "evaluated_checkpoint_does_not_match_last_training_iteration",
        ),
        (
            {
                "training": {"actor_updates": None, "last_iteration": 3},
                "evaluation": {"ppo": {"iteration": 3}},
            },
            "missing_actor_update_counts",
        ),
    ],
)
def test_missing_or_inconsistent_training_evidence_never_qualifies(
    summarizer, entry, expected_reason
):
    evidence = summarizer.learning_evidence(
        entry, training_job={"state": "succeeded"}, evaluation_job={"state": "succeeded"}
    )
    assert evidence["eligible_for_learning_comparison"] is False
    assert expected_reason in evidence["reasons"]


def test_external_reference_panel_does_not_imply_control_actor_learning(
    summarizer, campaign, monkeypatch
):
    manifest_path = campaign / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    # Stand in for an extension-only manifest: the reference panel is present,
    # but its training jobs and metrics are deliberately outside this campaign.
    manifest["jobs"] = {
        label: job for label, job in manifest["jobs"].items() if label.endswith("critic-source")
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_architecture_campaign.py",
            str(manifest_path),
            "--reference-evaluation",
            str(campaign / "control-evaluation.json"),
        ],
    )
    summarizer.main()
    report = json.loads((campaign / "outcome-summary.json").read_text())
    assert report["reference"]["learning_evidence"]["status"] == "unknown_from_evaluation_alone"
    candidate = report["arms"]["critic-source"]
    assert candidate["learning_evidence"]["eligible_for_learning_comparison"] is True
    assert candidate["paired_to_control"]["seed_clusters"] == 256
    qualification = candidate["paired_to_control_qualification"]
    assert qualification["raw_gameplay_comparison"] == "valid_matched_panel_comparison"
    assert qualification["candidate_learning_eligible"] is True
    assert qualification["reference_learning_eligible"] is None
    assert qualification["both_arms_learning_eligible"] is False
