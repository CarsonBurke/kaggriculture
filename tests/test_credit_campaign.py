"""Model-free gates for a costly full-budget experiment and its evidence report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kaggriculture.production import production_ppo_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(autouse=True)
def script_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def argument(command, flag):
    assert command.count(flag) == 1
    return command[command.index(flag) + 1]


def test_credit_commands_use_bounded_budget_and_only_change_named_mechanisms(tmp_path):
    module = load_script("queue_credit_campaign")
    root = tmp_path
    source = root / "source"
    output = root / "experiment"
    actor = root / "bc.pt"
    commands = module.build_commands(root, source, output, actor, list(module.ARMS))
    for arm, values in commands.items():
        train = values["train"]
        for flag, value in {
            "--iterations": "500",
            "--games": "128",
            "--league-games": "64",
            "--minibatch-size": str(
                production_ppo_config(update_compile_mode="default")["minibatch_size"]
            ),
            "--episode-steps": "720",
            "--league-selection": "hardness",
            "--critic-source-read": "true",
            "--init-actor-from": str(actor),
            "--max-hours": str(module.TRAINER_HOURS),
        }.items():
            assert argument(train, flag) == value
        assert train[1] == str(source / "scripts/train_ppo.py")
        assert "--autocull" in train and "--external-eval" not in train
        assert argument(train, "--architecture-panel") == "0"
        assert argument(train, "--architecture") == "entity-attention"
        assert argument(train, "--actor-gae-lambda") == (
            "1.0" if arm == "monte-carlo" else str(module.HISTORICAL_ACTOR_GAE_LAMBDA)
        )
        assert argument(values["benchmark"], "--actor-gae-lambda") == argument(
            train, "--actor-gae-lambda"
        )
        assert argument(train, "--critic-architecture") == (
            "economic" if arm == "economic" else "entity"
        )
        for kind in ("latent", "value"):
            assert argument(train, f"--structured-critic-{kind}-coefficient") == (
                "0.0" if arm == "no-nextlat" else "1.0"
            )
            assert argument(values["benchmark"], f"--structured-critic-{kind}-coefficient") == (
                argument(train, f"--structured-critic-{kind}-coefficient")
            )
        assert argument(values["benchmark"], "--auxiliary-mode") == (
            "off" if arm == "no-nextlat" else "enabled"
        )
        for mode, evaluation in values["evaluations"].items():
            assert argument(evaluation, "--decoding") == mode
            assert len([a for a in evaluation if a == "--artifact"]) == 2


@pytest.mark.parametrize("submit", [False, True])
def test_campaign_gates_learning_and_evaluates_successful_runs(tmp_path, monkeypatch, submit):
    module = load_script("queue_credit_campaign")
    script = tmp_path / "scripts/queue_credit_campaign.py"
    monkeypatch.setattr(module, "__file__", str(script))
    (tmp_path / "tests").mkdir()
    for name in ("conftest.py", "test_entity.py", "test_economic_critic.py"):
        (tmp_path / "tests" / name).write_text("# frozen contract\n")
    actor = tmp_path / "baseline-bc.pt"
    actor.write_bytes(b"opaque actor bytes; must never be loaded by planner")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "initial_actor": {"path": str(actor), "sha256": module.file_sha256(actor)},
                "environment": {"PYTHONPATH": "old", "PYTHONDONTWRITEBYTECODE": "1"},
                "plan": {"autocull": "existing policy"},
                "source_identity": {"sha256": "b" * 64},
                "jobs": {
                    f"evaluate-{mode}": {"submission": {"id": 8000 + i}}
                    for i, mode in enumerate(("argmax", "sampled"))
                }
                | {"train": {"command": ["train", "--run-dir", str(tmp_path / "baseline-run")]}},
            }
        )
    )
    monkeypatch.setattr(module, "source_identity", lambda: {"sha256": "a" * 64})
    monkeypatch.setattr(module, "freeze_source", lambda path: None)
    queued = []

    def check_output(command, *, text):
        assert submit and text
        queued.append(command)
        return json.dumps({"id": 9000 + len(queued), "state": "queued"})

    monkeypatch.setattr(module.subprocess, "check_output", check_output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--baseline",
            str(baseline),
            "--name",
            "credit-test",
            *(["--submit"] if submit else []),
        ],
    )
    module.main()
    manifest = json.loads((tmp_path / "artifacts/probes/credit-test/campaign.json").read_text())
    jobs = manifest["jobs"]
    validation = jobs["correctness"]["submission"]["id"]
    for arm in module.ARMS:
        gate, train = jobs[f"gate-{arm}"], jobs[f"train-{arm}"]
        assert gate["after_success"] == [validation]
        assert train["after_success"] == [gate["submission"]["id"]]
        assert train["time_limit_minutes"] == 30
        assert all(job["time_limit_minutes"] <= 30 for job in jobs.values())
        for mode in ("argmax", "sampled"):
            evaluation = jobs[f"evaluate-{arm}-{mode}"]
            assert evaluation["after_success"] == [train["submission"]["id"]]
            assert evaluation["after_terminal"] == []
    assert {8000, 8001} <= set(jobs["summarize"]["after_terminal"])
    assert (
        tmp_path / "artifacts/probes/credit-test/bc-actor.pt"
    ).read_bytes() == actor.read_bytes()
    for command in queued:
        assert argument(command, "--max-parallel-runs") == "1"
        assert argument(command, "--max-attempts") == "1"
        assert "--priority" not in command
        assert argument(command, "--cwd").endswith("a" * 64)


@pytest.mark.parametrize(
    "state,exit_code,stale,expected",
    [
        ("succeeded", 0, 0, True),
        ("failed", 75, 30, True),
        ("failed", 75, 29, False),
        ("failed", 1, 30, False),
        ("running", None, 30, False),
    ],
)
def test_learning_evidence_requires_clean_completion_or_verified_cull(
    state, exit_code, stale, expected
):
    module = load_script("summarize_credit_campaign")
    job = {"state": state, "attempts": [{"exitCode": exit_code}]}
    rows = [{"iteration": 50, "actor_updates": 29, "autocull_state": {"stale_observations": stale}}]
    result = module.training_evidence(job, rows, 50)
    assert result["valid_completed_learning_evidence"] is expected
    assert not module.training_evidence(job, rows, 49)["valid_completed_learning_evidence"]
    rows[0]["actor_updates"] = 0
    assert not module.training_evidence(job, rows, 50)["valid_completed_learning_evidence"]


def test_evaluation_rejects_wrong_decoding_and_changed_bc(tmp_path):
    module = load_script("summarize_credit_campaign")
    path = tmp_path / "evaluation.json"
    assert module.read_evaluation(path, "sampled", "a" * 64) is None
    path.write_text(
        json.dumps(
            {"complete": True, "decoding": "argmax", "artifacts": {"bc": {"sha256": "a" * 64}}}
        )
    )
    with pytest.raises(ValueError, match="wrong-decoding"):
        module.read_evaluation(path, "sampled", "a" * 64)
    with pytest.raises(ValueError, match="unmatched-BC"):
        module.read_evaluation(path, "argmax", "b" * 64)


@pytest.mark.parametrize("changed", [None, "path", "sha256", "source_identity"])
def test_evidence_is_bound_to_checkpoint_bytes_path_and_source(tmp_path, changed):
    module = load_script("summarize_credit_campaign")
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    artifact = {
        "path": str(checkpoint),
        "sha256": module.file_sha256(checkpoint),
        "source_identity": {"sha256": "a" * 64},
    }
    if changed:
        artifact[changed] = {
            "path": str(tmp_path / "another.pt"),
            "sha256": "b" * 64,
            "source_identity": {"sha256": "b" * 64},
        }[changed]
        with pytest.raises(ValueError, match="do not match"):
            module.validate_checkpoint_binding(
                {"artifacts": {"ppo": artifact}}, checkpoint, "a" * 64
            )
    else:
        module.validate_checkpoint_binding({"artifacts": {"ppo": artifact}}, checkpoint, "a" * 64)
