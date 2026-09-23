"""Model-free contracts for the architecture campaign's commands and queue DAG."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ARMS = ("control", "critic-source", "writeback", "tile-bias", "hardness-league")
DELTAS = {
    "control": {},
    "critic-source": {"--critic-source-read": "true"},
    "writeback": {"--memory-writeback": "true"},
    "tile-bias": {"--unit-tile-bias": "true"},
    "hardness-league": {},
}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argument(command, flag):
    assert command.count(flag) == 1, f"expected exactly one {flag}: {command}"
    return command[command.index(flag) + 1]


@pytest.fixture(params=[False, True], ids=["dry-run", "mock-submission"])
def campaign(request, tmp_path, monkeypatch):
    module = _load_script("queue_architecture_campaign")
    fake_script = tmp_path / "scripts" / "queue_architecture_campaign.py"
    fake_script.parent.mkdir()
    fake_script.write_text("# unchanged source sentinel\n")
    original_source = fake_script.read_bytes()
    monkeypatch.setattr(module, "__file__", str(fake_script))
    identity = {"sha256": "d" * 64, "dirty": True}
    monkeypatch.setattr(module, "source_identity", lambda: dict(identity))
    frozen = []
    monkeypatch.setattr(module, "freeze_source", lambda path: frozen.append(path))
    submitted = []

    def check_output(command, *, text):
        assert request.param, "dry-run must never invoke a subprocess"
        assert text is True
        submitted.append(list(command))
        return json.dumps({"id": 9000 + len(submitted), "state": "queued"})

    monkeypatch.setattr(module.subprocess, "check_output", check_output)
    argv = [str(fake_script), "--name", "test-campaign", "--validation-job", "7659"]
    if request.param:
        argv.append("--submit")
    monkeypatch.setattr(sys, "argv", argv)
    before_model = module.production_model_config()
    module.main()
    manifest_path = tmp_path / "artifacts" / "probes" / "test-campaign" / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    assert fake_script.read_bytes() == original_source
    assert module.production_model_config() == before_model
    # The mocked freeze is the sole source operation. Launch planning only
    # writes its manifest; it must not create models, runs, or source patches.
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()} == {
        fake_script.relative_to(tmp_path),
        manifest_path.relative_to(tmp_path),
    }
    assert frozen == [tmp_path / "artifacts" / "source-snapshots" / identity["sha256"]]
    return SimpleNamespace(
        module=module,
        root=tmp_path,
        manifest=manifest,
        submitted=submitted,
        submit=request.param,
        identity=identity,
    )


def test_campaign_changes_only_the_named_arm_and_keeps_full_production_shape(campaign):
    jobs = campaign.manifest["jobs"]
    architecture = campaign.module.resolve_architecture(campaign.module.ENTITY_ATTENTION)
    config_args = campaign.module.model_config_arguments(
        architecture, campaign.module.production_model_config()
    )
    base = dict(zip(config_args[::2], config_args[1::2], strict=True))
    assert base["--shared-memory-kv"] == "false"
    assert base["--critic-source-read"] == "true"
    base["--critic-source-read"] = "false"
    for arm in ARMS:
        expected = base | DELTAS[arm]
        for prefix in ("bc", "gate", "learn"):
            job = jobs.get(f"{prefix}-{arm}")
            if job is None:
                continue
            command = job["command"]
            for flag, value in expected.items():
                assert _argument(command, flag) == value
            if prefix == "bc":
                assert _argument(command, "--epochs") == "2"
                assert _argument(command, "--batch-size") == "1024"
                assert _argument(command, "--device") == "cuda"
            else:
                assert _argument(command, "--games") == "128"
                assert _argument(command, "--league-games") == "64"
                assert _argument(command, "--minibatch-size") == "8192"
                assert _argument(command, "--reward-mode") == "terminal-outcome"
                assert _argument(command, "--rollout-forward-mode") == "inductor_graph"
                assert _argument(command, "--update-compile-mode") == "reduce-overhead"
                assert "--rollout-bfloat16" in command
                for term in ("latent", "value"):
                    assert _argument(command, f"--structured-critic-{term}-coefficient") == "1.0"
        learning = jobs[f"learn-{arm}"]["command"]
        assert "--external-eval" not in learning
        assert _argument(learning, "--max-hours") == "0.4"
        assert _argument(learning, "--seed") == "20260812"
        assert _argument(learning, "--episode-steps") == "720"
        assert _argument(learning, "--league-selection") == (
            "hardness" if arm == "hardness-league" else "stratified"
        )


def test_campaign_dependencies_gate_learning_and_reuse_the_exact_control_clone(campaign):
    jobs = campaign.manifest["jobs"]
    assert set(jobs) == {
        *(f"bc-{arm}" for arm in ("control", "writeback", "tile-bias")),
        *(f"gate-{arm}" for arm in ("control", "critic-source", "writeback", "tile-bias")),
        *(f"learn-{arm}" for arm in ARMS),
        *(f"evaluate-{arm}" for arm in ARMS),
        "common-policy-fit",
    }

    def identifier(label):
        return jobs[label]["submission"]["id"]

    control_clone = str(campaign.root / "runs" / "test-campaign" / "control" / "bc" / "bc-actor.pt")
    for arm in ARMS:
        clone_arm = "control" if arm in ("critic-source", "hardness-league") else arm
        gate_arm = "control" if arm == "hardness-league" else arm
        assert jobs[f"bc-{clone_arm}"]["parents"] == [7659]
        assert jobs[f"gate-{gate_arm}"]["parents"] == [identifier(f"bc-{clone_arm}")]
        assert jobs[f"learn-{arm}"]["parents"] == [identifier(f"gate-{gate_arm}")]
        assert jobs[f"evaluate-{arm}"]["parents"] == [identifier(f"learn-{arm}")]
        clone = _argument(jobs[f"learn-{arm}"]["command"], "--init-actor-from")
        assert clone == str(
            campaign.root / "runs" / "test-campaign" / clone_arm / "bc" / "bc-actor.pt"
        )
        assert _argument(jobs[f"gate-{gate_arm}"]["command"], "--init-actor-from") == clone
    assert _argument(jobs["common-policy-fit"]["command"], "--actor") == control_clone
    assert jobs["common-policy-fit"]["parents"] == [
        identifier("bc-control"),
        identifier("gate-critic-source"),
    ]


def test_campaign_submission_caps_and_freezes_every_job(campaign):
    jobs = campaign.manifest["jobs"]
    assert all(0 < job["minutes"] <= 25 for job in jobs.values())
    assert campaign.manifest["source_identity"] == campaign.identity
    for job in jobs.values():
        assert Path(job["command"][1]).parent == Path(campaign.manifest["source"]) / "scripts"
    if not campaign.submit:
        assert campaign.submitted == []
        assert all(job["submission"]["state"] == "dry-run" for job in jobs.values())
        return
    assert len(campaign.submitted) == len(jobs)
    for queue_command, job in zip(campaign.submitted, jobs.values(), strict=True):
        separator = queue_command.index("--")
        options = queue_command[:separator]
        assert options[:3] == ["mlq", "submit", "--json"]
        assert _argument(options, "--max-parallel-runs") == "1"
        assert _argument(options, "--max-attempts") == "1"
        assert _argument(options, "--time-limit") == f"{job['minutes']}m"
        assert _argument(options, "--cwd") == campaign.manifest["source"]
        assert queue_command[separator + 1 :] == job["command"]
        parents = [
            options[index + 1] for index, value in enumerate(options) if value == "--after-success"
        ]
        assert parents == [str(parent) for parent in job["parents"]]
        environments = [
            options[index + 1] for index, value in enumerate(options) if value == "--env"
        ]
        assert f"KRAGG_SOURCE_DIGEST={campaign.identity['sha256']}" in environments
        assert f"PYTHONPATH={campaign.manifest['source']}/src" in environments


def test_campaign_primary_evaluation_resolves_to_argmax_on_common_seeds(campaign, monkeypatch):
    evaluator = _load_script("evaluate_architecture_campaign")
    panels = []
    for arm in ARMS:
        command = campaign.manifest["jobs"][f"evaluate-{arm}"]["command"]
        monkeypatch.setattr(sys, "argv", command[1:])
        args = evaluator.parse_args()
        assert args.decoding == "argmax"
        assert args.games == 256
        assert [label for label, _path in args.artifact] == ["bc", "ppo"]
        panels.append((args.seed_start, args.games))
    assert len(set(panels)) == 1


def test_reusing_clones_records_exact_bytes_and_gates_without_new_bc_jobs(tmp_path, monkeypatch):
    module = _load_script("queue_architecture_campaign")
    monkeypatch.setattr(
        module, "__file__", str(tmp_path / "scripts" / "queue_architecture_campaign.py")
    )
    monkeypatch.setattr(module, "source_identity", lambda: {"sha256": "e" * 64})
    monkeypatch.setattr(module, "freeze_source", lambda _path: None)

    def no_submission(*_args, **_kwargs):
        pytest.fail("clone-reuse dry-run must not submit or load models")

    monkeypatch.setattr(module.subprocess, "check_output", no_submission)
    reuse_root = tmp_path / "prior-campaign"
    clones = {}
    for arm in ("control", "writeback", "tile-bias"):
        path = reuse_root / arm / "bc" / "bc-actor.pt"
        path.parent.mkdir(parents=True)
        contents = f"opaque {arm} checkpoint bytes".encode()
        path.write_bytes(contents)
        clones[arm] = (path, contents)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue_architecture_campaign.py",
            "--name",
            "remat-campaign",
            "--validation-job",
            "7700",
            "--reuse-bc-root",
            str(reuse_root),
        ],
    )
    module.main()
    manifest = json.loads(
        (tmp_path / "artifacts" / "probes" / "remat-campaign" / "campaign.json").read_text()
    )
    jobs = manifest["jobs"]
    assert len(jobs) == 15 and not any(label.startswith("bc-") for label in jobs)
    assert set(manifest["reused_bc"]) == set(clones)
    for arm, (path, contents) in clones.items():
        assert manifest["reused_bc"][arm] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
        assert path.read_bytes() == contents
    for arm in ARMS:
        clone_arm = "control" if arm in ("critic-source", "hardness-league") else arm
        gate_arm = "control" if arm == "hardness-league" else arm
        clone = str(clones[clone_arm][0])
        assert _argument(jobs[f"learn-{arm}"]["command"], "--init-actor-from") == clone
        assert _argument(jobs[f"gate-{gate_arm}"]["command"], "--init-actor-from") == clone
        assert jobs[f"gate-{gate_arm}"]["parents"] == [7700]
        assert jobs[f"learn-{arm}"]["parents"] == [jobs[f"gate-{gate_arm}"]["submission"]["id"]]
    assert jobs["common-policy-fit"]["parents"] == [
        7700,
        jobs["gate-critic-source"]["submission"]["id"],
    ]


def test_bixt_only_campaign_queues_its_four_job_chain_without_original_arms(tmp_path, monkeypatch):
    module = _load_script("queue_architecture_campaign")
    fake_script = tmp_path / "scripts" / "queue_architecture_campaign.py"
    monkeypatch.setattr(module, "__file__", str(fake_script))
    identity = {"sha256": "b" * 64, "dirty": True}
    monkeypatch.setattr(module, "source_identity", lambda: dict(identity))
    frozen = []
    monkeypatch.setattr(module, "freeze_source", lambda path: frozen.append(path))

    def no_submission(*_args, **_kwargs):
        pytest.fail("BiXT dry-run must not submit or load models")

    monkeypatch.setattr(module.subprocess, "check_output", no_submission)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(fake_script), "--name", "bixt-only", "--validation-job", "7701", "--arms", "bixt"],
    )
    before_model = module.production_model_config()
    module.main()
    manifest_path = tmp_path / "artifacts" / "probes" / "bixt-only" / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["arms"] == ["bixt"]
    assert module.DEFAULT_ARMS == ARMS
    assert module.production_model_config() == before_model
    assert frozen == [tmp_path / "artifacts" / "source-snapshots" / identity["sha256"]]
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [manifest_path]
    jobs = manifest["jobs"]
    assert list(jobs) == ["bc-bixt", "gate-bixt", "learn-bixt", "evaluate-bixt"]
    parent = 7701
    for label, job in jobs.items():
        assert job["parents"] == [parent]
        assert job["minutes"] <= 25
        parent = job["submission"]["id"]
        if label == "evaluate-bixt":
            continue
        command = job["command"]
        assert _argument(command, "--bixt-latents") == "32"
        assert _argument(command, "--global-modulation") == "false"
        assert _argument(command, "--shared-memory-kv") == "false"
        for flag in ("--critic-source-read", "--memory-writeback", "--unit-tile-bias"):
            assert _argument(command, flag) == "false"
    clone = tmp_path / "runs" / "bixt-only" / "bixt" / "bc" / "bc-actor.pt"
    assert _argument(jobs["bc-bixt"]["command"], "--output") == str(clone.parent)
    for label in ("gate-bixt", "learn-bixt"):
        command = jobs[label]["command"]
        assert _argument(command, "--init-actor-from") == str(clone)
        assert _argument(command, "--games") == "128"
        assert _argument(command, "--league-games") == "64"
        assert _argument(command, "--minibatch-size") == "8192"
        assert "--external-eval" not in command
