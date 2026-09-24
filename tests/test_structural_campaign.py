"""Model-free contracts for full-budget architectural comparisons and queue gates."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kaggriculture.modelargs import actor_model_config, model_config_from_args
from kaggriculture.production import production_ppo_config
from kaggriculture.registry import resolve_architecture

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _argument(command, flag):
    assert command.count(flag) == 1, flag
    return command[command.index(flag) + 1]


@pytest.fixture(params=[False, True], ids=["dry-run", "submission"])
def campaign(request, tmp_path, monkeypatch):
    monkeypatch.setenv("KRAGG_PROJECT_ROOT", str(tmp_path))
    module = _module("queue_structural_campaign")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    script = tmp_path / "scripts/queue_structural_campaign.py"
    monkeypatch.setattr(module, "__file__", str(script))
    monkeypatch.setattr(module, "source_identity", lambda: {"sha256": "f" * 64})
    frozen = []
    monkeypatch.setattr(module, "freeze_source", lambda path: frozen.append(path))
    submissions = []

    def submit(command, *, text):
        assert request.param and text
        submissions.append(command)
        return json.dumps({"id": 9000 + len(submissions), "state": "queued"})

    monkeypatch.setattr(module.subprocess, "check_output", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--name",
            "contracts",
            "--validation-job",
            "8124",
            "--causal-validation-job",
            "8134",
            "--arms",
            *module.ARMS,
            *(["--submit"] if request.param else []),
        ],
    )
    module.main()
    manifest = json.loads((tmp_path / "artifacts/probes/contracts/campaign.json").read_text())
    assert frozen == [Path(manifest["source"])]
    return module, manifest, submissions


def test_dependency_graph_shares_bc_and_evaluates_culled_runs(campaign):
    module, manifest, submissions = campaign
    jobs = manifest["jobs"]
    assert len([name for name in jobs if name.startswith("bc-")]) == 8
    for owner in (
        "entity",
        "entity-v4",
        "workspace",
        "shared-plan",
        "causal",
        "causal-v4",
        "lejepa",
        "lejepa-quantity-all",
    ):
        job = jobs[f"bc-{owner}"]
        assert job["parents"] == [8134 if owner.startswith("causal") else 8124]
        assert job["dependency"] == "success"
    for arm, (owner, *_rest) in module.ARMS.items():
        bc, gate, train = (jobs[f"bc-{owner}"], jobs[f"gate-{arm}"], jobs[f"learn-{arm}"])
        assert gate["parents"] == [bc["submission"]["id"]]
        assert train["parents"] == [gate["submission"]["id"]]
        assert gate["dependency"] == train["dependency"] == "success"
        assert train["minutes"] == 30
        assert all(job["minutes"] <= 30 for job in jobs.values())
        for mode in ("sampled", "argmax"):
            evaluation = jobs[f"evaluate-{arm}-{mode}"]
            assert evaluation["parents"] == [train["submission"]["id"]]
            assert evaluation["dependency"] == "terminal"
            assert _argument(evaluation["command"], "--decoding") == mode
            assert _argument(evaluation["command"], "--games") == "256"
    for command in submissions:
        assert _argument(command, "--max-parallel-runs") == "1"
        assert _argument(command, "--max-attempts") == "1"
        assert "--priority" not in command
        assert _argument(command, "--cwd") == manifest["source"]
        assert "--time-limit" in command


def test_commands_keep_production_shape_gae_precision_and_named_ablations(campaign):
    module, manifest, _ = campaign
    ppo = production_ppo_config(update_compile_mode="reduce-overhead")
    for arm, (_owner, family, _changes, ratio, forecast) in module.ARMS.items():
        train = manifest["jobs"][f"learn-{arm}"]["command"]
        gate = manifest["jobs"][f"gate-{arm}"]["command"]
        for command in (train, gate):
            for flag, expected in {
                "--architecture": family,
                "--device": "cuda",
                "--games": "128",
                "--league-games": "64",
                "--minibatch-size": str(ppo["minibatch_size"]),
                "--policy-ratio-scope": ratio,
                "--actor-gae-lambda": str(ppo["actor_gae_lambda"]),
                "--economic-forecast-coefficient": str(forecast),
                "--update-compile-mode": "reduce-overhead",
                "--rollout-forward-mode": "inductor_graph",
                "--reward-mode": "terminal-outcome",
            }.items():
                assert _argument(command, flag) == expected
            assert "--rollout-bfloat16" in command and "--no-bfloat16" not in command
            for suffix in ("latent", "decision", "critic-latent", "critic-value"):
                assert _argument(command, f"--structured-{suffix}-coefficient") == "0.0"
        assert _argument(train, "--iterations") == "500"
        assert _argument(train, "--max-hours") == str(module.TRAINER_HOURS)
        assert _argument(train, "--episode-steps") == "720"
        assert _argument(train, "--critic-gae-lambda") == str(ppo["critic_gae_lambda"])
        assert _argument(train, "--architecture-panel") == "25"
        assert _argument(train, "--league-selection") == "hardness"
        assert "--external-eval" not in train and "--autocull" not in train
        assert _argument(gate, "--repeats") == "6"
        assert _argument(gate, "--init-actor-from") == _argument(train, "--init-actor-from")
    assert (
        _argument(manifest["jobs"]["learn-economic-control"]["command"], "--critic-architecture")
        == "economic"
    )
    assert (
        _argument(manifest["jobs"]["learn-forecast"]["command"], "--critic-architecture")
        == "forecast"
    )


def test_real_argument_parsers_preserve_critic_arm_bc_compatibility(campaign, monkeypatch):
    module, manifest, _ = campaign
    train_module = _module("train_ppo")
    gate_module = _module("benchmark_ppo_iteration")
    for arm, (owner, *_rest) in module.ARMS.items():
        _architecture, expected = module.arm_config(arm)
        identities = []
        for label, parser in ((f"learn-{arm}", train_module), (f"gate-{arm}", gate_module)):
            command = manifest["jobs"][label]["command"]
            monkeypatch.setattr(sys, "argv", command[1:])
            args = parser.parse_args()
            config = model_config_from_args(resolve_architecture(args.architecture), args)
            assert config == expected
            identities.append(actor_model_config(config))
        _, bc_config = module.arm_config(
            next(name for name in module.ARMS if module.ARMS[name][0] == owner)
        )
        assert identities[0] == identities[1] == actor_model_config(bc_config)


def test_quantity_family_is_matched_except_for_the_action_interface(campaign):
    module, manifest, _ = campaign
    assert module.FAMILIES["quantity"] == ("lejepa", "lejepa-quantity-all")
    _, control = module.arm_config("lejepa")
    _, treatment = module.arm_config("lejepa-quantity-all")
    assert control.action_interface == 1
    assert treatment.action_interface == 2
    assert control.to_dict() | {"action_interface": 2} == treatment.to_dict()
    assert "--epochs" not in manifest["jobs"]["bc-lejepa-quantity-all"]["command"]


def test_causal_v4_family_uses_promoted_observation_schema(campaign):
    module, manifest, _ = campaign
    assert module.FAMILIES["causal-v4"] == ("joint-control-v4", "causal-v4")
    for arm in module.FAMILIES["causal-v4"]:
        _, config = module.arm_config(arm)
        assert config.observation_schema_version == 4
        owner = module.ARMS[arm][0]
        command = manifest["jobs"][f"bc-{owner}"]["command"]
        assert _argument(command, "--observation-schema-version") == "4"
        assert "--epochs" not in command


@pytest.mark.parametrize("job", ["0", "-1"])
def test_invalid_validation_dependency_fails_before_freezing(monkeypatch, job):
    module = _module("queue_structural_campaign")
    monkeypatch.setattr(sys, "argv", ["campaign", "--name", "invalid", "--validation-job", job])
    monkeypatch.setattr(
        module, "freeze_source", lambda _: pytest.fail("must validate before freezing")
    )
    with pytest.raises(SystemExit):
        module.main()


def test_the_lejepa_arm_trains_its_backbone_in_every_job(campaign, monkeypatch):
    """BC, the gate and PPO all carry the objective, since nothing else trains it.

    Demonstrations hold no reward, so the clone takes only the two transition
    terms, on the two-row runs its successor pairs need.
    """
    module, manifest, _ = campaign
    # `train_bc` declares dataclasses, which resolve their module by name.
    trainer_spec = importlib.util.spec_from_file_location("train_bc", SCRIPTS / "train_bc.py")
    trainer = importlib.util.module_from_spec(trainer_spec)
    monkeypatch.setitem(sys.modules, "train_bc", trainer)
    assert trainer_spec.loader is not None
    trainer_spec.loader.exec_module(trainer)
    parsers = {
        "bc-lejepa": trainer,
        "gate-lejepa": _module("benchmark_ppo_iteration"),
        "learn-lejepa": _module("train_ppo"),
    }
    for label, parser in parsers.items():
        command = manifest["jobs"][label]["command"]
        monkeypatch.setattr(sys, "argv", command[1:])
        args = parser.parse_args()
        assert args.architecture == "lejepa", label
        assert args.observation_schema_version == 4, label
        if label != "bc-lejepa":
            assert args.critic_readout_ffn is True, label
        assert args.jepa_prediction_coefficient == 1.0, label
        assert args.jepa_sigreg_coefficient == 0.09, label
        assert args.jepa_horizon == 1, label
        if label == "bc-lejepa":
            assert args.run_length == 2
            assert args.epochs == 2
            assert "--epochs" not in command
            assert not hasattr(args, "jepa_reward_coefficient")
        else:
            assert args.jepa_reward_coefficient == 0.1, label
    # PPO steps the backbone at its own, slower rate (JEPA_BACKBONE_LEARNING_RATE).
    monkeypatch.setattr(sys, "argv", manifest["jobs"]["learn-lejepa"]["command"][1:])
    learn = parsers["learn-lejepa"].parse_args()
    assert learn.structured_learning_rate == module.JEPA_BACKBONE_LEARNING_RATE
    assert learn.structured_learning_rate < learn.actor_lr
    # No other arm is handed the objective or the backbone rate.
    for label, job in manifest["jobs"].items():
        if "lejepa" not in label:
            assert not any("--jepa-" in token for token in job["command"]), label
            assert "--structured-learning-rate" not in job["command"], label
        if label.startswith("bc-") and label not in {
            "bc-lejepa",
            "bc-lejepa-quantity-all",
            "bc-entity-v4",
            "bc-causal-v4",
        }:
            command = job["command"]
            assert "--epochs" not in command, label
            assert command[command.index("--observation-schema-version") + 1] == "3", label
