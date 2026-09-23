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
    assert len([name for name in jobs if name.startswith("bc-")]) == 5
    for owner in ("entity", "workspace", "shared-plan", "causal", "lejepa"):
        job = jobs[f"bc-{owner}"]
        assert job["parents"] == [8134 if owner == "causal" else 8124]
        assert job["dependency"] == "success"
    for arm, (owner, *_rest) in module.ARMS.items():
        bc, gate, train = (jobs[f"bc-{owner}"], jobs[f"gate-{arm}"], jobs[f"learn-{arm}"])
        assert gate["parents"] == [bc["submission"]["id"]]
        assert train["parents"] == [gate["submission"]["id"]]
        assert gate["dependency"] == train["dependency"] == "success"
        assert train["minutes"] == 250
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
        assert _argument(train, "--max-hours") == "4.0"
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
        if label != "bc-lejepa":
            assert args.critic_readout_ffn is True, label
        assert args.jepa_prediction_coefficient == 1.0, label
        assert args.jepa_sigreg_coefficient == 0.09, label
        assert args.jepa_horizon == 1, label
        if label == "bc-lejepa":
            assert args.run_length == 2
            # Its clone converges later than the entity actor's (see BC_EPOCHS).
            assert args.epochs == module.BC_EPOCHS[1] > module.BC_EPOCHS[0]
            assert not hasattr(args, "jepa_reward_coefficient")
        else:
            assert args.jepa_reward_coefficient == 0.1, label
    # No other arm is handed the objective.
    for label, job in manifest["jobs"].items():
        if not label.endswith("lejepa"):
            assert not any("--jepa-" in token for token in job["command"]), label
        if label.startswith("bc-") and label != "bc-lejepa":
            command = job["command"]
            assert command[command.index("--epochs") + 1] == str(module.BC_EPOCHS[0]), label
