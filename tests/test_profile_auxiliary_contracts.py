from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggriculture.entity import EntityConfig
from kaggriculture.ppo import PpoConfig
from kaggriculture.production import production_ppo_config
from kaggriculture.registry import ENTITY_ATTENTION


def _script(name, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("auxiliary_steps", [0, 2])
def test_profiler_counts_plain_and_auxiliary_updates(monkeypatch, auxiliary_steps):
    module = _script("profile_update_phases", monkeypatch)
    metrics = {
        "states": 10,
        "actor_updates": 2,
        "updates": 2,
        "epochs": 1,
        "actor_minibatches_intended": 2,
        "kl_early_stop": 0,
    }
    if auxiliary_steps:
        metrics.update(
            structured_critic_predictor_updates=auxiliary_steps,
            structured_critic_auxiliary_updates=auxiliary_steps,
        )
    counts = module._work_counts(
        metrics,
        SimpleNamespace(trajectories=2, horizon=5),
        PpoConfig(minibatch_size=8),
    )
    assert counts["critic_predictor_steps"] == auxiliary_steps
    assert counts["critic_joint_auxiliary_updates"] == auxiliary_steps
    assert counts["actor_optimizer_valid_rows"] == 10
    assert counts["critic_optimizer_valid_rows"] == 10


def test_historical_benchmark_preserves_nextlat_when_production_disables_it(monkeypatch):
    module = _script("benchmark_entity_architecture", monkeypatch)
    assert not PpoConfig(
        **production_ppo_config(update_compile_mode="default")
    ).structured_critic_auxiliary_active
    historical = module.historical_ppo_config()
    command = module.iteration_command(
        ENTITY_ATTENTION, EntityConfig(), None, Path("unused.jsonl"), 42
    )
    for term in ("latent", "value"):
        field = f"structured_critic_{term}_coefficient"
        flag = f"--structured-critic-{term}-coefficient"
        assert getattr(historical, field) == 1.0
        assert float(command[command.index(flag) + 1]) == getattr(historical, field)
    assert not historical.structured_actor_auxiliary_active


def test_probe_recovery_accepts_disabled_predictors_without_rng(monkeypatch):
    module = _script("probe_lr_trust", monkeypatch)
    member = {"actor": {}, "critic": {}}
    monkeypatch.setattr(module, "require_checkpoint_format", lambda state: None)
    monkeypatch.setattr(module, "checkpoint_agent_states", lambda state: [member])
    assert module._auxiliary_recovery({}, PpoConfig()) == (member, None)


@pytest.mark.parametrize("critic", [False, True])
def test_probe_recovery_requires_only_enabled_predictor_state(monkeypatch, critic):
    module = _script("probe_lr_trust", monkeypatch)
    key = "structured_critic_dynamics" if critic else "structured_dynamics"
    field = "structured_critic_latent_coefficient" if critic else "structured_latent_coefficient"
    config = PpoConfig(**{field: 1.0})
    member = {}
    state = {"structured_auxiliary_rng": {"state": {"value": 1}}}
    monkeypatch.setattr(module, "require_checkpoint_format", lambda state: None)
    monkeypatch.setattr(module, "checkpoint_agent_states", lambda state: [member])
    with pytest.raises(ValueError, match=key):
        module._auxiliary_recovery(state, config)
    member.update({key: {}, f"{key}_optimizer": {}})
    actual, rng = module._auxiliary_recovery(state, config)
    assert actual is member
    assert rng == state["structured_auxiliary_rng"]
    assert rng is not state["structured_auxiliary_rng"]
