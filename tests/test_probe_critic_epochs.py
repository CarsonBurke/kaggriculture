from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kaggriculture.ppo import PpoConfig, make_optimizers
from kaggriculture.registry import STRUCTURED
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic


def _probe():
    path = Path(__file__).parents[1] / "scripts" / "probe_critic_epochs.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_probe_critic_epochs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _models_and_state():
    model_config = StructuredConfig(
        model_dim=16,
        attention_heads=2,
        attention_kv_heads=1,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=2,
        latents=4,
        core_layers=1,
        quantity_rank=4,
        critic_core_layers=1,
        critic_latents=4,
        value_atoms=11,
    ).to_dict()
    actor = StructuredActor(StructuredConfig(**model_config))
    critic = StructuredCritic(StructuredConfig(**model_config))
    config = PpoConfig(optimizer="adamw", critic_epochs=2, minibatch_size=8)
    _, optimizer = make_optimizers(actor, critic, config)
    loss = torch.stack([parameter.square().sum() for parameter in critic.parameters()]).sum()
    loss.backward()
    optimizer.step()
    optimizer.param_groups[0]["base_lr"] = 7.5e-4
    optimizer.param_groups[0]["warmup_step"] = 19
    state = {
        "critic": critic.state_dict(),
        "critic_optimizer": optimizer.state_dict(),
        "ppo_config": asdict(config),
    }
    return actor, model_config, config, state


def test_parser_requires_exactly_one_source() -> None:
    module = _probe()

    assert module.parse_args(["--actor", "actor.pt"]).actor == Path("actor.pt")
    assert module.parse_args(["--checkpoint", "checkpoint.pt"]).checkpoint == Path("checkpoint.pt")
    with pytest.raises(SystemExit):
        module.parse_args([])
    with pytest.raises(SystemExit):
        module.parse_args(["--actor", "actor.pt", "--checkpoint", "checkpoint.pt"])


def test_stage_rollout_includes_structured_unit_activity() -> None:
    module = _probe()
    rollout = SimpleNamespace(
        states={"tile": np.ones((2, 3), dtype=np.float32)},
        unit_actions=np.zeros((2, 4), dtype=np.int64),
        unit_active=np.ones((2, 4), dtype=np.bool_),
        market_active=np.ones((2, 4), dtype=np.bool_),
    )

    staged = module._stage_rollout(rollout, torch.device("cpu"))

    assert set(staged) == {"tile", "unit_actions", "unit_active", "market_active"}
    assert staged["unit_active"].dtype == torch.bool


def test_checkpoint_build_restores_isolated_critic_and_optimizer() -> None:
    module = _probe()
    actor, model_config, config, state = _models_and_state()

    first_critic, first_optimizer = module._build_critic(
        actor, STRUCTURED, model_config, config, state, torch.device("cpu")
    )
    expected = state["critic"]
    assert all(
        torch.equal(value, expected[name]) for name, value in first_critic.state_dict().items()
    )
    assert first_optimizer.param_groups[0]["base_lr"] == 7.5e-4
    assert first_optimizer.param_groups[0]["warmup_step"] == 19
    assert first_optimizer.state

    with torch.no_grad():
        next(first_critic.parameters()).add_(1.0)
    for optimizer_state in first_optimizer.state.values():
        tensor = next(value for value in optimizer_state.values() if torch.is_tensor(value))
        tensor.add_(1.0)
        break

    second_critic, second_optimizer = module._build_critic(
        actor, STRUCTURED, model_config, config, state, torch.device("cpu")
    )
    assert all(
        torch.equal(value, expected[name]) for name, value in second_critic.state_dict().items()
    )
    assert second_optimizer.param_groups[0]["base_lr"] == 7.5e-4
    assert second_optimizer.param_groups[0]["warmup_step"] == 19


def test_checkpoint_mode_rejects_synthetic_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _probe()
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe_critic_epochs.py", "--checkpoint", "unused.pt", "--warmup-iterations", "1"],
    )

    with pytest.raises(SystemExit, match="already supplies warm state"):
        module.main()


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"agents": []}, "single-learner"),
        ({"critic": {}}, "critic_optimizer, ppo_config"),
        ({"critic": {}, "critic_optimizer": {}}, "ppo_config"),
        (
            {"critic": {}, "critic_optimizer": {}, "ppo_config": {}},
            "recorded reward mode",
        ),
    ],
)
def test_checkpoint_loader_rejects_incomplete_or_population_state(
    tmp_path: Path, payload: dict, message: str
) -> None:
    module = _probe()
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        module._load_checkpoint_state(path)
