from __future__ import annotations

import importlib.util
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from kaggriculture.entity import EntityConfig
from kaggriculture.lejepa_model import LejepaConfig


@pytest.fixture
def diagnostic():
    path = Path(__file__).parents[1] / "scripts" / "fit_lejepa_critics.py"
    spec = importlib.util.spec_from_file_location("lejepa_critic_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entity_critic_takes_reference_critic_options_on_actor_geometry(diagnostic):
    actor = LejepaConfig(critic_readout_ffn=True, critic_source_read=False)
    reference = asdict(EntityConfig(critic_readout_ffn=False, critic_source_read=True))
    config = diagnostic.entity_critic_config(actor, reference)
    assert type(config) is EntityConfig
    assert not config.critic_readout_ffn and config.critic_source_read
    assert config.model_dim == actor.model_dim and config.core_layers == actor.core_layers


def test_entity_critic_refuses_a_reference_with_other_geometry(diagnostic):
    reference = asdict(replace(EntityConfig(), core_layers=3))
    with pytest.raises(ValueError, match="core_layers"):
        diagnostic.entity_critic_config(LejepaConfig(), reference)
