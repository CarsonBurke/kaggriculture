from __future__ import annotations

import pytest
import torch
from kaggle_environments import make

from kaggriculture.inference import (
    ACTOR_ARTIFACT_FORMAT_VERSION,
    CheckpointAgent,
    load_actor_artifact,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.provenance import source_identity
from kaggriculture.registry import (
    ARCHITECTURES,
    architecture_of,
    resolve_architecture,
)
from kaggriculture.structured import StructuredActor, StructuredConfig


def test_resolution_defaults_untagged_payloads_to_the_conv_family() -> None:
    assert resolve_architecture(None).actor_class is FarmActor
    assert resolve_architecture({}).actor_class is FarmActor
    assert resolve_architecture({"architecture": "structured"}).actor_class is StructuredActor
    assert resolve_architecture("structured").config_class is StructuredConfig
    with pytest.raises(ValueError, match="unknown actor architecture"):
        resolve_architecture("perceiver")


def test_architecture_of_maps_constructed_actors_back() -> None:
    tiny = StructuredConfig(
        model_dim=32,
        attention_heads=2,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
    )
    assert architecture_of(StructuredActor(tiny)).name == "structured"
    conv = FarmActor(
        ModelConfig(
            cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
        )
    )
    assert architecture_of(conv).name == "entity-cnn"


def test_structured_artifact_loads_and_acts_on_a_real_observation(tmp_path) -> None:
    torch.manual_seed(0)
    config = StructuredConfig(
        model_dim=32,
        attention_heads=2,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
    )
    actor = StructuredActor(config)
    payload = {
        "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
        "architecture": "structured",
        "model_config": config.to_dict(),
        "actor": actor.state_dict(),
        "iteration": 0,
        "metrics": {},
        "source_identity": source_identity(),
        "run_provenance": None,
    }
    path = tmp_path / "structured-actor.pt"
    torch.save(payload, path)

    loaded, loaded_payload = load_actor_artifact(path)
    assert isinstance(loaded, StructuredActor)
    assert loaded_payload["architecture"] == "structured"

    agent = CheckpointAgent(path)
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 11})
    state = environment.reset(2)
    action = agent(state[0].observation)
    assert isinstance(action, dict)
    assert "hands" in action or "orders" in action or action


def test_every_registered_family_round_trips_config() -> None:
    for architecture in ARCHITECTURES.values():
        config = architecture.config_class()
        actor = architecture.build_actor(config.to_dict())
        assert type(actor) is architecture.actor_class
