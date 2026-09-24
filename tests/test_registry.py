from __future__ import annotations

import argparse
from dataclasses import replace

import pytest
import torch
from kaggle_environments import make

from kaggriculture.entity import EntityConfig
from kaggriculture.inference import (
    ACTOR_ARTIFACT_FORMAT_VERSION,
    CheckpointAgent,
    load_actor_artifact,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.modelargs import (
    actor_model_config,
    add_model_config_arguments,
    model_config_arguments,
    model_config_from_args,
)
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


@pytest.mark.parametrize("schema_version,farm_width", [(3, 4), (4, 5)])
def test_structured_artifact_loads_and_acts_on_a_real_observation(
    tmp_path, schema_version, farm_width
) -> None:
    torch.manual_seed(0)
    config = StructuredConfig(
        observation_schema_version=schema_version,
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
    assert loaded.config.observation_schema_version == schema_version
    assert loaded.trunk.economy.farm_projection.in_features == farm_width

    agent = CheckpointAgent(path)
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 11})
    state = environment.reset(2)
    action = agent(state[0].observation)
    assert isinstance(action, dict)
    assert "hands" in action or "orders" in action or action


def test_every_registered_family_round_trips_config() -> None:
    for architecture in ARCHITECTURES.values():
        config = architecture.config_class()
        assert architecture.build_config(config.to_dict()) == config


@pytest.mark.parametrize(
    "field,value",
    [
        ("shared_memory_kv", None),
        ("inter_attention_ffn", True),
        ("unit_local_readout", True),
        ("critic_readout_ffn", True),
        ("unit_local_init", False),
        ("tile_cross_rope", True),
    ],
)
def test_entity_ablation_cli_roundtrip_preserves_independent_selection(field, value) -> None:
    family = resolve_architecture("entity-attention")
    parser = argparse.ArgumentParser()
    add_model_config_arguments(parser)
    baseline = model_config_from_args(family, parser.parse_args([]))
    if value is None:
        value = not getattr(baseline, field)
    selected = model_config_from_args(
        family, parser.parse_args(["--" + field.replace("_", "-"), str(value).lower()])
    )
    assert selected == replace(baseline, **{field: value})
    assert selected != baseline
    restored = model_config_from_args(
        family, parser.parse_args(model_config_arguments(family, selected.to_dict()))
    )
    assert family.build_config(restored.to_dict()) == selected


def test_entity_actor_identity_excludes_only_critic_readout_ablation() -> None:
    baseline = EntityConfig()
    critic_changed = replace(baseline, critic_readout_ffn=True)
    assert actor_model_config(critic_changed) == actor_model_config(baseline)
    assert actor_model_config(critic_changed.to_dict()) == actor_model_config(baseline)
    assert (
        resolve_architecture("entity-attention").build_config(critic_changed.to_dict()) != baseline
    )
    for field, value in (
        ("shared_memory_kv", not baseline.shared_memory_kv),
        ("inter_attention_ffn", True),
        ("unit_local_readout", True),
        ("unit_local_init", False),
        ("tile_cross_rope", True),
    ):
        assert actor_model_config(replace(baseline, **{field: value})) != actor_model_config(
            baseline
        )


@pytest.mark.parametrize(
    "heads,kv_heads",
    [
        pytest.param(4, 4, id="mha"),
        pytest.param(4, 8, id="more-kv-than-query-heads"),
        pytest.param(4, 3, id="nondivisible-query-groups"),
        pytest.param(5, 2, id="nondivisible-model-width"),
    ],
)
def test_entity_config_rejects_attention_without_strict_gqa(heads, kv_heads) -> None:
    with pytest.raises(ValueError):
        EntityConfig(model_dim=96, attention_heads=heads, attention_kv_heads=kv_heads)


@pytest.mark.parametrize("builder", ["build_actor", "build_critic"])
@pytest.mark.parametrize("version", [None, 1, 2, 5])
@pytest.mark.parametrize("architecture", ["structured", "entity-attention"])
def test_structured_artifacts_reject_stale_observation_schema(
    builder, version, architecture
) -> None:
    family = resolve_architecture(architecture)
    config = family.config_class().to_dict()
    if version is None:
        del config["observation_schema_version"]
    else:
        config["observation_schema_version"] = version
    with pytest.raises(ValueError, match="stale structured observation schema"):
        getattr(family, builder)(config)
