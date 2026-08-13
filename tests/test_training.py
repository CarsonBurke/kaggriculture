from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.provenance import source_identity
from kaggriculture.training import (
    CHECKPOINT_FORMAT_VERSION,
    append_iteration_jsonl,
    load_checkpoint,
    metrics_journal_iteration,
    save_checkpoint,
)
from kaggriculture.vapo import VapoConfig, make_optimizers


def test_checkpoint_round_trips_local_training_generator(tmp_path) -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    vapo_config = VapoConfig(epochs=1, minibatch_size=4, use_bfloat16=False)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, vapo_config)
    generator = np.random.default_rng(17)
    generator.random(5)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        model_config=model_config,
        vapo_config=vapo_config,
        iteration=3,
        next_seed=41,
        metrics={"score_rate": 0.75},
        source_identity=source_identity(),
        training_rng_state=generator.bit_generator.state,
        training_data_config={"games": 112},
        league_snapshot_manifest={0: "a" * 64, 3: "b" * 64},
    )
    expected = generator.random(8)

    payload = load_checkpoint(
        path,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        device=torch.device("cpu"),
    )
    restored = np.random.default_rng()
    restored.bit_generator.state = payload["training_rng"]

    assert payload["iteration"] == 3
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["next_seed"] == 41
    assert payload["training_data_config"] == {"games": 112}
    assert payload["league_snapshot_manifest"] == {0: "a" * 64, 3: "b" * 64}
    assert payload["source_identity"] == source_identity()
    assert restored.random(8).tolist() == expected.tolist()


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4])
def test_checkpoint_rejects_incompatible_format(tmp_path, version) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"format_version": version}, path)
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )

    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(
            path,
            FarmActor(model_config),
            DistributionalCritic(model_config),
            device=torch.device("cpu"),
        )


def test_iteration_metrics_journal_is_idempotent_and_conflict_detecting(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    first = {"iteration": 1, "loss": 0.5}
    second = {"iteration": 2, "loss": 0.25}

    assert append_iteration_jsonl(path, first)
    assert not append_iteration_jsonl(path, first)
    assert append_iteration_jsonl(path, second)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    with pytest.raises(ValueError, match="conflicts"):
        append_iteration_jsonl(path, {"iteration": 2, "loss": 9.0})
    with pytest.raises(ValueError, match="ahead"):
        append_iteration_jsonl(path, first)


def test_iteration_metrics_journal_recovers_an_unterminated_crash_suffix(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_iteration_jsonl(path, {"iteration": 1, "loss": 0.5})
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"iteration": 2, "loss"')

    assert append_iteration_jsonl(path, {"iteration": 2, "loss": 0.25})

    assert [json.loads(line)["iteration"] for line in path.read_text().splitlines()] == [1, 2]
    assert metrics_journal_iteration(path) == 2


def test_iteration_metrics_journal_supports_a_portable_contiguous_suffix(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"

    assert append_iteration_jsonl(path, {"iteration": 100, "loss": 1.0})
    assert append_iteration_jsonl(path, {"iteration": 101, "loss": 0.5})
    assert metrics_journal_iteration(path) == 101

    with pytest.raises(ValueError, match="missing iterations"):
        append_iteration_jsonl(path, {"iteration": 103, "loss": 0.25})
