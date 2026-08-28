from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.ppo import PpoConfig, make_optimizers
from kaggriculture.provenance import source_identity
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic
from kaggriculture.structured_dynamics import StructuredDynamics
from kaggriculture.training import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingAgent,
    append_iteration_jsonl,
    load_checkpoint,
    metrics_journal_iteration,
    replace_checkpoint_alias,
    save_checkpoint,
    write_immutable_checkpoint,
)


def test_checkpoint_round_trips_local_training_generator(tmp_path) -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    ppo_config = PpoConfig(epochs=1, minibatch_size=4, use_bfloat16=False)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
    generator = np.random.default_rng(17)
    generator.random(5)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        agents=[TrainingAgent(actor, critic, actor_optimizer, critic_optimizer)],
        model_config=model_config,
        ppo_config=ppo_config,
        iteration=3,
        next_seed=41,
        metrics={"score_rate": 0.75},
        source_identity=source_identity(),
        training_rng_state=generator.bit_generator.state,
        training_data_config={"games": 112},
        league_snapshot_manifest={0: "a" * 64, 3: "b" * 64},
        league_score_rates={1: 0.5, 3: 0.75},
    )
    expected = generator.random(8)

    payload = load_checkpoint(
        path,
        [TrainingAgent(actor, critic, actor_optimizer, critic_optimizer)],
        device=torch.device("cpu"),
    )
    restored = np.random.default_rng()
    restored.bit_generator.state = payload["training_rng"]

    assert payload["iteration"] == 3
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["next_seed"] == 41
    assert payload["training_data_config"] == {"games": 112}
    assert payload["league_snapshot_manifest"] == {0: "a" * 64, 3: "b" * 64}
    assert payload["league_score_rates"] == {1: 0.5, 3: 0.75}
    assert payload["source_identity"] == source_identity()
    assert restored.random(8).tolist() == expected.tolist()


def test_structured_auxiliary_recovery_round_trips_predictor_rng_and_optimizer(
    tmp_path,
) -> None:
    model_config = StructuredConfig(
        model_dim=16,
        attention_heads=2,
        ffn_multiplier=1,
        farm_blocks=1,
        opponent_latents=2,
        latents=4,
        core_layers=1,
        quantity_rank=4,
    )
    ppo_config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=4,
        use_bfloat16=False,
        structured_decision_coefficient=0.5,
        structured_opponent_summary_coefficient=0.5,
        structured_opponent_patch_coefficient=0.5,
    )
    actor = StructuredActor(model_config)
    critic = StructuredCritic(model_config)
    dynamics = StructuredDynamics(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(
        actor,
        critic,
        ppo_config,
        structured_dynamics=dynamics,
    )
    actor_optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().mean() for parameter in dynamics.parameters()).backward()
    actor_optimizer.step()
    auxiliary_generator = np.random.default_rng(43)
    auxiliary_state = auxiliary_generator.bit_generator.state
    expected = auxiliary_generator.random(5)
    path = tmp_path / "structured.pt"

    save_checkpoint(
        path,
        agents=[
            TrainingAgent(
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                dynamics,
            )
        ],
        model_config=model_config,
        ppo_config=ppo_config,
        iteration=2,
        next_seed=9,
        metrics={},
        source_identity=source_identity(),
        auxiliary_rng_state=auxiliary_state,
    )

    restored_actor = StructuredActor(model_config)
    restored_critic = StructuredCritic(model_config)
    restored_dynamics = StructuredDynamics(model_config)
    restored_actor_optimizer, restored_critic_optimizer = make_optimizers(
        restored_actor,
        restored_critic,
        ppo_config,
        structured_dynamics=restored_dynamics,
    )
    restored_agent = TrainingAgent(
        restored_actor,
        restored_critic,
        restored_actor_optimizer,
        restored_critic_optimizer,
        restored_dynamics,
    )
    payload = load_checkpoint(path, [restored_agent], device=torch.device("cpu"))
    restored_generator = np.random.default_rng()
    restored_generator.bit_generator.state = payload["structured_auxiliary_rng"]

    for name, value in dynamics.state_dict().items():
        torch.testing.assert_close(restored_dynamics.state_dict()[name], value)
    assert restored_generator.random(5).tolist() == expected.tolist()
    assert restored_actor_optimizer.state_dict()["state"]
    assert len(restored_actor_optimizer.state_dict()["state"]) == len(
        actor_optimizer.state_dict()["state"]
    )

    inactive_config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=4,
        use_bfloat16=False,
    )
    inactive_actor = StructuredActor(model_config)
    inactive_critic = StructuredCritic(model_config)
    inactive_optimizers = make_optimizers(
        inactive_actor,
        inactive_critic,
        inactive_config,
    )
    with pytest.raises(ValueError, match="auxiliary RNG presence"):
        load_checkpoint(
            path,
            [TrainingAgent(inactive_actor, inactive_critic, *inactive_optimizers)],
            device=torch.device("cpu"),
        )

    missing_path = tmp_path / "missing-structured.pt"
    missing_payload = torch.load(path, weights_only=False)
    missing_payload.pop("structured_dynamics")
    missing_payload.pop("structured_auxiliary_rng")
    torch.save(missing_payload, missing_path)
    with pytest.raises(ValueError, match="auxiliary RNG presence"):
        load_checkpoint(
            missing_path,
            [restored_agent],
            device=torch.device("cpu"),
        )


def test_zero_auxiliary_checkpoint_keeps_the_historical_state_shape(tmp_path) -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    ppo_config = PpoConfig(epochs=1, minibatch_size=4, use_bfloat16=False)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    optimizers = make_optimizers(actor, critic, ppo_config)
    path = tmp_path / "plain.pt"

    save_checkpoint(
        path,
        agents=[TrainingAgent(actor, critic, *optimizers)],
        model_config=model_config,
        ppo_config=ppo_config,
        iteration=0,
        next_seed=0,
        metrics={},
        source_identity=source_identity(),
    )
    payload = torch.load(path, weights_only=False)

    assert "structured_dynamics" not in payload
    assert "structured_auxiliary_rng" not in payload


def test_latest_alias_atomically_tracks_immutable_regular_checkpoints(tmp_path) -> None:
    first = tmp_path / "checkpoint-000001.pt"
    second = tmp_path / "checkpoint-000002.pt"
    latest = tmp_path / "latest.pt"
    write_immutable_checkpoint(first, {"iteration": 1})
    write_immutable_checkpoint(second, {"iteration": 2})

    assert replace_checkpoint_alias(first, latest)
    assert latest.is_file() and not latest.is_symlink()
    assert latest.stat().st_ino == first.stat().st_ino
    assert not replace_checkpoint_alias(first, latest)

    assert replace_checkpoint_alias(second, latest)
    assert latest.stat().st_ino == second.stat().st_ino
    assert torch.load(latest, weights_only=False) == {"iteration": 2}
    assert torch.load(first, weights_only=False) == {"iteration": 1}
    with pytest.raises(FileExistsError):
        write_immutable_checkpoint(first, {"iteration": 99})


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_checkpoint_rejects_incompatible_format(tmp_path, version) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"format_version": version}, path)
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )

    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(
            path,
            [TrainingAgent(FarmActor(model_config), DistributionalCritic(model_config))],
            device=torch.device("cpu"),
        )


def test_a_checkpoints_update_mode_record_must_match_its_ppo_config() -> None:
    """The record and the config are independent parameters, so nothing else
    makes them agree, and a checkpoint whose record names one update mode while
    its config names another carries provenance for a run that did not happen.

    Compared by value rather than by identity: the knob is a mode string now, and
    a string read back out of a checkpoint or a JSON decision is not the same
    object as the interned literal the config was built from. An `is not`
    comparison here would reject agreeing pairs -- which the non-interned
    spelling below is built to catch.
    """
    from kaggriculture.training import checkpoint_payload

    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    ppo_config = PpoConfig(epochs=1, minibatch_size=4, update_compile_mode="max-autotune")
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
    rng_states = dict.fromkeys(("torch_rng", "cuda_rng", "numpy_rng", "python_rng"))

    def build(recorded: str):
        return checkpoint_payload(
            agents=[
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                }
            ],
            model_config=model_config,
            ppo_config=ppo_config,
            iteration=1,
            next_seed=2,
            metrics={},
            source_identity=source_identity(),
            rng_states=rng_states,
            training_data_config={"update_compile_mode": recorded},
        )

    # Built at runtime rather than written as a literal, so it is a distinct
    # object from the one `PpoConfig` holds: an identity comparison would reject
    # this agreeing pair, which is the regression this line exists to catch.
    agreeing = "-".join(("max", "autotune"))
    assert build(agreeing)["training_data_config"]["update_compile_mode"] == "max-autotune"

    with pytest.raises(ValueError, match="update_compile_mode does not match its ppo config"):
        build("default")


def test_checkpoint_load_rejects_a_foreign_model_before_touching_the_actor(tmp_path) -> None:
    """A resume whose checkpoint was written by another architecture, or by
    the same architecture at another capacity, must fail on model identity --
    and must fail before strict loading has already replaced any weights."""
    from kaggriculture.registry import architecture_of_config
    from kaggriculture.structured import StructuredConfig

    ppo_config = PpoConfig(epochs=1, minibatch_size=4, use_bfloat16=False)
    conv = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )

    def store(path, model_config) -> None:
        architecture = architecture_of_config(model_config)
        actor = architecture.actor_class(model_config)
        critic = architecture.critic_class(model_config)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
        save_checkpoint(
            path,
            agents=[TrainingAgent(actor, critic, actor_optimizer, critic_optimizer)],
            model_config=model_config,
            ppo_config=ppo_config,
            iteration=1,
            next_seed=2,
            metrics={},
            source_identity=source_identity(),
        )

    foreign_family = tmp_path / "structured.pt"
    store(
        foreign_family,
        StructuredConfig(
            model_dim=16,
            attention_heads=2,
            ffn_multiplier=1,
            farm_blocks=1,
            opponent_latents=2,
            latents=4,
            core_layers=1,
            quantity_rank=4,
        ),
    )
    foreign_capacity = tmp_path / "wider-conv.pt"
    store(
        foreign_capacity,
        ModelConfig(
            cnn_width=16, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
        ),
    )

    actor = FarmActor(conv)
    critic = DistributionalCritic(conv)
    before = {name: value.clone() for name, value in actor.state_dict().items()}

    with pytest.raises(ValueError, match="checkpoint architecture does not match"):
        load_checkpoint(foreign_family, [TrainingAgent(actor, critic)], device=torch.device("cpu"))
    with pytest.raises(ValueError, match="checkpoint model configuration does not match"):
        load_checkpoint(
            foreign_capacity, [TrainingAgent(actor, critic)], device=torch.device("cpu")
        )

    assert all(torch.equal(value, before[name]) for name, value in actor.state_dict().items())


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


def test_iteration_metrics_journal_rejects_a_blank_final_line(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_iteration_jsonl(path, {"iteration": 1, "loss": 0.5})
    with path.open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(ValueError, match="invalid final record"):
        append_iteration_jsonl(path, {"iteration": 2, "loss": 0.25})
