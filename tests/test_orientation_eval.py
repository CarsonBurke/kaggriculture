"""Orientation plumbing through evaluation surfaces: act_batch, checkpoints, artifacts."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from kaggle_environments import make

from kaggriculture.actions import UnitAction
from kaggriculture.encoding import EncodedObservation, encode_observation
from kaggriculture.inference import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointAgent,
    actor_artifact_from_checkpoint,
)
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.orientation import (
    MEMBER_ORIENTATIONS,
    Orientation,
    flip_board,
    member_orientation,
    movement_permutation,
    orient_unit_features,
    orient_unit_masks,
)
from kaggriculture.ppo import PpoConfig, make_optimizers
from kaggriculture.policy import _sample_numpy_categorical, act_batch, stack_encoded
from kaggriculture.provenance import source_identity
from kaggriculture.training import (
    TrainingAgent,
    checkpoint_member_orientations,
    checkpoint_payload,
    load_checkpoint,
    require_checkpoint_format,
    save_checkpoint,
)

_MOVE_NAMES = {
    int(UnitAction.NORTH): "NORTH",
    int(UnitAction.SOUTH): "SOUTH",
    int(UnitAction.EAST): "EAST",
    int(UnitAction.WEST): "WEST",
}


def _small_config() -> ModelConfig:
    return ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )


def _observations(seed: int = 37) -> list[dict]:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": seed})
    return [row.observation for row in environment.reset(2)]


def _pin_unit_head_to_moves_and_pass(actor: FarmActor) -> None:
    """Make the decode choose EAST where legal and PASS everywhere else.

    Neither choice mutates the evolving mask state -- no seeds are planted, no
    shed or tile effects fire -- so the real movement masks stay frozen across
    units and steps. That is what lets the reference composition below be
    exact for every unit instead of only the first one.
    """
    with torch.no_grad():
        bias = actor.unit_head[-1].bias
        bias.fill_(-12.0)
        bias[UnitAction.PASS] = -6.0
        bias[UnitAction.EAST] = 6.0


def _copy_encoded(row: EncodedObservation) -> EncodedObservation:
    """A mutable twin of one encoded row, leaving the caller's original alone."""
    return EncodedObservation(
        board=row.board.copy(),
        global_features=row.global_features.copy(),
        critic_features=row.critic_features.copy(),
        units=row.units.copy(),
        unit_positions=row.unit_positions.copy(),
        unit_active=row.unit_active,
    )


def _flip_encoded(row: EncodedObservation, orientation: Orientation) -> EncodedObservation:
    flipped = _copy_encoded(row)
    flip_board(flipped.board, orientation)
    orient_unit_features(
        flipped.units[np.newaxis],
        flipped.unit_positions[np.newaxis],
        np.array([int(orientation)], dtype=np.int8),
    )
    return flipped


def test_act_batch_under_mirror_x_equals_the_flipped_forward_composition() -> None:
    """The documented composition, checked against a hand-rolled reference.

    Orienting must be exactly: flip the encoded surfaces, forward, decode over
    the restriped movement columns, map every chosen index back through the
    orientation's permutation. The factor arrays stay oriented -- they pair
    with the stored features a replay re-forwards -- while the engine-facing
    commands carry the real movements.
    """
    orientation = Orientation.MIRROR_X
    codes = np.full(2, int(orientation), dtype=np.int8)
    perm = movement_permutation(orientation)
    actor = FarmActor(_small_config())
    _pin_unit_head_to_moves_and_pass(actor)
    observations = _observations()
    pristine_encodings = [encode_observation(observation) for observation in observations]

    baseline = act_batch(actor, observations, deterministic=True)
    mirrored = act_batch(actor, observations, deterministic=True, orientation=orientation)

    # The stored encodings are the flipped ones the forward actually consumed.
    for original, stored in zip(pristine_encodings, mirrored.encoded, strict=True):
        expected = _flip_encoded(original, orientation)
        np.testing.assert_array_equal(stored.board, flip_board(original.board.copy(), orientation))
        np.testing.assert_array_equal(stored.units, expected.units)
        np.testing.assert_array_equal(stored.unit_positions, expected.unit_positions)

    # Reference decode: the real masks restriped into oriented columns, argmax
    # over the flipped forward's logits, then back through the permutation.
    oriented_masks = orient_unit_masks(baseline.factors.unit_masks, codes)
    flipped_rows = [_flip_encoded(row, orientation) for row in pristine_encodings]
    tensors = stack_encoded(flipped_rows, torch.device("cpu"))
    with torch.inference_mode():
        output = actor(
            tensors.board, tensors.global_features, tensors.units, tensors.unit_positions
        )
    oriented_logits = output.unit_logits.float().numpy()
    expected_oriented = np.zeros_like(mirrored.factors.unit_actions)
    for unit_index in range(expected_oriented.shape[1]):
        expected_oriented[:, unit_index], _, _ = _sample_numpy_categorical(
            oriented_logits[:, unit_index],
            oriented_masks[:, unit_index],
            deterministic=True,
            temperature=1.0,
            generator=np.random.default_rng(0),
        )

    np.testing.assert_array_equal(mirrored.factors.unit_actions, expected_oriented)
    np.testing.assert_array_equal(mirrored.factors.unit_masks, oriented_masks)

    # The engine-facing commands carry the real image of every chosen movement;
    # passes and moves render statelessly, so they pin the mapping end to end.
    real_expected = perm[expected_oriented]
    for row, action in enumerate(mirrored.actions):
        commands = [action["farmer"], *action["hands"]]
        for unit_index, command in enumerate(commands):
            if not mirrored.factors.unit_active[row, unit_index]:
                continue
            real = int(real_expected[row, unit_index])
            if real == int(UnitAction.PASS):
                assert command == ["PASS"]
            elif real in _MOVE_NAMES:
                assert command == [_MOVE_NAMES[real]]

    # Where EAST was legal upright, the mirror must execute its image WEST:
    # the same oriented preference, played on the mirrored board.
    farmer_x = int(observations[0]["farms"][0]["farmer"][0])
    east_legal = baseline.factors.unit_masks[0, 0, int(UnitAction.EAST)]
    if east_legal and farmer_x < 9:
        assert baseline.actions[0]["farmer"] == ["EAST"]
        assert mirrored.actions[0]["farmer"] == ["WEST"]


def test_act_batch_stores_oriented_masks_for_asymmetric_positions() -> None:
    """Stored mask columns describe stored action indices, visibly.

    A farmer on the west edge has an asymmetric movement mask (WEST closed,
    EAST open), so the oriented restripe is distinguishable from the real
    columns and the factor contract can be pinned without relying on a
    symmetric scenario to make the two coincide.
    """
    actor = FarmActor(_small_config())
    _pin_unit_head_to_moves_and_pass(actor)
    observations = _observations()
    for farm in observations[0]["farms"]:
        farm["farmer"] = [0, int(farm["farmer"][1])]
    orientation = Orientation.MIRROR_X

    upright = act_batch(actor, observations, deterministic=True)
    mirrored = act_batch(actor, observations, deterministic=True, orientation=orientation)
    expected = orient_unit_masks(
        upright.factors.unit_masks,
        np.full(upright.factors.unit_masks.shape[0], int(orientation), dtype=np.int8),
    )

    assert not np.array_equal(upright.factors.unit_masks, expected)
    np.testing.assert_array_equal(mirrored.factors.unit_masks, expected)


def test_act_batch_rejects_orientation_for_structured_actors() -> None:
    from kaggriculture.structured import StructuredActor, StructuredConfig

    actor = StructuredActor(StructuredConfig())
    with pytest.raises(NotImplementedError, match="orientation"):
        act_batch(actor, _observations(), deterministic=True, orientation=Orientation.MIRROR_X)


def test_checkpoint_round_trip_preserves_member_orientations(tmp_path) -> None:
    """Every member's code survives a save/load cycle, stamped from its index."""
    model_config = _small_config()
    ppo_config = PpoConfig(epochs=1, minibatch_size=4, use_bfloat16=False)
    agents = []
    for _ in range(2):
        actor = FarmActor(model_config)
        critic = DistributionalCritic(model_config)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
        agents.append(TrainingAgent(actor, critic, actor_optimizer, critic_optimizer))
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        agents=agents,
        model_config=model_config,
        ppo_config=ppo_config,
        iteration=3,
        next_seed=41,
        metrics={},
        source_identity=source_identity(),
    )

    payload = load_checkpoint(path, agents, device=torch.device("cpu"))
    assert checkpoint_member_orientations(payload) == [
        member_orientation(index) for index in range(len(agents))
    ]
    assert [member["orientation"] for member in payload["agents"]] == [
        int(value) for value in MEMBER_ORIENTATIONS[:2]
    ]

    # A single learner keeps its code at the top level, beside its states.
    single = checkpoint_payload(
        agents=[agent.state() for agent in agents[:1]],
        model_config=model_config,
        ppo_config=ppo_config,
        iteration=0,
        next_seed=0,
        metrics={},
        source_identity=source_identity(),
        rng_states={
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
    )
    require_checkpoint_format(single)
    assert single["orientation"] == int(Orientation.IDENTITY)


def test_resume_treats_an_absent_orientation_as_identity(tmp_path) -> None:
    """A payload without orientation codes loads and reads as identity."""
    model_config = _small_config()
    ppo_config = PpoConfig(epochs=1, minibatch_size=4, use_bfloat16=False)

    def build_agents(count: int) -> list[TrainingAgent]:
        agents = []
        for _ in range(count):
            actor = FarmActor(model_config)
            critic = DistributionalCritic(model_config)
            actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
            agents.append(TrainingAgent(actor, critic, actor_optimizer, critic_optimizer))
        return agents

    writers = build_agents(2)
    payload = checkpoint_payload(
        agents=[agent.state() for agent in writers],
        model_config=model_config,
        ppo_config=ppo_config,
        iteration=0,
        next_seed=0,
        metrics={},
        source_identity=source_identity(),
        rng_states={
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
    )
    for member in payload["agents"]:
        del member["orientation"]
    require_checkpoint_format(payload)
    assert checkpoint_member_orientations(payload) == [Orientation.IDENTITY] * 2

    path = tmp_path / "legacy_shaped.pt"
    torch.save(payload, path)
    readers = build_agents(2)
    load_checkpoint(path, readers, device=torch.device("cpu"))


def test_exported_artifact_carries_the_source_members_orientation() -> None:
    """The artifact records whose rendering it ships; legacy reads as identity."""
    model_config = _small_config()
    actor = FarmActor(model_config)
    population = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": model_config.to_dict(),
        "source_identity": source_identity(),
        "agents": [
            {"actor": actor.state_dict(), "orientation": int(Orientation.MIRROR_Y)},
            {"actor": actor.state_dict(), "orientation": int(Orientation.ROTATE_180)},
        ],
    }
    assert actor_artifact_from_checkpoint(population, agent=0)["orientation"] == int(
        Orientation.MIRROR_Y
    )
    assert actor_artifact_from_checkpoint(population, agent=1)["orientation"] == int(
        Orientation.ROTATE_180
    )

    legacy = {
        "format_version": CHECKPOINT_FORMAT_VERSION - 1,
        "model_config": model_config.to_dict(),
        "actor": actor.state_dict(),
        "source_identity": source_identity(),
    }
    artifact = actor_artifact_from_checkpoint(legacy)
    assert artifact["orientation"] == int(Orientation.IDENTITY)


def test_checkpoint_agent_plays_under_the_recorded_orientation(tmp_path) -> None:
    model_config = _small_config()
    actor = FarmActor(model_config)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": model_config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 1,
            "metrics": {},
            "source_identity": source_identity(),
            "orientation": int(Orientation.MIRROR_X),
        },
        path,
    )

    assert CheckpointAgent(path).orientation == Orientation.MIRROR_X
