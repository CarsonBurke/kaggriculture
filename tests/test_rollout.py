from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.rollout import (
    collect_frozen_opponent_play,
    collect_self_play,
    concatenate_rollouts,
)


def test_short_self_play_rollout_shapes_and_telescoping() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)

    rollout = collect_self_play(
        actor,
        critic,
        games=2,
        seed_start=50,
        episode_steps=8,
        deterministic=False,
        sampling_seed=9,
    )

    assert rollout.trajectories == 4
    assert rollout.horizon == 7
    assert rollout.states == 28
    assert rollout.board.shape[:2] == (4, 7)
    assert rollout.unit_masks.shape[:2] == (4, 7)
    outcomes = (rollout.final_money > rollout.opponent_money).astype(float)
    outcomes -= (rollout.final_money < rollout.opponent_money).astype(float)
    assert rollout.rewards.sum(axis=1).tolist() == pytest.approx(outcomes.tolist(), abs=1e-6)
    assert rollout.seats.tolist() == [0, 1, 0, 1]
    assert rollout.episode_seeds.tolist() == [50, 50, 51, 51]
    np.testing.assert_allclose(rollout.rewards[0], -rollout.rewards[1], atol=1e-7)
    np.testing.assert_allclose(rollout.rewards[2], -rollout.rewards[3], atol=1e-7)


def test_frozen_opponent_rollout_and_concatenation() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    opponent = FarmActor(config)
    opponent.load_state_dict(actor.state_dict())
    self_play = collect_self_play(
        actor, critic, games=1, seed_start=70, episode_steps=8, sampling_seed=1
    )
    league = collect_frozen_opponent_play(
        actor,
        critic,
        opponent,
        games=2,
        seed_start=80,
        episode_steps=8,
        sampling_seed=2,
    )

    combined = concatenate_rollouts([self_play, league])

    assert league.trajectories == 2
    assert league.seats.tolist() == [0, 1]
    assert combined.trajectories == 4
    assert combined.horizon == 7


def test_stored_behavior_likelihoods_replay_from_identical_features() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    rollout = collect_self_play(
        actor, critic, games=2, seed_start=110, episode_steps=8, sampling_seed=5
    )

    def flatten(values):
        return torch.from_numpy(values.reshape(-1, *values.shape[2:]))

    with torch.inference_mode():
        output = actor(
            flatten(rollout.board).float(),
            flatten(rollout.global_features).float(),
            flatten(rollout.units).float(),
            flatten(rollout.unit_positions).long(),
        )
        quantity_logits = actor.quantity_logits(
            output.market_quantity_context,
            flatten(rollout.market_kinds).long(),
        )
        unit, kind, quantity, *_ = component_logprobs(
            output,
            quantity_logits,
            flatten(rollout.unit_actions).long(),
            flatten(rollout.market_kinds).long(),
            flatten(rollout.market_quantities).long(),
            flatten(rollout.unit_masks).bool(),
            flatten(rollout.market_kind_masks).bool(),
            flatten(rollout.market_quantity_masks).bool(),
        )
        replay_values = critic.value(
            critic(flatten(rollout.board).float(), flatten(rollout.critic_features).float())
        )

    old_unit = flatten(rollout.old_unit_logprobs)
    old_kind = flatten(rollout.old_market_kind_logprobs)
    old_quantity = flatten(rollout.old_market_quantity_logprobs)
    unit_active = flatten(rollout.unit_active).bool()
    kind_active = flatten(rollout.market_active).bool()
    quantity_active = flatten(rollout.market_quantity_active).bool()
    np.testing.assert_allclose(unit[unit_active], old_unit[unit_active], atol=2e-6)
    np.testing.assert_allclose(kind[kind_active], old_kind[kind_active], atol=2e-6)
    np.testing.assert_allclose(quantity[quantity_active], old_quantity[quantity_active], atol=2e-6)
    np.testing.assert_allclose(replay_values, rollout.old_values.reshape(-1), atol=2e-6)
