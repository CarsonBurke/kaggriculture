from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.rollout import collect_self_play
from kaggriculture.vapo import (
    VapoConfig,
    _clipped_surrogate_sums,
    generalized_advantage_and_targets,
    length_adaptive_lambda,
    make_optimizers,
    update_vapo,
)


def test_default_lambda_is_exact_monte_carlo() -> None:
    values = length_adaptive_lambda(torch.tensor([1, 2, 5, 100, 720]))

    assert values.tolist() == [1.0] * 5


def test_length_adaptive_lambda_remains_an_explicit_ablation() -> None:
    values = length_adaptive_lambda(torch.tensor([1, 2, 5, 100, 720]), alpha=0.5)

    assert values.tolist() == pytest.approx([0.0, 0.5, 0.6, 0.98, 1.0 - 1.0 / 360.0])


def test_gae_terminal_reward_reaches_opening() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros_like(rewards)
    valid = torch.ones_like(rewards)

    advantages, targets = generalized_advantage_and_targets(rewards, values, valid, torch.ones(1))

    assert advantages.tolist() == [[1.0, 1.0, 1.0]]
    assert targets.tolist() == [[1.0, 1.0, 1.0]]


def test_exact_mc_preserves_telescoping_shaping_for_every_state() -> None:
    # Potential-shaped rewards telescope to outcome minus the current
    # potential. At the symmetric opening the potential is zero, so the actor
    # receives the raw terminal outcome across the complete 719-step horizon.
    potentials = torch.tensor([[0.2, -0.1, 0.4, 0.3]])
    outcome = torch.tensor([[-1.0]])
    rewards = torch.cat(
        (potentials[:, 1:] - potentials[:, :-1], outcome - potentials[:, -1:]), dim=1
    )
    valid = torch.ones_like(rewards)

    values = torch.zeros_like(rewards)
    advantages, targets = generalized_advantage_and_targets(rewards, values, valid, torch.ones(1))
    expected = outcome - potentials

    torch.testing.assert_close(targets, expected)
    torch.testing.assert_close(advantages, expected)


def test_masked_gae_does_not_bootstrap_through_padding() -> None:
    rewards = torch.tensor([[0.0, 1.0, 100.0], [0.0, 0.0, -1.0]])
    values = torch.tensor([[0.25, 0.5, 99.0], [0.1, 0.2, 0.3]])
    valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    advantages, targets = generalized_advantage_and_targets(rewards, values, valid, torch.ones(2))

    torch.testing.assert_close(targets[0], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(advantages[0], torch.tensor([0.75, 0.5, 0.0]))
    torch.testing.assert_close(targets[1], torch.tensor([-1.0, -1.0, -1.0]))


def test_asymmetric_clipping_leaves_harmful_direction_unclipped() -> None:
    old = torch.zeros(2, 1)
    new = torch.tensor([[math.log(2.0)], [math.log(2.0)]])
    advantages = torch.tensor([1.0, -1.0])
    active = torch.ones_like(old)

    objective, approximate_kl, clipped = _clipped_surrogate_sums(
        new, old, advantages, active, 0.80, 1.28
    )

    # Positive advantage is clipped to 1.28; the harmful negative-advantage
    # move stays at ratio 2.0 so PPO retains the corrective gradient.
    assert objective.item() == pytest.approx(1.28 - 2.0)
    assert approximate_kl.item() == pytest.approx(2.0 * (1.0 - math.log(2.0)))
    assert clipped.item() == 2.0


def test_optimizer_warmup_is_checkpointed_in_param_group() -> None:
    model_config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    config = VapoConfig(
        epochs=1,
        minibatch_size=8,
        lr_warmup_steps=4,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    rollout = collect_self_play(
        actor, critic, games=1, seed_start=91, episode_steps=3, sampling_seed=4
    )

    update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(5),
    )

    actor_group = actor_optimizer.param_groups[0]
    assert actor_group["warmup_step"] == 1
    assert actor_group["base_lr"] == config.actor_learning_rate
    assert actor_group["lr"] == pytest.approx(config.actor_learning_rate / 4)


def test_unchanged_actor_replay_has_unit_importance_ratios() -> None:
    model_config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(
        actor, critic, games=1, seed_start=92, episode_steps=3, sampling_seed=6
    )
    flat = rollout.valid.reshape(-1)
    board = torch.from_numpy(rollout.board.reshape(-1, *rollout.board.shape[2:])[flat]).float()
    global_features = torch.from_numpy(
        rollout.global_features.reshape(-1, *rollout.global_features.shape[2:])[flat]
    ).float()
    units = torch.from_numpy(rollout.units.reshape(-1, *rollout.units.shape[2:])[flat]).float()
    positions = torch.from_numpy(
        rollout.unit_positions.reshape(-1, *rollout.unit_positions.shape[2:])[flat]
    ).long()

    output = actor(board, global_features, units, positions)
    market_kinds = torch.from_numpy(
        rollout.market_kinds.reshape(-1, *rollout.market_kinds.shape[2:])[flat]
    ).long()
    replayed = component_logprobs(
        output,
        actor.quantity_logits(output.market_quantity_context, market_kinds),
        torch.from_numpy(
            rollout.unit_actions.reshape(-1, *rollout.unit_actions.shape[2:])[flat]
        ).long(),
        market_kinds,
        torch.from_numpy(
            rollout.market_quantities.reshape(-1, *rollout.market_quantities.shape[2:])[flat]
        ).long(),
        torch.from_numpy(rollout.unit_masks.reshape(-1, *rollout.unit_masks.shape[2:])[flat]),
        torch.from_numpy(
            rollout.market_kind_masks.reshape(-1, *rollout.market_kind_masks.shape[2:])[flat]
        ),
        torch.from_numpy(
            rollout.market_quantity_masks.reshape(-1, *rollout.market_quantity_masks.shape[2:])[
                flat
            ]
        ),
    )[:3]
    behavior = (
        rollout.old_unit_logprobs.reshape(-1, *rollout.old_unit_logprobs.shape[2:])[flat],
        rollout.old_market_kind_logprobs.reshape(-1, *rollout.old_market_kind_logprobs.shape[2:])[
            flat
        ],
        rollout.old_market_quantity_logprobs.reshape(
            -1, *rollout.old_market_quantity_logprobs.shape[2:]
        )[flat],
    )

    for new_logprobs, old_logprobs in zip(replayed, behavior, strict=True):
        ratios = (new_logprobs - torch.from_numpy(old_logprobs)).exp()
        torch.testing.assert_close(ratios, torch.ones_like(ratios), atol=1e-5, rtol=1e-5)


def test_one_vapo_update_is_finite() -> None:
    model_config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(
        actor, critic, games=2, seed_start=90, episode_steps=8, sampling_seed=3
    )
    config = VapoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    metrics = update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(4),
    )

    assert metrics["updates"] == 4
    assert metrics["epochs"] == 1
    assert math.isfinite(metrics["policy_loss"])
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["lambda_mean"] == 1.0
