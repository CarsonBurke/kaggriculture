from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.rollout import collect_self_play
from kaggriculture.vapo import (
    VapoConfig,
    _clipped_surrogate_sums,
    _validate_rollout_action_masks,
    generalized_advantage_and_targets,
    length_adaptive_lambda,
    make_optimizers,
    update_vapo,
)


def test_vapo_config_has_no_entropy_bonus() -> None:
    assert "entropy_coefficient" not in asdict(VapoConfig())


def test_rollout_action_masks_are_validated_once_before_replay() -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(
        actor, critic, games=1, seed_start=89, episode_steps=3, sampling_seed=2
    )
    rollout.unit_masks[0, 0, 0].fill(False)

    with pytest.raises(ValueError, match="unit mask has no valid category"):
        _validate_rollout_action_masks(rollout)


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
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
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

    restored_actor = FarmActor(model_config)
    restored_critic = DistributionalCritic(model_config)
    restored_actor_optimizer, _ = make_optimizers(restored_actor, restored_critic, config)
    restored_actor_optimizer.load_state_dict(actor_optimizer.state_dict())
    restored_group = restored_actor_optimizer.param_groups[0]
    assert restored_group["warmup_step"] == actor_group["warmup_step"]
    assert restored_group["base_lr"] == actor_group["base_lr"]
    assert restored_group["lr"] == actor_group["lr"]


def test_unchanged_actor_replay_has_unit_importance_ratios() -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
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


def test_over_target_pre_step_kl_does_not_update_actor() -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(
        actor, critic, games=1, seed_start=93, episode_steps=3, sampling_seed=8
    )
    # Simulate behavior likelihoods from a stale policy. The unchanged actor is
    # already far beyond the configured trust region before any optimizer step.
    rollout.old_unit_logprobs[...] -= 1.0
    rollout.old_market_kind_logprobs[...] -= 1.0
    rollout.old_market_quantity_logprobs[...] -= 1.0
    config = VapoConfig(
        epochs=2,
        minibatch_size=8,
        target_kl=1e-4,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    critic_before = {
        name: parameter.detach().clone() for name, parameter in critic.named_parameters()
    }

    metrics = update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(9),
    )

    assert metrics["updates"] == config.epochs
    assert metrics["actor_updates"] == 0
    assert metrics["kl_early_stop"] == 1
    assert metrics["max_approx_kl"] > config.target_kl
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(parameter, critic_before[name])
        for name, parameter in critic.named_parameters()
    )


def test_one_vapo_update_is_finite() -> None:
    model_config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
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
    value_targets = generalized_advantage_and_targets(
        torch.from_numpy(rollout.rewards),
        torch.from_numpy(rollout.old_values),
        torch.from_numpy(rollout.valid),
        torch.ones(rollout.trajectories),
    )[1]
    valid_targets = value_targets[torch.from_numpy(rollout.valid)]
    assert metrics["value_target_min"] == pytest.approx(float(valid_targets.min()))
    assert metrics["value_target_max"] == pytest.approx(float(valid_targets.max()))


def test_entropy_is_diagnostic_only_when_policy_advantage_is_zero() -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(
        actor, critic, games=1, seed_start=94, episode_steps=3, sampling_seed=10
    )
    rollout.rewards.fill(0.0)
    rollout.old_values.fill(0.0)
    config = VapoConfig(
        epochs=1,
        minibatch_size=rollout.states,
        lr_warmup_steps=0,
        weight_decay=0.0,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}

    metrics = update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(11),
    )

    assert metrics["actor_updates"] == 1
    assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["entropy"] > 0.0
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)
