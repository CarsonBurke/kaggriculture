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
    CRITIC_GAE_LAMBDA,
    DEFAULT_ACTOR_GAE_LAMBDA,
    VapoConfig,
    _balanced_minibatch_slices,
    _clipped_surrogate_sums,
    _stage_tensor,
    _validate_config,
    _validate_staged_action_masks,
    generalized_advantage_and_targets,
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
    device = torch.device("cpu")
    staged = {
        name: _stage_tensor(getattr(rollout, name), device)
        for name in (
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "unit_masks",
            "market_kind_masks",
            "market_quantity_masks",
        )
    }
    valid = torch.from_numpy(rollout.valid.reshape(-1)).to(device)

    with pytest.raises(ValueError, match="unit mask has no valid category"):
        _validate_staged_action_masks(staged, valid)


def test_default_gae_matches_fixed_horizon_vapo() -> None:
    config = VapoConfig()

    assert config.gamma == 1.0
    assert config.actor_gae_lambda == pytest.approx(1.0 - 1.0 / (0.05 * 719.0))
    assert config.actor_gae_lambda == pytest.approx(699.0 / 719.0)
    assert config.actor_gae_lambda == DEFAULT_ACTOR_GAE_LAMBDA
    assert 1.0 / (1.0 - config.actor_gae_lambda) == pytest.approx(0.05 * 719.0)
    assert CRITIC_GAE_LAMBDA == 1.0


def test_discounted_bank_delta_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="undiscounted gamma=1"):
        _validate_config(VapoConfig(gamma=0.99))


def test_minibatches_are_balanced_without_dropping_the_tail() -> None:
    slices = _balanced_minibatch_slices(230_080, 2048)
    sizes = [row.stop - row.start for row in slices]

    assert len(slices) == 113
    assert sum(sizes) == 230_080
    assert max(sizes) <= 2048
    assert max(sizes) - min(sizes) <= 1
    assert slices[0].start == 0
    assert slices[-1].stop == 230_080


def test_actor_lambda_decays_terminal_residual_but_critic_target_does_not() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros_like(rewards)
    valid = torch.ones_like(rewards)

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, actor_gae_lambda=0.5
    )

    assert advantages.tolist() == [[0.25, 0.5, 1.0]]
    assert targets.tolist() == [[1.0, 1.0, 1.0]]


def test_exact_mc_preserves_dense_bank_delta_for_every_state() -> None:
    potentials = torch.tensor([[0.2, -0.1, 0.4, 0.3, 0.6]])
    rewards = potentials[:, 1:] - potentials[:, :-1]
    valid = torch.ones_like(rewards)

    values = torch.zeros_like(rewards)
    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, actor_gae_lambda=0.5
    )
    expected = potentials[:, -1:] - potentials[:, :-1]

    torch.testing.assert_close(targets, expected)
    assert not torch.equal(advantages, expected)


def test_dense_gae_matches_reference_recurrence_at_scale() -> None:
    generator = torch.Generator().manual_seed(17)
    rewards = torch.randn(4, 719, generator=generator)
    values = torch.randn(4, 719, generator=generator)
    valid = torch.ones_like(rewards)
    gamma, actor_gae_lambda = 1.0, DEFAULT_ACTOR_GAE_LAMBDA

    advantages, targets = generalized_advantage_and_targets(
        rewards,
        values,
        valid,
        actor_gae_lambda=actor_gae_lambda,
        gamma=gamma,
    )

    deltas = rewards.clone()
    deltas[:, :-1] += gamma * values[:, 1:]
    deltas -= values
    expected = torch.empty_like(deltas)
    running = torch.zeros(deltas.size(0))
    for step in range(deltas.size(1) - 1, -1, -1):
        running = deltas[:, step] + gamma * actor_gae_lambda * running
        expected[:, step] = running

    torch.testing.assert_close(advantages, expected)
    expected_targets = torch.empty_like(rewards)
    running_return = torch.zeros(rewards.size(0))
    for step in range(rewards.size(1) - 1, -1, -1):
        running_return = rewards[:, step] + gamma * running_return
        expected_targets[:, step] = running_return
    torch.testing.assert_close(targets, expected_targets)


def test_actor_gae_and_critic_monte_carlo_targets_are_decoupled() -> None:
    rewards = torch.tensor([[0.2, -0.1, 0.3]])
    values = torch.tensor([[0.4, 0.1, -0.2]])
    valid = torch.ones_like(rewards)

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, actor_gae_lambda=0.5, gamma=0.9
    )

    delta_2 = 0.3 - (-0.2)
    delta_1 = -0.1 + 0.9 * (-0.2) - 0.1
    delta_0 = 0.2 + 0.9 * 0.1 - 0.4
    expected_2 = delta_2
    expected_1 = delta_1 + 0.9 * 0.5 * expected_2
    expected_0 = delta_0 + 0.9 * 0.5 * expected_1
    expected_advantages = torch.tensor([[expected_0, expected_1, expected_2]])
    expected_targets = torch.tensor([[0.2 + 0.9 * (-0.1 + 0.9 * 0.3), -0.1 + 0.9 * 0.3, 0.3]])
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(targets, expected_targets)
    assert not torch.equal(targets, expected_advantages + values)


def test_critic_targets_are_independent_of_actor_lambda_and_old_values() -> None:
    rewards = torch.tensor([[0.25, -0.4, 0.6], [-0.1, 0.2, -0.3]])
    valid = torch.ones_like(rewards)
    first = generalized_advantage_and_targets(
        rewards,
        torch.tensor([[10.0, -7.0, 3.0], [4.0, 1.0, -8.0]]),
        valid,
        actor_gae_lambda=0.1,
    )[1]
    second = generalized_advantage_and_targets(
        rewards,
        torch.tensor([[-2.0, 6.0, 9.0], [-5.0, 11.0, 0.5]]),
        valid,
        actor_gae_lambda=0.99,
    )[1]

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first,
        torch.tensor([[0.45, 0.2, 0.6], [-0.2, -0.1, -0.3]]),
    )


def test_masked_gae_does_not_bootstrap_through_padding() -> None:
    rewards = torch.tensor([[0.0, 1.0, 100.0], [0.0, 0.0, -1.0]])
    values = torch.tensor([[0.25, 0.5, 99.0], [0.1, 0.2, 0.3]])
    valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, actor_gae_lambda=1.0
    )

    torch.testing.assert_close(targets[0], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(advantages[0], torch.tensor([0.75, 0.5, 0.0]))
    torch.testing.assert_close(targets[1], torch.tensor([-1.0, -1.0, -1.0]))


def test_masked_gae_ignores_nonfinite_padding_but_rejects_nonfinite_valid_data() -> None:
    rewards = torch.tensor([[0.0, 1.0, float("nan")]])
    values = torch.tensor([[0.25, 0.5, float("inf")]])
    valid = torch.tensor([[True, True, False]])

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, actor_gae_lambda=1.0
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.75, 0.5, 0.0]]))
    torch.testing.assert_close(targets, torch.tensor([[1.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="valid rewards must be finite"):
        generalized_advantage_and_targets(rewards, values, torch.ones_like(valid))


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
    assert metrics["actor_gae_lambda"] == config.actor_gae_lambda
    assert metrics["critic_gae_lambda"] == 1.0
    assert metrics["gamma"] == config.gamma
    value_targets = generalized_advantage_and_targets(
        torch.from_numpy(rollout.rewards),
        torch.from_numpy(rollout.old_values),
        torch.from_numpy(rollout.valid),
        actor_gae_lambda=config.actor_gae_lambda,
        gamma=config.gamma,
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
