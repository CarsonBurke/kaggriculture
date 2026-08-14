from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

import kaggriculture.vapo
from kaggriculture.actions import MarketKind
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.rollout import collect_self_play
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic
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
    replay_behavior_values,
    update_replay_parity,
    update_vapo,
)


def test_vapo_config_has_no_entropy_bonus() -> None:
    assert "entropy_coefficient" not in asdict(VapoConfig())


def test_rollout_action_masks_are_validated_once_before_replay() -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=89, episode_steps=3, sampling_seed=2)
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


def test_critic_targets_are_independent_of_actor_lambda_and_behavior_values() -> None:
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
    rollout = collect_self_play(actor, games=1, seed_start=91, episode_steps=3, sampling_seed=4)

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
    rollout = collect_self_play(actor, games=1, seed_start=92, episode_steps=3, sampling_seed=6)
    flat = rollout.valid.reshape(-1)
    board_states = rollout.states["board"]
    board = torch.from_numpy(board_states.reshape(-1, *board_states.shape[2:])[flat]).float()
    global_states = rollout.states["global_features"]
    global_features = torch.from_numpy(
        global_states.reshape(-1, *global_states.shape[2:])[flat]
    ).float()
    unit_states = rollout.states["units"]
    units = torch.from_numpy(unit_states.reshape(-1, *unit_states.shape[2:])[flat]).float()
    position_states = rollout.states["unit_positions"]
    positions = torch.from_numpy(
        position_states.reshape(-1, *position_states.shape[2:])[flat]
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


def test_update_replay_parity_gates_the_update_path_forward() -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    # A horizon long enough for hires and executed trades keeps every action
    # component active, so the gate covers all three heads non-vacuously.
    rollout = collect_self_play(actor, games=1, seed_start=94, episode_steps=32, sampling_seed=12)
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}

    parity = update_replay_parity(actor, rollout, minibatch_size=3)

    for component in ("unit", "kind", "quantity"):
        assert parity[f"update_replay_{component}_active_count"] > 0
        assert parity[f"update_replay_{component}_ratio_max_abs_error"] < 1e-4
    assert parity["update_replay_max_ratio_error"] == max(
        parity[f"update_replay_{component}_ratio_max_abs_error"]
        for component in ("unit", "kind", "quantity")
    )
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)

    # A behavior/update mismatch must be visible as ratio drift: stale stored
    # likelihoods shifted by log(2) produce ratios near two.
    rollout.old_market_kind_logprobs[...] -= math.log(2.0)
    rollout.old_market_quantity_logprobs[...] -= math.log(2.0)
    drifted = update_replay_parity(actor, rollout, minibatch_size=3)
    assert drifted["update_replay_kind_ratio_max_abs_error"] > 0.9
    assert drifted["update_replay_quantity_ratio_max_abs_error"] > 0.9

    with pytest.raises(ValueError, match="minibatch size"):
        update_replay_parity(actor, rollout, minibatch_size=0)


def test_update_ratio_is_pinned_to_one_regardless_of_stored_likelihoods() -> None:
    """The behavior replay makes the update immune to sampling-path numerics.

    Corrupting the rollout's stored likelihoods must not disturb the update:
    behavior likelihoods are recomputed through the update-path forward, so
    the first minibatch's importance ratio is exactly one at unchanged
    weights (identical eager function on CPU) and the actor still trains.
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=95, episode_steps=3, sampling_seed=10)
    rollout.old_unit_logprobs[...] -= 3.0
    rollout.old_market_kind_logprobs[...] -= 3.0
    rollout.old_market_quantity_logprobs[...] -= 3.0
    config = VapoConfig(epochs=1, minibatch_size=1 << 12, target_kl=1e-6, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    metrics = update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(11),
    )

    # One minibatch covers the whole rollout, so the sole actor update ran at
    # unchanged weights: any nonzero KL would be numerics, and the corrupted
    # stored likelihoods would have produced KL near e^3.
    assert metrics["first_minibatch_approx_kl"] == 0.0
    assert metrics["actor_updates"] == 1
    assert metrics["kl_early_stop"] == 0


def test_over_target_pre_step_kl_does_not_update_actor(monkeypatch) -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=93, episode_steps=3, sampling_seed=8)
    # Simulate a behavior policy far from the current actor at the interface
    # where behavior likelihoods now enter the update: the in-update replay.
    # The unchanged actor is then already beyond the trust region before any
    # optimizer step.
    genuine_replay = kaggriculture.vapo.replay_behavior_logprobs

    def stale_replay(*args, **kwargs):
        replayed = genuine_replay(*args, **kwargs)
        return {name: values - 1.0 for name, values in replayed.items()}

    monkeypatch.setattr(kaggriculture.vapo, "replay_behavior_logprobs", stale_replay)
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
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
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
    # Critic targets are exact Monte Carlo suffix returns, independent of the
    # replayed behavior values, so a zero reference reproduces them exactly.
    value_targets = generalized_advantage_and_targets(
        torch.from_numpy(rollout.rewards),
        torch.zeros_like(torch.from_numpy(rollout.rewards)),
        torch.from_numpy(rollout.valid),
        actor_gae_lambda=config.actor_gae_lambda,
        gamma=config.gamma,
    )[1]
    valid_targets = value_targets[torch.from_numpy(rollout.valid)]
    assert metrics["value_target_min"] == pytest.approx(float(valid_targets.min()))
    assert metrics["value_target_max"] == pytest.approx(float(valid_targets.max()))


def test_extra_critic_epochs_refit_the_critic_without_touching_the_actor() -> None:
    model_config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
    config = VapoConfig(epochs=1, critic_epochs=3, minibatch_size=8, use_bfloat16=False)
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

    # Four minibatches per epoch: the critic steps in all three epochs while
    # the actor participates only in the first.
    assert metrics["epochs"] == 3
    assert metrics["updates"] == 12
    assert metrics["actor_updates"] == 4
    assert math.isfinite(metrics["value_loss"])

    with pytest.raises(ValueError, match="critic epochs"):
        _validate_config(VapoConfig(epochs=4, critic_epochs=2))


def test_entropy_is_diagnostic_only_when_policy_advantage_is_zero(monkeypatch) -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=94, episode_steps=3, sampling_seed=10)
    rollout.rewards.fill(0.0)
    # Zero rewards alone leave value-driven GAE deltas; zero replayed values
    # too so every advantage is exactly zero and entropy carries no gradient.
    monkeypatch.setattr(
        kaggriculture.vapo,
        "replay_behavior_values",
        lambda critic, architecture, staged, **kwargs: torch.zeros(staged["unit_actions"].shape[0]),
    )
    config = VapoConfig(
        epochs=1,
        minibatch_size=rollout.state_count,
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


def test_behavior_values_are_replayed_before_the_update_mutates_the_critic(monkeypatch) -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=96, episode_steps=3, sampling_seed=14)
    critic_before = {
        name: parameter.detach().clone() for name, parameter in critic.named_parameters()
    }
    observed: dict[str, float | int] = {"calls": 0, "critic_drift": float("inf")}
    real_replay = kaggriculture.vapo.replay_behavior_values

    def recording_replay(critic_module, architecture, staged, **kwargs):
        observed["calls"] = int(observed["calls"]) + 1
        observed["critic_drift"] = max(
            (parameter - critic_before[name]).abs().max().item()
            for name, parameter in critic_module.named_parameters()
        )
        return real_replay(critic_module, architecture, staged, **kwargs)

    monkeypatch.setattr(kaggriculture.vapo, "replay_behavior_values", recording_replay)
    config = VapoConfig(epochs=2, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(15),
    )

    # Replaying collection-time values is only faithful while the critic still
    # holds its behavior-time weights: exactly one replay, at zero drift, even
    # though the following epochs then mutate the critic.
    assert observed["calls"] == 1
    assert observed["critic_drift"] == 0.0
    assert any(
        not torch.equal(parameter, critic_before[name])
        for name, parameter in critic.named_parameters()
    )


def test_behavior_value_replay_is_chunk_invariant_and_restores_mode() -> None:
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(
        FarmActor(model_config), games=1, seed_start=95, episode_steps=4, sampling_seed=12
    )
    device = torch.device("cpu")
    staged = {
        "board": _stage_tensor(rollout.states["board"], device),
        "critic_features": _stage_tensor(rollout.states["critic_features"], device),
        "unit_actions": _stage_tensor(rollout.unit_actions, device),
    }
    critic.train()

    replayed = replay_behavior_values(critic, CONV_ENTITY, staged, chunk_size=2)

    assert critic.training
    with torch.inference_mode():
        critic.eval()
        expected = critic.value(critic(staged["board"].float(), staged["critic_features"].float()))
        critic.train()
    assert replayed.shape == (staged["board"].shape[0],)
    torch.testing.assert_close(replayed, expected)
    with pytest.raises(ValueError, match="chunk size"):
        replay_behavior_values(critic, CONV_ENTITY, staged, chunk_size=0)


def test_zero_actor_epochs_runs_a_critic_only_warmup_update(monkeypatch) -> None:
    """The warm-start phase fits the critic without touching a pretrained actor.

    With `actor_epochs=0` the actor's weights must be byte-identical after the
    update, the critic must still train, and the behavior-likelihood replay
    must not run at all (nothing consumes it).
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=97, episode_steps=3, sampling_seed=14)

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("behavior replay must be skipped when the actor sits out")

    monkeypatch.setattr(kaggriculture.vapo, "replay_behavior_logprobs", forbidden_replay)
    config = VapoConfig(epochs=2, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    actor_before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    critic_before = {name: value.detach().clone() for name, value in critic.named_parameters()}

    metrics = update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(15),
        actor_epochs=0,
    )

    assert metrics["actor_updates"] == 0
    assert metrics["updates"] > 0
    assert metrics["epochs"] == 2
    assert all(torch.equal(value, actor_before[name]) for name, value in actor.named_parameters())
    assert any(
        not torch.equal(value, critic_before[name]) for name, value in critic.named_parameters()
    )
    with pytest.raises(ValueError, match="actor epoch override"):
        update_vapo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(15),
            actor_epochs=3,
        )


def _small_structured_config() -> StructuredConfig:
    return StructuredConfig(
        model_dim=16,
        attention_heads=2,
        ffn_multiplier=1,
        farm_blocks=1,
        opponent_latents=2,
        latents=4,
        core_layers=1,
        quantity_rank=4,
    )


def _structured_rollout_with_quantity_orders(seed_start: int, sampling_seed: int):
    actor = StructuredActor(_small_structured_config())
    with torch.no_grad():
        # Pin the market heads to a quantified buy so the quantity component
        # participates non-vacuously; the conservative production prior can
        # otherwise sample whole short episodes without one.
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-12.0)
        actor.market_kind.bias[MarketKind.STOP] = -6.0
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 6.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[MarketKind.BUY_SEED_WHEAT, -1] = 50.0
    rollout = collect_self_play(
        actor, games=1, seed_start=seed_start, episode_steps=8, sampling_seed=sampling_seed
    )
    return actor, rollout


def test_structured_update_replay_parity_covers_every_component() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=201, sampling_seed=21)
    assert rollout.architecture == STRUCTURED
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}

    parity = update_replay_parity(actor, rollout, minibatch_size=3)

    for component in ("unit", "kind", "quantity"):
        assert parity[f"update_replay_{component}_active_count"] > 0
        assert parity[f"update_replay_{component}_ratio_max_abs_error"] < 1e-4
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)


def test_structured_update_vapo_trains_both_networks() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=203, sampling_seed=23)
    critic = StructuredCritic(_small_structured_config())
    config = VapoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    actor_before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    critic_before = {name: value.detach().clone() for name, value in critic.named_parameters()}

    metrics = update_vapo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(25),
    )

    assert metrics["states"] == rollout.state_count
    assert metrics["actor_updates"] >= 1
    assert math.isfinite(metrics["value_loss"])
    assert metrics["first_minibatch_approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert any(
        not torch.equal(value, actor_before[name]) for name, value in actor.named_parameters()
    )
    assert any(
        not torch.equal(value, critic_before[name]) for name, value in critic.named_parameters()
    )
