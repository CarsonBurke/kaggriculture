from __future__ import annotations

import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._functorch.config

import kaggriculture.ppo
from kaggriculture import telemetry
from kaggriculture.actions import MarketKind
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.ppo import (
    DEFAULT_ACTOR_GAE_LAMBDA,
    DEFAULT_CRITIC_GAE_LAMBDA,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    UNCOMPILED_UPDATE_COMPILE_MODE,
    UPDATE_REPLAY_TAIL_LOGPROB,
    PpoConfig,
    _actor_batch_args,
    _balanced_minibatch_slices,
    _cached_update_callable,
    _clipped_surrogate_sums,
    _contiguous_run_indices,
    _credit_quality_metrics,
    _epoch_value_losses,
    _explained_variance,
    _fit_explained_variance,
    _observe_auxiliary_gradient,
    _stage_tensor,
    _structured_auxiliary_terms,
    _structured_critic_auxiliary_terms,
    _structured_transition_order,
    _target_correlation,
    _validate_config,
    _validate_optimizer_ownership,
    _validate_staged_action_masks,
    actor_forward_args,
    generalized_advantage_and_targets,
    make_optimizers,
    make_structured_dynamics_optimizer,
    prepare_advantages,
    replay_behavior_values,
    update_ppo,
    update_replay_parity,
)
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.rollout import (
    _SHARED_ROLLOUT_FIELDS,
    _TRAJECTORY_METADATA_FIELDS,
    collect_self_play,
)
from kaggriculture.structured import (
    FusedFeedForward,
    StructuredActor,
    StructuredBelief,
    StructuredConfig,
    StructuredCritic,
)
from kaggriculture.structured_dynamics import (
    StructuredCriticDynamics,
    StructuredDynamics,
    _active_belief_fields,
    structured_horizon_plan,
)


def test_structured_actor_gradient_ratio_is_removed() -> None:
    assert not hasattr(PpoConfig(), "structured_actor_gradient_ratio")


def test_entropy_bonus_is_not_configurable() -> None:
    """Entropy is telemetry and never enters the optimized objective."""
    assert not hasattr(PpoConfig(), "entropy_coefficient")


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


def test_default_gae_uses_vapo_decoupled_contract() -> None:
    config = PpoConfig()

    assert config.gamma == pytest.approx(0.997)
    assert config.actor_gae_lambda == pytest.approx(1.0 - 1.0 / (0.05 * 719.0))
    assert config.actor_gae_lambda == DEFAULT_ACTOR_GAE_LAMBDA
    assert config.critic_gae_lambda == pytest.approx(1.0)
    assert config.critic_gae_lambda == DEFAULT_CRITIC_GAE_LAMBDA


def test_out_of_range_critic_gae_lambda_is_rejected() -> None:
    with pytest.raises(ValueError, match="critic GAE lambda must be finite"):
        _validate_config(PpoConfig(critic_gae_lambda=1.5))
    with pytest.raises(ValueError, match="critic GAE lambda must be finite"):
        _validate_config(PpoConfig(critic_gae_lambda=float("nan")))


def test_out_of_range_gamma_is_rejected() -> None:
    with pytest.raises(ValueError, match="gamma must be finite"):
        _validate_config(PpoConfig(gamma=1.5))
    with pytest.raises(ValueError, match="gamma must be finite"):
        _validate_config(PpoConfig(gamma=0.0))


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_structured_auxiliary_coefficients_must_be_finite_and_nonnegative(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="coefficient must be finite and nonnegative"):
        _validate_config(PpoConfig(structured_decision_coefficient=value))
    with pytest.raises(ValueError, match="coefficient must be finite and nonnegative"):
        _validate_config(PpoConfig(structured_critic_latent_coefficient=value))


def test_active_structured_auxiliary_horizons_must_be_positive() -> None:
    with pytest.raises(ValueError, match="decision horizon must be positive"):
        _validate_config(
            PpoConfig(
                structured_decision_coefficient=0.5,
                structured_decision_horizon=0,
            )
        )
    with pytest.raises(ValueError, match="patch horizon must be positive"):
        _validate_config(
            PpoConfig(
                structured_opponent_patch_coefficient=0.5,
                structured_patch_horizon=0,
            )
        )


def test_active_structured_critic_horizon_must_be_positive() -> None:
    with pytest.raises(ValueError, match="structured critic horizon must be positive"):
        _validate_config(
            PpoConfig(
                structured_critic_latent_coefficient=1.0,
                structured_critic_horizon=0,
            )
        )


def test_contiguous_runs_keep_every_valid_state_inside_trajectory_segments() -> None:
    valid_indices = np.array([0, 1, 2, 3, 6, 7, 8, 10], dtype=np.int64)
    order = _contiguous_run_indices(
        valid_indices,
        steps_per_trajectory=5,
        run_length=2,
        rng=np.random.default_rng(0),
    )

    assert np.sort(order) == pytest.approx(valid_indices)
    positions = {int(index): position for position, index in enumerate(order)}
    assert abs(positions[0] - positions[1]) == 1
    assert abs(positions[2] - positions[3]) == 1
    assert abs(positions[6] - positions[7]) == 1


def test_structured_critic_learning_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="structured critic learning rate"):
        _validate_config(PpoConfig(structured_critic_learning_rate=0.0))


def test_minibatches_are_balanced_without_dropping_the_tail() -> None:
    slices = _balanced_minibatch_slices(230_080, 2048)
    sizes = [row.stop - row.start for row in slices]

    assert len(slices) == 113
    assert sum(sizes) == 230_080
    assert max(sizes) <= 2048
    assert max(sizes) - min(sizes) <= 1
    assert slices[0].start == 0
    assert slices[-1].stop == 230_080


def test_gae_at_a_lambda_returns_advantage_plus_value() -> None:
    """The recurrence itself still yields ``A + V`` at the lambda it was given.

    Decoupling happens in `prepare_advantages`, which calls this twice.
    """
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.2, -0.1, 0.5]])
    valid = torch.ones_like(rewards)

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, gae_lambda=0.5, gamma=1.0
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.125, 0.85, 0.5]]))
    torch.testing.assert_close(targets, torch.tensor([[0.325, 0.75, 1.0]]))
    assert not torch.allclose(targets, torch.ones_like(targets))


def test_prepare_advantages_fits_the_critic_on_the_lambda_one_return() -> None:
    """VAPO decoupled GAE: critic targets are Monte Carlo, not the policy lambda-return."""
    rewards = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    values = np.array([[0.2, -0.1, 0.5]], dtype=np.float32)
    valid = np.ones_like(rewards, dtype=np.bool_)
    rollout = SimpleNamespace(rewards=rewards, valid=valid)

    prepared = prepare_advantages(
        rollout, values, PpoConfig(actor_gae_lambda=0.5, critic_gae_lambda=1.0, gamma=1.0)
    )

    np.testing.assert_allclose(prepared.policy_lambda_returns, [[0.325, 0.75, 1.0]], atol=1e-6)
    np.testing.assert_allclose(prepared.value_targets, [[1.0, 1.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(prepared.monte_carlo_returns, prepared.value_targets, atol=1e-6)
    assert not np.allclose(prepared.value_targets, prepared.policy_lambda_returns)


def test_prepare_advantages_uses_the_configured_critic_lambda() -> None:
    """A non-default critic lambda must change the target, not silently stay MC."""
    rewards = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    values = np.array([[0.2, -0.1, 0.5]], dtype=np.float32)
    valid = np.ones_like(rewards, dtype=np.bool_)
    prepared = prepare_advantages(
        SimpleNamespace(rewards=rewards, valid=valid),
        values,
        PpoConfig(actor_gae_lambda=0.5, critic_gae_lambda=0.5, gamma=1.0),
    )

    np.testing.assert_allclose(prepared.value_targets, [[0.325, 0.75, 1.0]], atol=1e-6)
    np.testing.assert_allclose(prepared.policy_lambda_returns, prepared.value_targets, atol=1e-6)
    assert not np.allclose(prepared.value_targets, prepared.monte_carlo_returns)


def test_policy_lambda_explained_variance_is_the_advantage_identity() -> None:
    """EV against G_policy is ``1 - Var(A)/Var(G_policy)``, not critic fit."""
    rewards = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    values = np.array([[0.2, -0.1, 0.5]], dtype=np.float32)
    valid = np.ones_like(rewards, dtype=np.bool_)
    prepared = prepare_advantages(
        SimpleNamespace(rewards=rewards, valid=valid),
        values,
        PpoConfig(actor_gae_lambda=0.5, critic_gae_lambda=1.0, gamma=1.0),
    )
    residual = prepared.policy_lambda_returns - values
    identity = 1.0 - residual[valid].var() / prepared.policy_lambda_returns[valid].var()
    assert _explained_variance(prepared.policy_lambda_returns, values, valid) == pytest.approx(
        identity
    )
    assert _explained_variance(prepared.value_targets, values, valid) != pytest.approx(identity)


def test_an_exact_critic_makes_the_target_exact_at_any_lambda() -> None:
    """Bootstrapping costs nothing where the value function is already right.

    A synthetic potential-difference sequence gives an analytic exact value for
    every state. Feed that in and every temporal-difference residual vanishes:
    the advantage is zero and the target is the Monte Carlo return at any lambda.
    This tests the GAE recurrence, not the environment's reward construction.
    """
    potentials = torch.tensor([[0.2, -0.1, 0.4, 0.3, 0.6]])
    rewards = potentials[:, 1:] - potentials[:, :-1]
    valid = torch.ones_like(rewards)
    exact = potentials[:, -1:] - potentials[:, :-1]

    for gae_lambda in (0.0, 0.5, 1.0):
        advantages, targets = generalized_advantage_and_targets(
            rewards, exact, valid, gae_lambda=gae_lambda, gamma=1.0
        )

        torch.testing.assert_close(advantages, torch.zeros_like(advantages))
        torch.testing.assert_close(targets, exact)

    # And that the agreement is a property of the critic being right, not of the
    # target ignoring it: displace the critic and the target moves with it,
    # which the Monte Carlo suffix return it replaced would not have done.
    displaced = generalized_advantage_and_targets(
        rewards, exact + 0.5, valid, gae_lambda=0.5, gamma=1.0
    )[1]
    assert not torch.allclose(displaced, exact)


def test_dense_gae_matches_reference_recurrence_at_scale() -> None:
    generator = torch.Generator().manual_seed(17)
    rewards = torch.randn(4, 719, generator=generator)
    values = torch.randn(4, 719, generator=generator)
    valid = torch.ones_like(rewards)
    gamma, gae_lambda = 1.0, DEFAULT_ACTOR_GAE_LAMBDA

    advantages, targets = generalized_advantage_and_targets(
        rewards,
        values,
        valid,
        gae_lambda=gae_lambda,
        gamma=gamma,
    )

    deltas = rewards.clone()
    deltas[:, :-1] += gamma * values[:, 1:]
    deltas -= values
    expected = torch.empty_like(deltas)
    running = torch.zeros(deltas.size(0))
    for step in range(deltas.size(1) - 1, -1, -1):
        running = deltas[:, step] + gamma * gae_lambda * running
        expected[:, step] = running

    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(targets, expected + values)


def test_discounted_advantages_and_targets_match_the_reference_recurrence() -> None:
    rewards = torch.tensor([[0.2, -0.1, 0.3]])
    values = torch.tensor([[0.4, 0.1, -0.2]])
    valid = torch.ones_like(rewards)

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, gae_lambda=0.5, gamma=0.9
    )

    delta_2 = 0.3 - (-0.2)
    delta_1 = -0.1 + 0.9 * (-0.2) - 0.1
    delta_0 = 0.2 + 0.9 * 0.1 - 0.4
    expected_2 = delta_2
    expected_1 = delta_1 + 0.9 * 0.5 * expected_2
    expected_0 = delta_0 + 0.9 * 0.5 * expected_1
    expected_advantages = torch.tensor([[expected_0, expected_1, expected_2]])
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(targets, expected_advantages + values)
    # The general recurrence still supports explicit gamma-one experiments;
    # production uses the shared discounted shaping/PPO gamma.
    monte_carlo = torch.tensor([[0.2 + 0.9 * (-0.1 + 0.9 * 0.3), -0.1 + 0.9 * 0.3, 0.3]])
    assert not torch.allclose(targets, monte_carlo)


def test_lambda_one_recovers_the_monte_carlo_return_whatever_the_critic_says() -> None:
    """The lambda-one target is the one the behavior values cannot move.

    At lambda one the residuals telescope: whatever the critic predicts is added
    back by the advantage and cancels, leaving the exact suffix return. That is
    what makes it unbiased, and also what makes it the noisiest choice -- it is
    the endpoint the shipped lambda deliberately backs away from, so it is worth
    pinning that this implementation still reaches it exactly.
    """
    rewards = torch.tensor([[0.25, -0.4, 0.6], [-0.1, 0.2, -0.3]])
    valid = torch.ones_like(rewards)
    monte_carlo = torch.tensor([[0.45, 0.2, 0.6], [-0.2, -0.1, -0.3]])

    for values in (
        torch.tensor([[10.0, -7.0, 3.0], [4.0, 1.0, -8.0]]),
        torch.tensor([[-2.0, 6.0, 9.0], [-5.0, 11.0, 0.5]]),
    ):
        targets = generalized_advantage_and_targets(
            rewards, values, valid, gae_lambda=1.0, gamma=1.0
        )[1]

        torch.testing.assert_close(targets, monte_carlo)

    shortened = generalized_advantage_and_targets(
        rewards,
        torch.tensor([[10.0, -7.0, 3.0], [4.0, 1.0, -8.0]]),
        valid,
        gae_lambda=0.99,
        gamma=1.0,
    )[1]
    assert not torch.allclose(shortened, monte_carlo)


def test_masked_gae_does_not_bootstrap_through_padding() -> None:
    rewards = torch.tensor([[0.0, 1.0, 100.0], [0.0, 0.0, -1.0]])
    values = torch.tensor([[0.25, 0.5, 99.0], [0.1, 0.2, 0.3]])
    valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, gae_lambda=1.0, gamma=1.0
    )
    torch.testing.assert_close(targets[0], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(advantages[0], torch.tensor([0.75, 0.5, 0.0]))
    torch.testing.assert_close(targets[1], torch.tensor([-1.0, -1.0, -1.0]))


def test_a_truncated_trajectory_anchors_its_last_target_on_the_reward_alone() -> None:
    """Padding must not leak into a target that now contains a value.

    At lambda one the two masking tests above cannot separate a leak from a
    correct bootstrap, because the value cancels out of the target entirely.
    Below one it does not: every target carries a value, and the last valid step
    of a short trajectory is the one place where the value that would be added
    belongs to padding. It must bootstrap from nothing, leaving exactly the
    reward, whatever the padded slot holds.
    """
    rewards = torch.tensor([[0.2, -0.3, 0.5, 1000.0], [-0.1, 0.6, 1000.0, 1000.0]])
    values = torch.tensor([[0.1, 0.4, -0.2, -1000.0], [0.3, -0.5, -1000.0, -1000.0]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, gae_lambda=0.5, gamma=1.0
    )
    torch.testing.assert_close(
        advantages, torch.tensor([[0.225, -0.55, 0.7, 0.0], [-0.35, 1.1, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        targets, torch.tensor([[0.325, -0.15, 0.5, 0.0], [-0.05, 0.6, 0.0, 0.0]])
    )
    # Each row's last valid target is its last reward and nothing else.
    assert targets[0, 2] == rewards[0, 2]
    assert targets[1, 1] == rewards[1, 1]


def test_masked_gae_ignores_nonfinite_padding_but_rejects_nonfinite_valid_data() -> None:
    rewards = torch.tensor([[0.0, 1.0, float("nan")]])
    values = torch.tensor([[0.25, 0.5, float("inf")]])
    valid = torch.tensor([[True, True, False]])

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, gae_lambda=1.0, gamma=1.0
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
    config = PpoConfig(
        epochs=1,
        minibatch_size=8,
        lr_warmup_steps=4,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    rollout = collect_self_play(actor, games=1, seed_start=91, episode_steps=3, sampling_seed=4)

    update_ppo(
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

    parity = update_replay_parity(
        actor,
        rollout,
        minibatch_size=3,
        compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
        autocast_enabled=False,
    )

    for component in ("unit", "kind", "quantity"):
        assert parity[f"update_replay_{component}_active_count"] > 0
        assert parity[f"update_replay_{component}_ratio_max_abs_error"] < 1e-4
        assert parity[f"update_replay_{component}_kl"] < 1e-8
        assert parity[f"update_replay_{component}_tail_fraction"] == 0.0
    assert parity["update_replay_max_ratio_error"] == max(
        parity[f"update_replay_{component}_ratio_max_abs_error"]
        for component in ("unit", "kind", "quantity")
    )
    assert parity["update_replay_max_kl"] == max(
        parity[f"update_replay_{component}_kl"] for component in ("unit", "kind", "quantity")
    )
    assert parity["update_replay_max_tail_fraction"] == max(
        parity[f"update_replay_{component}_tail_fraction"]
        for component in ("unit", "kind", "quantity")
    )
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)

    # A behavior/update mismatch must be visible in the gated statistic, not
    # only in the extreme value. Stale stored likelihoods shifted by log(2)
    # give every component a ratio near two, whose k3 divergence is
    # (2 - 1) - log 2, so both affected heads land far above MAX_UPDATE_REPLAY_KL
    # while the untouched unit head stays at the numerical floor.
    rollout.old_market_kind_logprobs[...] -= math.log(2.0)
    rollout.old_market_quantity_logprobs[...] -= math.log(2.0)
    drifted = update_replay_parity(
        actor,
        rollout,
        minibatch_size=3,
        compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
        autocast_enabled=False,
    )
    assert drifted["update_replay_kind_ratio_max_abs_error"] > 0.9
    assert drifted["update_replay_quantity_ratio_max_abs_error"] > 0.9
    expected_kl = 1.0 - math.log(2.0)
    assert drifted["update_replay_kind_kl"] == pytest.approx(expected_kl, rel=1e-3)
    assert drifted["update_replay_quantity_kl"] == pytest.approx(expected_kl, rel=1e-3)
    assert drifted["update_replay_unit_kl"] < 1e-8
    assert drifted["update_replay_max_kl"] > MAX_UPDATE_REPLAY_KL

    with pytest.raises(ValueError, match="minibatch size"):
        update_replay_parity(
            actor,
            rollout,
            minibatch_size=0,
            compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
            autocast_enabled=False,
        )


def test_the_joint_kl_is_the_statistic_the_every_iteration_gate_samples() -> None:
    """The per-head audit and the per-iteration gate must measure one quantity.

    `update_ppo` gates `first_minibatch_approx_kl`, which is k3 summed over
    every active component of all three heads divided by their total count, on
    a single minibatch. That is a component-weighted mean of the three audited
    per-head means, so `MAX_UPDATE_REPLAY_KL` bounds it by construction and the
    only open question is how far one minibatch strays from the whole batch.
    None of that was visible while the audit reported three per-head numbers
    and the gate reported an unrelated-looking fourth, which is how a bound
    calibrated on a randomly initialized actor -- where this divergence is
    ~7e-8 rather than the clone's ~2e-3 -- survived two recalibrations of its
    own siblings and stopped a 500-iteration run at iteration 41.
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=94, episode_steps=32, sampling_seed=12)

    # Two heads shifted by log 2 give a joint value that is neither zero nor
    # equal to any one head, so the weighting is actually exercised.
    rollout.old_market_kind_logprobs[...] -= math.log(2.0)
    rollout.old_market_quantity_logprobs[...] -= math.log(2.0)
    parity = update_replay_parity(
        actor,
        rollout,
        minibatch_size=3,
        compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
        autocast_enabled=False,
    )

    heads = ("unit", "kind", "quantity")
    counts = {head: parity[f"update_replay_{head}_active_count"] for head in heads}
    weighted = sum(parity[f"update_replay_{head}_kl"] * counts[head] for head in heads) / sum(
        counts.values()
    )
    assert parity["update_replay_joint_kl"] == pytest.approx(weighted, rel=1e-9)

    # The inequality that lets one bound govern both statistics.
    assert parity["update_replay_joint_kl"] <= parity["update_replay_max_kl"]
    # The unit head is untouched and the other two are far above it, so the
    # weighted mean must land strictly between them rather than tracking either.
    assert parity["update_replay_unit_kl"] < parity["update_replay_joint_kl"]
    # A per-minibatch maximum cannot fall below the mean the minibatches make up.
    assert parity["update_replay_minibatch_kl"] >= parity["update_replay_joint_kl"]


def test_update_replay_kl_averages_over_components_while_the_maximum_does_not() -> None:
    """The gated statistic must not be an extreme value over the component count.

    A maximum grows with the number of samples drawn, so gating on it makes the
    threshold a property of the rollout size rather than of the policy. Shifting
    a known subset of one head's stored likelihoods separates the two: the worst
    component is identical whether one component is affected or all of them,
    while the KL tracks the affected fraction.
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=94, episode_steps=32, sampling_seed=12)

    # The audit only visits valid rows, so the denominator it divides by is the
    # active count restricted to those rows, not the whole arena.
    active = rollout.market_active.astype(bool) & rollout.valid.astype(bool)[..., None]
    total_active = int(active.sum())
    assert total_active > 8

    # Comfortably past UPDATE_REPLAY_TAIL_LOGPROB so every shifted component
    # counts as material and the tail fraction equals the shifted share exactly.
    shift = UPDATE_REPLAY_TAIL_LOGPROB * 1.5

    def shifted_parity(share: int) -> tuple[dict[str, float | int], float]:
        """Shift `1/share` of the active kind components by exactly `shift`."""
        fresh = collect_self_play(actor, games=1, seed_start=94, episode_steps=32, sampling_seed=12)
        flat_active = active.reshape(-1)
        count = total_active // share
        shift_mask = np.zeros(flat_active.shape, dtype=bool)
        shift_mask[np.flatnonzero(flat_active)[:count]] = True
        fresh.old_market_kind_logprobs[shift_mask.reshape(active.shape)] -= shift
        parity = update_replay_parity(
            actor,
            fresh,
            minibatch_size=3,
            compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
            autocast_enabled=False,
        )
        assert parity["update_replay_kind_active_count"] == total_active
        return parity, count / total_active

    # Every shifted component carries an identical k3, so the expected mean is
    # that value scaled by the affected fraction exactly.
    per_component_kl = math.expm1(shift) - shift
    half, half_fraction = shifted_parity(2)
    eighth, eighth_fraction = shifted_parity(8)

    assert half["update_replay_kind_kl"] == pytest.approx(
        per_component_kl * half_fraction, rel=1e-3
    )
    assert eighth["update_replay_kind_kl"] == pytest.approx(
        per_component_kl * eighth_fraction, rel=1e-3
    )
    # The mean tracks how much of the head moved; the extreme value is identical
    # in both cases, which is exactly why it cannot serve as the bound.
    assert eighth["update_replay_kind_kl"] < half["update_replay_kind_kl"]
    # The tail fraction is what stays proportional to the corrupted share
    # without depending on how far the worst component strayed, so it reads
    # directly as "what portion of sampled actions disagree materially".
    assert half["update_replay_kind_tail_fraction"] == pytest.approx(half_fraction, rel=1e-9)
    assert eighth["update_replay_kind_tail_fraction"] == pytest.approx(eighth_fraction, rel=1e-9)
    # The extreme value is identical in both cases, which is the point: it
    # cannot distinguish a head that moved wholesale from one that barely did.
    expected_max = math.expm1(shift)
    assert half["update_replay_kind_ratio_max_abs_error"] == pytest.approx(expected_max, rel=1e-3)
    assert eighth["update_replay_kind_ratio_max_abs_error"] == pytest.approx(expected_max, rel=1e-3)


def test_the_tail_statistic_separates_concentration_from_the_mean_it_shares() -> None:
    """Two corruptions of equal total KL must differ in the tail statistic.

    This is the discrimination the mean cannot make on its own. A large
    likelihood error confined to a thin slice of components — the shape an
    off-by-one on a single minibatch, one seat, or one rare mask path takes —
    and a small error spread across many can carry the identical summed k3 and
    therefore the identical mean. Counting the affected share instead of
    averaging its magnitude away is what tells them apart.
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=94, episode_steps=64, sampling_seed=12)
    active = rollout.unit_active.astype(bool) & rollout.valid.astype(bool)[..., None]
    flat_active = np.flatnonzero(active.reshape(-1))
    total_active = flat_active.size

    concentrated_shift = UPDATE_REPLAY_TAIL_LOGPROB * 1.2
    concentrated_count = max(1, total_active // 64)
    # Solve for the diffuse shift that spreads the same total k3 over every
    # component, so the two corruptions are indistinguishable by the mean.
    total_kl = concentrated_count * (math.expm1(concentrated_shift) - concentrated_shift)
    target = total_kl / total_active
    low, high = 0.0, concentrated_shift
    for _ in range(200):
        # k3 is strictly increasing on the positive axis, so plain bisection
        # inverts it to machine precision without pulling in a solver.
        middle = 0.5 * (low + high)
        low, high = (middle, high) if math.expm1(middle) - middle < target else (low, middle)
    diffuse_shift = 0.5 * (low + high)
    assert diffuse_shift < UPDATE_REPLAY_TAIL_LOGPROB, "the diffuse arm must stay immaterial"

    def parity_after(shift: float, count: int) -> dict[str, float | int]:
        fresh = collect_self_play(actor, games=2, seed_start=94, episode_steps=64, sampling_seed=12)
        mask = np.zeros(active.reshape(-1).shape, dtype=bool)
        mask[flat_active[:count]] = True
        fresh.old_unit_logprobs[mask.reshape(active.shape)] -= shift
        return update_replay_parity(
            actor,
            fresh,
            minibatch_size=3,
            compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
            autocast_enabled=False,
        )

    concentrated = parity_after(concentrated_shift, concentrated_count)
    diffuse = parity_after(diffuse_shift, total_active)

    # Same mean, by construction.
    assert concentrated["update_replay_unit_kl"] == pytest.approx(
        diffuse["update_replay_unit_kl"], rel=1e-6
    )
    # Wholly different tails: every concentrated component is material, no
    # diffuse one is.
    assert concentrated["update_replay_unit_tail_fraction"] == pytest.approx(
        concentrated_count / total_active, rel=1e-9
    )
    assert diffuse["update_replay_unit_tail_fraction"] == 0.0


def test_the_tail_bound_is_not_redundant_against_the_kl_bound() -> None:
    """The tail gate must be able to fail something the KL gate would pass.

    The two bounds police overlapping regions: a defect of share f at
    log-likelihood error d contributes f * k3(d) to the mean, so a share large
    enough to trip the tail bound may already have tripped the KL bound, at
    which point the second gate is decoration. It stays additive only while the
    tail bound sits below the KL budget divided by k3 at the threshold. The
    budget here is half of MAX_UPDATE_REPLAY_KL, which is roughly what the
    bf16 floor measured on a trained actor leaves of it.
    """
    threshold_kl = math.expm1(UPDATE_REPLAY_TAIL_LOGPROB) - UPDATE_REPLAY_TAIL_LOGPROB
    assert MAX_UPDATE_REPLAY_TAIL_FRACTION * threshold_kl < MAX_UPDATE_REPLAY_KL / 2.0


def test_update_uses_rollout_stored_sampler_likelihoods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shifted behavior likelihood must change the PPO ratio and objective."""
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    baseline_actor = FarmActor(model_config)
    baseline_critic = DistributionalCritic(model_config)
    shifted_actor = copy.deepcopy(baseline_actor)
    shifted_critic = copy.deepcopy(baseline_critic)
    baseline_rollout = collect_self_play(
        baseline_actor, games=1, seed_start=95, episode_steps=3, sampling_seed=10
    )
    shifted_rollout = copy.deepcopy(baseline_rollout)
    baseline_rollout.rewards[:] = 1.0
    shifted_rollout.rewards[:] = 1.0
    for name in (
        "old_unit_logprobs",
        "old_market_kind_logprobs",
        "old_market_quantity_logprobs",
    ):
        getattr(shifted_rollout, name)[...] -= math.log(2.0)

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("the PPO surrogate must not replay its behavior likelihoods")

    monkeypatch.setattr(kaggriculture.ppo, "replay_behavior_logprobs", forbidden_replay)
    config = PpoConfig(epochs=1, minibatch_size=1 << 12, target_kl=10.0, use_bfloat16=False)
    baseline_optimizers = make_optimizers(baseline_actor, baseline_critic, config)
    shifted_optimizers = make_optimizers(shifted_actor, shifted_critic, config)

    baseline = update_ppo(
        baseline_actor,
        baseline_critic,
        *baseline_optimizers,
        baseline_rollout,
        config,
        generator=np.random.default_rng(11),
    )
    shifted = update_ppo(
        shifted_actor,
        shifted_critic,
        *shifted_optimizers,
        shifted_rollout,
        config,
        generator=np.random.default_rng(11),
    )

    assert baseline["first_minibatch_approx_kl"] == pytest.approx(0.0, abs=1e-8)
    assert shifted["first_minibatch_approx_kl"] == pytest.approx(1.0 - math.log(2.0), rel=1e-4)
    assert shifted["policy_loss"] != pytest.approx(baseline["policy_loss"], abs=1e-6)


def _small_update_models():
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    return FarmActor(model_config), DistributionalCritic(model_config)


class _FoundInfRecordingOptimizer(torch.optim.Optimizer):
    supports_found_inf = True

    def __init__(self, parameters, learning_rate: float) -> None:
        super().__init__(parameters, {"lr": learning_rate})
        self.found_inf_values: list[float] = []

    @torch.no_grad()
    def step(self, closure=None):
        found_inf = self.found_inf
        self.found_inf_values.append(float(found_inf))
        if bool(found_inf):
            return None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["moment"] = parameter.grad.detach().clone()
                parameter.add_(parameter.grad, alpha=-group["lr"])
        return None


def test_optimizer_step_keeps_found_inf_on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))

    class DeviceGateOptimizer(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__((parameter,), {"lr": 0.1})
            self.received_found_inf: torch.Tensor | None = None

        def step(self, closure=None):
            self.received_found_inf = self.found_inf
            return None

    optimizer = DeviceGateOptimizer()
    found_inf = torch.zeros(())

    def forbidden_item(_tensor):
        raise AssertionError("optimizer step must not read a device scalar")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    kaggriculture.ppo._optimizer_step(
        optimizer,
        base_learning_rate=0.1,
        warmup_steps=4,
        found_inf=found_inf,
    )

    assert optimizer.received_found_inf is found_inf
    assert optimizer.param_groups[0]["warmup_step"] == 1


def test_update_rejects_deterministic_learner_rollout_before_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _small_update_models()
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=93,
        episode_steps=3,
        deterministic=True,
    )
    assert not rollout.learner_stochastic
    config = PpoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    actor_before = {
        name: parameter.detach().clone() for name, parameter in actor.named_parameters()
    }
    critic_before = {
        name: parameter.detach().clone() for name, parameter in critic.named_parameters()
    }

    def forbidden_stage(*_args, **_kwargs):
        raise AssertionError("invalid learner rollout must be rejected before staging")

    monkeypatch.setattr(kaggriculture.ppo, "_stage_tensor", forbidden_stage)
    with pytest.raises(ValueError, match="stochastic learner collection"):
        update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(9),
        )

    assert not actor_optimizer.state
    assert not critic_optimizer.state
    for name, parameter in actor.named_parameters():
        assert torch.equal(parameter, actor_before[name]), name
    for name, parameter in critic.named_parameters():
        assert torch.equal(parameter, critic_before[name]), name


def test_unchanged_on_policy_first_minibatch_has_unit_ratio() -> None:
    actor, critic = _small_update_models()
    rollout = collect_self_play(actor, games=1, seed_start=93, episode_steps=3, sampling_seed=8)
    config = PpoConfig(epochs=1, minibatch_size=1 << 12, target_kl=1e-4, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(9),
    )

    assert metrics["updates"] == 1
    assert metrics["first_minibatch_approx_kl"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["actor_updates"] == 1
    assert metrics["kl_early_stop"] == 0
    assert metrics["max_approx_kl"] == pytest.approx(0.0, abs=1e-8)


def test_over_target_sampler_kl_stops_actor_before_first_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale sampler likelihood rejects the gradient and does not refresh caches."""
    actor, critic = _small_update_models()
    rollout = collect_self_play(actor, games=1, seed_start=93, episode_steps=3, sampling_seed=8)
    for name in (
        "old_unit_logprobs",
        "old_market_kind_logprobs",
        "old_market_quantity_logprobs",
    ):
        getattr(rollout, name)[...] -= 1.0
    genuine_refresh = kaggriculture.ppo.refresh_fused_mlp_fp8
    actor_refreshes: list[bool | None] = []

    def record_actor_refresh(module, *, bootstrap_down=None):
        if module is actor:
            actor_refreshes.append(bootstrap_down)
        genuine_refresh(module, bootstrap_down=bootstrap_down)

    monkeypatch.setattr(kaggriculture.ppo, "refresh_fused_mlp_fp8", record_actor_refresh)
    config = PpoConfig(epochs=2, minibatch_size=8, target_kl=1e-4, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    actor_before = {
        name: parameter.detach().clone() for name, parameter in actor.named_parameters()
    }
    critic_before = {
        name: parameter.detach().clone() for name, parameter in critic.named_parameters()
    }

    metrics = update_ppo(
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
    assert actor_refreshes == [None]
    assert metrics["first_minibatch_approx_kl"] > config.target_kl
    assert metrics["max_approx_kl"] > config.target_kl
    for name, parameter in actor.named_parameters():
        assert torch.equal(parameter, actor_before[name]), name
    assert any(
        not torch.equal(parameter, critic_before[name])
        for name, parameter in critic.named_parameters()
    )


def test_over_target_kl_still_steps_the_actor_predictor() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=227, sampling_seed=55)
    critic = StructuredCritic(_small_structured_config())
    for name in (
        "old_unit_logprobs",
        "old_market_kind_logprobs",
        "old_market_quantity_logprobs",
    ):
        getattr(rollout, name)[...] -= 1.0
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        target_kl=1e-4,
        use_bfloat16=False,
        structured_decision_coefficient=0.5,
    )
    dynamics = StructuredDynamics(_small_structured_config())
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    dynamics_optimizer = make_structured_dynamics_optimizer(dynamics, config)
    actor_before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    dynamics_before = {name: value.detach().clone() for name, value in dynamics.named_parameters()}

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(56),
        structured_dynamics=dynamics,
        structured_dynamics_optimizer=dynamics_optimizer,
        structured_actor_auxiliary=True,
        auxiliary_generator=np.random.default_rng(57),
    )

    assert metrics["actor_updates"] == 0
    assert metrics["kl_early_stop"] == 1
    assert metrics["structured_actor_predictor_updates"] >= 1
    assert metrics["structured_actor_auxiliary_updates"] == 0
    assert all(torch.equal(value, actor_before[name]) for name, value in actor.named_parameters())
    assert any(
        not torch.equal(value, dynamics_before[name]) for name, value in dynamics.named_parameters()
    )


def test_one_ppo_update_is_finite() -> None:
    model_config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
    # Eight steps of a fresh game leaves both banks equal and every shaped
    # reward at around 1e-9, where any relation between target statistics holds
    # to any absolute tolerance. Give the fixture the reward scale a real
    # episode reaches so the identity below is a measurement.
    rollout.rewards[:] = (
        np.random.default_rng(11).normal(0.0, 0.05, size=rollout.rewards.shape).astype(np.float32)
    )
    config = PpoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    metrics = update_ppo(
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
    assert metrics["critic_gae_lambda"] == config.critic_gae_lambda
    assert metrics["gamma"] == config.gamma
    # Critic targets are the lambda-one return, not ``A_policy + V``.
    assert metrics["value_target_std"] > 0.05
    assert metrics["value_target_min"] < metrics["value_target_max"]
    # A random-init critic on this support stays well inside atoms that span
    # the reward range with headroom, so nothing needed saturating.
    assert metrics["value_target_saturated_fraction"] == 0.0
    # Four explained variances against three different targets, all reported
    # every iteration. One key carrying all of them is what let the run be read
    # as "the critic explains nothing" when the number quoted was measured
    # against a target the critic never regressed on, so the names have to stay
    # distinct and every one has to survive an update.
    for name in (
        "monte_carlo_explained_variance",
        "lambda_return_explained_variance",
        "critic_fit_explained_variance_first_epoch",
        "critic_fit_explained_variance_last_epoch",
    ):
        assert name in metrics, name
        assert math.isfinite(metrics[name]), name
    assert "explained_variance" not in metrics
    # Running an update is the only honest way to enumerate what it reports, so
    # this is where the mirror's coverage of those names is checkable. A metric
    # added here and nowhere else still reaches TensorBoard, but it lands in
    # `misc` -- an unbounded category nobody reads -- and the telemetry tests
    # cannot see it, because it appears in no table they can walk.
    unfiled = sorted(
        name
        for name in metrics
        if (placement := telemetry._placement(name)) is not None
        and placement[1].startswith("misc/")
    )
    assert not unfiled, unfiled


def test_policy_and_critic_gradients_are_not_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _small_update_models()
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=94,
        episode_steps=4,
        sampling_seed=10,
    )
    rollout.rewards[:] = np.random.default_rng(12).normal(0.0, 0.05, size=rollout.rewards.shape)
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        nextlat_max_gradient_norm=1.0e-8,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    original_clip = torch.nn.utils.clip_grad_norm_
    clipped_parameter_sets: list[set[int]] = []

    def recording_clip(parameters, *args, **kwargs):
        materialized = tuple(parameters)
        clipped_parameter_sets.append({id(parameter) for parameter in materialized})
        return original_clip(materialized, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(13),
    )

    assert clipped_parameter_sets == []
    assert metrics["actor_gradient_norm"] > config.nextlat_max_gradient_norm
    assert metrics["critic_gradient_norm"] == pytest.approx(
        math.hypot(
            metrics["critic_trunk_gradient_norm"],
            metrics["critic_head_gradient_norm"],
        )
    )


def test_a_critic_only_refit_aborts_on_a_non_finite_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A poisoned critic loss must stop before parameters or moments change.

    Gateable optimizers receive their native `found_inf` skip signal; other
    optimizers are rejected before `step`. Neither path may refresh a cache for
    an update that did not commit.
    """
    model_config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        critic_epochs=2,
        minibatch_size=8,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    before = {name: value.detach().clone() for name, value in critic.named_parameters()}

    original = kaggriculture.ppo._critic_minibatch_loss

    def poisoned(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        loss, predicted = original(*args, **kwargs)  # type: ignore[arg-type]
        return loss * float("nan"), predicted

    monkeypatch.setattr(kaggriculture.ppo, "_critic_minibatch_loss", poisoned)

    with pytest.raises(FloatingPointError, match="non-finite critic loss"):
        update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(4),
            actor_epochs=0,
        )

    assert not critic_optimizer.state
    for name, value in critic.named_parameters():
        assert torch.equal(before[name], value.detach()), name


def test_nonfinite_actor_gradient_uses_found_inf_without_committing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _small_update_models()
    rollout = collect_self_play(actor, games=1, seed_start=96, episode_steps=3, sampling_seed=12)
    config = PpoConfig(
        epochs=1,
        minibatch_size=1 << 12,
        target_kl=1.0,
        use_bfloat16=False,
    )
    actor_optimizer = _FoundInfRecordingOptimizer(actor.parameters(), config.actor_learning_rate)
    _, critic_optimizer = make_optimizers(actor, critic, config)
    actor_before = {
        name: parameter.detach().clone() for name, parameter in actor.named_parameters()
    }
    genuine_norm = torch.nn.utils.get_total_norm
    norm_calls = 0

    def inject_actor_inf(gradients, *args, **kwargs):
        nonlocal norm_calls
        materialized = tuple(gradients)
        norm_calls += 1
        if norm_calls == 1:
            return materialized[0].new_tensor(float("inf"))
        return genuine_norm(materialized, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", inject_actor_inf)

    with pytest.raises(FloatingPointError, match="non-finite actor gradient norm"):
        update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(13),
        )

    assert actor_optimizer.found_inf_values == [1.0]
    assert not actor_optimizer.state
    assert all("warmup_step" not in group for group in actor_optimizer.param_groups)
    assert all(group["lr"] == config.actor_learning_rate for group in actor_optimizer.param_groups)
    for name, parameter in actor.named_parameters():
        assert torch.equal(parameter, actor_before[name]), name


def test_nonfinite_critic_gradient_uses_found_inf_without_committing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _small_update_models()
    rollout = collect_self_play(actor, games=1, seed_start=97, episode_steps=3, sampling_seed=14)
    config = PpoConfig(
        epochs=1,
        minibatch_size=1 << 12,
        target_kl=1.0,
        use_bfloat16=False,
    )
    actor_optimizer, _ = make_optimizers(actor, critic, config)
    critic_optimizer = _FoundInfRecordingOptimizer(critic.parameters(), config.critic_learning_rate)
    critic_before = {
        name: parameter.detach().clone() for name, parameter in critic.named_parameters()
    }

    original_norm = torch.nn.utils.get_total_norm
    norm_calls = 0

    def inject_critic_inf(parameters, *args, **kwargs):
        nonlocal norm_calls
        norm_calls += 1
        materialized = tuple(parameters)
        if norm_calls == 2:
            return materialized[0].new_tensor(float("inf"))
        return original_norm(materialized, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", inject_critic_inf)

    with pytest.raises(FloatingPointError, match="non-finite critic loss or gradient norm"):
        update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(15),
        )

    assert critic_optimizer.found_inf_values == [1.0]
    assert not critic_optimizer.state
    assert all("warmup_step" not in group for group in critic_optimizer.param_groups)
    assert all(
        group["lr"] == config.critic_learning_rate for group in critic_optimizer.param_groups
    )
    for name, parameter in critic.named_parameters():
        assert torch.equal(parameter, critic_before[name]), name


def test_a_return_past_the_outermost_atom_saturates_and_is_reported() -> None:
    """A bootstrapped target has no bound the support can be sized against.

    The environment's complete economic return is bounded inside (-2, 2), but a
    lambda-return adds the critic's own prediction to a truncated advantage. The
    critic is only bounded by the same support, so no width contains that target
    by construction.
    Saturating at the outermost atom is what a categorical projection does; the
    fraction it had to saturate is what tells you the critic is off, and killing
    a five-hundred-iteration run to say so would be strictly less informative.
    """
    model_config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
    # A reward far outside the critic's calibrated support, so the target
    # saturates no matter what the critic predicts.
    rollout.rewards[:, -1] = np.float32(9.0)
    config = PpoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(4),
    )

    assert metrics["value_target_max"] > float(critic.support[-1])
    assert 0.0 < metrics["value_target_saturated_fraction"] <= 1.0
    assert math.isfinite(metrics["value_loss"])


def test_extra_critic_epochs_refit_the_critic_without_touching_the_actor() -> None:
    model_config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
    config = PpoConfig(epochs=1, critic_epochs=3, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    metrics = update_ppo(
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
        _validate_config(PpoConfig(epochs=4, critic_epochs=2))


def test_zero_policy_advantage_leaves_actor_unchanged(monkeypatch) -> None:
    """Measured entropy must not create a gradient outside the PPO surrogate."""
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=94, episode_steps=3, sampling_seed=10)
    rollout.rewards.fill(0.0)
    # Zero rewards alone leave value-driven GAE deltas; zero replayed values too
    # so every advantage and therefore every policy gradient is exactly zero.
    monkeypatch.setattr(
        kaggriculture.ppo,
        "replay_behavior_values",
        lambda critic, architecture, staged, **kwargs: torch.zeros(staged["unit_actions"].shape[0]),
    )
    config = PpoConfig(
        epochs=1,
        minibatch_size=rollout.state_count,
        lr_warmup_steps=0,
        use_bfloat16=False,
        actor_learning_rate=1.0e-2,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}

    metrics = update_ppo(
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
    real_replay = kaggriculture.ppo.replay_behavior_values

    def recording_replay(critic_module, architecture, staged, **kwargs):
        observed["calls"] = int(observed["calls"]) + 1
        observed["critic_drift"] = max(
            (parameter - critic_before[name]).abs().max().item()
            for name, parameter in critic_module.named_parameters()
        )
        return real_replay(critic_module, architecture, staged, **kwargs)

    monkeypatch.setattr(kaggriculture.ppo, "replay_behavior_values", recording_replay)
    config = PpoConfig(epochs=2, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    update_ppo(
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


def test_behavior_value_replay_can_be_restricted_to_named_states() -> None:
    """A population member's critic is replayed on that member's rows alone.

    The restriction is the saving that keeps N per-member updates at one
    whole-wave critic pass between them instead of N. It must return exactly the
    values a whole-wave pass produces, in the order asked for, and it has to
    hold across chunk boundaries because the chunk is a memory bound rather than
    a semantic one.
    """
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
    whole_wave = replay_behavior_values(critic, CONV_ENTITY, staged)
    selected = torch.tensor([4, 1, 0], dtype=torch.long)

    restricted = replay_behavior_values(critic, CONV_ENTITY, staged, states=selected, chunk_size=2)

    assert restricted.shape == (selected.numel(),)
    torch.testing.assert_close(restricted, whole_wave[selected])


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

    monkeypatch.setattr(kaggriculture.ppo, "replay_behavior_logprobs", forbidden_replay)
    config = PpoConfig(epochs=2, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    actor_before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    critic_before = {name: value.detach().clone() for name, value in critic.named_parameters()}

    metrics = update_ppo(
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
        update_ppo(
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


def _pin_structured_quantity_orders(actor: StructuredActor) -> None:
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-12.0)
        actor.market_kind.bias[MarketKind.STOP] = -6.0
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 6.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[MarketKind.BUY_SEED_WHEAT, -1] = 50.0


def _structured_rollout_with_quantity_orders(seed_start: int, sampling_seed: int):
    actor = StructuredActor(_small_structured_config())
    _pin_structured_quantity_orders(actor)
    rollout = collect_self_play(
        actor, games=1, seed_start=seed_start, episode_steps=8, sampling_seed=sampling_seed
    )
    return actor, rollout


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_update_stream_working_set_is_reused_across_iterations() -> None:
    """Repeated updates must not strand activation caches on fresh streams."""
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("stream-local cache accounting requires the native allocator")
    device = torch.device("cuda", torch.cuda.current_device())
    caller = torch.cuda.current_stream(device)

    def allocate_working_set() -> None:
        for stream in kaggriculture.ppo._update_cuda_streams(device):
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                activation = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
                activation.fill_(1)
            caller.wait_stream(stream)
            del activation
        caller.synchronize()

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    try:
        allocate_working_set()
        warm_reserved = torch.cuda.memory_reserved(device)
        for _ in range(6):
            allocate_working_set()
        assert torch.cuda.memory_reserved(device) == warm_reserved
    finally:
        torch.cuda.empty_cache()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_actor_refreshes_projection_caches_after_one_ppo_minibatch() -> None:
    torch.manual_seed(0)
    fused_config = replace(
        _small_structured_config(),
        model_dim=128,
        attention_heads=4,
        ffn_multiplier=2,
        fused_mlp=True,
    )
    actor = StructuredActor(fused_config).cuda()
    _pin_structured_quantity_orders(actor)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rollout = collect_self_play(
            actor,
            games=1,
            seed_start=219,
            episode_steps=8,
            sampling_seed=41,
        )
    rollout.rewards[:] = np.random.default_rng(42).normal(
        0.0,
        0.05,
        size=rollout.rewards.shape,
    )
    critic = StructuredCritic(_small_structured_config()).cuda()
    config = PpoConfig(
        actor_learning_rate=1.0e-2,
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=True,
        update_compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    fused_layers = [module for module in actor.modules() if isinstance(module, FusedFeedForward)]
    assert fused_layers
    master_before = [module.up_weight.detach().clone() for module in fused_layers]
    cache_before = [module._up_weight_bf16.detach().clone() for module in fused_layers]

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
    )

    assert metrics["actor_minibatches_intended"] == 1
    assert metrics["actor_updates"] == 1
    changed_layers = [
        index
        for index, module in enumerate(fused_layers)
        if not torch.equal(module.up_weight, master_before[index])
    ]
    assert changed_layers
    assert any(
        not torch.equal(fused_layers[index]._up_weight_bf16, cache_before[index])
        for index in changed_layers
    )
    for module in fused_layers:
        assert module._fp8_ready
        torch.testing.assert_close(
            module._up_weight_bf16,
            module.up_weight.bfloat16(),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            module._down_weight_bf16,
            module.down_weight.bfloat16(),
            rtol=0.0,
            atol=0.0,
        )
        assert module._up_weight_f8.numel() == module.up_weight.numel()
        assert module._down_weight_f8_storage.numel() == module.down_weight.numel()
        assert torch.isfinite(module._up_weight_f8.float()).all()
        assert torch.isfinite(module._down_weight_f8_storage.float()).all()

    flat_valid_index = int(np.flatnonzero(rollout.valid.reshape(-1))[0])
    forward_states = {
        name: values.reshape((-1, *values.shape[2:]))[[flat_valid_index]]
        for name, values in rollout.states.items()
    }
    forward_states["unit_active"] = rollout.unit_active.reshape(
        (-1, *rollout.unit_active.shape[2:])
    )[[flat_valid_index]]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = actor(*actor_forward_args(STRUCTURED, forward_states, torch.device("cuda")))
        quantity_logits = actor.quantity_logits(
            output.market_quantity_context,
            output.market_kind_logits.argmax(dim=-1),
        )
    assert torch.isfinite(output.unit_logits).all()
    assert torch.isfinite(output.market_kind_logits).all()
    assert torch.isfinite(output.market_quantity_context).all()
    assert torch.isfinite(quantity_logits).all()


def test_structured_update_replay_parity_covers_every_component() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=201, sampling_seed=21)
    assert rollout.architecture == STRUCTURED
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}

    parity = update_replay_parity(
        actor,
        rollout,
        minibatch_size=3,
        compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
        autocast_enabled=False,
    )

    for component in ("unit", "kind", "quantity"):
        assert parity[f"update_replay_{component}_active_count"] > 0
        assert parity[f"update_replay_{component}_ratio_max_abs_error"] < 1e-4
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)


def test_structured_update_ppo_trains_both_networks() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=203, sampling_seed=23)
    critic = StructuredCritic(_small_structured_config())
    config = PpoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    actor_before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    critic_before = {name: value.detach().clone() for name, value in critic.named_parameters()}

    metrics = update_ppo(
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


def test_structured_transition_order_never_crosses_trajectory_or_terminal_boundaries() -> None:
    valid = np.array(
        [
            [True, True, True, True, False, False],
            [True, True, False, True, True, True],
            [True, True, True, True, True, True],
        ],
        dtype=np.bool_,
    )
    order = _structured_transition_order(
        valid,
        np.array([0, 1], dtype=np.int64),
        2,
        np.random.default_rng(9),
    )

    assert order.shape == (3, 3)
    for window in order:
        trajectories, steps = np.divmod(window, valid.shape[1])
        assert np.unique(trajectories).size == 1
        np.testing.assert_array_equal(steps, np.arange(steps[0], steps[0] + 3))
        assert valid[trajectories, steps].all()
        assert trajectories[0] != 2


def test_structured_window_loss_matches_generic_masked_unroll() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(
        seed_start=205,
        sampling_seed=27,
    )
    config = PpoConfig(
        epochs=1,
        minibatch_size=1 << 12,
        use_bfloat16=False,
        structured_decision_coefficient=0.5,
        structured_patch_coefficient=0.5,
        structured_economy_coefficient=0.5,
        structured_opponent_summary_coefficient=0.5,
        structured_opponent_patch_coefficient=0.5,
        structured_decision_horizon=2,
        structured_patch_horizon=1,
    )
    dynamics = StructuredDynamics(_small_structured_config())
    device = torch.device("cpu")
    staged = {name: _stage_tensor(array, device) for name, array in rollout.states.items()}
    staged |= {
        "unit_actions": _stage_tensor(rollout.unit_actions, device),
        "market_kinds": _stage_tensor(rollout.market_kinds, device),
        "market_quantities": _stage_tensor(rollout.market_quantities, device),
        "unit_masks": _stage_tensor(rollout.unit_masks, device),
        "market_kind_masks": _stage_tensor(rollout.market_kind_masks, device),
        "market_quantity_masks": _stage_tensor(rollout.market_quantity_masks, device),
        "unit_active": _stage_tensor(rollout.unit_active, device),
        "market_active": _stage_tensor(rollout.market_active, device),
        "market_quantity_active": _stage_tensor(rollout.market_quantity_active, device),
    }
    windows = _structured_transition_order(
        rollout.valid,
        None,
        2,
        np.random.default_rng(28),
    )
    indices = torch.from_numpy(windows.reshape(-1))
    unique_indices, inverse = np.unique(windows.reshape(-1), return_inverse=True)
    belief_indices = torch.from_numpy(unique_indices)
    belief_inverse = torch.from_numpy(inverse)

    with torch.inference_mode():
        generic_loss, generic = _structured_auxiliary_terms(
            actor,
            dynamics,
            staged,
            indices,
            steps_per_trajectory=rollout.valid.shape[1],
            config=config,
            autocast_enabled=False,
            model_grad=False,
            complete_windows=False,
        )
        window_loss, windowed = _structured_auxiliary_terms(
            actor,
            dynamics,
            staged,
            indices,
            steps_per_trajectory=rollout.valid.shape[1],
            config=config,
            autocast_enabled=False,
            model_grad=False,
            complete_windows=True,
            belief_indices=belief_indices,
            belief_inverse=belief_inverse,
        )
        episodes, steps = np.divmod(indices.numpy(), rollout.valid.shape[1])
        compact_loss, compact = _structured_auxiliary_terms(
            actor,
            dynamics,
            staged,
            indices,
            steps_per_trajectory=rollout.valid.shape[1],
            config=config,
            autocast_enabled=False,
            model_grad=False,
            complete_windows=False,
            plan=structured_horizon_plan(episodes, steps, 2),
        )

    torch.testing.assert_close(window_loss, generic_loss)
    for window_value, generic_value in zip(windowed, generic, strict=True):
        torch.testing.assert_close(window_value, generic_value)
    torch.testing.assert_close(compact_loss, generic_loss)
    for compact_value, generic_value in zip(compact, generic, strict=True):
        torch.testing.assert_close(compact_value, generic_value)


def test_multistep_structured_dynamics_activates_central_workspace() -> None:
    common = {
        "decision_horizon": 0,
        "own_patches_active": True,
        "economy_active": False,
        "opponent_summary_active": False,
        "opponent_patches_active": False,
    }

    single_step = _active_belief_fields(recurrent_workspace=False, **common)
    multi_step = _active_belief_fields(recurrent_workspace=True, **common)

    assert not single_step[StructuredBelief._fields.index("central_latents")]
    assert multi_step[StructuredBelief._fields.index("central_latents")]


def test_sparse_structured_transition_preserves_active_belief_families() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(
        seed_start=206,
        sampling_seed=28,
    )
    dynamics = StructuredDynamics(_small_structured_config())
    device = torch.device("cpu")
    staged = {name: _stage_tensor(array, device) for name, array in rollout.states.items()}
    staged["unit_active"] = _stage_tensor(rollout.unit_active, device)
    staged["unit_actions"] = _stage_tensor(rollout.unit_actions, device)
    staged["market_kinds"] = _stage_tensor(rollout.market_kinds, device)
    staged["market_quantities"] = _stage_tensor(rollout.market_quantities, device)
    indices = torch.arange(3)
    (inputs,) = _actor_batch_args(STRUCTURED, staged, indices)
    _, belief = actor.forward_with_belief(inputs)

    full = dynamics(
        belief,
        staged["unit_actions"][indices].long(),
        staged["market_kinds"][indices].long(),
        staged["market_quantities"][indices].long(),
        inputs.unit_categorical,
        inputs.unit_active,
    )
    active_fields = (False, False, True, False, False, True, True)
    sparse = dynamics(
        belief,
        staged["unit_actions"][indices].long(),
        staged["market_kinds"][indices].long(),
        staged["market_quantities"][indices].long(),
        inputs.unit_categorical,
        inputs.unit_active,
        active_fields=active_fields,
    )
    for position, (full_value, sparse_value, input_value) in enumerate(
        zip(full, sparse, belief, strict=True)
    ):
        if active_fields[position]:
            torch.testing.assert_close(sparse_value, full_value)
        else:
            torch.testing.assert_close(sparse_value, input_value)


def test_active_structured_auxiliary_clips_nextlat_without_clipping_the_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=207, sampling_seed=29)
    control_actor = copy.deepcopy(actor)
    active_actor = copy.deepcopy(actor)
    control_critic = StructuredCritic(_small_structured_config())
    active_critic = copy.deepcopy(control_critic)
    control_config = PpoConfig(
        epochs=1,
        minibatch_size=1 << 12,
        use_bfloat16=False,
    )
    active_config = replace(
        control_config,
        structured_decision_coefficient=0.5,
        structured_opponent_summary_coefficient=0.5,
        structured_opponent_patch_coefficient=0.5,
        structured_decision_horizon=2,
        structured_patch_horizon=1,
        nextlat_max_gradient_norm=1.0e-4,
    )
    dynamics = StructuredDynamics(_small_structured_config())
    control_optimizers = make_optimizers(control_actor, control_critic, control_config)
    active_optimizers = make_optimizers(active_actor, active_critic, active_config)
    dynamics_optimizer = make_structured_dynamics_optimizer(dynamics, active_config)
    actor_optimized_parameters = {
        id(parameter)
        for group in active_optimizers[0].param_groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) not in actor_optimized_parameters for parameter in dynamics.parameters()
    )
    actor_before = {name: value.detach().clone() for name, value in active_actor.named_parameters()}
    dynamics_before = {name: value.detach().clone() for name, value in dynamics.named_parameters()}
    control_generator = np.random.default_rng(31)
    active_generator = np.random.default_rng(31)

    backward_calls = 0
    original_backward = torch.Tensor.backward
    predictor_requires_grad: list[bool] = []

    def record_backward(tensor, *args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        return original_backward(tensor, *args, **kwargs)

    def record_predictor_state(_module, _args):
        predictor_requires_grad.append(
            any(parameter.requires_grad for parameter in dynamics.parameters())
        )

    monkeypatch.setattr(torch.Tensor, "backward", record_backward)
    dynamics.register_forward_pre_hook(record_predictor_state)
    control_metrics = update_ppo(
        control_actor,
        control_critic,
        *control_optimizers,
        rollout,
        control_config,
        generator=control_generator,
    )
    active_metrics = update_ppo(
        active_actor,
        active_critic,
        *active_optimizers,
        rollout,
        active_config,
        generator=active_generator,
        structured_dynamics=dynamics,
        structured_dynamics_optimizer=dynamics_optimizer,
        structured_actor_auxiliary=True,
        auxiliary_generator=np.random.default_rng(32),
    )

    assert np.random.default_rng(31).bit_generator.state == active_generator.bit_generator.state
    assert control_generator.bit_generator.state != active_generator.bit_generator.state
    assert not any(name.startswith("structured_") for name in control_metrics)
    assert active_metrics["structured_actor_predictor_updates"] >= 1
    assert active_metrics["structured_actor_auxiliary_updates"] == 1
    assert active_metrics["structured_actor_eligible"] > 0.0
    assert backward_calls == (
        control_metrics["actor_updates"]
        + control_metrics["updates"]
        + active_metrics["actor_updates"]
        + active_metrics["updates"]
    )
    assert active_metrics["actor_gradient_norm"] > active_config.nextlat_max_gradient_norm
    assert predictor_requires_grad
    assert all(predictor_requires_grad)
    assert all(parameter.requires_grad for parameter in dynamics.parameters())
    assert active_metrics["structured_actor_combined_gradient_norm"] == pytest.approx(
        active_metrics["actor_gradient_norm"]
    )
    for name in (
        "structured_preupdate_combined",
        "structured_preupdate_decision",
        "structured_preupdate_persistence_combined",
        "structured_preupdate_persistence_decision",
        "structured_preupdate_persistence_patch",
        "structured_preupdate_persistence_economy",
        "structured_preupdate_persistence_opponent_summary",
        "structured_preupdate_persistence_opponent_patches",
        "structured_actor_opponent_summary",
        "structured_actor_opponent_patches",
        "structured_actor_opponent_patch_all",
        "structured_actor_opponent_patch_changed",
        "structured_actor_opponent_patch_unchanged",
        "structured_actor_decision",
        "structured_actor_residual_ratio",
    ):
        assert math.isfinite(active_metrics[name])
    assert active_metrics["structured_preupdate_decision"] > 0.0
    assert active_metrics["structured_preupdate_opponent_summary"] > 0.0
    assert active_metrics["structured_preupdate_opponent_patches"] > 0.0
    assert active_metrics["structured_actor_combined_gradient_norm"] > 0.0
    assert active_metrics["structured_actor_opponent_patches"] == pytest.approx(
        0.5
        * (
            active_metrics["structured_actor_opponent_patch_all"]
            + active_metrics["structured_actor_opponent_patch_changed"]
        )
    )
    assert active_metrics["structured_actor_opponent_patch_unchanged"] > 0.0
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in active_actor.named_parameters()
    )
    assert any(
        not torch.equal(value, dynamics_before[name]) for name, value in dynamics.named_parameters()
    )


def test_structured_predictor_trains_during_critic_only_warmup_without_actor_gradients() -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=211, sampling_seed=41)
    critic = StructuredCritic(_small_structured_config())
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        use_bfloat16=False,
        structured_decision_coefficient=0.5,
        structured_opponent_summary_coefficient=0.5,
        structured_opponent_patch_coefficient=0.5,
    )
    dynamics = StructuredDynamics(_small_structured_config())
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    dynamics_optimizer = make_structured_dynamics_optimizer(dynamics, config)
    actor_before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    dynamics_before = {name: value.detach().clone() for name, value in dynamics.named_parameters()}

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
        actor_epochs=0,
        structured_dynamics=dynamics,
        structured_dynamics_optimizer=dynamics_optimizer,
        structured_actor_auxiliary=False,
        auxiliary_generator=np.random.default_rng(44),
    )

    assert metrics["actor_updates"] == 0
    assert metrics["structured_actor_predictor_updates"] >= 1
    assert metrics["structured_actor_auxiliary_enabled"] == 0
    assert all(torch.equal(value, actor_before[name]) for name, value in actor.named_parameters())
    assert any(
        not torch.equal(value, dynamics_before[name]) for name, value in dynamics.named_parameters()
    )
    assert all(parameter.grad is None for parameter in actor.parameters())
    # Warmup still advances the fresh-wave gate, even without gradient diagnostics.
    gate_fields = (
        "combined",
        "decision",
        "patch",
        "economy",
        "opponent_summary",
        "opponent_patches",
    )
    for name in gate_fields:
        assert math.isfinite(metrics[f"structured_preupdate_persistence_{name}"])


def test_actor_and_critic_nextlat_clip_only_the_auxiliary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=221, sampling_seed=47)
    critic = StructuredCritic(_small_structured_config())
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
        structured_decision_coefficient=0.5,
        structured_opponent_summary_coefficient=0.5,
        structured_critic_latent_coefficient=0.5,
        structured_critic_value_coefficient=0.5,
        structured_critic_horizon=1,
    )
    actor_dynamics = StructuredDynamics(_small_structured_config())
    critic_dynamics = StructuredCriticDynamics(_small_structured_config())
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    actor_dynamics_optimizer = make_structured_dynamics_optimizer(actor_dynamics, config)
    critic_dynamics_optimizer = make_structured_dynamics_optimizer(critic_dynamics, config)
    actor_parameters = {id(parameter) for parameter in actor.parameters()}
    critic_head_ids = {id(parameter) for parameter in critic.value_head.parameters()}
    critic_trunk_parameters = {
        id(parameter) for parameter in critic.parameters() if id(parameter) not in critic_head_ids
    }
    backward_calls = 0
    clipped_parameter_sets: list[set[int]] = []
    critic_predictor_requires_grad: list[bool] = []
    original_backward = torch.Tensor.backward
    original_clip = torch.nn.utils.clip_grad_norm_
    original_optimizer_step = kaggriculture.ppo._optimizer_step
    model_clock_observations: list[tuple[int, int, int, int]] = []

    def record_backward(tensor, *args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        return original_backward(tensor, *args, **kwargs)

    def record_clip(parameters, *args, **kwargs):
        materialized = tuple(parameters)
        clipped_parameter_sets.append({id(parameter) for parameter in materialized})
        return original_clip(materialized, *args, **kwargs)

    def record_critic_predictor_state(_module, _args):
        critic_predictor_requires_grad.append(
            any(parameter.requires_grad for parameter in critic_dynamics.parameters())
        )

    def record_optimizer_step(optimizer, *args, **kwargs):
        if optimizer is actor_optimizer or optimizer is critic_optimizer:
            model_clock_observations.append(
                (
                    actor_optimizer.param_groups[0]["warmup_step"],
                    critic_optimizer.param_groups[0]["warmup_step"],
                    actor_dynamics_optimizer.param_groups[0]["warmup_step"],
                    critic_dynamics_optimizer.param_groups[0]["warmup_step"],
                )
            )
        return original_optimizer_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", record_backward)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    monkeypatch.setattr(kaggriculture.ppo, "_optimizer_step", record_optimizer_step)
    critic_dynamics.register_forward_pre_hook(record_critic_predictor_state)

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(48),
        structured_dynamics=actor_dynamics,
        structured_dynamics_optimizer=actor_dynamics_optimizer,
        structured_actor_auxiliary=True,
        structured_critic_dynamics=critic_dynamics,
        structured_critic_dynamics_optimizer=critic_dynamics_optimizer,
        structured_critic_auxiliary=True,
        auxiliary_generator=np.random.default_rng(49),
    )

    assert backward_calls == metrics["actor_updates"] + metrics["updates"]
    assert metrics["actor_updates"] == 1
    assert metrics["updates"] == 1
    assert metrics["structured_actor_predictor_updates"] == 1
    assert metrics["structured_critic_predictor_updates"] == 1
    assert actor_parameters not in clipped_parameter_sets
    assert critic_trunk_parameters not in clipped_parameter_sets
    assert {id(parameter) for parameter in actor_dynamics.parameters()} in clipped_parameter_sets
    assert {id(parameter) for parameter in critic_dynamics.parameters()} in clipped_parameter_sets
    assert critic_predictor_requires_grad
    assert all(critic_predictor_requires_grad)
    assert (
        torch.nn.utils.get_total_norm(
            [
                parameter.grad
                for parameter in actor_dynamics.parameters()
                if parameter.grad is not None
            ]
        )
        <= config.nextlat_max_gradient_norm + 1e-7
    )
    assert (
        torch.nn.utils.get_total_norm(
            [
                parameter.grad
                for parameter in critic_dynamics.parameters()
                if parameter.grad is not None
            ]
        )
        <= config.nextlat_max_gradient_norm + 1e-7
    )
    assert model_clock_observations[0] == (0, 0, 0, 0)
    assert actor_optimizer.param_groups[0]["warmup_step"] == metrics["actor_updates"]
    assert critic_optimizer.param_groups[0]["warmup_step"] == metrics["updates"]
    assert (
        actor_dynamics_optimizer.param_groups[0]["warmup_step"]
        == metrics["structured_actor_predictor_updates"]
    )
    assert (
        critic_dynamics_optimizer.param_groups[0]["warmup_step"]
        == metrics["structured_critic_predictor_updates"]
    )
    for name in (
        "structured_actor_auxiliary_loss",
        "structured_actor_combined_loss",
        "structured_actor_combined_gradient_norm",
        "structured_critic_auxiliary_loss",
        "structured_critic_combined_loss",
        "structured_critic_combined_gradient_norm",
        "structured_critic_preupdate_combined",
        "structured_critic_preupdate_latent",
        "structured_critic_preupdate_value",
        "structured_critic_preupdate_persistence_combined",
        "structured_critic_preupdate_persistence_latent",
        "structured_critic_preupdate_persistence_value",
        "structured_critic_latent",
        "structured_critic_value",
    ):
        assert math.isfinite(metrics[name]), name


def test_critic_auxiliary_gradient_excludes_predictor_and_value_teacher() -> None:
    _actor, rollout = _structured_rollout_with_quantity_orders(seed_start=223, sampling_seed=51)
    critic = StructuredCritic(_small_structured_config())
    dynamics = StructuredCriticDynamics(_small_structured_config())
    config = PpoConfig(
        structured_critic_latent_coefficient=1.0,
        structured_critic_value_coefficient=1.0,
        structured_critic_horizon=1,
        use_bfloat16=False,
    )
    staged = {
        name: _stage_tensor(array, torch.device("cpu")) for name, array in rollout.states.items()
    }
    staged |= {
        "unit_actions": _stage_tensor(rollout.unit_actions, torch.device("cpu")),
        "market_kinds": _stage_tensor(rollout.market_kinds, torch.device("cpu")),
        "market_quantities": _stage_tensor(rollout.market_quantities, torch.device("cpu")),
        "unit_active": _stage_tensor(rollout.unit_active, torch.device("cpu")),
    }
    windows = _structured_transition_order(
        rollout.valid,
        None,
        config.structured_critic_horizon,
        np.random.default_rng(52),
    )
    indices = torch.from_numpy(windows[:1].reshape(-1))
    for parameter in dynamics.parameters():
        parameter.requires_grad_(False)

    loss, terms = _structured_critic_auxiliary_terms(
        critic,
        dynamics,
        staged,
        indices,
        steps_per_trajectory=rollout.valid.shape[1],
        config=config,
        autocast_enabled=False,
        model_grad=True,
        complete_windows=True,
    )
    loss.backward()

    head_parameters = tuple(critic.value_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    assert terms.eligible > 0
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum())
        for parameter in critic.parameters()
        if id(parameter) not in head_ids
    )
    assert all(parameter.grad is None for parameter in head_parameters)
    assert all(parameter.grad is None for parameter in dynamics.parameters())


def test_optimizer_ownership_requires_exact_disjoint_pairs() -> None:
    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    first_optimizer = torch.optim.SGD(first.parameters(), lr=0.1)
    second_optimizer = torch.optim.SGD(second.parameters(), lr=0.1)
    _validate_optimizer_ownership(
        (
            ("first", first, first_optimizer),
            ("second", second, second_optimizer),
        )
    )

    incomplete = torch.optim.SGD((first.weight,), lr=0.1)
    with pytest.raises(ValueError, match="must own exactly"):
        _validate_optimizer_ownership((("first", first, incomplete),))

    second.weight = first.weight
    overlapping = torch.optim.SGD(second.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="parameters must be disjoint"):
        _validate_optimizer_ownership(
            (
                ("first", first, first_optimizer),
                ("second", second, overlapping),
            )
        )


@pytest.mark.parametrize("source", ["actor", "critic"])
def test_nonfinite_combined_auxiliary_loss_does_not_advance_source_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    actor, rollout = _structured_rollout_with_quantity_orders(seed_start=225, sampling_seed=53)
    critic = StructuredCritic(_small_structured_config())
    actor_active = source == "actor"
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
        structured_decision_coefficient=0.5 if actor_active else 0.0,
        structured_critic_latent_coefficient=0.5 if not actor_active else 0.0,
    )
    dynamics = (
        StructuredDynamics(_small_structured_config())
        if actor_active
        else StructuredCriticDynamics(_small_structured_config())
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    dynamics_optimizer = make_structured_dynamics_optimizer(dynamics, config)
    source_model = actor if actor_active else critic
    source_optimizer = actor_optimizer if actor_active else critic_optimizer
    source_before = {
        name: parameter.detach().clone() for name, parameter in source_model.named_parameters()
    }
    helper_name = (
        "_structured_auxiliary_terms" if actor_active else "_structured_critic_auxiliary_terms"
    )
    original_helper = getattr(kaggriculture.ppo, helper_name)

    def poison_representation_loss(*args, **kwargs):
        loss, terms = original_helper(*args, **kwargs)
        if kwargs["model_grad"]:
            loss = loss * loss.new_tensor(float("nan"))
        return loss, terms

    monkeypatch.setattr(kaggriculture.ppo, helper_name, poison_representation_loss)
    kwargs = (
        {
            "structured_dynamics": dynamics,
            "structured_dynamics_optimizer": dynamics_optimizer,
            "structured_actor_auxiliary": True,
        }
        if actor_active
        else {
            "structured_critic_dynamics": dynamics,
            "structured_critic_dynamics_optimizer": dynamics_optimizer,
            "structured_critic_auxiliary": True,
        }
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(54),
            auxiliary_generator=np.random.default_rng(55),
            **kwargs,
        )

    assert source_optimizer.param_groups[0]["warmup_step"] == 0
    assert not source_optimizer.state
    for name, parameter in source_model.named_parameters():
        torch.testing.assert_close(parameter, source_before[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("optimizer", ["adamw", "normuon"])
def test_ppo_optimizers_never_apply_weight_decay(optimizer: str) -> None:
    model_config = ModelConfig(
        cnn_width=8,
        cnn_blocks=1,
        model_dim=16,
        transformer_layers=3,
        attention_heads=2,
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    dynamics = StructuredDynamics(_small_structured_config())
    config = PpoConfig(optimizer=optimizer)

    optimizers = (
        *make_optimizers(actor, critic, config),
        make_structured_dynamics_optimizer(dynamics, config),
    )

    assert all(
        group["weight_decay"] == 0.0 for optimizer in optimizers for group in optimizer.param_groups
    )


@pytest.mark.parametrize("optimizer", ["adamw", "normuon"])
def test_structured_predictor_optimizers_keep_independent_base_rates(optimizer: str) -> None:
    actor_dynamics = StructuredDynamics(_small_structured_config())
    critic_dynamics = StructuredCriticDynamics(_small_structured_config())
    config = PpoConfig(
        optimizer=optimizer,
        actor_learning_rate=3.0e-5,
        critic_learning_rate=2.5e-4,
    )

    actor_optimizer = make_structured_dynamics_optimizer(actor_dynamics, config)
    critic_optimizer = make_structured_dynamics_optimizer(critic_dynamics, config)

    assert actor_optimizer.param_groups[0]["base_lr"] == pytest.approx(
        config.resolved_structured_learning_rate
    )
    assert critic_optimizer.param_groups[0]["base_lr"] == pytest.approx(
        config.resolved_structured_critic_learning_rate
    )


def test_target_correlation_separates_noise_from_a_mis_scaled_critic() -> None:
    """Explained variance goes negative two ways that need different fixes.

    A critic predicting uncorrelated noise and one that ranks states correctly
    at the wrong scale both score below zero on explained variance. Only the
    correlation tells them apart, which is why it is recorded beside it.
    """
    valid = np.ones(256, dtype=np.bool_)
    generator = np.random.default_rng(11)
    targets = generator.normal(size=256)

    noise = generator.normal(size=256)
    assert _explained_variance(targets, noise, valid) < 0.0
    assert abs(_target_correlation(targets, noise, valid)) < 0.2

    # Perfectly ranked, three times too large: the residual is twice the target
    # so explained variance lands on exactly -3, while the correlation is one.
    mis_scaled = targets * 3.0
    assert _explained_variance(targets, mis_scaled, valid) == pytest.approx(-3.0)
    assert _target_correlation(targets, mis_scaled, valid) == pytest.approx(1.0)

    # A constant predictor has no correlation to report rather than a wrong one.
    assert _target_correlation(targets, np.zeros(256), valid) == 0.0


def test_a_mis_scaled_critic_explains_the_policy_lambda_return_not_monte_carlo() -> None:
    """Decoupled GAE: the critic target is Monte Carlo; the policy lambda-return is not.

    A critic predicting three times the true return scores exactly -3 against
    the suffix return. That suffix is now also the critic target, so the same
    -3 is the value-target reading. The policy lambda-return is still built
    from that prediction, so the same critic scores near one against it.
    """
    generator = np.random.default_rng(5)
    rewards = generator.normal(0.002, 0.001, size=(4, 64)).astype(np.float32)
    valid = np.ones_like(rewards, dtype=np.bool_)
    monte_carlo = np.flip(
        np.cumsum(np.flip(rewards, axis=1), axis=1),
        axis=1,
    ).copy()
    mis_scaled = (3.0 * monte_carlo).astype(np.float32)

    prepared = prepare_advantages(
        SimpleNamespace(rewards=rewards, valid=valid),
        mis_scaled,
        PpoConfig(actor_gae_lambda=0.5, critic_gae_lambda=1.0, gamma=1.0),
    )

    np.testing.assert_allclose(prepared.monte_carlo_returns, monte_carlo, atol=1e-6)
    np.testing.assert_allclose(prepared.value_targets, monte_carlo, atol=1e-6)
    assert _explained_variance(prepared.monte_carlo_returns, mis_scaled, valid) == pytest.approx(
        -3.0
    )
    assert _explained_variance(prepared.value_targets, mis_scaled, valid) == pytest.approx(-3.0)
    assert _explained_variance(prepared.policy_lambda_returns, mis_scaled, valid) > 0.8


def _fit_sums(targets: np.ndarray, predictions: np.ndarray) -> dict[str, torch.Tensor]:
    """The four float64 accumulators `update_ppo` streams over its last epoch."""
    residuals = targets - predictions
    return {
        "target": torch.tensor(targets.sum(), dtype=torch.float64),
        "target_square": torch.tensor(np.square(targets).sum(), dtype=torch.float64),
        "residual": torch.tensor(residuals.sum(), dtype=torch.float64),
        "residual_square": torch.tensor(np.square(residuals).sum(), dtype=torch.float64),
    }


def test_streamed_fit_explained_variance_matches_the_array_form() -> None:
    """The streamed sums must compute the same statistic as the array helper.

    `_fit_explained_variance` exists only to avoid retaining a prediction per
    state, so it has to agree with `_explained_variance` to the last digit that
    matters. Sums of squares reach a variance through a different route --
    `E[x^2] - E[x]^2` rather than a subtracted mean -- and the mistakes that
    route invites leave numbers that are still plausible on a chart: dropping
    the mean subtraction reads 0.89 where the truth is 0.81, and a ddof applied
    to one of the two variances but not the other reads 0.806. A consistent
    ddof cancels between numerator and denominator and is the one such slip
    that cannot matter. Only a comparison against the array form on data with a
    non-zero mean can see the rest.
    """
    generator = np.random.default_rng(23)
    states = 512
    valid = np.ones(states, dtype=np.bool_)
    # A mean well away from zero, since that is what separates the two variance
    # formulas, and a prediction that explains some but not all of the target.
    targets = generator.normal(4.0, 1.5, size=states)
    predictions = 0.7 * targets + generator.normal(0.0, 0.5, size=states)

    streamed = _fit_explained_variance(_fit_sums(targets, predictions), states)

    assert streamed == pytest.approx(_explained_variance(targets, predictions, valid), abs=1e-9)
    # Not near the degenerate ends, so the agreement above is a measurement.
    assert 0.2 < streamed < 0.95

    # A critic that hits its target exactly explains all of it.
    assert _fit_explained_variance(_fit_sums(targets, targets.copy()), states) == pytest.approx(
        1.0, abs=1e-9
    )

    # A constant target has no variance to explain, and the ratio would divide
    # by the float64 residue of `E[x^2] - E[x]^2` rather than by zero, so the
    # result would be arbitrary rather than an obvious error.
    constant = np.full(states, 2.5)
    assert (
        _fit_explained_variance(_fit_sums(constant, generator.normal(size=states)), states) == 0.0
    )

    # Fewer than two states cannot carry a variance: the population variance of
    # a single observation is zero, so the ratio would be 0/0. Reporting nothing
    # measured is the honest answer.
    assert _fit_explained_variance(_fit_sums(targets[:1], predictions[:1]), 1) == 0.0
    assert _fit_explained_variance(_fit_sums(targets[:0], predictions[:0]), 0) == 0.0


def test_the_critic_fit_reading_is_the_only_one_that_can_see_a_working_refit() -> None:
    """Why a third explained variance had to exist beside the conventional one.

    `lambda_return_explained_variance` is scored against `advantages + values`
    using those same pre-update `values`, so its residual is identically the
    advantage and the whole number is the algebra `1 - Var(A)/Var(G_lambda)` --
    which this test recomputes from the two standard deviations the update
    already reports, to pin that it is an identity and not a measurement of any
    fit. Nothing the critic learns during the update can move it, and it rises
    on its own whenever the critic's predictions merely gain variance.

    So a critic here refits its targets well enough to explain a third of their
    variance while that conventional reading sits near zero. Quoting it as
    "explained variance" is what made a working critic look broken; the two
    `critic_fit_explained_variance_*` readings, taken from the critic's own
    predictions during the update, are what say the regression worked. They are
    reported as a pair because the last epoch scores states it has already been
    fitted to and the first does not, so only the gap between them separates a
    critic that is fitting from one that is memorizing the batch.
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    # Pinned because the readings below are compared against thresholds, and an
    # unseeded init makes those a property of whichever tests ran first: this
    # file seeds nothing, so the global stream's position when it arrives here
    # decides the numbers. It previously drew a last-epoch fit of 0.052 -- below
    # this test's own floor -- purely because tests were added above it.
    torch.manual_seed(6)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=32, sampling_seed=3)
    # A fresh game's shaped rewards are all around 1e-9, which leaves the
    # targets with no variance for a critic to explain. This is the reward
    # scale a real episode reaches.
    rollout.rewards[:] = (
        np.random.default_rng(11).normal(0.0, 0.3, size=rollout.rewards.shape).astype(np.float32)
    )
    # Warmup is off and the critic learning rate is an order of magnitude above
    # production because 64 states have to be fitted inside one test: the
    # default 32-step warmup would not have finished ramping before the update
    # ends. `critic_epochs` above `epochs` is exactly the critic-only refit the
    # production warm-start phase runs.
    #
    # The rate is in NorMuon units, where it is the fraction of itself a matrix
    # moves per step; the 3e-3 this read under AdamW was an absolute per-element
    # step, worth `3e-3 * sqrt(fan_in)` relative at this model's width of 16.
    # Keeping the same relative step is what keeps this a test of the metrics
    # rather than of the optimizer.
    config = PpoConfig(
        epochs=1,
        critic_epochs=24,
        minibatch_size=32,
        use_bfloat16=False,
        lr_warmup_steps=0,
        critic_learning_rate=1.2e-2,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    # A randomly initialized distributional critic predicts one constant for
    # every state, which makes `Var(A)` exactly `Var(G_lambda)` and the
    # conventional reading exactly zero -- true, but degenerate, and it would
    # let the identity below hold as 0 == 0. One discarded update moves the
    # critic off that constant so the identity is checked against real numbers.
    update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(4),
    )
    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(5),
    )

    assert metrics["value_prediction_std"] > 0.0
    # The policy-lambda EV is an algebraic identity of A and G_policy, not a
    # critic-fit score. The fit reading is the one that can see a working
    # refit; this test keeps them distinct.

    # The fit reading is materially positive on a critic the conventional
    # reading calls weak, and their separation is what the pair exists for.
    # The long discounted suffix increases the target's explained variance;
    # this pinned fixture reads a 2.66x separation. A factor of two still fails
    # if the fit reading collapses toward the algebraic identity without
    # encoding an exact threshold for one sampled return distribution.
    assert metrics["critic_fit_explained_variance_last_epoch"] > 0.1
    assert metrics["critic_fit_explained_variance_last_epoch"] > 2.0 * abs(
        metrics["lambda_return_explained_variance"]
    )

    # Twenty-four critic epochs over 64 states is deliberately the memorizing
    # end of the scale, so the in-sample reading must sit above the epoch that
    # scored those states before this update had fitted them. Equality would
    # mean the pair carries one number twice.
    assert (
        metrics["critic_fit_explained_variance_last_epoch"]
        > metrics["critic_fit_explained_variance_first_epoch"]
    )


def test_advantage_statistics_match_the_surrogate_advantages() -> None:
    rewards = np.zeros((2, 4), dtype=np.float32)
    rewards[:, -1] = [6.0, -6.0]
    rollout = SimpleNamespace(
        rewards=rewards,
        valid=np.ones((2, 4), dtype=np.bool_),
    )
    values = np.full((2, 4), 2.0, dtype=np.float32)

    prepared = prepare_advantages(rollout, values, PpoConfig())

    raw, _targets = generalized_advantage_and_targets(
        torch.from_numpy(rewards),
        torch.from_numpy(values),
        torch.ones_like(torch.from_numpy(rewards)),
        gae_lambda=PpoConfig().actor_gae_lambda,
        gamma=PpoConfig().gamma,
    )
    selected = raw.reshape(-1)
    np.testing.assert_allclose(prepared.advantages.reshape(-1), selected.numpy(), rtol=1e-6)
    assert prepared.raw_advantage_mean == pytest.approx(float(selected.mean()), rel=1e-6)
    assert prepared.raw_advantage_std == pytest.approx(
        float(selected.std(unbiased=False)), rel=1e-6
    )
    assert prepared.raw_advantage_std > 2.0


def test_first_and_last_critic_epoch_losses_separate_fitting_from_memorizing() -> None:
    """One averaged loss cannot tell a generalizing critic from a memorizing one.

    Four critic epochs run over the same targets, so the reported mean mixes a
    pass the critic has never seen with three it has. The first epoch is the
    only out-of-sample reading, and its gap to the last is the diagnosis.
    """
    # Cumulative (loss-sum, state-count) marks at each epoch boundary.
    marks = [
        (torch.tensor(400.0, dtype=torch.float64), 100),
        (torch.tensor(600.0, dtype=torch.float64), 200),
        (torch.tensor(700.0, dtype=torch.float64), 300),
    ]

    first, last = _epoch_value_losses(marks)

    assert first == pytest.approx(4.0)
    assert last == pytest.approx(1.0)
    # A single epoch is its own first and last rather than an undefined last.
    assert _epoch_value_losses(marks[:1]) == (pytest.approx(4.0), pytest.approx(4.0))
    # A critic that never ran reports absence, not a loss of zero it achieved.
    assert _epoch_value_losses([]) == (0.0, 0.0)


def test_the_audited_first_minibatch_kl_uses_sampler_likelihoods() -> None:
    """The audit and update compare the same stored behavior distribution."""
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=41, episode_steps=4, sampling_seed=7)

    metrics = update_replay_parity(
        actor,
        rollout,
        minibatch_size=1 << 12,
        compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
        autocast_enabled=False,
    )

    sampling = metrics["update_replay_minibatch_kl"]
    update_graph = metrics["update_replay_first_minibatch_kl"]
    assert update_graph >= 0.0
    # One minibatch contains the whole fixture, so row shuffling cannot change
    # the component-weighted sampler-versus-update statistic.
    assert update_graph == pytest.approx(sampling, rel=1e-9, abs=1e-12)


def _population_wave() -> tuple[FarmActor, DistributionalCritic, object]:
    """Two games' four trajectories, laid out the way a population wave stores them.

    Game-major and seat-minor, so a member holding one seat of each game owns
    rows 0 and 3: never a contiguous block, which is why the update partitions
    on row indices instead of slicing the batch. The two members are given
    return scales a hundredfold apart so a pooled statistic cannot be mistaken
    for a partitioned one.
    """
    actor, critic = _small_update_models()
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
    generator = np.random.default_rng(11)
    for rows, scale in ((_QUIET_ROWS, 0.05), (_LOUD_ROWS, 5.0)):
        rollout.rewards[rows] = generator.normal(
            0.0, scale, size=(rows.size, rollout.horizon)
        ).astype(np.float32)
    return actor, critic, rollout


#: One member's rows and the other's, in the layout `_population_wave` builds.
_QUIET_ROWS = np.array([0, 3], dtype=np.int64)
_LOUD_ROWS = np.array([1, 2], dtype=np.int64)


def _trajectory_subset(rollout, rows: np.ndarray):
    """A batch physically containing only `rows`, in that order."""
    fields = {*_SHARED_ROLLOUT_FIELDS, *_TRAJECTORY_METADATA_FIELDS, "agents"}
    return replace(
        rollout,
        states={name: array[rows] for name, array in rollout.states.items()},
        **{field: getattr(rollout, field)[rows] for field in fields},
    )


def _partitioned_update(actor, critic, rollout, config: PpoConfig, rows: np.ndarray | None):
    """One update on fresh copies of the pair, so the runs stay independent.

    Phase timings are wall clock, not a function of the update, so they are the
    one thing two bit-identical updates legitimately disagree on.
    """
    run_actor = copy.deepcopy(actor)
    run_critic = copy.deepcopy(critic)
    actor_optimizer, critic_optimizer = make_optimizers(run_actor, run_critic, config)
    metrics = update_ppo(
        run_actor,
        run_critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(4),
        rows=rows,
    )
    metrics = {name: value for name, value in metrics.items() if not name.endswith("_seconds")}
    return metrics, run_actor, run_critic


def _same_weights(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    right_state = right.state_dict()
    return all(torch.equal(value, right_state[name]) for name, value in left.state_dict().items())


def test_a_partition_naming_every_row_is_the_unpartitioned_update_bit_for_bit() -> None:
    """`rows=None` stays production's single-learner path and must not move.

    A partition naming every row is the same states in the same order with the
    same RNG draws, so it has to reproduce the unpartitioned update exactly --
    every metric and every weight, not merely closely.
    """
    actor, critic, rollout = _population_wave()
    config = PpoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)

    unpartitioned, pooled_actor, pooled_critic = _partitioned_update(
        actor, critic, rollout, config, None
    )
    every_row, row_actor, row_critic = _partitioned_update(
        actor, critic, rollout, config, np.arange(rollout.trajectories, dtype=np.int64)
    )

    assert every_row == unpartitioned
    assert every_row["states"] == rollout.state_count
    assert _same_weights(row_actor, pooled_actor)
    assert _same_weights(row_critic, pooled_critic)


def test_a_row_restricted_update_is_the_update_on_a_batch_of_only_those_rows() -> None:
    """The partition must be a partition, not a reweighting.

    Restricting two rows of a four-row wave in place must reproduce the update a
    batch physically holding only those rows produces. Anything the other rows
    still reach -- the minibatch partition, the KL, the critic's targets, any
    reported statistic -- differs here if it leaks.
    """
    actor, critic, rollout = _population_wave()
    config = PpoConfig(epochs=1, minibatch_size=8, use_bfloat16=False)

    restricted, restricted_actor, restricted_critic = _partitioned_update(
        actor, critic, rollout, config, _QUIET_ROWS
    )
    physical, physical_actor, physical_critic = _partitioned_update(
        actor, critic, _trajectory_subset(rollout, _QUIET_ROWS), config, None
    )

    assert restricted == physical
    # Half the wave's states, so the restriction reached the sample index rather
    # than merely the metrics computed off it.
    assert restricted["states"] == rollout.state_count // 2
    assert _same_weights(restricted_actor, physical_actor)
    assert _same_weights(restricted_critic, physical_critic)


def test_a_row_restricted_parity_audit_is_the_audit_of_only_those_rows() -> None:
    """Every wave row was sampled by its own member's weights.

    Auditing the whole wave through one member's actor measures the distance
    between two policies and reports it as a staging defect, so the audit takes
    the same row partition the update does -- and restricting in place must
    equal auditing a batch that physically holds only those rows.
    """
    actor, _critic, rollout = _population_wave()
    audit = {
        "minibatch_size": 8,
        "compile_mode": UNCOMPILED_UPDATE_COMPILE_MODE,
        "autocast_enabled": False,
    }

    restricted = update_replay_parity(actor, rollout, **audit, rows=_QUIET_ROWS)
    physical = update_replay_parity(actor, _trajectory_subset(rollout, _QUIET_ROWS), **audit)
    whole_wave = update_replay_parity(actor, rollout, **audit)

    assert restricted == physical
    # The whole-wave audit is a different measurement over strictly more
    # components, which is what makes the restriction load-bearing rather than
    # cosmetic once those components belong to other members' policies.
    assert (
        whole_wave["update_replay_unit_active_count"]
        > restricted["update_replay_unit_active_count"]
    )


def test_actor_forward_args_reproduce_the_staged_minibatch_arguments() -> None:
    """Both actor-argument builders read one field and dtype table.

    A second copy of that table is silent when it drifts: a swapped argument or
    a narrowed dtype changes the logits without raising anything, so the
    host-side builder is pinned against the staged one the update itself uses,
    on both architectures.
    """
    device = torch.device("cpu")
    conv_rollout = collect_self_play(
        _small_update_models()[0], games=1, seed_start=98, episode_steps=4, sampling_seed=16
    )
    _structured_actor, structured_rollout = _structured_rollout_with_quantity_orders(
        seed_start=205, sampling_seed=27
    )
    selected = np.array([0, 3], dtype=np.int64)

    def flattened(args) -> list[torch.Tensor]:
        """The argument tuple's tensors, entering the structured named tuple."""
        return [
            tensor for value in args for tensor in (value if isinstance(value, tuple) else (value,))
        ]

    for architecture, rollout in ((CONV_ENTITY, conv_rollout), (STRUCTURED, structured_rollout)):
        # The structured actor's unit mask is a shared rollout field rather than
        # a state field, and the update stages it beside the states.
        source = {**rollout.states, "unit_active": rollout.unit_active}
        staged = {name: _stage_tensor(array, device) for name, array in source.items()}
        host_states = {
            name: array.reshape(-1, *array.shape[2:])[selected] for name, array in source.items()
        }

        host = flattened(actor_forward_args(architecture, host_states, device))
        minibatch = flattened(_actor_batch_args(architecture, staged, torch.from_numpy(selected)))

        assert len(host) == len(minibatch) > 0
        for left, right in zip(host, minibatch, strict=True):
            assert left.dtype == right.dtype
            assert torch.equal(left, right)


def test_an_unknown_architecture_is_refused_with_the_known_names() -> None:
    """A `KeyError` naming a dict key is not something a caller can act on.

    Both actor-argument builders fail on an unregistered architecture rather
    than guessing at one, and they name the alternatives the way
    `resolve_architecture` does.
    """
    builders = (
        lambda: actor_forward_args("entity-mlp", {}, torch.device("cpu")),
        lambda: _actor_batch_args("entity-mlp", {}, slice(None)),
    )

    for builder in builders:
        with pytest.raises(ValueError, match="unknown actor architecture 'entity-mlp'") as raised:
            builder()
        assert CONV_ENTITY in str(raised.value)
        assert STRUCTURED in str(raised.value)


def test_credit_diagnostics_distinguish_potential_fit_from_terminal_skill() -> None:
    # Perfect shaped-return fit alone can conceal that most predictable signal
    # is the known current-state potential. Remove it before scoring bank skill.
    valid = np.ones((2, 40), dtype=bool)
    utility = np.array([0.5, -0.5])
    potential = np.broadcast_to(np.linspace(-2.0, 2.0, 40), valid.shape)
    terminal = np.broadcast_to(utility[:, None], valid.shape)
    returns = terminal - potential
    rollout = SimpleNamespace(
        valid=valid,
        final_money=3000.0 * np.expm1(np.array([1.0, 0.5])),
        opponent_money=3000.0 * np.expm1(np.array([0.5, 1.0])),
    )
    baseline = _credit_quality_metrics(
        rollout,
        returns,
        -potential,
        valid,
        1.0,
        {"league": np.array([True, False])},
    )
    perfect = _credit_quality_metrics(rollout, returns, returns, valid, 1.0, None)
    assert baseline[
        "credit_preupdate_all_all_terminal_residual_explained_variance"
    ] == pytest.approx(0.0)
    assert perfect[
        "credit_preupdate_all_all_terminal_residual_explained_variance"
    ] == pytest.approx(1.0)
    assert baseline["credit_preupdate_all_all_potential_only_mse"] == pytest.approx(0.25)
    assert baseline["credit_preupdate_all_all_terminal_residual_mse"] == pytest.approx(0.25)
    assert baseline["credit_preupdate_all_ttg_1_32_states"] == 64
    assert baseline["credit_preupdate_all_ttg_33_128_states"] == 16
    assert baseline["credit_preupdate_league_all_states"] == 40
    assert all(math.isfinite(value) for value in baseline.values())


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auxiliary_gradient_observation_preserves_single_backward_updates(compiled) -> None:
    from kaggriculture.structured import StructuredBelief

    model = torch.nn.Linear(16, 16, bias=False, device="cuda")
    reference = copy.deepcopy(model)
    inputs = torch.linspace(-1, 1, 512, device="cuda").reshape(32, 16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    mode = "default" if compiled else "eager"
    forward = _cached_update_callable(model, "_gradient_probe", model.forward, mode)

    def auxiliary(*fields):
        return sum(field.float().sin().square().mean() for field in fields)

    auxiliary_fn = _cached_update_callable(model, "_auxiliary_probe", auxiliary, mode)
    for observed in (False, True, False, True):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = forward(inputs)
            reference_output = reference(inputs)
        belief = StructuredBelief(*(output[:, None, :] for _ in StructuredBelief._fields))
        squares = []
        source = _observe_auxiliary_gradient(belief, squares) if observed else belief
        auxiliary_loss = auxiliary_fn(*source)
        # Primary gradients deliberately oppose the auxiliary. A hook on shared
        # outputs rather than the auxiliary-only views would include this term.
        primary_loss = -3 * output.float().square().mean()
        (primary_loss + auxiliary_loss).backward()
        reference_auxiliary = auxiliary(
            *(reference_output[:, None, :] for _ in StructuredBelief._fields)
        )
        (-3 * reference_output.float().square().mean() + reference_auxiliary).backward()
        if observed:
            expected = (
                (2 * output.detach().float().sin() * output.detach().float().cos() / output.numel())
                .to(output.dtype)
                .float()
            )
            torch.testing.assert_close(
                torch.stack(squares).sum(),
                expected.square().sum() * len(StructuredBelief._fields),
                rtol=1e-2,
                atol=2e-5,
            )
        torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-2, atol=2e-4)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.weight, reference.weight, rtol=2e-3, atol=5e-5)
