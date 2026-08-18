from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import kaggriculture.ppo
from kaggriculture import telemetry
from kaggriculture.actions import MarketKind
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.ppo import (
    DEFAULT_ACTOR_GAE_LAMBDA,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    UNCOMPILED_UPDATE_COMPILE_MODE,
    UPDATE_REPLAY_TAIL_LOGPROB,
    PpoConfig,
    _balanced_minibatch_slices,
    _clipped_surrogate_sums,
    _epoch_value_losses,
    _explained_variance,
    _fit_explained_variance,
    _stage_tensor,
    _target_correlation,
    _validate_config,
    _validate_staged_action_masks,
    generalized_advantage_and_targets,
    make_optimizers,
    prepare_advantages,
    replay_behavior_values,
    update_ppo,
    update_replay_parity,
)
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.rollout import collect_self_play
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic


def test_entropy_bonus_exists_and_ships_disabled_until_measured() -> None:
    """DAPO drops the entropy bonus for Clip-Higher; this pipeline needs both.

    Clip-Higher only permits a *sampled* low-probability action's probability to
    grow, so it preserves exploration rather than restoring it. That presupposes
    DAPO's setting -- RL from a pretrained model whose policy is still diffuse.
    This actor is warm-started from behavior cloning and measured at 0.1395 nats
    per active component on the first actor-active iteration, when 40 iterations
    of critic warmup had left it byte-identical to the clone. The collapse is
    therefore inherited, not caused by the update, and Clip-Higher cannot act on
    actions that are never drawn.

    The coefficient nonetheless ships at zero until a run measures it, so this
    pins the mechanism's existence and its inert default separately.
    """
    assert "entropy_coefficient" in asdict(PpoConfig())
    assert PpoConfig().entropy_coefficient == 0.0


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


def test_default_gae_matches_the_length_adaptive_value_for_the_fixed_horizon() -> None:
    config = PpoConfig()

    assert config.gamma == 1.0
    assert config.actor_gae_lambda == pytest.approx(1.0 - 1.0 / (0.05 * 719.0))
    assert config.actor_gae_lambda == pytest.approx(699.0 / 719.0)
    assert config.actor_gae_lambda == DEFAULT_ACTOR_GAE_LAMBDA
    assert 1.0 / (1.0 - config.actor_gae_lambda) == pytest.approx(0.05 * 719.0)


def test_discounted_bank_delta_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="undiscounted gamma=1"):
        _validate_config(PpoConfig(gamma=0.99))


def test_minibatches_are_balanced_without_dropping_the_tail() -> None:
    slices = _balanced_minibatch_slices(230_080, 2048)
    sizes = [row.stop - row.start for row in slices]

    assert len(slices) == 113
    assert sum(sizes) == 230_080
    assert max(sizes) <= 2048
    assert max(sizes) - min(sizes) <= 1
    assert slices[0].start == 0
    assert slices[-1].stop == 230_080


def test_the_critic_target_is_the_lambda_return_the_actor_advantage_came_from() -> None:
    """Standard PPO: the critic regresses on `advantage + value`.

    Only the terminal target is the Monte Carlo return, because nothing is left
    to bootstrap from. Every earlier one is pulled toward the value function by
    the same lambda that truncates the actor's advantage, which is the whole
    point of bootstrapping a dense reward: the target for a state 719 steps from
    the end stops carrying 719 steps of sampling noise.
    """
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.2, -0.1, 0.5]])
    valid = torch.ones_like(rewards)

    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, actor_gae_lambda=0.5
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.125, 0.85, 0.5]]))
    torch.testing.assert_close(targets, torch.tensor([[0.325, 0.75, 1.0]]))
    # The lambda-one Monte Carlo suffix return the critic used to fit.
    assert not torch.allclose(targets, torch.ones_like(targets))


def test_an_exact_critic_makes_the_target_the_dense_bank_delta_at_any_lambda() -> None:
    """Bootstrapping costs nothing where the value function is already right.

    The reward is a potential difference, so the exact value of a state is the
    remaining potential delta. Feed that in and every temporal-difference
    residual vanishes: the advantage is exactly zero and the target is exactly
    the Monte Carlo return, for any lambda. The bias the lambda introduces is
    therefore entirely the critic's own error, not a property of the target.
    """
    potentials = torch.tensor([[0.2, -0.1, 0.4, 0.3, 0.6]])
    rewards = potentials[:, 1:] - potentials[:, :-1]
    valid = torch.ones_like(rewards)
    exact = potentials[:, -1:] - potentials[:, :-1]

    for actor_gae_lambda in (0.0, 0.5, 1.0):
        advantages, targets = generalized_advantage_and_targets(
            rewards, exact, valid, actor_gae_lambda=actor_gae_lambda
        )

        torch.testing.assert_close(advantages, torch.zeros_like(advantages))
        torch.testing.assert_close(targets, exact)

    # And that the agreement is a property of the critic being right, not of the
    # target ignoring it: displace the critic and the target moves with it,
    # which the Monte Carlo suffix return it replaced would not have done.
    displaced = generalized_advantage_and_targets(
        rewards, exact + 0.5, valid, actor_gae_lambda=0.5
    )[1]
    assert not torch.allclose(displaced, exact)


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
    torch.testing.assert_close(targets, expected + values)


def test_discounted_advantages_and_targets_match_the_reference_recurrence() -> None:
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
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(targets, expected_advantages + values)
    # Discounting a potential difference would price holding cash early, which
    # is why production fixes gamma at one; the recurrence still has to be right
    # for the general case it is written for.
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
        targets = generalized_advantage_and_targets(rewards, values, valid, actor_gae_lambda=1.0)[1]

        torch.testing.assert_close(targets, monte_carlo)

    shortened = generalized_advantage_and_targets(
        rewards,
        torch.tensor([[10.0, -7.0, 3.0], [4.0, 1.0, -8.0]]),
        valid,
        actor_gae_lambda=0.99,
    )[1]
    assert not torch.allclose(shortened, monte_carlo)


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
        rewards, values, valid, actor_gae_lambda=0.5
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
    config = PpoConfig(epochs=1, minibatch_size=1 << 12, target_kl=1e-6, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    metrics = update_ppo(
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


def _stale_behavior_replay(monkeypatch) -> None:
    """Put the behavior policy far from the actor at the interface the update
    reads it from, so every minibatch's k3 divergence is ~0.72 nats."""
    genuine_replay = kaggriculture.ppo.replay_behavior_logprobs

    def stale_replay(*args, **kwargs):
        replayed = genuine_replay(*args, **kwargs)
        return {name: values - 1.0 for name, values in replayed.items()}

    monkeypatch.setattr(kaggriculture.ppo, "replay_behavior_logprobs", stale_replay)


def _small_update_models():
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    return FarmActor(model_config), DistributionalCritic(model_config)


def test_the_first_minibatch_is_exempt_from_the_trust_region(monkeypatch) -> None:
    """At unchanged weights the first minibatch's divergence is numerical
    residual between the replay's graph and the update's, not policy movement.
    Feeding it to the trust region reads rounding as staleness and can stop the
    actor before it takes a single step -- the audit's worst recorded draw over
    339 minibatches was 4.060e-2, above the production trust region of 0.03.
    Its own gate is MAX_FIRST_MINIBATCH_KL, which `train_ppo` raises on.
    """
    actor, critic = _small_update_models()
    rollout = collect_self_play(actor, games=1, seed_start=93, episode_steps=3, sampling_seed=8)
    _stale_behavior_replay(monkeypatch)
    config = PpoConfig(epochs=1, minibatch_size=1 << 12, target_kl=1e-4, use_bfloat16=False)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}

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
    assert metrics["first_minibatch_approx_kl"] > config.target_kl
    assert metrics["actor_updates"] == 1
    assert metrics["kl_early_stop"] == 0
    # The exempt minibatch is excluded from the statistic paired with the bound,
    # so with no later actor minibatch this stays at its initial value.
    assert metrics["max_approx_kl"] == 0.0
    assert any(
        not torch.equal(parameter, before[name]) for name, parameter in actor.named_parameters()
    )


def test_over_target_kl_stops_the_actor_after_the_first_minibatch(monkeypatch) -> None:
    """Past the exempt first minibatch the trust region is enforced before the
    policy is mutated, and the stop latches for the rest of the update while the
    critic keeps refitting."""
    actor, critic = _small_update_models()
    rollout = collect_self_play(actor, games=1, seed_start=93, episode_steps=3, sampling_seed=8)
    _stale_behavior_replay(monkeypatch)
    config = PpoConfig(
        epochs=2,
        minibatch_size=8,
        target_kl=1e-4,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
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
    # Two actor minibatches ran; only the exempt first one was applied, which is
    # the proof the violating one was skipped rather than merely counted.
    assert metrics["actor_updates"] == 1
    assert metrics["kl_early_stop"] == 1
    assert metrics["max_approx_kl"] > config.target_kl
    assert any(
        not torch.equal(parameter, critic_before[name])
        for name, parameter in critic.named_parameters()
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
    assert "critic_gae_lambda" not in metrics
    assert metrics["gamma"] == config.gamma
    # The critic target is the lambda-return `advantage + value` at every valid
    # state, so the three means the update reports are the same identity read
    # off the whole batch. The Monte Carlo target it replaced satisfied no such
    # relation, because it never saw the values at all.
    assert metrics["value_target_std"] > 0.05
    assert metrics["value_target_mean"] == pytest.approx(
        metrics["advantage_mean"] + metrics["value_prediction_mean"], abs=1e-6
    )
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


def test_a_critic_only_refit_aborts_on_a_non_finite_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A poisoned critic loss must still stop the run when no actor is present.

    Critic-only minibatches no longer read their loss back to the host every
    step, so the abort has moved. On an unfused optimizer it stays immediate; on
    a fused one the step is skipped on the device and the raise lands at the
    epoch boundary. Either way the run must not continue, and the critic must
    never have absorbed the poisoned gradient.
    """
    model_config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=90, episode_steps=8, sampling_seed=3)
    config = PpoConfig(epochs=1, critic_epochs=2, minibatch_size=8, use_bfloat16=False)
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

    for name, value in critic.named_parameters():
        assert torch.equal(before[name], value.detach()), name


def test_a_return_past_the_outermost_atom_saturates_and_is_reported() -> None:
    """A bootstrapped target has no bound the support can be sized against.

    The Monte Carlo target was a potential difference and could not leave
    [-2, 2], so a target outside the support meant a bug and was raised on. The
    lambda-return adds the critic's own prediction to that, and the critic is
    only bounded by the same support, so no width contains it by construction.
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
    # A reward far outside the potential difference the environment can pay, so
    # the return leaves the support no matter what the critic predicts.
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


def test_entropy_is_the_only_gradient_when_policy_advantage_is_zero(monkeypatch) -> None:
    """Zero advantages silence the surrogate, isolating the entropy term.

    With every advantage exactly zero the clipped surrogate has no gradient, so
    whatever the actor does next is attributable to the entropy bonus alone.
    That makes this the sharp test of the coefficient in both directions: at
    zero it must leave the actor bit-identical, and above zero it must move the
    policy toward higher entropy on the very states just measured.
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=94, episode_steps=3, sampling_seed=10)
    rollout.rewards.fill(0.0)
    # Zero rewards alone leave value-driven GAE deltas; zero replayed values
    # too so every advantage is exactly zero.
    monkeypatch.setattr(
        kaggriculture.ppo,
        "replay_behavior_values",
        lambda critic, architecture, staged, **kwargs: torch.zeros(staged["unit_actions"].shape[0]),
    )

    def run(coefficient: float, actor: FarmActor, critic: DistributionalCritic) -> dict:
        config = PpoConfig(
            epochs=1,
            minibatch_size=rollout.state_count,
            lr_warmup_steps=0,
            weight_decay=0.0,
            use_bfloat16=False,
            entropy_coefficient=coefficient,
            # Far above the shipped rate so one normalized step is unambiguous;
            # `max_gradient_norm` makes the applied step `lr * g / ||g||`.
            actor_learning_rate=1.0e-2,
        )
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
        return update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(11),
        )

    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    disabled = run(0.0, actor, critic)
    assert disabled["actor_updates"] == 1
    assert disabled["policy_loss"] == pytest.approx(0.0, abs=1e-12)
    assert disabled["entropy"] > 0.0
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)

    # A fresh pair, so the enabled run starts from the same weights the disabled
    # run left untouched rather than from its own first step.
    enabled_actor = FarmActor(model_config)
    enabled_actor.load_state_dict(actor.state_dict())
    enabled_critic = DistributionalCritic(model_config)
    enabled_critic.load_state_dict(critic.state_dict())
    first = run(1.0, enabled_actor, enabled_critic)
    second = run(1.0, enabled_actor, enabled_critic)

    # The surrogate stays silent throughout, so the reported entropy rising over
    # the identical states is the bonus doing the only work there is to do.
    assert first["policy_loss"] == pytest.approx(0.0, abs=1e-12)
    assert second["entropy"] > first["entropy"]
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in enabled_actor.named_parameters()
    )


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


def test_a_mis_scaled_critic_still_explains_its_own_bootstrapped_target() -> None:
    """Why the Monte Carlo reading is kept after the target stopped being it.

    A critic predicting three times the true return is badly wrong, and the
    explained variance against the suffix return says so: exactly -3. But the
    bootstrapped target is built from that same prediction, so its residual is
    only the advantage -- one lambda-window of reward against a full trajectory
    of value -- and the same critic scores near one against it. Reporting only
    the bootstrapped reading would have retired the measurement that diagnosed
    this critic in the first place.
    """
    # A relative bank drifting a little each step over a horizon long against
    # the lambda window, which is the shape of the real 719-step episode: one
    # transition moves the potential far less than the rest of the game does.
    generator = np.random.default_rng(5)
    potentials = np.cumsum(generator.normal(0.0, 0.05, size=(4, 65)), axis=1).astype(np.float32)
    rewards = np.diff(potentials, axis=1)
    valid = np.ones_like(rewards, dtype=np.bool_)
    monte_carlo = potentials[:, -1:] - potentials[:, :-1]
    mis_scaled = (3.0 * monte_carlo).astype(np.float32)

    prepared = prepare_advantages(
        SimpleNamespace(rewards=rewards, valid=valid),
        mis_scaled,
        PpoConfig(actor_gae_lambda=0.5),
    )

    np.testing.assert_allclose(prepared.monte_carlo_returns, monte_carlo, atol=1e-6)
    assert _explained_variance(prepared.monte_carlo_returns, mis_scaled, valid) == pytest.approx(
        -3.0
    )
    assert _explained_variance(prepared.value_targets, mis_scaled, valid) > 0.8


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
    config = PpoConfig(
        epochs=1,
        critic_epochs=24,
        minibatch_size=32,
        use_bfloat16=False,
        lr_warmup_steps=0,
        critic_learning_rate=3e-3,
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
    # Both standard deviations are population ones, so the identity is exact up
    # to the float32 the advantages and targets are held in.
    identity = 1.0 - (metrics["advantage_std"] / metrics["value_target_std"]) ** 2
    assert metrics["lambda_return_explained_variance"] == pytest.approx(identity, abs=1e-5)

    # The fit reading is materially positive on a critic the conventional reading
    # calls near-worthless, and their SEPARATION is what the pair exists for.
    # Swept over twelve initializations, the absolute levels do not support a
    # fixed threshold -- the last-epoch fit spans 0.111 to 0.412 and the identity
    # 0.018 to 0.096, so the two ranges overlap and either bound can be a hair
    # from firing. The ratio is the stable statistic: 3.8x at its tightest,
    # 16.8x at its widest, so a factor of three separates them on every init
    # sampled while still failing if the fit reading collapses onto the identity.
    assert metrics["critic_fit_explained_variance_last_epoch"] > 0.1
    assert abs(metrics["lambda_return_explained_variance"]) < 0.1
    assert metrics["critic_fit_explained_variance_last_epoch"] > 3.0 * abs(
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


def test_advantage_statistics_describe_the_rollout_not_the_normalizer() -> None:
    """`advantage_mean`/`advantage_std` must not report their own normalizer.

    `prepare_advantages` returns a zero-mean unit-variance array, so measuring
    the returned array reports 0 and 1 at every iteration regardless of what the
    rollout contained -- two charts that are constants by construction. The raw
    scale is the useful one: it is the size of the advantage signal, and it
    shrinks as the critic starts explaining the return.
    """
    rewards = np.zeros((2, 4), dtype=np.float32)
    rewards[:, -1] = [6.0, -6.0]
    rollout = SimpleNamespace(
        rewards=rewards,
        valid=np.ones((2, 4), dtype=np.bool_),
    )
    values = np.full((2, 4), 2.0, dtype=np.float32)

    prepared = prepare_advantages(rollout, values, PpoConfig())

    normalized = prepared.advantages[rollout.valid]
    assert normalized.mean() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std() == pytest.approx(1.0, abs=1e-6)

    raw, _targets = generalized_advantage_and_targets(
        torch.from_numpy(rewards),
        torch.from_numpy(values),
        torch.ones_like(torch.from_numpy(rewards)),
        actor_gae_lambda=PpoConfig().actor_gae_lambda,
        gamma=PpoConfig().gamma,
    )
    selected = raw.reshape(-1)
    assert prepared.raw_advantage_mean == pytest.approx(float(selected.mean()), rel=1e-6)
    assert prepared.raw_advantage_std == pytest.approx(
        float(selected.std(unbiased=False)), rel=1e-6
    )
    # The raw scale is nowhere near one, which is why reporting the normalized
    # array instead threw the measurement away.
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


def test_the_audited_first_minibatch_kl_is_the_replay_to_update_residual() -> None:
    """The gate's statistic is replay-vs-update, not sampling-vs-update.

    `update_ppo` overwrites the rollout's sampling likelihoods with a replay
    before the minibatch loop, so its ratio starts at one by construction and
    the residual it gates is only compiled-graph and batch-composition noise.
    Auditing a sampling-path number against that bound compares two different
    quantities, which on a cloned actor happen to sit at a similar magnitude --
    the coincidence that made the confusion survive.
    """
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    rollout = collect_self_play(actor, games=2, seed_start=41, episode_steps=4, sampling_seed=7)

    metrics = update_replay_parity(
        actor,
        rollout,
        minibatch_size=8,
        compile_mode=UNCOMPILED_UPDATE_COMPILE_MODE,
        autocast_enabled=False,
    )

    sampling = metrics["update_replay_minibatch_kl"]
    replay = metrics["update_replay_first_minibatch_kl"]
    # k3 is non-negative, so a bound is one-sided and zero is the floor.
    assert replay >= 0.0
    # In eager float32 both sides are deterministic and agree to rounding, while
    # the sampling path went through a different forward entirely. The gap is
    # the whole point: these are not interchangeable measurements.
    assert replay < 1e-9
    assert sampling > replay
