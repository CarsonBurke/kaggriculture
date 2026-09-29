"""Shared-plan likelihood algebra and replay metadata; no model forward on CPU."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import kaggriculture.ppo as ppo
from kaggriculture.strategic_actor import PlanChoice


def _bc():
    path = Path(__file__).parents[1] / "scripts" / "train_bc.py"
    spec = importlib.util.spec_from_file_location("strategic_bc_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_plan_bc_marginalizes_joint_actions_and_reaches_plan_prior():
    bc = _bc()
    prior_logits = torch.tensor([[0.2, -0.2]], requires_grad=True)
    log_prior = prior_logits.log_softmax(-1)
    # A single plan must explain BOTH observed actions; independently mixing
    # their probabilities would assign incompatible plans to the same turn.
    selected = torch.tensor([[0.9, 0.8], [0.2, 0.1]]).log().requires_grad_()
    active = torch.ones(1, 2, dtype=torch.bool)
    loss = bc._marginal_plan_nll(log_prior, (selected,), (active,))
    expected = -(prior_logits.softmax(-1) * torch.tensor([[0.72, 0.02]])).sum().log() / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prior_logits.grad[0, 0] < 0 < prior_logits.grad[0, 1]
    assert selected.grad.isfinite().all()
    assert (selected.grad < 0).all()


def test_bc_prefix_statistics_match_exact_turn_marginal_with_inactive_factors():
    bc = _bc()
    prior = torch.tensor([[0.6, 0.4], [0.3, 0.7]]).log()
    # [batch * plan, slot, class], with unequal plan distributions.
    units = torch.tensor([[[0.9, 0.1]], [[0.2, 0.8]], [[0.4, 0.6]], [[0.7, 0.3]]]).log()
    kinds = torch.tensor([[[0.8, 0.2]], [[0.3, 0.7]], [[0.9, 0.1]], [[0.2, 0.8]]]).log()
    quantities = torch.tensor([[[0.7, 0.3]], [[0.1, 0.9]], [[0.8, 0.2]], [[0.3, 0.7]]]).log()
    logits = (units, kinds, quantities)
    masks = tuple(torch.ones_like(value, dtype=torch.bool) for value in logits)
    targets = (torch.tensor([[0], [1]]), torch.tensor([[1], [0]]), torch.tensor([[0], [1]]))
    active = (
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([[True], [False]]),
    )
    statistics = bc._plan_prefix_statistics(prior, logits, masks, targets, active)
    # Exhaustive sum over the same plan shared by every active physical factor.
    selected = tuple(
        head.gather(-1, target[:, None].expand(-1, 2, -1).reshape(4, 1, 1)).squeeze(-1)
        for head, target in zip(logits, targets, strict=True)
    )
    expected = bc._marginal_plan_nll(prior, selected, active)
    observed = -sum((stats[0] * mask).sum() for stats, mask in zip(statistics, active, strict=True))
    observed /= sum(mask.sum() for mask in active)
    torch.testing.assert_close(observed, expected)
    # Kind marginal after seeing unit=0 has P(plan=0)=0.54/(0.54+0.08).
    expected_kind = (0.54 * 0.2 + 0.08 * 0.7) / 0.62
    assert statistics[1][0][0, 0].exp() == pytest.approx(expected_kind)
    for _, entropy, distribution in statistics:
        assert entropy.isfinite().all()
        torch.testing.assert_close(distribution.exp().sum(-1), torch.ones(2, 1))


def test_plan_is_a_joint_ppo_factor_with_masked_entropy_and_gradient(monkeypatch):
    physical = torch.tensor([[math.log(1.1)], [0.0], [float("nan")]], requires_grad=True)
    plan = torch.tensor([[math.log(1.1)], [math.log(1.4)], [float("nan")]], requires_grad=True)
    inactive = torch.zeros(3, 1)
    ignored = torch.full_like(inactive, float("nan"))
    monkeypatch.setattr(
        ppo,
        "_replayed_component_logprobs",
        lambda *args: (
            physical,
            ignored,
            ignored,
            plan,
            torch.full_like(physical, 0.5),
            ignored,
            ignored,
            torch.full_like(plan, 0.7),
        ),
    )
    choice = PlanChoice(
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3),
        torch.tensor([True, False, True]),
    )
    arguments = (
        None,
        physical,
        inactive,
        inactive,
        inactive,
        inactive,
        inactive,
        torch.ones_like(physical),
        inactive,
        inactive,
        torch.zeros_like(physical),
        ignored,
        ignored,
        torch.ones(3),
        0.8,
        1.28,
        False,
        None,
        choice,
    )
    terms = ppo._actor_minibatch_terms(
        *arguments, sample_weight=torch.tensor([1.0, 1.0, 0.0]), policy_ratio_scope="joint"
    )
    torch.testing.assert_close(terms[0], torch.tensor(1.21 + 1.0))
    torch.testing.assert_close(terms[1], torch.tensor(0.5 + 0.5 + 0.7))
    terms[0].backward()
    torch.testing.assert_close(plan.grad[:2], torch.tensor([[1.21], [0.0]]))
    assert plan.grad[2] == 0
    with pytest.raises(ValueError, match="joint policy ratios"):
        ppo._actor_minibatch_terms(*arguments)


def test_the_entropy_bonus_reaches_the_plan_factor_over_active_states_only(monkeypatch):
    physical = torch.tensor([[math.log(1.1)], [0.0], [float("nan")]])
    plan = torch.tensor([[math.log(1.1)], [math.log(1.4)], [float("nan")]])
    physical_entropy = torch.full_like(physical, 0.5, requires_grad=True)
    plan_entropy = torch.full_like(plan, 0.7, requires_grad=True)
    inactive = torch.zeros(3, 1)
    ignored = torch.full_like(inactive, float("nan"))
    monkeypatch.setattr(
        ppo,
        "_replayed_component_logprobs",
        lambda *args: (
            physical,
            ignored,
            ignored,
            plan,
            physical_entropy,
            ignored,
            ignored,
            plan_entropy,
        ),
    )
    choice = PlanChoice(
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3),
        torch.tensor([True, False, True]),
    )
    terms = ppo._actor_minibatch_terms(
        None,
        physical,
        inactive,
        inactive,
        inactive,
        inactive,
        inactive,
        torch.ones_like(physical),
        inactive,
        inactive,
        torch.zeros_like(physical),
        ignored,
        ignored,
        torch.ones(3),
        0.8,
        1.28,
        False,
        None,
        choice,
        sample_weight=torch.tensor([1.0, 1.0, 0.0]),
        policy_ratio_scope="joint",
        entropy_gradient=True,
    )
    torch.testing.assert_close(terms[1], torch.tensor(0.5 + 0.5 + 0.7))
    terms[1].backward()
    # The plan is active on the first state only; the third is wrapped padding.
    torch.testing.assert_close(plan_entropy.grad, torch.tensor([[1.0], [0.0], [0.0]]))
    torch.testing.assert_close(physical_entropy.grad, torch.tensor([[1.0], [1.0], [0.0]]))


def test_plan_replay_args_use_recorded_indices_and_critic_ignores_plan(monkeypatch):
    # The typed argument boundary is independent of actual observation values.
    fields = {
        name: torch.zeros(3, 1, dtype=dtype)
        for name, dtype in ppo._actor_forward_fields("strategic-plan")
    }
    fields |= {
        "plan_indices": torch.tensor([2, 0, 1], dtype=torch.int8),
        "old_plan_logprobs": torch.tensor([-0.2, -0.3, -0.4]),
        "plan_active": torch.tensor([True, True, False]),
    }
    indices = torch.tensor([2, 0])
    args = ppo._actor_batch_args("strategic-plan", fields, indices)
    choice = args[1]
    torch.testing.assert_close(choice.indices, torch.tensor([1, 2]))
    torch.testing.assert_close(choice.old_logprobs, torch.tensor([-0.4, -0.2]))
    torch.testing.assert_close(choice.active, torch.tensor([False, True]))
    assert (
        len(
            ppo._actor_batch_args(
                "strategic-plan",
                {k: v for k, v in fields.items() if not k.startswith(("plan_", "old_plan"))},
                indices,
            )
        )
        == 1
    )
    for name in ("products", "animals", "crops"):
        fields[f"critic_{name}"] = torch.zeros(3, 1)
    fields["opponent_unit_categorical"] = torch.zeros(3, 1, dtype=torch.long)
    fields["opponent_unit_continuous"] = torch.zeros(3, 1)
    fields["opponent_unit_active"] = torch.zeros(3, 1, dtype=torch.bool)
    critic_args = ppo._critic_batch_args("strategic-plan", fields, indices, actor_args=args)
    assert len(critic_args) == 4
    assert all(not isinstance(argument, PlanChoice) for argument in critic_args)


def test_plan_likelihood_is_in_every_replay_audit_and_saved_replay(monkeypatch):
    horizon = 3
    rollout = SimpleNamespace(
        architecture="strategic-plan",
        valid=np.ones((1, horizon), dtype=np.bool_),
        states={
            "plan_indices": np.zeros((1, horizon), dtype=np.int8),
            "old_plan_logprobs": np.full((1, horizon), -1.0, dtype=np.float32),
            "plan_active": np.ones((1, horizon), dtype=np.bool_),
        },
    )
    for name in ("unit_actions", "market_kinds", "market_quantities"):
        setattr(rollout, name, np.zeros((1, horizon, 1), dtype=np.int8))
    for name in ("unit_masks", "market_kind_masks", "market_quantity_masks"):
        setattr(rollout, name, np.ones((1, horizon, 1, 2), dtype=np.bool_))
    for name in ("unit_active", "market_active", "market_quantity_active"):
        setattr(rollout, name, np.ones((1, horizon, 1), dtype=np.bool_))
    for name in ("old_unit_logprobs", "old_market_kind_logprobs", "old_market_quantity_logprobs"):
        setattr(rollout, name, np.full((1, horizon, 1), -1.0, dtype=np.float32))
    # Every actor carries its model configuration; the replay reads its action
    # interface to choose the market-set path, which the plan actor never takes.
    actor = SimpleNamespace(
        parameters=lambda: iter((torch.zeros(1),)), config=SimpleNamespace(action_interface=1)
    )

    def actor_args(architecture, staged, indices):
        old = staged["old_plan_logprobs"][indices]
        return (
            None,
            PlanChoice(
                staged["plan_indices"][indices],
                torch.zeros_like(old),
                torch.ones_like(old),
                torch.zeros_like(old, dtype=torch.bool),
                old,
                staged["plan_active"][indices],
            ),
        )

    def replay(*args):
        choice = args[-1]
        fixed = torch.full((choice.indices.numel(), 1), -1.0)
        return fixed, fixed, fixed, choice.old_logprobs[:, None] + 0.3

    def replay_with_entropy(*args):
        result = replay(*args)
        return (*result, *(torch.ones_like(part) for part in result))

    monkeypatch.setattr(ppo, "_actor_batch_args", actor_args)
    monkeypatch.setattr(ppo, "_replayed_selected_logprobs", replay)
    monkeypatch.setattr(ppo, "_replayed_component_logprobs", replay_with_entropy)
    metrics = ppo.update_replay_parity(
        actor, rollout, minibatch_size=2, compile_mode="eager", autocast_enabled=False
    )
    expected_kl = math.expm1(0.3) - 0.3
    assert metrics["update_replay_plan_active_count"] == horizon
    assert metrics["update_replay_plan_kl"] == pytest.approx(expected_kl, rel=1e-6)
    assert metrics["update_replay_joint_kl"] == pytest.approx(expected_kl, rel=1e-6)
    assert metrics["update_replay_max_kl"] == pytest.approx(expected_kl, rel=1e-6)
    assert metrics["update_replay_first_minibatch_kl"] == pytest.approx(expected_kl / 4, rel=1e-5)
    assert metrics["update_replay_mean_minibatch_kl"] == pytest.approx(expected_kl / 4, rel=1e-5)
    staged = {
        name: ppo._stage_tensor(array, torch.device("cpu"))
        for name, array in {
            **rollout.states,
            **{
                name: value
                for name, value in vars(rollout).items()
                if isinstance(value, np.ndarray) and name != "valid"
            },
        }.items()
    }
    replayed = ppo.replay_behavior_logprobs(
        actor,
        rollout.architecture,
        staged,
        np.arange(horizon),
        minibatch_size=2,
        autocast_enabled=False,
    )
    torch.testing.assert_close(replayed["old_plan_logprobs"], torch.full((horizon,), -0.7))
