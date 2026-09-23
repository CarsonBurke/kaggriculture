"""Compiled BF16 recomputation equivalence; execute only through MLQ CUDA."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from test_entity import config as config
from test_entity import native_batch as native_batch
from test_entity import native_rollout as native_rollout

from kaggriculture.entity import EntityActor, EntityCritic
from kaggriculture.model import policy_compile_options
from kaggriculture.ppo import _value_objective
from kaggriculture.structured import StructuredInputs
from kaggriculture.structured_dynamics import (
    StructuredCriticDynamics,
    structured_critic_window_loss,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required"),
]


def _differentiable_inputs(inputs):
    values = []
    roots = {}
    for name, value in inputs._asdict().items():
        cloned = value.detach().clone()
        if cloned.is_floating_point():
            cloned.requires_grad_()
            roots[f"input.{name}"] = cloned
        values.append(cloned)
    return StructuredInputs(*values), roots


def _compare_gradients(actual, reference):
    """Check every tensor separately so a large head cannot hide a bad small branch."""
    assert actual.keys() == reference.keys()
    live = 0
    for name in actual:
        left, right = actual[name].grad, reference[name].grad
        assert (left is None) == (right is None), name
        if left is None:
            continue
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
        error = torch.linalg.vector_norm(left.float() - right.float())
        magnitude = torch.linalg.vector_norm(right.float())
        if magnitude == 0:
            assert torch.count_nonzero(left) == 0, name
        else:
            live += 1
            # BF16 backward may choose a different legal reduction after
            # recomputation. Bound its relative error per parameter/input.
            assert error <= 0.02 * magnitude + 1e-7, (name, float(error), float(magnitude))
    assert live > 0


def _actor_objective(actor, kinds):
    def objective(inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = actor(inputs)
            quantities = actor.quantity_logits(output.market_quantity_context, kinds)
            outputs = (output.unit_logits, output.market_kind_logits, quantities)
            loss = sum(value.float().square().mean() for value in outputs)
            return loss, outputs

    return torch.compile(
        objective, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )


def test_actor_writeback_recomputation_matches_all_parameter_and_input_gradients(
    config, native_batch
):
    _, inputs, _, factors = native_batch
    torch.manual_seed(20260922)
    actual = EntityActor(replace(config, memory_writeback=True)).cuda().eval()
    reference = copy.deepcopy(actual)
    reference.trunk.rematerialize = False
    assert actual.trunk.rematerialize
    assert actual.state_dict().keys() == reference.state_dict().keys()
    actual_inputs, actual_roots = _differentiable_inputs(inputs)
    reference_inputs, reference_roots = _differentiable_inputs(inputs)
    actual_loss, actual_outputs = _actor_objective(actual, factors["market_kinds"])(actual_inputs)
    actual_loss.backward()
    reference_loss, reference_outputs = _actor_objective(reference, factors["market_kinds"])(
        reference_inputs
    )
    reference_loss.backward()
    torch.testing.assert_close(actual_loss, reference_loss, rtol=0.005, atol=1e-5)
    for left, right in zip(actual_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(left, right, rtol=0.01, atol=0.005)
    _compare_gradients(dict(actual.named_parameters()), dict(reference.named_parameters()))
    _compare_gradients(actual_roots, reference_roots)
    assert actual_roots["input.unit_continuous"].grad is not None
    assert torch.count_nonzero(actual_roots["input.unit_continuous"].grad[~inputs.unit_active]) == 0


def _critic_objective(critic, dynamics, factors):
    def objective(inputs, categorical, continuous, active):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # PPO calls encode_belief directly and branches the same belief
            # into its HL value loss and source-live NextLat predictor.
            belief = critic.encode_belief(inputs, categorical, continuous, active)
            logits = critic.decode_belief(belief)
            targets = torch.linspace(-1, 1, logits.shape[0], device=logits.device)
            primary = _value_objective(critic, logits, targets)
            auxiliary = structured_critic_window_loss(
                dynamics,
                belief,
                inputs,
                factors,
                value_head=critic.value_head,
                horizon=1,
            )
            return primary + auxiliary.latent + auxiliary.value, logits, belief.value_decision

    return torch.compile(
        objective, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )


@pytest.mark.parametrize("flag", ["critic_source_read", "memory_writeback"])
def test_critic_recomputation_matches_joint_value_nextlat_gradients(config, native_batch, flag):
    _, _, critic_args, factors = native_batch
    torch.manual_seed(20260923)
    configured = replace(config, **({"critic_source_read": False} | {flag: True}))
    actual = EntityCritic(configured).cuda()
    with torch.no_grad():
        actual.value_head.weight.normal_(std=0.02)
    reference = copy.deepcopy(actual)
    reference.trunk.rematerialize = False
    assert actual.trunk.rematerialize
    actual_dynamics = StructuredCriticDynamics(configured).cuda()
    reference_dynamics = copy.deepcopy(actual_dynamics)
    actual_inputs, actual_roots = _differentiable_inputs(critic_args[0])
    reference_inputs, reference_roots = _differentiable_inputs(critic_args[0])
    actual_private = critic_args[2].detach().clone().requires_grad_()
    reference_private = critic_args[2].detach().clone().requires_grad_()
    actual_roots["input.opponent_units"] = actual_private
    reference_roots["input.opponent_units"] = reference_private
    actual_loss, actual_logits, actual_belief = _critic_objective(actual, actual_dynamics, factors)(
        actual_inputs, critic_args[1], actual_private, critic_args[3]
    )
    actual_loss.backward()
    reference_loss, reference_logits, reference_belief = _critic_objective(
        reference, reference_dynamics, factors
    )(reference_inputs, critic_args[1], reference_private, critic_args[3])
    reference_loss.backward()
    torch.testing.assert_close(actual_loss, reference_loss, rtol=0.005, atol=1e-5)
    torch.testing.assert_close(actual_logits, reference_logits, rtol=0.01, atol=0.005)
    torch.testing.assert_close(actual_belief, reference_belief, rtol=0.01, atol=0.005)
    _compare_gradients(dict(actual.named_parameters()), dict(reference.named_parameters()))
    _compare_gradients(
        dict(actual_dynamics.named_parameters()), dict(reference_dynamics.named_parameters())
    )
    _compare_gradients(actual_roots, reference_roots)
    assert actual_private.grad[critic_args[3]].abs().sum() > 0
    assert torch.count_nonzero(actual_private.grad[~critic_args[3]]) == 0
