"""Compiled BF16 shared-plan actor/BC contracts; execute only through MLQ CUDA."""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch
from test_entity import config as config
from test_entity import native_batch as native_batch
from test_entity import native_rollout as native_rollout

from kaggriculture.model import policy_compile_options
from kaggriculture.ppo import _actor_minibatch_terms, _replayed_selected_logprobs
from kaggriculture.provenance import repository_root
from kaggriculture.strategic_actor import PlanChoice, StrategicActor, StrategicConfig

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required"),
]


def _compiled(function):
    return torch.compile(
        function, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )


def _choice(batch, plans):
    return PlanChoice(
        torch.arange(batch, device="cuda") % plans,
        torch.zeros(batch, device="cuda"),
        torch.ones(batch, device="cuda"),
        torch.zeros(batch, device="cuda", dtype=torch.bool),
        torch.zeros(batch, device="cuda"),
        torch.ones(batch, device="cuda", dtype=torch.bool),
    )


def _assert_strategic_gradients(actor):
    for name in ("trunk.plan_head", "trunk.plans", "trunk.workspace", "trunk.encoder.tiles"):
        parameters = tuple(actor.get_submodule(name).parameters())
        gradients = [parameter.grad for parameter in parameters]
        assert all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
        ), name
        assert sum(gradient.float().abs().sum() for gradient in gradients) > 0, name


def test_compiled_all_plans_matches_conditioned_decode(native_batch):
    _, inputs, _, _ = native_batch
    actor = StrategicActor(StrategicConfig()).cuda().eval()
    choice = _choice(inputs.unit_active.shape[0], actor.config.plan_count)

    def compare(inputs, choice):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            heads, prior = actor.all_plans(inputs)
            selected = actor(inputs, choice)
        return heads, prior, selected

    with torch.no_grad():
        heads, prior, selected = _compiled(compare)(inputs, choice)
    indices = torch.arange(choice.indices.numel(), device="cuda") * actor.config.plan_count
    indices += choice.indices
    for all_values, selected_values in zip(heads, selected[:3], strict=True):
        torch.testing.assert_close(all_values[indices], selected_values, atol=0.03, rtol=0.03)
    torch.testing.assert_close(
        prior.gather(-1, choice.indices[:, None]).squeeze(-1),
        selected.plan[:, 1],
        atol=0.005,
        rtol=0.005,
    )


def test_compiled_joint_ppo_loss_reaches_plan_prior_embedding_and_workspace(native_batch):
    _, inputs, _, factors = native_batch
    torch.manual_seed(20260927)
    actor = StrategicActor(StrategicConfig()).cuda().eval()
    batch = inputs.unit_active.shape[0]
    choice = _choice(batch, actor.config.plan_count)
    physical = tuple(
        factors[name]
        for name in (
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "unit_masks",
            "market_kind_masks",
            "market_quantity_masks",
        )
    )
    with torch.no_grad():
        old = _compiled(_replayed_selected_logprobs)(actor, *physical, True, inputs, choice)
    choice = choice._replace(old_logprobs=old[3].squeeze(-1))
    active = tuple(
        factors[name].float()
        for name in (
            "unit_active",
            "market_active",
            "market_quantity_active",
        )
    )
    advantages = torch.linspace(-1, 1, batch, device="cuda")
    result = _compiled(_actor_minibatch_terms)(
        actor,
        *physical,
        *active,
        *old[:3],
        advantages,
        0.8,
        1.28,
        True,
        inputs,
        choice,
        policy_ratio_scope="joint",
    )
    assert all(torch.isfinite(term) for term in result)
    (-result[0] / batch).backward()
    _assert_strategic_gradients(actor)


def test_compiled_exact_bc_marginal_reaches_plan_prior_embedding_and_workspace(native_batch):
    _, inputs, _, factors = native_batch
    path = repository_root() / "scripts" / "train_bc.py"
    spec = importlib.util.spec_from_file_location("strategic_bc_gradient_test", path)
    bc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bc
    spec.loader.exec_module(bc)
    torch.manual_seed(20260928)
    actor = StrategicActor(StrategicConfig()).cuda().eval()
    loss = _compiled(bc._clone_loss)(actor, (inputs,), factors, True)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    _assert_strategic_gradients(actor)
