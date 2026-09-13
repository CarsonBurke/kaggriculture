"""Behavioral contracts for balanced source cotangents and full-return GAE."""

import copy

import pytest
import torch

from kaggriculture.model import Linear, policy_compile_options
from kaggriculture.ppo import PpoConfig, _BalancedSource, generalized_advantage_and_targets


def test_default_gae_keeps_delayed_payoffs_independent_of_critic_errors() -> None:
    config = PpoConfig()
    rewards = torch.zeros(1, 719)
    rewards[0, 0] = -0.2
    rewards[0, 240] = 0.6
    rewards[0, -1] = 0.4
    values = torch.linspace(-1.0, 1.0, 719)[None]
    expected = torch.empty_like(rewards)
    running = 0.0
    for step in range(718, -1, -1):
        running = rewards[0, step] + config.gamma * running
        expected[0, step] = running
    for coefficient in (config.actor_gae_lambda, config.critic_gae_lambda):
        advantages, targets = generalized_advantage_and_targets(
            rewards, values, torch.ones_like(rewards), coefficient, config.gamma
        )
        torch.testing.assert_close(targets, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(advantages, expected - values, atol=2e-6, rtol=2e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_source_balance_spans_heads_and_preserves_predictor_gradient() -> None:
    observed = []
    for multiplier in (1.0, 40.0):
        first = torch.tensor([2.0], device="cuda", requires_grad=True)
        second = torch.tensor([3.0], device="cuda", requires_grad=True)
        predictor = torch.tensor(5.0, device="cuda", requires_grad=True)
        main_first, main_second, aux_first, aux_second = _BalancedSource.apply(first, second)
        main = 3 * main_first.sum() + 4 * main_second.sum() - 1000
        auxiliary = multiplier * (12 * aux_first.sum() - 5 * aux_second.sum() + predictor.square())
        (main + auxiliary).backward()
        expected = torch.tensor([1.5, 2.0], device="cuda") + (2.5 / 13) * torch.tensor(
            [12.0, -5.0], device="cuda"
        )
        assert first.grad is not None and second.grad is not None
        actual = torch.cat((first.grad, second.grad))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(predictor.grad, torch.tensor(10 * multiplier, device="cuda"))
        observed.append(actual)
    torch.testing.assert_close(*observed)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("zero_main", [False, True])
def test_source_balance_zero_objective_does_not_create_direction(zero_main: bool) -> None:
    source = torch.tensor([2.0, 3.0], device="cuda", requires_grad=True)
    predictor = torch.tensor(5.0, device="cuda", requires_grad=True)
    primary, auxiliary = _BalancedSource.apply(source)
    main = primary.sum() * (0.0 if zero_main else 1.0)
    auxiliary_loss = auxiliary.square().sum() * (1.0 if zero_main else 0.0) + predictor.square()
    (main + auxiliary_loss).backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 0.0 if zero_main else 1.0))
    torch.testing.assert_close(predictor.grad, torch.tensor(10.0, device="cuda"))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_source_balance_matches_explicit_cotangent_reference() -> None:
    torch.manual_seed(391)
    model = Linear(8, 8).cuda()
    predictor = Linear(8, 8).cuda()
    reference_model, reference_predictor = copy.deepcopy(model), copy.deepcopy(predictor)
    inputs = torch.randn(32, 8, device="cuda")

    def model_terms(inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            primary, auxiliary = _BalancedSource.apply(model(inputs))
            return primary.float().square().mean(), auxiliary

    def auxiliary_terms(belief):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return (predictor(belief).float() - belief.detach().float()).square().mean()

    compiled_model = torch.compile(
        model_terms, options=policy_compile_options("default"), fullgraph=True
    )
    compiled_auxiliary = torch.compile(
        auxiliary_terms, options=policy_compile_options("default"), fullgraph=True
    )
    for _ in range(2):
        for module in (model, predictor, reference_model, reference_predictor):
            module.zero_grad(set_to_none=True)
        main, belief = compiled_model(inputs)
        (main + compiled_auxiliary(belief)).backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference_belief = reference_model(inputs)
            reference_main = reference_belief.float().square().mean()
            source = reference_belief.detach().requires_grad_()
            reference_auxiliary = (
                (reference_predictor(source).float() - source.detach().float()).square().mean()
            )
        primary_gradient = torch.autograd.grad(reference_main, reference_belief, retain_graph=True)[
            0
        ].float()
        reference_auxiliary.backward()
        auxiliary_gradient = source.grad.float()
        scale = 0.5 * primary_gradient.norm() / auxiliary_gradient.norm()
        reference_belief.backward(
            (0.5 * primary_gradient + scale * auxiliary_gradient).to(reference_belief.dtype)
        )
        for actual, reference in zip(model.parameters(), reference_model.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, reference.grad)
        for actual, reference in zip(
            predictor.parameters(), reference_predictor.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, reference.grad)
