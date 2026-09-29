"""PPO importance ratios for the effective market-set action interface."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from kaggriculture.model import ActorOutput
from kaggriculture.ppo import (
    _market_set_minibatch_terms,
    _market_set_selected_logprobs,
    _policy_factor_batch_args,
    _validate_staged_action_masks,
)


class _SetActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_interface=3)
        self.units = nn.Parameter(torch.tensor([[[0.2, -0.1]]]))
        self.sets = nn.Parameter(torch.tensor([[[0.0, 0.8, -0.2], [0.4, -0.5, 0.1]]]))

    def forward(self, inputs: torch.Tensor) -> ActorOutput:
        batch = inputs.shape[0]
        return ActorOutput(
            self.units.expand(batch, -1, -1),
            torch.empty(batch, 2, 0),
            torch.zeros(batch, 2, 1),
        )

    def market_set_logits(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.sets.expand(context.shape[0], -1, -1)


def _factors() -> tuple[torch.Tensor, ...]:
    unit_actions = torch.tensor([[0], [1]])
    set_values = torch.tensor([[1, 0], [0, 2]])
    unit_masks = torch.ones(2, 1, 2, dtype=torch.bool)
    set_masks = torch.tensor(
        [[[True, True, True], [True, False, False]], [[True, True, True], [True, True, True]]]
    )
    return unit_actions, set_values, unit_masks, set_masks


def test_market_set_policy_batch_excludes_compiled_slot_factors() -> None:
    actor = _SetActor()
    unit_actions, set_values, unit_masks, set_masks = _factors()
    staged = {
        "unit_actions": unit_actions,
        "market_set_values": set_values,
        "unit_masks": unit_masks,
        "market_set_masks": set_masks,
        "unit_active": torch.ones(2, 1, dtype=torch.bool),
        "market_set_active": torch.tensor([[True, False], [True, True]]),
        "old_unit_logprobs": torch.zeros(2, 1),
        "old_market_set_logprobs": torch.zeros(2, 2),
    }
    selected = _policy_factor_batch_args(actor, staged, slice(None))
    assert len(selected) == 8
    assert torch.equal(selected[1], set_values)
    _validate_staged_action_masks(staged, torch.ones(2, dtype=torch.bool))


def test_market_set_joint_ratio_ignores_inactive_decisions() -> None:
    actor = _SetActor()
    factors = _factors()
    inputs = torch.zeros(2, 1)
    original_unit, original_set = _market_set_selected_logprobs(actor, *factors, False, inputs)
    with torch.no_grad():
        actor.units[0, 0, 0] += 0.3
        actor.sets[0, 0, 1] += 0.4
        actor.sets[0, 1, 0] += 2.0
    active_unit = torch.ones(2, 1)
    active_set = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    advantages = torch.tensor([1.0, -0.5])
    terms = _market_set_minibatch_terms(
        actor,
        *factors,
        active_unit,
        active_set,
        original_unit.detach(),
        original_set.detach(),
        advantages,
        0.8,
        1.2,
        False,
        inputs,
    )
    new_unit, new_set = _market_set_selected_logprobs(actor, *factors, False, inputs)
    joint_log_ratio = (new_unit - original_unit).sum(-1) + (
        (new_set - original_set) * active_set
    ).sum(-1)
    ratio = joint_log_ratio.exp()
    expected = (torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)).sum()
    torch.testing.assert_close(terms[0], expected)
    terms[0].backward()
    assert actor.sets.grad is not None
    assert torch.isfinite(actor.sets.grad).all()


def test_market_set_rejects_component_ratio() -> None:
    actor = _SetActor()
    factors = _factors()
    old_unit, old_set = _market_set_selected_logprobs(actor, *factors, False, torch.zeros(2, 1))
    try:
        _market_set_minibatch_terms(
            actor,
            *factors,
            torch.ones(2, 1),
            torch.ones(2, 2),
            old_unit,
            old_set,
            torch.ones(2),
            0.8,
            1.2,
            False,
            torch.zeros(2, 1),
            policy_ratio_scope="components",
        )
    except ValueError as error:
        assert "joint" in str(error)
    else:
        raise AssertionError("component ratios must be rejected")


def test_market_set_entropy_gradient_is_the_masked_active_entropy() -> None:
    """Both interface-3 factors carry the bonus, over active decisions only."""
    actor = _SetActor()
    factors = _factors()
    set_masks = factors[3]
    inputs = torch.zeros(2, 1)
    old_unit, old_set = _market_set_selected_logprobs(actor, *factors, False, inputs)
    # The first state's second slot is active with two masked entries, which is
    # where a differentiated near-minimum log-probability overflows; the first
    # state's inactive unit keeps the active weighting in view.
    active_unit = torch.tensor([[0.0], [1.0]])
    active_set = torch.ones(2, 2)
    arguments = (
        actor,
        *factors,
        active_unit,
        active_set,
        old_unit.detach(),
        old_set.detach(),
        torch.zeros(2),
        0.8,
        1.2,
        False,
        inputs,
    )
    assert not _market_set_minibatch_terms(*arguments)[1].requires_grad
    entropy_sum = _market_set_minibatch_terms(*arguments, entropy_gradient=True)[1]
    # A cotangent above one is what overflowed the masked log-probabilities.
    (4.0 * entropy_sum).backward()

    # By hand, over only the legal entries of each active decision.
    reference = _SetActor()

    def entropy(logits: torch.Tensor) -> torch.Tensor:
        logprob = logits.log_softmax(-1)
        return -(logprob.exp() * logprob).sum()

    expected = sum(
        entropy(reference.units[0, 0]) * active_unit[row, 0]
        + sum(
            entropy(reference.sets[0, slot][set_masks[row, slot]]) * active_set[row, slot]
            for slot in range(2)
        )
        for row in range(2)
    )
    torch.testing.assert_close(entropy_sum, expected)
    (4.0 * expected).backward()
    for name in ("units", "sets"):
        gradient = getattr(actor, name).grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        torch.testing.assert_close(gradient, getattr(reference, name).grad, msg=name)
