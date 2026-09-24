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
