"""Behavioral checks for metric-only matched transition baselines."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from kaggriculture.structured import StructuredBelief, StructuredCriticBelief, StructuredInputs
from kaggriculture.structured_dynamics import (
    PersistenceDynamics,
    ShuffledActionDynamics,
    _belief_latent_smooth_l1,
    _belief_rms_ratio,
    _critic_value_kl,
    _eligible_rms_ratio,
    _latent_smooth_l1,
    structured_critic_horizon_loss,
    structured_critic_window_loss,
    structured_horizon_loss,
    structured_horizon_plan,
)
from kaggriculture.tokens import TILE_COUNT


def test_action_conditioning_beats_persistence_and_shuffled_actions_without_rng() -> None:
    class ExactTransition(nn.Module):
        def forward(self, belief, unit_actions, *context):
            delta = unit_actions[:, :1, None].float()
            return type(belief)(*(value + delta for value in belief))

    # Two complete windows have different action effects, but the same source.
    states = torch.tensor([0.0, 1.0, 0.0, 3.0]).reshape(4, 1, 1)
    belief = StructuredCriticBelief(*(states.clone() for _ in StructuredCriticBelief._fields))
    inputs = StructuredInputs(
        **{name: torch.zeros(4, 1, dtype=torch.long) for name in StructuredInputs._fields}
    )
    factors = {
        "unit_actions": torch.tensor([[1], [0], [3], [0]]),
        "market_kinds": torch.zeros(4, 1, dtype=torch.long),
        "market_quantities": torch.zeros(4, 1, dtype=torch.long),
    }
    head = nn.Linear(1, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[-1.0], [1.0]]))
        head.bias.zero_()
    rng = torch.get_rng_state().clone()

    def loss(dynamics):
        return structured_critic_window_loss(
            dynamics,
            belief,
            inputs,
            factors,
            value_head=head,
            horizon=1,
        )

    model = loss(ExactTransition())
    persistence = loss(PersistenceDynamics())
    shuffled = loss(ShuffledActionDynamics(ExactTransition()))
    assert model.latent == 0
    assert model.value.abs() < 1e-7
    assert persistence.latent > model.latent
    assert shuffled.latent > model.latent
    assert persistence.value > model.value
    assert shuffled.value > model.value
    assert torch.equal(rng, torch.get_rng_state())
    assert head.weight.grad is None

    # A fresh zero readout has no value prediction task. When PPO changes the
    # readout, the same current-state persistence baseline becomes informative.
    with torch.no_grad():
        head.weight.zero_()
    assert loss(PersistenceDynamics()).value == 0
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[-1.0], [1.0]]))
    assert loss(PersistenceDynamics()).value > 0


class _RecurrentTransition(nn.Module):
    """Row-independent transition with observable recurrent central ancestry."""

    def __init__(self, fields: int) -> None:
        super().__init__()
        self.scales = nn.Parameter(torch.linspace(0.05, 0.2, fields))

    def forward(self, belief, unit_actions, *context, active_fields=None):
        central = belief.central_latents.mean(dim=1, keepdim=True)
        action = unit_actions[:, :1, None].float() * 0.01
        active_fields = active_fields or (True,) * len(belief)
        return type(belief)(
            *(
                value + self.scales[kind] * (value.sin() + central + action) if active else value
                for kind, (value, active) in enumerate(zip(belief, active_fields, strict=True))
            )
        )


@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize("layout", ["pairs", "discontinuous", "empty"])
def test_compact_horizons_preserve_losses_combined_backward_and_empty_steps(
    critic: bool, layout: str
) -> None:
    if layout == "pairs":
        # 65 sources exercise the 64-row alignment boundary and false padding.
        episodes = np.repeat(np.arange(65), 2)
        steps = np.tile(np.arange(2), 65)
        horizon = 1
    elif layout == "discontinuous":
        # Offset two admits source zero despite its discontinuous intermediate:
        # dropping it at offset one would sever a valid recursive prediction.
        episodes = np.array([0, 1, 0, 0, 2, 2, 2, 3])
        steps = np.array([0, 9, 2, 3, 0, 1, 2, 0])
        horizon = 3
    else:
        episodes = np.arange(5)
        steps = np.zeros(5, dtype=np.int64)
        horizon = 3
    rows = len(steps)
    generator = torch.Generator().manual_seed(947)
    belief_type = StructuredCriticBelief if critic else StructuredBelief
    sizes = (
        (TILE_COUNT, TILE_COUNT, 2, 3, 2, 1) if critic else (TILE_COUNT, TILE_COUNT, 2, 3, 2, 2, 1)
    )
    values = tuple(torch.randn(rows, size, 4, generator=generator) for size in sizes)
    inputs = StructuredInputs(
        **{
            name: (
                torch.zeros(rows, 2 * TILE_COUNT, 1)
                if name in {"tile_categorical", "tile_continuous"}
                else torch.zeros(rows, 1, dtype=torch.long)
            )
            for name in StructuredInputs._fields
        }
    )
    factors = {
        "episode_index": torch.from_numpy(episodes),
        "step": torch.from_numpy(steps),
        "unit_actions": torch.arange(rows).remainder(5).reshape(rows, 1),
        "market_kinds": torch.zeros(rows, 1, dtype=torch.long),
        "market_quantities": torch.zeros(rows, 1, dtype=torch.long),
    }
    plan = structured_horizon_plan(episodes, steps, horizon)
    dynamics = _RecurrentTransition(len(sizes))
    head = nn.Linear(4, 3)
    outcomes = []
    for selection in (None, plan):
        model = copy.deepcopy(dynamics)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        belief = belief_type(*(value.clone().requires_grad_() for value in values))
        if critic:
            terms = structured_critic_horizon_loss(
                model, belief, inputs, factors, value_head=head, horizon=horizon, plan=selection
            )
            auxiliary = terms.latent + terms.value
        else:
            target = StructuredBelief(*(value.clone().requires_grad_() for value in values))
            terms = structured_horizon_loss(
                model,
                belief,
                inputs,
                factors,
                decode=None,
                decision_horizon=0,
                latent_horizon=horizon,
                patch_horizon=horizon,
                economy_active=True,
                opponent_summary_active=True,
                opponent_patches_active=True,
                target_belief=target,
                plan=selection,
            )
            auxiliary = (
                terms.latent
                + terms.patch
                + terms.economy
                + terms.opponent_summary
                + terms.opponent_patches
            )
        # The diagnostic retains this same graph; the source and predictor then
        # receive one combined primary+NextLat backward rather than a second trunk.
        parameters = (*belief, *model.parameters())
        diagnostic = torch.autograd.grad(
            auxiliary, parameters, retain_graph=True, allow_unused=True
        )
        primary = sum(value.square().mean() for value in belief)
        (primary + auxiliary).backward()
        assert head.weight.grad is None
        if not critic:
            assert all(value.grad is None for value in target)
        if layout == "empty":
            assert auxiliary == 0
            assert model.scales.grad is not None
            assert torch.count_nonzero(model.scales.grad) == 0
        else:
            assert torch.count_nonzero(model.scales.grad) > 0
            assert any(
                gradient is not None and torch.count_nonzero(gradient) > 0
                for gradient in diagnostic[: len(belief)]
            )
        optimizer.step()
        outcomes.append(
            (
                terms,
                diagnostic,
                tuple(value.grad for value in parameters),
                model.scales.detach().clone(),
                optimizer.state[model.scales]["step"],
            )
        )
    dense, compact = outcomes
    torch.testing.assert_close(compact, dense)


def test_field_reductions_match_joined_element_weights_and_global_rms() -> None:
    generator = torch.Generator().manual_seed(514)
    predicted = tuple(
        torch.randn(4, size, 5, generator=generator).requires_grad_() for size in (1, 3, 13)
    )
    previous = tuple(
        torch.randn(4, size, 5, generator=generator).requires_grad_() for size in (1, 3, 13)
    )
    eligible = torch.tensor([True, False, True, False])
    joined_predicted, joined_previous = torch.cat(predicted, 1), torch.cat(previous, 1)
    expected = _latent_smooth_l1(joined_predicted, joined_previous, eligible)
    actual = _belief_latent_smooth_l1(predicted, previous, eligible)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, predicted),
        torch.autograd.grad(expected, predicted),
    )
    assert all(value.grad is None for value in previous)
    torch.testing.assert_close(
        _belief_rms_ratio(predicted, previous, eligible),
        _eligible_rms_ratio(joined_predicted, joined_previous, eligible),
    )


def test_critic_value_kl_masks_rows_without_cross_batch_broadcast() -> None:
    head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))
    predicted = torch.tensor([[[0.4, -0.2]], [[9.0, -9.0]], [[-9.0, 9.0]]], requires_grad=True)
    target = torch.tensor([[[-0.3, 0.5]], [[-9.0, 9.0]], [[9.0, -9.0]]], requires_grad=True)
    expected = _critic_value_kl(predicted[:1], target[:1], head)
    actual = _critic_value_kl(predicted, target, head, torch.tensor([True, False, False]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(predicted.grad[0]) > 0
    assert torch.count_nonzero(predicted.grad[1:]) == 0
    assert target.grad is None
    assert head.weight.grad is None
