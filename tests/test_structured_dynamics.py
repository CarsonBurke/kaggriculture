"""Behavioral checks for metric-only matched transition baselines."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from kaggriculture.model import Linear, softcap_value_logits
from kaggriculture.structured import (
    StructuredBelief,
    StructuredCriticBelief,
    StructuredInputs,
)
from kaggriculture.structured_dynamics import (
    PersistenceDynamics,
    ShuffledActionDynamics,
    _belief_latent_smooth_l1,
    _belief_rms_ratio,
    _critic_value_kl,
    _eligible_rms_ratio,
    _latent_smooth_l1,
    _target_index,
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
    """Row-independent transition with observable recursive belief ancestry."""

    def __init__(self, fields: int) -> None:
        super().__init__()
        self.scales = nn.Parameter(torch.linspace(0.05, 0.2, fields))

    def forward(self, belief, unit_actions, *context, active_fields=None):
        central = belief[-1].mean(dim=1, keepdim=True)
        action = unit_actions[:, :1, None].float() * 0.01
        active_fields = active_fields or (True,) * len(belief)
        return type(belief)(
            *(
                value + self.scales[kind] * (value.sin() + central + action) if active else value
                for kind, (value, active) in enumerate(zip(belief, active_fields, strict=True))
            )
        )


@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize(
    "layout", ["pairs", "non_power_of_two", "discontinuous", "step_gap", "empty"]
)
def test_compact_horizons_preserve_losses_combined_backward_and_empty_steps(
    critic: bool, layout: str
) -> None:
    if layout == "pairs":
        # 65 sources exercise the 64-row alignment boundary and false padding.
        episodes = np.repeat(np.arange(65), 2)
        steps = np.tile(np.arange(2), 65)
        horizon = 1
    elif layout == "non_power_of_two":
        # A larger, non-power-of-two batch exercises compact occupancy padding.
        episodes = np.repeat(np.arange(2397), 2)
        steps = np.tile(np.arange(2), 2397)
        horizon = 1
    elif layout == "discontinuous":
        # Matching endpoints cannot repair the invalid intermediate row at one.
        episodes = np.array([0, 1, 0, 0, 2, 2, 2, 3])
        steps = np.array([0, 9, 2, 3, 0, 1, 2, 0])
        horizon = 3
    elif layout == "step_gap":
        episodes = np.zeros(8, dtype=np.int64)
        steps = np.array([0, 9, 2, 3, 10, 11, 12, 13])
        horizon = 3
    else:
        episodes = np.arange(5)
        steps = np.zeros(5, dtype=np.int64)
        horizon = 3
    rows = len(steps)
    generator = torch.Generator().manual_seed(947)
    belief_type = StructuredCriticBelief if critic else StructuredBelief
    sizes = (1,) if critic else (TILE_COUNT, TILE_COUNT, 2, 3, 2, 2, 1)
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
    expected_eligibility = [
        [
            source + offset < rows
            and all(
                episodes[index] == episodes[index - 1] and steps[index] == steps[index - 1] + 1
                for index in range(source + 1, source + offset + 1)
            )
            for source in range(rows)
        ]
        for offset in range(1, horizon + 1)
    ]
    for offset, expected_rows in enumerate(expected_eligibility, start=1):
        _, dense_mask = _target_index(factors["episode_index"], factors["step"], offset)
        assert dense_mask.tolist() == expected_rows
        expected_sources = [source for source, valid in enumerate(expected_rows) if valid]
        assert plan.indices[0][plan.eligible[offset - 1]].tolist() == expected_sources
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
        assert terms.eligible.item() == pytest.approx(
            sum(sum(mask) for mask in expected_eligibility) / horizon
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


def test_scalar_critic_value_kl_preserves_decoded_supervision_and_stop_gradients() -> None:
    head = nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[2.0, -1.0]]))
        head.bias.fill_(0.5)
    predicted = torch.tensor([[[1.0, 2.0]], [[99.0, 99.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 1.0]], [[-99.0, -99.0]]], requires_grad=True)
    loss = _critic_value_kl(predicted, target, head, torch.tensor([True, False]))
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    torch.testing.assert_close(predicted.grad, torch.tensor([[[2.0, -1.0]], [[0.0, 0.0]]]))
    assert target.grad is None
    assert head.weight.grad is None
    assert head.bias.grad is None


@pytest.mark.parametrize("bfloat16", [False, True])
def test_categorical_critic_kl_matches_the_actual_capped_value_readout(bfloat16: bool) -> None:
    head = Linear(2, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.73, 0.41], [-0.35, 1.13]]))
        head.bias.copy_(torch.tensor([0.303, -0.27]))
    predicted = torch.tensor([[[3.0, -2.0]]], requires_grad=True)
    target = torch.tensor([[[-3.0, 2.0]]], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bfloat16):
        teacher_log_probs = softcap_value_logits(head(target).detach()).log_softmax(dim=-1)
        student_log_probs = softcap_value_logits(head(predicted)).log_softmax(dim=-1)
        expected = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum()
        actual = _critic_value_kl(predicted, target, head)
    expected_gradient = torch.autograd.grad(expected, predicted)[0]
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(predicted.grad, expected_gradient)
    assert target.grad is None
    assert head.weight.grad is None
    assert head.bias.grad is None


def test_an_invalid_transition_breaks_every_chain_through_it() -> None:
    """Row 2's recorded action did not produce row 3 (a recovery demonstration's
    perturbed step), so no horizon may cross the edge 2 -> 3."""
    episode = torch.zeros(6, dtype=torch.long)
    step = torch.arange(6)
    valid = torch.tensor([True, True, False, True, True, True])
    expected = {
        1: [True, True, False, True, True, False],
        2: [True, False, False, True, False, False],
        3: [False, False, False, False, False, False],
    }
    for offset, rows in expected.items():
        index, eligible = _target_index(episode, step, offset, valid)
        assert eligible.tolist() == rows
        _, unmasked = _target_index(episode, step, offset)
        assert unmasked.tolist() == [source + offset < 6 for source in range(6)]
        assert index.tolist() == [min(source + offset, 5) for source in range(6)]
