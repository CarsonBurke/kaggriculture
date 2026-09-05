"""Behavioral checks for metric-only matched transition baselines."""

import torch
from torch import nn

from kaggriculture.structured import StructuredCriticBelief, StructuredInputs
from kaggriculture.structured_dynamics import (
    PersistenceDynamics,
    ShuffledActionDynamics,
    structured_critic_window_loss,
)


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
