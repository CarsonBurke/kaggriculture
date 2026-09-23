from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kaggriculture.constants import MAX_UNITS
from kaggriculture.critic_diagnostics import critic_replay_arrays, terminal_outcomes
from kaggriculture.ppo import _actor_forward_fields, _critic_batch_args
from kaggriculture.registry import ENTITY_ATTENTION
from kaggriculture.rollout import _state_field_specs


def test_real_rollout_state_contract_supplies_shared_mask_to_critic_inputs() -> None:
    """Exercise real input assembly with tensors only; never construct or run a model."""
    states = {
        name: np.zeros((2, 3, *shape), dtype=dtype)
        for name, (shape, dtype) in _state_field_specs(ENTITY_ATTENTION).items()
    }
    assert "unit_active" not in states
    active = np.zeros((2, 3, MAX_UNITS), dtype=bool)
    active[:, :, 0] = True
    rollout = SimpleNamespace(
        states=states,
        unit_active=active,
        unit_actions=np.zeros((2, 3, MAX_UNITS), dtype=np.int8),
    )
    arrays = critic_replay_arrays(rollout)
    assert arrays["unit_active"] is active
    assert arrays["unit_actions"] is rollout.unit_actions
    assert set(name for name, _dtype in _actor_forward_fields(ENTITY_ATTENTION)) <= arrays.keys()
    staged = {
        name: torch.from_numpy(array.reshape((-1, *array.shape[2:])))
        for name, array in arrays.items()
    }
    inputs, private_categorical, private_continuous, private_active = _critic_batch_args(
        ENTITY_ATTENTION, staged, slice(None)
    )
    assert inputs.unit_active.shape == (6, MAX_UNITS)
    assert inputs.unit_active[:, 0].all()
    assert not inputs.unit_active[:, 1:].any()
    assert (
        inputs.products.shape[-1]
        == states["products"].shape[-1] + states["critic_products"].shape[-1]
    )
    assert (
        private_categorical.shape[0] == private_continuous.shape[0] == private_active.shape[0] == 6
    )
    assert "unit_active" not in rollout.states


def _terminal_rollout():
    return SimpleNamespace(
        reward_mode="terminal-outcome",
        rewards=np.asarray([[0, 0, 1], [0, 0, -1], [0, 0, 0]], dtype=np.float32),
        valid=np.ones((3, 3), dtype=bool),
        final_money=np.asarray([100_000_001.0] * 3, dtype=np.float32),
        opponent_money=np.asarray([100_000_000.0] * 3, dtype=np.float32),
    )


def test_critic_targets_use_native_outcomes_even_when_rounded_banks_are_equal():
    rollout = _terminal_rollout()
    assert np.array_equal(rollout.final_money, rollout.opponent_money)
    assert terminal_outcomes(rollout).tolist() == [1, -1, 0]


@pytest.mark.parametrize("corruption", ["shape", "nan", "nonterminal", "range", "mode", "invalid"])
def test_terminal_outcomes_reject_malformed_reward_evidence(corruption):
    rollout = _terminal_rollout()
    if corruption == "shape":
        rollout.rewards = rollout.rewards[:, -1]
    elif corruption == "nan":
        rollout.rewards[0, -1] = np.nan
    elif corruption == "nonterminal":
        rollout.rewards[0, 0] = 1
    elif corruption == "range":
        rollout.rewards[0, -1] = 0.5
    elif corruption == "mode":
        rollout.reward_mode = "terminal-bank"
    else:
        rollout.valid[0, 0] = False
    with pytest.raises(ValueError, match="native terminal outcomes"):
        terminal_outcomes(rollout)
