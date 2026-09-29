"""Outcome supervision preserves the match-score objective and actor identity."""

import argparse
from dataclasses import replace

import numpy as np
import pytest
import torch

from kaggriculture.lejepa_model import LejepaConfig, LejepaCritic
from kaggriculture.modelargs import (
    actor_model_config,
    add_model_config_arguments,
    model_config_from_args,
)
from kaggriculture.outcome_value import (
    match_score_calibration,
    outcome_value_loss,
    validate_outcome_objective,
    validate_terminal_outcome_rewards,
)
from kaggriculture.ppo import _value_objective
from kaggriculture.registry import resolve_architecture


def test_wdl_head_value_loss_and_checkpoint_contract():
    config = LejepaConfig(model_dim=16, attention_heads=2, attention_kv_heads=1, wdl_value=True)
    critic = LejepaCritic(config)
    assert critic.value_head.out_features == 3
    torch.testing.assert_close(critic.support, torch.tensor([-1.0, 0.0, 1.0]))
    probabilities = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    logits = probabilities.log().requires_grad_()
    targets = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
    torch.testing.assert_close(critic.value(logits), torch.tensor([0.6, -0.7, 0.0]))
    actual = _value_objective(critic, logits, targets)
    expected = -torch.tensor([0.7, 0.8, 0.8]).log().mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None and targets.grad is None
    assert logits.grad[0, 2] < 0 and logits.grad[1, 0] < 0 and logits.grad[2, 1] < 0
    restored = resolve_architecture("lejepa").build_critic(config.to_dict())
    restored.load_state_dict(critic.state_dict(), strict=True)
    assert actor_model_config(config) == actor_model_config(replace(config, wdl_value=False))


def test_wdl_is_the_default_and_rejects_scalar_conflict():
    parser = argparse.ArgumentParser()
    add_model_config_arguments(parser)
    architecture = resolve_architecture("lejepa")
    assert model_config_from_args(architecture, parser.parse_args([])).wdl_value
    assert not model_config_from_args(
        architecture, parser.parse_args(["--wdl-value", "false"])
    ).wdl_value
    with pytest.raises(ValueError, match="requires wdl_value=False"):
        LejepaConfig(scalar_value=True)
    assert LejepaConfig(wdl_value=False, scalar_value=True).scalar_value


def test_saved_configs_without_the_field_restore_their_distributional_critic():
    saved = LejepaConfig(model_dim=16, attention_heads=2, attention_kv_heads=1).to_dict()
    del saved["wdl_value"]
    assert not resolve_architecture("lejepa").build_config(saved).wdl_value


@pytest.mark.parametrize(
    "mode,gamma,lam",
    [
        ("shaped", 1.0, 1.0),
        ("terminal-bank", 1.0, 1.0),
        ("terminal-outcome", 0.99, 1.0),
        ("terminal-outcome", 1.0, 0.95),
    ],
)
def test_wdl_rejects_surrogates_and_bootstrapped_targets(mode, gamma, lam):
    with pytest.raises(ValueError, match="WDL critic requires"):
        validate_outcome_objective(mode, gamma, lam)


@pytest.mark.parametrize("value", [0.01, -0.9999, float("nan"), 2.0])
def test_wdl_does_not_round_margin_or_bootstrap_into_an_outcome(value):
    with pytest.raises(ValueError, match="exact completed-game outcomes"):
        outcome_value_loss(torch.zeros(1, 3), torch.tensor([value]))


def test_match_score_calibration_preserves_draws_and_invalid_probabilities():
    outcomes = np.array([-1.0, 0.0, 1.0])
    perfect = match_score_calibration(outcomes, outcomes)
    assert all(value == 0 for value in perfect.values())
    uniform = match_score_calibration(np.zeros(3), outcomes)
    assert uniform["behavior_match_score_mse"] == pytest.approx(1 / 6)
    assert uniform["behavior_match_score_bias"] == 0
    assert uniform["behavior_match_score_calibration_error"] == 0
    # Same calibration can conceal different discrimination; report both MSE
    # and calibration error. Out-of-range values must remain visible.
    bad = match_score_calibration(np.array([3.0]), np.array([1.0]))
    assert bad["behavior_match_score_mse"] == 1
    assert bad["behavior_match_score_out_of_range_fraction"] == 1


def test_outcome_validation_handles_padding_and_rejects_dense_integer_returns():
    rewards = np.array([[0.0, 1.0, np.nan], [0.0, 0.0, -1.0]])
    valid = np.array([[True, True, False], [True, True, True]])
    validate_terminal_outcome_rewards(rewards, valid)
    # Even when every suffix return is one of -1/0/1, intermediate rewards
    # are not outcome supervision. Do not infer correctness from class range.
    rewards[0, :2] = [-1.0, 1.0]
    with pytest.raises(ValueError, match="zero intermediate rewards"):
        validate_terminal_outcome_rewards(rewards, valid)
