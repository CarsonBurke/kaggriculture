"""Observable forecasting labels and loss contracts; no CPU model execution."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.constants import MAX_UNITS
from kaggriculture.economic_forecasting import (
    ECONOMIC_FEATURE_MASK,
    FORECAST_GROUPS,
    VALUATION_STATES,
    build_economic_forecast_targets,
    economic_forecast_loss,
)
from kaggriculture.registry import ENTITY_ATTENTION
from kaggriculture.rollout import _state_field_specs
from kaggriculture.tokens import TILE_COUNT, TILE_KINDS, UNIT_CONTINUOUS_FIELDS


def _states(trajectories=2, steps=6):
    arrays = {
        name: np.zeros((trajectories, steps, *shape), dtype=dtype)
        for name, (shape, dtype) in _state_field_specs(ENTITY_ATTENTION).items()
    }
    arrays["unit_active"] = np.zeros((trajectories, steps, MAX_UNITS), dtype=np.bool_)
    arrays["farms"][..., 0] = np.arange(steps)[None, :, None] / 8
    return arrays


def test_future_indices_respect_terminal_holes_and_owned_rows():
    valid = np.asarray([[True, True, False, True, True, True], [False] * 6])
    batch = build_economic_forecast_targets(_states(), valid, (1, 2, 6))
    np.testing.assert_array_equal(batch.valid[0, :, 0], [True, False, False, True, True, False])
    np.testing.assert_array_equal(batch.valid[0, :, 1], [False, False, False, True, False, False])
    assert not batch.valid[..., 2].any()
    assert not batch.valid[1].any()
    assert not batch.features[1].any()
    # Invalid targets are finite persistence labels; even t=last never reads
    # the next physical trajectory's first state.
    source = np.arange(12).reshape(2, 6)
    np.testing.assert_array_equal(
        batch.future_indices[~batch.valid],
        np.broadcast_to(source[..., None], (2, 6, 3))[~batch.valid],
    )
    assert batch.future_indices[0, 3, 1] == 5


def test_features_have_fixed_scales_ownership_and_no_static_clock_targets():
    states = _states(1, 3)
    # Through the stock columns; later schemas' valuation columns are not labels.
    states["products"][0, :, 0, :5] = [1, 2, 99, 3, 4]
    states["products"][0, :, 0, 5:] = 99
    states["critic_products"][0, :, 0, :2] = [5, 6]
    states["critic_products"][0, :, 0, 2:] = 99
    states["animals"][0, :, 0, :3] = [99, 7, 8]
    states["animals"][0, :, 0, 3:] = 99
    states["critic_animals"][0, :, 0] = [9, 10]
    states["crops"][0, :, 0, 1] = 11
    states["crops"][0, :, 0, 6:] = 99
    states["critic_crops"][0, :, 0, 0] = 12
    states["town"][..., :6] = 99
    states["town"][..., 6:14] = np.arange(8) / 8
    states["tile_categorical"][..., :TILE_COUNT, 0] = TILE_KINDS.index("PLANT")
    states["unit_active"][..., :2] = True
    states["opponent_unit_active"][..., :4] = True
    held = UNIT_CONTINUOUS_FIELDS.index("holds_total")
    states["unit_continuous"][..., :2, held] = 0.5
    # Inactive unit garbage must not train the workforce representation.
    states["unit_continuous"][..., 2:, held] = 99
    result = build_economic_forecast_targets(states, np.ones((1, 3), dtype=np.bool_), (1,))
    features = result.features[0, 0]
    np.testing.assert_array_equal(features[4, :6], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(features[13, :4], [7, 8, 9, 10])
    np.testing.assert_array_equal(features[16, :2], [11, 12])
    np.testing.assert_array_equal(features[-1], np.arange(8) / 8)
    assert features[0, 0] == 1 and features[1, 0] == 0
    assert features[2, 0] == 2 / MAX_UNITS and features[3, 0] == 4 / MAX_UNITS
    assert features[2, 1] == 1 / MAX_UNITS
    assert np.all(features[~ECONOMIC_FEATURE_MASK] == 0)
    assert states["products"][0, 0, 0, 2] == 99  # Input remains untouched.


def test_label_gather_uses_exact_future_observation_and_preserves_current_input():
    states = _states(2, 6)
    states["farms"][1, :, :, 0] += 2
    original = states["farms"].copy()
    result = build_economic_forecast_targets(states, np.ones((2, 6), dtype=np.bool_), (1, 4))
    flat = result.features.reshape(-1, VALUATION_STATES, 8).astype(np.float32)
    source = flat[:, None]
    labels = flat[result.future_indices.reshape(-1, 2)] - source
    assert labels[0, 0, 21, 0] == 1 / 8
    assert labels[0, 1, 21, 0] == 4 / 8
    assert labels[6, 1, 21, 0] == 4 / 8
    assert not labels[5].any()
    np.testing.assert_array_equal(states["farms"], original)


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (1, 1), (True,), (1.5,)])
def test_invalid_horizons_fail(horizons):
    with pytest.raises(ValueError, match="horizons"):
        build_economic_forecast_targets(_states(), np.ones((2, 6), dtype=np.bool_), horizons)


def test_invalid_labels_fail_and_unowned_nonfinite_states_are_ignored():
    states = _states()
    states["products"][0, 0, 0, 0] = np.nan
    valid = np.ones((2, 6), dtype=np.bool_)
    with pytest.raises(ValueError, match="finite"):
        build_economic_forecast_targets(states, valid)
    valid[0] = False
    result = build_economic_forecast_targets(states, valid)
    assert np.isfinite(result.features).all()


def test_loss_balances_families_horizons_masks_padding_and_detaches_labels():
    predictions = torch.zeros((2, 2, VALUATION_STATES, 8), requires_grad=True)
    targets = torch.ones_like(predictions, requires_grad=True)
    valid = torch.tensor([[True, False], [True, True]])
    # Last sample models the minibatch padding duplicate. Unavailable horizon
    # must not halve the loss, and duplicated sample must not alter it.
    loss = economic_forecast_loss(predictions, targets, valid, torch.tensor([1.0, 0.0]))
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert targets.grad is None
    assert predictions.grad is not None
    assert not predictions.grad[1].any()
    assert not predictions.grad[:, 1].any()
    for start, stop in FORECAST_GROUPS:
        assert predictions.grad[0, 0, start:stop].abs().sum().item() == pytest.approx(
            1 / len(FORECAST_GROUPS)
        )
    assert not predictions.grad[..., ~torch.from_numpy(ECONOMIC_FEATURE_MASK)].any()


def test_loss_without_valid_horizons_is_zero_with_zero_gradients():
    predictions = torch.ones((2, 2, VALUATION_STATES, 8), requires_grad=True)
    loss = economic_forecast_loss(predictions, torch.zeros_like(predictions), torch.zeros((2, 2)))
    assert loss.item() == 0
    loss.backward()
    assert not predictions.grad.any()


def test_forecast_config_keeps_actor_identity_and_roundtrips_cli():
    import argparse
    from dataclasses import replace

    from kaggriculture.entity import EntityConfig
    from kaggriculture.modelargs import (
        actor_model_config,
        add_model_config_arguments,
        model_config_arguments,
        model_config_from_args,
    )
    from kaggriculture.registry import resolve_architecture

    default = EntityConfig()
    forecast = replace(default, critic_architecture="forecast")
    assert actor_model_config(forecast) == actor_model_config(default)
    architecture = resolve_architecture(ENTITY_ATTENTION)
    parser = argparse.ArgumentParser()
    add_model_config_arguments(parser)
    args = parser.parse_args(model_config_arguments(architecture, forecast.to_dict()))
    assert model_config_from_args(architecture, args) == forecast


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")
def test_compiled_forecast_value_is_action_independent_and_loss_reaches_trunk():
    from dataclasses import replace

    from kaggriculture.actions import N_UNIT_ACTIONS
    from kaggriculture.economic_forecasting import FORECAST_HORIZONS
    from kaggriculture.entity import EntityConfig, EntityCritic
    from kaggriculture.model import policy_compile_options
    from kaggriculture.optim import route_parameters
    from kaggriculture.ppo import (
        _critic_batch_args,
        _forecast_critic_minibatch_fit_terms,
        _stage_tensor,
    )

    # Synthetic observations exercise shape/gradient contracts only; this is not
    # a learning experiment and does not substitute for full native PPO runs.
    arrays = _states(1, 4)
    arrays["unit_active"][..., 0] = True
    arrays["opponent_unit_active"][..., 0] = True
    staged = {name: _stage_tensor(value, torch.device("cuda")) for name, value in arrays.items()}
    args = _critic_batch_args(ENTITY_ATTENTION, staged, slice(None))
    config = replace(EntityConfig(), critic_architecture="forecast")
    critic = EntityCritic(config).cuda().eval()
    _, adam, _ = route_parameters(critic)
    assert id(critic.forecast_heads.forecast_head.weight) in {id(p) for p in adam}
    with torch.no_grad():
        critic.value_head.weight.normal_(std=0.02)
        critic.forecast_heads.forecast_head.weight.normal_(std=0.02)
    from kaggriculture.constants import MAX_MARKET_ORDERS

    actions = dict(
        unit_actions=torch.zeros((4, MAX_UNITS), dtype=torch.long, device="cuda"),
        market_kinds=torch.zeros((4, MAX_MARKET_ORDERS), dtype=torch.long, device="cuda"),
        market_quantities=torch.zeros((4, MAX_MARKET_ORDERS), dtype=torch.long, device="cuda"),
        market_active=torch.ones((4, MAX_MARKET_ORDERS), dtype=torch.bool, device="cuda"),
        market_quantity_active=torch.zeros((4, MAX_MARKET_ORDERS), dtype=torch.bool, device="cuda"),
    )
    options = policy_compile_options("default")
    read = torch.compile(
        critic.forward_with_forecasts, fullgraph=True, dynamic=False, options=options
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        values, predictions = read(*args, **actions)
        changed_actions = actions | {"unit_actions": (actions["unit_actions"] + 1) % N_UNIT_ACTIONS}
        changed_values, changed_predictions = read(*args, **changed_actions)
        ignored_actions = actions | {"market_quantities": actions["market_quantities"] + 1}
        ignored_values, ignored_predictions = read(*args, **ignored_actions)
    torch.testing.assert_close(values, changed_values, rtol=0, atol=0)
    torch.testing.assert_close(values, ignored_values, rtol=0, atol=0)
    torch.testing.assert_close(predictions, ignored_predictions, rtol=0, atol=0)
    assert not torch.equal(predictions, changed_predictions)
    fit = torch.compile(
        _forecast_critic_minibatch_fit_terms, fullgraph=True, dynamic=False, options=options
    )
    value_loss, moments, forecast_loss, persistence = fit(
        critic,
        torch.linspace(-1, 1, 4, device="cuda"),
        True,
        *args,
        sample_weight=torch.ones(4, device="cuda"),
        forecast_targets=torch.ones_like(predictions),
        forecast_valid=torch.ones((4, len(FORECAST_HORIZONS)), dtype=torch.bool, device="cuda"),
        **actions,
    )
    assert torch.isfinite(moments).all() and torch.isfinite(persistence)
    (value_loss + forecast_loss).backward()
    assert critic.trunk.valuation_roles.weight.grad.abs().sum() > 0
    assert critic.forecast_heads.unit_action.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())


@pytest.mark.parametrize("script", ["train_ppo", "benchmark_ppo_iteration"])
def test_forecast_coefficient_cli_preserves_explicit_value(monkeypatch, script):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / f"{script}.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(script, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = [script, "--economic-forecast-coefficient", "0.375"]
    if script == "train_ppo":
        arguments += ["--run-dir", "unused"]
    monkeypatch.setattr("sys.argv", arguments)
    assert module.parse_args().economic_forecast_coefficient == 0.375


def test_production_launcher_serializes_forecast_coefficient():
    from pathlib import Path

    from kaggriculture.ppo import PpoConfig
    from kaggriculture.production import build_training_command

    command = build_training_command(
        Path("unused"),
        iterations=500,
        max_hours=24,
        seed=42,
        rollout_forward_mode="inductor_graph",
        update_compile_mode="default",
        initial_actors=(Path("unused-bc.pt"),),
    )
    index = command.index("--economic-forecast-coefficient")
    assert float(command[index + 1]) == PpoConfig.economic_forecast_coefficient
