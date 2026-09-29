from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.actions import N_MARKET_KINDS, MarketKind, MarketLedger, UnitAction
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS, PRODUCTS
from kaggriculture.resource_conditioning import (
    RESOURCE_FEATURE_NAMES,
    RESOURCE_FEATURES,
    MarketResourceConditioner,
    market_resource_features,
    replay_market_resources,
)


def test_zero_residual_preserves_both_heads_and_receives_gradient() -> None:
    head = MarketResourceConditioner(7)
    resources = torch.arange(2 * 3 * RESOURCE_FEATURES).reshape(2, 3, -1).float() / 100
    kinds = torch.randn(2, 3, N_MARKET_KINDS, requires_grad=True)
    quantities = torch.randn(2, 3, 7, requires_grad=True)
    adjusted_kinds, adjusted_quantities = head.condition(kinds, quantities, resources)
    assert torch.equal(adjusted_kinds, kinds)
    assert torch.equal(adjusted_quantities, quantities)
    (adjusted_kinds.square().sum() + adjusted_quantities.square().sum()).backward()
    assert head.kind.weight.grad.abs().sum() > 0
    assert head.quantity.weight.grad.abs().sum() > 0
    assert torch.equal(kinds.grad, kinds.detach() * 2)
    assert torch.equal(quantities.grad, quantities.detach() * 2)


def test_numpy_host_residual_matches_torch_replay_with_nonzero_weights() -> None:
    head = MarketResourceConditioner(5)
    generator = np.random.default_rng(73)
    features = generator.normal(size=(4, 10, RESOURCE_FEATURES)).astype(np.float32)
    with torch.no_grad():
        head.kind.weight.copy_(torch.from_numpy(generator.normal(size=head.kind.weight.shape)))
        head.quantity.weight.copy_(
            torch.from_numpy(generator.normal(size=head.quantity.weight.shape))
        )
    kind_delta, quantity_delta = head(torch.from_numpy(features))
    np.testing.assert_allclose(
        kind_delta.detach(), features @ head.kind.weight.detach().numpy().T, atol=5e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        quantity_delta.detach(),
        features @ head.quantity.weight.detach().numpy().T,
        atol=5e-6,
        rtol=1e-6,
    )


def test_features_preserve_negative_inventory_and_do_not_clip_cash() -> None:
    ledger = MarketLedger(10**12, {"WHEAT": 50, "COW": 2}, 4, 2, dict.fromkeys(PRODUCTS, -3))
    features = market_resource_features(ledger)
    assert features.dtype == np.float32
    assert features.shape == (29,)
    assert features[0] > 1
    assert features[RESOURCE_FEATURE_NAMES.index("shed_WHEAT")] == 0.5
    assert features[RESOURCE_FEATURE_NAMES.index("market_WHEAT_signed_log")] < 0
    assert features[22] == 4 / MAX_UNITS
    assert features[23] == pytest.approx(2 / 3)


def test_teacher_prefix_uses_post_unit_shed_and_previous_order_cash() -> None:
    observation = {
        "player": 0,
        "farms": [
            {
                "money": 200,
                "farmer": [0, 0],
                "hands": [],
                "tiles": [["SHED"]],
                "unlocked_quadrants": [0],
            }
        ],
        "private": {"shed": {"WHEAT": 2}, "inventories": [{"WHEAT": 3}]},
        "market": {"inventory": dict.fromkeys(PRODUCTS, 100)},
    }
    units = np.zeros(MAX_UNITS, dtype=np.int64)
    units[0] = UnitAction.DROP
    kinds = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    kinds[:2] = [MarketKind.BUY_SEED_WHEAT, MarketKind.BUY_SEED_WHEAT]
    quantities[:2] = [1, 2]  # Buy two, then three seeds.
    features = replay_market_resources(observation, units, kinds, quantities)
    assert features[0, RESOURCE_FEATURE_NAMES.index("shed_WHEAT")] == pytest.approx(0.05)
    np.testing.assert_allclose(features[:3, 0], np.log1p([200, 180, 150]) / 12)
    np.testing.assert_array_equal(features[2:], np.broadcast_to(features[2], features[2:].shape))
    assert observation["private"]["shed"] == {"WHEAT": 2}


def test_native_prefix_residuals_match_replay_and_gumbel_utility_decoder() -> None:
    from kaggriculture.actions import N_QUANTITIES, N_UNIT_ACTIONS
    from kaggriculture.model import ActorOutput
    from kaggriculture.policy import categorical_statistics
    from kaggriculture.rollout import _fill_gpu_policy_statistics
    from kaggriculture.rust_env import load_native

    rows, rank = 2, 3
    rng = np.random.default_rng(117)
    native = load_native()
    environments = [native.BatchEnv(np.asarray([41], dtype=np.uint64)) for _ in range(2)]
    kind_weights = rng.normal(0, 0.2, (1, N_MARKET_KINDS, RESOURCE_FEATURES)).astype(np.float32)
    quantity_weights = rng.normal(0, 0.2, (1, rank, RESOURCE_FEATURES)).astype(np.float32)
    for environment in environments:
        environment.set_market_resource_heads(kind_weights, quantity_weights)
    unit = np.full((rows, MAX_UNITS, N_UNIT_ACTIONS), -20.0, dtype=np.float32)
    unit[:, :, UnitAction.PASS] = 20.0
    kinds = np.full((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), -10.0, dtype=np.float32)
    kinds[:, :, MarketKind.STOP] = 0.0
    kinds[:, :3, MarketKind.BUY_SEED_WHEAT] = 5.0
    context = rng.normal(size=(rows, MAX_MARKET_ORDERS, rank)).astype(np.float32)
    gates = rng.normal(0, 0.1, (1, N_MARKET_KINDS, rank)).astype(np.float32)
    values = rng.normal(0, 0.1, (1, N_QUANTITIES, rank)).astype(np.float32)
    biases = np.full((1, N_MARKET_KINDS, N_QUANTITIES), -5.0, dtype=np.float32)
    biases[:, :, 1] = 2.0  # Two seeds per order; several purchases remain affordable.
    heads = np.zeros(rows, dtype=np.uint16)
    draws = np.zeros((rows, MAX_MARKET_ORDERS), dtype=np.float32)
    deterministic = np.ones(rows, dtype=np.bool_)
    temperatures = np.asarray([0.6, 1.4], dtype=np.float32)
    builtin = np.zeros(rows, dtype=np.uint8)
    sampled, selected = (environment.sample_buffers() for environment in environments)
    environments[0].sample_and_step_into(
        unit,
        kinds,
        context,
        gates,
        values,
        biases,
        heads,
        np.zeros((rows, MAX_UNITS), dtype=np.float32),
        draws,
        draws,
        deterministic,
        temperatures,
        builtin,
        sampled,
    )
    environments[1].select_and_step_into(
        unit / temperatures[:, None, None],
        kinds / temperatures[:, None, None],
        context,
        gates,
        values,
        biases,
        heads,
        draws,
        deterministic,
        temperatures,
        builtin,
        selected,
    )
    _fill_gpu_policy_statistics(
        selected,
        ActorOutput(torch.from_numpy(unit), torch.from_numpy(kinds), torch.from_numpy(context)),
        torch.from_numpy(builtin),
        torch.from_numpy(temperatures),
    )
    for name in ("unit_actions", "market_kinds", "market_quantities", "market_resources"):
        np.testing.assert_array_equal(sampled[name], selected[name])
    for name in ("market_kind_logprobs", "market_quantity_logprobs", "entropy"):
        np.testing.assert_allclose(sampled[name], selected[name], atol=2e-5)
    resources = np.asarray(sampled["market_resources"])
    assert (resources[:, 1, 0] < resources[:, 0, 0]).all()
    seed_column = RESOURCE_FEATURE_NAMES.index("seeds_WHEAT_signed_log")
    original_seeds = np.rint(np.expm1(resources[:, 0, seed_column] * 12))
    first_purchase = np.asarray(sampled["market_quantities"])[:, 0].astype(np.int64) + 1
    np.testing.assert_allclose(
        resources[:, 1, seed_column], np.log1p(original_seeds + first_purchase) / 12, atol=1e-7
    )
    head = MarketResourceConditioner(rank)
    with torch.no_grad():
        head.kind.weight.copy_(torch.from_numpy(kind_weights[0]))
        head.quantity.weight.copy_(torch.from_numpy(quantity_weights[0]))
    conditioned_kinds, conditioned_context = head.condition(
        torch.from_numpy(kinds), torch.from_numpy(context), torch.from_numpy(resources)
    )
    expected_kind, _ = categorical_statistics(
        conditioned_kinds / torch.from_numpy(temperatures[:, None, None]),
        torch.from_numpy(np.asarray(sampled["market_kind_masks"])),
        torch.from_numpy(np.asarray(sampled["market_kinds"]).astype(np.int64)),
    )
    np.testing.assert_allclose(expected_kind.detach(), sampled["market_kind_logprobs"], atol=2e-5)
    chosen = np.asarray(sampled["market_kinds"]).astype(np.int64)
    conditioned_context = conditioned_context.detach().numpy() * (1 + gates[0][chosen])
    quantity_logits = conditioned_context @ values[0].T + biases[0][chosen]
    expected_quantity, _ = categorical_statistics(
        torch.from_numpy(quantity_logits / temperatures[:, None, None]),
        torch.from_numpy(np.asarray(sampled["market_quantity_masks"])),
        torch.from_numpy(np.asarray(sampled["market_quantities"]).astype(np.int64)),
    )
    np.testing.assert_allclose(expected_quantity, sampled["market_quantity_logprobs"], atol=2e-5)


def test_ppo_refuses_conditioned_replay_without_saved_resources() -> None:
    from types import SimpleNamespace

    from kaggriculture.ppo import _require_market_replay

    actor = SimpleNamespace(market_resource_conditioner=MarketResourceConditioner(2))
    with pytest.raises(ValueError, match="recorded pre-order ledgers"):
        _require_market_replay(actor, (object(),))
    _require_market_replay(actor, (object(), torch.zeros(1, 10, RESOURCE_FEATURES)))
    _require_market_replay(SimpleNamespace(market_resource_conditioner=None), (object(),))


def test_seed_resources_apply_planting_once_then_each_purchase() -> None:
    observation = {
        "player": 0,
        "step": 0,
        "farms": [
            {
                "money": 200,
                "farmer": [0, 0],
                "hands": [],
                "tiles": [["EMPTY"]],
                "unlocked_quadrants": [0],
            }
        ],
        "private": {"shed": {}, "seeds": {"WHEAT": 4}, "inventories": [{}]},
        "market": {"inventory": dict.fromkeys(PRODUCTS, 100)},
    }
    units = np.zeros(MAX_UNITS, dtype=np.int64)
    units[0] = UnitAction.PLANT_WHEAT
    kinds = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    kinds[:2] = [MarketKind.BUY_SEED_WHEAT, MarketKind.BUY_SEED_WHEAT]
    quantities[:2] = [1, 2]
    features = replay_market_resources(observation, units, kinds, quantities)
    seed_column = RESOURCE_FEATURE_NAMES.index("seeds_WHEAT_signed_log")
    np.testing.assert_allclose(features[:3, seed_column], np.log1p([3, 5, 8]) / 12)
    assert observation["private"]["seeds"] == {"WHEAT": 4}
    assert observation["farms"][0]["tiles"] == [["EMPTY"]]
