"""Exact policy-prefix parity, including native pricing and inventory insertion order."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import cache

import numpy as np
import pytest
import torch

from kaggriculture.actions import (
    MarketKind,
    MarketLedger,
    UnitAction,
    _apply_ledger_order,
    _ledger_kind_mask,
    _ledger_quantity_mask,
    apply_unit_shed_effect,
    apply_unit_tile_effect,
    copy_tile_grid,
    unit_action_mask,
)
from kaggriculture.constants import CROPS, PRODUCTS, market_price
from kaggriculture.device_ledger import (
    POLICY_LEDGER_WIDTH,
    DeviceLedger,
    initial_ledger,
    pack_observations,
    validate_packed,
)
from kaggriculture.rust_env import load_native


@pytest.fixture(scope="module")
def rules():
    return DeviceLedger.from_native("cpu", minimum=-20000, maximum=30000)


def observations(env, games):
    result = []
    for game in range(games):
        snapshot = json.loads(env.snapshot_json(game))
        result.extend(dict(snapshot, player=p, private=snapshot["privates"][p]) for p in range(2))
    return result


@pytest.fixture(scope="module")
def corpus():
    env = load_native().BatchEnv(np.array([7, 381], dtype=np.uint64))
    rng = np.random.default_rng(4167)
    packed_rows, factors, expected = [], [], []
    for step in range(720):
        acts = env.builtin_actions(np.array([4, 4, 2, 3], dtype=np.uint8))
        if step % 24 == 0:
            packed = env.policy_ledger()
            staged = env.policy_ledger_buffers()
            env.policy_ledger_into(staged)
            np.testing.assert_array_equal(staged, packed)
            np.testing.assert_array_equal(pack_observations(observations(env, 2)), packed)
            native_factors = [
                acts[name].reshape(2, 2, -1)
                for name in ("unit_actions", "market_kinds", "market_quantities")
            ]
            random_factors = [
                rng.integers(0, count, value.shape, dtype=np.uint8)
                for value, count in zip(native_factors, (68, 22, 100), strict=True)
            ]
            for selected in (native_factors, random_factors):
                packed_rows.append(packed.copy())
                factors.append([a.reshape(4, -1).copy() for a in selected])
                expected.append(env.factor_masks(*selected))
        env.step_factors(
            *(
                acts[name].reshape(2, 2, -1)
                for name in ("unit_actions", "market_kinds", "market_quantities")
            )
        )
    return (
        np.concatenate(packed_rows),
        [np.concatenate([f[i] for f in factors]) for i in range(3)],
        {key: np.concatenate([item[key] for item in expected]) for key in expected[0]},
    )


def assert_oracle(actual, expected):
    for name, value in zip(actual._fields, actual, strict=True):
        np.testing.assert_array_equal(value.cpu().numpy(), expected[name], err_msg=name)


def test_native_bridge_and_prefix_masks_across_full_games(corpus, rules):
    packed, actions, expected = corpus
    validate_packed(packed, rules.minimum, rules.maximum)
    actual = rules.replay_masks(
        torch.from_numpy(packed), *(torch.from_numpy(a).long() for a in actions)
    )
    assert_oracle(actual, expected)


def test_native_quotes_and_boundary_validation(rules):
    for item, product in enumerate(PRODUCTS):
        for inventory in (-20000, -1, 0, 9000, 10000, 11000, 30000):
            assert rules.prices[item, inventory - rules.minimum].item() == market_price(
                product, inventory
            )
    native = load_native()
    with pytest.raises(ValueError):
        native.BatchEnv.policy_market_prices(-1_444_001, 0)
    with pytest.raises(ValueError):
        native.BatchEnv.policy_market_prices(0, 1_452_001)
    env = native.BatchEnv(np.array([1], dtype=np.uint64))
    with pytest.raises(ValueError):
        env.policy_ledger_into(np.empty((2, POLICY_LEDGER_WIDTH - 1), dtype=np.int64))
    with pytest.raises(ValueError, match="prefix bounds"):
        validate_packed(env.policy_ledger(), 9500, 10500)


def test_colocated_units_reserve_tiles_seeds_and_ordered_shed_capacity(rules):
    env = load_native().BatchEnv(np.array([7], dtype=np.uint64))
    observation = observations(env, 1)[0]
    farm = observation["farms"][0]
    farm["hands"] = [[4, 4]] * 15
    private = observation["private"]
    private["seeds"] = dict.fromkeys(CROPS, 1)
    private["shed"] = {"WHEAT": 96}
    # Capacity allocation must follow insertion order, not canonical item order.
    private["inventories"] = [
        {"MILK": 3, "WHEAT": 4},
        {"FERTILIZER": 1},
        {"GOOSE": 1},
        {},
        *({} for _ in range(12)),
    ]
    scenarios = [
        [5, 6, 45, 50, 52, 53, 54, 42, 58, 56, 57, 51, 53, 45, 46, 0],
        [54, 53, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    for selected in scenarios:
        state = initial_ledger(torch.from_numpy(pack_observations([observation])))
        seeds, shed = dict(private["seeds"]), dict(private["shed"])
        tiles = copy_tile_grid(farm["tiles"])
        for unit, action in enumerate(selected):
            expected = unit_action_mask(observation, unit, seeds, shed, tiles)
            np.testing.assert_array_equal(rules.unit_mask(state, unit)[0].numpy(), expected)
            action = action if expected[action] else int(UnitAction.PASS)
            if 45 <= action <= 49:
                seeds[CROPS[action - 45]] -= 1
            apply_unit_shed_effect(observation, unit, action, shed, tiles)
            apply_unit_tile_effect(observation, unit, action, tiles)
            state = rules.apply_unit(state, unit, torch.tensor([action]))
            np.testing.assert_array_equal(state.seeds[0].numpy(), [seeds.get(p, 0) for p in CROPS])
            np.testing.assert_array_equal(
                state.shed[0].numpy(),
                [shed.get(p, 0) for p in (*PRODUCTS, "GOOSE", "COW", "SHEEP")],
            )


@pytest.mark.parametrize("inventory", [-19000, 9538, 10000, 29000])
def test_market_prefix_exact_cost_floor_and_quantity_boundary(rules, inventory):
    env = load_native().BatchEnv(np.array([7], dtype=np.uint64))
    observation = observations(env, 1)[0]
    observation["market"]["inventory"] = dict.fromkeys(PRODUCTS, inventory)
    observation["farms"][0]["money"] = 100000
    observation["private"]["shed"] = {"WHEAT": 30, "FERTILIZER": 40, "MILK": 10}
    state = initial_ledger(torch.from_numpy(pack_observations([observation])))
    ledger = MarketLedger.from_observation(observation)
    for kind, requested in (
        (13, 30),
        (21, 40),
        (8, 45),
        (9, 10),
        (10, 1),
        (1, 1),
        (2, 1),
        (3, 100),
        (18, 10),
        (0, 1),
    ):
        mask = _ledger_kind_mask(observation, ledger)
        np.testing.assert_array_equal(rules.market_kind_mask(state)[0].numpy(), mask)
        if not mask[kind]:
            continue
        quote = rules.market_quote(state, torch.tensor([kind]))
        expected = _ledger_quantity_mask(observation, MarketKind(kind), ledger)
        np.testing.assert_array_equal(quote.mask[0].numpy(), expected)
        quantity = min(requested, int(expected.sum()))
        state = rules.apply_market(state, quote, torch.tensor([quantity - 1]))
        _apply_ledger_order(observation, MarketKind(kind), quantity, ledger)
        assert state.money.item() == ledger.money
        assert state.hires.item() == ledger.hires
        assert state.land.item() == ledger.extra_land
        np.testing.assert_array_equal(
            state.market[0].numpy(), [ledger.inventory[p] for p in PRODUCTS]
        )
        np.testing.assert_array_equal(
            state.shed[0, :9].numpy(), [ledger.shed.get(p, 0) for p in PRODUCTS]
        )


@pytest.mark.skipif(
    os.environ.get("KAGG_DEVICE_LEDGER_CUDA") != "1", reason="GPU checks require mlq"
)
def test_compiled_cuda_fullgraph_parity(corpus):
    packed, actions, expected = corpus
    rules = DeviceLedger.from_native("cuda", minimum=-20000, maximum=30000)
    from kaggriculture.model import policy_compile_options

    compiled = torch.compile(
        rules.replay_masks, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )
    tensors = [torch.as_tensor(a, device="cuda", dtype=torch.int64) for a in (packed, *actions)]
    actual = compiled(*tensors)
    assert_oracle(actual, expected)
    assert_oracle(compiled(*tensors), expected)


@pytest.mark.skipif(
    os.environ.get("KAGG_DEVICE_LEDGER_CUDA") != "1", reason="GPU checks require mlq"
)
def test_compiled_exact_scan_handles_large_integers_strides_and_frozen_lanes():
    from kaggriculture.device_ledger_kernels import ledger_cumsum
    from kaggriculture.model import policy_compile_options

    compiled = torch.compile(
        torch.vmap(ledger_cumsum),
        fullgraph=True,
        dynamic=False,
        options=policy_compile_options("default"),
    )
    values = (torch.arange(1600, device="cuda", dtype=torch.int64) + 2**40).reshape(8, 200)
    for width in (1, 12, 31, 32, 33, 100):
        inputs = values[:, : 2 * width : 2].unsqueeze(0).expand(3, -1, -1)
        torch.testing.assert_close(compiled(inputs), inputs.cumsum(-1), rtol=0, atol=0)

    # The mapped dimension need not lead the tensor. Keep both positive and
    # negative high-bit values so rounding or unsigned accumulation is visible.
    nonleading = torch.compile(
        torch.vmap(ledger_cumsum, in_dims=1, out_dims=1), fullgraph=True, dynamic=False
    )
    signed = values[:, :24:2].reshape(2, 4, 12).transpose(0, 1)
    signed = torch.where(signed.remainder(3) == 0, -signed, signed)
    torch.testing.assert_close(nonleading(signed), signed.cumsum(-1), rtol=0, atol=0)


def test_exact_scan_fake_tensor_and_nested_vmap_contracts():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from kaggriculture.device_ledger_kernels import ledger_cumsum

    # Fake CUDA tensors allocate no device storage; large strides verify the
    # abstract operator accepts layouts used by callers without CUDA execution.
    with FakeTensorMode():
        inputs = torch.empty_strided((2, 3, 12), (2**32, 100, 2), device="cuda", dtype=torch.int64)
        output = ledger_cumsum(inputs)
        assert output.shape == inputs.shape
        assert output.device == inputs.device and output.dtype == torch.int64
        assert output.is_contiguous()
        assert output is not inputs
        nested = torch.vmap(torch.vmap(ledger_cumsum))(inputs)
        assert nested.shape == inputs.shape
        nonleading = torch.vmap(ledger_cumsum, in_dims=1, out_dims=1)(inputs)
        assert nonleading.shape == inputs.shape
        empty = ledger_cumsum(torch.empty((0, 12), device="cuda", dtype=torch.int64))
        assert empty.shape == (0, 12)
        vector = ledger_cumsum(torch.empty(1, device="cuda", dtype=torch.int64))
        assert vector.shape == (1,)
        for shape in ((), (0,), (101,)):
            with pytest.raises(ValueError, match="width"):
                ledger_cumsum(torch.empty(shape, device="cuda", dtype=torch.int64))
        with pytest.raises(ValueError, match="int64"):
            ledger_cumsum(torch.empty(12, device="cuda", dtype=torch.int32))


def test_device_tables_are_shared_across_aliases_and_frozen_replica_copies(monkeypatch):
    import kaggriculture.device_ledger as ledger_module

    created = []

    def construct(device):
        created.append(device)
        return DeviceLedger(torch.ones((9, 3), dtype=torch.int32), -1)

    # Test the actual cache while keeping this contract independent of any
    # process-global cache already populated by other tests or a GPU context.
    monkeypatch.setattr(
        ledger_module,
        "_cached_device_ledger",
        cache(ledger_module._cached_device_ledger.__wrapped__),
    )
    monkeypatch.setattr(DeviceLedger, "from_native", staticmethod(construct))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    first = ledger_module.get_device_ledger("cuda")
    assert ledger_module.get_device_ledger("cuda:0") is first
    assert ledger_module.get_device_ledger(torch.device("cuda:0")) is first
    replicas = deepcopy([{"ledger": first}, {"ledger": first}])
    assert all(replica["ledger"] is first for replica in replicas)
    assert created == [torch.device("cuda:0")]
    assert ledger_module.get_device_ledger("cuda:1") is not first
    assert created == [torch.device("cuda:0"), torch.device("cuda:1")]


@pytest.mark.parametrize("use_builtins", [False, True])
def test_buffered_supplied_factors_keep_builtin_state_and_engine_results(use_builtins):
    native = load_native()
    supplied = native.BatchEnv(np.array([71, 383], dtype=np.uint64))
    reference = native.BatchEnv(np.array([71, 383], dtype=np.uint64))
    output = supplied.sample_buffers()
    reference_codes = np.array([3, 4, 3, 2], dtype=np.uint8)
    builtin_codes = reference_codes if use_builtins else np.zeros(4, dtype=np.uint8)
    for step in range(720):
        actions = reference.builtin_actions(reference_codes)
        factors = [actions[name] for name in ("unit_actions", "market_kinds", "market_quantities")]
        oracle = reference.factor_masks(*(a.reshape(2, 2, -1) for a in factors))
        supplied.step_factors_into(*factors, builtin_codes, output)
        expected = reference.step_factors(
            *(a.reshape(2, 2, -1) for a in factors), external=use_builtins
        )
        for name in ("unit_actions", "market_kinds", "market_quantities"):
            np.testing.assert_array_equal(output[name], actions[name])
        if not use_builtins:
            for name in oracle:
                np.testing.assert_array_equal(output[name], oracle[name])
        for name in (
            "rewards",
            "dones",
            "final_money",
            "previous_potentials",
            "potentials",
            "terminal_utilities",
        ):
            np.testing.assert_array_equal(output[name], expected[name])
        if step % 24 == 0:
            for game in range(2):
                assert supplied.snapshot_json(game) == reference.snapshot_json(game)


def test_causal_depth_is_independent_of_critic_and_rejects_invalid_values():
    from kaggriculture.causal_actor import CausalConfig

    config = CausalConfig(decoder_layers=3, core_layers=4)
    assert config.decoder_layers == 3 and config.core_layers == 4
    assert CausalConfig().decoder_layers == 2
    for depth in (False, 0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="decoder_layers"):
            CausalConfig(decoder_layers=depth)


@pytest.mark.parametrize(
    "tile,actions",
    [
        (
            {
                "kind": "PLANT",
                "crop": "WHEAT",
                "planted_day": 8,
                "yield_units": 5,
                "fertilized_until_day": 10,
            },
            [50, 50, 51, 45, 52, 53, 54, 42, 58, 56, 57],
        ),
        (
            {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 0, "yield_units": 4},
            [51, 51, 50, 52, 53, 55, 43, 58, 56, 57],
        ),
        (
            {
                "kind": "PASTURE",
                "animal": "COW",
                "placed_day": 0,
                "yield_units": 3,
                "fertilizer_available": True,
            },
            [51, 51, 56, 56, 58, 58, 57, 57, 53],
        ),
    ],
)
def test_local_tile_updates_match_python_for_colocated_successors(rules, tile, actions):
    env = load_native().BatchEnv(np.array([3], dtype=np.uint64))
    observation = observations(env, 1)[0]
    observation["day"] = 10
    observation["farms"][0]["hands"] = [[4, 4]] * 15
    observation["farms"][0]["tiles"][4][4] = tile
    observation["private"]["seeds"] = dict.fromkeys(CROPS, 1)
    observation["private"]["inventories"] = [
        {"WHEAT": 1, "FERTILIZER": 1, "GOOSE": 1, "COW": 1} for _ in range(16)
    ]
    state = initial_ledger(torch.from_numpy(pack_observations([observation])))
    tiles = copy_tile_grid(observation["farms"][0]["tiles"])
    seeds = dict(observation["private"]["seeds"])
    shed = dict(observation["private"]["shed"])
    for unit, action in enumerate(actions):
        expected = unit_action_mask(observation, unit, seeds, shed, tiles)
        np.testing.assert_array_equal(rules.unit_mask(state, unit)[0].numpy(), expected)
        action = action if expected[action] else 0
        if 45 <= action <= 49:
            seeds[CROPS[action - 45]] -= 1
        apply_unit_shed_effect(observation, unit, action, shed, tiles)
        apply_unit_tile_effect(observation, unit, action, tiles)
        state = rules.apply_unit(state, unit, torch.tensor([action]))


def test_demonstration_contract_accepts_native_projection_and_rejects_stale_support():
    from kaggriculture.demonstrations import project_demonstration
    from kaggriculture.device_ledger import validate_replay_contract

    env = load_native().BatchEnv(np.array([13], dtype=np.uint64))
    raw = observations(env, 1)
    projected = [
        project_demonstration(
            ob, {"farmer": ["PASS"], "market": [["HIRE"], ["BUY_SEED", "WHEAT", 3]]}
        )
        for ob in raw
    ]
    fields = (
        "unit_masks",
        "market_kind_masks",
        "market_quantity_masks",
        "unit_active",
        "market_active",
        "market_quantity_active",
    )
    masks = {name: np.stack([getattr(row, name) for row in projected]) for name in fields}
    factors = [
        np.stack([getattr(row, name) for row in projected])
        for name in ("unit_actions", "market_kinds", "market_quantities")
    ]
    packed = pack_observations(raw)
    validate_replay_contract(packed, *factors, masks)
    for name in fields:
        corrupted = dict(masks)
        corrupted[name] = masks[name].copy()
        corrupted[name].flat[0] = ~corrupted[name].flat[0]
        with pytest.raises(ValueError, match=name):
            validate_replay_contract(packed, *factors, corrupted)
    # A forbidden final selection need not change a subsequent mask, but must
    # still be rejected as a training label even if every archived mask matches.
    factors[0][:, 0] = 51  # HARVEST on empty soil.
    with pytest.raises(ValueError, match="illegal selected action in unit_masks"):
        validate_replay_contract(packed, *factors, masks)
