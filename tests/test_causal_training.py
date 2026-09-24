"""Causal teacher-forcing storage and cache contracts, without model execution."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from kaggriculture.causal_actor import CausalReplay
from kaggriculture.constants import MARKET_I0, MAX_MARKET_ORDERS, MAX_UNITS, PRODUCTS, market_price
from kaggriculture.device_ledger import POLICY_LEDGER_WIDTH, pack_observations
from kaggriculture.ppo import _actor_batch_args, _critic_batch_args
from kaggriculture.registry import CAUSAL
from kaggriculture.rollout import _state_field_specs


@pytest.fixture
def bc(monkeypatch):
    script = Path(__file__).parents[1] / "scripts/train_bc.py"
    spec = importlib.util.spec_from_file_location("causal_training_test_bc", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _observation():
    farm = {
        "farmer": [0, 0],
        "hands": [],
        "money": 23456,
        "unlocked_quadrants": [0],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    return {
        "player": 0,
        "day": 3,
        "hour": 2,
        "step": 50,
        "farms": [copy.deepcopy(farm), copy.deepcopy(farm)],
        "private": {"seeds": {"WHEAT": 7}, "shed": {"WHEAT": 3}, "inventories": [{}]},
        "market": {
            "inventory": {item: MARKET_I0 for item in PRODUCTS},
            "prices": {item: market_price(item, MARKET_I0) for item in PRODUCTS},
        },
        "town": {},
    }


def test_causal_cache_is_separate_and_bound_to_ledger_and_native_sources(bc, monkeypatch):
    files = {name: "same" for name in bc._ENCODING_SOURCE_FILES}
    files.update(
        {
            "scripts/train_bc.py": "bc",
            "src/kaggriculture/device_ledger.py": "ledger",
            "rust/kagg_env/src/lib.rs": "native",
        }
    )
    monkeypatch.setattr(bc, "source_identity", lambda: {"files": dict(files)})
    ordinary = bc._encoding_cache_schema("entity-attention")
    causal = bc._encoding_cache_schema(CAUSAL)
    assert causal != ordinary
    for name in ("src/kaggriculture/device_ledger.py", "rust/kagg_env/src/lib.rs"):
        files[name] += "changed"
        assert bc._encoding_cache_schema(CAUSAL) != causal
        assert bc._encoding_cache_schema("entity-attention") == ordinary
        causal = bc._encoding_cache_schema(CAUSAL)


def test_causal_episode_cache_packs_only_current_observations(bc, tmp_path):
    observation = _observation()
    other = copy.deepcopy(observation)
    other["private"]["seeds"]["WHEAT"] = 11
    # Opponent private changes cannot become policy ledger inputs.
    raw = {
        "observations": [
            {"observation": ob, "opponent_private": {"seeds": {"WHEAT": 99999}}}
            for ob in (observation, other)
        ]
    }
    from kaggriculture.demonstrations import project_demonstration

    projected = [
        project_demonstration(ob, {"farmer": ["PASS"], "market": []}) for ob in (observation, other)
    ]
    factors = {
        name: np.stack([getattr(row, name) for row in projected]) for name in bc._FACTOR_FIELDS
    }
    episode, cache = tmp_path / "episode.npz", tmp_path / "cache.npz"
    np.savez(
        episode,
        raw_json_zlib=np.frombuffer(zlib.compress(json.dumps(raw).encode()), dtype=np.uint8),
        **factors,
    )
    encoded = bc._encode_episode_file(str(episode), str(cache), architecture_name=CAUSAL)
    assert encoded["policy_ledger"].dtype == np.int64
    assert encoded["policy_ledger"].shape == (2, POLICY_LEDGER_WIDTH)
    np.testing.assert_array_equal(encoded["policy_ledger"], pack_observations((observation, other)))
    episode.unlink()  # A valid cache is complete without consulting raw data.
    cached = bc._encode_episode_file(str(episode), str(cache), architecture_name=CAUSAL)
    np.testing.assert_array_equal(cached["policy_ledger"], encoded["policy_ledger"])
    with pytest.raises(ValueError, match="encoded cache fields"):
        bc._encode_episode_file(str(episode), str(cache), architecture_name="entity-attention")


def test_causal_batchargs_use_recorded_prefix_and_critic_never_needs_actions():
    staged = {
        name: torch.from_numpy(np.zeros((3, *shape), dtype=dtype))
        for name, (shape, dtype) in _state_field_specs(CAUSAL).items()
    }
    staged["unit_active"] = torch.ones((3, MAX_UNITS), dtype=torch.bool)
    staged["unit_actions"] = torch.arange(3)[:, None].expand(-1, MAX_UNITS).to(torch.int8)
    staged["market_kinds"] = torch.ones((3, MAX_MARKET_ORDERS), dtype=torch.int8)
    staged["market_quantities"] = torch.full((3, MAX_MARKET_ORDERS), 2, dtype=torch.int8)
    indices = torch.tensor([2, 0])
    inputs, replay = _actor_batch_args(CAUSAL, staged, indices)
    assert isinstance(replay, CausalReplay)
    assert replay.packed_ledger.shape == (2, POLICY_LEDGER_WIDTH)
    assert all(value.dtype == torch.int64 for value in replay)
    torch.testing.assert_close(replay.unit_actions[:, 0], torch.tensor([2, 0]))
    # Building V(s) must require no action or prefix-ledger metadata, even for
    # critic-only diagnostic replays and GAE before actor minibatches exist.
    critic_states = {
        name: tensor
        for name, tensor in staged.items()
        if name not in {"policy_ledger", "unit_actions", "market_kinds", "market_quantities"}
    }
    critic_args = _critic_batch_args(CAUSAL, critic_states, indices)
    torch.testing.assert_close(critic_args[0].unit_continuous, inputs.unit_continuous)
    assert len(critic_args) == 4


def test_all_architectures_can_stage_declared_state_dtypes_in_pinned_storage():
    from kaggriculture.registry import ARCHITECTURES
    from kaggriculture.rollout import _TORCH_STORAGE_DTYPES, _state_field_specs

    for architecture in ARCHITECTURES:
        for _, dtype in _state_field_specs(architecture).values():
            dtype = np.dtype(dtype)
            assert _TORCH_STORAGE_DTYPES[dtype] == torch.from_numpy(np.empty(0, dtype=dtype)).dtype
