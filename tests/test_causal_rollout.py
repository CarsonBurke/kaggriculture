"""Execution-order native collection, authoritative factors and replay contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.causal_actor import CausalActor, CausalConfig
from kaggriculture.causal_rollout import SelectedFactorTransfer
from kaggriculture.device_ledger import POLICY_LEDGER_WIDTH
from kaggriculture.model import policy_compile_options
from kaggriculture.ppo import (
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    update_replay_parity,
)
from kaggriculture.rollout import collect_mixed_play_rust
from kaggriculture.rust_env import load_native


def test_native_step_statistics_restore_physical_execution_order_and_activity():
    # Native stepping intentionally clears neural statistics. Restore the exact
    # sampler density afterward, using authoritative native activity for entropy.
    transfer = object.__new__(SelectedFactorTransfer)
    transfer.arrays = {
        "factor_logprobs": -np.arange(72, dtype=np.float32).reshape(2, 36),
        "factor_entropies": np.arange(72, dtype=np.float32).reshape(2, 36),
    }
    sampled = {
        "unit_logprobs": np.zeros((2, 16), dtype=np.float32),
        "market_kind_logprobs": np.zeros((2, 10), dtype=np.float32),
        "market_quantity_logprobs": np.zeros((2, 10), dtype=np.float32),
        "entropy": np.zeros(2, dtype=np.float32),
        "unit_active": np.zeros((2, 16), dtype=bool),
        "market_active": np.zeros((2, 10), dtype=bool),
        "market_quantity_active": np.zeros((2, 10), dtype=bool),
    }
    sampled["unit_active"][0, 0] = True
    sampled["market_active"][0, :2] = True
    sampled["market_quantity_active"][0, 0] = True
    transfer.store_statistics(sampled)
    np.testing.assert_array_equal(sampled["unit_logprobs"][0], -np.arange(16))
    np.testing.assert_array_equal(sampled["market_kind_logprobs"][0], -np.arange(16, 36, 2))
    np.testing.assert_array_equal(sampled["market_quantity_logprobs"][0], -np.arange(17, 36, 2))
    np.testing.assert_array_equal(sampled["entropy"], [(0 + 16 + 17 + 18) / 4, 0])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")
def test_native_causal_full_horizon_frozen_lanes_and_parallel_replay():
    if not torch.cuda.is_bf16_supported():
        pytest.fail("causal native contracts require CUDA BF16")
    torch.manual_seed(20261004)
    config = CausalConfig()
    actor = CausalActor(config).cuda().eval()
    opponents = [CausalActor(config).cuda().eval() for _ in range(3)]
    for index, opponent in enumerate(opponents):
        opponent.load_state_dict(actor.state_dict())
        with torch.no_grad():
            opponent.market_kind.bias[index + 1] += 0.5
        opponent.requires_grad_(False)
    # Unequal lane widths and three neural opponents exercise padding/vmap;
    # a builtin executes independently of the neural output's zero-filled row.
    first_seed = 20261004
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=1,
        league_games=5,
        builtin_lanes=("starter",),
        opponent_indices=np.asarray([0, 0, 1, 2, 3]),
        seed_start=first_seed,
        episode_steps=720,
        sampling_seed=20261005,
        temperature=1.0,
        opponent_temperatures=(0.75, 1.0, 1.25),
        deterministic_opponents=(False, True, False),
        reward_mode="terminal-outcome",
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    assert rollout.architecture == "causal-execution" and rollout.learner_stochastic
    assert rollout.valid.shape == (7, 719) and rollout.valid.all()
    ledgers = rollout.states["policy_ledger"]
    assert ledgers.shape == (7, 719, POLICY_LEDGER_WIDTH) and ledgers.dtype == np.int64
    # Stored ledgers must precede this turn's selected actions, rather than the
    # next turn's asynchronously uploaded double buffer.
    env = load_native().BatchEnv(np.arange(first_seed, first_seed + 6, dtype=np.uint64))
    seats = np.arange(first_seed + 1, first_seed + 6) % 2
    stored = np.concatenate((np.arange(2), 2 + 2 * np.arange(5) + seats))
    np.testing.assert_array_equal(ledgers[:, 0], env.policy_ledger()[stored])
    assert np.any(ledgers[:, 1] != ledgers[:, 0])
    for name in ("old_unit_logprobs", "old_market_kind_logprobs", "old_market_quantity_logprobs"):
        values = getattr(rollout, name)
        assert np.isfinite(values).all() and (values <= 1e-6).all()
    assert np.isfinite(rollout.entropy_sums).all() and rollout.mean_entropy > 0

    # The public packed state plus recorded prefix must reconstruct every
    # authoritative native mask over all 719 turns, including STOP descendants.
    rules = actor.device_ledger
    assert all(opponent.device_ledger is rules for opponent in opponents)
    replay_masks = torch.compile(
        rules.replay_masks, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )
    actions = [
        torch.as_tensor(getattr(rollout, name).reshape(-1, width), device="cuda").long()
        for name, width in (("unit_actions", 16), ("market_kinds", 10), ("market_quantities", 10))
    ]
    with torch.no_grad():
        masks = replay_masks(
            torch.as_tensor(ledgers.reshape(-1, POLICY_LEDGER_WIDTH), device="cuda"), *actions
        )
    for name, actual in zip(masks._fields, masks, strict=True):
        expected = getattr(rollout, name)
        np.testing.assert_array_equal(
            actual.cpu().numpy(), expected.reshape(actual.shape), err_msg=name
        )
    del masks, actions
    replay = update_replay_parity(
        actor, rollout, minibatch_size=512, compile_mode="default", autocast_enabled=True
    )
    assert replay["update_replay_max_kl"] <= MAX_UPDATE_REPLAY_KL
    assert replay["update_replay_max_tail_fraction"] <= MAX_UPDATE_REPLAY_TAIL_FRACTION
    assert replay["update_replay_joint_kl"] <= MAX_UPDATE_REPLAY_KL
