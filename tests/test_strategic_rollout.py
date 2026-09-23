"""Native shared-plan collection contracts; model execution requires MLQ CUDA."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.ppo import (
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    update_replay_parity,
)
from kaggriculture.registry import STRATEGIC
from kaggriculture.rollout import _state_field_specs, collect_mixed_play_rust
from kaggriculture.strategic_actor import StrategicActor, StrategicConfig


def test_plan_storage_preserves_indices_above_int8_range():
    config = StrategicConfig(plan_count=256)
    _, dtype = _state_field_specs(STRATEGIC)["plan_indices"]
    stored = np.asarray([0, 127, 128, config.plan_count - 1], dtype=dtype)
    np.testing.assert_array_equal(stored, [0, 127, 128, 255])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")
def test_native_strategic_rollout_frozen_lanes_and_augmented_likelihood():
    if not torch.cuda.is_bf16_supported():
        pytest.fail("strategic native contracts require CUDA BF16")
    torch.manual_seed(20260930)
    config = StrategicConfig()
    actor = StrategicActor(config).cuda().eval()
    opponents = [StrategicActor(config).cuda().eval() for _ in range(3)]
    for index, opponent in enumerate(opponents):
        opponent.load_state_dict(actor.state_dict())
        with torch.no_grad():
            # Distinct plan priors exercise vmapped independent frozen weights.
            opponent.trunk.plan_head[-1].bias[index] += 1
        opponent.requires_grad_(False)
    # Three active neural lanes bucket to four; unequal row counts pad lane
    # widths too. The final built-in lane has no plan head and is never stored.
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=1,
        league_games=5,
        builtin_lanes=("starter",),
        opponent_indices=np.asarray([0, 0, 1, 2, 3], dtype=np.int64),
        seed_start=20260930,
        episode_steps=720,
        sampling_seed=20261001,
        temperature=1.0,
        opponent_temperatures=(0.75, 1.0, 1.25),
        deterministic_opponents=(False, True, False),
        reward_mode="terminal-outcome",
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    assert rollout.architecture == STRATEGIC and rollout.learner_stochastic
    assert rollout.valid.shape == (7, 719) and rollout.valid.all()
    assert rollout.states["plan_active"].all()
    plans = rollout.states["plan_indices"]
    assert plans.dtype == np.int32 and plans.min() >= 0 and plans.max() < config.plan_count
    assert np.unique(plans).size > 1
    assert np.isfinite(rollout.states["old_plan_logprobs"]).all()
    assert (rollout.states["old_plan_logprobs"] <= 0).all()
    assert np.isfinite(rollout.entropy_sums).all() and rollout.mean_entropy > 0
    replay = update_replay_parity(
        actor,
        rollout,
        minibatch_size=512,
        compile_mode="default",
        autocast_enabled=True,
    )
    assert replay["update_replay_plan_active_count"] == rollout.state_count
    assert replay["update_replay_max_kl"] <= MAX_UPDATE_REPLAY_KL
    assert replay["update_replay_max_tail_fraction"] <= MAX_UPDATE_REPLAY_TAIL_FRACTION
    assert replay["update_replay_joint_kl"] <= MAX_UPDATE_REPLAY_KL
