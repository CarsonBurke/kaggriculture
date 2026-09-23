"""Experimental entity contracts; run only in an MLQ CUDA allocation.

These are forward/backward and full-horizon sampler correctness checks, not
training runs or evidence of learning quality.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    _actor_batch_args,
    _critic_batch_args,
    _stage_tensor,
    _value_objective,
    update_replay_parity,
)
from kaggriculture.registry import ENTITY_ATTENTION
from kaggriculture.rollout import _StackedActorEnsemble, collect_mixed_play_rust
from kaggriculture.structured import StructuredInputs

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required"),
]


@pytest.fixture(scope="module")
def biased_native_wave():
    if not torch.cuda.is_bf16_supported():
        pytest.fail("entity contracts require native CUDA BF16")
    torch.manual_seed(20260919)
    config = EntityConfig(unit_tile_bias=True, shared_memory_kv=False)
    actor = EntityActor(config).cuda().eval()
    # Zero initialization would hide bias indexing and train/collection drift.
    with torch.no_grad():
        actor.trunk.tile_bias.weight.normal_(std=0.4)
    opponents = [EntityActor(config).cuda().eval() for _ in range(2)]
    for index, opponent in enumerate(opponents):
        opponent.load_state_dict(actor.state_dict())
        with torch.no_grad():
            opponent.trunk.tile_bias.weight.copy_(
                actor.trunk.tile_bias.weight.roll(index + 1, dims=0) * (index + 1)
            )
        opponent.requires_grad_(False)
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=2,
        league_games=4,
        opponent_indices=np.asarray([0, 1, 0, 1], dtype=np.int64),
        seed_start=20260919,
        episode_steps=720,
        sampling_seed=20260920,
        temperature=1.0,
        opponent_temperature=1.0,
        reward_mode="terminal-outcome",
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    return actor, opponents, rollout


@pytest.fixture(scope="module")
def biased_native_inputs(biased_native_wave):
    _, _, rollout = biased_native_wave
    staged = {
        name: _stage_tensor(array[:1, :32], torch.device("cuda"))
        for name, array in {**rollout.states, "unit_active": rollout.unit_active}.items()
    }
    actor_args = _actor_batch_args(ENTITY_ATTENTION, staged, slice(None))
    critic_args = _critic_batch_args(ENTITY_ATTENTION, staged, slice(None), actor_args=actor_args)
    return actor_args[0], critic_args


def test_nonzero_bias_vmapped_frozen_lanes_match_individual_compiled_actors(
    biased_native_wave, biased_native_inputs
):
    _, opponents, _ = biased_native_wave
    inputs, _ = biased_native_inputs
    # Distinct input rows as well as distinct learned tables expose lane mixing.
    lane_inputs = StructuredInputs(*(torch.stack((field, field.flip(0))) for field in inputs))
    ensemble = _StackedActorEnsemble(opponents)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = [
            torch.compile(actor, fullgraph=True, mode="default")(
                StructuredInputs(*(field[lane] for field in lane_inputs))
            )
            for lane, actor in enumerate(opponents)
        ]
        actual = ensemble(lane_inputs, mode="inductor_graph")
    for component, values in zip(actual, zip(*expected, strict=True), strict=True):
        torch.testing.assert_close(
            component.float(), torch.stack(values).float(), atol=0.01, rtol=0.02
        )


def test_nonzero_bias_full_horizon_sampler_replays_in_grad_tracking_update(biased_native_wave):
    actor, _, rollout = biased_native_wave
    assert rollout.valid.shape == (8, 719) and rollout.valid.all()
    assert rollout.learner_stochastic
    assert torch.count_nonzero(actor.trunk.tile_bias.weight) > 0
    parity = update_replay_parity(
        actor,
        rollout,
        minibatch_size=8192,
        compile_mode="default",
        autocast_enabled=True,
    )
    assert parity["update_replay_max_kl"] <= MAX_UPDATE_REPLAY_KL
    assert parity["update_replay_max_tail_fraction"] <= MAX_UPDATE_REPLAY_TAIL_FRACTION
    assert parity["update_replay_first_minibatch_kl"] <= MAX_FIRST_MINIBATCH_KL


@pytest.mark.parametrize("flag", ["memory_writeback", "unit_tile_bias"])
def test_experimental_critic_backward_reaches_branch_and_masks_private_units(
    biased_native_inputs, flag
):
    _, critic_args = biased_native_inputs
    config = replace(EntityConfig(shared_memory_kv=False), **{flag: True})
    torch.manual_seed(20260921)
    critic = EntityCritic(config).cuda()
    with torch.no_grad():
        critic.value_head.weight.normal_(std=0.02)
        if flag == "unit_tile_bias":
            critic.trunk.tile_bias.weight.normal_(std=0.4)
    private_units = critic_args[2].detach().clone().requires_grad_()
    assert critic_args[3].any() and (~critic_args[3]).any()
    targets = torch.linspace(-1, 1, private_units.shape[0], device="cuda")

    def loss(private):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = critic(critic_args[0], critic_args[1], private, critic_args[3])
            return _value_objective(critic, logits, targets)

    torch.compile(loss, fullgraph=True, mode="default")(private_units).backward()
    branch = critic.trunk.source_writeback if flag == "memory_writeback" else critic.trunk.tile_bias
    gradients = [parameter.grad for parameter in branch.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.float().abs().sum()) for gradient in gradients) > 0
    assert private_units.grad is not None and torch.isfinite(private_units.grad).all()
    assert private_units.grad[critic_args[3]].abs().sum() > 0
    assert torch.count_nonzero(private_units.grad[~critic_args[3]]) == 0
