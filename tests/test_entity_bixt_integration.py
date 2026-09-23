"""BiXT model boundaries; CUDA cases must execute through MLQ."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_entity import config as config
from test_entity import native_batch as native_batch
from test_entity import native_rollout as native_rollout

from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.inference import ACTOR_ARTIFACT_FORMAT_VERSION, load_actor_artifact
from kaggriculture.model import policy_compile_options
from kaggriculture.modelargs import actor_model_config
from kaggriculture.optim import route_parameters
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    _actor_batch_args,
    _critic_batch_args,
    _value_objective,
    update_replay_parity,
)
from kaggriculture.provenance import source_identity
from kaggriculture.registry import ENTITY_ATTENTION, resolve_architecture
from kaggriculture.rollout import collect_mixed_play_rust
from kaggriculture.structured import SMALL_QUERY_INITIAL_SCALE

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")


def test_bixt_is_opt_in_and_prior_entity_config_still_describes_the_same_actor():
    default = EntityConfig()
    assert default.bixt_latents == 0
    assert default.shared_memory_kv is False
    old = default.to_dict()
    del old["bixt_latents"]
    restored = resolve_architecture(ENTITY_ATTENTION).build_config(old)
    assert restored == default
    assert actor_model_config(restored) == actor_model_config(default)
    enabled = replace(default, bixt_latents=32, global_modulation=False, critic_source_read=False)
    assert actor_model_config(enabled) != actor_model_config(default)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_bixt_latent_count_rejects_invalid_values(count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        EntityConfig(bixt_latents=count, global_modulation=False)


@pytest.mark.parametrize(
    "flag",
    [
        "global_modulation",
        "shared_memory_kv",
        "inter_attention_ffn",
        "unit_local_readout",
        "tile_cross_rope",
        "critic_source_read",
        "memory_writeback",
        "unit_tile_bias",
    ],
)
def test_bixt_rejects_unimplemented_mixed_architecture_flags(flag):
    with pytest.raises(ValueError, match=flag):
        replace(
            EntityConfig(bixt_latents=32, global_modulation=False, critic_source_read=False),
            **{flag: True},
        )


@pytest.fixture
def bixt_config(config):
    return replace(config, bixt_latents=32, global_modulation=False, critic_source_read=False)


def _compiled(function):
    return torch.compile(
        function, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )


@pytest.mark.cuda
@cuda_only
def test_bixt_actor_and_critic_compiled_backward_have_no_dead_parameters(bixt_config, native_batch):
    _, inputs, critic_args, factors = native_batch
    torch.manual_seed(20260924)
    actor = EntityActor(bixt_config).cuda().eval()
    critic = EntityCritic(bixt_config).cuda()
    with torch.no_grad():
        critic.value_head.weight.normal_(std=0.02)
    own = inputs.unit_continuous.detach().clone().requires_grad_()
    private = critic_args[2].detach().clone().requires_grad_()

    def objective(own, private):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy = actor(inputs._replace(unit_continuous=own))
            quantity = actor.quantity_logits(
                policy.market_quantity_context, factors["market_kinds"]
            )
            logits = critic(
                critic_args[0]._replace(unit_continuous=own),
                critic_args[1],
                private,
                critic_args[3],
            )
            targets = torch.linspace(-1, 1, logits.shape[0], device=logits.device)
            return (
                policy.unit_logits.float().square().mean()
                + policy.market_kind_logits.float().square().mean()
                + quantity.square().mean()
                + _value_objective(critic, logits, targets)
            )

    loss = _compiled(objective)(own, private)
    assert torch.isfinite(loss)
    loss.backward()
    for module in (actor, critic):
        assert module.trunk.memory is module.trunk.memory_norm is None
        latent_bank = module.trunk.bixt_latents
        assert isinstance(latent_bank, torch.nn.Embedding)
        assert latent_bank.weight.shape == (32, 96)
        matrices, adam, multipliers = route_parameters(module)
        assert all(parameter is not latent_bank.weight for parameter in matrices)
        bank_rates = [
            rate
            for parameter, rate in zip(adam, multipliers, strict=True)
            if parameter is latent_bank.weight
        ]
        assert bank_rates == [SMALL_QUERY_INITIAL_SCALE]
        assert SMALL_QUERY_INITIAL_SCALE == 0.02
        assert [block.output_tokens for block in module.trunk.core] == [None, None, 26, 26]
        assert [block.refine_latents for block in module.trunk.core] == [True, True, True, False]
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    assert own.grad[inputs.unit_active].abs().sum() > 0
    assert torch.count_nonzero(own.grad[~inputs.unit_active]) == 0
    assert private.grad[critic_args[3]].abs().sum() > 0
    assert torch.count_nonzero(private.grad[~critic_args[3]]) == 0


@pytest.mark.cuda
@cuda_only
def test_bixt_masks_inactive_slots_and_keeps_critic_private_inputs_out_of_actor(
    bixt_config, native_batch
):
    staged, inputs, critic_args, _ = native_batch
    actor = EntityActor(bixt_config).cuda().eval()
    critic = EntityCritic(bixt_config).cuda().eval()
    changed = dict(staged)
    for name in ("critic_products", "critic_animals", "critic_crops", "opponent_unit_continuous"):
        changed[name] = staged[name] + 0.5
    private_actor_args = _actor_batch_args(ENTITY_ATTENTION, changed, slice(None))
    private_critic_args = _critic_batch_args(ENTITY_ATTENTION, changed, slice(None))
    own_changed = inputs.unit_continuous.clone()
    own_changed[~inputs.unit_active] = 1000
    private_changed = critic_args[2].clone()
    private_changed[~critic_args[3]] = 1000

    actor_forward = _compiled(actor.forward_with_belief)
    critic_forward = _compiled(critic.encode_belief)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        policy, belief = actor_forward(inputs)
        private_policy, _ = actor_forward(*private_actor_args)
        masked_policy, masked_belief = actor_forward(inputs._replace(unit_continuous=own_changed))
        value = critic_forward(*critic_args).value_decision
        changed_value = critic_forward(*private_critic_args).value_decision
        masked_value = critic_forward(
            critic_args[0]._replace(unit_continuous=own_changed),
            critic_args[1],
            private_changed,
            critic_args[3],
        ).value_decision
    for expected, observed, masked in zip(policy, private_policy, masked_policy, strict=True):
        torch.testing.assert_close(expected, observed, atol=0, rtol=0)
        torch.testing.assert_close(expected, masked, atol=0, rtol=0)
    torch.testing.assert_close(value, masked_value, atol=0, rtol=0)
    assert (value.float() - changed_value.float()).abs().max() > 0
    assert torch.count_nonzero(belief.unit_decisions[~inputs.unit_active]) == 0
    torch.testing.assert_close(belief.unit_decisions, masked_belief.unit_decisions, atol=0, rtol=0)


@pytest.mark.cuda
@cuda_only
@pytest.mark.parametrize("bixt", [False, True], ids=["pre-bixt-artifact", "bixt-artifact"])
def test_bixt_and_prior_actor_artifacts_preserve_compiled_outputs(
    config, native_batch, tmp_path, bixt
):
    _, inputs, _, _ = native_batch
    configuration = (
        replace(config, bixt_latents=32, global_modulation=False, critic_source_read=False)
        if bixt
        else config
    )
    actor = EntityActor(configuration).cuda().eval()
    model_config = configuration.to_dict()
    if not bixt:
        del model_config["bixt_latents"]
        assert not any("bixt" in name for name in actor.state_dict())
    artifact = tmp_path / "actor.pt"
    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "architecture": ENTITY_ATTENTION,
            "model_config": model_config,
            "actor": actor.state_dict(),
            "source_identity": source_identity(),
            "run_provenance": None,
        },
        artifact,
    )
    restored, _ = load_actor_artifact(artifact, device="cuda")
    assert restored.state_dict().keys() == actor.state_dict().keys()
    assert actor_model_config(restored.config) == actor_model_config(configuration)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _compiled(actor)(inputs)
        observed = _compiled(restored)(inputs)
    for left, right in zip(expected, observed, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.cuda
@cuda_only
def test_bixt_full_native_horizon_with_vmapped_frozen_opponents_replays(bixt_config):
    torch.manual_seed(20260925)
    actor = EntityActor(bixt_config).cuda().eval()
    opponents = [EntityActor(bixt_config).cuda().eval().requires_grad_(False) for _ in range(2)]
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=2,
        league_games=2,
        opponent_indices=np.asarray([0, 1], dtype=np.int64),
        seed_start=20260925,
        episode_steps=720,
        sampling_seed=20260926,
        temperature=1.0,
        opponent_temperature=1.0,
        reward_mode="terminal-outcome",
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    assert rollout.valid.shape == (6, 719) and rollout.valid.all()
    assert rollout.learner_stochastic
    parity = update_replay_parity(
        actor, rollout, minibatch_size=8192, compile_mode="default", autocast_enabled=True
    )
    assert parity["update_replay_max_kl"] <= MAX_UPDATE_REPLAY_KL
    assert parity["update_replay_max_tail_fraction"] <= MAX_UPDATE_REPLAY_TAIL_FRACTION
    assert parity["update_replay_first_minibatch_kl"] <= MAX_FIRST_MINIBATCH_KL
