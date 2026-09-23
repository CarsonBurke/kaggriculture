"""Economic-critic configuration and compiled CUDA contracts (GPU only via MLQ)."""

from __future__ import annotations

import argparse
import io
from dataclasses import replace

import pytest
import torch
from test_entity import config as config
from test_entity import native_batch as native_batch
from test_entity import native_rollout as native_rollout

from kaggriculture.economic_critic import ECONOMY_STATES, VALUATION_STATES, EconomicCriticTrunk
from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.model import policy_compile_options
from kaggriculture.modelargs import (
    actor_model_config,
    add_model_config_arguments,
    model_config_arguments,
    model_config_from_args,
)
from kaggriculture.optim import route_parameters
from kaggriculture.ppo import _value_objective
from kaggriculture.registry import ENTITY_ATTENTION, resolve_architecture
from kaggriculture.structured_dynamics import StructuredCriticDynamics

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")


def test_economic_critic_is_opt_in_with_identical_actor_artifact_identity():
    default = EntityConfig()
    economic = replace(default, critic_architecture="economic")
    assert default.critic_architecture == "entity" and default.critic_source_read
    assert actor_model_config(default) == actor_model_config(economic)
    old = default.to_dict()
    del old["critic_architecture"]
    architecture = resolve_architecture(ENTITY_ATTENTION)
    assert architecture.build_config(old) == default
    assert architecture.build_config(economic.to_dict()) == economic
    parser = argparse.ArgumentParser()
    add_model_config_arguments(parser)
    args = parser.parse_args(model_config_arguments(architecture, economic.to_dict()))
    assert model_config_from_args(architecture, args) == economic
    assert ECONOMY_STATES == 20 and VALUATION_STATES == 24


@pytest.mark.parametrize("value", ["forecaster", "", True, None])
def test_invalid_critic_architecture_is_rejected(value):
    with pytest.raises(ValueError, match="critic_architecture"):
        EntityConfig(critic_architecture=value)


def test_economic_critic_and_bixt_are_separate_experiments():
    with pytest.raises(ValueError, match="BiXT requires critic_architecture"):
        EntityConfig(
            critic_architecture="economic",
            bixt_latents=32,
            global_modulation=False,
            critic_source_read=False,
        )


def _compiled(function):
    return torch.compile(
        function, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )


@pytest.mark.cuda
@cuda_only
@pytest.mark.parametrize("shared_memory_kv", [False, True])
def test_economic_value_loss_reaches_sources_and_every_parameter(
    config, native_batch, shared_memory_kv
):
    _, public, args, _ = native_batch
    configuration = replace(
        config, critic_architecture="economic", shared_memory_kv=shared_memory_kv
    )
    torch.manual_seed(20260927)
    critic = EntityCritic(configuration).cuda()
    assert isinstance(critic.trunk, EconomicCriticTrunk)
    assert critic.source_pool_norm is None
    assert not any("market_queries" in name for name, _ in critic.named_parameters())
    assert critic.trunk.units.gather_projection is None
    matrices, vectors, _ = route_parameters(critic)
    assert {id(p) for p in (*matrices, *vectors)} == {id(p) for p in critic.parameters()}
    assert id(critic.trunk.valuation_roles.weight) in {id(p) for p in vectors}
    # A fresh zero value head intentionally delays encoder gradients one update.
    with torch.no_grad():
        critic.value_head.weight.normal_(std=0.02)
    sources = {
        name: getattr(args[0], name).detach().clone().requires_grad_()
        for name in ("tile_continuous", "unit_continuous", "products", "animals", "crops")
    }
    private_units = args[2].detach().clone().requires_grad_()

    def objective(tile, own, products, animals, crops, private):
        inputs = args[0]._replace(
            tile_continuous=tile,
            unit_continuous=own,
            products=products,
            animals=animals,
            crops=crops,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, belief = critic.forward_with_belief(inputs, args[1], private, args[3])
            targets = torch.linspace(-1, 1, logits.shape[0], device="cuda")
            return _value_objective(critic, logits, targets), belief.value_decision

    loss, belief = _compiled(objective)(*sources.values(), private_units)
    assert belief.shape == (public.unit_active.shape[0], 1, config.model_dim)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in critic.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for round_ in critic.trunk.core:
        assert round_.cross_attention.query.weight.grad.abs().sum() > 0
        if round_.memory is not None:
            for gradient in round_.memory.key_value.weight.grad.chunk(2):
                assert gradient.abs().sum() > 0
    assert critic.trunk.valuation_roles.weight.grad.abs().sum() > 0
    assert sources["tile_continuous"].grad[:, :100].abs().sum() > 0
    assert sources["tile_continuous"].grad[:, 100:].abs().sum() > 0
    for name in ("products", "animals", "crops"):
        private = sources[name].grad[..., getattr(public, name).shape[-1] :]
        assert private.abs().sum() > 0, name
    for unit_source, active in (
        (sources["unit_continuous"], public.unit_active),
        (private_units, args[3]),
    ):
        assert unit_source.grad[active].abs().sum() > 0
        assert torch.count_nonzero(unit_source.grad[~active]) == 0


@pytest.mark.cuda
@cuda_only
def test_economic_critic_masks_both_players_and_handles_empty_workforces(config, native_batch):
    _, _, args, _ = native_batch
    critic = EntityCritic(replace(config, critic_architecture="economic")).cuda().eval()
    own_active = args[0].unit_active.clone()
    opponent_active = args[3].clone()
    own_active[0] = False
    opponent_active[0] = False
    original = args[0]._replace(unit_active=own_active)
    own = original.unit_continuous.clone()
    private = args[2].clone()
    own[~own_active] += 1000
    private[~opponent_active] += 1000
    own_cat = original.unit_categorical.clone()
    private_cat = args[1].clone()
    own_cat[~own_active] = 0
    private_cat[~opponent_active] = 0
    modified = original._replace(unit_continuous=own, unit_categorical=own_cat)
    read = _compiled(critic.encode_belief)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = read(original, args[1], args[2], opponent_active).value_decision
        after = read(modified, private_cat, private, opponent_active).value_decision
        live_private = args[2].clone()
        live_private[opponent_active] += 10
        changed = read(original, args[1], live_private, opponent_active).value_decision
    assert torch.isfinite(before).all()
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert not torch.equal(before[1:], changed[1:])


@pytest.mark.cuda
@cuda_only
def test_economic_critic_serialization_actor_privacy_and_nextlat_interface(config, native_batch):
    _, public, args, factors = native_batch
    configuration = replace(config, critic_architecture="economic")
    actor = EntityActor(config).cuda().eval()
    actor_economic = EntityActor(configuration).cuda().eval()
    actor_economic.load_state_dict(actor.state_dict(), strict=True)
    critic = EntityCritic(configuration).cuda().eval()
    with torch.no_grad():
        critic.value_head.weight.normal_(std=0.02)
    assert {p.data_ptr() for p in actor.parameters()}.isdisjoint(
        p.data_ptr() for p in critic.parameters()
    )
    serialized = io.BytesIO()
    torch.save({"config": configuration.to_dict(), "critic": critic.state_dict()}, serialized)
    serialized.seek(0)
    payload = torch.load(serialized, weights_only=True, map_location="cuda")
    restored = EntityCritic(EntityConfig(**payload["config"])).cuda().eval()
    restored.load_state_dict(payload["critic"], strict=True)
    dynamics = StructuredCriticDynamics(configuration).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for left, right in zip(
            _compiled(actor)(public), _compiled(actor_economic)(public), strict=True
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        expected, belief = _compiled(critic.forward_with_belief)(*args)
        actual = _compiled(restored)(*args)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        predicted = _compiled(dynamics)(
            belief,
            factors["unit_actions"],
            factors["market_kinds"],
            factors["market_quantities"],
            public.unit_categorical,
            public.unit_active,
        )
        logits = _compiled(critic.decode_belief)(predicted)
    assert predicted.value_decision.shape == belief.value_decision.shape
    assert logits.shape == expected.shape and torch.isfinite(logits).all()
