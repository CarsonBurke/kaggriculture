from __future__ import annotations

import pytest
import torch
from kaggle_environments import make

from kaggriculture.actions import N_MARKET_KINDS, N_UNIT_ACTIONS
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.structured import (
    StructuredActor,
    StructuredConfig,
    StructuredCritic,
    StructuredInputs,
    stack_structured,
)
from kaggriculture.tokens import encode_structured_observation


def _tiny_config() -> StructuredConfig:
    return StructuredConfig(
        model_dim=32,
        attention_heads=2,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
    )


@pytest.fixture(scope="module")
def real_pairs() -> list[tuple[dict, dict]]:
    environment = make("kaggriculture", configuration={"episodeSteps": 30, "seed": 3})
    environment.run(["starter", "starter"])
    return [
        (
            environment.steps[step][seat].observation,
            environment.steps[step][1 - seat].observation,
        )
        for step in (1, 25)
        for seat in (0, 1)
    ]


@pytest.fixture(scope="module")
def real_inputs(real_pairs: list[tuple[dict, dict]]) -> StructuredInputs:
    rows = [encode_structured_observation(observation) for observation, _ in real_pairs]
    inputs, extras = stack_structured(rows)
    assert extras is None
    return inputs


def test_structured_actor_preserves_the_output_contract(real_inputs: StructuredInputs) -> None:
    torch.manual_seed(0)
    actor = StructuredActor(_tiny_config())

    output = actor(real_inputs)

    batch = real_inputs.tile_categorical.shape[0]
    assert output.unit_logits.shape == (batch, MAX_UNITS, N_UNIT_ACTIONS)
    assert output.market_kind_logits.shape == (batch, MAX_MARKET_ORDERS, N_MARKET_KINDS)
    assert output.market_quantity_context.shape == (batch, MAX_MARKET_ORDERS, 32)
    assert torch.isfinite(output.unit_logits).all()
    assert torch.isfinite(output.market_kind_logits).all()

    kinds = output.market_kind_logits.argmax(dim=-1)
    quantities = actor.quantity_logits(output.market_quantity_context, kinds)
    assert quantities.shape == (batch, MAX_MARKET_ORDERS, 100)
    assert torch.isfinite(quantities).all()

    # Inactive unit slots produce exactly the head bias, not model garbage.
    inactive = ~real_inputs.unit_active
    assert inactive.any()
    bias_logits = actor.unit_head(torch.zeros(1, 1, actor.config.model_dim))
    expanded = bias_logits.expand_as(output.unit_logits)
    assert torch.allclose(output.unit_logits[inactive], expanded[inactive])


def test_structured_actor_shares_the_head_bias_prior_with_farm_actor() -> None:
    torch.manual_seed(0)
    structured = StructuredActor(_tiny_config())
    convolutional = FarmActor(
        ModelConfig(
            cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
        )
    )

    assert torch.equal(structured.unit_head[-1].bias, convolutional.unit_head[-1].bias)
    assert torch.equal(structured.market_kind.bias, convolutional.market_kind.bias)
    assert torch.equal(structured.market_quantity_bias, convolutional.market_quantity_bias)


def test_structured_actor_gradients_reach_every_input_family(
    real_inputs: StructuredInputs,
) -> None:
    torch.manual_seed(0)
    actor = StructuredActor(_tiny_config())

    output = actor(real_inputs)
    loss = (
        output.unit_logits[real_inputs.unit_active].sum()
        + output.market_kind_logits.sum()
        + output.market_quantity_context.sum()
    )
    loss.backward()

    # Inactive units run the local-tile decoder with no valid gathers; a
    # fully masked SDPA row would emit NaN and poison every shared gradient
    # even though the forward values are discarded. All gradients must stay
    # finite with real observations (15 of 16 slots inactive).
    for name, parameter in actor.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"

    reached = {
        "tiles": actor.trunk.tiles.continuous[0].weight.grad,
        "tile_kind": actor.trunk.tiles.kind.weight.grad,
        "units": actor.trunk.units.continuous[0].weight.grad,
        "economy": actor.trunk.economy.product_projection.weight.grad,
        "town": actor.trunk.economy.town_projection.weight.grad,
        "opponent": actor.trunk.opponent_queries.grad,
        "latents": actor.trunk.latent_queries.grad,
        "core": actor.trunk.core[0].ffn.input.weight.grad,
    }
    for name, gradient in reached.items():
        assert gradient is not None and gradient.abs().sum() > 0, f"no gradient into {name}"


def test_structured_critic_reads_both_private_states(
    real_pairs: list[tuple[dict, dict]],
) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    critic = StructuredCritic(config)

    rows = [
        encode_structured_observation(observation, opponent["private"])
        for observation, opponent in real_pairs
    ]
    stacked, extras = stack_structured(rows)
    assert extras is not None
    assert extras.products.shape[-1] == 2 and extras.crops.shape[-1] == 1
    batch = stacked.tile_categorical.shape[0]
    inputs = stacked._replace(
        products=torch.cat((stacked.products, extras.products), dim=-1),
        crops=torch.cat((stacked.crops, extras.crops), dim=-1),
    )

    logits = critic(inputs, extras.unit_categorical, extras.unit_continuous, extras.unit_active)

    assert logits.shape == (batch, config.value_atoms)
    assert torch.isfinite(logits).all()
    # Zero-initialized head starts at the uniform distribution: value 0.
    assert float(critic.value(logits).detach().abs().max()) == pytest.approx(0.0, abs=1e-5)

    # Opponent private inputs must actually reach the value estimate. The
    # zero-initialized head blocks trunk gradients at init, so perturb it.
    torch.nn.init.normal_(critic.value_head.weight, std=0.01)
    second = critic(inputs, extras.unit_categorical, extras.unit_continuous, extras.unit_active)
    second.sum().backward()
    gate = critic.trunk.units.farm.weight.grad
    assert gate is not None and gate[1].abs().sum() > 0


def test_structured_config_validation() -> None:
    with pytest.raises(ValueError, match="attention head width"):
        StructuredConfig(model_dim=24, attention_heads=4)
    with pytest.raises(ValueError, match="latents"):
        StructuredConfig(latents=0)
    round_trip = StructuredConfig(**StructuredConfig().to_dict())
    assert round_trip == StructuredConfig()
