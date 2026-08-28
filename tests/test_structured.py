from __future__ import annotations

import argparse
from dataclasses import replace

import pytest
import torch
from kaggle_environments import make

from kaggriculture.actions import N_MARKET_KINDS, N_UNIT_ACTIONS
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.modelargs import add_model_config_arguments, model_config_from_args
from kaggriculture.registry import resolve_architecture
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


def test_structured_actor_exposes_typed_training_belief(
    real_inputs: StructuredInputs,
) -> None:
    actor = StructuredActor(_tiny_config())

    output, belief = actor.forward_with_belief(real_inputs)

    assert output.unit_logits.shape[:2] == (real_inputs.unit_active.shape)
    assert belief.own_patches.shape == (real_inputs.unit_active.shape[0], 100, 32)
    assert belief.opponent_patches.shape == belief.own_patches.shape
    assert belief.opponent_summary.shape == (real_inputs.unit_active.shape[0], 4, 32)
    assert belief.central_latents.shape == (real_inputs.unit_active.shape[0], 8, 32)
    assert belief.unit_decisions.shape == (real_inputs.unit_active.shape[0], 16, 32)
    assert belief.market_decisions.shape == (real_inputs.unit_active.shape[0], 10, 32)


@pytest.mark.parametrize(
    "changes",
    [
        {"global_refresh_layers": (1,), "global_refresh_context": "economy"},
        {"global_refresh_layers": (1,), "global_refresh_context": "all"},
        {"input_reinject_layers": (1,)},
        {"core_skip_source": 1, "core_skip_target": 2},
        {"global_modulation": True},
        {"mudd_lite": True, "core_layers": 6},
    ],
)
def test_zero_initialized_transport_paths_begin_as_noops(
    real_inputs: StructuredInputs,
    changes: dict,
) -> None:
    torch.manual_seed(7)
    baseline = StructuredActor(replace(_tiny_config(), core_layers=changes.get("core_layers", 2)))
    torch.manual_seed(11)
    candidate = StructuredActor(replace(baseline.config, **changes))
    common = {
        name: value
        for name, value in baseline.state_dict().items()
        if name in candidate.state_dict() and candidate.state_dict()[name].shape == value.shape
    }
    candidate.load_state_dict(common, strict=False)

    expected = baseline(real_inputs)
    actual = candidate(real_inputs)

    torch.testing.assert_close(actual.unit_logits, expected.unit_logits)
    torch.testing.assert_close(actual.market_kind_logits, expected.market_kind_logits)
    torch.testing.assert_close(
        actual.market_quantity_context,
        expected.market_quantity_context,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"fuse_market_decoder": True},
        {"fuse_unit_decoder": True},
        {"split_clock_token": True},
        {"zero_init_branches": True},
    ],
)
def test_structured_variants_preserve_finite_output_contract(
    real_inputs: StructuredInputs,
    changes: dict,
) -> None:
    actor = StructuredActor(replace(_tiny_config(), **changes))

    output = actor(real_inputs)

    assert torch.isfinite(output.unit_logits).all()
    assert torch.isfinite(output.market_kind_logits).all()
    assert torch.isfinite(output.market_quantity_context).all()


def test_structured_model_arguments_parse_typed_regression_fields() -> None:
    parser = argparse.ArgumentParser()
    add_model_config_arguments(parser)
    args = parser.parse_args(
        [
            "--global-refresh-layers",
            "2,5",
            "--global-refresh-context",
            "all",
            "--zero-init-branches",
            "true",
        ]
    )

    config = model_config_from_args(resolve_architecture("structured"), args)

    assert config.global_refresh_layers == (2, 5)
    assert config.global_refresh_context == "all"
    assert config.zero_init_branches is True

def test_structured_farm_batch_matches_separate_canonical_encoding(
    real_inputs: StructuredInputs,
) -> None:
    torch.manual_seed(0)
    actor = StructuredActor(_tiny_config())
    trunk = actor.trunk
    tiles = trunk.tiles(real_inputs.tile_categorical, real_inputs.tile_continuous)
    batch = tiles.shape[0]
    board = torch.stack(
        torch.meshgrid(
            torch.arange(10),
            torch.arange(10),
            indexing="ij",
        )[::-1],
        dim=-1,
    ).reshape(1, 100, 2)
    reference_rotation = trunk.rope.rotation(board.expand(batch, -1, -1))

    def separately(farm: torch.Tensor) -> torch.Tensor:
        hidden = farm
        for block in trunk.farm_local:
            hidden = block(
                hidden,
                query_rotation=reference_rotation,
                key_rotation=reference_rotation,
            )
        return hidden

    expected = (
        separately(tiles[:, :100]),
        separately(tiles[:, 100:]),
    )
    batched_rotation = (
        trunk.rope.cosine.view(1, 1, 100, -1).expand(batch * 2, -1, -1, -1),
        trunk.rope.sine.view(1, 1, 100, -1).expand(batch * 2, -1, -1, -1),
    )
    actual = trunk.encode_farms(tiles, batched_rotation)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

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
