from __future__ import annotations

import copy
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from kaggle_environments import make
from torch import Tensor

from kaggriculture.actions import MarketKind
from kaggriculture.actor_dynamics import ActorDynamics
from kaggriculture.compilewatch import CompileWatch
from kaggriculture.constants import EPISODE_STEPS, MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.lejepa import (
    JEPA_GROUPS,
    JEPA_METRICS,
    JepaObjective,
    JepaPersistenceControl,
    JepaShuffledControl,
    JepaTerms,
    SIGReg,
    belief_groups,
    jepa_horizon_loss,
    sigreg_directions,
)
from kaggriculture.lejepa_model import (
    LejepaActor,
    LejepaBackbone,
    LejepaConfig,
    LejepaCritic,
    build_lejepa_pair,
)
from kaggriculture.optim import route_parameters
from kaggriculture.ppo import (
    BACKBONE_ROLE,
    PpoConfig,
    _stage_tensor,
    _validate_config,
    _validate_structured_auxiliary_modules,
    _validate_structured_critic_auxiliary_modules,
    actor_forward_args,
    make_optimizers,
    make_structured_dynamics_optimizer,
    replay_behavior_values,
    set_lr_cooldown,
    update_ppo,
)
from kaggriculture.provenance import source_identity
from kaggriculture.registry import LEJEPA, architecture_of_config, resolve_architecture
from kaggriculture.rollout import (
    behavior_value_key,
    collect_mixed_play_rust,
    collect_self_play,
    collect_self_play_rust,
    concatenate_rollouts,
    slice_trajectories,
)
from kaggriculture.structured import (
    JepaBelief,
    StructuredCriticBelief,
    StructuredInputs,
    stack_structured,
)
from kaggriculture.structured_dynamics import StructuredCriticDynamics, structured_horizon_plan
from kaggriculture.tokens import (
    PRODUCT_TOKEN_FIELDS,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TILE_COUNT,
    animal_token_fields,
    crop_token_fields,
    encode_structured_observation,
    farm_token_fields,
    product_private_fields,
    product_token_fields,
    town_token_fields,
)

_ROWS_PER_SEAT = 4


def _tiny_config(**overrides) -> LejepaConfig:
    base = dict(
        # These contracts isolate the original readout/world-model family;
        # destination/resource/affordance-head contracts live in their
        # dedicated tests.
        unit_target_navigation=False,
        market_resource_conditioning=False,
        unit_affordance_scorer=False,
        model_dim=32,
        attention_heads=2,
        attention_kv_heads=1,
        ffn_multiplier=2,
        farm_blocks=1,
        core_layers=2,
        jepa_hidden_dim=48,
        jepa_slices=16,
        jepa_tile_samples=8,
        jepa_sigreg_rows=64,
    )
    base.update(overrides)
    return LejepaConfig(**base)


@pytest.fixture(scope="module")
def real_observations() -> list[tuple[dict, dict]]:
    environment = make("kaggriculture", configuration={"episodeSteps": 30, "seed": 3})
    environment.run(["starter", "starter"])
    # Seat-major, so consecutive rows are consecutive steps of one episode. Any
    # other order makes neighbouring rows the two seats of one step, whose early
    # observations are byte-identical and whose prediction loss is a fake zero.
    return [
        (
            environment.steps[step][seat].observation,
            environment.steps[step][1 - seat].observation,
        )
        for seat in (0, 1)
        for step in range(1, _ROWS_PER_SEAT + 1)
    ]


@pytest.fixture(scope="module")
def actor_inputs(real_observations: list[tuple[dict, dict]]) -> StructuredInputs:
    rows = [encode_structured_observation(own) for own, _ in real_observations]
    inputs, extras = stack_structured(rows)
    assert extras is None
    return inputs


@pytest.fixture(scope="module")
def critic_inputs(
    real_observations: list[tuple[dict, dict]],
) -> tuple[StructuredInputs, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows = [encode_structured_observation(own, other) for own, other in real_observations]
    inputs, extras = stack_structured(rows)
    assert extras is not None
    # `_critic_batch_args` concatenates the private economy columns; the critic
    # trunk's embedder is built for the concatenated width.
    inputs = inputs._replace(
        products=torch.cat((inputs.products, extras.products), dim=-1),
        animals=torch.cat((inputs.animals, extras.animals), dim=-1),
        crops=torch.cat((inputs.crops, extras.crops), dim=-1),
    )
    opponent = (extras.unit_categorical, extras.unit_continuous, extras.unit_active)
    return inputs, opponent


@pytest.fixture(scope="module")
def factors() -> dict[str, torch.Tensor]:
    rows = 2 * _ROWS_PER_SEAT
    generator = torch.Generator().manual_seed(11)
    episode = np.repeat(np.arange(2), _ROWS_PER_SEAT)
    step = np.tile(np.arange(_ROWS_PER_SEAT), 2)
    rewards = torch.zeros(rows)
    # Terminal-outcome shaped: the whole signal sits on each episode's last row,
    # which is never a transition source.
    rewards[_ROWS_PER_SEAT - 1] = 1.0
    rewards[-1] = -1.0
    return {
        "unit_actions": torch.randint(0, 4, (rows, MAX_UNITS), generator=generator),
        "market_kinds": torch.randint(0, 3, (rows, MAX_MARKET_ORDERS), generator=generator),
        "market_quantities": torch.randint(0, 3, (rows, MAX_MARKET_ORDERS), generator=generator),
        "episode_index": torch.from_numpy(episode),
        "step": torch.from_numpy(step),
        "rewards": rewards,
    }


@pytest.fixture(scope="module")
def plan():
    episode = np.repeat(np.arange(2), _ROWS_PER_SEAT)
    step = np.tile(np.arange(_ROWS_PER_SEAT), 2)
    return structured_horizon_plan(episode, step, 1)


# --------------------------------------------------------------------------
# SIGReg
# --------------------------------------------------------------------------


def _statistic(samples: torch.Tensor, slices: int = 64, seed: int = 0) -> float:
    directions = torch.from_numpy(
        sigreg_directions(samples.shape[1], slices, np.random.default_rng(seed))
    )
    return float(SIGReg()(samples, directions))


def test_sigreg_vanishes_on_a_standard_gaussian_and_rejects_every_departure() -> None:
    torch.manual_seed(0)
    reference = torch.randn(4096, 16)

    standard = _statistic(reference)
    scaled = _statistic(reference * 3.0)
    shifted = _statistic(reference + 2.0)
    constant = _statistic(torch.zeros(4096, 16))

    assert standard < 1e-3
    # Every departure the objective exists to reject is orders above the null.
    assert scaled > 100 * standard
    assert shifted > 100 * standard
    assert constant > 100 * standard


def test_sigreg_is_invariant_to_the_sample_count() -> None:
    """The reference's ``* n`` factor is dropped, so one coefficient serves every
    minibatch size, token count and unit occupancy."""
    torch.manual_seed(0)
    small = torch.randn(512, 16) * 2.0
    large = small.repeat(8, 1)

    assert _statistic(small) == pytest.approx(_statistic(large), rel=1e-5)


def test_sigreg_weight_selects_exactly_the_unmasked_subset() -> None:
    torch.manual_seed(0)
    samples = torch.randn(256, 8) * 1.7
    keep = torch.zeros(256, dtype=torch.bool)
    keep[::3] = True
    directions = torch.from_numpy(sigreg_directions(8, 32, np.random.default_rng(1)))
    statistic = SIGReg()

    masked = statistic(samples, directions, keep)
    subset = statistic(samples[keep], directions)

    assert float(masked) == pytest.approx(float(subset), rel=1e-5)


def test_sigreg_gradient_drives_a_degenerate_cloud_onto_the_standard_normal() -> None:
    torch.manual_seed(0)
    samples = (torch.randn(1024, 8) * 0.05 + 1.0).requires_grad_(True)
    directions = torch.from_numpy(sigreg_directions(8, 64, np.random.default_rng(2)))
    statistic = SIGReg()
    optimizer = torch.optim.Adam([samples], lr=0.05)

    for _ in range(1500):
        optimizer.zero_grad()
        statistic(samples, directions).backward()
        optimizer.step()

    settled = samples.detach()
    assert float(settled.mean().abs()) < 0.05
    assert float(settled.std()) == pytest.approx(1.0, abs=0.1)


def test_sigreg_chunking_matches_a_single_pass() -> None:
    from kaggriculture import lejepa

    torch.manual_seed(0)
    samples = torch.randn(4000, 8) * 1.3
    directions = torch.from_numpy(sigreg_directions(8, 8, np.random.default_rng(3)))
    statistic = SIGReg()

    whole = float(statistic(samples, directions))
    budget = lejepa.SIGREG_CHUNK_ELEMENTS
    try:
        lejepa.SIGREG_CHUNK_ELEMENTS = 8 * SIGReg().t.numel()  # ~8 rows per chunk
        chunked = float(statistic(samples, directions))
    finally:
        lejepa.SIGREG_CHUNK_ELEMENTS = budget

    assert chunked == pytest.approx(whole, rel=1e-5)


def test_sigreg_refuses_malformed_arguments() -> None:
    statistic = SIGReg()
    with pytest.raises(ValueError):
        statistic(torch.randn(4, 3, 2, 2), torch.randn(2, 4))
    with pytest.raises(ValueError):
        statistic(torch.randn(4, 3), torch.randn(5, 4))
    with pytest.raises(ValueError):
        statistic(torch.randn(4, 3, 2), torch.randn(3, 4))


def test_sigreg_rejects_the_slot_constant_cloud_a_pooled_test_would_pass() -> None:
    """The statistic runs per token position, as `../le-wm/module.py` does.

    An encoder that hands every token slot its own constant is degenerate -- its
    prediction loss is exactly zero -- while the union of those atoms is an
    ordinary cloud. `TileEmbedder.slot_embedding` alone can express it, so this is
    a reachable collapse basin, and flattening the token axis into the sample axis
    is what would leave it open.
    """
    generator = torch.Generator().manual_seed(11)
    features, tokens, rows, slices = 24, 32, 512, 64
    directions = torch.from_numpy(sigreg_directions(features, slices, np.random.default_rng(3)))
    statistic = SIGReg()
    normal = torch.randn(rows, tokens, features, generator=generator)
    atoms = torch.randn(1, tokens, features, generator=generator).expand(rows, -1, -1).contiguous()

    assert float(statistic(normal, directions)) < 1e-2
    # Pooled, the atoms cost a twentieth of what they cost per position -- and
    # per position they cost more than total collapse, which is the right order.
    assert float(statistic(atoms, directions)) > 10.0 * float(
        statistic(atoms.reshape(-1, features), directions)
    )
    assert float(statistic(atoms, directions)) > float(
        statistic(torch.zeros(rows, tokens, features), directions)
    )


def test_sigreg_ignores_token_positions_no_row_supervises() -> None:
    """A position with no weight is dropped, not scored against an empty sample."""
    generator = torch.Generator().manual_seed(4)
    features, tokens, rows, slices = 16, 8, 256, 32
    directions = torch.from_numpy(sigreg_directions(features, slices, np.random.default_rng(1)))
    statistic = SIGReg()
    values = torch.randn(rows, tokens, features, generator=generator)
    weight = torch.ones(rows, tokens)
    weight[:, 0] = 0.0

    assert float(statistic(values, directions, weight)) == pytest.approx(
        float(statistic(values[:, 1:], directions)), rel=1e-5
    )


def test_sigreg_is_flat_in_how_many_rows_support_a_token_position() -> None:
    """Positions combine in proportion to their support, not equally.

    A position's statistic falls like ``1 / n_position``, so an equally weighted
    mean lets one thinly supported slot -- a unit the agent has just learned to
    hire -- read as collapse on sampling noise alone, and the gradient that
    follows pulls that slot toward the origin. Support weighting is the
    reference's ``* n`` restored per position and divided out once at the end.
    """
    generator = torch.Generator().manual_seed(17)
    features, tokens, rows, slices = 32, 16, 1024, 128
    directions = torch.from_numpy(sigreg_directions(features, slices, np.random.default_rng(7)))
    statistic = SIGReg()
    values = torch.randn(rows, tokens, features, generator=generator)
    full = float(statistic(values, directions))
    collapsed = float(statistic(torch.zeros(rows, tokens, features), directions))

    for supported in (1, 2, 5, 100):
        weight = torch.ones(rows, tokens)
        weight[supported:, 0] = 0.0
        thin = float(statistic(values, directions, weight))
        # Within a few percent of the fully supported null, and nowhere near the
        # collapse it would otherwise be mistaken for.
        assert thin == pytest.approx(full, rel=0.1)
        assert thin < 0.05 * collapsed


def test_slice_directions_are_unit_norm_and_stream_bound() -> None:
    columns = sigreg_directions(12, 7, np.random.default_rng(5))

    assert columns.shape == (12, 7)
    assert columns.dtype == np.float32
    assert np.allclose(np.linalg.norm(columns, axis=0), 1.0, atol=1e-5)
    # Same seed, same draw: the objective replays from the run's seed alone.
    assert np.array_equal(columns, sigreg_directions(12, 7, np.random.default_rng(5)))
    with pytest.raises(ValueError):
        sigreg_directions(0, 4, np.random.default_rng(5))


# --------------------------------------------------------------------------
# The objective's buffers
# --------------------------------------------------------------------------


def test_refresh_slices_moves_values_without_moving_buffers() -> None:
    objective = JepaObjective(_tiny_config())
    pointers = {name: objective.directions(name).data_ptr() for name in objective.groups}
    tile_pointer = objective.tile_index.data_ptr()
    before = {name: objective.directions(name).clone() for name in objective.groups}
    before_tiles = objective.tile_index.clone()

    objective.refresh_slices(np.random.default_rng(31))

    for name in objective.groups:
        current = objective.directions(name)
        # Buffer identity, shape and dtype are what keep the compiled update
        # graph from retracing after a refresh.
        assert current.data_ptr() == pointers[name]
        assert current.shape == before[name].shape
        assert not torch.equal(current, before[name])
        assert torch.allclose(current.norm(dim=0), torch.ones(objective.slices), atol=1e-5)
    assert objective.tile_index.data_ptr() == tile_pointer
    assert not torch.equal(objective.tile_index, before_tiles)
    assert torch.equal(objective.tile_index.sort().values, objective.tile_index)
    assert objective.tile_index.unique().numel() == objective.tile_samples
    assert int(objective.tile_index.max()) < 2 * TILE_COUNT


def test_objective_buffers_stay_out_of_the_state_dict() -> None:
    objective = JepaObjective(_tiny_config())

    keys = set(objective.state_dict())

    assert not any(key.startswith("directions_") or key == "tile_index" for key in keys)
    assert objective.groups == JEPA_GROUPS


def test_objective_refuses_a_tile_sample_wider_than_the_board() -> None:
    with pytest.raises(ValueError):
        LejepaConfig(jepa_tile_samples=2 * TILE_COUNT + 1)
    with pytest.raises(ValueError):
        LejepaConfig(jepa_slices=0)
    with pytest.raises(ValueError):
        LejepaConfig(bixt_latents=4)


# --------------------------------------------------------------------------
# The model family
# --------------------------------------------------------------------------


def test_registry_round_trips_the_family(actor_inputs: StructuredInputs) -> None:
    family = resolve_architecture(LEJEPA)

    assert family.structured_inputs
    assert family.config_class is LejepaConfig
    assert family.actor_class is LejepaActor
    assert family.critic_class is LejepaCritic
    assert family.actor_belief_class is JepaBelief
    # The critic decides from a pooled state it builds itself on top of the
    # shared backbone; it contributes nothing to the world-model objective.
    assert family.critic_belief_class is StructuredCriticBelief
    assert architecture_of_config(_tiny_config()).name == LEJEPA


def test_actor_belief_carries_the_encoded_observation(actor_inputs: StructuredInputs) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    actor = LejepaActor(config)
    rows = actor_inputs.tile_categorical.shape[0]

    belief = actor.encode_belief(actor_inputs)

    assert isinstance(belief, JepaBelief)
    assert belief.unit_decisions.shape == (rows, MAX_UNITS, config.model_dim)
    assert belief.market_decisions.shape == (rows, MAX_MARKET_ORDERS, config.model_dim)
    assert belief.tiles.shape == (rows, 2 * TILE_COUNT, config.model_dim)
    assert belief.economy.shape[0] == rows and belief.economy.shape[1] > 0
    assert all(torch.isfinite(value).all() for value in belief)


def test_decoding_the_belief_reproduces_the_forward_pass(actor_inputs: StructuredInputs) -> None:
    torch.manual_seed(0)
    actor = LejepaActor(_tiny_config()).eval()

    with torch.no_grad():
        direct = actor(actor_inputs)
        decoded = actor.decode_belief(actor.encode_belief(actor_inputs), actor_inputs.unit_active)

    assert torch.allclose(direct.unit_logits, decoded.unit_logits, atol=1e-5)
    assert torch.allclose(direct.market_kind_logits, decoded.market_kind_logits, atol=1e-5)
    assert torch.allclose(
        direct.market_quantity_context, decoded.market_quantity_context, atol=1e-5
    )


def _perturbed(belief: JepaBelief, field: str) -> JepaBelief:
    value = getattr(belief, field)
    return belief._replace(**{field: value + torch.randn_like(value)})


@pytest.mark.parametrize("field", ["economy", "tiles"])
def test_decision_slots_read_the_encoded_observation(actor_inputs, field) -> None:
    """The readout is how a decision reaches what the world model kept elsewhere.

    Without rounds each slot is decoded from itself alone, so the economy and
    the farm, which the world model keeps in their own tokens, never reach a
    market or a unit logit. That is the linear-probe control; per-slot heads,
    even with a gated feed-forward, cloned market kinds to only 0.76.
    """
    torch.manual_seed(0)
    active = actor_inputs.unit_active
    for layers, moves in ((1, True), (0, False)):
        actor = LejepaActor(_tiny_config(policy_readout_layers=layers)).eval()
        with torch.no_grad():
            belief = actor.encode_belief(actor_inputs)
            base = actor.decode_belief(belief, active)
            other = actor.decode_belief(_perturbed(belief, field), active)
        changed = not torch.allclose(base.market_kind_logits, other.market_kind_logits)
        assert changed is moves
        assert (not torch.equal(base.unit_logits[active], other.unit_logits[active])) is moves


def test_inactive_unit_slots_are_invisible_to_the_readout(actor_inputs) -> None:
    torch.manual_seed(0)
    active = actor_inputs.unit_active
    assert (~active).any() and active.any()
    actor = LejepaActor(_tiny_config()).eval()
    with torch.no_grad():
        belief = actor.encode_belief(actor_inputs)
        base = actor.decode_belief(belief, active)
        noise = torch.randn_like(belief.unit_decisions) * (~active).unsqueeze(-1)
        other = actor.decode_belief(
            belief._replace(unit_decisions=belief.unit_decisions + noise), active
        )

    torch.testing.assert_close(other.market_kind_logits, base.market_kind_logits)
    torch.testing.assert_close(other.unit_logits[active], base.unit_logits[active])


@pytest.mark.parametrize("layers", [-1, True])
def test_readout_depth_is_validated(layers) -> None:
    with pytest.raises(ValueError, match="policy_readout_layers"):
        _tiny_config(policy_readout_layers=layers)


def test_the_critic_reads_the_shared_backbone_and_owns_no_encoder(critic_inputs) -> None:
    torch.manual_seed(0)
    inputs, opponent = critic_inputs
    config = _tiny_config()
    actor, critic = build_lejepa_pair(config)
    rows = inputs.tile_categorical.shape[0]

    logits, belief = critic.forward_with_belief(inputs, *opponent)

    assert isinstance(belief, StructuredCriticBelief)
    assert logits.shape[0] == rows
    assert belief.value_decision.shape == (rows, 1, config.model_dim)
    # One backbone: the critic holds the actor's by reference, and carries
    # neither a copy of it in its state dict nor its parameters in its own set.
    assert critic.backbone is actor.trunk
    assert not any("trunk" in key for key in critic.state_dict())
    backbone = {id(parameter) for parameter in actor.backbone_parameters()}
    assert backbone and not (backbone & {id(p) for p in critic.parameters()})


def test_a_handed_backbone_belief_replaces_the_critic_pass_exactly(
    actor_inputs, critic_inputs
) -> None:
    """The update hands the critic the actor's encoding instead of a second pass.

    Same module, same public inputs, same weights: the value must be identical,
    and the handed belief must be read detached, so no value gradient reaches the
    backbone even when the caller's copy is attached.
    """
    torch.manual_seed(0)
    inputs, opponent = critic_inputs
    actor, critic = build_lejepa_pair(_tiny_config())
    critic.value_head.weight.data.normal_()

    encoded = actor.auxiliary_belief(actor_inputs)
    handed = critic(inputs, *opponent, encoded)
    own = critic(inputs, *opponent)
    handed.sum().backward()

    assert torch.equal(handed, own)
    assert all(parameter.grad is None for parameter in actor.backbone_parameters())
    assert critic.value_head.weight.grad is not None


def test_an_unattached_critic_refuses_rather_than_encoding_something_else(critic_inputs) -> None:
    """A critic loaded from a checkpoint is inert until it is given a backbone."""
    inputs, opponent = critic_inputs
    critic = LejepaCritic(_tiny_config())

    with pytest.raises(RuntimeError, match="no backbone"):
        critic(inputs, *opponent)

    with pytest.raises(TypeError):
        critic.attach_backbone(LejepaActor(_tiny_config()))
    with pytest.raises(ValueError, match="model configuration"):
        critic.attach_backbone(LejepaBackbone(_tiny_config(jepa_slices=8)))


@pytest.mark.parametrize("optimizer", ["adamw", "normuon", "fused"])
def test_the_behavior_value_key_moves_with_every_weight_the_value_reads(optimizer) -> None:
    """A collected value is reused only while this key stands still.

    The critic's own tower and the backbone it borrows both feed the value, and
    they step in two different optimizers; a checkpoint load writes both. Any of
    them left out of the key would let the update read values from weights it no
    longer holds. A fused AdamW -- what CUDA builds -- writes its parameters
    without advancing their version counters, so the update's step does.
    """
    import kaggriculture.ppo as ppo_module

    torch.manual_seed(0)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    config = PpoConfig(
        optimizer="adamw" if optimizer == "fused" else optimizer,
        lr_warmup_steps=0,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
    )
    _, critic_optimizer = make_optimizers(actor, critic, config)
    backbone_optimizer = make_structured_dynamics_optimizer(
        JepaObjective(model_config), config, critic=False, actor=actor
    )
    if optimizer == "fused":
        critic_optimizer = torch.optim.AdamW(critic.parameters(), fused=True)
        backbone_optimizer = torch.optim.AdamW(actor.backbone_parameters(), fused=True)
    key = behavior_value_key(critic, True)

    assert behavior_value_key(critic, True) == key
    assert behavior_value_key(critic, False) != key
    assert behavior_value_key(build_lejepa_pair(model_config)[1], True) != key

    for stepped, parameters in (
        (critic_optimizer, list(critic.parameters())),
        (backbone_optimizer, list(actor.backbone_parameters())),
    ):
        before = behavior_value_key(critic, True)
        weights = [parameter.detach().clone() for parameter in parameters]
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
        ppo_module._optimizer_step(stepped, 1.0e-3, 0)
        assert any(
            not torch.equal(parameter, weight)
            for parameter, weight in zip(parameters, weights, strict=True)
        )
        assert behavior_value_key(critic, True) != before

    before = behavior_value_key(critic, True)
    actor.load_state_dict(actor.state_dict())
    assert behavior_value_key(critic, True) != before


def _carried_rollout(critic: LejepaCritic):
    """A small wave that claims values read at `critic`'s current weights."""
    rollout = collect_self_play(
        LejepaActor(_tiny_config()),
        games=1,
        seed_start=227,
        episode_steps=6,
        sampling_seed=19,
        reward_mode="terminal-outcome",
    )
    values = np.arange(rollout.rewards.size, dtype=np.float32).reshape(rollout.rewards.shape)
    return replace(
        rollout, behavior_values=values, behavior_value_key=behavior_value_key(critic, False)
    )


def test_the_update_reads_collected_values_only_at_the_weights_they_were_read_at(
    monkeypatch,
) -> None:
    import kaggriculture.ppo as ppo_module

    torch.manual_seed(0)
    _, critic = build_lejepa_pair(_tiny_config())
    rollout = _carried_rollout(critic)
    replays: list[dict] = []

    def replay(*args, **kwargs):
        replays.append(kwargs)
        rows = rollout.rewards.size if kwargs.get("states") is None else kwargs["states"].numel()
        return torch.full((rows,), -1.0)

    monkeypatch.setattr(ppo_module, "replay_behavior_values", replay)

    def owned(rows=None, *, autocast=False, entities=False):
        return ppo_module._owned_behavior_values(
            critic,
            LEJEPA,
            {"unit_actions": torch.zeros(rollout.rewards.size, 1)},
            rollout,
            rows,
            compile_mode="none",
            autocast_enabled=autocast,
            include_entities=entities,
        )

    assert owned() is rollout.behavior_values
    assert not replays
    # Each way the carried values stop describing what the update would compute.
    assert (owned(autocast=True) == -1.0).all()
    assert (owned(rows=np.asarray([0]))[0] == -1.0).all()
    assert (owned(entities=True) == -1.0).all()
    with torch.no_grad():
        critic.value_head.bias.add_(1.0)
    assert (owned() == -1.0).all()
    assert len(replays) == 4


def test_the_update_reports_whether_it_carried_the_collected_values() -> None:
    torch.manual_seed(0)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    plain = collect_self_play(
        actor,
        games=1,
        seed_start=229,
        episode_steps=6,
        sampling_seed=23,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
        update_compile_mode="eager",
        structured_learning_rate=1.0e-2,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
        jepa_reward_coefficient=0.1,
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    objective_optimizer = make_structured_dynamics_optimizer(
        objective, config, critic=False, actor=actor
    )

    def carried(rollout) -> int:
        metrics = update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(47),
            auxiliary_generator=np.random.default_rng(48),
            structured_dynamics=objective,
            structured_dynamics_optimizer=objective_optimizer,
            structured_actor_auxiliary=True,
        )
        return metrics["update_behavior_values_carried"]

    # Keyed before the first update: that update's steps move the key, so the
    # same wave offered again is replayed.
    keyed = replace(
        plain,
        behavior_values=np.zeros(plain.rewards.shape, dtype=np.float32),
        behavior_value_key=behavior_value_key(critic, False),
    )
    assert carried(keyed) == 1
    assert carried(keyed) == 0
    assert carried(plain) == 0


def test_collected_values_survive_a_slice_and_only_a_whole_merge() -> None:
    torch.manual_seed(0)
    _, critic = build_lejepa_pair(_tiny_config())
    rollout = _carried_rollout(critic)
    stopped = rollout.trajectories

    head = slice_trajectories(rollout, 0, 1)
    np.testing.assert_array_equal(head.behavior_values, rollout.behavior_values[:1])
    assert head.behavior_value_key == rollout.behavior_value_key

    merged = concatenate_rollouts([head, slice_trajectories(rollout, 1, stopped)])
    np.testing.assert_array_equal(merged.behavior_values, rollout.behavior_values)
    assert merged.behavior_value_key == rollout.behavior_value_key

    for other in (
        replace(head, behavior_values=None, behavior_value_key=None),
        replace(head, behavior_value_key=behavior_value_key(critic, True)),
    ):
        merged = concatenate_rollouts([other, slice_trajectories(rollout, 1, stopped)])
        assert merged.behavior_values is None and merged.behavior_value_key is None


def test_the_collector_reads_values_only_through_the_learner_backbone() -> None:
    """Values from a critic reading another encoder would be another critic's."""
    torch.manual_seed(0)
    actor, critic = build_lejepa_pair(_tiny_config())
    _, stranger = build_lejepa_pair(_tiny_config())

    def collect(value_critic):
        return collect_mixed_play_rust(
            actor,
            self_play_games=1,
            seed_start=229,
            sampling_seed=23,
            forward_mode="eager",
            critic=value_critic,
        )

    with pytest.raises(ValueError, match="backbone"):
        collect(stranger)
    # Off CUDA there is no captured step to ride on: the wave carries nothing
    # and the update replays, exactly as without a critic.
    carried = collect(critic)
    assert carried.behavior_values is None and carried.behavior_value_key is None
    assert critic.training


def test_the_critic_sees_its_privileged_inputs_and_the_actor_cannot(
    actor_inputs, critic_inputs
) -> None:
    """Opponent units and private economy columns enter strictly after the JEPA."""
    torch.manual_seed(0)
    inputs, opponent = critic_inputs
    actor, critic = build_lejepa_pair(_tiny_config())
    categorical, continuous, active = opponent

    with torch.no_grad():
        # The valuation state, not the value logits: `value_head` is
        # zero-initialized, so every logit is constant until it is trained.
        baseline = critic.encode_belief(inputs, *opponent).value_decision
        moved_units = critic.encode_belief(
            inputs, categorical, continuous + 1.0, active
        ).value_decision
        private = inputs.products.clone()
        private[..., actor_inputs.products.shape[-1] :] += 1.0
        moved_economy = critic.encode_belief(
            inputs._replace(products=private), *opponent
        ).value_decision
        # And the backbone reads the seat's own observation exactly: the critic's
        # widened tokens, trimmed, are the actor's own.
        public = critic._public_inputs(inputs)
        encoded = actor.encode_belief(public)
        own = actor.encode_belief(actor_inputs)

    assert active.any()
    assert not torch.allclose(baseline, moved_units)
    assert not torch.allclose(baseline, moved_economy)
    assert public.products.shape[-1] == actor_inputs.products.shape[-1]
    for left, right in zip(encoded, own, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize("schema", [4, 5, 6, 7, 8])
def test_a_pair_reads_only_its_schemas_columns(actor_inputs, critic_inputs, schema) -> None:
    """Saved pairs act and value bit-identically on tokens staged for newer schemas.

    Each schema's added product (public and private), animal, crop and farm
    columns move the actor and the critic exactly when the pair's schema
    reads them.
    """
    inputs, opponent = critic_inputs
    staged = len(PRODUCT_TOKEN_FIELDS)

    def added(fields, version: int) -> list[int]:
        return list(range(len(fields(version - 1)), len(fields(version))))

    def moved(tokens: StructuredInputs, version: int) -> StructuredInputs:
        products, farms = tokens.products.clone(), tokens.farms.clone()
        animals, crops = tokens.animals.clone(), tokens.crops.clone()
        products[..., added(product_token_fields, version)] += 1.0
        private = [staged + column for column in added(product_private_fields, version)]
        if products.shape[-1] > staged:
            products[..., private] += 1.0
        animals[..., added(animal_token_fields, version)] += 1.0
        crops[..., added(crop_token_fields, version)] += 1.0
        farms[..., added(farm_token_fields, version)] -= 1.0
        return tokens._replace(products=products, animals=animals, crops=crops, farms=farms)

    torch.manual_seed(0)
    actor, critic = build_lejepa_pair(_tiny_config(observation_schema_version=schema))
    with torch.no_grad():
        acted = actor.encode_belief(actor_inputs)
        valued = critic.encode_belief(inputs, *opponent).value_decision
        for version in (6, 7, 8):
            acted_moved = actor.encode_belief(moved(actor_inputs, version))
            valued_moved = critic.encode_belief(moved(inputs, version), *opponent).value_decision
            if schema >= version:
                assert not torch.equal(acted.economy, acted_moved.economy)
                assert not torch.equal(valued, valued_moved)
            else:
                for left, right in zip(acted, acted_moved, strict=True):
                    assert (left is None and right is None) or torch.equal(left, right)
                assert torch.equal(valued, valued_moved)


def _save_actor_artifact(path: Path, config: LejepaConfig) -> tuple[LejepaActor, JepaObjective]:
    actor, objective = LejepaActor(config), JepaObjective(config)
    torch.save(
        {
            "format_version": 17,
            "architecture": "lejepa",
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "structured_dynamics": objective.state_dict(),
            "iteration": 5,
            "source_identity": source_identity(),
            "run_provenance": None,
            "seed_usage": [{"domain": "online_rl", "start": 20_500_000, "count": 16}],
        },
        path,
    )
    return actor, objective


def _schema_upgrade(
    path: Path, inputs: StructuredInputs, source: int, target: int
) -> tuple[LejepaActor, LejepaActor, dict[str, StructuredInputs]]:
    """A schema-``source`` artifact, its warm start at ``target``, and their inputs.

    The artifact reads its own schema's encodings at their legacy widths
    (``legacy``); the upgraded actor reads the newest staged tokens as encoded
    (``staged``) and with every column the artifact's schema lacks redrawn as
    noise (``noised``).
    """
    torch.manual_seed(0)
    config = _tiny_config(observation_schema_version=source)
    pretrained, objective = _save_actor_artifact(path, config)
    upgraded_config = replace(config, observation_schema_version=target)
    actor, warm_objective = LejepaActor(upgraded_config), JepaObjective(upgraded_config)
    _training_script()._load_initial_actor(
        path, actor, "lejepa", upgraded_config, torch.device("cpu"), warm_objective
    )
    for key, value in objective.state_dict().items():
        assert torch.equal(warm_objective.state_dict()[key], value)

    widths = {
        "products": len(product_token_fields(source)),
        "animals": len(animal_token_fields(source)),
        "crops": len(crop_token_fields(source)),
        "farms": len(farm_token_fields(source)),
        "town": len(town_token_fields(source)),
    }
    assert any(getattr(inputs, name).shape[-1] > width for name, width in widths.items()), (
        "the upgrade adds no columns to read"
    )
    generator = torch.Generator().manual_seed(1)
    noised = {}
    for name, width in widths.items():
        tokens = getattr(inputs, name).clone()
        tokens[..., width:] = torch.randn(tokens[..., width:].shape, generator=generator)
        noised[name] = tokens
    return (
        pretrained.eval(),
        actor.eval(),
        {
            "legacy": inputs._replace(
                **{
                    name: getattr(inputs, name)[..., :width].contiguous()
                    for name, width in widths.items()
                }
            ),
            "staged": inputs,
            "noised": inputs._replace(**noised),
        },
    )


_SCHEMA_UPGRADES = [
    (source, target)
    for source in sorted(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
    for target in sorted(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
    if source < target
]


@pytest.mark.parametrize(("source", "target"), _SCHEMA_UPGRADES)
def test_a_schema_upgrade_warm_start_acts_as_its_artifact(
    tmp_path, actor_inputs, source, target
) -> None:
    """An older-schema clone starts a newer-schema PPO run acting as it did.

    The new columns meet zero weights only, so the upgraded actor ignores them
    exactly. Against the artifact itself its logits agree to float32 rounding:
    the one difference is each widened projection's GEMM, which may
    reassociate the older terms.
    """
    pretrained, actor, inputs = _schema_upgrade(tmp_path / "clone.pt", actor_inputs, source, target)
    with torch.no_grad():
        for path in ("forward_with_belief", "forward_with_auxiliary_belief"):
            original = getattr(pretrained, path)(inputs["legacy"])[0]
            candidate = getattr(actor, path)(inputs["staged"])[0]
            noised = getattr(actor, path)(inputs["noised"])[0]
            for left, right, redrawn in zip(original, candidate, noised, strict=True):
                assert torch.equal(right, redrawn)
                torch.testing.assert_close(right, left)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("source", "target"), _SCHEMA_UPGRADES)
def test_a_schema_upgrade_warm_start_preserves_the_compiled_bf16_policy(
    tmp_path, actor_inputs, source, target
) -> None:
    """The same on the update path's compiled BF16 arithmetic.

    A run compiles only its upgraded actor. Each case here compiles two
    schemas' actors, and every one guards the same `forward`, so the cases
    start from an empty Dynamo cache instead of accumulating past its
    recompile limit.
    """
    torch._dynamo.reset()
    pretrained, actor, inputs = _schema_upgrade(tmp_path / "clone.pt", actor_inputs, source, target)
    inputs = {
        name: tokens._replace(**{field: value.cuda() for field, value in tokens._asdict().items()})
        for name, tokens in inputs.items()
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = torch.compile(pretrained.cuda(), fullgraph=True)(inputs["legacy"])
        upgraded = torch.compile(actor.cuda(), fullgraph=True)
        actual = upgraded(inputs["staged"])
        noised = upgraded(inputs["noised"])
    for expected_field, actual_field, noised_field in zip(expected, actual, noised, strict=True):
        assert torch.equal(actual_field, noised_field)
        torch.testing.assert_close(actual_field, expected_field)


def test_a_warm_start_refuses_a_schema_downgrade_or_another_difference(tmp_path) -> None:
    newest = max(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
    config = _tiny_config(observation_schema_version=newest)
    _save_actor_artifact(tmp_path / "clone.pt", config)
    module = _training_script()
    for mismatched in (
        replace(config, observation_schema_version=newest - 1),
        replace(config, core_layers=3),
    ):
        with pytest.raises(ValueError, match="model configuration does not match"):
            module._load_initial_actor(
                tmp_path / "clone.pt",
                LejepaActor(mismatched),
                "lejepa",
                mismatched,
                torch.device("cpu"),
            )


# --------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------


def _actor_terms(actor, objective, inputs, factors, plan=None, horizon=1) -> JepaTerms:
    return jepa_horizon_loss(
        objective, actor.auxiliary_belief(inputs), inputs, factors, horizon=horizon, plan=plan
    )


def test_every_journalled_term_is_finite(actor_inputs, factors, plan) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    terms = _actor_terms(LejepaActor(config), JepaObjective(config), actor_inputs, factors, plan)

    assert isinstance(terms, JepaTerms)
    assert JepaTerms._fields == JEPA_METRICS
    for name, value in zip(JEPA_METRICS, terms, strict=True):
        assert torch.isfinite(value).all(), name
        assert value.ndim == 0, name


def test_the_prediction_target_carries_gradient(actor_inputs, factors, plan) -> None:
    """The whole point of the transplant: no stop-gradient on the target."""
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    latents = {
        "unit": torch.randn(8, MAX_UNITS, config.model_dim, requires_grad=True),
        "market": torch.randn(8, MAX_MARKET_ORDERS, config.model_dim, requires_grad=True),
        "economy": torch.randn(8, 6, config.model_dim, requires_grad=True),
        "tile": torch.randn(8, 2 * TILE_COUNT, config.model_dim, requires_grad=True),
    }
    belief = JepaBelief(latents["unit"], latents["market"], latents["economy"], latents["tile"])

    terms = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1, plan=plan)
    terms.prediction.backward()

    source = plan.indices[0]
    target = plan.indices[plan.eligible.shape[0] + 1]
    rows_with_gradient = {
        int(row)
        for row in range(8)
        if latents["tile"].grad is not None and float(latents["tile"].grad[row].abs().sum()) > 0.0
    }
    # Both ends of every admitted edge receive gradient, which a detached
    # objective would restrict to the sources.
    assert rows_with_gradient >= set(source[plan.eligible[0]].tolist())
    assert rows_with_gradient >= set(target[plan.eligible[0]].tolist())


def test_the_detached_target_ablation_moves_only_the_target_side_gradient(
    actor_inputs, factors, plan
) -> None:
    """cleanrl's JEPA-PPO stop-gradient, as an ablation of LeWM's attached target.

    Detaching changes where the prediction term's gradient lands and nothing
    else: every returned value is bitwise identical, a row that is only ever a
    successor stops receiving prediction gradient, and a row that is only ever a
    source receives exactly what it did with the target attached.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    shapes = {
        "unit": (8, MAX_UNITS, config.model_dim),
        "market": (8, MAX_MARKET_ORDERS, config.model_dim),
        "economy": (8, 6, config.model_dim),
        "tile": (8, 2 * TILE_COUNT, config.model_dim),
    }
    values = {name: torch.randn(shape) for name, shape in shapes.items()}

    def run(detach_target: bool) -> tuple[JepaTerms, Tensor]:
        latents = {name: value.clone().requires_grad_() for name, value in values.items()}
        belief = JepaBelief(*(latents[name] for name in shapes))
        terms = jepa_horizon_loss(
            objective,
            belief,
            actor_inputs,
            factors,
            horizon=1,
            plan=plan,
            detach_target=detach_target,
        )
        terms.prediction.backward()
        # Per-row gradient magnitude over every latent group.
        per_row = sum(latents[name].grad.abs().flatten(1).sum(1) for name in shapes)
        return terms, per_row

    attached, attached_gradient = run(detach_target=False)
    detached, detached_gradient = run(detach_target=True)

    for name, left, right in zip(JEPA_METRICS, attached, detached, strict=True):
        assert torch.equal(left.detach(), right.detach()), name
    eligible = plan.eligible[0]
    sources = set(plan.indices[0][eligible].tolist())
    targets = set(plan.indices[plan.eligible.shape[0] + 1][eligible].tolist())
    successor_only = sorted(targets - sources)
    source_only = sorted(sources - targets)
    assert successor_only and source_only
    assert (attached_gradient[successor_only] > 0.0).all()
    assert (detached_gradient[successor_only] == 0.0).all()
    torch.testing.assert_close(detached_gradient[source_only], attached_gradient[source_only])


def test_the_reward_term_is_scored_on_rows_the_transition_cannot_reach(
    actor_inputs, factors, plan
) -> None:
    """Under terminal-outcome rewards the entire signal sits on rows that are
    never transition sources, so a reward scored only on sources trains on zeros."""
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    actor = LejepaActor(config)

    terms = _actor_terms(actor, objective, actor_inputs, factors, plan)

    source_rows = set(plan.indices[0][plan.eligible[0]].tolist())
    nonzero_reward_rows = set(torch.nonzero(factors["rewards"]).flatten().tolist())
    assert nonzero_reward_rows and not (nonzero_reward_rows & source_rows)
    # `reward_scale` is the RMS of the target over the same population the
    # reward loss uses; it is nonzero only because that population is every row.
    expected = float(factors["rewards"].square().mean().sqrt())
    assert float(terms.reward_scale) == pytest.approx(expected, rel=1e-5)
    assert float(terms.reward.detach()) > 0.0


def test_the_transition_is_scored_only_on_plan_admitted_sources(
    actor_inputs, factors, plan
) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    actor = LejepaActor(config)
    objective = JepaObjective(config)

    terms = _actor_terms(actor, objective, actor_inputs, factors, plan)

    # Six of eight rows have a successor inside their own episode.
    assert float(terms.eligible) == pytest.approx(float(plan.eligible[0].sum()))
    assert float(terms.eligible) == pytest.approx(2.0 * (_ROWS_PER_SEAT - 1))


def test_sigreg_reads_every_staged_row_not_only_the_sources(actor_inputs, factors, plan) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    actor = LejepaActor(config)
    objective = JepaObjective(config)
    belief = actor.auxiliary_belief(actor_inputs)

    planned = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1, plan=plan)
    unplanned = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1)

    # The plan restricts the transition, never the representation constraint.
    assert float(planned.sigreg.detach()) == pytest.approx(
        float(unplanned.sigreg.detach()), rel=1e-6
    )
    assert float(planned.dispersion) == pytest.approx(float(unplanned.dispersion), rel=1e-6)


def test_the_sigreg_row_budget_truncates_the_population(actor_inputs, factors, plan) -> None:
    """The budget takes the leading slice of an already shuffled minibatch."""
    torch.manual_seed(0)
    config = _tiny_config(jepa_sigreg_rows=3)
    objective = JepaObjective(config)
    actor = LejepaActor(config)
    with torch.no_grad():
        belief = JepaBelief(*(value.detach() for value in actor.auxiliary_belief(actor_inputs)))
        terms = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1, plan=plan)
        embedded = objective.project(belief_groups(belief, objective.tile_index))["economy"][:3]
        expected = objective.statistic(embedded, objective.directions("economy"))

    assert float(terms.sigreg_economy) == pytest.approx(float(expected), rel=1e-5)


def test_the_row_weight_excludes_the_tail_the_trainer_duplicated(
    actor_inputs, factors, plan
) -> None:
    """`_fixed_minibatch_positions` wraps the epoch's last minibatch back to the
    head of the ordering and requires its callers to zero-weight the duplicates.
    The transition term is covered by the plan's sentinels; SIGReg and the reward
    read every staged row, so they need the weight itself."""
    torch.manual_seed(0)
    config = _tiny_config(jepa_sigreg_rows=8)
    objective = JepaObjective(config)
    actor = LejepaActor(config)
    with torch.no_grad():
        belief = JepaBelief(*(value.detach() for value in actor.auxiliary_belief(actor_inputs)))
    rows = belief.economy.shape[0]
    weight = torch.ones(rows)
    weight[:2] = 0.0

    poisoned_belief = JepaBelief(*(value.clone() for value in belief))
    for field in poisoned_belief:
        field[:2] = 1e3
    poisoned_factors = {**factors, "rewards": factors["rewards"].clone()}
    poisoned_factors["rewards"][:2] = 1e3

    def score(chosen_belief, chosen_factors, sample_weight):
        with torch.no_grad():
            return jepa_horizon_loss(
                objective,
                chosen_belief,
                actor_inputs,
                chosen_factors,
                horizon=1,
                plan=plan,
                sample_weight=sample_weight,
            )

    reference = score(belief, factors, weight)
    perturbed = score(poisoned_belief, poisoned_factors, weight)
    unweighted = score(poisoned_belief, poisoned_factors, None)

    assert float(perturbed.sigreg) == pytest.approx(float(reference.sigreg), rel=1e-5)
    assert float(perturbed.reward) == pytest.approx(float(reference.reward), rel=1e-5)
    # And the weight is doing the work: without it the same rows move both terms.
    # The statistic is not monotone in an outlier -- a projection far enough out
    # oscillates the characteristic function back toward zero -- so what is
    # asserted of it is that it moves, not which way.
    assert float(unweighted.sigreg) != pytest.approx(float(reference.sigreg), rel=1e-2)
    assert float(unweighted.reward) > 10.0 * float(reference.reward)


def test_a_zero_reward_coefficient_skips_the_reward_graph(actor_inputs, factors, plan) -> None:
    """The head reads every staged row, so computing it for a term that is then
    multiplied by zero is the most expensive nothing in the objective."""
    torch.manual_seed(0)
    objective = JepaObjective(_tiny_config())
    actor = LejepaActor(_tiny_config())
    belief = actor.auxiliary_belief(actor_inputs)
    terms = jepa_horizon_loss(
        objective, belief, actor_inputs, factors, horizon=1, plan=plan, score_reward=False
    )

    assert float(terms.reward.detach()) == 0.0
    assert not terms.reward.requires_grad
    # The baseline goes with it: a zero loss beside a live baseline reads as a
    # perfect predictor, and the pair is the only thing that says which it is.
    assert float(terms.reward_scale.detach()) == 0.0
    # Everything else is untouched.
    assert terms.prediction.requires_grad
    terms.prediction.backward()
    assert all(parameter.grad is None for parameter in objective.predictor.reward_head.parameters())


def test_motion_sees_the_trajectory_constant_encoder_that_sigreg_cannot(
    actor_inputs, factors, plan
) -> None:
    """The one column that reads the failure the module docstring names.

    An encoder that is constant along a trajectory but varies across the batch
    drives the prediction term to zero, drags the persistence baseline down with
    it, and leaves both SIGReg and the dispersion reading clean -- across the
    batch such an embedding is still perfectly ordinary.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    actor = LejepaActor(config)
    with torch.no_grad():
        belief = JepaBelief(*(value.detach() for value in actor.auxiliary_belief(actor_inputs)))
        # Every row replaced by the first row of its own run: constant along a
        # trajectory, still varying across the batch.
        anchor = (
            torch.div(torch.arange(belief.economy.shape[0]), _ROWS_PER_SEAT, rounding_mode="floor")
            * _ROWS_PER_SEAT
        )
        frozen = JepaBelief(*(value[anchor] for value in belief))
        reference = jepa_horizon_loss(
            objective, belief, actor_inputs, factors, horizon=1, plan=plan
        )
        collapsed = jepa_horizon_loss(
            objective, frozen, actor_inputs, factors, horizon=1, plan=plan
        )

    assert float(reference.motion.detach()) > 1e-3
    assert float(collapsed.motion.detach()) == pytest.approx(0.0, abs=1e-6)
    # The failure is invisible to everything else: the prediction term is now
    # exactly zero and the dispersion is unmoved.
    assert float(collapsed.prediction.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(collapsed.dispersion.detach()) > 0.5 * float(reference.dispersion.detach())


def test_the_reward_head_reads_groups_and_their_concatenation_alike(actor_inputs, factors) -> None:
    """Pooling per group spares a `[rows, tokens, width]` concatenation, and the
    masked mean is the same either way."""
    torch.manual_seed(0)
    objective = JepaObjective(_tiny_config())
    actor = LejepaActor(_tiny_config())
    with torch.no_grad():
        belief = actor.auxiliary_belief(actor_inputs)
        groups = belief_groups(belief, objective.tile_index)
        embedded = objective.project(groups)
        order = objective.groups
        valid = {
            name: torch.ones(value.shape[:2], dtype=torch.bool) for name, value in groups.items()
        }
        valid["unit"] = actor_inputs.unit_active
        actions = objective.embed_action(
            factors["unit_actions"],
            factors["market_kinds"],
            factors["market_quantities"],
            actor_inputs.unit_categorical,
            actor_inputs.unit_active,
        )
        split = objective.reward_prediction(
            [embedded[name] for name in order], actions, [valid[name] for name in order]
        )
        joined = objective.reward_prediction(
            torch.cat([embedded[name] for name in order], dim=1),
            actions,
            torch.cat([valid[name] for name in order], dim=1),
        )

    assert torch.allclose(split, joined, atol=1e-5)


def test_inactive_unit_slots_are_excluded_from_both_populations(
    actor_inputs, factors, plan
) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    actor = LejepaActor(config)
    assert (~actor_inputs.unit_active).any()

    belief = actor.auxiliary_belief(actor_inputs)
    reference = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1, plan=plan)
    # Padding slots carry an exactly zero latent; perturbing them must not move
    # a single term, because nothing supervises them.
    poisoned = JepaBelief(
        belief.unit_decisions.masked_fill((~actor_inputs.unit_active).unsqueeze(-1), 1e3),
        belief.market_decisions,
        belief.economy,
        belief.tiles,
    )
    perturbed = jepa_horizon_loss(objective, poisoned, actor_inputs, factors, horizon=1, plan=plan)

    assert float(perturbed.unit.detach()) == pytest.approx(float(reference.unit.detach()), rel=1e-4)
    assert float(perturbed.sigreg_unit.detach()) == pytest.approx(
        float(reference.sigreg_unit.detach()), rel=1e-4
    )


def test_tile_term_upweights_the_slots_that__moved(actor_inputs, factors, plan) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    terms = _actor_terms(LejepaActor(config), JepaObjective(config), actor_inputs, factors, plan)

    assert float(terms.tile.detach()) == pytest.approx(
        0.5 * (float(terms.tile_all.detach()) + float(terms.tile_changed.detach())), rel=1e-5
    )
    assert float(terms.tile_unchanged.detach()) >= 0.0


def test_the_predictor_starts_as_the_identity(actor_inputs, factors, plan) -> None:
    """A zero-initialized residual output makes the first step's transition exactly
    the persistence control, so any separation later is learned, not initial."""
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    actor = LejepaActor(config)
    with torch.no_grad():
        belief = JepaBelief(*(v.detach() for v in actor.auxiliary_belief(actor_inputs)))

    with torch.no_grad():
        live = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1, plan=plan)
        persistence = jepa_horizon_loss(
            JepaPersistenceControl(objective), belief, actor_inputs, factors, horizon=1, plan=plan
        )

    assert float(live.prediction) == pytest.approx(float(persistence.prediction), rel=1e-5)
    assert float(live.residual_ratio) == pytest.approx(0.0, abs=1e-6)


def test_controls_share_the_objective_state_without_owning_parameters(
    actor_inputs, factors, plan
) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    actor = LejepaActor(config)
    with torch.no_grad():
        belief = JepaBelief(*(v.detach() for v in actor.auxiliary_belief(actor_inputs)))

    for control_class in (JepaPersistenceControl, JepaShuffledControl):
        control = control_class(objective)
        assert not isinstance(control, torch.nn.Module)
        assert not hasattr(control, "parameters")
        assert control.tile_index is objective.tile_index
        for name in objective.groups:
            assert control.directions(name) is objective.directions(name)
        with torch.no_grad():
            terms = jepa_horizon_loss(control, belief, actor_inputs, factors, horizon=1, plan=plan)
        assert all(torch.isfinite(value).all() for value in terms)


def test_the_shuffled_control_reads_another_trajectory_action(actor_inputs, factors, plan) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    # Give the predictor a non-identity transition so the control can differ.
    with torch.no_grad():
        for parameter in objective.predictor.output.parameters():
            parameter.normal_(0.0, 0.2)
    actor = LejepaActor(config)
    with torch.no_grad():
        belief = JepaBelief(*(v.detach() for v in actor.auxiliary_belief(actor_inputs)))
        live = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1, plan=plan)
        shuffled = jepa_horizon_loss(
            JepaShuffledControl(objective, 2), belief, actor_inputs, factors, horizon=1, plan=plan
        )

    assert float(shuffled.prediction) != pytest.approx(float(live.prediction), rel=1e-6)


def test_the_shuffled_control_strides_past_its_own_run(actor_inputs, factors, plan) -> None:
    """A stride of one would hand most rows their own trajectory's last action.

    The minibatch is built of contiguous same-trajectory runs, so `roll(1)` is
    the previous *step* of the same episode for every row but the first of each
    run. That is autocorrelated enough that an action-conditioned predictor still
    scores well against it, which is exactly the reading the control exists to
    refuse. Striding by the run length lands on a different run.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    with torch.no_grad():
        for parameter in objective.predictor.output.parameters():
            parameter.normal_(0.0, 0.2)
    actor = LejepaActor(config)
    with torch.no_grad():
        belief = JepaBelief(*(v.detach() for v in actor.auxiliary_belief(actor_inputs)))
        live = jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=1, plan=plan)
        near = jepa_horizon_loss(
            JepaShuffledControl(objective, 1), belief, actor_inputs, factors, horizon=1, plan=plan
        )
        far = jepa_horizon_loss(
            JepaShuffledControl(objective, _ROWS_PER_SEAT),
            belief,
            actor_inputs,
            factors,
            horizon=1,
            plan=plan,
        )

    live_loss = float(live.prediction)
    assert float(far.prediction) - live_loss > float(near.prediction) - live_loss
    with pytest.raises(ValueError, match="stride must be positive"):
        JepaShuffledControl(objective, 0)


def test_the_loss_refuses_a_horizon_the_plan_does_not_cover(actor_inputs, factors, plan) -> None:
    config = _tiny_config()
    objective = JepaObjective(config)
    actor = LejepaActor(config)
    belief = actor.auxiliary_belief(actor_inputs)

    with pytest.raises(ValueError):
        jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=0, plan=plan)
    with pytest.raises(ValueError):
        jepa_horizon_loss(objective, belief, actor_inputs, factors, horizon=2, plan=plan)


def test_the_objective_reaches_the_trunk_and_the_projectors_but_no_head(
    actor_inputs, factors, plan
) -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    objective = JepaObjective(config)
    actor = LejepaActor(config)

    terms = _actor_terms(actor, objective, actor_inputs, factors, plan)
    (terms.prediction + 0.09 * terms.sigreg + 0.1 * terms.reward).backward()

    trunk = [
        name
        for name, parameter in actor.named_parameters()
        if name.startswith("trunk.") and parameter.grad is not None
    ]
    assert len(trunk) > 20
    assert all(parameter.grad is not None for parameter in objective.projectors.parameters())
    # The readouts below the belief are not on this objective's graph at all:
    # the objective reaches the trunk and the per-family norm that defines the
    # belief, and stops there.
    assert len(actor.readout) > 0
    assert all(parameter.grad is None for parameter in actor.readout.parameters())
    assert actor.unit_head[-1].weight.grad is None
    assert actor.market_kind.weight.grad is None


def test_belief_groups_applies_the_shared_tile_sample() -> None:
    config = _tiny_config()
    objective = JepaObjective(config)
    belief = JepaBelief(
        torch.zeros(2, MAX_UNITS, config.model_dim),
        torch.zeros(2, MAX_MARKET_ORDERS, config.model_dim),
        torch.zeros(2, 6, config.model_dim),
        torch.arange(2 * 2 * TILE_COUNT * config.model_dim, dtype=torch.float32).reshape(
            2, 2 * TILE_COUNT, config.model_dim
        ),
    )

    groups = belief_groups(belief, objective.tile_index)

    assert set(groups) == set(JEPA_GROUPS)
    assert groups["tile"].shape == (2, config.jepa_tile_samples, config.model_dim)
    assert torch.equal(groups["tile"], belief.tiles.index_select(1, objective.tile_index))


# --------------------------------------------------------------------------
# Trainer wiring
# --------------------------------------------------------------------------


def _jepa_ppo_config(**overrides) -> PpoConfig:
    base = dict(
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
        jepa_reward_coefficient=0.1,
    )
    base.update(overrides)
    config = PpoConfig(**base)
    _validate_config(config)
    return config


def test_an_attached_target_without_sigreg_is_refused() -> None:
    _validate_config(_jepa_ppo_config())
    with pytest.raises(ValueError, match="collapses"):
        _jepa_ppo_config(jepa_sigreg_coefficient=0.0)
    with pytest.raises(ValueError, match="collapses"):
        _validate_config(PpoConfig(jepa_sigreg_coefficient=0.09))
    with pytest.raises(ValueError, match="collapses"):
        _validate_config(PpoConfig(jepa_prediction_coefficient=1.0, jepa_reward_coefficient=0.1))


def test_lejepa_and_detached_nextlat_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _jepa_ppo_config(structured_decision_coefficient=0.1, structured_decision_horizon=1)
    # The detached critic predictor is refused by the same rule: there is one
    # representation now, and it cannot carry two conventions about the gradient.
    with pytest.raises(ValueError, match="mutually exclusive"):
        _jepa_ppo_config(structured_critic_latent_coefficient=0.1, structured_critic_horizon=1)


def test_the_detached_target_ablation_requires_the_objective_it_ablates() -> None:
    assert _jepa_ppo_config(jepa_detach_target=True).jepa_detach_target
    with pytest.raises(ValueError, match="requires the LeJEPA objective"):
        _validate_config(PpoConfig(jepa_detach_target=True))


def test_active_flags_follow_the_coefficients() -> None:
    config = _jepa_ppo_config()

    assert config.jepa_active
    assert config.structured_actor_auxiliary_active
    # The world model is the actor arm's objective; nothing turns on a critic
    # auxiliary, because the critic carries no encoder to regularize.
    assert not config.structured_critic_auxiliary_active
    assert not PpoConfig().jepa_active


def test_the_world_model_optimizer_owns_the_backbone_beside_the_objective() -> None:
    config = _jepa_ppo_config()
    model_config = _tiny_config()
    actor, _ = build_lejepa_pair(model_config)
    objective = JepaObjective(model_config)

    # It is not the predictor's optimizer any more, and it refuses to be built
    # as though it were: the backbone has no other loss, so leaving it out would
    # quietly hand the run an encoder that never steps.
    with pytest.raises(ValueError, match="pass the actor"):
        make_structured_dynamics_optimizer(objective, config, critic=False)
    with pytest.raises(ValueError, match="no critic arm"):
        make_structured_dynamics_optimizer(objective, config, critic=True, actor=actor)
    with pytest.raises(ValueError, match="only the LeJEPA"):
        make_structured_dynamics_optimizer(
            StructuredCriticDynamics(model_config), config, critic=True, actor=actor
        )

    optimizer = make_structured_dynamics_optimizer(objective, config, critic=False, actor=actor)
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        config.resolved_structured_learning_rate
    )
    assert owned == {id(p) for p in objective.parameters()} | {
        id(p) for p in actor.backbone_parameters()
    }


def test_the_actor_optimizer_steps_the_heads_and_never_the_backbone() -> None:
    config = _jepa_ppo_config()
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)

    actor_optimizer, _ = make_optimizers(actor, critic, config)
    owned = {id(p) for group in actor_optimizer.param_groups for p in group["params"]}
    backbone = {id(p) for p in actor.backbone_parameters()}

    assert backbone
    assert not (owned & backbone)
    assert owned == {id(p) for p in actor.head_parameters()}
    assert owned | backbone == {id(p) for p in actor.parameters()}


def _training_script():
    path = Path(__file__).parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_train_ppo_lejepa", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_args(module, monkeypatch, run_dir, *extra: str):
    command = ["train_ppo.py", "--run-dir", str(run_dir), *extra]
    monkeypatch.setattr(sys, "argv", command)
    return module.parse_args()


def test_train_ppo_binds_the_coefficients_to_the_architecture(monkeypatch, tmp_path) -> None:
    module = _training_script()
    objective = (
        "--jepa-prediction-coefficient",
        "1.0",
        "--jepa-sigreg-coefficient",
        "0.09",
        "--jepa-reward-coefficient",
        "0.1",
    )

    module._validate_args(
        _cli_args(module, monkeypatch, tmp_path, "--architecture", LEJEPA, *objective)
    )
    with pytest.raises(ValueError, match="require --architecture lejepa"):
        module._validate_args(
            _cli_args(
                module, monkeypatch, tmp_path, "--architecture", "entity-attention", *objective
            )
        )
    # An unflagged launch inherits the family's objective, and only an explicit
    # zero can switch it off.
    inherited = _cli_args(module, monkeypatch, tmp_path, "--architecture", LEJEPA)
    assert (inherited.jepa_prediction_coefficient, inherited.jepa_sigreg_coefficient) == (
        1.0,
        0.09,
    )
    module._validate_args(inherited)
    ablated = (
        "--architecture",
        LEJEPA,
        "--jepa-prediction-coefficient",
        "0",
        "--jepa-sigreg-coefficient",
        "0",
    )
    with pytest.raises(ValueError, match="without its objective"):
        module._validate_args(_cli_args(module, monkeypatch, tmp_path, *ablated))
    # Fine-tuning a clone by the policy's gradient alone ablates the objective.
    module._validate_args(
        _cli_args(
            module, monkeypatch, tmp_path, *ablated, "--init-actor-from", str(tmp_path / "clone.pt")
        )
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        module._validate_args(
            _cli_args(
                module,
                monkeypatch,
                tmp_path,
                "--architecture",
                LEJEPA,
                "--jepa-prediction-coefficient",
                "-1.0",
                "--jepa-sigreg-coefficient",
                "0.09",
            )
        )


def test_every_jepa_knob_has_a_flag_that_defaults_to_the_config(monkeypatch, tmp_path) -> None:
    """A field the launcher cannot set is a field a run cannot record."""
    module = _training_script()
    args = _cli_args(module, monkeypatch, tmp_path, "--architecture", "entity-attention")
    fields = [name for name in PpoConfig.__dataclass_fields__ if name.startswith("jepa_")]

    assert len(fields) == 5
    for name in fields:
        assert hasattr(args, name), name
        assert getattr(args, name) == getattr(PpoConfig, name), name


# --------------------------------------------------------------------------
# End to end through the update
# --------------------------------------------------------------------------


def _pin_quantity_orders(actor: LejepaActor) -> None:
    """Force a market order every step, so market slots carry live decisions."""
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-12.0)
        actor.market_kind.bias[MarketKind.STOP] = -6.0
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 6.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[MarketKind.BUY_SEED_WHEAT, -1] = 50.0


def test_the_lejepa_family_refuses_a_detached_predictor_on_either_arm() -> None:
    """One representation, one convention about where the gradient stops.

    The coefficient rule already refuses a detached NextLat beside the LeJEPA
    objective. These are the module-level refusals behind it: the family's actor
    takes the world model and nothing else, and its critic -- which now owns no
    encoder at all -- takes no predictor.
    """
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    jepa = PpoConfig(jepa_prediction_coefficient=1.0, jepa_sigreg_coefficient=0.09)
    _validate_config(jepa)

    with pytest.raises(ValueError, match="require exactly a LeJEPA objective"):
        _validate_structured_auxiliary_modules(actor, ActorDynamics(model_config), jepa)
    # And with the coefficients off, the family still refuses the detached one.
    with pytest.raises(ValueError, match="admits only the LeJEPA objective"):
        _validate_structured_auxiliary_modules(
            actor,
            ActorDynamics(model_config),
            PpoConfig(structured_decision_coefficient=0.1, structured_decision_horizon=1),
        )
    with pytest.raises(ValueError, match="admits no predictor"):
        _validate_structured_critic_auxiliary_modules(
            critic,
            StructuredCriticDynamics(model_config),
            PpoConfig(structured_critic_latent_coefficient=0.1, structured_critic_horizon=1),
        )
    # And the objective itself is refused on the critic side outright.
    with pytest.raises(ValueError, match="belongs to the actor arm"):
        _validate_structured_critic_auxiliary_modules(critic, JepaObjective(model_config), jepa)


def test_the_detached_target_flag_parses_and_is_bound_to_the_objective(
    monkeypatch, tmp_path
) -> None:
    module = _training_script()
    objective = (
        "--architecture",
        LEJEPA,
        "--jepa-prediction-coefficient",
        "1.0",
        "--jepa-sigreg-coefficient",
        "0.09",
    )

    assert not _cli_args(module, monkeypatch, tmp_path, *objective).jepa_detach_target
    enabled = _cli_args(module, monkeypatch, tmp_path, *objective, "--jepa-detach-target")
    module._validate_args(enabled)
    assert enabled.jepa_detach_target
    assert not _cli_args(
        module, monkeypatch, tmp_path, *objective, "--no-jepa-detach-target"
    ).jepa_detach_target
    with pytest.raises(ValueError, match="requires the LeJEPA objective"):
        module._validate_args(
            _cli_args(
                module,
                monkeypatch,
                tmp_path,
                "--architecture",
                "entity-attention",
                "--jepa-detach-target",
            )
        )


def test_the_cli_refuses_the_lejepa_family_beside_a_detached_nextlat(monkeypatch, tmp_path) -> None:
    """A run should learn this from its arguments, not an hour into setup."""
    module = _training_script()
    objective = (
        "--architecture",
        LEJEPA,
        "--jepa-prediction-coefficient",
        "1.0",
        "--jepa-sigreg-coefficient",
        "0.09",
    )
    for flag in ("--structured-decision-coefficient", "--structured-critic-latent-coefficient"):
        with pytest.raises(ValueError, match="admits only the LeJEPA objective"):
            module._validate_args(_cli_args(module, monkeypatch, tmp_path, *objective, flag, "0.1"))


def test_update_ppo_trains_the_shared_backbone_and_nothing_else_does() -> None:
    torch.manual_seed(0)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=225,
        episode_steps=8,
        sampling_seed=53,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
        structured_learning_rate=1.0e-2,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
        jepa_reward_coefficient=0.1,
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    before_objective = _snapshot(objective.named_parameters())
    before_backbone = _snapshot(actor.trunk.named_parameters())
    before_critic = _snapshot(critic.named_parameters())

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
        auxiliary_generator=np.random.default_rng(44),
        structured_dynamics=objective,
        structured_dynamics_optimizer=make_structured_dynamics_optimizer(
            objective, config, critic=False, actor=actor
        ),
        structured_actor_auxiliary=True,
    )

    assert metrics["structured_actor_predictor_updates"] >= 1
    for name in JEPA_METRICS:
        assert math.isfinite(metrics[f"structured_actor_{name}"]), name
    # The detached family's columns must not appear beside them, and the critic
    # arm journals nothing at all: it has no auxiliary.
    assert "structured_actor_latent" not in metrics
    assert not any(key.startswith("structured_critic_preupdate") for key in metrics)
    assert metrics["structured_actor_dispersion"] > 0.0
    assert metrics["structured_actor_motion"] > 0.0
    # Both matched baselines are journaled. Persistence alone cannot separate an
    # idle predictor from an encoder that went constant along the trajectory.
    for label in ("persistence", "shuffled"):
        assert math.isfinite(metrics[f"structured_preupdate_{label}_combined"]), label
        assert math.isfinite(metrics[f"structured_preupdate_{label}_prediction"]), label
    # And the train loop's consumer reads them. It is keyed on the detached
    # predictor's field names, which this arm does not journal at all, so an
    # unarmed consumer is a `KeyError` on the first completed iteration -- a
    # failure no test stopping at `update_ppo` can see.
    module = _training_script()
    ratios = module._structured_persistence_diagnostics(metrics, kind="actor", jepa=True)
    for label in ("persistence", "shuffled"):
        assert math.isfinite(ratios[f"structured_{label}_combined_ratio"]), label
        assert math.isfinite(ratios[f"structured_{label}_prediction_ratio"]), label

    # The point of the family, asserted on the weights the update actually left
    # behind: the world model stepped the backbone beside its own projectors and
    # predictor, and the value objective stepped the critic. The actor's heads
    # are not asserted here -- this fixture is one eight-step game, whose
    # normalized advantages are identically zero, so the policy's gradient is a
    # true zero and no optimizer would move them. Where the policy gradient
    # goes is `test_ppo_carries_the_policy_gradient_into_the_backbone`.
    assert math.isfinite(metrics["actor_gradient_norm"])
    assert _moved(objective.named_parameters(), before_objective)
    assert _moved(actor.trunk.named_parameters(), before_backbone)
    assert _moved(critic.named_parameters(), before_critic)


def test_update_rematerialization_trades_memory_and_nothing_else() -> None:
    """Replaying the farm blocks in backward recomputes the same function."""

    def update(rematerialize_actor_update: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model_config = _tiny_config()
        actor, critic = build_lejepa_pair(model_config)
        _pin_quantity_orders(actor)
        rollout = collect_self_play(
            actor,
            games=1,
            seed_start=225,
            episode_steps=8,
            sampling_seed=53,
            reward_mode="terminal-outcome",
        )
        config = PpoConfig(
            optimizer="adamw",
            epochs=1,
            minibatch_size=1 << 12,
            lr_warmup_steps=0,
            target_kl=1.0,
            use_bfloat16=False,
            structured_learning_rate=1.0e-2,
            jepa_prediction_coefficient=1.0,
            jepa_sigreg_coefficient=0.09,
            rematerialize_actor_update=rematerialize_actor_update,
        )
        objective = JepaObjective(model_config)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
        update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(43),
            auxiliary_generator=np.random.default_rng(44),
            structured_dynamics=objective,
            structured_dynamics_optimizer=make_structured_dynamics_optimizer(
                objective, config, critic=False, actor=actor
            ),
            structured_actor_auxiliary=True,
        )
        return _snapshot(actor.named_parameters())

    reference = update(True)
    for name, value in update(False).items():
        torch.testing.assert_close(value, reference[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("rematerialize", [False, True])
def test_the_entropy_bonus_trains_beside_the_world_model(rematerialize: bool) -> None:
    """The bonus reaches the policy through the promoted recipe's update path.

    Three waves over the same states, with and without the bonus. The world
    model and the surrogate are identical between the arms, so the higher
    entropy the bonus arm reports on its last wave is the bonus's doing, and
    `policy_loss` still reports the surrogate beside a separate bonus column.
    Both arms are seeded CPU eager runs, so the comparison is deterministic;
    the measured gap is 0.634 against 0.677 nats.
    """

    def waves(entropy_coefficient: float) -> dict[str, float]:
        torch.manual_seed(0)
        model_config = _tiny_config()
        actor, critic = build_lejepa_pair(model_config)
        _pin_quantity_orders(actor)
        rollout = collect_self_play(
            actor,
            games=2,
            seed_start=225,
            episode_steps=16,
            sampling_seed=53,
            reward_mode="terminal-outcome",
        )
        # Opposed outcomes, so the surrogate has a real gradient in both arms.
        last = rollout.valid.shape[1] - 1 - rollout.valid[:, ::-1].argmax(axis=1)
        rollout.rewards[np.arange(last.size), last] = np.where(np.arange(last.size) % 2, -1.0, 1.0)
        config = PpoConfig(
            optimizer="adamw",
            epochs=1,
            minibatch_size=1 << 12,
            lr_warmup_steps=0,
            target_kl=10.0,
            use_bfloat16=False,
            actor_learning_rate=1.0e-2,
            structured_learning_rate=1.0e-3,
            jepa_prediction_coefficient=1.0,
            jepa_sigreg_coefficient=0.09,
            jepa_reward_coefficient=0.1,
            rematerialize_actor_update=rematerialize,
            entropy_coefficient=entropy_coefficient,
        )
        objective = JepaObjective(model_config)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
        dynamics_optimizer = make_structured_dynamics_optimizer(
            objective, config, critic=False, actor=actor
        )
        for wave in range(3):
            metrics = update_ppo(
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                rollout,
                config,
                generator=np.random.default_rng(43 + wave),
                auxiliary_generator=np.random.default_rng(44 + wave),
                structured_dynamics=objective,
                structured_dynamics_optimizer=dynamics_optimizer,
                structured_actor_auxiliary=True,
            )
            assert metrics["actor_updates"] == 1
            assert metrics["structured_actor_predictor_updates"] >= 1
        return metrics

    plain = waves(0.0)
    bonus = waves(1.0)
    assert "entropy_bonus" not in plain
    assert math.isfinite(bonus["policy_loss"]) and bonus["entropy_bonus"] > 0.0
    assert bonus["entropy"] > plain["entropy"]


def test_update_ppo_hands_the_detached_target_switch_to_the_objective(monkeypatch) -> None:
    """The config field is read inside the compiled update, not only validated."""
    import kaggriculture.ppo as ppo_module

    torch.manual_seed(0)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=225,
        episode_steps=8,
        sampling_seed=53,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
        jepa_detach_target=True,
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    seen: list[bool] = []

    def spy(*args, detach_target: bool = False, **kwargs):
        seen.append(detach_target)
        return jepa_horizon_loss(*args, detach_target=detach_target, **kwargs)

    monkeypatch.setattr(ppo_module, "jepa_horizon_loss", spy)
    update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
        auxiliary_generator=np.random.default_rng(44),
        structured_dynamics=objective,
        structured_dynamics_optimizer=make_structured_dynamics_optimizer(
            objective, config, critic=False, actor=actor
        ),
        structured_actor_auxiliary=True,
    )

    assert seen and all(seen)


@pytest.mark.parametrize("wdl_value", [False, True])
def test_update_ppo_encodes_each_minibatch_once_for_both_towers(monkeypatch, wdl_value) -> None:
    """Only the whole-wave behavior replay runs the critic's own backbone pass.

    Every minibatch's critic forward must read the encoding the actor side of the
    same minibatch already produced; a critic that re-encoded would be running
    the identical trunk forward twice per minibatch.
    """
    torch.manual_seed(0)
    model_config = _tiny_config(wdl_value=wdl_value)
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=225,
        episode_steps=8,
        sampling_seed=53,
        reward_mode="terminal-outcome" if wdl_value else "shaped",
    )
    config = _jepa_ppo_config(
        optimizer="adamw", epochs=2, minibatch_size=4, lr_warmup_steps=0, use_bfloat16=False
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    handed: list[bool] = []
    encode = LejepaCritic.encode_belief

    def spy(self, *arguments, **keywords):
        handed.append(len(arguments) == 5 and arguments[4] is not None)
        return encode(self, *arguments, **keywords)

    monkeypatch.setattr(LejepaCritic, "encode_belief", spy)
    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
        auxiliary_generator=np.random.default_rng(44),
        structured_dynamics=objective,
        structured_dynamics_optimizer=make_structured_dynamics_optimizer(
            objective, config, critic=False, actor=actor
        ),
        structured_actor_auxiliary=True,
    )

    minibatches = int(metrics["updates"])
    assert minibatches > 1
    assert handed.count(True) == minibatches
    # The behavior replay is the one pass that has no actor encoding to borrow.
    assert not handed[0]
    if wdl_value:
        assert critic.value_head.out_features == 3
        assert math.isfinite(metrics["behavior_match_score_mse"])
        assert metrics["behavior_match_score_out_of_range_fraction"] == 0


def test_the_detached_ablation_stops_both_gradients_at_the_backbone(
    actor_inputs, critic_inputs
) -> None:
    """The pure world-model ablation, from both consumers at once.

    A policy loss reaches the actor's heads and stops; a value loss reaches the
    critic's tower and stops. Neither reaches the encoder they share, so the
    world model is shaped by its own objective and by nothing else.
    """
    torch.manual_seed(0)
    inputs, opponent = critic_inputs
    actor, critic = build_lejepa_pair(_tiny_config(policy_shapes_backbone=False))

    decoded = actor.decode_belief(actor.auxiliary_belief(actor_inputs), actor_inputs.unit_active)
    sum(value.square().mean() for value in decoded).backward()
    head_grads = {name for name, value in actor.named_parameters() if value.grad is not None}

    assert head_grads
    assert not any(name.startswith("trunk.") for name in head_grads)
    assert all(value.grad is None for value in actor.backbone_parameters())
    # The head normalizations are what go missing if `decode_belief` forgets
    # them -- silently, since the readouts still run, just on an input whose
    # scale their initialization did not assume. `EntityActor` applies them at
    # encode time, which this family cannot do.
    assert {"unit_head.0.weight", "market_norm.weight"} <= head_grads
    readout = {name for name, _ in actor.named_parameters() if name.startswith("readout")}
    assert readout and readout <= head_grads

    actor.zero_grad(set_to_none=True)
    critic(inputs, *opponent).square().mean().backward()

    assert any(value.grad is not None for value in critic.parameters())
    assert all(value.grad is None for value in actor.backbone_parameters())
    assert all(value.grad is None for value in actor.parameters())


def test_the_policy_gradient_reaches_the_backbone_and_the_value_gradient_does_not(
    actor_inputs, critic_inputs
) -> None:
    """The family's default: the policy shapes the encoder, the critic never does."""
    torch.manual_seed(0)
    inputs, opponent = critic_inputs
    actor, critic = build_lejepa_pair(_tiny_config())

    decoded = actor.decode_belief(actor.auxiliary_belief(actor_inputs), actor_inputs.unit_active)
    sum(value.square().mean() for value in decoded).backward()

    assert any(value.grad is not None for value in actor.backbone_parameters())
    assert actor.market_kind.weight.grad is not None

    actor.zero_grad(set_to_none=True)
    critic(inputs, *opponent).square().mean().backward()

    assert any(value.grad is not None for value in critic.parameters())
    assert all(value.grad is None for value in actor.parameters())


@pytest.mark.parametrize("shapes", [True, False], ids=["attached", "detached"])
def test_ppo_carries_the_policy_gradient_into_the_backbone(monkeypatch, shapes) -> None:
    """Through the whole update, with the objective's own gradient silenced.

    What the backbone's optimizer then clips and steps is the policy's gradient
    alone: nonzero when the heads read the attached belief, a true zero under
    the detached ablation. The rewards are replaced by noise because this
    fixture's own are zero, and zero advantages carry no policy gradient. Noise
    is not a completed-game outcome, so the critic is the HL-Gauss head.
    """
    import kaggriculture.ppo as ppo_module

    torch.manual_seed(0)
    model_config = _tiny_config(policy_shapes_backbone=shapes, wdl_value=False)
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor, games=1, seed_start=225, episode_steps=8, sampling_seed=53, reward_mode="shaped"
    )
    noise = np.random.default_rng(7).standard_normal(rollout.rewards.shape)
    rollout = replace(rollout, rewards=noise.astype(rollout.rewards.dtype))
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    objective_optimizer = make_structured_dynamics_optimizer(
        objective, config, critic=False, actor=actor
    )
    jepa_terms = ppo_module._jepa_auxiliary_terms

    def silenced(*arguments, **keywords):
        loss, terms = jepa_terms(*arguments, **keywords)
        return loss * 0.0, terms

    monkeypatch.setattr(ppo_module, "_jepa_auxiliary_terms", silenced)
    backbone_norms: list[float] = []
    step = ppo_module._optimizer_step

    def record(stepped, *arguments, **keywords):
        if stepped is objective_optimizer:
            grads = [value.grad for value in actor.backbone_parameters() if value.grad is not None]
            backbone_norms.append(float(torch.nn.utils.get_total_norm(grads)) if grads else 0.0)
        return step(stepped, *arguments, **keywords)

    monkeypatch.setattr(ppo_module, "_optimizer_step", record)
    before_backbone = _snapshot(actor.trunk.named_parameters())

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
        auxiliary_generator=np.random.default_rng(44),
        structured_dynamics=objective,
        structured_dynamics_optimizer=objective_optimizer,
        structured_actor_auxiliary=True,
    )

    assert metrics["actor_gradient_norm"] > 0.0
    assert metrics["actor_updates"] >= 1 and backbone_norms
    assert (max(backbone_norms) > 0.0) == shapes
    # Clipped with the objective it joined, and applied rather than dropped.
    assert max(backbone_norms) <= config.nextlat_max_gradient_norm * (1 + 1e-5)
    assert bool(_moved(actor.trunk.named_parameters(), before_backbone)) == shapes


def test_a_frozen_copy_traces_whole_under_grad_mode() -> None:
    """A frozen clone of the actor compiles whole with grad mode on.

    Nothing in that forward requires grad, the one case where Dynamo cannot
    trace the tiny-vocabulary embedding's vararg Function; the production gate
    died there. The trace must stay whole, and equal to the eager forward.
    """
    torch.manual_seed(0)
    actor, _critic = build_lejepa_pair(_tiny_config())
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=225,
        episode_steps=4,
        sampling_seed=53,
        reward_mode="terminal-outcome",
    )
    states = {name: array[rollout.valid] for name, array in rollout.states.items()}
    states["unit_active"] = rollout.unit_active[rollout.valid]
    args = actor_forward_args(rollout.architecture, states, torch.device("cpu"))
    reference = copy.deepcopy(actor).eval().requires_grad_(False)

    torch._dynamo.reset()
    traced = torch.compile(reference, backend="eager", fullgraph=True)(*args)
    eager = reference(*args)

    torch.testing.assert_close(traced.market_kind_logits, eager.market_kind_logits)
    torch.testing.assert_close(traced.unit_logits, eager.unit_logits)


@pytest.mark.parametrize("optimizer", ["adamw", "normuon"])
def test_without_its_objective_the_actor_optimizer_owns_the_backbone(optimizer) -> None:
    """The backbone joins the policy's optimizer, in groups of its own.

    At the structured rate the objective would have stepped it at, tagged
    `BACKBONE_ROLE`, and never both ways at once: while the objective runs the
    actor's optimizer holds the heads alone.
    """
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    config = PpoConfig(optimizer=optimizer, structured_learning_rate=2e-5)

    actor_optimizer, _ = make_optimizers(actor, critic, config)
    backbone = {id(p) for p in actor.backbone_parameters()}
    tagged = [g for g in actor_optimizer.param_groups if g.get("role") == BACKBONE_ROLE]

    assert {id(p) for g in actor_optimizer.param_groups for p in g["params"]} == {
        id(p) for p in actor.parameters()
    }
    assert {id(p) for g in tagged for p in g["params"]} == backbone
    for group in tagged:
        rate = config.resolved_structured_learning_rate
        if group.get("kind") == "adam":
            rate *= config.adam_learning_rate_ratio
        assert group["base_lr"] == pytest.approx(rate)
    for group in actor_optimizer.param_groups:
        if group.get("role") != BACKBONE_ROLE:
            assert not ({id(p) for p in group["params"]} & backbone)


def test_a_detached_backbone_without_its_objective_is_refused() -> None:
    actor, critic = build_lejepa_pair(_tiny_config(policy_shapes_backbone=False))

    with pytest.raises(ValueError, match="no loss"):
        make_optimizers(actor, critic, PpoConfig())
    # With the objective the detached ablation is well defined.
    make_optimizers(actor, critic, _jepa_ppo_config())


@pytest.mark.parametrize("optimizer", ["adamw", "normuon"])
def test_without_its_objective_the_policy_gradient_alone_trains_the_backbone(optimizer) -> None:
    """Through the whole update: the heads and the backbone step together.

    The rewards are replaced by noise because this fixture's own are zero, and
    zero advantages carry no policy gradient. Noise is not a completed-game
    outcome, so the critic is the HL-Gauss head.
    """
    torch.manual_seed(0)
    model_config = _tiny_config(wdl_value=False)
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor, games=1, seed_start=225, episode_steps=8, sampling_seed=53, reward_mode="shaped"
    )
    noise = np.random.default_rng(7).standard_normal(rollout.rewards.shape)
    rollout = replace(rollout, rewards=noise.astype(rollout.rewards.dtype))
    config = PpoConfig(
        optimizer=optimizer,
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    before_backbone = _snapshot(actor.trunk.named_parameters())

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
    )

    assert metrics["actor_updates"] >= 1
    assert _moved(actor.trunk.named_parameters(), before_backbone)
    # The critic reads the backbone detached and holds none of it, so the
    # backbone moved by the policy's gradient and nothing else.
    assert not ({id(p) for p in critic.parameters()} & {id(p) for p in actor.backbone_parameters()})
    backbone_clocks, head_clocks = _warmup_clocks(actor_optimizer)
    assert backbone_clocks == head_clocks == {metrics["actor_updates"]}


def test_frozen_waves_leave_an_objective_free_backbone_untouched() -> None:
    """Critic warmup (`actor_epochs=0`) steps neither the heads nor the backbone."""
    torch.manual_seed(0)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=225,
        episode_steps=8,
        sampling_seed=53,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=False,
    )
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    before = _snapshot(actor.named_parameters())

    update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(43),
        actor_epochs=0,
    )

    assert not _moved(actor.named_parameters(), before)


def _snapshot(named) -> dict[str, Tensor]:
    """The trainable weights as they stand, by name."""
    return {name: value.detach().clone() for name, value in named if value.requires_grad}


def _moved(named, before: dict[str, Tensor]) -> set[str]:
    """Which of the snapshotted weights an update actually changed."""
    return {
        name
        for name, value in named
        if name in before and not torch.equal(value.detach(), before[name])
    }


def _record_backbone_steps(monkeypatch, actor, optimizer) -> list[bool]:
    """Per step of `optimizer`, whether it carried a gradient into the backbone."""
    import kaggriculture.ppo as ppo_module

    steps: list[bool] = []
    original = ppo_module._optimizer_step

    def record(stepped, *arguments, **keywords):
        if stepped is optimizer:
            steps.append(any(value.grad is not None for value in actor.trunk.parameters()))
        return original(stepped, *arguments, **keywords)

    monkeypatch.setattr(ppo_module, "_optimizer_step", record)
    return steps


def _warmup_clocks(optimizer) -> tuple[set[int], set[int]]:
    """The warmup steps of the backbone's parameter groups and of the rest."""
    backbone = {g["warmup_step"] for g in optimizer.param_groups if g.get("role") == BACKBONE_ROLE}
    rest = {g["warmup_step"] for g in optimizer.param_groups if g.get("role") != BACKBONE_ROLE}
    return backbone, rest


@pytest.mark.parametrize("optimizer", ["adamw", "normuon"])
def test_the_backbone_owns_its_warmup_clock_in_the_objective_optimizer(optimizer) -> None:
    """Its own groups, routed as inside the actor, holding exactly the backbone."""
    model_config = _tiny_config()
    actor, _critic = build_lejepa_pair(model_config)
    objective = JepaObjective(model_config)
    config = PpoConfig(
        optimizer=optimizer,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
    )
    built = make_structured_dynamics_optimizer(objective, config, critic=False, actor=actor)

    def held(role_is_backbone: bool) -> list[int]:
        return [
            id(parameter)
            for group in built.param_groups
            if (group.get("role") == BACKBONE_ROLE) == role_is_backbone
            for parameter in group["params"]
        ]

    assert sorted(held(True)) == sorted(id(p) for p in actor.backbone_parameters())
    assert sorted(held(False)) == sorted(id(p) for p in objective.parameters())
    # The actor's cooldown reaches the backbone, which the heads read, and
    # nothing else the objective owns.
    set_lr_cooldown(built, 0.25, role=BACKBONE_ROLE)
    assert {
        (group.get("role") == BACKBONE_ROLE, group.get("cooldown_scale", 1.0))
        for group in built.param_groups
    } == {(True, 0.25), (False, 1.0)}
    if optimizer == "normuon":
        kinds = [g["kind"] for g in built.param_groups if g.get("role") == BACKBONE_ROLE]
        matrices, vectors, _ = route_parameters(actor.trunk)
        assert kinds.count("normuon") == 1 and len(kinds) > 1
        assert {
            id(p)
            for g in built.param_groups
            if g.get("role") == BACKBONE_ROLE and g["kind"] == "normuon"
            for p in g["params"]
        } == {id(p) for p in matrices}
        assert vectors


def test_a_warmup_wave_freezes_the_backbone_and_fits_the_objective(monkeypatch) -> None:
    """The heads read the backbone, so a frozen policy is a frozen backbone.

    A warmup wave holds the policy still so the critic can catch up with it; an
    encoder step would change the policy just the same. What the wave does fit
    is the projector and the predictor, against the backbone as it stands --
    which is how a warm start's fresh objective settles on the cloned encoder
    before anything moves it.
    """
    torch.manual_seed(0)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=231,
        episode_steps=8,
        sampling_seed=67,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        use_bfloat16=False,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
        structured_learning_rate=1.0e-2,
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    objective_optimizer = make_structured_dynamics_optimizer(
        objective, config, critic=False, actor=actor
    )
    backbone_steps = _record_backbone_steps(monkeypatch, actor, objective_optimizer)
    before_actor = _snapshot(actor.named_parameters())
    before_objective = _snapshot(objective.named_parameters())

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(51),
        auxiliary_generator=np.random.default_rng(52),
        actor_epochs=0,
        structured_dynamics=objective,
        structured_dynamics_optimizer=objective_optimizer,
        # As `train_ppo` passes it through a warmup wave.
        structured_actor_auxiliary=False,
    )

    assert metrics["actor_updates"] == 0
    assert metrics["structured_actor_predictor_updates"] > 0
    assert backbone_steps and not any(backbone_steps)
    # The backbone's warmup is still all ahead of it when the policy is released.
    assert _warmup_clocks(objective_optimizer) == ({0}, {len(backbone_steps)})
    assert not _moved(actor.named_parameters(), before_actor)
    assert _moved(objective.named_parameters(), before_objective)


def test_the_backbone_steps_exactly_when_the_policy_does(monkeypatch) -> None:
    """A KL stop freezes the encoder on the very minibatch that trips it.

    Every backbone step moves the policy through the heads, so one taken past
    the stop is a policy change the trust region has already refused. The
    fixture's advantages are identically zero, so the heads never move and the
    first minibatch's KL is only precision noise: the stop below is tripped by
    the backbone's own first step, which is exactly the drift it has to bound.
    """
    torch.manual_seed(0)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=231,
        episode_steps=8,
        sampling_seed=67,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        critic_epochs=2,
        minibatch_size=4,
        lr_warmup_steps=0,
        target_kl=1.0e-6,
        use_bfloat16=False,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
        structured_learning_rate=1.0e-1,
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    objective_optimizer = make_structured_dynamics_optimizer(
        objective, config, critic=False, actor=actor
    )
    backbone_steps = _record_backbone_steps(monkeypatch, actor, objective_optimizer)

    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(51),
        auxiliary_generator=np.random.default_rng(52),
        structured_dynamics=objective,
        structured_dynamics_optimizer=objective_optimizer,
        structured_actor_auxiliary=True,
    )

    assert metrics["kl_early_stop"] == 1
    # The objective kept fitting through the stop and the critic's extra epoch,
    # but the backbone moved on exactly the minibatches the policy stepped on.
    assert len(backbone_steps) > metrics["actor_updates"] >= 1
    assert sum(backbone_steps) == metrics["actor_updates"]
    assert all(backbone_steps[: int(metrics["actor_updates"])])
    # Its warmup clock counts the steps it took, in lockstep with the heads'.
    assert _warmup_clocks(objective_optimizer) == (
        {int(metrics["actor_updates"])},
        {len(backbone_steps)},
    )
    assert {g["warmup_step"] for g in actor_optimizer.param_groups} == {
        int(metrics["actor_updates"])
    }


def test_the_frozen_actor_warm_pass_traces_the_branch_the_release_wave_uses(
    monkeypatch,
) -> None:
    """The warm pass exists to keep compilation out of the release wave.

    `jepa_horizon_loss` branches on whether a row weight was handed to it, and
    `None` against a tensor is a Dynamo guard, so warming without one traces a
    graph the release wave then discards and recompiles -- putting back exactly
    the cost this path was added to move.
    """
    torch.manual_seed(0)
    import kaggriculture.ppo as ppo_module

    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    _pin_quantity_orders(actor)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=229,
        episode_steps=8,
        sampling_seed=61,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        use_bfloat16=False,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
    )
    objective = JepaObjective(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    warmed: list[dict] = []
    left_behind: list[str] = []
    original = ppo_module._warm_actor_update_graphs

    def record(*arguments, **keywords):
        warmed.append(keywords)
        result = original(*arguments, **keywords)
        # Checked here rather than after the update, because the release wave
        # legitimately leaves its own gradients behind. The warm pass runs a
        # real backward and promises to leave nothing: the backbone is the one
        # place that promise is easy to break, since it belongs to the
        # objective's optimizer rather than to the objective's module tree and
        # `JepaObjective.zero_grad` does not reach it.
        left_behind.extend(
            name
            for name, value in actor.named_parameters()
            if value.grad is not None and name.startswith("trunk.")
        )
        left_behind.extend(
            name for name, value in objective.named_parameters() if value.grad is not None
        )
        return result

    monkeypatch.setattr(ppo_module, "_warm_actor_update_graphs", record)

    update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        config,
        generator=np.random.default_rng(45),
        auxiliary_generator=np.random.default_rng(46),
        actor_epochs=0,
        structured_dynamics=objective,
        structured_dynamics_optimizer=make_structured_dynamics_optimizer(
            objective, config, critic=False, actor=actor
        ),
        structured_actor_auxiliary=True,
    )

    assert len(warmed) == 1
    weight = warmed[0]["auxiliary_sample_weight"]
    assert weight is not None
    assert torch.equal(weight, warmed[0]["sample_weight"])
    assert not left_behind


def test_a_jepa_objective_is_refused_outside_its_family() -> None:
    torch.manual_seed(0)
    entity_config = EntityConfig(
        model_dim=32,
        attention_heads=2,
        attention_kv_heads=1,
        ffn_multiplier=2,
        farm_blocks=1,
        core_layers=2,
    )
    actor = EntityActor(entity_config)
    critic = EntityCritic(entity_config)
    rollout = collect_self_play(
        actor,
        games=1,
        seed_start=231,
        episode_steps=6,
        sampling_seed=17,
        reward_mode="terminal-outcome",
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=1,
        minibatch_size=1 << 12,
        lr_warmup_steps=0,
        use_bfloat16=False,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
    )
    objective = JepaObjective(_tiny_config())
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)

    # Two refusals, because the objective is reachable from two directions: the
    # optimizer that would have to own a backbone this actor does not have, and
    # the update that would have to run an arm this family does not admit.
    with pytest.raises(ValueError, match="shared backbone"):
        make_structured_dynamics_optimizer(objective, config, critic=False, actor=actor)

    with pytest.raises(ValueError, match="lejepa"):
        update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(1),
            auxiliary_generator=np.random.default_rng(2),
            structured_dynamics=objective,
            structured_dynamics_optimizer=torch.optim.AdamW(
                list(objective.parameters()), lr=1.0e-3
            ),
            structured_actor_auxiliary=True,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_compiled_bf16_update_settles_and_a_slice_refresh_never_retraces() -> None:
    """The contract the objective was shaped around.

    Slice directions and the tile sample change every minibatch. They live in
    non-persistent buffers written in place, so a settled wave must compile
    nothing: any event here means the objective is retracing the whole update
    once per minibatch, which is the difference between an affordable auxiliary
    and an unaffordable one.
    """
    torch.manual_seed(37)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    actor, critic = actor.cuda(), critic.cuda()
    _pin_quantity_orders(actor)
    rollout = collect_self_play_rust(
        actor,
        games=2,
        seed_start=233,
        episode_steps=EPISODE_STEPS,
        sampling_seed=61,
        reward_mode="terminal-outcome",
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    config = PpoConfig(
        optimizer="adamw",
        epochs=2,
        minibatch_size=1536,
        lr_warmup_steps=0,
        target_kl=1.0,
        use_bfloat16=True,
        update_compile_mode="default",
        structured_learning_rate=1.0e-2,
        structured_critic_learning_rate=1.0e-2,
        jepa_prediction_coefficient=1.0,
        jepa_sigreg_coefficient=0.09,
        jepa_reward_coefficient=0.1,
    )
    objective = JepaObjective(model_config).cuda()
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    arms = dict(
        structured_dynamics=objective,
        structured_dynamics_optimizer=make_structured_dynamics_optimizer(
            objective, config, critic=False, actor=actor
        ),
        structured_actor_auxiliary=True,
    )
    watch = CompileWatch()

    metrics = None
    for wave in range(2):
        metrics = update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            # The auxiliary stream is held fixed across waves because it draws
            # the contiguous-run shuffle as well as the slice directions and the
            # tile sample, keeping both waves' inputs comparable. The buffers
            # still refresh on every one of this wave's minibatches, which is
            # what the claim is about.
            generator=np.random.default_rng(70 + wave),
            auxiliary_generator=np.random.default_rng(80),
            **arms,
        )
        events, _ = watch.drain()
        if wave:
            assert not events, [event.describe() for event in events]
        else:
            assert events, "the first wave must compile, not fall back to eager"

    assert metrics is not None
    assert metrics["structured_actor_predictor_updates"] > 1
    for name in JEPA_METRICS:
        assert math.isfinite(metrics[f"structured_actor_{name}"]), name
    assert metrics["structured_actor_dispersion"] > 0.0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
# One segment, and two: 33 games split 17 self-play / 3 self-play + 13 league.
@pytest.mark.parametrize(("self_play_games", "league_games"), [(1, 2), (20, 13)])
def test_collected_behavior_values_match_the_replay_and_leave_the_wave_alone(
    self_play_games: int, league_games: int
) -> None:
    """What the update would have replayed, read off the rollout's own encoding.

    Not bitwise: the rollout reads the belief its captured bf16 actor forward
    produced, and the replay re-encodes under the update's compiled forward. The
    difference must stay at bf16 rounding of the value, far below its spread.
    The sampled wave itself must not notice the critic was riding along.
    """
    torch.manual_seed(41)
    model_config = _tiny_config()
    actor, critic = build_lejepa_pair(model_config)
    actor, critic = actor.cuda(), critic.cuda()
    with torch.no_grad():
        critic.value_head.weight.normal_(std=0.5)
    opponent = LejepaActor(model_config).cuda().requires_grad_(False)

    def collect(value_critic):
        return collect_mixed_play_rust(
            actor,
            (opponent,),
            self_play_games=self_play_games,
            league_games=league_games,
            opponent_indices=np.zeros(league_games, dtype=np.int64),
            seed_start=241,
            sampling_seed=67,
            forward_mode="inductor_graph",
            forward_autocast=True,
            critic=value_critic,
        )

    plain = collect(None)
    watch = CompileWatch()
    carried = collect(critic)
    events, _ = watch.drain()
    # The tail compiles with the wave that first reads it instead of running
    # eagerly, and every later wave of the same shape reuses it.
    assert any(event.name == "_collected_value_tail" for event in events), [
        event.describe() for event in events
    ]
    collect(critic)
    events, _ = watch.drain()
    assert not events, [event.describe() for event in events]

    assert plain.behavior_values is None
    assert carried.behavior_value_key == behavior_value_key(critic, True)
    assert critic.training
    for name in (
        "valid",
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "old_unit_logprobs",
        "old_market_kind_logprobs",
        "old_market_quantity_logprobs",
        "rewards",
    ):
        np.testing.assert_array_equal(getattr(carried, name), getattr(plain, name), err_msg=name)

    device = torch.device("cuda")
    staged = {name: _stage_tensor(array, device) for name, array in carried.states.items()}
    for name in ("unit_actions", "unit_active"):
        staged[name] = _stage_tensor(getattr(carried, name), device)
    replayed = (
        replay_behavior_values(
            critic, LEJEPA, staged, compile_mode="default", autocast_enabled=True
        )
        .cpu()
        .numpy()
        .reshape(carried.rewards.shape)
    )
    valid = carried.valid
    gap = np.abs(carried.behavior_values[valid] - replayed[valid])
    spread = float(replayed[valid].std())
    assert spread > 0.0
    assert float(gap.max()) <= 0.05 * spread, (float(gap.max()), float(gap.mean()), spread)
