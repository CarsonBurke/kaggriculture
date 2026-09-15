"""Production-width entity contracts. Execute only on an MLQ CUDA allocation.

These are GPU correctness tests, not reduced-training learning evidence. The
native integration case keeps the complete 719-step horizon; throughput belongs
to benchmark_entity_architecture.py and the full production iteration pair.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from kaggriculture.actor_dynamics import ActorDynamics, actor_window_loss
from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.inference import CHECKPOINT_FORMAT_VERSION, load_actor_artifact
from kaggriculture.league import (
    FrozenActorPool,
    load_actor_snapshot,
    save_actor_snapshot,
    snapshot_sha256,
)
from kaggriculture.policy import component_selected_logprobs
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    PpoConfig,
    _actor_batch_args,
    _critic_batch_args,
    _stage_tensor,
    _structured_actor_minibatch_terms,
    _value_objective,
    make_optimizers,
    make_structured_dynamics_optimizer,
    update_ppo,
    update_replay_parity,
)
from kaggriculture.production import production_ppo_config
from kaggriculture.provenance import source_identity
from kaggriculture.registry import ENTITY_ATTENTION
from kaggriculture.rollout import collect_mixed_play_rust, collect_population_play_rust
from kaggriculture.structured import StructuredDecisionBelief
from kaggriculture.structured_dynamics import (
    StructuredCriticDynamics,
    structured_critic_window_loss,
)
from kaggriculture.training import TrainingAgent, load_checkpoint, save_checkpoint

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required"),
]


@pytest.fixture(scope="module")
def config():
    return EntityConfig(model_dim=96, farm_blocks=2, core_layers=4)


@pytest.fixture(scope="module")
def native_rollout(config):
    if not torch.cuda.is_bf16_supported():
        pytest.fail("entity production contracts require CUDA BF16")
    torch.manual_seed(20260914)
    # Fresh inverted-critic learners must still play historical ordinary actors.
    actor = EntityActor(replace(config, critic_inverted_attention=True)).cuda().eval()
    opponent = EntityActor(config).cuda().eval()
    opponent.load_state_dict(actor.state_dict())
    opponent.requires_grad_(False)
    rollout = collect_mixed_play_rust(
        actor,
        [opponent],
        self_play_games=2,
        league_games=2,
        opponent_indices=np.zeros(2, dtype=np.int64),
        seed_start=20260914,
        episode_steps=720,
        sampling_seed=20260915,
        temperature=1.0,
        opponent_temperature=1.0,
        reward_mode="terminal-outcome",
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    return actor, rollout


@pytest.fixture(scope="module")
def native_batch(native_rollout):
    _, rollout = native_rollout
    names = (
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "unit_masks",
        "market_kind_masks",
        "market_quantity_masks",
        "unit_active",
        "market_active",
        "market_quantity_active",
    )
    # One real contiguous native run: actor/critic NextLat targets are actual
    # successors, not unrelated seed rows masquerading as transitions.
    arrays = {**rollout.states, **{name: getattr(rollout, name) for name in names}}
    staged = {
        name: _stage_tensor(array[:1, :32], torch.device("cuda")) for name, array in arrays.items()
    }
    actor_args = _actor_batch_args(ENTITY_ATTENTION, staged, slice(None))
    critic_args = _critic_batch_args(ENTITY_ATTENTION, staged, slice(None), actor_args=actor_args)
    factors = {
        name: staged[name].long() if name in names[:3] else staged[name].bool() for name in names
    }
    return staged, actor_args[0], critic_args, factors


def _compiled(function):
    return torch.compile(function, fullgraph=True, mode="default")


def _nonzero_finite_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients, "the observable loss must reach this model"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.float().abs().sum()) for gradient in gradients) > 0


def test_inverted_critic_flag_preserves_actor_and_initialization(config, native_batch):
    _, inputs, critic_args, factors = native_batch
    inverted_config = replace(config, critic_inverted_attention=True)
    torch.manual_seed(11)
    actor = EntityActor(config).cuda().eval()
    critic = EntityCritic(config).cuda().eval()
    rng = torch.get_rng_state()
    torch.manual_seed(11)
    inverted_actor = EntityActor(inverted_config).cuda().eval()
    inverted_critic = EntityCritic(inverted_config).cuda().eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for ordinary, inverted in ((actor, inverted_actor), (critic, inverted_critic)):
        assert dict(ordinary.named_parameters()).keys() == dict(inverted.named_parameters()).keys()
        ordinary_state, inverted_state = ordinary.state_dict(), inverted.state_dict()
        assert ordinary_state.keys() == inverted_state.keys()
        for name, value in ordinary_state.items():
            torch.testing.assert_close(value, inverted_state[name], rtol=0, atol=0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _compiled(actor)(inputs)
        actual = _compiled(inverted_actor)(inputs)
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(
            actor.quantity_logits(expected.market_quantity_context, factors["market_kinds"]),
            inverted_actor.quantity_logits(actual.market_quantity_context, factors["market_kinds"]),
            rtol=0,
            atol=0,
        )
        ordinary_value = _compiled(critic.encode_belief)(*critic_args).value_decision
        inverted_value = _compiled(inverted_critic.encode_belief)(*critic_args).value_decision
    assert torch.isfinite(inverted_value).all()
    assert not torch.allclose(ordinary_value, inverted_value, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("zero_init_branches", [False, True])
@pytest.mark.parametrize("critic_inverted_attention", [False, True])
def test_one_query_critic_gqa_matches_repeated_kv_belief_and_gradients(
    config, zero_init_branches, critic_inverted_attention
):
    """The production readout must preserve masking and summed GQA cotangents."""
    torch.manual_seed(13)
    critic = EntityCritic(
        replace(
            config,
            zero_init_branches=zero_init_branches,
            critic_inverted_attention=critic_inverted_attention,
        )
    ).cuda()
    batch = 8192
    context = torch.randn(batch, 26, 96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    valid = torch.ones(batch, 26, device="cuda", dtype=torch.bool)
    valid[:, :16] = torch.rand(batch, 16, device="cuda") > 0.5
    valid[0, :16] = False
    valid[1, :16] = True
    attention = critic.pool_attention
    direction = torch.randn(batch, 1, 96, device="cuda", dtype=torch.bfloat16)

    def readout(states):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            query = critic.value_query.unsqueeze(0).expand(batch, -1, -1)
            return critic.value_norm(
                attention(query, critic.pool_norm(states), context_valid=valid)
            )

    def repeated_kv_reference(states):
        # Reuse only the projections/norms, never the production attention
        # helper: explicit repetition independently specifies query-head groups.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            query = critic.value_query.unsqueeze(0).expand(batch, -1, -1)
            query = attention.query(query).view(batch, 1, 4, 24).transpose(1, 2)
            key, value = (
                attention.key_value(critic.pool_norm(states))
                .view(batch, 26, 2, 2, 24)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
            query = attention.query_norm(query)
            key = attention.key_norm(key).repeat_interleave(2, dim=1)
            value = value.repeat_interleave(2, dim=1)
            # FP32 oracle arithmetic is not a production model fallback.
            # Avoid sharing fused SDPA, its padded scale, or its mask layout.
            with torch.autocast("cuda", enabled=False):
                scores = (query.float() @ key.float().transpose(-2, -1)) * 24**-0.5
                scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
                attended = scores.softmax(-1) @ value.float()
            pooled = attended.to(query.dtype).transpose(1, 2).reshape(batch, 1, 96)
            return critic.value_norm(attention.output(pooled))

    actual_readout = _compiled(readout)
    actual = actual_readout(context)
    expected = _compiled(repeated_kv_reference)(context)
    assert actual.shape == (batch, 1, 96) and actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    parameters = (
        critic.value_query,
        *critic.pool_norm.parameters(),
        *attention.parameters(),
        *critic.value_norm.parameters(),
    )
    actual_gradients = torch.autograd.grad(
        (actual.float() * direction).sum(), (context, *parameters)
    )
    expected_gradients = torch.autograd.grad(
        (expected.float() * direction).sum(), (context, *parameters)
    )
    for observed, reference in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(observed).all() and torch.isfinite(reference).all()
        reference_norm = torch.linalg.vector_norm(reference.float())
        assert reference_norm > 0
        relative_error = torch.linalg.vector_norm(observed.float() - reference.float())
        assert relative_error < 0.04 * reference_norm
    # Inspect K and V separately so a correct value path cannot conceal broken
    # key-head mapping or a second accumulation of shared-KV gradients.
    kv_index = 1 + next(
        index
        for index, parameter in enumerate(parameters)
        if parameter is attention.key_value.weight
    )
    for observed, reference in zip(
        actual_gradients[kv_index].chunk(2),
        expected_gradients[kv_index].chunk(2),
        strict=True,
    ):
        assert torch.linalg.vector_norm(reference.float()) > 0
        assert torch.linalg.vector_norm(observed.float() - reference.float()) < (
            0.04 * torch.linalg.vector_norm(reference.float())
        )
    assert torch.count_nonzero(actual_gradients[0][~valid]) == 0
    assert actual_gradients[0][:, 16:].abs().sum() > 0
    changed = context.detach().clone()
    changed[~valid] += 100
    with torch.no_grad():
        unchanged = actual_readout(context.detach())
        perturbed = actual_readout(changed)
    torch.testing.assert_close(unchanged, perturbed, rtol=0, atol=0)


@pytest.mark.parametrize("critic_inverted_attention", [False, True])
def test_inactive_units_cannot_change_live_policy_or_pooled_value(
    config, native_batch, critic_inverted_attention
):
    config = replace(config, critic_inverted_attention=critic_inverted_attention)
    _, inputs, critic_args, _ = native_batch
    torch.manual_seed(17)
    actor = EntityActor(config).cuda().eval()
    critic = EntityCritic(config).cuda().eval()
    actor_forward = _compiled(actor.forward_with_belief)
    critic_forward = _compiled(critic.forward_with_belief)
    inactive = ~inputs.unit_active
    assert inactive.any() and inputs.unit_active.any()
    # Change every inactive unit's continuous content and local relation while
    # retaining legal categorical/gather domains. A mask must erase all paths.
    changed_units = inputs.unit_continuous.clone()
    changed_units[inactive] += 100
    categorical = inputs.unit_categorical.clone()
    categorical[inactive] = 0
    gather = inputs.unit_tile_gather.clone()
    gather[inactive] = 99
    gather_valid = inputs.unit_tile_gather_valid.clone()
    gather_valid[inactive] = True
    changed = inputs._replace(
        unit_continuous=changed_units,
        unit_categorical=categorical,
        unit_tile_gather=gather,
        unit_tile_gather_valid=gather_valid,
    )
    opponent_inactive = ~critic_args[3]
    opponent_units = critic_args[2].clone()
    opponent_units[opponent_inactive] += 100
    opponent_categorical = critic_args[1].clone()
    opponent_categorical[opponent_inactive] = 0
    changed_critic = (
        critic_args[0]._replace(
            unit_continuous=changed_units,
            unit_categorical=categorical,
            unit_tile_gather=gather,
            unit_tile_gather_valid=gather_valid,
        ),
        opponent_categorical,
        opponent_units,
        critic_args[3],
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before, belief = actor_forward(inputs)
        after, changed_belief = actor_forward(changed)
        _, value_belief = critic_forward(*critic_args)
        _, changed_value = critic_forward(*changed_critic)
    assert value_belief.value_decision.shape == (inputs.unit_active.shape[0], 1, 96)
    torch.testing.assert_close(
        before.unit_logits[inputs.unit_active],
        after.unit_logits[inputs.unit_active],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(before.market_kind_logits, after.market_kind_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        before.market_quantity_context, after.market_quantity_context, rtol=0, atol=0
    )
    torch.testing.assert_close(
        value_belief.value_decision, changed_value.value_decision, rtol=0, atol=0
    )
    assert torch.count_nonzero(belief.unit_decisions[inactive]) == 0
    assert torch.count_nonzero(changed_belief.unit_decisions[inactive]) == 0


@pytest.mark.parametrize("critic_inverted_attention", [False, True])
def test_private_state_changes_critic_not_actor_and_has_live_gradients(
    config, native_batch, critic_inverted_attention
):
    config = replace(config, critic_inverted_attention=critic_inverted_attention)
    staged, inputs, critic_args, _ = native_batch
    torch.manual_seed(19)
    actor = EntityActor(config).cuda().eval()
    critic = EntityCritic(config).cuda().eval()
    torch.nn.init.normal_(critic.value_head.weight, std=0.02)
    changed_staged = dict(staged)
    for name in ("critic_products", "critic_animals", "critic_crops", "opponent_unit_continuous"):
        changed_staged[name] = staged[name].float() + 0.5
    public_changed = _actor_batch_args(ENTITY_ATTENTION, changed_staged, slice(None))[0]
    private_changed = _critic_batch_args(ENTITY_ATTENTION, changed_staged, slice(None))
    actor_forward = _compiled(actor)
    critic_forward = _compiled(critic.forward_with_belief)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        before = actor_forward(inputs)
        after = actor_forward(public_changed)
        value_before, belief_before = critic_forward(*critic_args)
        value_after, belief_after = critic_forward(*private_changed)
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.equal(belief_before.value_decision, belief_after.value_decision)
    assert not torch.equal(value_before, value_after)
    # Probe each private input family independently; a zero readout must not
    # turn an absent private-information path into a vacuous passing test.
    private_inputs = critic_args[0]
    leaves = {
        name: getattr(private_inputs, name).detach().clone().requires_grad_()
        for name in ("products", "animals", "crops")
    }
    units = critic_args[2].detach().clone().requires_grad_()
    private_inputs = private_inputs._replace(**leaves)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, belief = critic_forward(private_inputs, critic_args[1], units, critic_args[3])
        direction = torch.linspace(-1, 1, config.model_dim, device="cuda")
        objective = (
            belief.value_decision.float() * direction
        ).sum() + logits.float().square().mean()
    gradients = torch.autograd.grad(objective, (*leaves.values(), units))
    for gradient, public in zip(
        gradients[:3], (inputs.products, inputs.animals, inputs.crops), strict=True
    ):
        private_gradient = gradient[..., public.shape[-1] :]
        assert torch.isfinite(private_gradient).all() and private_gradient.abs().sum() > 0
    assert torch.isfinite(gradients[-1]).all()
    assert gradients[-1][critic_args[3]].abs().sum() > 0


def test_every_round_contributes_gradients_to_shared_memory_kv(config):
    torch.manual_seed(23)
    actor = EntityActor(config).cuda()
    batch = 4
    memory = torch.randn(batch, 220, 96, device="cuda", dtype=torch.bfloat16)
    states = torch.randn(batch, 26, 96, device="cuda", dtype=torch.bfloat16)
    direction = torch.randn_like(states)
    valid = torch.ones(batch, 26, device="cuda", dtype=torch.bool)
    conditioning = torch.randn(batch, 96, device="cuda", dtype=torch.bfloat16)
    # Hold query ancestry constant so an earlier round cannot conceal a
    # detached K/V read in a later round. Each deployed round must deliver
    # nonzero cotangents to BOTH halves of the same live memory projection.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        key, value = _compiled(actor.trunk.memory)(memory)
        losses = [
            (
                _compiled(round_)(states, key, value, valid, None, conditioning).float() * direction
            ).sum()
            for round_ in actor.trunk.core
        ]
    weight = actor.trunk.memory.key_value.weight
    contributions = [torch.autograd.grad(loss, weight, retain_graph=True)[0] for loss in losses]
    for gradient in contributions:
        key_gradient, value_gradient = gradient.chunk(2, dim=0)
        assert torch.isfinite(gradient).all()
        assert key_gradient.abs().sum() > 0 and value_gradient.abs().sum() > 0
    total = torch.autograd.grad(sum(losses), weight)[0]
    torch.testing.assert_close(total, torch.stack(contributions).sum(0), rtol=3e-2, atol=3e-2)


def test_ppo_and_three_head_actor_and_critic_nextlat_combined_compiled_backward(
    config, native_batch
):
    _, inputs, critic_args, factors = native_batch
    torch.manual_seed(29)
    actor = EntityActor(config).cuda().train()
    critic = EntityCritic(config).cuda().train()
    actor_dynamics = ActorDynamics(config).cuda().train()
    critic_dynamics = StructuredCriticDynamics(config).cuda().train()
    torch.nn.init.normal_(critic.value_head.weight, std=0.02)
    ppo = PpoConfig(**production_ppo_config(update_compile_mode="default"))
    rows = inputs.unit_active.shape[0]
    advantages = torch.linspace(-1, 1, rows, device="cuda")
    value_targets = torch.linspace(-0.5, 0.5, rows, device="cuda")
    assert factors["market_quantity_active"][1:].any(), "the batch must exercise quantity targets"

    def behavior_logprobs():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = actor(inputs)
            return component_selected_logprobs(
                output,
                actor.quantity_logits(output.market_quantity_context, factors["market_kinds"]),
                factors["unit_actions"],
                factors["market_kinds"],
                factors["market_quantities"],
                factors["unit_masks"],
                factors["market_kind_masks"],
                factors["market_quantity_masks"],
                validate_masks=False,
            )

    with torch.no_grad():
        old_logprobs = _compiled(behavior_logprobs)()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        critic_belief = _compiled(critic.encode_belief)(*critic_args)
    assert critic_belief.value_decision.shape == (rows, 1, 96)

    def critic_auxiliary():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            terms = structured_critic_window_loss(
                critic_dynamics,
                critic_belief,
                critic_args[0],
                factors,
                value_head=critic.value_head,
                horizon=1,
            )
            return terms.latent + terms.value

    auxiliary = _compiled(critic_auxiliary)()
    auxiliary_inputs = (
        critic_belief.value_decision,
        critic.value_query,
        critic.pool_attention.query.weight,
        critic.pool_attention.key_value.weight,
        critic.pool_attention.output.weight,
        critic.trunk.memory.key_value.weight,
    )
    gradients = torch.autograd.grad(
        auxiliary,
        (*auxiliary_inputs, *critic.value_head.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    for gradient in gradients[: len(auxiliary_inputs)]:
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert all(gradient is None for gradient in gradients[-2:]), (
        "NextLat must not train its value teacher head"
    )
    assert gradients[0][::2].abs().sum() > 0
    assert torch.count_nonzero(gradients[0][1::2]) == 0, "successor beliefs are detached targets"

    def objective():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ppo_terms = _structured_actor_minibatch_terms(
                actor,
                factors["unit_actions"],
                factors["market_kinds"],
                factors["market_quantities"],
                factors["unit_masks"],
                factors["market_kind_masks"],
                factors["market_quantity_masks"],
                factors["unit_active"],
                factors["market_active"],
                factors["market_quantity_active"],
                *old_logprobs,
                advantages,
                ppo.clip_low,
                ppo.clip_high,
                True,
                inputs,
                policy_ratio_scope=ppo.policy_ratio_scope,
            )
            actor_belief = StructuredDecisionBelief(*ppo_terms[5:])
            logits = critic.decode_belief(critic_belief)
            actor_terms = actor_window_loss(
                actor_dynamics,
                actor,
                actor_belief,
                inputs,
                factors,
                decision_horizon=1,
                latent_horizon=1,
            )
            primary = -ppo_terms[0] / rows + _value_objective(critic, logits, value_targets)
            return primary + actor_terms.latent + actor_terms.decision

    loss = _compiled(objective)() + auxiliary
    assert torch.isfinite(loss)
    loss.backward()
    for module in (
        actor,
        critic,
        actor_dynamics.unit_predictor,
        actor_dynamics.market_kind_predictor,
        actor_dynamics.market_quantity_predictor,
        actor_dynamics.action,
        actor_dynamics.action_projection,
        critic_dynamics,
    ):
        _nonzero_finite_gradients(module)
    assert actor.trunk.memory.key_value.weight.grad.abs().sum() > 0
    assert critic.trunk.memory.key_value.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("historical_config", [False, True])
def test_artifact_roundtrip_preserves_compiled_policy_and_rejects_missing_weights(
    config, native_batch, tmp_path, historical_config
):
    _, inputs, _, factors = native_batch
    torch.manual_seed(31)
    actor = EntityActor(config).cuda().eval()
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ENTITY_ATTENTION,
        "model_config": config.to_dict(),
        "actor": actor.state_dict(),
        "source_identity": source_identity(),
        "run_provenance": None,
    }
    if historical_config:
        del payload["model_config"]["critic_inverted_attention"]
    artifact = tmp_path / "entity.pt"
    torch.save(payload, artifact)
    restored, _ = load_actor_artifact(artifact, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _compiled(actor)(inputs)
        actual = _compiled(restored)(inputs)
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(
            restored.quantity_logits(actual.market_quantity_context, factors["market_kinds"]),
            actor.quantity_logits(expected.market_quantity_context, factors["market_kinds"]),
            rtol=0,
            atol=0,
        )
    payload["actor"] = dict(payload["actor"])
    del payload["actor"]["market_quantity_bias"]
    torch.save(payload, artifact)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_actor_artifact(artifact, device="cuda")


def test_historical_actor_warm_start_accepts_inverted_critic_only(config, native_batch, tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_train_ppo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, inputs, _, _ = native_batch
    pretrained = EntityActor(config).cuda().eval()
    historical = config.to_dict()
    del historical["critic_inverted_attention"]
    artifact = tmp_path / "historical-bc.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": ENTITY_ATTENTION,
            "model_config": historical,
            "actor": pretrained.state_dict(),
            "source_identity": source_identity(),
            "run_provenance": None,
        },
        artifact,
    )
    inverted = replace(config, critic_inverted_attention=True)
    actor = EntityActor(inverted).cuda().eval()
    module._load_initial_actor(artifact, actor, ENTITY_ATTENTION, inverted, torch.device("cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _compiled(pretrained)(inputs)
        actual = _compiled(actor)(inputs)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    incompatible = replace(inverted, zero_init_branches=not config.zero_init_branches)
    with pytest.raises(ValueError, match="model configuration"):
        module._load_initial_actor(
            artifact, actor, ENTITY_ATTENTION, incompatible, torch.device("cuda")
        )


@pytest.mark.parametrize("historical_config", [False, True])
def test_frozen_entity_opponents_allow_inverted_critic_without_mutating_snapshots(
    config, native_batch, tmp_path, historical_config
):
    _, inputs, _, _ = native_batch
    actor = EntityActor(config).cuda().eval()
    snapshot = save_actor_snapshot(tmp_path, actor, 0)
    if historical_config:
        payload = torch.load(snapshot.path, map_location="cpu", weights_only=True)
        del payload["model_config"]["critic_inverted_attention"]
        torch.save(payload, snapshot.path)
    digest = snapshot_sha256(snapshot.path)
    inverted = replace(config, critic_inverted_attention=True)
    loaded = load_actor_snapshot(snapshot.path, expected_model_config=inverted, device="cuda")
    pooled = FrozenActorPool(inverted, "cuda").acquire([snapshot.path])[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _compiled(actor)(inputs)
        for opponent in (loaded, pooled):
            actual = _compiled(opponent)(inputs)
            for left, right in zip(expected, actual, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert snapshot_sha256(snapshot.path) == digest
    incompatible = replace(inverted, attention_kv_heads=1)
    with pytest.raises(ValueError, match="model configuration"):
        load_actor_snapshot(snapshot.path, expected_model_config=incompatible, device="cuda")


def test_inverted_critic_recovery_binds_full_configuration(config, tmp_path):
    inverted = replace(config, critic_inverted_attention=True)
    actor = EntityActor(inverted).cuda()
    critic = EntityCritic(inverted).cuda()
    ppo = PpoConfig()
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo)
    agent = TrainingAgent(actor, critic, actor_optimizer, critic_optimizer)
    path = tmp_path / "inverted-recovery.pt"
    save_checkpoint(
        path,
        agents=[agent],
        model_config=inverted,
        ppo_config=ppo,
        iteration=1,
        next_seed=2,
        metrics={},
        source_identity=source_identity(),
    )
    payload = load_checkpoint(path, [agent], device=torch.device("cuda"))
    assert payload["model_config"]["critic_inverted_attention"] is True
    ordinary_actor = EntityActor(config).cuda()
    ordinary_critic = EntityCritic(config).cuda()
    before = {name: value.clone() for name, value in ordinary_actor.state_dict().items()}
    with pytest.raises(ValueError, match="checkpoint model configuration"):
        load_checkpoint(
            path, [TrainingAgent(ordinary_actor, ordinary_critic)], device=torch.device("cuda")
        )
    for name, value in ordinary_actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_critic_checkpoint_roundtrip_rejects_old_pool_and_partial_readout(
    config, native_batch, tmp_path
):
    _, _, critic_args, _ = native_batch
    torch.manual_seed(37)
    critic = EntityCritic(config).cuda().eval()
    torch.nn.init.normal_(critic.value_head.weight, std=0.02)
    path = tmp_path / "entity-critic.pt"
    torch.save(critic.state_dict(), path)
    state = torch.load(path, map_location="cuda", weights_only=True)
    restored = EntityCritic(config).cuda().eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected_logits, expected_belief = _compiled(critic.forward_with_belief)(*critic_args)
        actual_logits, actual_belief = _compiled(restored.forward_with_belief)(*critic_args)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_belief.value_decision, expected_belief.value_decision, rtol=0, atol=0
    )
    old_state = {
        name: value
        for name, value in state.items()
        if name != "value_query" and not name.startswith("pool_attention.")
    }
    old_state["pool_score.weight"] = torch.randn(1, 96, device="cuda")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(old_state, strict=True)
    partial_state = dict(state)
    del partial_state["pool_attention.key_value.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(partial_state, strict=True)


def test_native_population_policy_is_invariant_to_critic_configuration(config, native_rollout):
    actor, _ = native_rollout
    matched = EntityActor(actor.config).cuda().eval()
    historical = EntityActor(config).cuda().eval()
    for member in (matched, historical):
        member.load_state_dict(actor.state_dict())
    arguments = {
        "games": 2,
        "seed_start": 20260918,
        "sampling_seed": 20260919,
        "episode_steps": 720,
        "reward_mode": "terminal-outcome",
        "forward_mode": "inductor_graph",
        "forward_autocast": True,
    }
    expected = collect_population_play_rust([actor, matched], **arguments)
    actual = collect_population_play_rust([actor, historical], **arguments)
    for name in (
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "rewards",
        "valid",
        "final_money",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize(
    ("compile_mode", "critic_inverted_attention"),
    [
        pytest.param("default", False, id="default"),
        pytest.param("reduce-overhead", False, id="reduce-overhead"),
        pytest.param("reduce-overhead", True, id="inverted-reduce-overhead"),
    ],
)
def test_native_full_horizon_ppo_replay_and_update(
    config, native_rollout, compile_mode, critic_inverted_attention
):
    config = replace(config, critic_inverted_attention=critic_inverted_attention)
    behavior_actor, rollout = native_rollout
    actor = EntityActor(config).cuda().eval()
    actor.load_state_dict(behavior_actor.state_dict())
    assert rollout.architecture == ENTITY_ATTENTION
    assert rollout.valid.shape == (6, 719) and rollout.valid.all()
    assert rollout.state_count == 6 * 719
    ppo = replace(
        PpoConfig(**production_ppo_config(update_compile_mode=compile_mode)),
        minibatch_size=8192,
    )
    critic = EntityCritic(config).cuda()
    dynamics = StructuredCriticDynamics(config).cuda()
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo)
    dynamics_optimizer = make_structured_dynamics_optimizer(dynamics, ppo)
    before = actor.market_kind.weight.detach().clone()
    parity = update_replay_parity(
        actor,
        rollout,
        minibatch_size=ppo.minibatch_size,
        compile_mode=ppo.update_compile_mode,
        autocast_enabled=True,
    )
    assert parity["update_replay_max_kl"] <= MAX_UPDATE_REPLAY_KL
    assert parity["update_replay_max_tail_fraction"] <= MAX_UPDATE_REPLAY_TAIL_FRACTION
    metrics = update_ppo(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        rollout,
        ppo,
        generator=np.random.default_rng(20260916),
        structured_critic_dynamics=dynamics,
        structured_critic_dynamics_optimizer=dynamics_optimizer,
        auxiliary_generator=np.random.default_rng(20260917),
    )
    assert metrics["updates"] > 0 and metrics["actor_updates"] > 0
    assert metrics["first_minibatch_component_kl"] <= MAX_FIRST_MINIBATCH_KL
    assert not torch.equal(actor.market_kind.weight, before)
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int)))
