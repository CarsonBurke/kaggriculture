"""Production-width entity contracts. Execute only on an MLQ CUDA allocation.

These are GPU correctness tests, not reduced-training learning evidence. The
native integration case keeps the complete 719-step horizon; throughput belongs
to benchmark_entity_architecture.py and the full production iteration pair.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from kaggriculture.actor_dynamics import ActorDynamics, actor_window_loss
from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.inference import CHECKPOINT_FORMAT_VERSION, load_actor_artifact
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
    update_ppo,
    update_replay_parity,
)
from kaggriculture.production import production_ppo_config
from kaggriculture.provenance import source_identity
from kaggriculture.registry import ENTITY_ATTENTION
from kaggriculture.rollout import collect_mixed_play_rust
from kaggriculture.structured import StructuredDecisionBelief
from kaggriculture.structured_dynamics import (
    StructuredCriticDynamics,
    structured_critic_window_loss,
)

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
    actor = EntityActor(config).cuda().eval()
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


@pytest.mark.parametrize("flag", ["critic_source_read", "memory_writeback", "unit_tile_bias"])
def test_experimental_entity_paths_compiled_backward(config, native_batch, flag):
    """The actual actor/critic loss reaches each proposed information path."""
    _, inputs, critic_args, _ = native_batch
    configuration = replace(config, **{flag: True})
    model = (EntityCritic if flag == "critic_source_read" else EntityActor)(configuration).cuda()
    if flag == "critic_source_read":
        torch.nn.init.normal_(model.value_head.weight, std=0.01)

    def loss():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if flag == "critic_source_read":
                output = model(*critic_args)
            else:
                output = model(inputs).unit_logits
            return output.float().square().mean()

    _compiled(loss)().backward()
    branch = {
        "critic_source_read": "source_pool_norm",
        "memory_writeback": "trunk.source_writeback",
        "unit_tile_bias": "trunk.tile_bias",
    }[flag]
    _nonzero_finite_gradients(model.get_submodule(branch))


def test_unit_tile_bias_dense_gradient_oracle():
    """Relative bias applies only to own units/tiles, with exact per-head GQA routing."""
    from kaggriculture.relative_attention import unit_tile_attention

    torch.manual_seed(71)
    q = torch.randn(3, 4, 26, 24, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(3, 2, 236, 24, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    bias = (torch.randn(361, 4, device="cuda") * 0.1).requires_grad_()
    positions = torch.randint(0, 10, (3, 16, 2), device="cuda")
    valid = torch.ones(3, 236, dtype=torch.bool, device="cuda")
    valid[:, 221::2] = False

    def actual(q, k, v, bias):
        return unit_tile_attention(q, k, v, valid, bias, positions, scale=24**-0.5)

    def reference(q, k, v, bias):
        scores = q.float() @ k.float().repeat_interleave(2, dim=1).transpose(-1, -2)
        scores = scores * 24**-0.5
        tiles = torch.arange(100, device="cuda")
        dx = tiles % 10 - positions[..., 0, None] + 9
        dy = tiles // 10 - positions[..., 1, None] + 9
        offsets = bias[dy * 19 + dx].permute(0, 3, 1, 2)
        scores = scores + torch.nn.functional.pad(offsets, (0, 136, 0, 10))
        scores = scores.masked_fill(~valid[:, None, None], -float("inf"))
        return scores.softmax(-1) @ v.float().repeat_interleave(2, dim=1)

    observed = _compiled(actual)(q, k, v, bias)
    expected = reference(q, k, v, bias)
    torch.testing.assert_close(observed.float(), expected, atol=0.008, rtol=0.02)
    direction = torch.randn_like(expected)
    actual_grad = torch.autograd.grad((observed.float() * direction).sum(), (q, k, v, bias))
    expected_grad = torch.autograd.grad((expected * direction).sum(), (q, k, v, bias))
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left.float(), right.float(), atol=0.02, rtol=0.05)
    with torch.no_grad():
        inference = _compiled(actual)(q, k, v, bias)
    torch.testing.assert_close(inference.float(), expected, atol=0.01, rtol=0.03)


def test_critic_source_read_masked_private_units_and_actor_identity(config, native_batch):
    from kaggriculture.modelargs import actor_model_config

    _, _inputs, critic_args, _ = native_batch
    configuration = replace(config, critic_source_read=True)
    control = replace(config, critic_source_read=False)
    assert actor_model_config(configuration) == actor_model_config(control)
    model = EntityCritic(configuration).cuda()
    categorical, continuous, active = critic_args[1:]
    changed = continuous.clone()
    changed[~active] = 1000

    def belief(extra):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.encode_belief(critic_args[0], categorical, extra, active).value_decision

    read = _compiled(belief)
    torch.testing.assert_close(read(continuous), read(changed), atol=0, rtol=0)


@pytest.mark.parametrize("zero_init_branches", [False, True])
def test_one_query_critic_gqa_matches_repeated_kv_belief_and_gradients(config, zero_init_branches):
    """The production readout must preserve masking and summed GQA cotangents."""
    torch.manual_seed(13)
    critic = EntityCritic(replace(config, zero_init_branches=zero_init_branches)).cuda()
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


@pytest.mark.parametrize("ablations", [False, True], ids=["baseline", "combined-ablations"])
def test_inactive_units_cannot_change_live_policy_or_pooled_value(config, native_batch, ablations):
    _, inputs, critic_args, _ = native_batch
    if ablations:
        config = replace(
            config,
            shared_memory_kv=False,
            inter_attention_ffn=True,
            unit_local_readout=True,
            critic_readout_ffn=True,
            unit_local_init=False,
            tile_cross_rope=True,
        )
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
    with torch.set_grad_enabled(ablations), torch.autocast("cuda", dtype=torch.bfloat16):
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
    if ablations:
        direction = torch.linspace(-1, 1, config.model_dim, device="cuda")
        (
            (belief.unit_decisions.float() * direction).sum()
            + (belief.market_decisions.float() * direction).sum()
            + (value_belief.value_decision.float() * direction).sum()
        ).backward()
        for model in (actor, critic):
            for block in model.trunk.core:
                _nonzero_finite_gradients(block.inter_ffn)
                _nonzero_finite_gradients(block.memory)
        _nonzero_finite_gradients(actor.trunk.unit_local_decoder)
        _nonzero_finite_gradients(critic.value_ffn)


def test_private_state_changes_critic_not_actor_and_has_live_gradients(config, native_batch):
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
    config = replace(config, shared_memory_kv=True)
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


def test_tile_cross_rope_matches_dense_spatial_oracle_and_gradients(config, native_batch):
    """Both farm grids rotate, while market queries and non-tile keys do not."""
    config = replace(config, shared_memory_kv=True)
    _, inputs, _, _ = native_batch
    torch.manual_seed(43)
    actual = EntityActor(replace(config, tile_cross_rope=True)).cuda().trunk
    reference = EntityActor(config).cuda().trunk
    reference.load_state_dict(actual.state_dict())
    inputs = inputs._replace(
        tile_continuous=inputs.tile_continuous.detach().clone().requires_grad_(),
        unit_continuous=inputs.unit_continuous.detach().clone().requires_grad_(),
    )
    batch = inputs.unit_active.shape[0]
    opponent = torch.randn(batch, 16, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    opponent_valid = torch.ones(batch, 16, device="cuda", dtype=torch.bool)
    opponent_valid[:, 1::2] = False
    head_dim = config.model_dim // config.attention_heads
    frequency = 10_000.0 ** (
        -torch.arange(0, head_dim, 4, device="cuda", dtype=torch.float32) / head_dim
    )
    # Independent angle construction, not the production cached rotation tables.
    unit_angles = torch.cat(
        (
            inputs.unit_categorical[..., 3, None] * frequency,
            inputs.unit_categorical[..., 2, None] * frequency,
        ),
        dim=-1,
    )
    query_angles = torch.cat((unit_angles, unit_angles.new_zeros(batch, 10, head_dim // 2)), dim=1)[
        :, None
    ]
    tiles = torch.arange(100, device="cuda")
    tile_angles = torch.cat(
        ((tiles % 10)[:, None] * frequency, (tiles // 10)[:, None] * frequency), dim=-1
    )
    key_angles = torch.cat(
        (tile_angles, tile_angles, tile_angles.new_zeros(36, head_dim // 2)), dim=0
    )[None, None]

    class DenseSpatialRead(torch.nn.Module):
        def __init__(self, read):
            super().__init__()
            self.read = read

        @staticmethod
        def rotate(tensor, angles):
            cosine, sine = angles.cos().to(tensor.dtype), angles.sin().to(tensor.dtype)
            real, imaginary = tensor[..., 0::2], tensor[..., 1::2]
            return torch.stack(
                (real * cosine - imaginary * sine, real * sine + imaginary * cosine), dim=-1
            ).flatten(-2)

        def forward(self, queries, key, value, memory_valid, unit_rotation=None):
            read = self.read
            query = read.query_norm(
                read.query(queries).view(batch, 26, read.heads, read.head_dim).transpose(1, 2)
            )
            query = self.rotate(query, query_angles)
            key = self.rotate(key, key_angles).repeat_interleave(read.heads // read.kv_heads, dim=1)
            value = value.repeat_interleave(read.heads // read.kv_heads, dim=1)
            # FP32 dense oracle only; the production path remains fused CUDA BF16.
            with torch.autocast("cuda", enabled=False):
                scores = query.float() @ key.float().transpose(-1, -2)
                scores = (scores * read.head_dim**-0.5).masked_fill(
                    ~memory_valid[:, None, None], float("-inf")
                )
                attended = scores.softmax(dim=-1) @ value.float()
            return read.output(attended.to(queries.dtype).transpose(1, 2).reshape(batch, 26, 96))

    for block in reference.core:
        block.cross_attention = DenseSpatialRead(block.cross_attention)

    def forward(trunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return trunk(inputs, opponent, opponent_valid)

    observed = _compiled(forward)(actual)
    expected = _compiled(forward)(reference)
    error = torch.linalg.vector_norm(observed.float() - expected.float())
    assert error < 0.02 * torch.linalg.vector_norm(expected.float())
    direction = torch.randn_like(expected)
    roots = (inputs.tile_continuous, inputs.unit_continuous, opponent)
    actual_gradients = torch.autograd.grad(
        (observed.float() * direction).sum(),
        (
            *roots,
            actual.memory.key_value.weight,
            *(block.cross_attention.query.weight for block in actual.core),
        ),
    )
    expected_gradients = torch.autograd.grad(
        (expected.float() * direction).sum(),
        (
            *roots,
            reference.memory.key_value.weight,
            *(block.cross_attention.read.query.weight for block in reference.core),
        ),
    )
    for observed_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        error = torch.linalg.vector_norm(observed_gradient.float() - expected_gradient.float())
        assert error < 0.04 * torch.linalg.vector_norm(expected_gradient.float()) + 1e-6
    assert torch.count_nonzero(actual_gradients[2][~opponent_valid]) == 0


@pytest.mark.parametrize("tile_cross_rope", [False, True], ids=["plain", "tile-rope"])
def test_untied_memory_matches_shared_outputs_and_summed_gradients(config, tile_cross_rope):
    """Tying round-local projections must recover shared-K/V forward and backward."""
    config = replace(config, shared_memory_kv=True, tile_cross_rope=tile_cross_rope)
    shared = EntityActor(config).cuda().trunk
    untied = EntityActor(replace(config, shared_memory_kv=False)).cuda().trunk
    untied.memory_norm.load_state_dict(shared.memory.norm.state_dict())
    for shared_round, untied_round in zip(shared.core, untied.core, strict=True):
        untied_round.load_state_dict(
            {
                **shared_round.state_dict(),
                **{
                    "memory." + name: value
                    for name, value in shared.memory.state_dict().items()
                    if not name.startswith("norm.")
                },
            }
        )
    batch = 4
    memory = torch.randn(batch, 220, 96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    states = torch.randn(batch, 26, 96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    conditioning = torch.randn(batch, 96, device="cuda", dtype=torch.bfloat16)
    state_valid = torch.ones(batch, 26, device="cuda", dtype=torch.bool)
    state_valid[:, 1:16:2] = False
    memory_valid = torch.ones(batch, 220, device="cuda", dtype=torch.bool)
    memory_valid[:, -16:] = False
    direction = torch.randn_like(states)
    unit_rotation = tile_rotation = None
    if tile_cross_rope:
        positions = torch.randint(0, 10, (batch, 16, 2), device="cuda")
        unit_rotation = shared.rope.rotation(positions)
        tile_rotation = (shared.rope.cosine, shared.rope.sine)

    def shared_forward(memory, states):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            key, value = shared.memory(memory, tile_rotation)
            for block in shared.core:
                states = block(
                    states,
                    key,
                    value,
                    state_valid,
                    memory_valid,
                    conditioning,
                    unit_rotation=unit_rotation,
                    tile_rotation=tile_rotation,
                )
            return states

    def untied_forward(memory, states):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            memory = untied.memory_norm(memory)
            for block in untied.core:
                states = block(
                    states,
                    memory,
                    None,
                    state_valid,
                    memory_valid,
                    conditioning,
                    unit_rotation=unit_rotation,
                    tile_rotation=tile_rotation,
                )
            return states

    actual = _compiled(untied_forward)(memory, states)
    expected = _compiled(shared_forward)(memory, states)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    assert torch.count_nonzero(actual[~state_valid]) == 0
    shared_parameters = dict(shared.core.named_parameters())
    untied_parameters = dict(untied.core.named_parameters())
    memory_parameters = dict(shared.memory.named_parameters())
    untied_norm_parameters = dict(untied.memory_norm.named_parameters())
    expected_gradients = torch.autograd.grad(
        (expected.float() * direction).sum(),
        (memory, states, *shared_parameters.values(), *memory_parameters.values()),
    )
    actual_gradients = torch.autograd.grad(
        (actual.float() * direction).sum(),
        (memory, states, *untied_parameters.values(), *untied_norm_parameters.values()),
    )
    actual_by_name = dict(
        zip(untied_parameters, actual_gradients[2 : 2 + len(untied_parameters)], strict=True)
    )
    actual_norm_by_name = dict(
        zip(untied_norm_parameters, actual_gradients[2 + len(untied_parameters) :], strict=True)
    )
    compared = list(zip(actual_gradients[:2], expected_gradients[:2], strict=True))
    compared.extend(
        (actual_by_name[name], gradient)
        for name, gradient in zip(
            shared_parameters, expected_gradients[2 : 2 + len(shared_parameters)], strict=True
        )
    )
    for name, expected_gradient in zip(
        memory_parameters, expected_gradients[2 + len(shared_parameters) :], strict=True
    ):
        contributions = (
            [actual_norm_by_name[name.removeprefix("norm.")]]
            if name.startswith("norm.")
            else [actual_by_name[f"{index}.memory.{name}"] for index in range(config.core_layers)]
        )
        for gradient in contributions:
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        compared.append((torch.stack(contributions).sum(0), expected_gradient))
    for observed, reference in compared:
        assert torch.isfinite(observed).all() and torch.isfinite(reference).all()
        error = torch.linalg.vector_norm(observed.float() - reference.float())
        assert error <= 0.04 * torch.linalg.vector_norm(reference.float()) + 1e-6
    assert torch.count_nonzero(actual_gradients[0][~memory_valid]) == 0
    assert actual_gradients[0][memory_valid].abs().sum() > 0


def test_no_local_init_ignores_gathers_but_preserves_unit_information(config, native_batch):
    _, inputs, critic_args, _ = native_batch
    config = replace(config, unit_local_init=False, unit_local_readout=False)
    torch.manual_seed(31)
    actor = EntityActor(config).cuda().eval()
    # Actor-only final readout must not restore gather dependence in the critic.
    critic = EntityCritic(replace(config, unit_local_readout=True)).cuda().eval()
    actor_forward = _compiled(actor.forward_with_belief)
    critic_forward = _compiled(critic.forward_with_belief)
    changed = {
        "unit_tile_gather": (inputs.unit_tile_gather + 37) % 100,
        "unit_tile_gather_valid": ~inputs.unit_tile_gather_valid,
    }
    own_units = inputs.unit_continuous.detach().clone().requires_grad_()
    opponent_units = critic_args[2].detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        before, belief = actor_forward(inputs._replace(unit_continuous=own_units))
        after, changed_belief = actor_forward(inputs._replace(unit_continuous=own_units, **changed))
        _, critic_belief = critic_forward(
            critic_args[0]._replace(unit_continuous=own_units),
            critic_args[1],
            opponent_units,
            critic_args[3],
        )
        _, changed_critic_belief = critic_forward(
            critic_args[0]._replace(unit_continuous=own_units, **changed),
            critic_args[1],
            opponent_units,
            critic_args[3],
        )
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(belief, changed_belief, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(
        critic_belief.value_decision, changed_critic_belief.value_decision, rtol=0, atol=0
    )
    direction = torch.linspace(-1, 1, config.model_dim, device="cuda")
    # The ablation removes local tile seeding, not inventory, identity, position,
    # or the private opponent memory available to the critic.
    actor_objective = (belief.unit_decisions.float() * direction).sum()
    critic_objective = (critic_belief.value_decision.float() * direction).sum()
    identity_parameters = tuple(
        getattr(actor.trunk.units, name).weight for name in ("role", "slot", "row", "column")
    )
    actor_gradients = torch.autograd.grad(actor_objective, (own_units, *identity_parameters))
    critic_gradients = torch.autograd.grad(critic_objective, (own_units, opponent_units))
    for gradient in (*actor_gradients, *critic_gradients):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert torch.count_nonzero(actor_gradients[0][~inputs.unit_active]) == 0
    assert torch.count_nonzero(critic_gradients[1][~critic_args[3]]) == 0


def test_final_local_readout_is_unit_local_and_masks_invalid_neighbors(config, native_batch):
    _, inputs, _, _ = native_batch
    torch.manual_seed(37)
    actor = (
        EntityActor(replace(config, unit_local_init=False, unit_local_readout=True)).cuda().eval()
    )
    forward = _compiled(actor.forward_with_belief)
    valid = inputs.unit_tile_gather_valid.clone()
    valid[~inputs.unit_active] = False
    # Mask one neighbor even for interior units; HERE stays real for live units.
    valid[..., -1] = False
    inputs = inputs._replace(unit_tile_gather_valid=valid)
    masked_gather = torch.where(
        valid, inputs.unit_tile_gather, (inputs.unit_tile_gather + 41) % 100
    )
    target = torch.zeros_like(inputs.unit_active)
    target[:, 0] = inputs.unit_active[:, 0]
    assert target.any() and (~inputs.unit_active).any()
    targeted_gather = torch.where(
        target.unsqueeze(-1) & valid, (inputs.unit_tile_gather + 17) % 100, inputs.unit_tile_gather
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline, belief = forward(inputs)
        masked, masked_belief = forward(inputs._replace(unit_tile_gather=masked_gather))
        changed, changed_belief = forward(inputs._replace(unit_tile_gather=targeted_gather))
        # Invalid slots still carry relation embeddings after gather masking.
        # Perturbing one must not leak through the decoder's attention mask.
        actor.trunk.units.gather_relation.weight[-1].add_(100)
        masked_relation, relation_belief = forward(inputs)
    for left, right in zip(baseline, masked, strict=True):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip((*baseline, *belief), (*masked_relation, *relation_belief), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for candidate in (belief, masked_belief, changed_belief):
        assert torch.isfinite(candidate.unit_decisions).all()
        assert torch.count_nonzero(candidate.unit_decisions[~inputs.unit_active]) == 0
    assert not torch.equal(belief.unit_decisions[target], changed_belief.unit_decisions[target])
    assert not torch.equal(baseline.unit_logits[target], changed.unit_logits[target])
    torch.testing.assert_close(
        belief.unit_decisions[~target], changed_belief.unit_decisions[~target], rtol=0, atol=0
    )
    torch.testing.assert_close(
        belief.market_decisions, changed_belief.market_decisions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline.market_kind_logits, changed.market_kind_logits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline.market_quantity_context, changed.market_quantity_context, rtol=0, atol=0
    )


def test_critic_readout_ffn_preserves_actor_and_receives_value_gradients(config, native_batch):
    _, inputs, critic_args, _ = native_batch
    enhanced = replace(config, critic_readout_ffn=True)
    torch.manual_seed(41)
    actor = EntityActor(config).cuda().eval()
    torch.manual_seed(41)
    enhanced_actor = EntityActor(enhanced).cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before, belief = _compiled(actor.forward_with_belief)(inputs)
        after, enhanced_belief = _compiled(enhanced_actor.forward_with_belief)(inputs)
    for left, right in zip((*before, *belief), (*after, *enhanced_belief), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    critic = EntityCritic(enhanced).cuda().eval()
    torch.nn.init.normal_(critic.value_head.weight, std=0.02)
    baseline = EntityCritic(config).cuda().eval()
    enhanced_state = critic.state_dict()
    baseline.load_state_dict({name: enhanced_state[name] for name in baseline.state_dict()})
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, value_belief = _compiled(critic.forward_with_belief)(*critic_args)
        with torch.no_grad():
            baseline_logits, baseline_belief = _compiled(baseline.forward_with_belief)(*critic_args)
    assert not torch.equal(value_belief.value_decision, baseline_belief.value_decision)
    assert not torch.equal(logits, baseline_logits)
    direction = torch.linspace(-1, 1, config.value_atoms, device="cuda")
    (logits.float() * direction).sum().backward()
    _nonzero_finite_gradients(critic.value_ffn)
    _nonzero_finite_gradients(critic.value_ffn_norm)
    _nonzero_finite_gradients(critic.value_ffn_gate)
    _nonzero_finite_gradients(critic.pool_attention)


def test_ppo_and_three_head_actor_and_critic_nextlat_combined_compiled_backward(
    config, native_batch
):
    _, inputs, critic_args, factors = native_batch
    assert config.shared_memory_kv is False
    torch.manual_seed(29)
    actor = EntityActor(config).cuda().train()
    critic = EntityCritic(config).cuda().train()
    assert actor.trunk.memory is critic.trunk.memory is None
    critic_memory_weights = tuple(round_.memory.key_value.weight for round_ in critic.trunk.core)
    assert len(critic_memory_weights) == config.core_layers
    assert len({id(weight) for weight in critic_memory_weights}) == config.core_layers
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
        *critic_memory_weights,
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
    memory_gradients = gradients[: len(auxiliary_inputs)][-len(critic_memory_weights) :]
    for gradient in memory_gradients:
        key_gradient, value_gradient = gradient.chunk(2, dim=0)
        assert key_gradient.abs().sum() > 0 and value_gradient.abs().sum() > 0
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
            actor_belief = StructuredDecisionBelief(*ppo_terms[8:])
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
    for module in (actor, critic):
        weights = tuple(round_.memory.key_value.weight for round_ in module.trunk.core)
        assert len(weights) == config.core_layers
        assert len({id(weight) for weight in weights}) == config.core_layers
        for weight in weights:
            gradient = weight.grad
            assert gradient is not None and torch.isfinite(gradient).all()
            key_gradient, value_gradient = gradient.chunk(2, dim=0)
            assert key_gradient.abs().sum() > 0 and value_gradient.abs().sum() > 0


def test_artifact_roundtrip_preserves_compiled_policy_and_rejects_missing_weights(
    config, native_batch, tmp_path
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


@pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead"])
def test_native_full_horizon_ppo_replay_and_update(config, native_rollout, compile_mode):
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
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo)
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
    )
    assert metrics["updates"] > 0 and metrics["actor_updates"] > 0
    assert metrics["first_minibatch_component_kl"] <= MAX_FIRST_MINIBATCH_KL
    assert not torch.equal(actor.market_kind.weight, before)
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int)))
