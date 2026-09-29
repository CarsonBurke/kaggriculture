from __future__ import annotations

import argparse
import copy
from dataclasses import replace

import pytest
import torch
from kaggle_environments import make

from kaggriculture import structured
from kaggriculture.actions import N_MARKET_KINDS, N_UNIT_ACTIONS
from kaggriculture.compilewatch import CompileWatch
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS, PRODUCTS
from kaggriculture.entity import EntityConfig
from kaggriculture.latent_dynamics import DecodeHeads
from kaggriculture.model import (
    FarmActor,
    ModelConfig,
    ReluSquared,
    policy_compile_options,
    softcap_value_logits,
)
from kaggriculture.modelargs import add_model_config_arguments, model_config_from_args
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import (
    _CapturedStep,
    _league_layout,
    _stacked_actor_ensemble,
    _StackedActorEnsemble,
)
from kaggriculture.structured import (
    Attention,
    EconomyEmbedder,
    FeedForward,
    FusedFeedForward,
    StructuredActor,
    StructuredConfig,
    StructuredCritic,
    StructuredCriticBelief,
    StructuredInputs,
    refresh_fused_mlp_fp8,
    stack_structured,
)
from kaggriculture.structured_dynamics import (
    StructuredCriticDynamics,
    _latent_smooth_l1,
    structured_critic_window_loss,
)
from kaggriculture.tokens import (
    FARM_TOKEN_FIELDS,
    PRODUCT_PRIVATE_FIELDS,
    PRODUCT_TOKEN_FIELDS,
    TOWN_TOKEN_FIELDS,
    encode_structured_observation,
    product_private_fields,
    product_token_fields,
    town_token_fields,
)
from kaggriculture.triton_mlp import _fused_relu_squared_mlp_bf16


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


def test_structured_latent_loss_matches_reference_element_mean() -> None:
    predicted = torch.zeros(2, 2, 2, requires_grad=True)
    target = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0]], [[9.0, 9.0], [9.0, 9.0]]],
        requires_grad=True,
    )

    loss = _latent_smooth_l1(predicted, target, torch.tensor([True, False]))
    loss.backward()

    assert float(loss.detach()) == pytest.approx(0.5)
    assert predicted.grad is not None
    assert target.grad is None


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_masked_cuda_attention_ignores_padded_context_forward_and_backward() -> None:
    torch.manual_seed(0)
    config = replace(
        _tiny_config(),
        model_dim=128,
        attention_heads=8,
    )
    tiny = Attention(config).cuda().train()
    general = copy.deepcopy(tiny)
    queries = torch.randn(8, 1, 128, device="cuda", requires_grad=True)
    context = torch.randn(8, 5, 128, device="cuda", requires_grad=True)
    general_queries = queries.detach().clone().requires_grad_()
    general_context = context.detach().clone().requires_grad_()
    padding = torch.randn(8, 4, 128, device="cuda", requires_grad=True)
    valid = torch.tensor(
        [
            [False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
            [False, True, False, True, False],
            [True, False, True, False, True],
        ],
        device="cuda",
    )
    general_valid = torch.cat(
        (valid, torch.zeros(8, 4, dtype=torch.bool, device="cuda")),
        dim=1,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        tiny_output = torch.compile(
            tiny, options=policy_compile_options("default"), fullgraph=True
        )(
            queries,
            context,
            context_valid=valid,
        )
        general_output = general(
            general_queries,
            torch.cat((general_context, padding), dim=1),
            context_valid=general_valid,
        )
    upstream = torch.randn_like(tiny_output)
    tiny_output.backward(upstream)
    general_output.backward(upstream)

    assert torch.count_nonzero(tiny_output[0]) == 0
    torch.testing.assert_close(tiny_output, general_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(queries.grad, general_queries.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(context.grad, general_context.grad, rtol=3e-2, atol=3e-2)
    assert torch.count_nonzero(queries.grad[0]) == 0
    assert torch.count_nonzero(context.grad[~valid]) == 0
    assert torch.count_nonzero(padding.grad) == 0
    for tiny_parameter, general_parameter in zip(
        tiny.parameters(), general.parameters(), strict=True
    ):
        assert tiny_parameter.grad is not None
        assert general_parameter.grad is not None
        torch.testing.assert_close(
            tiny_parameter.grad,
            general_parameter.grad,
            rtol=3e-2,
            atol=3e-2,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("model_dim,heads", [(80, 4), (128, 4)])
def test_attention_policy_is_invariant_to_rollout_and_update_batch_sizes(model_dim, heads) -> None:
    """The same state must not switch BF16 attention arithmetic in a larger batch."""
    if not torch.cuda.is_available():
        pytest.skip("attention batch-size parity requires CUDA")
    # Each parameter case owns its shape/mode specializations, not prior tests'.
    torch._dynamo.reset_code(Attention.forward.__code__)
    torch.manual_seed(0)
    config = replace(
        _tiny_config(),
        model_dim=model_dim,
        attention_heads=heads,
        attention_kv_heads=heads // 2,
    )
    module = torch.compile(
        Attention(config).cuda().train(),
        options=policy_compile_options("default"),
        fullgraph=True,
        dynamic=False,
    )
    # Farm attention doubles actor rows; cover a rollout row and a PPO batch.
    query = torch.randn(1, 100, model_dim, device="cuda")
    context = torch.randn(1, 100, model_dim, device="cuda")
    valid = torch.ones(1, 100, dtype=torch.bool, device="cuda")
    valid[:, 70:] = False

    def run(batch: int) -> tuple[torch.Tensor, ...]:
        queries = query.repeat(batch, 1, 1).requires_grad_()
        contexts = context.repeat(batch, 1, 1).requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(queries, contexts, context_valid=valid.repeat(batch, 1))
        query_grad, context_grad = torch.autograd.grad(output[0].sum(), (queries, contexts))
        return output[0].detach(), query_grad[0], context_grad[0]

    rollout = run(1)
    update = run(840)
    for left, right in zip(rollout, update, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
    for mode in (torch.no_grad, torch.inference_mode):
        with mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in (1, 840):
                output = module(
                    query.repeat(batch, 1, 1),
                    context.repeat(batch, 1, 1),
                    context_valid=valid.repeat(batch, 1),
                )
                torch.testing.assert_close(output[0], rollout[0], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("kv_heads,head_dim", [(2, 20), (2, 32), (4, 32)])
@pytest.mark.parametrize("mask_axes", [None, (1, 1), (1, 12), (4, 1), (4, 12)])
def test_fused_attention_matches_reference_forward_and_backward(
    device: str, kv_heads: int, head_dim: int, mask_axes: tuple[int, int] | None
) -> None:
    """GQA preserves key-, query-, and head-specific masks and fully masked rows."""
    torch.manual_seed(0)
    heads = 4
    # Exercise token-major projection strides, including unpadded GQA heads.
    query = (
        torch.randn(4, 12, heads, head_dim, dtype=torch.bfloat16, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    key = (
        torch.randn(4, 17, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    value = (
        torch.randn(4, 17, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    mask = None
    if mask_axes is not None:
        mask = torch.rand(4, *mask_axes, 17, device=device) > 0.3
        mask[0] = False
    forward = structured._fused_attention
    if device == "cuda":
        # Each mask/stride/head case deliberately owns a distinct specialization.
        torch._dynamo.reset_code(structured._fused_attention.__code__)
        forward = torch.compile(forward, options=policy_compile_options("default"), fullgraph=True)
    actual = forward(query, key, value, mask, enable_gqa=heads != kv_heads, scale=head_dim**-0.5)

    # Independent FP32 mathematics deliberately uses the original head width.
    # Reusing a production attention helper would miss a shared scale/mask bug.
    reference_query, reference_key, reference_value = (
        tensor.detach().float().requires_grad_() for tensor in (query, key, value)
    )
    repeated_key = reference_key.repeat_interleave(heads // kv_heads, dim=1)
    repeated_value = reference_value.repeat_interleave(heads // kv_heads, dim=1)
    scores = (reference_query @ repeated_key.transpose(-2, -1)) * head_dim**-0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
        scores = torch.where(mask.any(dim=-1, keepdim=True), scores, 0.0)
    probabilities = scores.softmax(dim=-1)
    if mask is not None:
        probabilities = probabilities.masked_fill(~mask, 0.0)
    expected = probabilities @ repeated_value
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.float())

    assert actual.dtype == query.dtype
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    for tensor, reference in zip(
        (query, key, value), (reference_query, reference_key, reference_value), strict=True
    ):
        assert tensor.grad is not None
        assert reference.grad is not None
        torch.testing.assert_close(tensor.grad.float(), reference.grad, rtol=3e-2, atol=3e-2)
        if mask_axes is not None:
            assert torch.count_nonzero(tensor.grad[0]) == 0
    if mask_axes is not None:
        assert torch.count_nonzero(actual[0]) == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_masked_attention_supports_production_unit_decoder_batch() -> None:
    """Unit decoding exceeds cuDNN's 65,535-row limit even with short contexts."""
    torch.manual_seed(0)
    query = torch.randn(8, 4, 1, 20, dtype=torch.bfloat16, device="cuda")
    key = torch.randn(8, 2, 5, 20, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    mask = torch.rand(8, 1, 1, 5, device="cuda") > 0.3
    mask[0] = False
    forward = torch.compile(
        structured._fused_attention,
        options=policy_compile_options("default"),
        fullgraph=True,
        dynamic=False,
    )
    repeats = 10_240
    with torch.inference_mode():
        expected = forward(query, key, value, mask, enable_gqa=True, scale=20**-0.5)
        actual = forward(
            query.repeat(repeats, 1, 1, 1),
            key.repeat(repeats, 1, 1, 1),
            value.repeat(repeats, 1, 1, 1),
            mask.repeat(repeats, 1, 1, 1),
            enable_gqa=True,
            scale=20**-0.5,
        )
    torch.testing.assert_close(
        actual.reshape(repeats, *expected.shape),
        expected.unsqueeze(0).expand(repeats, *expected.shape),
        rtol=2e-3,
        atol=2e-3,
    )
    assert torch.count_nonzero(actual[::8]) == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_masked_attention_preserves_per_batch_masks_and_gradients() -> None:
    """Cross the Flex grid-Y limit with a nondivisible final batch tile."""
    torch.manual_seed(17)
    batch, heads, kv_heads, width = 65537, 4, 2, 24
    query = torch.randn(batch, heads, 1, width, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(batch, kv_heads, 5, width, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query.requires_grad_()
    key.requires_grad_()
    value.requires_grad_()
    mask = torch.rand(batch, 1, 1, 5, device="cuda") > 0.3
    mask[::32768] = False
    forward = torch.compile(
        structured._fused_attention,
        options=policy_compile_options("default"),
        fullgraph=True,
        dynamic=False,
    )
    actual = forward(query, key, value, mask, enable_gqa=True, scale=width**-0.5)
    rq, rk, rv = (tensor.detach().float().requires_grad_() for tensor in (query, key, value))
    scores = (rq @ rk.repeat_interleave(heads // kv_heads, dim=1).transpose(-2, -1)) * width**-0.5
    scores = scores.masked_fill(~mask, -torch.inf)
    scores = torch.where(mask.any(dim=-1, keepdim=True), scores, 0.0)
    weights = scores.softmax(dim=-1).masked_fill(~mask, 0.0)
    expected = weights @ rv.repeat_interleave(heads // kv_heads, dim=1)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.float())
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    for tensor, reference in zip((query, key, value), (rq, rk, rv), strict=True):
        assert tensor.grad is not None and reference.grad is not None
        torch.testing.assert_close(tensor.grad.float(), reference.grad, rtol=3e-2, atol=3e-2)
        assert torch.count_nonzero(tensor.grad[::32768]) == 0
    assert torch.count_nonzero(actual[::32768]) == 0


def test_hardware_native_mlp_rejects_cpu_execution() -> None:
    module = FusedFeedForward(
        replace(
            _tiny_config(),
            model_dim=128,
            attention_heads=4,
            ffn_multiplier=2,
            fused_mlp=True,
        )
    ).eval()

    with pytest.raises(RuntimeError, match="require CUDA"):
        module(torch.randn(2, 4, 128, dtype=torch.bfloat16))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hardware_native_mlp_compiles_fp8_forward_and_backward() -> None:
    torch.manual_seed(0)
    config = replace(
        _tiny_config(),
        model_dim=128,
        attention_heads=4,
        ffn_multiplier=2,
        fused_mlp=True,
    )
    module = FusedFeedForward(config).cuda().train()
    refresh_fused_mlp_fp8(module, bootstrap_down=True)
    values = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(module, options=policy_compile_options("default"), fullgraph=True)

    compiled(values).float().square().mean().backward()

    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert module.up_weight.grad is not None and torch.isfinite(module.up_weight.grad).all()
    assert module.down_weight.grad is not None and torch.isfinite(module.down_weight.grad).all()
    prior = module._up_weight_f8.clone()
    with torch.no_grad():
        module.up_weight.add_(0.01)
    refresh_fused_mlp_fp8(module, bootstrap_down=False)
    assert not torch.equal(prior, module._up_weight_f8)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hardware_native_mlp_compiles_bf16_backward() -> None:
    torch.manual_seed(0)
    config = replace(
        _tiny_config(),
        model_dim=128,
        attention_heads=4,
        ffn_multiplier=2,
        fused_mlp=True,
    )
    module = FusedFeedForward(config).cuda().eval()
    values = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    torch.compile(module, options=policy_compile_options("default"), fullgraph=True)(
        values
    ).float().square().mean().backward()

    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert module.up_weight.grad is not None and torch.isfinite(module.up_weight.grad).all()
    assert module.down_weight.grad is not None and torch.isfinite(module.down_weight.grad).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hardware_native_actor_runs_autocast_with_fp32_master_weights(
    real_inputs: StructuredInputs,
) -> None:
    torch.manual_seed(0)
    config = replace(
        _tiny_config(),
        model_dim=128,
        attention_heads=4,
        ffn_multiplier=2,
        fused_mlp=True,
    )
    actor = StructuredActor(config).cuda().train()
    refresh_fused_mlp_fp8(actor, bootstrap_down=True)
    cuda_inputs = StructuredInputs(*(field.cuda() for field in real_inputs))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = actor(cuda_inputs)
        loss = (
            output.unit_logits.float().square().mean()
            + output.market_kind_logits.float().square().mean()
            + output.market_quantity_context.float().square().mean()
        )
    loss.backward()

    fused_parameters = [
        parameter
        for name, parameter in actor.named_parameters()
        if name.endswith(("up_weight", "down_weight"))
    ]
    assert fused_parameters
    assert all(parameter.dtype == torch.float32 for parameter in fused_parameters)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in fused_parameters
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hardware_native_structured_ensemble_compiles_batched_forward(
    real_inputs: StructuredInputs,
) -> None:
    torch.manual_seed(0)
    config = replace(
        _tiny_config(),
        model_dim=128,
        attention_heads=4,
        ffn_multiplier=2,
        fused_mlp=True,
        global_modulation=True,
    )
    actors = [StructuredActor(config).cuda().eval() for _ in range(2)]
    with torch.no_grad():
        actors[1].unit_head[-1].weight.add_(0.01)
    cuda_inputs = StructuredInputs(*(field.cuda() for field in real_inputs))
    lane_inputs = StructuredInputs(*(torch.stack((field, field)) for field in cuda_inputs))
    ensemble = _StackedActorEnsemble(actors)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = [actor(cuda_inputs) for actor in actors]
        actual = ensemble(lane_inputs, mode="inductor")

    for component, references in (
        (actual.unit_logits, [output.unit_logits for output in expected]),
        (
            actual.market_kind_logits,
            [output.market_kind_logits for output in expected],
        ),
        (
            actual.market_quantity_context,
            [output.market_quantity_context for output in expected],
        ),
    ):
        # The rollout packer explicitly converts every head to FP32. Inductor's
        # functional vmap may retain that dtype while the single-lane autocast
        # reference returns BF16, so the observable contract here is numeric.
        torch.testing.assert_close(
            component,
            torch.stack(references),
            rtol=1e-2,
            atol=1e-1,
            check_dtype=False,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bucketed_ensemble_layouts_reuse_compilation_and_reload_captured_policies(
    real_inputs: StructuredInputs,
) -> None:
    torch.manual_seed(41)
    models = [StructuredActor(_tiny_config()).cuda().eval() for _ in range(4)]
    inputs = StructuredInputs(*(field.cuda() for field in real_inputs))
    watch = CompileWatch()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, (lanes, width) in enumerate(((3, 5), (4, 7), (3, 8), (4, 6))):
                selected = [models[(index + lane) % len(models)] for lane in range(lanes)]
                padded_lanes, padded_width = _league_layout(
                    lanes, width, device=torch.device("cuda"), mode="inductor_graph"
                )
                rows = (
                    torch.arange(padded_lanes * padded_width, device="cuda")
                    % inputs.unit_active.shape[0]
                )
                lane_inputs = StructuredInputs(
                    *(
                        field[rows].view(padded_lanes, padded_width, *field.shape[1:])
                        for field in inputs
                    )
                )
                ensemble = _stacked_actor_ensemble(
                    selected + selected[:1] * (padded_lanes - lanes), namespace=1_000_037
                )
                ensemble(lane_inputs, mode="inductor_graph")
                events, _ = watch.drain()
                if index:
                    assert not events, [event.describe() for event in events]
                else:
                    assert events, "the first forward must compile, not fall back to eager"
                captured = _CapturedStep(
                    lambda ensemble=ensemble, lane_inputs=lane_inputs: ensemble(
                        lane_inputs, mode="inductor_graph"
                    )
                )
                try:
                    # Both same-shape refills and count changes must expose the
                    # newly selected policy, including through an existing graph.
                    for policies in (selected, selected[::-1]):
                        ensemble.load(policies + policies[:1] * (padded_lanes - lanes))
                        actual = captured()
                        expected = [
                            model(StructuredInputs(*(field[lane, :width] for field in lane_inputs)))
                            for lane, model in enumerate(policies)
                        ]
                        for component, references in zip(
                            actual, zip(*expected, strict=True), strict=True
                        ):
                            torch.testing.assert_close(
                                component[:lanes, :width],
                                torch.stack(references),
                                rtol=1e-2,
                                atol=1e-2,
                                check_dtype=False,
                            )
                        del actual
                finally:
                    captured.close()
                events, _ = watch.drain()
                assert not events, [event.describe() for event in events]
    finally:
        watch.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hardware_native_mlp_supports_frozen_ensemble_vmap() -> None:
    torch.manual_seed(0)
    values = torch.randn(3, 2, 4, 128, device="cuda", dtype=torch.bfloat16)
    up_weight = torch.randn(3, 256, 128, device="cuda", dtype=torch.bfloat16)
    down_weight = torch.randn(3, 256, 128, device="cuda", dtype=torch.bfloat16)

    actual, _ = torch.vmap(_fused_relu_squared_mlp_bf16)(
        values,
        up_weight.float(),
        down_weight.float(),
        up_weight,
        down_weight,
    )
    expected = torch.stack(
        [
            torch.relu(value @ up.T).square() @ down
            for value, up, down in zip(values, up_weight, down_weight, strict=True)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-1)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_structured_actor_exposes_typed_training_belief(
    real_inputs: StructuredInputs,
) -> None:
    actor = StructuredActor(_tiny_config()).cuda()
    real_inputs = StructuredInputs(*(value.cuda() for value in real_inputs))

    output, belief = actor.forward_with_belief(real_inputs)

    assert output.unit_logits.shape[:2] == (real_inputs.unit_active.shape)
    assert belief.own_patches.shape == (real_inputs.unit_active.shape[0], 100, 32)
    assert belief.opponent_patches.shape == belief.own_patches.shape
    assert belief.opponent_summary.shape == (real_inputs.unit_active.shape[0], 4, 32)
    assert belief.central_latents.shape == (real_inputs.unit_active.shape[0], 8, 32)
    assert belief.unit_decisions.shape == (real_inputs.unit_active.shape[0], 16, 32)
    assert belief.market_decisions.shape == (real_inputs.unit_active.shape[0], 10, 32)

    # Non-unit norm weights expose accidental second normalization. Reconstruct
    # the original Sequential head from its raw input to pin unchanged logits.
    with torch.no_grad():
        actor.unit_head[0].weight.copy_(torch.linspace(0.5, 1.5, actor.config.model_dim))
    raw_units = []
    handle = actor.unit_head[0].register_forward_pre_hook(
        lambda _module, arguments: raw_units.append(arguments[0])
    )
    try:
        output, belief = actor.forward_with_belief(real_inputs)
    finally:
        handle.remove()
    torch.testing.assert_close(output.unit_logits, actor.unit_head(raw_units[0]), rtol=0, atol=0)
    torch.testing.assert_close(
        belief.unit_decisions, actor.unit_head[0](raw_units[0]), rtol=0, atol=0
    )
    decoded = DecodeHeads.from_actor(actor, normalized_units=True).decode(
        torch.cat((belief.unit_decisions, belief.market_decisions), dim=1)
    )
    torch.testing.assert_close(
        tuple(decoded), tuple(value.float() for value in output), rtol=2e-2, atol=2e-3
    )
    kinds = output.market_kind_logits.argmax(dim=-1)
    torch.testing.assert_close(
        DecodeHeads.from_actor(actor, normalized_units=True).quantity_logits(
            decoded.market_quantity_context, kinds
        ),
        actor.quantity_logits(output.market_quantity_context, kinds),
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fuse_unit_decoder", [False, True])
@pytest.mark.parametrize("fuse_market_decoder", [False, True])
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_actor_auxiliary_export_preserves_policy_and_checkpoint(
    real_inputs: StructuredInputs,
    fuse_unit_decoder: bool,
    fuse_market_decoder: bool,
) -> None:
    torch.manual_seed(94)
    config = replace(
        _tiny_config(),
        fuse_unit_decoder=fuse_unit_decoder,
        fuse_market_decoder=fuse_market_decoder,
    )
    actor = StructuredActor(config).cuda()
    checkpoint = copy.deepcopy(actor.state_dict())
    inputs = StructuredInputs(*(value[:3].cuda() for value in real_inputs))
    expected, bc_belief = actor.forward_with_belief(inputs)
    output, belief = actor.forward_with_auxiliary_belief(inputs)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # Rematerializing the auxiliary-enabled trunk must preserve ordinary PPO
    # gradients, not merely its forward values.
    parameters = tuple(actor.parameters())
    expected_gradients = torch.autograd.grad(
        sum(value.float().square().mean() for value in expected),
        parameters,
        allow_unused=True,
    )
    actual_gradients = torch.autograd.grad(
        sum(value.float().square().mean() for value in output),
        parameters,
        allow_unused=True,
    )
    torch.testing.assert_close(actual_gradients, expected_gradients, rtol=0, atol=0)
    torch.testing.assert_close(belief.unit_decisions, bc_belief.unit_decisions, rtol=0, atol=0)
    torch.testing.assert_close(belief.market_decisions, bc_belief.market_decisions, rtol=0, atol=0)
    assert all(value.requires_grad for value in belief)
    assert all(value.requires_grad for value in bc_belief)
    torch.testing.assert_close(actor.auxiliary_belief(inputs), belief, rtol=0, atol=0)

    # Auxiliary calls must leave the original checkpoint contract and ordinary
    # inference intact, rather than registering a copied or nested decoder.
    actor.load_state_dict(checkpoint, strict=True)
    restored = StructuredActor(config).cuda()
    restored.load_state_dict(actor.state_dict(), strict=True)
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    torch.testing.assert_close(actor(inputs), expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fused_mlp", [False, True])
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_actor_auxiliary_frozen_decode_keeps_combined_policy_gradients(
    real_inputs: StructuredInputs, fused_mlp: bool
) -> None:
    torch.manual_seed(95)
    config = replace(
        _tiny_config(),
        model_dim=128,
        attention_heads=4,
        fused_mlp=fused_mlp,
        fuse_unit_decoder=fused_mlp,
        fuse_market_decoder=fused_mlp,
    )
    actor = StructuredActor(config).cuda().train()
    if fused_mlp:
        refresh_fused_mlp_fp8(actor, bootstrap_down=True)
    inputs = StructuredInputs(*(value[:3].cuda() for value in real_inputs))
    output, belief = actor.forward_with_auxiliary_belief(inputs)
    predicted = -torch.cat(belief, dim=1)
    predicted.retain_grad()
    heads = DecodeHeads.from_actor(actor, normalized_units=True)
    frozen_decode = torch.compile(
        heads.decode, fullgraph=True, options=policy_compile_options("default")
    )
    student = frozen_decode(predicted)
    teacher = heads.decode(torch.cat(tuple(value.detach() for value in belief), dim=1))
    kinds = teacher.market_kind_logits.argmax(dim=-1)
    student_logits = (
        student.unit_logits,
        student.market_kind_logits,
        heads.quantity_logits(student.market_quantity_context, kinds),
    )
    teacher_logits = (
        teacher.unit_logits,
        teacher.market_kind_logits,
        heads.quantity_logits(teacher.market_quantity_context, kinds),
    )
    auxiliary_loss = sum(
        torch.nn.functional.kl_div(
            current.log_softmax(dim=-1),
            target.softmax(dim=-1),
            reduction="batchmean",
        )
        for current, target in zip(student_logits, teacher_logits, strict=True)
    )
    policy_loss = sum(value.float().square().mean() for value in output)
    decoder_parameters = tuple(
        parameter
        for name, parameter in actor.named_parameters()
        if name.startswith(("unit_head.1.", "market_kind.", "market_quantity_"))
    )
    expected_gradients = torch.autograd.grad(
        policy_loss, decoder_parameters, retain_graph=True, allow_unused=True
    )
    assert any(
        gradient is not None and gradient.count_nonzero() > 0 for gradient in expected_gradients
    )

    (policy_loss + auxiliary_loss).backward()

    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert all(parameter.requires_grad for parameter in actor.parameters())
    for parameter, expected_gradient in zip(decoder_parameters, expected_gradients, strict=True):
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    "changes",
    [
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
            "--attention-kv-heads",
            "2",
            "--ffn-multiplier",
            "2",
            "--global-refresh-layers",
            "2,5",
            "--global-refresh-context",
            "all",
            "--zero-init-branches",
            "true",
            "--per-entity-critic",
            "true",
        ]
    )

    config = model_config_from_args(resolve_architecture("structured"), args)

    assert config.attention_kv_heads == 2
    assert config.ffn_multiplier == 2
    assert config.global_refresh_layers == (2, 5)
    assert config.global_refresh_context == "all"
    assert config.zero_init_branches is True
    assert config.per_entity_critic is True


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


def test_tile_embedder_matches_per_token_lookups_and_their_gradients(
    real_inputs: StructuredInputs,
) -> None:
    """The slot-table forward and GEMM backward must equal six plain lookups."""
    torch.manual_seed(0)
    embedder = structured.TileEmbedder(_tiny_config())
    categorical, continuous = real_inputs.tile_categorical, real_inputs.tile_continuous

    def reference() -> torch.Tensor:
        return (
            embedder.kind(categorical[..., 0])
            + embedder.occupant(categorical[..., 1])
            + embedder.farm(categorical[..., 2])
            + embedder.row(categorical[..., 3])
            + embedder.column(categorical[..., 4])
            + embedder.quadrant(categorical[..., 5])
            + embedder.continuous(continuous.float())
        )

    for tokens in (categorical.shape[1], categorical.shape[1] // 2):
        categorical, continuous = categorical[:, :tokens], continuous[:, :tokens]
        weights = torch.randn_like(embedder(categorical, continuous))
        (embedder(categorical, continuous) * weights).sum().backward()
        actual = {name: parameter.grad.clone() for name, parameter in embedder.named_parameters()}
        embedder.zero_grad()
        (reference() * weights).sum().backward()
        expected = {name: parameter.grad.clone() for name, parameter in embedder.named_parameters()}
        embedder.zero_grad()

        torch.testing.assert_close(embedder(categorical, continuous), reference())
        # Batch reductions replace scatter-adds, so only summation order differs.
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


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
        "animals": actor.trunk.economy.animal_projection.weight.grad,
        "town": actor.trunk.economy.town_projection.weight.grad,
        "opponent": actor.trunk.opponent_queries.grad,
        "latents": actor.trunk.latent_queries.grad,
        "core": actor.trunk.core[0].ffn.input.weight.grad,
    }
    for name, gradient in reached.items():
        assert gradient is not None and gradient.abs().sum() > 0, f"no gradient into {name}"


@pytest.mark.parametrize("actor_opponent_farm", [True, False])
def test_actor_opponent_farm_gates_what_the_policy_can_see(
    real_pairs: list[tuple[dict, dict]],
    actor_opponent_farm: bool,
) -> None:
    """The knob is a contract about information, not a speed setting.

    Disabled, the actor must be *exactly* invariant to the opponent half of the
    tile tokens -- not approximately, since it never reads them. Enabled, it
    must not be, or the knob would be measuring nothing. The centralized critic
    keeps both farms either way, which is what makes it safe to gate: the value
    function measurably uses the opponent board even where the policy does not.
    """
    torch.manual_seed(0)
    config = replace(_tiny_config(), actor_opponent_farm=actor_opponent_farm)
    actor = StructuredActor(config).eval()
    critic = StructuredCritic(config).eval()

    rows = [
        encode_structured_observation(observation, opponent["private"])
        for observation, opponent in real_pairs
    ]
    stacked, extras = stack_structured(rows)
    assert extras is not None
    critic_inputs = stacked._replace(
        products=torch.cat((stacked.products, extras.products), dim=-1),
        animals=torch.cat((stacked.animals, extras.animals), dim=-1),
        crops=torch.cat((stacked.crops, extras.crops), dim=-1),
    )
    # The value readout initializes to zero so a fresh critic predicts zero for
    # every state; give it a readout before asking what it depends on.
    torch.nn.init.normal_(critic.value_head.weight, std=0.01)

    def permute_opponent_half(inputs: StructuredInputs) -> StructuredInputs:
        """Reverse the batch order of the opponent farm's tokens.

        Every marginal of the input survives; only the pairing between a state
        and its own opponent's board is destroyed.
        """
        half = slice(structured.TILE_COUNT, 2 * structured.TILE_COUNT)
        categorical = inputs.tile_categorical.clone()
        continuous = inputs.tile_continuous.clone()
        categorical[:, half] = inputs.tile_categorical[:, half].flip(0)
        continuous[:, half] = inputs.tile_continuous[:, half].flip(0)
        return inputs._replace(tile_categorical=categorical, tile_continuous=continuous)

    with torch.no_grad():
        first = actor(stacked)
        second = actor(permute_opponent_half(stacked))
        critic_arguments = (extras.unit_categorical, extras.unit_continuous, extras.unit_active)
        critic_first = critic(critic_inputs, *critic_arguments)
        critic_second = critic(permute_opponent_half(critic_inputs), *critic_arguments)

    if actor_opponent_farm:
        assert not torch.equal(first.unit_logits, second.unit_logits)
        assert hasattr(actor.trunk, "opponent_queries")
    else:
        torch.testing.assert_close(first.unit_logits, second.unit_logits, rtol=0, atol=0)
        torch.testing.assert_close(
            first.market_kind_logits, second.market_kind_logits, rtol=0, atol=0
        )
        # The unused query bank must not reach the optimizer or the checkpoint.
        assert not hasattr(actor.trunk, "opponent_queries")
        assert "opponent_queries" not in actor.trunk.adam_learning_rate_multipliers
    assert not torch.equal(critic_first, critic_second)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("zero_init_branches", [False, True])
def test_state_read_uses_context_content_without_query_shortcut(zero_init_branches) -> None:
    torch.manual_seed(81)
    config = replace(_tiny_config(), zero_init_branches=zero_init_branches)
    block = structured.Block(config, state_read=True).cuda().eval()
    context = torch.randn(2, 1, config.model_dim, device="cuda")
    first_queries = torch.randn(2, 3, config.model_dim, device="cuda")
    other_queries = torch.randn_like(first_queries) * 4
    forward = torch.compile(block, options=policy_compile_options("default"), fullgraph=True)

    # With one context token, queries cannot change what is read. They must not
    # leak into the residual content, even when residual branches initialize zero.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = forward(first_queries, context)
        other = forward(other_queries, context)
    torch.testing.assert_close(first, other, rtol=0, atol=0)
    assert not torch.allclose(first[0], first[1])


@pytest.mark.parametrize("scalar_value", [False, True])
@pytest.mark.parametrize("per_entity_critic", [False, True])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_structured_critic_exposes_only_normalized_value_head_input(
    real_pairs: list[tuple[dict, dict]],
    scalar_value: bool,
    per_entity_critic: bool,
) -> None:
    torch.manual_seed(0)
    config = replace(
        _tiny_config(),
        critic_latents=5,
        scalar_value=scalar_value,
        per_entity_critic=per_entity_critic,
    )
    critic = StructuredCritic(config).cuda()

    rows = [
        encode_structured_observation(observation, opponent["private"])
        for observation, opponent in real_pairs
    ]
    stacked, extras = stack_structured(rows)
    assert extras is not None
    stacked = StructuredInputs(*(value.cuda() for value in stacked))
    extras = type(extras)(*(value.cuda() for value in extras))
    assert extras.products.shape[-1] == len(PRODUCT_PRIVATE_FIELDS)
    assert extras.crops.shape[-1] == 1
    batch = stacked.tile_categorical.shape[0]
    inputs = stacked._replace(
        products=torch.cat((stacked.products, extras.products), dim=-1),
        animals=torch.cat((stacked.animals, extras.animals), dim=-1),
        crops=torch.cat((stacked.crops, extras.crops), dim=-1),
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        initial = critic(
            inputs, extras.unit_categorical, extras.unit_continuous, extras.unit_active
        )
    assert initial.shape == (batch, 1 if scalar_value else config.value_atoms)
    assert torch.isfinite(initial).all()
    # Either zero-initialized readout predicts value zero.
    assert float(critic.value(initial).detach().abs().max()) == pytest.approx(0.0, abs=1e-5)

    torch.nn.init.normal_(critic.value_head.weight, std=0.01)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = critic(
            inputs, extras.unit_categorical, extras.unit_continuous, extras.unit_active
        )
        raw_value = []
        handle = critic.value_norm.register_forward_pre_hook(
            lambda _module, arguments: raw_value.append(arguments[0])
        )
        try:
            actual, belief = critic.forward_with_belief(
                inputs,
                extras.unit_categorical,
                extras.unit_continuous,
                extras.unit_active,
            )
        finally:
            handle.remove()
        assert torch.equal(actual, expected)
        assert belief._fields == ("value_decision",)
        with torch.autocast("cuda", enabled=False):
            normalized = torch.nn.functional.rms_norm(
                raw_value[0], (config.model_dim,), critic.value_norm.weight, eps=1e-5
            )
        torch.testing.assert_close(belief.value_decision, normalized, rtol=0, atol=0)
        torch.testing.assert_close(
            actual,
            (
                critic.value_head(belief.value_decision[:, 0])
                if scalar_value
                else softcap_value_logits(critic.value_head(belief.value_decision[:, 0]))
            ),
            rtol=0,
            atol=0,
        )
        slots = 1 + MAX_UNITS + MAX_MARKET_ORDERS if per_entity_critic else 1
        assert belief.value_decision.shape == (batch, slots, config.model_dim)
        if per_entity_critic:
            entity_logits = critic.decode_entity_belief(belief)
            assert entity_logits.shape == (
                batch,
                slots - 1,
                1 if scalar_value else config.value_atoms,
            )
            torch.testing.assert_close(
                critic.value(entity_logits),
                critic.value(
                    critic.value_head(belief.value_decision[:, 1:])
                    if scalar_value
                    else softcap_value_logits(critic.value_head(belief.value_decision[:, 1:]))
                ),
                rtol=0,
                atol=0,
            )
            assert not torch.equal(entity_logits[:, 0], entity_logits[:, 1])

        product_values = inputs.products.clone()
        product_values[..., -1] += 0.25
        _, product_belief = critic.forward_with_belief(
            inputs._replace(products=product_values),
            extras.unit_categorical,
            extras.unit_continuous,
            extras.unit_active,
        )
        assert not torch.equal(product_belief.value_decision, belief.value_decision)

        animal_values = inputs.animals.clone()
        animal_values[..., -1] += 0.25
        _, animal_belief = critic.forward_with_belief(
            inputs._replace(animals=animal_values),
            extras.unit_categorical,
            extras.unit_continuous,
            extras.unit_active,
        )
        assert not torch.equal(animal_belief.value_decision, belief.value_decision)

        crop_values = inputs.crops.clone()
        crop_values[..., -1] += 0.25
        _, crop_belief = critic.forward_with_belief(
            inputs._replace(crops=crop_values),
            extras.unit_categorical,
            extras.unit_continuous,
            extras.unit_active,
        )
        assert not torch.equal(crop_belief.value_decision, belief.value_decision)

        assert extras.unit_active.any()
        opponent_unit_continuous = extras.unit_continuous.clone()
        opponent_unit_continuous[extras.unit_active] += 0.25
        _, unit_belief = critic.forward_with_belief(
            inputs,
            extras.unit_categorical,
            opponent_unit_continuous,
            extras.unit_active,
        )
        assert not torch.equal(unit_belief.value_decision, belief.value_decision)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_structured_critic_dynamics_loss_is_recursive_and_detached(
    real_inputs: StructuredInputs,
) -> None:
    torch.manual_seed(13)
    config = replace(_tiny_config(), critic_latents=5)
    dynamics = StructuredCriticDynamics(config).cuda()
    rows = 3
    inputs = StructuredInputs(*(value[:rows].cuda() for value in real_inputs))
    belief = StructuredCriticBelief(
        torch.randn(
            rows, 1, config.model_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
    )
    factors = {
        "unit_actions": torch.zeros(rows, MAX_UNITS, dtype=torch.long, device="cuda"),
        "market_kinds": torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.long, device="cuda"),
        "market_quantities": torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.long, device="cuda"),
    }
    value_head = torch.nn.Linear(config.model_dim, config.value_atoms).cuda()

    terms = structured_critic_window_loss(
        dynamics,
        belief,
        inputs,
        factors,
        value_head=value_head,
        horizon=2,
    )

    assert terms.eligible.item() == pytest.approx(1.5)
    assert all(torch.isfinite(term) for term in terms)
    (terms.latent + terms.value).backward()
    for value in belief:
        assert value.grad is not None
        assert value.grad[0].abs().sum() > 0
        assert value.grad[-1].count_nonzero() == 0
    predictor_gradients = [parameter.grad for parameter in dynamics.parameters()]
    assert all(gradient is not None for gradient in predictor_gradients)
    assert sum(gradient.abs().sum() for gradient in predictor_gradients) > 0
    assert dynamics.action.unit_action.weight.grad is not None
    assert dynamics.action.market_kind.weight.grad is not None
    assert value_head.weight.grad is None
    assert value_head.bias.grad is None

    source = StructuredCriticBelief(*(value[:1].detach() for value in belief))
    unit_actions = factors["unit_actions"][:1]
    market_kinds = factors["market_kinds"][:1]
    market_quantities = factors["market_quantities"][:1]
    baseline = dynamics(
        source,
        unit_actions,
        market_kinds,
        market_quantities,
        inputs.unit_categorical[:1],
        inputs.unit_active[:1],
    )
    changed_unit_actions = unit_actions.clone()
    changed_unit_actions[:, 0] = 1
    unit_conditioned = dynamics(
        source,
        changed_unit_actions,
        market_kinds,
        market_quantities,
        inputs.unit_categorical[:1],
        inputs.unit_active[:1],
    )
    changed_market_kinds = market_kinds.clone()
    changed_market_quantities = market_quantities.clone()
    changed_market_kinds[:, 0] = 1
    changed_market_quantities[:, 0] = 1
    market_conditioned = dynamics(
        source,
        unit_actions,
        changed_market_kinds,
        changed_market_quantities,
        inputs.unit_categorical[:1],
        inputs.unit_active[:1],
    )
    assert any(
        not torch.equal(left, right) for left, right in zip(baseline, unit_conditioned, strict=True)
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(baseline, market_conditioned, strict=True)
    )

    with pytest.raises(ValueError, match="horizon must be positive"):
        structured_critic_window_loss(
            dynamics,
            belief,
            inputs,
            factors,
            value_head=value_head,
            horizon=0,
        )
    with pytest.raises(ValueError, match="complete windows"):
        structured_critic_window_loss(
            dynamics,
            belief,
            inputs,
            factors,
            value_head=value_head,
            horizon=3,
        )


def test_structured_config_uses_two_head_gqa_and_two_x_squared_relu() -> None:
    config = StructuredConfig()
    attention = Attention(config)
    feed_forward = FeedForward(config)

    assert config.attention_kv_heads == 2
    assert attention.key_value.out_features == 2 * config.attention_kv_heads * attention.head_dim
    assert isinstance(feed_forward.activation, ReluSquared)
    assert feed_forward.input.out_features == 2 * config.model_dim


def test_structured_defaults_do_not_silently_enable_global_refresh() -> None:
    config = StructuredConfig()

    assert config.global_refresh_layers == ()
    assert config.global_refresh_context == "none"


def test_default_core_interleaves_full_entity_refreshes(
    real_inputs: StructuredInputs,
) -> None:
    config = replace(_tiny_config(), core_layers=8, global_refresh_context="all")
    actor = StructuredActor(config)
    context_lengths: list[int] = []

    def capture_context(_module, arguments) -> None:
        context_lengths.append(arguments[1].shape[1])

    modules = [
        actor.trunk.latent_read.attention,
        *(block.attention for block in actor.trunk.global_refresh.values()),
    ]
    handles = [module.register_forward_pre_hook(capture_context) for module in modules]
    try:
        actor(real_inputs)
    finally:
        for handle in handles:
            handle.remove()

    assert config.global_refresh_layers == (3, 6)
    assert config.global_refresh_context == "all"
    assert context_lengths == [context_lengths[0]] * 3
    assert context_lengths[0] > 100
    for block in actor.trunk.global_refresh.values():
        torch.testing.assert_close(
            block.attention_gate.gate,
            torch.full_like(block.attention_gate.gate, 0.1),
        )


def test_structured_config_validation() -> None:
    with pytest.raises(ValueError, match="attention head width"):
        StructuredConfig(model_dim=24, attention_heads=4)
    for kv_heads in (0, 3, 8):
        with pytest.raises(ValueError, match="attention KV heads"):
            StructuredConfig(attention_heads=4, attention_kv_heads=kv_heads)
    with pytest.raises(ValueError, match="latents"):
        StructuredConfig(latents=0)
    round_trip = StructuredConfig(**StructuredConfig().to_dict())
    assert round_trip == StructuredConfig()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_conditioning_mean_is_batch_invariant_with_correct_gradient() -> None:
    torch.manual_seed(20260912)
    tokens = torch.randn(320, 20, 80, device="cuda")
    row_ids = torch.randperm(5120, device="cuda") % tokens.shape[0]
    compiled = torch.compile(
        structured._token_mean,
        options=policy_compile_options("default"),
        fullgraph=True,
        dynamic=False,
    )
    with torch.inference_mode():
        baseline = compiled(tokens).clone()
        repeated = compiled(tokens.index_select(0, row_ids)).clone()
    torch.testing.assert_close(repeated, baseline[row_ids], rtol=0, atol=0)
    torch.testing.assert_close(baseline, tokens.double().mean(1).float(), rtol=2e-6, atol=1e-7)
    tokens.requires_grad_()
    output = compiled(tokens)
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    torch.testing.assert_close(
        tokens.grad, gradient[:, None, :].expand_as(tokens) / tokens.shape[1], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "config_class", [StructuredConfig, EntityConfig], ids=["structured", "entity"]
)
def test_economy_reads_only_its_schemas_farm_columns(config_class) -> None:
    """v3 weights see exactly the v3 inputs; only a v4 model reads the margin."""
    margin = FARM_TOKEN_FIELDS.index("money_margin")
    generator = torch.Generator().manual_seed(3)

    def economy(width: int) -> tuple[torch.Tensor, ...]:
        shapes = (
            (9, len(PRODUCT_TOKEN_FIELDS)),
            (3, 3),
            (5, 6),
            (2, width),
            (len(TOWN_TOKEN_FIELDS),),
        )
        return tuple(torch.randn(2, *shape, generator=generator) for shape in shapes)

    products, animals, crops, farms, town = economy(len(FARM_TOKEN_FIELDS))
    moved = farms.clone()
    moved[..., margin] += 1.0
    for version, width in ((3, 4), (4, 5), (5, 5), (6, 7)):
        torch.manual_seed(0)
        embedder = EconomyEmbedder(
            config_class(observation_schema_version=version), private_columns=False
        )
        assert embedder.farm_projection.in_features == width
        with torch.no_grad():
            baseline = embedder(products, animals, crops, farms, town)
            shifted = embedder(products, animals, crops, moved, town)
            prefix = embedder(products, animals, crops, farms[..., :width].clone(), town)
        assert torch.equal(prefix, baseline)
        assert torch.equal(shifted, baseline) == (version == 3)


@pytest.mark.parametrize(
    "config_class,split_clock",
    [(StructuredConfig, False), (StructuredConfig, True), (EntityConfig, False)],
    ids=["structured", "structured-split-clock", "entity"],
)
def test_economy_reads_only_its_schemas_town_columns(config_class, split_clock) -> None:
    """v3/v4 weights see exactly their 14 town inputs; only v5 reads unlock ranks."""
    ranks = len(town_token_fields(4))
    generator = torch.Generator().manual_seed(5)
    shapes = (
        (9, len(PRODUCT_TOKEN_FIELDS)),
        (3, 3),
        (5, 6),
        (2, len(FARM_TOKEN_FIELDS)),
        (len(TOWN_TOKEN_FIELDS),),
    )
    products, animals, crops, farms, town = (
        torch.randn(2, *shape, generator=generator) for shape in shapes
    )
    reordered = town.clone()
    reordered[..., ranks:] = reordered[..., ranks:].flip(-1)
    for version, width in ((3, 14), (4, 14), (5, 22), (6, 22)):
        torch.manual_seed(0)
        embedder = EconomyEmbedder(
            config_class(observation_schema_version=version, split_clock_token=split_clock),
            private_columns=False,
        )
        assert embedder.town_projection.in_features == width - 6 * split_clock
        with torch.no_grad():
            baseline = embedder(products, animals, crops, farms, town)
            shifted = embedder(products, animals, crops, farms, reordered)
            prefix = embedder(products, animals, crops, farms, town[..., :width].clone())
        assert torch.equal(prefix, baseline)
        assert torch.equal(shifted, baseline) == (version < 5)


@pytest.mark.parametrize("private_columns", [False, True], ids=["actor", "critic"])
@pytest.mark.parametrize(
    "config_class", [StructuredConfig, EntityConfig], ids=["structured", "entity"]
)
def test_economy_reads_only_its_schemas_product_columns(config_class, private_columns) -> None:
    """v3-v5 weights see exactly their product inputs, public and private alike."""
    staged = len(PRODUCT_TOKEN_FIELDS)
    held = [PRODUCT_TOKEN_FIELDS.index("held_value")]
    if private_columns:
        held.append(staged + PRODUCT_PRIVATE_FIELDS.index("opponent_held_value"))
    generator = torch.Generator().manual_seed(7)
    width = staged + len(PRODUCT_PRIVATE_FIELDS) * private_columns
    shapes = (
        (9, width),
        (3, 3 + 2 * private_columns),
        (5, 6 + private_columns),
        (2, len(FARM_TOKEN_FIELDS)),
        (len(TOWN_TOKEN_FIELDS),),
    )
    products, animals, crops, farms, town = (
        torch.randn(2, *shape, generator=generator) for shape in shapes
    )
    moved = products.clone()
    moved[..., held] += 1.0
    for version in (3, 4, 5, 6):
        public = len(product_token_fields(version))
        private = len(product_private_fields(version)) * private_columns
        torch.manual_seed(0)
        embedder = EconomyEmbedder(
            config_class(observation_schema_version=version), private_columns=private_columns
        )
        assert embedder.product_projection.in_features == public + private
        with torch.no_grad():
            baseline = embedder(products, animals, crops, farms, town)
            shifted = embedder(moved, animals, crops, farms, town)
            # The pre-v6 embedder projected its own contiguous columns.
            legacy = torch.cat(
                (products[..., :public], products[..., staged : staged + private]), dim=-1
            )
            expected = embedder.product_projection(legacy) + embedder.product_identity.weight
        assert torch.equal(baseline[:, : len(PRODUCTS)], expected)
        assert torch.equal(shifted, baseline) == (version < 6)
