"""BiXT contracts; CUDA correctness tests must run through MLQ."""

from __future__ import annotations

import operator
from copy import deepcopy

import pytest
import torch

from kaggriculture.bixt import BiXTRound, shared_score_attention
from kaggriculture.entity import EntityConfig
from kaggriculture.model import policy_compile_options


def test_bidirectional_attention_graph_computes_reference_product_once():
    """Trace symbolic tensors only; the shared score and two value products total three."""
    graph = torch.fx.symbolic_trace(
        shared_score_attention, concrete_args={"output_tokens": None}
    ).graph
    matmuls = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is operator.matmul
    ]
    assert len(matmuls) == 3
    first, *_value_products = matmuls
    assert first.args[0].target == "latent_references"


def _compiled(function):
    return torch.compile(
        function, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")
@pytest.mark.parametrize("tokens,output_tokens", [(246, None), (262, 26), (246, 246), (262, 1)])
@pytest.mark.parametrize("direction", ["both", "latent", "token"])
def test_bounded_attention_matches_dense_bf16_forward_and_backward(
    tokens, output_tokens, direction
):
    from kaggriculture.bixt_attention import bounded_shared_score_attention

    torch.manual_seed(240212141)

    # Cross the internal workspace tile boundary and use the projection layout.
    def projected(count):
        return torch.randn(257, count, 4, 24, device="cuda", dtype=torch.bfloat16).transpose(1, 2)

    inputs = tuple(projected(count).requires_grad_() for count in (32, tokens, 32, tokens))
    valid = torch.ones(257, tokens, dtype=torch.bool, device="cuda")
    valid[:, 2::5] = False
    valid[0, 1:] = False

    def dense(*values):
        return shared_score_attention(*values, valid, output_tokens=output_tokens)

    def bounded(*values):
        return bounded_shared_score_attention(*values, valid, output_tokens=output_tokens)

    expected = _compiled(dense)(*inputs)
    observed = _compiled(bounded)(*inputs)
    for actual, reference in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.015)
    cotangents = tuple(torch.randn_like(value) for value in expected)

    def loss(outputs):
        selected = (0, 1) if direction == "both" else ((0,) if direction == "latent" else (1,))
        return sum((outputs[index].float() * cotangents[index]).sum() for index in selected)

    expected_gradients = torch.autograd.grad(loss(expected), inputs, allow_unused=True)
    observed_gradients = torch.autograd.grad(loss(observed), inputs, allow_unused=True)
    for actual, reference, value in zip(
        observed_gradients, expected_gradients, inputs, strict=True
    ):
        if reference is None:
            reference = torch.zeros_like(value)
        if actual is None:
            actual = torch.zeros_like(value)
        torch.testing.assert_close(actual, reference, atol=0.004, rtol=0.03)
        relative_error = (
            actual.float() - reference.float()
        ).norm() / reference.float().norm().clamp_min(1e-12)
        assert relative_error < 0.015, relative_error.item()
    for index in (1, 3):
        gradient = observed_gradients[index]
        if gradient is not None:
            assert torch.count_nonzero(gradient.masked_select(~valid[:, None, :, None])) == 0
    if direction == "token":
        assert torch.count_nonzero(observed_gradients[3]) == 0
        if output_tokens is not None:
            assert torch.count_nonzero(observed_gradients[1][:, :, output_tokens:]) == 0
    elif direction == "latent":
        assert torch.count_nonzero(observed_gradients[2]) == 0
    assert (
        torch.count_nonzero(observed[1].masked_select(~valid[:, None, :output_tokens, None])) == 0
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")
def test_shared_scores_dense_oracle_values_gradients_and_masks():
    torch.manual_seed(240212138)
    lr = torch.randn(3, 4, 32, 24, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tr = torch.randn(3, 4, 262, 24, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lv = torch.randn_like(lr, requires_grad=True)
    tv = torch.randn_like(tr, requires_grad=True)
    valid = torch.ones(3, 262, dtype=torch.bool, device="cuda")
    valid[:, 2::5] = False

    def actual(lr, tr, lv, tv):
        return shared_score_attention(lr, tr, lv, tv, valid)

    # Independently express both dense directions with einsum and FP32 math.
    score = torch.einsum("bhld,bhnd->bhln", lr.float(), tr.float()) * 24**-0.5
    expected_latent = torch.einsum(
        "bhln,bhnd->bhld",
        score.masked_fill(~valid[:, None, None], -float("inf")).softmax(-1),
        tv.float(),
    )
    expected_token = torch.einsum("bhln,bhld->bhnd", score.softmax(-2), lv.float())
    expected_token = torch.where(valid[:, None, :, None], expected_token, 0)
    observed = _compiled(actual)(lr, tr, lv, tv)
    for left, right in zip(observed, (expected_latent, expected_token), strict=True):
        torch.testing.assert_close(left.float(), right, atol=0.009, rtol=0.035)
    direction_latent, direction_token = (
        torch.randn_like(expected_latent),
        torch.randn_like(expected_token),
    )
    actual_grads = torch.autograd.grad(
        (observed[0].float() * direction_latent).sum()
        + (observed[1].float() * direction_token).sum(),
        (lr, tr, lv, tv),
    )
    reference_grads = torch.autograd.grad(
        (expected_latent * direction_latent).sum() + (expected_token * direction_token).sum(),
        (lr, tr, lv, tv),
    )
    for left, right in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(left.float(), right.float(), atol=0.03, rtol=0.07)
    assert torch.count_nonzero(actual_grads[1].masked_select(~valid[:, None, :, None])) == 0
    assert torch.count_nonzero(actual_grads[3].masked_select(~valid[:, None, :, None])) == 0
    assert torch.isfinite(observed[1]).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")
def test_two_round_token_output_has_no_dead_parameters_and_ignores_padding():
    torch.manual_seed(240212139)
    config = EntityConfig(global_modulation=False)
    first = BiXTRound(config).cuda()
    final = BiXTRound(config, refine_latents=False).cuda()
    latents = torch.randn(2, 32, 96, device="cuda", requires_grad=True)
    tokens = torch.randn(2, 262, 96, device="cuda", requires_grad=True)
    valid = torch.ones(2, 262, dtype=torch.bool, device="cuda")
    valid[:, 2::5] = False

    def forward(latents, tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            updated_latents, updated_tokens = first(latents, tokens, valid)
            return final(updated_latents, updated_tokens[:, :26], valid[:, :26])[1]

    run = _compiled(forward)
    observed = run(latents, tokens)
    with torch.no_grad():
        inference_output = run(latents, tokens)
    torch.testing.assert_close(observed, inference_output, atol=0.025, rtol=0.03)
    direction = torch.randn_like(observed)
    (observed * direction).sum().backward()
    for module in (first, final):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.count_nonzero(tokens.grad.masked_select(~valid.unsqueeze(-1))) == 0
    assert final.token_value is final.latent_output is final.latent_self_attention is None
    altered = torch.where(valid.unsqueeze(-1), tokens, torch.full_like(tokens, 1000.0))
    repeated = run(latents, altered)
    torch.testing.assert_close(observed, repeated, atol=0, rtol=0)
    assert torch.count_nonzero(observed.masked_select(~valid[:, :26, None])) == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required")
def test_penultimate_token_pruning_preserves_outputs_and_gradients():
    torch.manual_seed(240212140)
    config = EntityConfig(global_modulation=False)
    complete = torch.nn.ModuleList(
        [BiXTRound(config, refine_latents=index != 3) for index in range(4)]
    ).cuda()
    pruned = deepcopy(complete)
    pruned[2].output_tokens = 26
    latents = torch.randn(2, 32, 96, device="cuda", requires_grad=True)
    tokens = torch.randn(2, 262, 96, device="cuda", requires_grad=True)
    valid = torch.ones(2, 262, dtype=torch.bool, device="cuda")
    valid[:, 2::5] = False

    def apply(blocks, latents, tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for block in blocks:
                latents, tokens = block(latents, tokens, valid[:, : tokens.shape[1]])
            return tokens[:, :26]

    full_output = _compiled(lambda latents, tokens: apply(complete, latents, tokens))(
        latents, tokens
    )
    pruned_output = _compiled(lambda latents, tokens: apply(pruned, latents, tokens))(
        latents, tokens
    )
    torch.testing.assert_close(full_output, pruned_output, atol=0.025, rtol=0.03)
    direction = torch.randn_like(full_output)
    full_gradients = torch.autograd.grad(
        (full_output * direction).sum(), (*complete.parameters(), latents, tokens)
    )
    pruned_gradients = torch.autograd.grad(
        (pruned_output * direction).sum(), (*pruned.parameters(), latents, tokens)
    )
    for full, small in zip(full_gradients, pruned_gradients, strict=True):
        torch.testing.assert_close(full, small, atol=0.05, rtol=0.08)
