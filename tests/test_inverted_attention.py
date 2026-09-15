from __future__ import annotations

import pytest
import torch

from kaggriculture.triton_inverted_attention import inverted_attention

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _oracle(query, key, value, query_valid, memory_valid, scale):
    """Independent double-precision weighted-mean definition, on the GPU."""
    repeats = query.shape[1] // key.shape[1]
    key = key.double().repeat_interleave(repeats, dim=1)
    value = value.double().repeat_interleave(repeats, dim=1)
    scores = (query.double() @ key.transpose(-1, -2)) * scale
    valid = query_valid[:, None, :, None]
    if memory_valid is not None:
        valid = valid & memory_valid[:, None, None, :]
    scores = scores.masked_fill(~valid, -torch.inf)
    maximum = scores.amax(dim=-2, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.where(valid, (scores - maximum).exp(), 0.0)
    weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1.0)
    mass = weights.sum(dim=-1, keepdim=True)
    weights = weights / mass.clamp_min(torch.finfo(torch.float64).tiny)
    return (weights @ value).to(query.dtype)


def _inputs(dtype, queries=26):
    generator = torch.Generator(device="cuda").manual_seed(817)
    # Match the head-transposed views and interleaved K/V storage used by EntityMemory.
    query = torch.randn(3, queries, 4, 24, device="cuda", dtype=dtype, generator=generator)
    query = query.transpose(1, 2).detach().requires_grad_()
    memory = torch.randn(3, 181, 2, 2, 24, device="cuda", dtype=dtype, generator=generator)
    key, value = memory.permute(2, 0, 3, 1, 4).unbind(0)
    key = key.detach().requires_grad_()
    value = value.detach().requires_grad_()
    query_valid = torch.ones(3, queries, device="cuda", dtype=torch.bool)
    memory_valid = torch.ones(3, 181, device="cuda", dtype=torch.bool)
    return query, key, value, query_valid, memory_valid


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_inverted_attention_matches_masked_oracle_and_gradients(dtype):
    query, key, value, query_valid, memory_valid = _inputs(dtype)
    query_valid[0] = False
    memory_valid[1] = False
    query_valid[2, 3::4] = False
    memory_valid[2, 5::7] = False
    scale = query.shape[-1] ** -0.5
    expected_inputs = [x.detach().clone().requires_grad_() for x in (query, key, value)]
    actual = inverted_attention(query, key, value, query_valid, memory_valid, scale)
    reference_query, reference_key, reference_value = expected_inputs
    expected = _oracle(
        reference_query, reference_key, reference_value, query_valid, memory_valid, scale
    )
    cotangent = torch.linspace(-1, 1, actual.numel(), device="cuda").reshape(actual.shape).to(dtype)
    actual_gradients = torch.autograd.grad(actual, (query, key, value), cotangent)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.0002
    for observed, reference in zip((actual, *actual_gradients), (expected, *expected_gradients), strict=True):
        difference = (observed.float() - reference.float()).norm()
        assert difference <= tolerance * reference.float().norm().clamp_min(1e-12)
        assert torch.isfinite(observed).all()
    assert torch.count_nonzero(actual[0:2]) == 0
    assert torch.count_nonzero(actual[2, :, ~query_valid[2]]) == 0
    assert torch.count_nonzero(actual_gradients[0][2, :, ~query_valid[2]]) == 0
    for gradient in actual_gradients[1:]:
        assert torch.count_nonzero(gradient[0:2]) == 0
        assert torch.count_nonzero(gradient[2, :, ~memory_valid[2]]) == 0


def test_single_query_is_mean_pooling_not_competition_between_gqa_heads():
    query, key, value, query_valid, memory_valid = _inputs(torch.float32, queries=1)
    memory_valid[:, 7::9] = False
    result = inverted_attention(query, key, value, query_valid, memory_valid, 24**-0.5)
    expected = (value * memory_valid[:, None, :, None]).sum(2, keepdim=True)
    expected = expected / memory_valid.sum(1)[:, None, None, None]
    expected = expected.repeat_interleave(2, dim=1)
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
    q_grad, k_grad = torch.autograd.grad(result.sum(), (query, key))
    torch.testing.assert_close(q_grad, torch.zeros_like(q_grad), rtol=0, atol=2e-6)
    torch.testing.assert_close(k_grad, torch.zeros_like(k_grad), rtol=0, atol=2e-6)


def test_common_key_salience_cancels_in_query_competition():
    query, key, value, query_valid, _ = _inputs(torch.float32)
    query = query.detach().clone()
    key = key.detach().clone()
    query[..., -1] = 1.0
    key[..., -1] = 0.0
    original = inverted_attention(query, key, value, query_valid, None, 24**-0.5)
    shifted = key.clone()
    shifted[..., -1] = torch.linspace(-12, 12, key.shape[-2], device="cuda")
    changed = inverted_attention(query, shifted, value, query_valid, None, 24**-0.5)
    torch.testing.assert_close(changed, original, rtol=2e-5, atol=2e-6)


def test_outcompeted_query_retains_normalized_memory_read():
    query, key, value, query_valid, memory_valid = _inputs(torch.float32)
    query = torch.zeros_like(query)
    query[..., 0] = 100.0
    query[:, :, 0, 0] = -100.0
    key = key.detach().clone()
    key[..., 0] = torch.linspace(3, 5, key.shape[-2], device="cuda")
    observed = inverted_attention(query, key, value, query_valid, memory_valid, 24**-0.5)
    expected = _oracle(query, key, value, query_valid, memory_valid, 24**-0.5)
    torch.testing.assert_close(observed, expected, rtol=0.0002, atol=0.00002)


def test_compiled_graph_replay_respects_changed_masks_and_gradients():
    query, key, value, query_valid, memory_valid = _inputs(torch.bfloat16)

    def objective(q, k, v, q_valid, m_valid):
        result = inverted_attention(q, k, v, q_valid, m_valid, 24**-0.5)
        return result.float().square().sum()

    compiled = torch.compile(objective, mode="reduce-overhead", fullgraph=True)
    for iteration in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        query_valid[:, iteration::4] = False
        memory_valid[:, iteration::7] = False
        expected_inputs = [x.detach().clone().requires_grad_() for x in (query, key, value)]
        reference_query, reference_key, reference_value = expected_inputs
        expected = _oracle(
            reference_query, reference_key, reference_value, query_valid, memory_valid, 24**-0.5
        )
        expected_loss = expected.float().square().sum()
        expected_gradients = torch.autograd.grad(expected_loss, expected_inputs)
        actual_loss = compiled(query, key, value, query_valid, memory_valid)
        actual_gradients = torch.autograd.grad(actual_loss, (query, key, value))
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0.01, atol=0.001)
        for observed, reference in zip(actual_gradients, expected_gradients, strict=True):
            assert (observed.float() - reference.float()).norm() <= 0.025 * reference.float().norm()
