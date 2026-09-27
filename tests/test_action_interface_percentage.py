"""Exact executed-integer likelihoods for the percentage quantity head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    MarketKind,
    UnitAction,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.entity import EntityConfig
from kaggriculture.model import FarmActor, ModelConfig, percentage_quantity_logits
from kaggriculture.policy import percentage_quantity_logits_numpy
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredConfig


@pytest.mark.parametrize("config", [ModelConfig, EntityConfig, StructuredConfig])
def test_percentage_interface_is_opt_in(config: type) -> None:
    assert config().action_interface == 1
    assert config(action_interface=4).action_interface == 4


@pytest.mark.parametrize("maximum", [1, 2, 3, 16, 100])
def test_fraction_masses_and_duplicate_atoms_match_numpy(maximum: int) -> None:
    parameters = np.asarray([[0.2, -0.5, 1.3, 2.0, -0.2, 0.4, -1.5]], dtype=np.float32)
    mask = np.arange(N_QUANTITIES)[None, :] < maximum
    torch_parameters = torch.tensor(parameters, requires_grad=True)
    actual = percentage_quantity_logits(torch_parameters, torch.from_numpy(mask))
    reference = percentage_quantity_logits_numpy(parameters.copy(), mask)
    np.testing.assert_allclose(actual.detach().numpy(), reference, atol=3e-5, rtol=3e-5)
    probabilities = torch.softmax(actual.masked_fill(~torch.from_numpy(mask), -torch.inf), -1)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    if maximum == 1:
        torch.testing.assert_close(probabilities[0, 0], torch.tensor(1.0))
    loss = torch.log_softmax(actual.masked_fill(~torch.from_numpy(mask), -torch.inf), -1)[
        0, maximum - 1
    ]
    loss.backward()
    assert torch.isfinite(torch_parameters.grad).all()


@pytest.mark.parametrize("scale_raw", [-1.5, 100.0])
def test_native_percentage_sampler_reports_same_integer_logprob(scale_raw: float) -> None:
    rows, rank = 2, 2
    environment = load_native().BatchEnv(np.asarray([841], dtype=np.uint64))
    output = environment.sample_buffers()
    unit = np.full((rows, MAX_UNITS, N_UNIT_ACTIONS), -100.0, dtype=np.float32)
    unit[:, :, UnitAction.PASS] = 100.0
    kinds = np.full((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), -100.0, dtype=np.float32)
    kinds[:, :, MarketKind.STOP] = 100.0
    kinds[:, 0, MarketKind.BUY_SEED_WHEAT] = 200.0
    context = np.zeros((rows, MAX_MARKET_ORDERS, rank), dtype=np.float32)
    gate = np.zeros((1, N_MARKET_KINDS, rank), dtype=np.float32)
    values = np.zeros((1, 7, rank), dtype=np.float32)
    bias = np.zeros((1, N_MARKET_KINDS, 7), dtype=np.float32)
    bias[0, MarketKind.BUY_SEED_WHEAT] = np.asarray(
        [0.2, -0.5, 1.3, 2.0, -0.2, 0.4, scale_raw], dtype=np.float32
    )
    environment.sample_and_step_into(
        unit, kinds, context, gate, values, bias,
        np.zeros(rows, dtype=np.uint16),
        np.zeros((rows, MAX_UNITS), dtype=np.float32),
        np.zeros((rows, MAX_MARKET_ORDERS), dtype=np.float32),
        np.zeros((rows, MAX_MARKET_ORDERS), dtype=np.float32),
        np.ones(rows, dtype=np.bool_),
        np.ones(rows, dtype=np.float32),
        np.zeros(rows, dtype=np.uint8),
        output,
    )
    masks = np.asarray(output["market_quantity_masks"])[:, 0]
    quantity = np.asarray(output["market_quantities"])[:, 0].astype(np.int64)
    logits = percentage_quantity_logits_numpy(
        np.broadcast_to(bias[0, MarketKind.BUY_SEED_WHEAT], (rows, 7)).copy(), masks
    )
    logits = np.where(masks, logits, -np.inf)
    high = logits.max(axis=-1, keepdims=True)
    log_probability = logits - high - np.log(np.exp(logits - high).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(
        np.asarray(output["market_quantity_logprobs"])[:, 0],
        log_probability[np.arange(rows), quantity],
        atol=3e-5,
    )
    replay_logits = percentage_quantity_logits(
        torch.from_numpy(np.broadcast_to(bias[0, MarketKind.BUY_SEED_WHEAT], (rows, 7)).copy()),
        torch.from_numpy(masks),
    ).masked_fill(~torch.from_numpy(masks), -torch.inf)
    replay_logprob = replay_logits.log_softmax(-1).gather(
        -1, torch.from_numpy(quantity[:, None])
    )[:, 0]
    np.testing.assert_allclose(
        np.asarray(output["market_quantity_logprobs"])[:, 0],
        replay_logprob.numpy(),
        atol=3e-5,
    )


def test_actor_percentage_head_scores_exact_amounts_and_backpropagates() -> None:
    actor = FarmActor(ModelConfig(action_interface=4))
    context = torch.randn(2, 1, actor.config.quantity_rank)
    kinds = torch.full((2, 1), MarketKind.SELL_WHEAT)
    mask = torch.arange(N_QUANTITIES)[None, None, :] < torch.tensor([[[3]], [[27]]])
    logits = actor.quantity_logits(context, kinds, mask)
    assert logits.shape == mask.shape
    logits[..., 0].sum().backward()
    assert actor.market_quantity_value.weight.grad is not None
    assert torch.isfinite(actor.market_quantity_value.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_percentage_cuda_integer_logits_match_cpu() -> None:
    parameters = torch.tensor(
        [[0.2, -0.5, 1.3, 2.0, -0.2, 0.4, 100.0]] * 5,
        dtype=torch.float32,
    )
    maxima = torch.tensor([1, 2, 3, 16, 100])
    mask = torch.arange(N_QUANTITIES)[None, :] < maxima[:, None]
    expected = percentage_quantity_logits(parameters, mask)
    actual = percentage_quantity_logits(parameters.cuda(), mask.cuda()).cpu()
    torch.testing.assert_close(actual.masked_fill(~mask, 0), expected.masked_fill(~mask, 0))
    assert torch.isfinite(actual[mask]).all()
