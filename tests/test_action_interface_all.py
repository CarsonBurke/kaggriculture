"""The version-2 ALL encoding has one engine quantity and one likelihood."""

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
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredConfig


@pytest.mark.parametrize("config", [ModelConfig, EntityConfig, StructuredConfig])
def test_all_interface_is_opt_in_and_validated(config: type) -> None:
    assert config().action_interface == 1
    assert config(action_interface=2).action_interface == 2
    with pytest.raises(ValueError, match="action_interface"):
        config(action_interface=4)


def test_all_row_merges_into_legal_maximum_and_backpropagates() -> None:
    actor = FarmActor(ModelConfig(action_interface=2))
    context = torch.zeros(3, 1, actor.config.quantity_rank)
    kinds = torch.full((3, 1), MarketKind.SELL_WHEAT)
    mask = torch.zeros(3, 1, N_QUANTITIES, dtype=torch.bool)
    mask[0, :, :4] = True
    mask[1, :, :35] = True
    with torch.no_grad():
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_bias.zero_()
        actor.market_quantity_bias[MarketKind.SELL_WHEAT, N_QUANTITIES] = 2.0
    logits = actor.quantity_logits(context, kinds, mask)
    assert logits.shape == mask.shape
    assert logits[0, 0, 3].item() == pytest.approx(
        float(torch.logaddexp(torch.tensor(0.0), torch.tensor(2.0)))
    )
    assert logits[1, 0, 34].item() == pytest.approx(logits[0, 0, 3].item())
    torch.testing.assert_close(logits[0, 0, :3], torch.zeros(3))
    torch.testing.assert_close(logits[2], torch.zeros_like(logits[2]))
    logits[0, 0, 3].backward()
    assert actor.market_quantity_bias.grad[MarketKind.SELL_WHEAT, N_QUANTITIES] > 0
    with pytest.raises(ValueError, match="legality mask"):
        actor.quantity_logits(context, kinds)


def test_native_all_head_selects_maximum_and_reports_merged_logprob() -> None:
    rows, rank = 2, 2
    environment = load_native().BatchEnv(np.asarray([41], dtype=np.uint64))
    output = environment.sample_buffers()
    unit = np.full((rows, MAX_UNITS, N_UNIT_ACTIONS), -100.0, dtype=np.float32)
    unit[:, :, UnitAction.PASS] = 100.0
    kinds = np.full((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), -100.0, dtype=np.float32)
    kinds[:, :, MarketKind.STOP] = 100.0
    kinds[:, 0, MarketKind.BUY_SEED_WHEAT] = 200.0
    context = np.zeros((rows, MAX_MARKET_ORDERS, rank), dtype=np.float32)
    gate = np.zeros((1, N_MARKET_KINDS, rank), dtype=np.float32)
    values = np.zeros((1, N_QUANTITIES + 1, rank), dtype=np.float32)
    bias = np.full((1, N_MARKET_KINDS, N_QUANTITIES + 1), -10.0, dtype=np.float32)
    bias[:, :, 0] = 0.0
    bias[:, :, N_QUANTITIES] = 4.0
    environment.sample_and_step_into(
        unit,
        kinds,
        context,
        gate,
        values,
        bias,
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
    assert np.asarray(output["market_quantity_active"])[:, 0].all()
    maximum = masks.sum(-1) - 1
    np.testing.assert_array_equal(np.asarray(output["market_quantities"])[:, 0], maximum)
    for row, limit in enumerate(maximum):
        merged = np.logaddexp(-10.0, 4.0)
        logits = np.full(N_QUANTITIES, -10.0)
        logits[0] = 0.0
        logits[limit] = merged
        legal = logits[: limit + 1]
        expected = merged - np.log(np.exp(legal - merged).sum()) - merged
        assert np.asarray(output["market_quantity_logprobs"])[row, 0] == pytest.approx(
            expected, abs=2e-5
        )
