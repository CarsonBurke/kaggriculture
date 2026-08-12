from __future__ import annotations

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
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
)
from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    ModelConfig,
    distributional_value_loss,
)


def test_actor_and_critic_shapes() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    batch = 3
    board = torch.zeros(batch, BOARD_CHANNELS, 10, 10)
    global_features = torch.zeros(batch, GLOBAL_FEATURES)
    critic_features = torch.zeros(batch, CRITIC_FEATURES)
    units = torch.zeros(batch, MAX_UNITS, UNIT_FEATURES)
    positions = torch.zeros(batch, MAX_UNITS, 2, dtype=torch.long)

    output = actor(board, global_features, units, positions)
    critic_logits = critic(board, critic_features)

    assert output.unit_logits.shape == (batch, MAX_UNITS, N_UNIT_ACTIONS)
    assert output.market_kind_logits.shape == (
        batch,
        MAX_MARKET_ORDERS,
        N_MARKET_KINDS,
    )
    assert output.market_quantity_context.shape == (
        batch,
        MAX_MARKET_ORDERS,
        config.quantity_rank,
    )
    selected_kinds = torch.zeros(batch, MAX_MARKET_ORDERS, dtype=torch.long)
    assert actor.quantity_logits(output.market_quantity_context, selected_kinds).shape == (
        batch,
        MAX_MARKET_ORDERS,
        N_QUANTITIES,
    )
    assert critic_logits.shape == (batch, config.value_atoms)
    assert critic.value(critic_logits).tolist() == pytest.approx([0.0] * batch, abs=1e-6)


def test_distributional_loss_is_small_for_matching_atom() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, value_atoms=5)
    critic = DistributionalCritic(config)
    logits = torch.full((1, 5), -20.0)
    logits[0, 3] = 20.0

    loss = distributional_value_loss(logits, torch.tensor([1.0]), critic.support)

    assert loss.item() < 1e-6


def test_non_multiple_of_eight_width_is_supported() -> None:
    config = ModelConfig(width=10, residual_blocks=1, hidden=24, query_features=6)

    actor = FarmActor(config)
    critic = DistributionalCritic(config)

    assert actor.spatial.output[0].num_groups == 5
    assert critic.spatial.output[0].num_groups == 5


def test_initial_policy_prior_reaches_productive_actions_without_destroying_investments() -> None:
    actor = FarmActor(ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8))

    unit_bias = actor.unit_head[-1].bias
    assert unit_bias[UnitAction.WATER] > unit_bias[UnitAction.NORTH]
    assert unit_bias[UnitAction.HARVEST] > unit_bias[UnitAction.WATER]
    assert unit_bias[UnitAction.PLACE_GOOSE] > unit_bias[UnitAction.NORTH]
    assert unit_bias[UnitAction.BUILD_COOP] < unit_bias[UnitAction.DIG]
    assert unit_bias[UnitAction.DIG] < unit_bias[UnitAction.NORTH]
    assert unit_bias[UnitAction.PICKUP_WHEAT_1] > unit_bias[UnitAction.PICKUP_WHEAT_16]
    assert unit_bias[UnitAction.PICKUP_FERTILIZER_1] > unit_bias[UnitAction.PICKUP_FERTILIZER_8]
    assert unit_bias[UnitAction.PICKUP_GOOSE_1] > unit_bias[UnitAction.PICKUP_GOOSE_4]


def test_initial_market_prior_preserves_cash_and_liquidates_products() -> None:
    actor = FarmActor(ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8))

    bias = actor.market_kind.bias
    opening_kinds = torch.tensor(
        [
            MarketKind.STOP,
            MarketKind.HIRE,
            MarketKind.BUY_LAND,
            *range(MarketKind.BUY_SEED_WHEAT, MarketKind.BUY_SEED_MELON + 1),
            *range(MarketKind.BUY_PRODUCT_WHEAT, MarketKind.BUY_PRODUCT_FERTILIZER + 1),
            *range(MarketKind.BUY_ANIMAL_GOOSE, MarketKind.BUY_ANIMAL_SHEEP + 1),
        ]
    )
    opening_probabilities = bias[opening_kinds].softmax(dim=0)

    assert opening_probabilities[0].item() > 0.93
    assert bias[MarketKind.HIRE] > bias[MarketKind.BUY_SEED_WHEAT]
    assert bias[MarketKind.BUY_SEED_WHEAT] > bias[MarketKind.BUY_ANIMAL_GOOSE]
    assert bias[MarketKind.BUY_ANIMAL_GOOSE] > bias[MarketKind.BUY_LAND]
    assert bias[MarketKind.SELL_WHEAT] > bias[MarketKind.HIRE]

    quantity_bias = actor.market_quantity_bias
    assert torch.all(
        quantity_bias[MarketKind.BUY_SEED_WHEAT, 1:] < quantity_bias[MarketKind.BUY_SEED_WHEAT, :-1]
    )
    assert torch.all(
        quantity_bias[MarketKind.BUY_ANIMAL_GOOSE, 1:]
        < quantity_bias[MarketKind.BUY_ANIMAL_GOOSE, :-1]
    )
    assert torch.all(
        quantity_bias[MarketKind.SELL_WHEAT, 1:] > quantity_bias[MarketKind.SELL_WHEAT, :-1]
    )


def test_low_rank_quantity_head_has_state_by_kind_interaction() -> None:
    config = ModelConfig(
        width=8,
        residual_blocks=1,
        hidden=8,
        query_features=4,
        quantity_rank=2,
    )
    actor = FarmActor(config)
    with torch.no_grad():
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_context.weight[0, 0] = 1.0
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_value.weight[0, 0] = 1.0
        actor.market_quantity_kind_gate.weight.zero_()
        actor.market_quantity_kind_gate.weight[MarketKind.SELL_WHEAT, 0] = 1.0
        actor.market_quantity_bias.zero_()
    market_hidden = torch.zeros(2, MAX_MARKET_ORDERS, config.hidden)
    market_hidden[0, :, 0] = 1.0
    market_hidden[1, :, 0] = 2.0

    context = actor.market_quantity_context(market_hidden)
    buy_kinds = torch.full((2, MAX_MARKET_ORDERS), MarketKind.BUY_SEED_WHEAT, dtype=torch.long)
    sell_kinds = torch.full((2, MAX_MARKET_ORDERS), MarketKind.SELL_WHEAT, dtype=torch.long)
    buy_logits = actor.quantity_logits(context, buy_kinds)
    sell_logits = actor.quantity_logits(context, sell_kinds)
    buy_slope = buy_logits[1, 0, 0] - buy_logits[0, 0, 0]
    sell_slope = sell_logits[1, 0, 0] - sell_logits[0, 0, 0]

    torch.testing.assert_close(sell_slope, 2.0 * buy_slope)


def test_quantity_head_rejects_misaligned_selected_kinds() -> None:
    actor = FarmActor(ModelConfig(width=8, residual_blocks=1, hidden=8, quantity_rank=4))
    context = torch.zeros(2, MAX_MARKET_ORDERS, 4)

    with pytest.raises(ValueError, match="must align"):
        actor.quantity_logits(context, torch.zeros(2, MAX_MARKET_ORDERS - 1, dtype=torch.long))
