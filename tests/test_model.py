from __future__ import annotations

import math

import pytest
import torch
from torch import nn

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
    AxialRotaryEmbedding,
    DistributionalCritic,
    FarmActor,
    ModelConfig,
    SelfAttention,
    distributional_value_loss,
    hl_gauss_value_targets,
)


def _small_config(**overrides: int | float) -> ModelConfig:
    values: dict[str, int | float] = {
        "cnn_width": 8,
        "cnn_blocks": 1,
        "model_dim": 16,
        "transformer_layers": 3,
        "attention_heads": 2,
        "ffn_multiplier": 2,
        "quantity_rank": 4,
        "value_atoms": 11,
        "value_min": -2.2,
        "value_max": 2.2,
        "value_sigma_ratio": 0.75,
    }
    values.update(overrides)
    return ModelConfig(**values)


def _actor_inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    board = torch.randn(batch, BOARD_CHANNELS, 10, 10)
    global_features = torch.randn(batch, GLOBAL_FEATURES)
    units = torch.zeros(batch, MAX_UNITS, UNIT_FEATURES)
    units[:, :3] = torch.randn(batch, 3, UNIT_FEATURES)
    units[:, :3, 0] = 1.0
    positions = torch.randint(0, 10, (batch, MAX_UNITS, 2))
    return board, global_features, units, positions


def test_actor_and_critic_outputs_are_finite_contiguous_and_fixed_shape() -> None:
    config = _small_config()
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    board, global_features, units, positions = _actor_inputs(batch=3)
    critic_features = torch.randn(3, CRITIC_FEATURES)

    output = actor(board, global_features, units, positions)
    critic_logits = critic(board, critic_features)

    assert output.unit_logits.shape == (3, MAX_UNITS, N_UNIT_ACTIONS)
    assert output.market_kind_logits.shape == (3, MAX_MARKET_ORDERS, N_MARKET_KINDS)
    assert output.market_quantity_context.shape == (
        3,
        MAX_MARKET_ORDERS,
        config.quantity_rank,
    )
    selected_kinds = torch.zeros(3, MAX_MARKET_ORDERS, dtype=torch.long)
    quantity_logits = actor.quantity_logits(output.market_quantity_context, selected_kinds)
    assert quantity_logits.shape == (3, MAX_MARKET_ORDERS, N_QUANTITIES)
    assert critic_logits.shape == (3, config.value_atoms)
    for tensor in (*output, quantity_logits, critic_logits):
        assert torch.isfinite(tensor).all()
        assert tensor.is_contiguous()
    assert critic.value(critic_logits).tolist() == pytest.approx([0.0] * 3, abs=1e-6)


def test_inactive_unit_garbage_cannot_change_any_actor_output() -> None:
    torch.manual_seed(17)
    actor = FarmActor(_small_config()).eval()
    board, global_features, units, positions = _actor_inputs(batch=1)
    corrupted_units = units.clone()
    corrupted_units[:, 3:, 1:] = torch.nan
    corrupted_positions = positions.clone()
    corrupted_positions[:, 3:] = 1_000_000

    with torch.inference_mode():
        expected = actor(board, global_features, units, positions)
        actual = actor(board, global_features, corrupted_units, corrupted_positions)

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


def test_model_is_deterministic_across_train_and_eval_modes() -> None:
    torch.manual_seed(23)
    actor = FarmActor(_small_config())
    inputs = _actor_inputs(batch=1)

    actor.eval()
    with torch.inference_mode():
        eval_output = actor(*inputs)
    actor.train()
    with torch.inference_mode():
        train_output = actor(*inputs)

    for eval_tensor, train_tensor in zip(eval_output, train_output, strict=True):
        torch.testing.assert_close(train_tensor, eval_tensor, rtol=0.0, atol=0.0)
    assert not any(isinstance(module, nn.SiLU) for module in actor.modules())
    assert not any("drop" in type(module).__name__.lower() for module in actor.modules())


def test_transformer_has_mirrored_long_residual_stages() -> None:
    config = _small_config(transformer_layers=7)
    actor = FarmActor(config)

    assert len(actor.transformer.encoder) == 3
    assert len(actor.transformer.decoder) == 3
    assert actor.transformer.bottleneck is not None
    assert actor.spatial.encoder is not actor.spatial.decoder


def test_axial_rope_rotates_each_coordinate_axis_independently() -> None:
    rope = AxialRotaryEmbedding(head_dim=8)
    query = torch.ones(1, 1, 3, 8)
    positions = torch.tensor([[[0, 0], [1, 0], [0, 1]]])

    rotated_query, rotated_key = rope(query, query.clone(), positions)

    torch.testing.assert_close(rotated_query, rotated_key)
    torch.testing.assert_close(rotated_query[..., 0, :], query[..., 0, :])
    assert not torch.equal(rotated_query[..., 1, :4], query[..., 1, :4])
    torch.testing.assert_close(rotated_query[..., 1, 4:], query[..., 1, 4:])
    torch.testing.assert_close(rotated_query[..., 2, :4], query[..., 2, :4])
    assert not torch.equal(rotated_query[..., 2, 4:], query[..., 2, 4:])
    torch.testing.assert_close(rotated_query.square().sum(-1), query.square().sum(-1))


def test_attention_uses_unmasked_deterministic_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = SelfAttention(_small_config())
    captured: dict[str, object] = {}

    def fake_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object):
        captured.update(kwargs)
        captured["dtypes"] = (query.dtype, key.dtype, value.dtype)
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    inputs = torch.randn(2, 5, 16)
    positions = torch.zeros(2, 5, 2, dtype=torch.long)

    output = attention(inputs, positions)

    assert output.shape == inputs.shape
    assert captured == {
        "dropout_p": 0.0,
        "dtypes": (torch.float32, torch.float32, torch.float32),
    }


def test_hl_gauss_targets_are_normal_cdf_bin_masses() -> None:
    support = torch.linspace(-2.2, 2.2, 11)
    target = torch.tensor([0.0])
    sigma_ratio = 0.75

    actual = hl_gauss_value_targets(target, support, sigma_ratio)
    width = support[1] - support[0]
    edges = torch.cat(
        (support[:1] - width * 0.5, (support[:-1] + support[1:]) * 0.5, support[-1:] + width * 0.5)
    )
    normal = torch.distributions.Normal(target.item(), width * sigma_ratio)
    expected = normal.cdf(edges[1:]) - normal.cdf(edges[:-1])
    expected = expected / expected.sum()

    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(actual[0], actual[0].flip(0))
    assert torch.count_nonzero(actual[0]) > 2


def test_distributional_loss_prefers_the_matching_hl_gauss_distribution() -> None:
    support = torch.linspace(-2.2, 2.2, 11)
    targets = torch.tensor([0.8])
    projected = hl_gauss_value_targets(targets, support)
    matching_logits = projected.clamp_min(1e-8).log()

    matching = distributional_value_loss(matching_logits, targets, support)
    reversed_prediction = distributional_value_loss(matching_logits.flip(-1), targets, support)

    assert matching.item() < reversed_prediction.item()


def test_value_support_has_tail_room_and_rejects_invalid_targets() -> None:
    config = ModelConfig()
    critic = DistributionalCritic(config)

    assert config.value_min == pytest.approx(-2.2)
    assert config.value_max == pytest.approx(2.2)
    assert config.value_sigma_ratio == pytest.approx(0.75)
    accepted = hl_gauss_value_targets(torch.tensor([-2.0, 2.0]), critic.support)
    assert accepted.shape == (2, config.value_atoms)
    with pytest.raises(ValueError, match="outside"):
        hl_gauss_value_targets(torch.tensor([-2.21]), critic.support)
    with pytest.raises(ValueError, match="outside"):
        hl_gauss_value_targets(torch.tensor([2.21]), critic.support)
    with pytest.raises(ValueError, match="finite"):
        hl_gauss_value_targets(torch.tensor([torch.nan]), critic.support)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"cnn_width": 0}, "cnn_width"),
        ({"cnn_blocks": 0}, "cnn_blocks"),
        ({"transformer_layers": 2}, "odd and at least 3"),
        ({"transformer_layers": 4}, "odd and at least 3"),
        ({"model_dim": 18, "attention_heads": 3}, "divisible by 4"),
        ({"model_dim": 16, "attention_heads": 3}, "evenly divide"),
        ({"ffn_multiplier": 0}, "ffn_multiplier"),
        ({"quantity_rank": 0}, "quantity_rank"),
        ({"value_atoms": 1}, "value_atoms"),
        ({"value_min": 1.0, "value_max": 1.0}, "value_min"),
        ({"value_sigma_ratio": 0.0}, "value_sigma_ratio"),
    ],
)
def test_model_config_rejects_invalid_dimensions(
    override: dict[str, int | float],
    message: str,
) -> None:
    values = _small_config().to_dict()
    values.update(override)

    with pytest.raises(ValueError, match=message):
        ModelConfig(**values)


def test_non_multiple_of_eight_cnn_width_is_supported() -> None:
    config = _small_config(cnn_width=10)

    actor = FarmActor(config)
    critic = DistributionalCritic(config)

    assert actor.spatial.output[0].num_groups == 5
    assert critic.spatial.output[0].num_groups == 5


def test_initial_policy_prior_reaches_productive_actions_without_destroying_investments() -> None:
    actor = FarmActor(_small_config())

    unit_bias = actor.unit_head[-1].bias
    assert unit_bias[UnitAction.WATER] > unit_bias[UnitAction.NORTH]
    assert unit_bias[UnitAction.HARVEST] > unit_bias[UnitAction.WATER]
    assert unit_bias[UnitAction.PLACE_GOOSE] > unit_bias[UnitAction.NORTH]
    assert unit_bias[UnitAction.BUILD_COOP] < unit_bias[UnitAction.DIG]
    assert unit_bias[UnitAction.DIG] < unit_bias[UnitAction.NORTH]
    assert unit_bias[UnitAction.PICKUP_WHEAT_1] > unit_bias[UnitAction.PICKUP_WHEAT_16]
    assert unit_bias[UnitAction.PICKUP_FERTILIZER_1] > unit_bias[UnitAction.PICKUP_FERTILIZER_8]
    assert unit_bias[UnitAction.PICKUP_GOOSE_1] > unit_bias[UnitAction.PICKUP_GOOSE_4]
    wheat_biases = torch.stack(
        [unit_bias[UnitAction[f"PICKUP_WHEAT_{quantity}"]] for quantity in range(1, 17)]
    )
    fertilizer_biases = torch.stack(
        [unit_bias[UnitAction[f"PICKUP_FERTILIZER_{quantity}"]] for quantity in range(1, 9)]
    )
    assert torch.all(wheat_biases[1:] < wheat_biases[:-1])
    assert torch.all(fertilizer_biases[1:] < fertilizer_biases[:-1])
    assert wheat_biases[-1].item() == pytest.approx(1.0 - math.log(16))


def test_initial_market_prior_preserves_cash_and_liquidates_products() -> None:
    actor = FarmActor(_small_config())

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
    config = _small_config(quantity_rank=2)
    actor = FarmActor(config)
    with torch.no_grad():
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_context.weight[0, 0] = 1.0
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_value.weight[0, 0] = 1.0
        actor.market_quantity_kind_gate.weight.zero_()
        actor.market_quantity_kind_gate.weight[MarketKind.SELL_WHEAT, 0] = 1.0
        actor.market_quantity_bias.zero_()
    market_hidden = torch.zeros(2, MAX_MARKET_ORDERS, config.model_dim)
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
    actor = FarmActor(_small_config())
    context = torch.zeros(2, MAX_MARKET_ORDERS, actor.config.quantity_rank)

    with pytest.raises(ValueError, match="must align"):
        actor.quantity_logits(context, torch.zeros(2, MAX_MARKET_ORDERS - 1, dtype=torch.long))
