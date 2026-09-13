from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

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
from kaggriculture.constants import BOARD_SIZE, MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
)
from kaggriculture.model import (
    AxialRotaryEmbedding,
    BeliefOutput,
    DistributionalCritic,
    EntityTransformer,
    FarmActor,
    Linear,
    ModelConfig,
    SelfAttention,
    SpatialUNet,
    distributional_value_loss,
    hl_gauss_value_targets,
    policy_compile_options,
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
        expected = actor.forward_with_belief(board, global_features, units, positions)
        actual = actor.forward_with_belief(
            board, global_features, corrupted_units, corrupted_positions
        )

    # The belief is included: it is the heads' own input, so if garbage reached it
    # the logits could only be unchanged by luck.
    for expected_tensor, actual_tensor in zip(
        (*expected.output, expected.belief), (*actual.output, actual.belief), strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


def test_model_is_deterministic_across_train_and_eval_modes() -> None:
    torch.manual_seed(23)
    actor = FarmActor(_small_config())
    inputs = _actor_inputs(batch=1)

    actor.eval()
    with torch.inference_mode():
        eval_output = actor.forward_with_belief(*inputs)
    actor.train()
    with torch.inference_mode():
        train_output = actor.forward_with_belief(*inputs)

    for eval_tensor, train_tensor in zip(
        (*eval_output.output, eval_output.belief),
        (*train_output.output, train_output.belief),
        strict=True,
    ):
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

    rotation = rope.rotation(positions)
    rotated_query, rotated_key = AxialRotaryEmbedding.apply_rotation(query, query.clone(), rotation)

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
    rotation = AxialRotaryEmbedding(head_dim=8).rotation(positions)

    output = attention(inputs, rotation)

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


def test_separable_upsample_matches_bilinear_interpolation() -> None:
    unet = SpatialUNet(_small_config())
    low_size = (BOARD_SIZE + 1) // 2
    low_resolution = torch.randn(4, 6, low_size, low_size)

    reference = torch.nn.functional.interpolate(
        low_resolution,
        size=(BOARD_SIZE, BOARD_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    upsampled = unet.upsample_weights @ low_resolution @ unet.upsample_weights.T

    assert torch.allclose(upsampled, reference, atol=2e-6, rtol=0.0)
    # Interpolation weights form a partition of unity, so constants are exact.
    assert torch.allclose(unet.upsample_weights.sum(dim=1), torch.ones(BOARD_SIZE))
    # The buffer must stay out of checkpoints to keep the format unchanged.
    assert "upsample_weights" not in unet.state_dict()
    assert "spatial.upsample_weights" not in FarmActor(_small_config()).state_dict()


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


def test_quantity_likelihood_and_gradients_are_invariant_to_bfloat16_autocast() -> None:
    torch.manual_seed(37)
    actor = FarmActor(_small_config())
    with torch.no_grad():
        actor.market_quantity_kind_gate.weight.normal_()
        actor.market_quantity_value.weight.normal_()
        actor.market_quantity_bias.normal_()
    context = torch.randn(2, MAX_MARKET_ORDERS, actor.config.quantity_rank).bfloat16()
    kinds = torch.full((2, MAX_MARKET_ORDERS), MarketKind.BUY_SEED_WHEAT, dtype=torch.long)

    def likelihood(autocast: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        features = context.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            logits = actor.quantity_logits(features, kinds)
            logprobs = logits.log_softmax(-1)[..., 17]
        gradients = torch.autograd.grad(
            logprobs.sum(),
            (
                features,
                actor.market_quantity_kind_gate.weight,
                actor.market_quantity_value.weight,
                actor.market_quantity_bias,
            ),
        )
        return logprobs, gradients

    expected, expected_gradients = likelihood(False)
    actual, actual_gradients = likelihood(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_quantity_head_rejects_misaligned_selected_kinds() -> None:
    actor = FarmActor(_small_config())
    context = torch.zeros(2, MAX_MARKET_ORDERS, actor.config.quantity_rank)

    with pytest.raises(ValueError, match="must align"):
        actor.quantity_logits(context, torch.zeros(2, MAX_MARKET_ORDERS - 1, dtype=torch.long))


def _captured_head_inputs(actor: FarmActor) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """Record the exact tensor each policy head is applied to, as it is applied."""
    captured: dict[str, torch.Tensor] = {}

    def record(name: str) -> Any:
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            captured[name] = inputs[0]

        return hook

    handles = [
        head.register_forward_pre_hook(record(name))
        for name, head in (
            ("unit_head", actor.unit_head),
            ("market_kind", actor.market_kind),
            ("market_quantity_context", actor.market_quantity_context),
        )
    ]
    return captured, handles


def test_forward_with_belief_runs_one_trunk_pass_and_moves_no_logit() -> None:
    """The belief must ride the forward the heads already ran, not a second one.

    Bit-identity of every head is the first half: the A/B that switches the latent
    auxiliary on must not simultaneously change the policy it is measuring.

    On its own that is nearly tautological, because both paths call `_head_inputs`
    and `_policy_heads` with the same arguments and CPU eager is deterministic -- an
    implementation that simply ran the trunk twice, once for the logits and once for
    the belief, would be bit-identical here and still wrong. Counting trunk calls is
    what rejects it. The cost of that bug is a doubled trunk on the update path, and
    on CUDA under bf16 autocast the two passes are not even guaranteed to agree:
    `_sdpa_inputs` drops to bf16 and the attention backend's tiling is free to
    differ, so the supervised belief would silently drift from what the heads saw.
    """
    torch.manual_seed(11)
    actor = FarmActor(_small_config()).eval()
    inputs = _actor_inputs(batch=3)
    trunk_calls: list[int] = []
    handle = actor.transformer.register_forward_hook(
        lambda _module, _args, _output: trunk_calls.append(1)
    )
    try:
        with torch.inference_mode():
            plain = actor(*inputs)
            assert len(trunk_calls) == 1
            with_belief = actor.forward_with_belief(*inputs)
            assert len(trunk_calls) == 2
    finally:
        handle.remove()

    for field in ("unit_logits", "market_kind_logits", "market_quantity_context"):
        assert torch.equal(getattr(plain, field), getattr(with_belief.output, field))


def test_belief_is_exactly_the_tensors_the_policy_heads_were_applied_to() -> None:
    """The belief must be the heads' own input, not a parallel read of the trunk.

    The invariant pinned here is bit-equality against the tensors the three heads
    actually received, recorded by pre-hooks during the same forward. Storage
    identity is unavailable by construction -- the two halves carry different
    preprocessing, `unit_hidden` raw and `market_hidden` post-`market_norm`, so no
    single slice of the trunk output holds both and the concatenation must copy.
    Bit-equality against the recorded head inputs is the strongest remaining
    claim, and it is far stronger than gradient reachability, which any tensor
    downstream of the trunk would satisfy: it fails the moment the belief is read
    from a different token range, taken before `market_norm`, or recomputed.
    """
    torch.manual_seed(29)
    config = _small_config()
    actor = FarmActor(config).eval()
    inputs = _actor_inputs(batch=2)

    captured, handles = _captured_head_inputs(actor)
    try:
        with torch.inference_mode():
            belief = actor.forward_with_belief(*inputs).belief
    finally:
        for handle in handles:
            handle.remove()

    # Both market heads must read one and the same tensor, or "the market half of
    # the belief" would not be well defined.
    assert captured["market_kind"] is captured["market_quantity_context"]
    assert belief.shape == (2, MAX_UNITS + MAX_MARKET_ORDERS, config.model_dim)
    assert torch.equal(belief[:, :MAX_UNITS], captured["unit_head"])
    assert torch.equal(belief[:, MAX_UNITS:], captured["market_kind"])


def test_belief_carries_the_head_inputs_gradient_and_no_head_parameter() -> None:
    """The belief sits between the trunk and the heads, and the gradient proves it.

    Three claims, none of which names a parameter, so none rots on a head rename.

    The belief is a concatenation of the two head inputs, so the derivative of its
    sum with respect to each half must be exactly ones. That is what catches a
    detach on one half only, which no parameter-reachability set can see: both
    halves descend from the same trunk output, so the surviving half already
    reaches every trunk parameter and the sets stay equal.

    Set equality against the gradient of the heads' own recorded inputs then says
    the belief carries exactly their reachability -- nothing extra, and in
    particular not a different token range: the trunk's state token, the obvious
    wrong answer, never passes through `market_norm` and fails here. Strict
    containment in the logits path says the belief is upstream of the heads, so no
    head's own weights move with it.
    """
    torch.manual_seed(31)
    actor = FarmActor(_small_config())
    inputs = _actor_inputs(batch=2)

    def run() -> tuple[BeliefOutput, dict[str, torch.Tensor]]:
        captured, handles = _captured_head_inputs(actor)
        try:
            return actor.forward_with_belief(*inputs), captured
        finally:
            for handle in handles:
                handle.remove()

    def reached(loss: Callable[[BeliefOutput, dict[str, torch.Tensor]], torch.Tensor]) -> set[str]:
        actor.zero_grad(set_to_none=True)
        result, captured = run()
        loss(result, captured).backward()
        return {name for name, parameter in actor.named_parameters() if parameter.grad is not None}

    result, captured = run()
    half_grads = torch.autograd.grad(
        result.belief.sum(),
        (captured["unit_head"], captured["market_kind"]),
        allow_unused=True,
    )
    for half_grad in half_grads:
        assert half_grad is not None
        assert torch.equal(half_grad, torch.ones_like(half_grad))

    from_belief = reached(lambda result, _captured: result.belief.sum())
    from_head_inputs = reached(
        lambda _result, captured: captured["unit_head"].sum() + captured["market_kind"].sum()
    )
    from_logits = reached(
        lambda result, _captured: (
            result.output.unit_logits.sum()
            + result.output.market_kind_logits.sum()
            + result.output.market_quantity_context.sum()
        )
    )

    assert from_belief == from_head_inputs
    assert from_belief < from_logits


def test_belief_is_fp32_one_token_per_unit_and_order_slot_under_bf16_autocast() -> None:
    """The dynamics MLP holds fp32 weights, so the belief must be widened here.

    Two ways the trunk can hand back bf16 are checked, because they are not the
    same mechanism, and only the second one makes the widen load-bearing. On CPU
    `RMSNorm`'s autocast-disabled `F.rms_norm` promotes an fp32 weight rather than
    taking the fused path, so the trunk output under `torch.autocast('cpu')` is
    already fp32 and that half only pins the shape and the dtype contract while
    confirming the region was live. With bf16 parameters the trunk tensors are
    unconditionally bf16, which is the state CUDA reaches under production
    autocast, and there the widen is the only reason the belief is fp32 -- so that
    half also checks the values survive it, since a widen that quietly substituted
    a differently-shaped or zeroed tensor would satisfy dtype alone.
    """
    torch.manual_seed(37)
    config = _small_config()
    actor = FarmActor(config).eval()
    board, global_features, units, positions = _actor_inputs(batch=4)
    expected_shape = (4, MAX_UNITS + MAX_MARKET_ORDERS, config.model_dim)

    with torch.autocast("cpu", dtype=torch.bfloat16), torch.inference_mode():
        under_autocast = actor.forward_with_belief(board, global_features, units, positions)
    # Confirms the autocast region was live rather than silently ignored.
    assert under_autocast.output.unit_logits.dtype == torch.bfloat16
    assert under_autocast.belief.shape == expected_shape
    assert under_autocast.belief.dtype == torch.float32

    half = FarmActor(config).eval().to(torch.bfloat16)
    captured, handles = _captured_head_inputs(half)
    try:
        with torch.inference_mode():
            in_bf16 = half.forward_with_belief(
                board.bfloat16(), global_features.bfloat16(), units.bfloat16(), positions
            )
    finally:
        for handle in handles:
            handle.remove()

    assert captured["unit_head"].dtype == torch.bfloat16
    assert captured["market_kind"].dtype == torch.bfloat16
    assert in_bf16.belief.shape == expected_shape
    assert in_bf16.belief.dtype == torch.float32
    assert torch.equal(in_bf16.belief[:, :MAX_UNITS], captured["unit_head"].float())
    assert torch.equal(in_bf16.belief[:, MAX_UNITS:], captured["market_kind"].float())


def _trunk_case() -> tuple[EntityTransformer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    config = _small_config()
    trunk = EntityTransformer(config).eval()
    tokens = torch.randn(3, 12, config.model_dim)
    positions = torch.randint(0, BOARD_SIZE, (3, 12, 2))
    return trunk, tokens, positions


def test_unmasked_trunk_matches_an_all_true_mask() -> None:
    """`valid=None` is the claim that masking every row true changes nothing.

    The critic relies on it to skip 26 full-tensor writes, so the two spellings
    must agree exactly, not approximately.
    """
    trunk, tokens, positions = _trunk_case()
    every_row = torch.ones(*tokens.shape[:2], 1, dtype=torch.bool)

    with torch.no_grad():
        masked = trunk(tokens, positions, every_row)
        unmasked = trunk(tokens, positions, None)

    assert torch.equal(masked, unmasked)


def test_single_query_readout_matches_the_full_width_trunk() -> None:
    """A narrowed final block must reproduce the rows it still computes.

    Every token stays a key and a value; only the queries are dropped. So the
    kept prefix has to match the full-width result, which is what lets the
    critic read its value from one row without paying for a hundred others.
    """
    trunk, tokens, positions = _trunk_case()

    with torch.no_grad():
        full = trunk(tokens, positions, None)
        narrowed = trunk(tokens, positions, None, readout=2)

    assert narrowed.shape == (tokens.shape[0], 2, tokens.shape[2])
    torch.testing.assert_close(narrowed, full[:, :2], rtol=0.0, atol=1e-6)


def test_a_masked_block_refuses_to_drop_query_rows() -> None:
    """Zeroed rows are still attended to, so a mixed mask cannot skip queries."""
    trunk, tokens, positions = _trunk_case()
    mixed = torch.ones(*tokens.shape[:2], 1, dtype=torch.bool)
    mixed[:, -1] = False

    with pytest.raises(ValueError, match="cannot drop query rows"):
        trunk(tokens, positions, mixed, readout=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_linear_preserves_policy_across_batch_and_grad_modes() -> None:
    torch.manual_seed(20260912)
    module = Linear(320, 1280).cuda()
    inputs = torch.randn(320, 320, device="cuda")
    compiled = torch.compile(
        module, options=policy_compile_options("default"), fullgraph=True, dynamic=False
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            baseline = compiled(inputs).clone()
            repeated = compiled(inputs.repeat(16, 1)).clone()
        actual = compiled(inputs)
        reference = torch.nn.functional.linear(inputs, module.weight, None)
        reference = reference + module.bias.to(reference.dtype)
    torch.testing.assert_close(repeated, baseline.repeat(16, 1), rtol=0, atol=0)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (module.weight, module.bias), probe)
    expected = torch.autograd.grad(reference, (module.weight, module.bias), probe)
    for value, target in zip(gradients, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
