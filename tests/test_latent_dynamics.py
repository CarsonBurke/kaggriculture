from __future__ import annotations

import pytest
import torch

from kaggriculture.constants import BOARD_SIZE, MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.encoding import BOARD_CHANNELS, GLOBAL_FEATURES, UNIT_FEATURES
from kaggriculture.latent_dynamics import (
    DecodeContext,
    DecodeHeads,
    DecodeMasks,
    LatentDynamics,
    belief_spread,
    consecutive_rows,
    latent_decode_kl,
    latent_dynamics_halves,
    latent_dynamics_loss,
    latent_horizon_loss,
)
from kaggriculture.model import FarmActor, ModelConfig

BELIEF_TOKENS = MAX_UNITS + MAX_MARKET_ORDERS


def _config() -> ModelConfig:
    return ModelConfig(cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3)


def _actor(seed: int = 0) -> FarmActor:
    torch.manual_seed(seed)
    return FarmActor(_config())


def _state(rows: int, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "board": torch.randn(rows, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE, generator=generator),
        "global_features": torch.randn(rows, GLOBAL_FEATURES, generator=generator),
        "units": torch.randn(rows, MAX_UNITS, UNIT_FEATURES, generator=generator),
        "unit_positions": torch.zeros(rows, MAX_UNITS, 2, dtype=torch.long),
    }


def _actions(rows: int, seed: int = 0) -> dict[str, torch.Tensor]:
    from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS

    generator = torch.Generator().manual_seed(seed)
    return {
        "unit_actions": torch.randint(0, N_UNIT_ACTIONS, (rows, MAX_UNITS), generator=generator),
        "market_kinds": torch.randint(
            0, N_MARKET_KINDS, (rows, MAX_MARKET_ORDERS), generator=generator
        ),
        "market_quantities": torch.randint(
            0, N_QUANTITIES, (rows, MAX_MARKET_ORDERS), generator=generator
        ),
    }


def _masks(rows: int, actions: dict[str, torch.Tensor]) -> DecodeMasks:
    from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS

    return DecodeMasks(
        unit_masks=torch.ones(rows, MAX_UNITS, N_UNIT_ACTIONS, dtype=torch.bool),
        market_kind_masks=torch.ones(rows, MAX_MARKET_ORDERS, N_MARKET_KINDS, dtype=torch.bool),
        market_quantity_masks=torch.ones(rows, MAX_MARKET_ORDERS, N_QUANTITIES, dtype=torch.bool),
        unit_active=torch.ones(rows, MAX_UNITS, dtype=torch.bool),
        market_active=torch.ones(rows, MAX_MARKET_ORDERS, dtype=torch.bool),
        market_quantity_active=torch.ones(rows, MAX_MARKET_ORDERS, dtype=torch.bool),
        market_kinds=actions["market_kinds"],
    )


def _horizon(dynamics, belief, actions, episode, step, **rest):
    """Call the unroll with named factors: a `**dict` splat defeats type checking."""
    return latent_horizon_loss(
        dynamics,
        belief,
        actions["unit_actions"],
        actions["market_kinds"],
        actions["market_quantities"],
        episode_index=episode,
        step=step,
        **rest,
    )


def test_the_decode_reproduces_the_actors_own_heads_exactly() -> None:
    """The tightest available pin on the belief contract.

    If the belief really is the tensor the heads read, then decoding it with the
    same (detached) weights must reproduce the actor's own logits bit-for-bit.
    This fails if the belief is split at the wrong index, if the market half is
    normalized a second time -- the actor already applies `market_norm` inside
    `_head_inputs` -- or if a row-level vector is broadcast to every slot.
    """
    actor = _actor()
    state = _state(4)
    with torch.no_grad():
        belief_output = actor.forward_with_belief(**state)
        decoded = DecodeHeads.from_actor(actor).decode(belief_output.belief)
    expected = belief_output.output
    torch.testing.assert_close(decoded.unit_logits, expected.unit_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        decoded.market_kind_logits, expected.market_kind_logits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        decoded.market_quantity_context, expected.market_quantity_context, rtol=0, atol=0
    )


def test_each_slot_gets_its_own_prediction() -> None:
    """The failure this module was rewritten to remove.

    A per-row belief broadcast to every slot gives all 16 unit slots the same
    distribution, which cannot express "unit 3 harvests while unit 7 walks". Two
    slots whose beliefs differ must decode to different logits.
    """
    dynamics = LatentDynamics(16)
    actions = _actions(2)
    belief = torch.randn(2, BELIEF_TOKENS, 16)
    predicted = dynamics(belief, **actions)
    assert predicted.shape == (2, BELIEF_TOKENS, 16)
    spread = (predicted[:, 0] - predicted[:, 1]).abs().max()
    assert spread > 1e-3

    heads = DecodeHeads.from_actor(_actor())
    logits = heads.decode(predicted).unit_logits
    assert logits.shape[-2] == MAX_UNITS
    assert (logits[:, 0] - logits[:, 1]).abs().max() > 1e-4


def test_the_predictor_is_shared_across_tokens() -> None:
    """Per-token with shared weights, not a flattened per-position regression.

    The reference predicts one token at a time; a flattened variant would scale
    the parameter count with the token count. Feeding the same token twice must
    therefore give the same prediction.
    """
    dynamics = LatentDynamics(16)
    actions = _actions(1)
    token = torch.randn(1, 1, 16)
    belief = token.expand(1, BELIEF_TOKENS, 16).contiguous()
    predicted = dynamics(belief, **actions)
    for index in range(1, BELIEF_TOKENS):
        torch.testing.assert_close(predicted[:, 0], predicted[:, index], rtol=1e-6, atol=1e-6)


def test_the_target_is_a_stop_gradient_and_the_prediction_is_not() -> None:
    dynamics = LatentDynamics(16)
    actions = _actions(3)
    belief = torch.randn(3, BELIEF_TOKENS, 16, requires_grad=True)
    target = torch.randn(3, BELIEF_TOKENS, 16, requires_grad=True)
    predicted = dynamics(belief, **actions)
    loss = latent_dynamics_loss(predicted, target, torch.ones(3, dtype=torch.bool))
    loss.backward()
    assert target.grad is None or bool(target.grad.abs().max() == 0)
    assert belief.grad is not None and bool(belief.grad.abs().max() > 0)
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().max() > 0)
        for parameter in dynamics.parameters()
    )


def test_an_ineligible_rows_belief_cannot_move_the_loss_by_one_bit() -> None:
    predicted = torch.randn(4, BELIEF_TOKENS, 16)
    target = torch.randn(4, BELIEF_TOKENS, 16)
    eligible = torch.tensor([True, False, True, False])
    before = latent_dynamics_loss(predicted, target, eligible)
    predicted[1] = 1e6
    target[3] = -1e6
    after = latent_dynamics_loss(predicted, target, eligible)
    assert before.item() == after.item()


def test_the_denominator_is_the_masked_element_count() -> None:
    """Masking rows out must not shrink a per-coordinate mean."""
    predicted = torch.zeros(8, BELIEF_TOKENS, 16)
    target = torch.full((8, BELIEF_TOKENS, 16), 0.5)
    all_rows = latent_dynamics_loss(predicted, target, torch.ones(8, dtype=torch.bool))
    half = torch.zeros(8, dtype=torch.bool)
    half[:4] = True
    torch.testing.assert_close(latent_dynamics_loss(predicted, target, half), all_rows)
    none = latent_dynamics_loss(predicted, target, torch.zeros(8, dtype=torch.bool))
    assert none.item() == 0.0


def test_the_halves_are_reported_separately_and_pool_back_to_the_total() -> None:
    predicted = torch.randn(5, BELIEF_TOKENS, 16)
    target = torch.randn(5, BELIEF_TOKENS, 16)
    eligible = torch.ones(5, dtype=torch.bool)
    unit, market = latent_dynamics_halves(predicted, target, eligible, MAX_UNITS)
    total = latent_dynamics_loss(predicted, target, eligible)
    # Both halves are per-coordinate means, so the pooled mean is their
    # token-weighted average -- the identity that makes the split reportable
    # rather than a second, unrelated number.
    expected = (unit * MAX_UNITS + market * MAX_MARKET_ORDERS) / BELIEF_TOKENS
    torch.testing.assert_close(total, expected)
    with pytest.raises(ValueError, match="must split the belief"):
        latent_dynamics_halves(predicted, target, eligible, BELIEF_TOKENS)


def test_the_decode_kl_is_zero_exactly_when_the_prediction_is_exact() -> None:
    actor = _actor()
    heads = DecodeHeads.from_actor(actor)
    rows = 4
    actions = _actions(rows)
    masks = _masks(rows, actions)
    truth = torch.randn(rows, BELIEF_TOKENS, 16)
    teacher = heads.decode(truth)
    eligible = torch.ones(rows, dtype=torch.bool)
    exact = latent_decode_kl(
        truth,
        teacher.unit_logits,
        teacher.market_kind_logits,
        teacher.market_quantity_context,
        heads,
        masks,
        eligible,
    )
    assert exact.item() == pytest.approx(0.0, abs=1e-6)
    wrong = latent_decode_kl(
        truth + 1.0,
        teacher.unit_logits,
        teacher.market_kind_logits,
        teacher.market_quantity_context,
        heads,
        masks,
        eligible,
    )
    assert wrong.item() > 1e-4


def test_the_decode_kl_never_moves_a_head_weight() -> None:
    """The gradient must reach the prediction and stop at the heads.

    Otherwise the term has a degenerate solution: flatten the policy until every
    decode agrees, which lowers the KL without predicting anything.
    """
    actor = _actor()
    heads = DecodeHeads.from_actor(actor)
    rows = 3
    actions = _actions(rows)
    masks = _masks(rows, actions)
    truth = torch.randn(rows, BELIEF_TOKENS, 16)
    teacher = heads.decode(truth)
    predicted = (truth + 0.5).requires_grad_(True)
    latent_decode_kl(
        predicted,
        teacher.unit_logits,
        teacher.market_kind_logits,
        teacher.market_quantity_context,
        heads,
        masks,
        torch.ones(rows, dtype=torch.bool),
    ).backward()
    assert predicted.grad is not None and bool(predicted.grad.abs().max() > 0)
    for name, parameter in actor.named_parameters():
        assert parameter.grad is None or bool(parameter.grad.abs().max() == 0), name


def test_consecutive_rows_pairs_only_adjacent_steps_of_one_episode() -> None:
    episode = torch.tensor([0, 0, 0, 1, 1, 2])
    step = torch.tensor([5, 6, 7, 0, 1, 9])
    assert consecutive_rows(episode, step).tolist() == [True, True, False, True, False, False]
    gap = consecutive_rows(torch.tensor([0, 0]), torch.tensor([3, 5]))
    assert gap.tolist() == [False, False]
    with pytest.raises(ValueError, match="one flat entry per row"):
        consecutive_rows(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))


def test_horizon_one_equals_the_single_step_call_bit_for_bit() -> None:
    dynamics = LatentDynamics(16)
    rows = 6
    actions = _actions(rows)
    belief = torch.randn(rows, BELIEF_TOKENS, 16)
    episode = torch.zeros(rows, dtype=torch.long)
    step = torch.arange(rows)

    horizon = _horizon(dynamics, belief, actions, episode, step)
    paired = consecutive_rows(episode, step)
    keep = rows - 1
    direct = latent_dynamics_loss(
        dynamics(
            belief[:keep],
            actions["unit_actions"][:keep],
            actions["market_kinds"][:keep],
            actions["market_quantities"][:keep],
        ),
        belief[1 : keep + 1],
        paired[:keep],
    )
    assert horizon.dynamics.item() == direct.item()
    assert horizon.decode.item() == 0.0
    assert horizon.eligible.tolist() == [keep]


def test_a_run_shorter_than_the_horizon_contributes_nothing() -> None:
    dynamics = LatentDynamics(16)
    rows = 4
    actions = _actions(rows)
    belief = torch.randn(rows, BELIEF_TOKENS, 16)
    # Two episode-seats of two rows each: one pair apiece, so a two-step unroll
    # has no row that survives both steps.
    episode = torch.tensor([0, 0, 1, 1])
    step = torch.tensor([0, 1, 0, 1])
    two = _horizon(dynamics, belief, actions, episode, step, horizon=2)
    assert two.eligible.tolist() == [2, 0]
    assert torch.isfinite(two.dynamics)
    with pytest.raises(ValueError, match="at least one step"):
        _horizon(dynamics, belief, actions, episode, step, horizon=0)


def test_the_horizon_runs_the_decode_only_when_asked() -> None:
    actor = _actor()
    dynamics = LatentDynamics(16)
    rows = 5
    actions = _actions(rows)
    belief = torch.randn(rows, BELIEF_TOKENS, 16)
    episode = torch.zeros(rows, dtype=torch.long)
    step = torch.arange(rows)
    context = DecodeContext(heads=DecodeHeads.from_actor(actor), masks=_masks(rows, actions))
    with_decode = _horizon(dynamics, belief, actions, episode, step, decode=context)
    assert with_decode.decode.item() > 0.0
    without = _horizon(dynamics, belief, actions, episode, step)
    assert without.decode.item() == 0.0
    assert with_decode.dynamics.item() == without.dynamics.item()


def test_belief_spread_detects_collapse_and_survives_one_row() -> None:
    collapsed = torch.ones(8, BELIEF_TOKENS, 16)
    spread = belief_spread(collapsed)
    assert spread.cosine_similarity.item() == pytest.approx(1.0, abs=1e-5)
    assert spread.dispersion.item() == pytest.approx(0.0, abs=1e-9)
    varied = belief_spread(torch.randn(64, BELIEF_TOKENS, 16))
    assert varied.dispersion.item() > spread.dispersion.item()
    single = belief_spread(torch.randn(1, BELIEF_TOKENS, 16))
    assert single.cosine_similarity.item() == 1.0


def test_a_two_dimensional_belief_is_refused() -> None:
    """The old contract, refused loudly rather than silently broadcast."""
    dynamics = LatentDynamics(16)
    actions = _actions(2)
    with pytest.raises(ValueError, match="per belief token"):
        dynamics(torch.randn(2, 16), **actions)
    heads = DecodeHeads.from_actor(_actor())
    with pytest.raises(ValueError, match="one token per unit slot"):
        heads.decode(torch.randn(2, 16))
