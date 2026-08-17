from __future__ import annotations

import numpy as np
import torch
from kaggle_environments import make

from kaggriculture.actions import MarketKind
from kaggriculture.constants import PRODUCTS
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.policy import (
    MarketLedger,
    _apply_ledger_order,
    _ledger_quantity_mask,
    _sample_numpy_categorical,
    act_batch,
    component_logprobs,
    component_selected_logprobs,
)


class _FixedGenerator:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, self.value)


def _pin_kind_head_to_a_quantified_buy(actor: FarmActor) -> None:
    """Make a deterministic decode choose a quantified market order.

    The production prior is conservative enough that an argmax decode picks STOP
    in every order slot, which leaves every quantity-conditioned selection empty
    and the assertions over it vacuous. Only the kind head is pinned: the
    quantity head keeps its random initialization so its log-probabilities stay
    data-dependent, which is what makes a replay comparison on that head a
    measurement instead of a comparison of two zeros.
    """
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-12.0)
        actor.market_kind.bias[MarketKind.STOP] = -6.0
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 6.0


def test_numpy_categorical_roundoff_fallback_stays_on_last_valid_action() -> None:
    logits = np.asarray([[0.0, -1.0, -2.0, 20.0, 20.0]], dtype=np.float32)
    mask = np.asarray([[True, True, True, False, False]])

    actions, logprobs, entropies = _sample_numpy_categorical(
        logits,
        mask,
        deterministic=False,
        temperature=1.0,
        generator=_FixedGenerator(np.nextafter(1.0, 0.0)),  # type: ignore[arg-type]
    )

    assert actions.tolist() == [2]
    assert np.isfinite(logprobs).all()
    assert np.isfinite(entropies).all()


def test_numpy_categorical_zero_draw_selects_first_valid_sparse_action() -> None:
    logits = np.zeros((1, 5), dtype=np.float32)
    mask = np.asarray([[False, False, True, False, True]])

    actions, _, _ = _sample_numpy_categorical(
        logits,
        mask,
        deterministic=False,
        temperature=1.0,
        generator=_FixedGenerator(0.0),  # type: ignore[arg-type]
    )

    assert actions.tolist() == [2]


def test_deterministic_policy_emits_masked_engine_actions() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 23})
    state = environment.reset(2)
    observations = [row.observation for row in state]
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)

    step = act_batch(
        actor,
        observations,
        [observations[1]["private"], observations[0]["private"]],
        deterministic=True,
    )

    assert len(step.actions) == 2
    assert step.factors.unit_active.sum(axis=1).tolist() == [1, 1]
    assert step.factors.market_kinds.shape == (2, 10)
    assert all(len(action["market"]) <= 10 for action in step.actions)
    for kinds, active in zip(step.factors.market_kinds, step.factors.market_active, strict=True):
        stopped = False
        for kind, is_active in zip(kinds, active, strict=True):
            assert bool(is_active) != stopped
            stopped |= kind == MarketKind.STOP

    next_state = environment.step(step.actions)
    assert all(row.status == "ACTIVE" for row in next_state)


def test_policy_skips_quantity_head_for_nonquantified_market_rows() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 41})
    observations = [row.observation for row in environment.reset(2)]
    actor = FarmActor(
        ModelConfig(
            cnn_width=8,
            cnn_blocks=1,
            model_dim=16,
            transformer_layers=3,
            attention_heads=2,
        )
    )
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-10)
        actor.market_kind.bias[MarketKind.STOP] = 10
        # Any accidental quantity-head evaluation would propagate NaNs into
        # the stored behavior log-probabilities.
        actor.market_quantity_context.weight.fill_(float("nan"))

    policy_step = act_batch(actor, observations, deterministic=True)

    assert not policy_step.factors.market_quantity_active.any()
    assert not policy_step.factors.market_quantities.any()
    assert not policy_step.factors.market_quantity_logprobs.any()
    assert np.isfinite(policy_step.factors.market_quantity_logprobs).all()


def test_market_ledger_matches_dynamic_engine_fill_and_round_trip() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 29})
    initial_state = environment.reset(2)
    observation = initial_state[0].observation
    initial_wheat_inventory = observation["market"]["inventory"]["WHEAT"]
    ledger = MarketLedger(
        money=observation["farms"][0]["money"],
        shed=dict(observation["private"]["shed"]),
        hires=observation["farms"][0]["hires_today"],
        extra_land=0,
        inventory={item: observation["market"]["inventory"][item] for item in PRODUCTS},
    )

    quantity_mask = _ledger_quantity_mask(observation, MarketKind.BUY_PRODUCT_WHEAT, ledger)
    assert quantity_mask[:95].all()
    assert not quantity_mask[95:].any()

    _apply_ledger_order(observation, MarketKind.BUY_PRODUCT_WHEAT, 100, ledger)
    bought_state = environment.step(
        [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_PRODUCT", "WHEAT", 100]],
            },
            {},
        ]
    )
    bought = bought_state[0].observation
    assert ledger.money == bought["farms"][0]["money"] == 5
    assert ledger.shed["WHEAT"] == bought["private"]["shed"]["WHEAT"] == 95
    # The town consumes one unit after orders resolve, so the action-local
    # ledger intentionally stops one transition before the next observation.
    assert ledger.inventory["WHEAT"] == initial_wheat_inventory - 95

    sell_ledger = MarketLedger(
        money=bought["farms"][0]["money"],
        shed=dict(bought["private"]["shed"]),
        hires=bought["farms"][0]["hires_today"],
        extra_land=0,
        inventory={item: bought["market"]["inventory"][item] for item in PRODUCTS},
    )
    _apply_ledger_order(bought, MarketKind.SELL_WHEAT, 95, sell_ledger)
    sold_state = environment.step(
        [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["SELL", "WHEAT", 95]],
            },
            {},
        ]
    )
    sold = sold_state[0].observation
    assert sell_ledger.money == sold["farms"][0]["money"]
    assert sell_ledger.shed["WHEAT"] == sold["private"]["shed"]["WHEAT"] == 0


def test_sell_quantity_mask_is_an_exact_inventory_prefix() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 31})
    observation = environment.reset(2)[0].observation
    observation["private"]["shed"]["STRAWBERRY"] = 53
    ledger = MarketLedger(
        money=observation["farms"][0]["money"],
        shed=dict(observation["private"]["shed"]),
        hires=0,
        extra_land=0,
        inventory={item: observation["market"]["inventory"][item] for item in PRODUCTS},
    )

    mask = _ledger_quantity_mask(observation, MarketKind.SELL_STRAWBERRY, ledger)

    assert mask[:53].all()
    assert not mask[53:].any()


def test_compact_cpu_quantity_logits_match_actor_exactly() -> None:
    config = ModelConfig(
        cnn_width=8,
        cnn_blocks=1,
        model_dim=16,
        transformer_layers=3,
        attention_heads=2,
        quantity_rank=5,
    )
    actor = FarmActor(config)
    context = torch.randn(3, 10, config.quantity_rank)
    kinds = torch.randint(0, 22, (3, 10))

    expected = actor.quantity_logits(context, kinds).detach().numpy()
    context_numpy = context.numpy()
    gate = actor.market_quantity_kind_gate.weight.detach().numpy()
    values = actor.market_quantity_value.weight.detach().numpy()
    bias = actor.market_quantity_bias.detach().numpy()
    features = context_numpy * (1.0 + gate[kinds.numpy()])
    actual = features @ values.T + bias[kinds.numpy()]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_component_logprobs_accepts_selected_kind_quantity_logits() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 37})
    observations = [row.observation for row in environment.reset(2)]
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    # Without this the comparison at the end of the test runs over an empty
    # selection: `market_quantity_active` was measured empty at all 24 global-RNG
    # stream positions sampled, and numpy passes an empty-vs-empty
    # `assert_allclose` silently, so the quantity head this test is named for was
    # never compared at all. Pinned, 20 components are active at every one of
    # those positions, the stored log-probabilities land near -0.49, and the
    # replay agrees to 8.9e-8..2.7e-7 -- inside the tolerance below.
    _pin_kind_head_to_a_quantified_buy(actor)
    policy_step = act_batch(actor, observations, deterministic=True)
    encoded = policy_step.encoded
    with torch.inference_mode():
        output = actor(
            torch.from_numpy(np.stack([row.board for row in encoded])).float(),
            torch.from_numpy(np.stack([row.global_features for row in encoded])).float(),
            torch.from_numpy(np.stack([row.units for row in encoded])).float(),
            torch.from_numpy(np.stack([row.unit_positions for row in encoded])).long(),
        )
        kinds = torch.from_numpy(policy_step.factors.market_kinds)
        quantity_logits = actor.quantity_logits(output.market_quantity_context, kinds)
        _, _, quantity_logprobs, *_ = component_logprobs(
            output,
            quantity_logits,
            torch.from_numpy(policy_step.factors.unit_actions),
            kinds,
            torch.from_numpy(policy_step.factors.market_quantities),
            torch.from_numpy(policy_step.factors.unit_masks),
            torch.from_numpy(policy_step.factors.market_kind_masks),
            torch.from_numpy(policy_step.factors.market_quantity_masks),
        )

    active = policy_step.factors.market_quantity_active
    # The guard is the finding, not ceremony: with no active component the
    # comparison below is empty and passes without comparing anything, which is
    # the state this test was in before the pinning above.
    assert active.any()
    np.testing.assert_allclose(
        quantity_logprobs.numpy()[active],
        policy_step.factors.market_quantity_logprobs[active],
        atol=2e-6,
    )


def test_component_selected_logprobs_matches_component_logprobs() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 41})
    observations = [row.observation for row in environment.reset(2)]
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    # Same reason as above, one head further on: with no quantified order both
    # paths returned identically zero quantity log-probabilities at all 24 stream
    # positions measured, so the "every head" claim below held as 0 == 0 for the
    # head that takes the externally supplied logits. Pinned, that head carries
    # 20 nonzero components and the two paths still agree exactly -- worst
    # deviation 0.0 across those 24 positions on all three heads, which is what
    # `rtol=0.0, atol=0.0` asks for.
    _pin_kind_head_to_a_quantified_buy(actor)
    policy_step = act_batch(actor, observations, deterministic=True)
    encoded = policy_step.encoded
    with torch.inference_mode():
        output = actor(
            torch.from_numpy(np.stack([row.board for row in encoded])).float(),
            torch.from_numpy(np.stack([row.global_features for row in encoded])).float(),
            torch.from_numpy(np.stack([row.units for row in encoded])).float(),
            torch.from_numpy(np.stack([row.unit_positions for row in encoded])).long(),
        )
        kinds = torch.from_numpy(policy_step.factors.market_kinds)
        arguments = (
            output,
            actor.quantity_logits(output.market_quantity_context, kinds),
            torch.from_numpy(policy_step.factors.unit_actions),
            kinds,
            torch.from_numpy(policy_step.factors.market_quantities),
            torch.from_numpy(policy_step.factors.unit_masks),
            torch.from_numpy(policy_step.factors.market_kind_masks),
            torch.from_numpy(policy_step.factors.market_quantity_masks),
        )
        full = component_logprobs(*arguments)
        selected = component_selected_logprobs(*arguments)

    # Nonzero on the quantity head specifically: masked-out rows are zero in
    # both paths, so a comparison over zeros alone would leave the head the
    # externally supplied logits reach unchecked.
    assert (full[2] != 0).any()
    # The entropy-free path must be the same masking/log_softmax/gather ops,
    # so it agrees bit-for-bit with the full statistics on every head.
    for lean, reference in zip(selected, full[:3], strict=True):
        torch.testing.assert_close(lean, reference, rtol=0.0, atol=0.0)
