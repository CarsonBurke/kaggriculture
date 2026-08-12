from __future__ import annotations

from kaggle_environments import make

from kaggriculture.actions import MarketKind
from kaggriculture.constants import PRODUCTS
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import (
    MarketLedger,
    _apply_ledger_order,
    _ledger_quantity_mask,
    act_batch,
)


def test_deterministic_policy_emits_masked_engine_actions() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 23})
    state = environment.reset(2)
    observations = [row.observation for row in state]
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)

    step = act_batch(
        actor,
        critic,
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
