from __future__ import annotations

import math

import numpy as np
from kaggle_environments import make

from kaggriculture.constants import MAX_UNITS, PRICE_FLOOR, market_price
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    ECONOMIC_SCALE,
    GLOBAL_FEATURES,
    ILLIQUID_LAND_CREDIT,
    ILLIQUID_PENDING_YIELD_CREDIT,
    ILLIQUID_PLACED_ANIMAL_CREDIT,
    ILLIQUID_PLANTED_SEED_CREDIT,
    ILLIQUID_SHED_ANIMAL_CREDIT,
    ILLIQUID_SHED_SEED_CREDIT,
    STARTING_MONEY,
    UNIT_FEATURES,
    economic_pair_reward,
    encode_observation,
    illiquid_value,
    liquidation_value,
    pair_economic_scores,
)


def _observations():
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 11})
    state = environment.reset(2)
    return state[0].observation, state[1].observation


def test_encoding_shapes_and_viewpoint_symmetry() -> None:
    zero, one = _observations()

    encoded = encode_observation(zero, one["private"])

    assert encoded.board.shape == (BOARD_CHANNELS, 10, 10)
    assert encoded.global_features.shape == (GLOBAL_FEATURES,)
    assert encoded.critic_features.shape == (CRITIC_FEATURES,)
    assert encoded.units.shape == (MAX_UNITS, UNIT_FEATURES)
    assert encoded.unit_positions.shape == (MAX_UNITS, 2)
    assert encoded.unit_active.sum() == 1
    assert encoded.board.dtype == np.float16
    assert encoded.global_features.dtype == np.float16
    assert encoded.critic_features.dtype == np.float16
    assert encoded.units.dtype == np.float16
    assert pair_economic_scores(zero, one) == (0.0, 0.0)


def test_pair_economic_scores_are_absolute_and_independent() -> None:
    zero, one = _observations()
    zero["farms"][0]["money"] = 9000
    one["farms"][1]["money"] = 3000

    assert pair_economic_scores(zero, one) == (
        math.tanh(6000 / ECONOMIC_SCALE),
        0.0,
    )
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 9000
    assert pair_economic_scores(zero, one) == (
        0.0,
        math.tanh(6000 / ECONOMIC_SCALE),
    )

    # Equal rich farms are positively reinforced; equal collapse is not a
    # zero-sum tie with the same reward.
    zero["farms"][0]["money"] = 5000
    one["farms"][1]["money"] = 5000
    rich_tie = pair_economic_scores(zero, one)
    assert rich_tie[0] == rich_tie[1] > 0.0
    zero["farms"][0]["money"] = 0
    one["farms"][1]["money"] = 0
    poor_tie = pair_economic_scores(zero, one)
    assert poor_tie[0] == poor_tie[1] < 0.0

    # Held products contribute their own exact liquidation proceeds and never
    # enter the other player's score.
    zero["farms"][0]["money"] = STARTING_MONEY
    one["farms"][1]["money"] = STARTING_MONEY
    zero["private"]["shed"]["WHEAT"] = 40
    one["private"]["shed"]["MILK"] = 5
    value_zero = liquidation_value(zero, 0)
    value_one = liquidation_value(one, 1)
    assert value_zero > STARTING_MONEY
    assert value_one > STARTING_MONEY
    assert pair_economic_scores(zero, one) == (
        math.tanh((value_zero - STARTING_MONEY) / ECONOMIC_SCALE),
        math.tanh((value_one - STARTING_MONEY) / ECONOMIC_SCALE),
    )


def test_liquidation_value_walks_the_engine_sell_curve_exactly() -> None:
    zero, one = _observations()
    zero["private"]["shed"]["WHEAT"] = 3
    zero["private"]["inventories"][0]["WHEAT"] = 2

    expected = float(zero["farms"][0]["money"])
    inventory_level = int(zero["market"]["inventory"]["WHEAT"])
    for _ in range(5):
        price = market_price("WHEAT", inventory_level)
        expected += float(price)
        if price > PRICE_FLOOR:
            inventory_level += 1

    assert liquidation_value(zero, 0) == expected

    # Walking that same curve is what keeps a market product sale exactly
    # potential-neutral: the proceeds land in the bank at precisely the quotes
    # the held units were already credited at, so no amount of trading back and
    # forth manufactures shaped reward. Any monotone squash of the same value
    # difference preserves this, so the margin rescale cannot have broken it.
    before = pair_economic_scores(zero, one)[0]
    zero["farms"][0]["money"] = expected
    zero["private"]["shed"]["WHEAT"] = 0
    zero["private"]["inventories"][0]["WHEAT"] = 0
    zero["market"]["inventory"]["WHEAT"] = inventory_level
    assert pair_economic_scores(zero, one)[0] == before


def test_illiquid_value_credits_cost_basis_fractions() -> None:
    zero, one = _observations()
    assert illiquid_value(zero, 0) == 0.0
    assert illiquid_value(one, 1) == 0.0

    zero["private"]["shed"]["GOOSE"] = 2
    zero["private"]["seeds"]["TOMATO"] = 4
    zero["farms"][0]["tiles"][0][3] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 2}
    zero["farms"][0]["tiles"][0][7] = {"animal": "COW", "yield_units": 3}
    zero["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW"]
    prices = zero["market"]["prices"]

    expected = (
        ILLIQUID_SHED_ANIMAL_CREDIT * 2 * 300
        + ILLIQUID_SHED_SEED_CREDIT * 4 * 50
        + ILLIQUID_PLANTED_SEED_CREDIT * 10
        + ILLIQUID_PENDING_YIELD_CREDIT * 2 * prices["WHEAT"]
        + ILLIQUID_PLACED_ANIMAL_CREDIT * 400
        + ILLIQUID_PENDING_YIELD_CREDIT * 3 * prices["MILK"]
        + ILLIQUID_LAND_CREDIT * (1000 + 2000)
    )
    assert abs(illiquid_value(zero, 0) - expected) < 1e-9
    # Investment moves the shaping potential instead of reading as pure loss.
    assert pair_economic_scores(zero, one)[0] > 0.0


def test_terminal_economic_scores_use_bank_only() -> None:
    zero, one = _observations()
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 1000
    zero["private"]["shed"]["WHEAT"] = 100

    terminal = pair_economic_scores(zero, one, terminal=True)
    assert terminal == (
        0.0,
        math.tanh(-2000 / ECONOMIC_SCALE),
    )
    # A hundred unsold WHEAT is useful mid-episode but scores nothing at the
    # terminal bank, so the final bonus still requires liquidating production.
    assert pair_economic_scores(zero, one)[0] > terminal[0]


def test_economic_score_cannot_pay_for_shrinking_an_opponent() -> None:
    """Each learner's reward is monotone in its own economy only."""
    zero, one = _observations()
    zero["farms"][0]["money"] = 9000
    one["farms"][1]["money"] = 3000
    original = pair_economic_scores(zero, one)

    one["farms"][1]["money"] = 0
    opponent_destroyed = pair_economic_scores(zero, one)
    assert opponent_destroyed[0] == original[0]
    assert opponent_destroyed[1] < original[1]

    zero["farms"][0]["money"] = 7000
    learner_destroyed = pair_economic_scores(zero, one)
    assert learner_destroyed[0] < opponent_destroyed[0]


def test_economic_rewards_do_not_telescope() -> None:
    terminal_scores = (0.4, 0.2)
    late_growth = [
        economic_pair_reward((0.0, 0.0), terminal=False, transitions=2),
        economic_pair_reward(terminal_scores, terminal=True, transitions=2),
    ]
    sustained_growth = [
        economic_pair_reward(terminal_scores, terminal=False, transitions=2),
        economic_pair_reward(terminal_scores, terminal=True, transitions=2),
    ]

    # Same endpoint, different path: sustained capital earns more. A potential
    # difference would telescope both paths to the same terminal score.
    assert tuple(map(sum, zip(*sustained_growth, strict=True))) > tuple(
        map(sum, zip(*late_growth, strict=True))
    )
    assert economic_pair_reward((0.5, 0.25), terminal=False, transitions=4) == (
        0.125,
        0.0625,
    )
    assert economic_pair_reward((0.5, 0.25), terminal=True, transitions=4) == (
        0.625,
        0.3125,
    )


def test_encoding_exposes_shed_pressure_and_exact_crop_decay_phase() -> None:
    zero, one = _observations()
    zero["step"] = 70
    zero["day"] = 2
    zero["hour"] = 22
    zero["private"]["shed"]["WHEAT"] = 60
    zero["private"]["inventories"][0]["MILK"] = 5
    zero["remainingOverageTime"] = 7
    zero["farms"][0]["tiles"][4][4] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 4,
        "max_lifespan_step": 72,
        "fertilized_until_day": -1,
    }

    before_decay = encode_observation(zero, one["private"])
    zero["step"] = 72
    zero["hour"] = 0
    at_decay = encode_observation(zero, one["private"])

    assert math.isclose(float(before_decay.global_features[-3]), 0.6, abs_tol=1e-3)
    assert math.isclose(float(before_decay.global_features[-2]), 0.05, abs_tol=1e-3)
    assert before_decay.global_features[-1] == 1
    assert float(before_decay.board[25, 4, 4]) > 0
    assert before_decay.board[26, 4, 4] == 0
    assert before_decay.board[27, 4, 4] == 0
    assert at_decay.board[25, 4, 4] == 0
    assert at_decay.board[26, 4, 4] == 1
    assert at_decay.board[27, 4, 4] == 1
    zero["step"] = 73
    after_decay_tick = encode_observation(zero, one["private"])
    assert after_decay_tick.board[27, 4, 4] == 0


def test_encoding_preserves_strategically_distinct_pending_care_bonuses() -> None:
    zero, one = _observations()
    animal = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 0,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": True,
        "cared_today": True,
        "fertilizer_available": False,
        "pending_care_bonus": 2,
    }
    zero["farms"][0]["tiles"][4][4] = animal
    care_two = encode_observation(zero, one["private"])
    animal["pending_care_bonus"] = 5
    care_five = encode_observation(zero, one["private"])
    animal["pending_care_bonus"] = 8
    care_above_cap = encode_observation(zero, one["private"])

    assert math.isclose(float(care_two.board[22, 4, 4]), 0.4, abs_tol=1e-3)
    assert care_five.board[22, 4, 4] == 1
    # Base production plus five banked units already fills the six-unit cap,
    # so larger banks are behaviorally equivalent at the next production.
    assert care_above_cap.board[22, 4, 4] == 1
