from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from kaggle_environments import make

from kaggriculture.constants import MAX_UNITS, PRICE_FLOOR, market_price
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
    encode_observation,
    illiquid_value,
    liquidation_value,
    pair_potential,
    shaped_pair_reward,
    terminal_pair_potential,
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
    assert pair_potential(zero, one) == 0.0


def test_pair_potential_is_relative_farm_value_with_exact_liquid_core() -> None:
    zero, one = _observations()
    zero["farms"][0]["money"] = 9000
    one["farms"][1]["money"] = 3000

    assert pair_potential(zero, one) == 0.5
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 9000
    assert pair_potential(zero, one) == -0.5

    zero["farms"][0]["money"] = 0
    one["farms"][1]["money"] = 0
    assert pair_potential(zero, one) == 0.0

    # Held products shift the potential by their exact sale proceeds.
    zero["farms"][0]["money"] = 1000
    one["farms"][1]["money"] = 1000
    zero["private"]["shed"]["WHEAT"] = 40
    one["private"]["shed"]["MILK"] = 5
    value_zero = liquidation_value(zero, 0)
    value_one = liquidation_value(one, 1)
    assert value_zero > 1000.0
    assert value_one > 1000.0
    assert pair_potential(zero, one) == (value_zero - value_one) / (value_zero + value_one)


def test_liquidation_value_walks_the_engine_sell_curve_exactly() -> None:
    zero, _ = _observations()
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
        0.82 * 2 * 300
        + 0.85 * 4 * 50
        + 0.6 * 10
        + 0.72 * 2 * prices["WHEAT"]
        + 0.72 * 400
        + 0.72 * 3 * prices["MILK"]
        + 0.45 * (1000 + 2000)
    )
    assert abs(illiquid_value(zero, 0) - expected) < 1e-9
    # Investment moves the shaping potential instead of reading as pure loss.
    assert pair_potential(zero, one) > 0.0


def test_terminal_pair_potential_scores_bank_only() -> None:
    zero, one = _observations()
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 1000
    zero["private"]["shed"]["WHEAT"] = 100

    assert terminal_pair_potential(zero, one) == 0.5
    assert pair_potential(zero, one) > 0.5


def test_dense_bank_rewards_telescope_without_a_terminal_override() -> None:
    potentials = [0.0, 0.15, -0.2, 0.4]
    rewards = [
        shaped_pair_reward(previous, following)[0] for previous, following in pairwise(potentials)
    ]

    assert math.isclose(sum(rewards), potentials[-1] - potentials[0])
    assert shaped_pair_reward(-0.25, 0.5) == (0.75, -0.75)


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
