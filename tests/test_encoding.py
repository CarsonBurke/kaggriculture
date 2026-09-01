from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from kaggle_environments import make

from kaggriculture.constants import (
    DEFAULT_REWARD_GAMMA,
    MAX_UNITS,
    PRICE_FLOOR,
    STARTING_MONEY,
    market_price,
)
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
    encode_observation,
    liquidation_value,
    pair_potential,
    shaped_pair_reward,
    terminal_pair_utility,
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


def test_pair_potential_is_a_zero_sum_log_liquid_asset_ratio() -> None:
    zero, one = _observations()
    zero["farms"][0]["money"] = 9000
    one["farms"][1]["money"] = 3000

    lead = math.log1p(9000 / STARTING_MONEY) - math.log1p(3000 / STARTING_MONEY)
    assert pair_potential(zero, one) == lead
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 9000
    assert pair_potential(zero, one) == -lead

    # Equal farms have zero potential at any absolute wealth, including zero.
    zero["farms"][0]["money"] = 5000
    one["farms"][1]["money"] = 5000
    assert pair_potential(zero, one) == 0.0
    zero["farms"][0]["money"] = 0
    one["farms"][1]["money"] = 0
    assert pair_potential(zero, one) == 0.0
    one["farms"][1]["money"] = 3000
    assert pair_potential(zero, one) == -math.log(2.0)
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 0
    assert pair_potential(zero, one) == math.log(2.0)

    # Held products contribute their exact liquidation proceeds.
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 3000
    zero["private"]["shed"]["WHEAT"] = 40
    one["private"]["shed"]["MILK"] = 5
    value_zero = liquidation_value(zero, 0)
    value_one = liquidation_value(one, 1)
    assert pair_potential(zero, one) == math.log1p(value_zero / STARTING_MONEY) - math.log1p(
        value_one / STARTING_MONEY
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

    # Walking that same curve keeps a market product sale exactly
    # potential-neutral: the proceeds land in the bank at precisely the quotes
    # the held units were already credited at.
    before = pair_potential(zero, one)
    zero["farms"][0]["money"] = expected
    zero["private"]["shed"]["WHEAT"] = 0
    zero["private"]["inventories"][0]["WHEAT"] = 0
    zero["market"]["inventory"]["WHEAT"] = inventory_level
    assert pair_potential(zero, one) == before


def test_pair_potential_excludes_assets_that_cannot_be_liquidated() -> None:
    zero, one = _observations()
    baseline = pair_potential(zero, one)

    zero["private"]["shed"]["GOOSE"] = 2
    zero["private"]["seeds"]["TOMATO"] = 4
    zero["farms"][0]["tiles"][0][3] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 2}
    zero["farms"][0]["tiles"][0][7] = {"animal": "COW", "yield_units": 3}
    zero["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW"]

    assert pair_potential(zero, one) == baseline


def test_terminal_pair_utility_uses_bank_only() -> None:
    zero, one = _observations()
    zero["farms"][0]["money"] = 3000
    one["farms"][1]["money"] = 1000
    zero["private"]["shed"]["WHEAT"] = 100

    terminal = terminal_pair_utility(zero, one)
    assert terminal == math.log(2.0) - math.log1p(1000 / STARTING_MONEY)
    # Unsold WHEAT contributes mid-episode but not to the terminal objective.
    assert pair_potential(zero, one) > terminal


def test_log_asset_ratio_tracks_relative_not_absolute_wealth() -> None:
    zero, one = _observations()
    zero["farms"][0]["money"] = 9000
    one["farms"][1]["money"] = 3000
    original = pair_potential(zero, one)

    # Lowering the opponent increases the learner's score through the resulting
    # relative liquid-asset advantage.
    one["farms"][1]["money"] = 0
    assert pair_potential(zero, one) > original

    # Scaling both (money + starting bank) values equally preserves the ratio.
    zero["farms"][0]["money"] = 3000
    assert math.isclose(pair_potential(zero, one), original, abs_tol=1e-15)


def test_discounted_shaped_rewards_preserve_terminal_utility_and_zero_sum() -> None:
    gamma = 0.5
    potentials = np.asarray([0.0, 0.125, -0.25], dtype=np.float32)
    terminal_utility = 0.5
    rewards = [
        shaped_pair_reward(previous, following, gamma=gamma)
        for previous, following in pairwise(potentials)
    ]
    rewards.append(
        shaped_pair_reward(potentials[-1], None, terminal_utility=terminal_utility, gamma=gamma)
    )

    returns = np.asarray(rewards, dtype=np.float64)
    discounted = (returns * np.asarray([1.0, gamma, gamma**2])[:, None]).sum(axis=0)
    np.testing.assert_allclose(
        discounted,
        [gamma**2 * terminal_utility, -(gamma**2 * terminal_utility)],
        atol=1e-15,
    )
    expected_stable = float(
        np.float32(np.float32(DEFAULT_REWARD_GAMMA) * np.float32(0.25) - np.float32(0.25))
    )
    assert shaped_pair_reward(0.25, 0.25) == (expected_stable, -expected_stable)
    assert shaped_pair_reward(-0.25, None, terminal_utility=0.5) == (0.75, -0.75)


def test_shaped_rewards_match_binary32_subtraction() -> None:
    # Both inputs narrow to the same cached potential. A binary64 subtraction
    # would invent a tiny backend-specific reward here.
    assert shaped_pair_reward(8.0, 8.0 + 1e-7, gamma=1.0) == (0.0, -0.0)


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
