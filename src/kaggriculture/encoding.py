"""Lossless-enough numeric observation encoding for the neural policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from kaggriculture.constants import (
    ANIMALS,
    BASE_PRICE,
    BOARD_SIZE,
    CROPS,
    DEFAULT_REWARD_GAMMA,
    EPISODE_STEPS,
    MARKET_I0,
    MAX_UNITS,
    PRICE_FLOOR,
    PRIVATE_ITEMS,
    PRODUCTS,
    SHOP_NAMES,
    STARTING_MONEY,
    TURNS_PER_DAY,
    market_price,
)

FARM_CHANNELS = 29
BOARD_CHANNELS = FARM_CHANNELS * 2
GLOBAL_FEATURES = 72
CRITIC_EXTRA_FEATURES = len(PRIVATE_ITEMS) * 2 + len(CROPS)
CRITIC_FEATURES = GLOBAL_FEATURES + CRITIC_EXTRA_FEATURES
UNIT_FEATURES = 5 + len(PRIVATE_ITEMS)

_CROP_INDEX = {crop: index for index, crop in enumerate(CROPS)}
_ANIMAL_INDEX = {animal: index for index, animal in enumerate(ANIMALS)}


@dataclass(frozen=True)
class EncodedObservation:
    board: np.ndarray
    global_features: np.ndarray
    critic_features: np.ndarray
    units: np.ndarray
    unit_positions: np.ndarray
    unit_active: np.ndarray


def _money_feature(value: Any) -> float:
    amount = float(value or 0)
    return math.copysign(math.log1p(abs(amount)) / 12.0, amount)


def _private_vector(private: dict[str, Any]) -> np.ndarray:
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    inventories = private.get("inventories") or []
    aggregate = {
        item: sum(int(inventory.get(item, 0) or 0) for inventory in inventories)
        for item in PRIVATE_ITEMS
    }
    return np.asarray(
        [*(float(shed.get(item, 0) or 0) / 100.0 for item in PRIVATE_ITEMS)]
        + [float(seeds.get(crop, 0) or 0) / 100.0 for crop in CROPS]
        + [float(aggregate[item]) / 100.0 for item in PRIVATE_ITEMS],
        dtype=np.float32,
    )


def _encode_farm(
    farm: dict[str, Any], day: int, step: int, board_size: int = BOARD_SIZE
) -> np.ndarray:
    encoded = np.zeros((FARM_CHANNELS, board_size, board_size), dtype=np.float32)
    tiles = farm.get("tiles") or []
    for y, row in enumerate(tiles[:board_size]):
        for x, tile in enumerate(row[:board_size]):
            if tile == "LOCKED":
                encoded[0, y, x] = 1.0
                continue
            encoded[28, y, x] = 1.0
            if tile is None:
                encoded[1, y, x] = 1.0
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                encoded[2, y, x] = 1.0
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop in _CROP_INDEX:
                    encoded[3 + _CROP_INDEX[crop], y, x] = 1.0
                encoded[13, y, x] = float(tile.get("yield_units", 0) or 0) / 6.0
                encoded[14, y, x] = max(
                    0.0, float(day - int(tile.get("planted_day", day) or 0)) / 30.0
                )
                encoded[15, y, x] = float(bool(tile.get("watered_today", False)))
                encoded[16, y, x] = min(1.0, float(tile.get("consecutive_unwatered", 0) or 0) / 2.0)
                encoded[17, y, x] = max(
                    0.0,
                    float(int(tile.get("fertilized_until_day", -1) or -1) - day + 1) / 3.0,
                )
            elif kind in {"COOP", "PASTURE"}:
                encoded[8 if kind == "COOP" else 9, y, x] = 1.0
                animal = tile.get("animal")
                if animal in _ANIMAL_INDEX:
                    encoded[10 + _ANIMAL_INDEX[animal], y, x] = 1.0
                    encoded[13, y, x] = float(tile.get("yield_units", 0) or 0) / 6.0
                    encoded[14, y, x] = max(
                        0.0, float(day - int(tile.get("placed_day", day) or 0)) / 30.0
                    )
                    encoded[18, y, x] = float(bool(tile.get("fed_today", False)))
                    encoded[19, y, x] = min(1.0, float(tile.get("consecutive_unfed", 0) or 0) / 2.0)
                    encoded[20, y, x] = float(bool(tile.get("cared_today", False)))
                    encoded[21, y, x] = float(bool(tile.get("fertilizer_available", False)))
                    encoded[22, y, x] = min(
                        1.0, float(tile.get("pending_care_bonus", 0) or 0) / 5.0
                    )
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                max_lifespan_step = int(tile.get("max_lifespan_step", -1) or -1)
                if max_lifespan_step >= 0:
                    encoded[25, y, x] = max(
                        0.0,
                        min(1.0, (max_lifespan_step - step) / 96.0),
                    )
                    encoded[26, y, x] = float(max_lifespan_step <= step)
                    encoded[27, y, x] = float(
                        step >= max_lifespan_step and (step - max_lifespan_step) % 2 == 0
                    )

    farmer = farm.get("farmer")
    if farmer is not None:
        x, y = map(int, farmer)
        if 0 <= x < board_size and 0 <= y < board_size:
            encoded[23, y, x] = 1.0
    for hand in farm.get("hands") or []:
        x, y = map(int, hand)
        if 0 <= x < board_size and 0 <= y < board_size:
            encoded[24, y, x] += 1.0 / MAX_UNITS
    return encoded


def encode_observation(
    observation: dict[str, Any],
    opponent_private: dict[str, Any] | None = None,
) -> EncodedObservation:
    """Encode one player's decentralized actor input and centralized critic input."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if len(farms) != 2:
        raise ValueError(f"expected exactly two farms, got {len(farms)}")
    opponent = 1 - player
    day = int(observation.get("day", 0) or 0)
    hour = int(observation.get("hour", 0) or 0)
    step = int(observation.get("step", day * TURNS_PER_DAY + hour) or 0)
    own_farm, opponent_farm = farms[player], farms[opponent]
    board = np.concatenate(
        (_encode_farm(own_farm, day, step), _encode_farm(opponent_farm, day, step)), axis=0
    )

    cycle = 2.0 * math.pi * hour / TURNS_PER_DAY
    features: list[float] = [
        day / 30.0,
        hour / float(TURNS_PER_DAY),
        step / float(EPISODE_STEPS - 1),
        (EPISODE_STEPS - 1 - step) / float(EPISODE_STEPS - 1),
        math.sin(cycle),
        math.cos(cycle),
    ]
    for farm in (own_farm, opponent_farm):
        features.extend(
            (
                _money_feature(farm.get("money", 0)),
                len(farm.get("unlocked_quadrants") or []) / 4.0,
                len(farm.get("hands") or []) / float(MAX_UNITS - 1),
                float(farm.get("hires_today", 0) or 0) / float(MAX_UNITS - 1),
            )
        )

    private = observation.get("private") or {}
    private_features = _private_vector(private)
    features.extend(private_features.tolist())
    market = observation.get("market") or {}
    inventory = market.get("inventory") or {}
    prices = market.get("prices") or {}
    features.extend((float(inventory.get(item, 10000) or 0) - 10000.0) / 500.0 for item in PRODUCTS)
    features.extend(
        float(prices.get(item, BASE_PRICE[item]) or 0) / (2.0 * BASE_PRICE[item])
        for item in PRODUCTS
    )
    shops = (observation.get("town") or {}).get("unlocked_shops") or []
    features.extend(shops.count(name) / 8.0 for name in SHOP_NAMES)
    features.extend(
        (
            sum(int(value or 0) for value in private.get("shed", {}).values()) / 100.0,
            sum(
                int(value or 0)
                for inventory in private.get("inventories", [])
                for value in inventory.values()
            )
            / 100.0,
            # Runtime overage is wrapper-dependent and always one in native
            # training. Keep this reserved slot stationary to avoid live OOD.
            1.0,
        )
    )
    global_features = np.asarray(features, dtype=np.float32)
    if global_features.shape != (GLOBAL_FEATURES,):
        raise AssertionError(f"global feature shape drifted: {global_features.shape}")

    critic_extra = (
        np.zeros(CRITIC_EXTRA_FEATURES, dtype=np.float32)
        if opponent_private is None
        else _private_vector(opponent_private)
    )
    critic_features = np.concatenate((global_features, critic_extra), axis=0)

    units = np.zeros((MAX_UNITS, UNIT_FEATURES), dtype=np.float32)
    positions = np.zeros((MAX_UNITS, 2), dtype=np.int64)
    active = np.zeros(MAX_UNITS, dtype=np.bool_)
    raw_positions = [own_farm.get("farmer"), *(own_farm.get("hands") or [])][:MAX_UNITS]
    inventories = private.get("inventories") or []
    for index, raw_position in enumerate(raw_positions):
        if raw_position is None:
            continue
        x, y = map(int, raw_position)
        positions[index] = (x, y)
        active[index] = True
        units[index, :5] = (
            1.0,
            float(index == 0),
            index / float(MAX_UNITS - 1),
            x / float(BOARD_SIZE - 1),
            y / float(BOARD_SIZE - 1),
        )
        unit_inventory = inventories[index] if index < len(inventories) else {}
        units[index, 5:] = [
            float(unit_inventory.get(item, 0) or 0) / 32.0 for item in PRIVATE_ITEMS
        ]
    return EncodedObservation(
        # Rollouts persist these features as float16. Quantizing before the
        # behavior-policy forward keeps replay likelihoods genuinely on-policy.
        board=board.astype(np.float16),
        global_features=global_features.astype(np.float16),
        critic_features=critic_features.astype(np.float16),
        units=units.astype(np.float16),
        unit_positions=positions,
        unit_active=active,
    )


def _validated_player(observation: dict[str, Any], expected_player: int) -> int:
    raw_player = observation.get("player", expected_player)
    player = expected_player if raw_player is None else int(raw_player)
    if player != expected_player:
        raise ValueError(f"expected player {expected_player} observation, got player {player}")
    farms = observation.get("farms") or []
    if len(farms) != 2:
        raise ValueError(f"expected exactly two farms, got {len(farms)}")
    return player


def _scored_money(observation: dict[str, Any], expected_player: int) -> float:
    player = _validated_player(observation, expected_player)
    farms = observation.get("farms") or []
    amount = float(farms[player].get("money", 0) or 0)
    if not math.isfinite(amount) or amount < 0.0:
        raise ValueError("farm money must be finite and non-negative")
    return amount


def liquidation_value(observation: dict[str, Any], expected_player: int) -> float:
    """Bank money plus the exact proceeds of liquidating every held product.

    Mirrors the engine's sell arithmetic unit by unit: each unit quotes at the
    current market inventory and a sale restocks the market only while the
    quote sits above the price floor.  Products in unit hands count at shed
    value, an optimistic bound: depositing needs shed room and one more turn.
    Quotes and counts are integers, so the sum is exact.
    """
    value = _scored_money(observation, expected_player)
    private = observation.get("private") or {}
    shed = private.get("shed") or {}
    inventories = private.get("inventories") or []
    market = observation.get("market") or {}
    market_inventory = market.get("inventory") or {}
    market_params = market.get("params")
    for item in PRODUCTS:
        held = int(shed.get(item, 0) or 0) + sum(
            int(inventory.get(item, 0) or 0) for inventory in inventories
        )
        inventory_level = int(market_inventory.get(item, MARKET_I0))
        for _ in range(held):
            price = market_price(item, inventory_level, market_params)
            value += float(price)
            if price > PRICE_FLOOR:
                inventory_level += 1
    return value


def _log_asset_ratio(zero: float, one: float) -> float:
    """Log-relative wealth regularized by each player's starting bank."""
    if not math.isfinite(zero) or zero < 0.0 or not math.isfinite(one) or one < 0.0:
        raise ValueError("economic values must be finite and non-negative")
    # The game-defined starting stake supplies the zero-asset prior. This keeps
    # the ratio percentage-like without the extreme log($1) slope near ruin.
    return math.log1p(zero / STARTING_MONEY) - math.log1p(one / STARTING_MONEY)


def pair_potential(observation_zero: dict[str, Any], observation_one: dict[str, Any]) -> float:
    """Log-relative actual liquid assets from player zero's perspective."""
    return _log_asset_ratio(
        liquidation_value(observation_zero, 0),
        liquidation_value(observation_one, 1),
    )


def terminal_pair_utility(
    observation_zero: dict[str, Any], observation_one: dict[str, Any]
) -> float:
    """Terminal log-relative bank utility from player zero's perspective."""
    return _log_asset_ratio(
        _scored_money(observation_zero, 0),
        _scored_money(observation_one, 1),
    )


def shaped_pair_reward(
    previous_potential: float,
    next_potential: float | None,
    *,
    terminal_utility: float | None = None,
    gamma: float = DEFAULT_REWARD_GAMMA,
) -> tuple[float, float]:
    """Discount-correct, exactly zero-sum binary32 potential shaping.

    Nonterminal rewards are ``gamma * Phi(next) - Phi(current)``. A terminal
    transition instead pays the bank utility minus ``Phi(current)`` and treats
    the terminal shaping potential as zero. Discounted returns therefore retain
    the terminal objective for any fixed horizon and matching PPO gamma.
    """
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    previous = np.float32(previous_potential)
    if not np.isfinite(previous):
        raise ValueError("previous potential must be finite")
    if terminal_utility is None:
        if next_potential is None:
            raise ValueError("nonterminal reward requires next potential")
        following = np.float32(next_potential)
        if not np.isfinite(following):
            raise ValueError("next potential must be finite")
        reward_zero = np.float32(np.float32(gamma) * following - previous)
    else:
        if next_potential is not None:
            raise ValueError("terminal reward must not receive next potential")
        utility = np.float32(terminal_utility)
        if not np.isfinite(utility):
            raise ValueError("terminal utility must be finite")
        reward_zero = np.float32(utility - previous)
    result = float(reward_zero)
    return result, -result
