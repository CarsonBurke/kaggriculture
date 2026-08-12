"""Lossless-enough numeric observation encoding for the neural policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from kaggriculture.constants import (
    ANIMAL_COST,
    ANIMALS,
    BASE_PRICE,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    LAND_PRICES,
    MAX_UNITS,
    PRIVATE_ITEMS,
    PRODUCTS,
    SEED_COST,
    SHOP_NAMES,
    TURNS_PER_DAY,
)

FARM_CHANNELS = 28
BOARD_CHANNELS = FARM_CHANNELS * 2
GLOBAL_FEATURES = 72
CRITIC_EXTRA_FEATURES = len(PRIVATE_ITEMS) * 2 + len(CROPS)
CRITIC_FEATURES = GLOBAL_FEATURES + CRITIC_EXTRA_FEATURES
UNIT_FEATURES = 5 + len(PRIVATE_ITEMS)

_CROP_INDEX = {crop: index for index, crop in enumerate(CROPS)}
_ANIMAL_INDEX = {animal: index for index, animal in enumerate(ANIMALS)}
_PRODUCT_OF_ANIMAL = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


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
            encoded[27, y, x] = 1.0
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
                        1.0, float(tile.get("pending_care_bonus", 0) or 0) / 2.0
                    )
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                max_lifespan_step = int(tile.get("max_lifespan_step", -1) or -1)
                if max_lifespan_step >= 0:
                    encoded[25, y, x] = max(
                        0.0,
                        min(1.0, (max_lifespan_step - step) / 96.0),
                    )
                    encoded[26, y, x] = float(max_lifespan_step <= step)

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
            float(observation.get("remainingOverageTime", 60) or 0) / 60.0,
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


def farm_equity(observation: dict[str, Any]) -> float:
    """Mark-to-market potential used only for objective-preserving reward shaping."""
    player = int(observation.get("player", 0) or 0)
    farm = (observation.get("farms") or [])[player]
    private = observation.get("private") or {}
    prices = (observation.get("market") or {}).get("prices") or {}
    value = float(farm.get("money", 0) or 0)

    def item_value(item: str) -> float:
        if item in BASE_PRICE:
            return 0.72 * float(prices.get(item, BASE_PRICE[item]) or 0)
        return 0.82 * ANIMAL_COST[item]

    shed = private.get("shed") or {}
    inventories = private.get("inventories") or []
    for item in PRIVATE_ITEMS:
        quantity = int(shed.get(item, 0) or 0) + sum(
            int(inventory.get(item, 0) or 0) for inventory in inventories
        )
        value += quantity * item_value(item)
    for crop in CROPS:
        value += 0.85 * int((private.get("seeds") or {}).get(crop, 0) or 0) * SEED_COST[crop]

    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal in ANIMAL_COST:
                product = _PRODUCT_OF_ANIMAL[animal]
                value += 0.72 * ANIMAL_COST[animal]
                value += (
                    0.72
                    * int(tile.get("yield_units", 0) or 0)
                    * float(prices.get(product, BASE_PRICE[product]) or 0)
                )
            elif tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop in SEED_COST:
                    value += 0.6 * SEED_COST[crop]
                    value += (
                        0.72
                        * int(tile.get("yield_units", 0) or 0)
                        * float(prices.get(crop, BASE_PRICE[crop]) or 0)
                    )
    extra_land = max(0, len(farm.get("unlocked_quadrants") or []) - 1)
    value += 0.45 * sum(LAND_PRICES[:extra_land])
    return value


def pair_potential(observation_zero: dict[str, Any], observation_one: dict[str, Any]) -> float:
    """Bounded zero-sum potential from player zero's perspective."""
    margin = farm_equity(observation_zero) - farm_equity(observation_one)
    return math.tanh(margin / 40_000.0)


def shaped_pair_reward(
    previous_potential: float,
    next_potential: float,
    terminal_money_margin: float | None = None,
) -> tuple[float, float]:
    """Potential shaping whose episode sum is exactly terminal win/loss."""
    if terminal_money_margin is None:
        reward_zero = next_potential - previous_potential
    else:
        outcome = float(terminal_money_margin > 0) - float(terminal_money_margin < 0)
        reward_zero = outcome - previous_potential
    return reward_zero, -reward_zero
