"""Semantic tokenization for the structured farm transformer (VIT_PLAN).

Turns a raw observation into per-entity token arrays: categorical index
columns (for learned embeddings) plus continuous columns normalized by known
mechanic bounds. No convolutional mixing, no undifferentiated scalar packing —
each field keeps its identity so attention can relate entities semantically.

Families: tile tokens (own and opponent farms share one tokenizer; a
farm-identity categorical distinguishes them), execution-ordered unit tokens
with local tile-gather indices, economic tokens (products, crops, farm
summaries, town/clock), and critic-only opponent-private columns.
``encode_structured_observation`` assembles them all in staging dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kaggriculture.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_MAX_HELD,
    ANIMALS,
    BASE_PRICE,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    CROPS,
    EPISODE_STEPS,
    MARKET_I0,
    MAX_UNITS,
    ONGOING_CROPS,
    PRIVATE_ITEMS,
    PRODUCTS,
    SEED_COST,
    SHED_CAPACITY,
    SHOP_NAMES,
    TURNS_PER_DAY,
    shed_access_tiles,
)

TILE_COUNT = BOARD_SIZE * BOARD_SIZE
OBSERVATION_SCHEMA_VERSION = 3

# Categorical vocabularies. Index 0 of the occupant vocabulary is the "no
# occupant" value so embeddings for absent fields are learned, not
# zero-imputed. Crops and animals never co-occupy a tile, so they share one
# occupant vocabulary.
TILE_KINDS = ("LOCKED", "EMPTY", "WEED", "PLANT", "COOP", "PASTURE")
TILE_KIND_INDEX = {name: index for index, name in enumerate(TILE_KINDS)}
TILE_OCCUPANTS = ("NONE", *CROPS, *ANIMALS)
TILE_OCCUPANT_INDEX = {name: index for index, name in enumerate(TILE_OCCUPANTS)}
FARM_IDENTITIES = ("OWN", "OPPONENT")
QUADRANT_COUNT = 4

# Categorical columns per tile token, in order.
TILE_CATEGORICAL_FIELDS = (
    "kind",  # TILE_KINDS
    "occupant",  # TILE_OCCUPANTS
    "farm",  # FARM_IDENTITIES
    "row",  # 0..BOARD_SIZE-1
    "column",  # 0..BOARD_SIZE-1
    "quadrant",  # NW, NE, SW, SE
)
# Continuous columns per tile token, in order. Every value is bounded to
# [0, 1] by a cited mechanic; booleans are exact {0, 1}.
TILE_CONTINUOUS_FIELDS = (
    "yield_fraction",  # yield_units / CROP_MAX_YIELD[crop] or ANIMAL_MAX_HELD
    "age_fraction",  # occupant age in days / 30-day episode horizon
    "maturity_fraction",  # age / occupant first_yield_day, clipped to 1
    "watered_today",
    "fed_today",
    "cared_today",
    "fertilizer_remaining",  # (fertilized_until_day - day + 1) / 3 (plants)
    "fertilizer_available",  # collectible fertilizer waiting (animals)
    "pending_care_bonus",  # banked care bonus units / 5 (animals)
    "decay_pressure",  # consecutive unwatered/unfed days / lethal threshold 2
    "lifespan_remaining",  # (max_lifespan_step - step) / 96, finite crops
    "lifespan_expired",  # past max_lifespan_step: losing 1 yield every 2 steps
    "lifespan_decay_tick",  # an expired plant loses a unit at this exact step
    "harvest_ready",  # mature crop with stock, or animal with stock
    "edge",
    "corner",
    "shed_distance",  # Manhattan distance to nearest shed-access tile / max
    "shed_access",  # exactly on a shed-access tile
    "farmer_present",  # public farmer position, for both farms
    "hand_count",  # public hands on this tile / (MAX_UNITS - 1)
)
N_TILE_CATEGORICAL = len(TILE_CATEGORICAL_FIELDS)
N_TILE_CONTINUOUS = len(TILE_CONTINUOUS_FIELDS)

_FIELD = {name: index for index, name in enumerate(TILE_CONTINUOUS_FIELDS)}
# Engine: a plant or animal dies when its consecutive unwatered/unfed counter
# reaches 2 at the daily refresh.
_DECAY_LETHAL = 2.0
# Remaining lifespan saturates at a 4-day horizon (the existing encoder's
# scale): a freshly planted finite crop may carry a far larger lifespan, but
# only the approach to expiry is decision-relevant.
_LIFESPAN_HORIZON_STEPS = 96.0
_CARE_BONUS_HORIZON = 5.0
_EPISODE_DAYS = EPISODE_STEPS / TURNS_PER_DAY


def _quadrant(x: int, y: int, board_size: int) -> int:
    half = board_size // 2
    return (0 if y < half else 2) + (0 if x < half else 1)


def _shed_distance_map(board_size: int) -> np.ndarray:
    access = shed_access_tiles(board_size)
    grid = np.empty((board_size, board_size), dtype=np.float32)
    for y in range(board_size):
        for x in range(board_size):
            grid[y, x] = min(abs(x - ax) + abs(y - ay) for ax, ay in access)
    return grid / grid.max()


_SHED_DISTANCE = _shed_distance_map(BOARD_SIZE)
_SHED_ACCESS = frozenset(shed_access_tiles(BOARD_SIZE))


@dataclass(frozen=True)
class TileTokens:
    """One farm's tile tokens: embedding indices plus bounded features."""

    categorical: np.ndarray  # [TILE_COUNT, N_TILE_CATEGORICAL] int64
    continuous: np.ndarray  # [TILE_COUNT, N_TILE_CONTINUOUS] float32


def tokenize_farm_tiles(farm: dict, day: int, step: int, *, opponent: bool) -> TileTokens:
    """Tokenize one farm's tile grid with the shared tile schema.

    Both farms pass through this exact function; ``opponent`` only sets the
    farm-identity categorical, which is how the model tells them apart.
    """
    tiles = farm.get("tiles") or []
    categorical = np.zeros((TILE_COUNT, N_TILE_CATEGORICAL), dtype=np.int64)
    continuous = np.zeros((TILE_COUNT, N_TILE_CONTINUOUS), dtype=np.float32)
    farm_identity = int(opponent)
    farmer = farm.get("farmer")
    if farmer is not None:
        x, y = map(int, farmer)
        continuous[y * BOARD_SIZE + x, _FIELD["farmer_present"]] = 1.0
    hand_counts = np.zeros(TILE_COUNT, dtype=np.int32)
    for x, y in farm.get("hands") or []:
        hand_counts[int(y) * BOARD_SIZE + int(x)] += 1
    continuous[:, _FIELD["hand_count"]] = hand_counts / float(MAX_UNITS - 1)

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            token = y * BOARD_SIZE + x
            tile = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else "LOCKED"
            row = categorical[token]
            row[2] = farm_identity
            row[3] = y
            row[4] = x
            row[5] = _quadrant(x, y, BOARD_SIZE)
            features = continuous[token]
            features[_FIELD["edge"]] = float(x in (0, BOARD_SIZE - 1) or y in (0, BOARD_SIZE - 1))
            features[_FIELD["corner"]] = float(
                x in (0, BOARD_SIZE - 1) and y in (0, BOARD_SIZE - 1)
            )
            features[_FIELD["shed_distance"]] = float(_SHED_DISTANCE[y, x])
            features[_FIELD["shed_access"]] = float((x, y) in _SHED_ACCESS)

            if tile == "LOCKED":
                row[0] = TILE_KIND_INDEX["LOCKED"]
                continue
            if tile is None:
                row[0] = TILE_KIND_INDEX["EMPTY"]
                continue
            if not isinstance(tile, dict):
                raise ValueError(f"unrecognized tile value at ({x}, {y}): {tile!r}")
            kind = tile.get("kind")
            if kind == "WEED":
                row[0] = TILE_KIND_INDEX["WEED"]
                continue
            if kind == "PLANT":
                row[0] = TILE_KIND_INDEX["PLANT"]
                crop = tile.get("crop")
                if crop not in TILE_OCCUPANT_INDEX:
                    raise ValueError(f"unrecognized crop at ({x}, {y}): {crop!r}")
                row[1] = TILE_OCCUPANT_INDEX[crop]
                age = max(0, day - int(tile.get("planted_day", day) or 0))
                stock = int(tile.get("yield_units", 0) or 0)
                features[_FIELD["yield_fraction"]] = min(1.0, stock / float(CROP_MAX_YIELD[crop]))
                features[_FIELD["age_fraction"]] = min(1.0, age / _EPISODE_DAYS)
                features[_FIELD["maturity_fraction"]] = min(
                    1.0, age / float(CROP_FIRST_YIELD_DAY[crop])
                )
                features[_FIELD["watered_today"]] = float(bool(tile.get("watered_today", False)))
                features[_FIELD["fertilizer_remaining"]] = min(
                    1.0,
                    max(0, int(tile.get("fertilized_until_day", -1) or -1) - day + 1) / 3.0,
                )
                features[_FIELD["decay_pressure"]] = min(
                    1.0,
                    max(0, int(tile.get("consecutive_unwatered", 0) or 0)) / _DECAY_LETHAL,
                )
                lifespan_step = int(tile.get("max_lifespan_step", -1) or -1)
                if lifespan_step >= 0:
                    features[_FIELD["lifespan_remaining"]] = min(
                        1.0, max(0, lifespan_step - step) / _LIFESPAN_HORIZON_STEPS
                    )
                    expired = step >= lifespan_step
                    features[_FIELD["lifespan_expired"]] = float(expired)
                    features[_FIELD["lifespan_decay_tick"]] = float(
                        expired and (step - lifespan_step) % 2 == 0
                    )
                features[_FIELD["harvest_ready"]] = float(
                    age >= CROP_FIRST_YIELD_DAY[crop] and stock > 0
                )
                continue
            if kind in ("COOP", "PASTURE"):
                row[0] = TILE_KIND_INDEX[kind]
                animal = tile.get("animal")
                if animal is None:
                    continue
                if animal not in TILE_OCCUPANT_INDEX:
                    raise ValueError(f"unrecognized animal at ({x}, {y}): {animal!r}")
                row[1] = TILE_OCCUPANT_INDEX[animal]
                age = max(0, day - int(tile.get("placed_day", day) or 0))
                stock = int(tile.get("yield_units", 0) or 0)
                features[_FIELD["yield_fraction"]] = min(
                    1.0, stock / float(ANIMAL_MAX_HELD[animal])
                )
                features[_FIELD["age_fraction"]] = min(1.0, age / _EPISODE_DAYS)
                features[_FIELD["maturity_fraction"]] = min(
                    1.0, age / float(ANIMAL_FIRST_YIELD_DAY[animal])
                )
                features[_FIELD["fed_today"]] = float(bool(tile.get("fed_today", False)))
                features[_FIELD["cared_today"]] = float(bool(tile.get("cared_today", False)))
                features[_FIELD["fertilizer_available"]] = float(
                    bool(tile.get("fertilizer_available", False))
                )
                features[_FIELD["pending_care_bonus"]] = min(
                    1.0,
                    max(0, int(tile.get("pending_care_bonus", 0) or 0)) / _CARE_BONUS_HORIZON,
                )
                features[_FIELD["decay_pressure"]] = min(
                    1.0,
                    max(0, int(tile.get("consecutive_unfed", 0) or 0)) / _DECAY_LETHAL,
                )
                features[_FIELD["harvest_ready"]] = float(stock > 0)
                continue
            raise ValueError(f"unrecognized tile kind at ({x}, {y}): {kind!r}")
    return TileTokens(categorical=categorical, continuous=continuous)


# Unit tokens: the farmer plus up to 15 hands, in engine execution order.
UNIT_ROLES = ("FARMER", "HAND")

# Categorical columns per unit token, in order.
UNIT_CATEGORICAL_FIELDS = (
    "role",  # UNIT_ROLES
    "slot",  # execution-order slot 0..MAX_UNITS-1
    "row",  # 0..BOARD_SIZE-1
    "column",  # 0..BOARD_SIZE-1
)
# Continuous columns per unit token: exact held counts, situational flags,
# then insertion ranks. DROP fills the shed in insertion order and discards
# overflow, so counts alone do not determine the next economic state.
UNIT_CONTINUOUS_FIELDS = (
    *(f"holds_{item}" for item in PRIVATE_ITEMS),
    "holds_total",
    "shed_access",  # standing on a shed-access tile
    *(f"inventory_rank_{item}" for item in PRIVATE_ITEMS),
)
N_UNIT_CATEGORICAL = len(UNIT_CATEGORICAL_FIELDS)
N_UNIT_CONTINUOUS = len(UNIT_CONTINUOUS_FIELDS)
_UNIT_INVENTORY_SCALE = 32.0

# Gather order for a unit's local tile context: its own tile, then NSEW.
UNIT_TILE_GATHERS = ("HERE", "NORTH", "SOUTH", "EAST", "WEST")
_GATHER_DELTA = ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0))


@dataclass(frozen=True)
class UnitTokens:
    """Execution-ordered unit tokens with local tile-gather indices."""

    categorical: np.ndarray  # [MAX_UNITS, N_UNIT_CATEGORICAL] int64
    continuous: np.ndarray  # [MAX_UNITS, N_UNIT_CONTINUOUS] float32
    positions: np.ndarray  # [MAX_UNITS, 2] int64 (x, y)
    active: np.ndarray  # [MAX_UNITS] bool
    tile_gather: np.ndarray  # [MAX_UNITS, 5] int64 own-farm tile token index
    tile_gather_valid: np.ndarray  # [MAX_UNITS, 5] bool (off-board neighbors)


def tokenize_units(farm: dict, private: dict) -> UnitTokens:
    """Tokenize the acting player's farmer and hands in execution order.

    Inactive slots stay zeroed and are meant to be excluded from attention via
    ``active`` rather than consumed as null tokens.
    """
    categorical = np.zeros((MAX_UNITS, N_UNIT_CATEGORICAL), dtype=np.int64)
    continuous = np.zeros((MAX_UNITS, N_UNIT_CONTINUOUS), dtype=np.float32)
    positions = np.zeros((MAX_UNITS, 2), dtype=np.int64)
    active = np.zeros(MAX_UNITS, dtype=np.bool_)
    tile_gather = np.zeros((MAX_UNITS, len(UNIT_TILE_GATHERS)), dtype=np.int64)
    tile_gather_valid = np.zeros((MAX_UNITS, len(UNIT_TILE_GATHERS)), dtype=np.bool_)

    raw_positions = [farm.get("farmer"), *(farm.get("hands") or [])][:MAX_UNITS]
    inventories = private.get("inventories") or []
    for slot, raw_position in enumerate(raw_positions):
        if raw_position is None:
            continue
        x, y = map(int, raw_position)
        active[slot] = True
        positions[slot] = (x, y)
        categorical[slot] = (
            UNIT_ROLES.index("FARMER") if slot == 0 else UNIT_ROLES.index("HAND"),
            slot,
            y,
            x,
        )
        inventory = inventories[slot] if slot < len(inventories) else {}
        held = [float(inventory.get(item, 0) or 0) for item in PRIVATE_ITEMS]
        continuous[slot, : len(PRIVATE_ITEMS)] = np.asarray(held) / _UNIT_INVENTORY_SCALE
        continuous[slot, len(PRIVATE_ITEMS)] = sum(held) / _UNIT_INVENTORY_SCALE
        continuous[slot, len(PRIVATE_ITEMS) + 1] = float((x, y) in _SHED_ACCESS)
        rank = 0
        for item, count in inventory.items():
            if item in PRIVATE_ITEMS and count:
                rank += 1
                continuous[slot, len(PRIVATE_ITEMS) + 2 + PRIVATE_ITEMS.index(item)] = (
                    rank / _UNIT_INVENTORY_SCALE
                )
        for gather, (dx, dy) in enumerate(_GATHER_DELTA):
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                tile_gather[slot, gather] = ny * BOARD_SIZE + nx
                tile_gather_valid[slot, gather] = True
    return UnitTokens(
        categorical=categorical,
        continuous=continuous,
        positions=positions,
        active=active,
        tile_gather=tile_gather,
        tile_gather_valid=tile_gather_valid,
    )


# Economic tokens: one per tradable product, one per plantable crop, one
# summary per farm, one shared town/clock token. Widths differ per family, so
# each family gets its own array and input projection in the model.
# Market columns reuse the proven flat encoder's scales; unlike tile tokens,
# inventory deviation and price are signed/unbounded by design.
PRODUCT_TOKEN_FIELDS = (
    "market_inventory",  # (inventory - MARKET_I0) / 500, matching the encoder
    "price",  # current price / (2 * BASE_PRICE)
    "base_price",  # BASE_PRICE / max BASE_PRICE
    "shed_stock",  # own shed count / SHED_CAPACITY
    "carried_stock",  # summed across own units / SHED_CAPACITY
)
ANIMAL_TOKEN_FIELDS = (
    "purchase_price",  # ANIMAL_COST / max ANIMAL_COST; animals have no market quote
    "shed_stock",  # own shed count / SHED_CAPACITY
    "carried_stock",  # summed across own units / SHED_CAPACITY
)
CROP_TOKEN_FIELDS = (
    "seed_cost",  # SEED_COST / max SEED_COST
    "seeds_held",  # own private seed count / SHED_CAPACITY
    "first_yield_day",  # CROP_FIRST_YIELD_DAY / max CROP_MAX_YIELD_DAY
    "max_yield_day",  # CROP_MAX_YIELD_DAY / max CROP_MAX_YIELD_DAY
    "max_yield",  # CROP_MAX_YIELD / max CROP_MAX_YIELD
    "ongoing",  # keeps producing after first yield
)
FARM_TOKEN_FIELDS = (
    "money",  # signed log1p scale shared with the flat encoder
    "unlocked_quadrants",  # / 4
    "hands",  # hired hands / (MAX_UNITS - 1)
    "hires_today",  # / (MAX_UNITS - 1)
)
TOWN_TOKEN_FIELDS = (
    "day",
    "hour",
    "progress",
    "remaining",
    "hour_sin",
    "hour_cos",
    *(f"shop_{name}" for name in SHOP_NAMES),
)


# Extra economy columns the centralized critic appends from the opponent's
# private state; the actor never sees them.
PRODUCT_PRIVATE_FIELDS = (
    "opponent_shed_stock",  # opponent shed count / SHED_CAPACITY
    "opponent_carried_stock",  # summed across opponent units / SHED_CAPACITY
)
ANIMAL_PRIVATE_FIELDS = (
    "opponent_shed_stock",
    "opponent_carried_stock",
)
CROP_PRIVATE_FIELDS = ("opponent_seeds_held",)  # opponent seeds / SHED_CAPACITY


@dataclass(frozen=True)
class EconomyTokens:
    """Market, crop, farm-summary, and town/clock tokens for one viewpoint."""

    products: np.ndarray  # [len(PRODUCTS), len(PRODUCT_TOKEN_FIELDS)] float32
    animals: np.ndarray  # [len(ANIMALS), len(ANIMAL_TOKEN_FIELDS)] float32
    crops: np.ndarray  # [len(CROPS), len(CROP_TOKEN_FIELDS)] float32
    farms: np.ndarray  # [2, len(FARM_TOKEN_FIELDS)] float32, own farm first
    town: np.ndarray  # [len(TOWN_TOKEN_FIELDS)] float32


def _money_feature(value: float) -> float:
    amount = float(value or 0)
    return float(np.copysign(np.log1p(abs(amount)) / 12.0, amount))


def tokenize_economy(observation: dict) -> EconomyTokens:
    """Tokenize market, crop, farm-summary, and town state for the actor."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if len(farms) != 2:
        raise ValueError(f"expected exactly two farms, got {len(farms)}")
    private = observation.get("private") or {}
    market = observation.get("market") or {}
    inventory = market.get("inventory") or {}
    prices = market.get("prices") or {}
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    carried = {
        item: sum(int(unit.get(item, 0) or 0) for unit in private.get("inventories") or [])
        for item in (*PRODUCTS, *ANIMALS)
    }
    max_base_price = float(max(BASE_PRICE.values()))

    products = np.asarray(
        [
            (
                (float(inventory.get(item, MARKET_I0) or 0) - MARKET_I0) / 500.0,
                float(prices.get(item, BASE_PRICE[item]) or 0) / (2.0 * BASE_PRICE[item]),
                BASE_PRICE[item] / max_base_price,
                float(shed.get(item, 0) or 0) / SHED_CAPACITY,
                carried[item] / SHED_CAPACITY,
            )
            for item in PRODUCTS
        ],
        dtype=np.float32,
    )
    max_animal_cost = float(max(ANIMAL_COST.values()))
    animals = np.asarray(
        [
            (
                ANIMAL_COST[animal] / max_animal_cost,
                float(shed.get(animal, 0) or 0) / SHED_CAPACITY,
                carried[animal] / SHED_CAPACITY,
            )
            for animal in ANIMALS
        ],
        dtype=np.float32,
    )
    max_seed_cost = float(max(SEED_COST.values()))
    max_yield_day = float(max(CROP_MAX_YIELD_DAY.values()))
    max_yield = float(max(CROP_MAX_YIELD.values()))
    crops = np.asarray(
        [
            (
                SEED_COST[crop] / max_seed_cost,
                float(seeds.get(crop, 0) or 0) / SHED_CAPACITY,
                CROP_FIRST_YIELD_DAY[crop] / max_yield_day,
                CROP_MAX_YIELD_DAY[crop] / max_yield_day,
                CROP_MAX_YIELD[crop] / max_yield,
                float(crop in ONGOING_CROPS),
            )
            for crop in CROPS
        ],
        dtype=np.float32,
    )
    farm_rows = []
    for farm_index in (player, 1 - player):
        farm = farms[farm_index]
        farm_rows.append(
            (
                _money_feature(farm.get("money", 0)),
                len(farm.get("unlocked_quadrants") or []) / 4.0,
                len(farm.get("hands") or []) / float(MAX_UNITS - 1),
                float(farm.get("hires_today", 0) or 0) / float(MAX_UNITS - 1),
            )
        )
    shops = (observation.get("town") or {}).get("unlocked_shops") or []
    town = np.concatenate(
        (
            clock_features(observation),
            np.asarray([shops.count(name) / 8.0 for name in SHOP_NAMES], dtype=np.float32),
        )
    )
    return EconomyTokens(
        products=products,
        animals=animals,
        crops=crops,
        farms=np.asarray(farm_rows, dtype=np.float32),
        town=town,
    )


def opponent_economy_columns(
    opponent_private: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Critic-only columns from the opponent's private shed, hands, and seeds."""
    shed = opponent_private.get("shed") or {}
    seeds = opponent_private.get("seeds") or {}
    carried = {
        item: sum(int(unit.get(item, 0) or 0) for unit in opponent_private.get("inventories") or [])
        for item in (*PRODUCTS, *ANIMALS)
    }
    products = np.asarray(
        [
            (float(shed.get(item, 0) or 0) / SHED_CAPACITY, carried[item] / SHED_CAPACITY)
            for item in PRODUCTS
        ],
        dtype=np.float32,
    )
    animals = np.asarray(
        [
            (float(shed.get(item, 0) or 0) / SHED_CAPACITY, carried[item] / SHED_CAPACITY)
            for item in ANIMALS
        ],
        dtype=np.float32,
    )
    crops = np.asarray(
        [(float(seeds.get(crop, 0) or 0) / SHED_CAPACITY,) for crop in CROPS],
        dtype=np.float32,
    )
    return products, animals, crops


@dataclass(frozen=True)
class StructuredObservation:
    """One player's full token bundle in rollout staging dtypes.

    Integer index arrays stage as int8 (every vocabulary and tile index fits)
    and continuous features as float16, mirroring how the flat encoder's
    outputs persist in rollouts; consumers upcast per batch. Critic-only
    fields are None when the opponent's private state is unavailable.
    """

    tile_categorical: np.ndarray  # [2 * TILE_COUNT, N_TILE_CATEGORICAL] int8
    tile_continuous: np.ndarray  # [2 * TILE_COUNT, N_TILE_CONTINUOUS] float16
    unit_categorical: np.ndarray  # [MAX_UNITS, N_UNIT_CATEGORICAL] int8
    unit_continuous: np.ndarray  # [MAX_UNITS, N_UNIT_CONTINUOUS] float16
    unit_active: np.ndarray  # [MAX_UNITS] bool
    unit_tile_gather: np.ndarray  # [MAX_UNITS, 5] int8
    unit_tile_gather_valid: np.ndarray  # [MAX_UNITS, 5] bool
    products: np.ndarray  # [len(PRODUCTS), len(PRODUCT_TOKEN_FIELDS)] float16
    animals: np.ndarray  # [len(ANIMALS), len(ANIMAL_TOKEN_FIELDS)] float16
    crops: np.ndarray  # [len(CROPS), len(CROP_TOKEN_FIELDS)] float16
    farms: np.ndarray  # [2, len(FARM_TOKEN_FIELDS)] float16
    town: np.ndarray  # [len(TOWN_TOKEN_FIELDS)] float16
    critic_products: np.ndarray | None  # [len(PRODUCTS), 2] float16
    critic_animals: np.ndarray | None  # [len(ANIMALS), 2] float16
    critic_crops: np.ndarray | None  # [len(CROPS), 1] float16
    opponent_unit_categorical: np.ndarray | None  # [MAX_UNITS, 4] int8
    opponent_unit_continuous: np.ndarray | None  # [MAX_UNITS, ...] float16
    opponent_unit_active: np.ndarray | None  # [MAX_UNITS] bool


def encode_structured_observation(
    observation: dict,
    opponent_private: dict | None = None,
) -> StructuredObservation:
    """Tokenize one observation into the structured model's input bundle."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if len(farms) != 2:
        raise ValueError(f"expected exactly two farms, got {len(farms)}")
    day = int(observation.get("day", 0) or 0)
    hour = int(observation.get("hour", 0) or 0)
    step = int(observation.get("step", day * TURNS_PER_DAY + hour) or 0)

    own = tokenize_farm_tiles(farms[player], day, step, opponent=False)
    other = tokenize_farm_tiles(farms[1 - player], day, step, opponent=True)
    units = tokenize_units(farms[player], observation.get("private") or {})
    economy = tokenize_economy(observation)

    critic_products = critic_animals = critic_crops = None
    opponent_categorical = opponent_continuous = opponent_active = None
    if opponent_private is not None:
        critic_products, critic_animals, critic_crops = opponent_economy_columns(opponent_private)
        critic_products = critic_products.astype(np.float16)
        critic_animals = critic_animals.astype(np.float16)
        critic_crops = critic_crops.astype(np.float16)
        opponent_units = tokenize_units(farms[1 - player], opponent_private)
        opponent_categorical = opponent_units.categorical.astype(np.int8)
        opponent_continuous = opponent_units.continuous.astype(np.float16)
        opponent_active = opponent_units.active

    return StructuredObservation(
        tile_categorical=np.concatenate((own.categorical, other.categorical)).astype(np.int8),
        tile_continuous=np.concatenate((own.continuous, other.continuous)).astype(np.float16),
        unit_categorical=units.categorical.astype(np.int8),
        unit_continuous=units.continuous.astype(np.float16),
        unit_active=units.active,
        unit_tile_gather=units.tile_gather.astype(np.int8),
        unit_tile_gather_valid=units.tile_gather_valid,
        products=economy.products.astype(np.float16),
        animals=economy.animals.astype(np.float16),
        crops=economy.crops.astype(np.float16),
        farms=economy.farms.astype(np.float16),
        town=economy.town.astype(np.float16),
        critic_products=critic_products,
        critic_animals=critic_animals,
        critic_crops=critic_crops,
        opponent_unit_categorical=opponent_categorical,
        opponent_unit_continuous=opponent_continuous,
        opponent_unit_active=opponent_active,
    )


def clock_features(observation: dict) -> np.ndarray:
    """Day/hour phase and horizon features shared by every token consumer."""
    day = int(observation.get("day", 0) or 0)
    hour = int(observation.get("hour", 0) or 0)
    step = int(observation.get("step", day * TURNS_PER_DAY + hour) or 0)
    cycle = 2.0 * np.pi * hour / TURNS_PER_DAY
    return np.asarray(
        (
            day / _EPISODE_DAYS,
            hour / float(TURNS_PER_DAY),
            step / float(EPISODE_STEPS - 1),
            (EPISODE_STEPS - 1 - step) / float(EPISODE_STEPS - 1),
            np.sin(cycle),
            np.cos(cycle),
        ),
        dtype=np.float32,
    )


__all__ = [
    "ANIMAL_PRIVATE_FIELDS",
    "ANIMAL_TOKEN_FIELDS",
    "CROP_PRIVATE_FIELDS",
    "CROP_TOKEN_FIELDS",
    "FARM_IDENTITIES",
    "FARM_TOKEN_FIELDS",
    "N_TILE_CATEGORICAL",
    "N_TILE_CONTINUOUS",
    "N_UNIT_CATEGORICAL",
    "N_UNIT_CONTINUOUS",
    "OBSERVATION_SCHEMA_VERSION",
    "PRODUCT_PRIVATE_FIELDS",
    "PRODUCT_TOKEN_FIELDS",
    "QUADRANT_COUNT",
    "TILE_CATEGORICAL_FIELDS",
    "TILE_CONTINUOUS_FIELDS",
    "TILE_COUNT",
    "TILE_KINDS",
    "TILE_KIND_INDEX",
    "TILE_OCCUPANTS",
    "TILE_OCCUPANT_INDEX",
    "TOWN_TOKEN_FIELDS",
    "UNIT_CATEGORICAL_FIELDS",
    "UNIT_CONTINUOUS_FIELDS",
    "UNIT_ROLES",
    "UNIT_TILE_GATHERS",
    "EconomyTokens",
    "StructuredObservation",
    "TileTokens",
    "UnitTokens",
    "clock_features",
    "encode_structured_observation",
    "opponent_economy_columns",
    "tokenize_economy",
    "tokenize_farm_tiles",
    "tokenize_units",
]
