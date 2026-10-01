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

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from kaggriculture.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    ANIMAL_YIELD_INTERVAL,
    ANIMALS,
    BASE_PRICE,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    CROP_YIELD_INTERVAL,
    CROPS,
    EPISODE_STEPS,
    MARKET_I0,
    MAX_SHOP_INSTANCES,
    MAX_UNITS,
    ONGOING_CROPS,
    PRIVATE_ITEMS,
    PRODUCTS,
    SEED_COST,
    SHED_CAPACITY,
    SHOP_NAMES,
    SHOP_PRODUCTS,
    TOWN_CENTER_PRODUCTS,
    TOWN_CENTER_SELL_INTERVAL,
    TOWN_SHOP_SELL_INTERVAL,
    TOWN_SHOP_UNLOCK_INTERVAL,
    TURNS_PER_DAY,
    market_price,
    restocked_inventory,
    sale_proceeds,
    shed_access_tiles,
)

TILE_COUNT = BOARD_SIZE * BOARD_SIZE
# Every supported schema reads a prefix of one tokenized layout, so a single
# tokenization (Python here, the native extension for rollouts) serves models of
# every supported version from the same staged arrays -- older artifacts keep
# acting on current rollouts and inference -- and one encoded BC cache serves
# them all. (League snapshots must still match the learner's whole config.)
# `OBSERVATION_SCHEMA_VERSION` names the newest schema, whose layout both
# tokenizers emit; a model's config names the schema it consumes, and its
# embedder slices that schema's prefix (`product_token_fields`,
# `farm_token_fields`, `town_token_fields`; the critic's private columns by
# `product_private_fields`).
#
# v3: per-unit carried-item insertion ranks (README).
# v4: adds the farm token's `money_margin`.
# v5: adds the town token's per-shop `first_unlock` positions.
# v6: adds the product token's `held_value` and the farm token's `liquidation`
#     and `liquidation_margin` (and the critic's `opponent_held_value`).
# v7: adds the product token's supply/demand outlook, `forecast_price` through
#     `town_draw_to_end` (and the critic's `opponent_forecast_price`).
# v8: adds the crop and animal tokens' `payback` and `forecast_payback`.
OBSERVATION_SCHEMA_VERSION = 8
SUPPORTED_OBSERVATION_SCHEMA_VERSIONS = frozenset((3, 4, 5, 6, 7, 8))
# The common model-config default remains v3 for entity-attention and the other
# structured families. Fresh LeJEPA configs, production's, override it to v4; saved v3 model
# configs still carry their explicit version for loading and resume.
DEFAULT_OBSERVATION_SCHEMA_VERSION = 3

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


def _tile_slot_positions(board_size: int) -> np.ndarray:
    """(row, column, quadrant) per tile token; token order is ``y * board_size + x``."""
    positions = np.empty((board_size * board_size, 3), dtype=np.int64)
    for y in range(board_size):
        for x in range(board_size):
            positions[y * board_size + x] = (y, x, _quadrant(x, y, board_size))
    return positions


#: The ``farm, row, column, quadrant`` categorical columns of every tile token in
#: a two-farm observation (own farm first). They depend only on the token slot,
#: never on game state; both encoders and the tile embedder rely on that.
TILE_SLOT_CATEGORICAL = np.concatenate(
    [
        np.concatenate(
            (np.full((TILE_COUNT, 1), farm, dtype=np.int64), _tile_slot_positions(BOARD_SIZE)),
            axis=1,
        )
        for farm in range(len(FARM_IDENTITIES))
    ]
)
TILE_SLOT_CATEGORICAL.setflags(write=False)


def _shed_steps_map(board_size: int) -> np.ndarray:
    """Moves from each tile to the nearest shed-access tile, one tile a step."""
    access = shed_access_tiles(board_size)
    grid = np.empty((board_size, board_size), dtype=np.int64)
    for y in range(board_size):
        for x in range(board_size):
            grid[y, x] = min(abs(x - ax) + abs(y - ay) for ax, ay in access)
    return grid


_SHED_STEPS = _shed_steps_map(BOARD_SIZE)
_SHED_DISTANCE = _SHED_STEPS.astype(np.float32) / np.float32(_SHED_STEPS.max())
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

    farm_slots = slice(farm_identity * TILE_COUNT, (farm_identity + 1) * TILE_COUNT)
    categorical[:, 2:6] = TILE_SLOT_CATEGORICAL[farm_slots]
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            token = y * BOARD_SIZE + x
            tile = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else "LOCKED"
            row = categorical[token]
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
    # Schema v6. The exact coins selling every unit of this product the seat
    # holds (shed and hands) into the current market would bank, one unit at a
    # time: each sale restocks the market, so the quote walks down the curve
    # (`sale_proceeds`, the engine's sell arithmetic). / HELD_VALUE_SCALE, so a
    # full shed of melons at base price is 1. `price` times the stock
    # overstates a large stock by exactly the impact this prices in.
    "held_value",
    # Schema v7: the product's outlook to the end of the game. Supply is what
    # the public tiles will yield under nominal care (`_farm_supply`): every
    # plant watered and every animal fed daily, each harvested when its yield
    # stops growing, no fertilizer or care beyond what the tiles already hold;
    # a harvest counts if it can still reach the shed and sell before the end
    # (`_sells_by_the_end`). Demand is the town's draw (`_town_draw`): the
    # open shop instances plus the ones still to open, each an expected
    # (uniform) shop. "Soon" is the next `OUTLOOK_SOON_STEPS` steps, "to end"
    # every step still to act.
    #
    # The quote, / (2 * BASE_PRICE) like `price`, at the market inventory
    # left if this seat sold everything it holds and both farms' supply to the
    # end, the animals ate their daily WHEAT, and the town drew its expected
    # demand (`_forecast_prices`), rounded to a unit. The opponent's holdings
    # are private, so the critic reads its own view as
    # `opponent_forecast_price`.
    "forecast_price",
    "supply_soon",  # units this farm's tiles yield soon / SHED_CAPACITY
    "supply_to_end",  # units this farm's tiles yield to the end / SHED_CAPACITY
    "opponent_supply_soon",  # the other farm's, the same way
    "opponent_supply_to_end",
    "town_draw_soon",  # expected units the town takes soon / SHED_CAPACITY
    "town_draw_to_end",  # expected units the town takes to the end / SHED_CAPACITY
)
_PRODUCT_TOKEN_WIDTHS = {3: 5, 4: 5, 5: 5, 6: 6, 7: 13, 8: 13}
# Coins per unit of `held_value`: a full shed at the highest base price.
HELD_VALUE_SCALE = float(SHED_CAPACITY * max(BASE_PRICE.values()))
# The outlook's near horizon: two days, about one crop or animal cycle.
OUTLOOK_SOON_STEPS = 2 * TURNS_PER_DAY
# Schema v8, on the crop and animal tokens: what starting one more now
# returns before the game ends. A crop sown, or an animal placed, this day
# yields what `_tile_arrivals` gives a fresh tile under nominal care, a crop
# still growing on the last acting day harvested then (`_started_now_yields`):
# for an animal, its product and a FERTILIZER a day. `payback` values that
# yield at the current prices over its cost -- the seed, or the animal and the
# WHEAT that feeds it daily until the last acting day (`_paybacks`) -- and
# `forecast_payback` at the seat's `forecast_price`s. Both are log1p of that
# ratio, unscaled: 0 once nothing more can be harvested, log 2 at breakeven.
_PAYBACK_FIELDS = ("payback", "forecast_payback")
ANIMAL_TOKEN_FIELDS = (
    "purchase_price",  # ANIMAL_COST / max ANIMAL_COST; animals have no market quote
    "shed_stock",  # own shed count / SHED_CAPACITY
    "carried_stock",  # summed across own units / SHED_CAPACITY
    *_PAYBACK_FIELDS,
)
_ANIMAL_TOKEN_WIDTHS = {3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 5}
CROP_TOKEN_FIELDS = (
    "seed_cost",  # SEED_COST / max SEED_COST
    "seeds_held",  # own private seed count / SHED_CAPACITY
    "first_yield_day",  # CROP_FIRST_YIELD_DAY / max CROP_MAX_YIELD_DAY
    "max_yield_day",  # CROP_MAX_YIELD_DAY / max CROP_MAX_YIELD_DAY
    "max_yield",  # CROP_MAX_YIELD / max CROP_MAX_YIELD
    "ongoing",  # keeps producing after first yield
    *_PAYBACK_FIELDS,
)
_CROP_TOKEN_WIDTHS = {3: 6, 4: 6, 5: 6, 6: 6, 7: 6, 8: 8}
FARM_TOKEN_FIELDS = (
    "money",  # signed log1p scale shared with the flat encoder
    "unlocked_quadrants",  # / 4
    "hands",  # hired hands / (MAX_UNITS - 1)
    "hires_today",  # / (MAX_UNITS - 1)
    # Schema v4. This farm's signed log1p money minus the other farm's,
    # unscaled: ln((1 + own) / (1 + other)) for nonnegative money, so the two
    # rows are exact negatives. `money` sits near 0.94 late in a game, where a
    # bf16 input step is 2^-8 -- about 5% of a bank -- so two close banks are
    # often the same number to the model. The margin is formed in float64
    # before any rounding and is small exactly when the game is close, where
    # floating point is finest: a 1% lead is 0.01, a 10x blowout 2.3. Leaving
    # it unscaled (not /12 like `money`) keeps the decision-relevant range at
    # the magnitude of the other [0, 1] inputs rather than 12x below them.
    "money_margin",
    # Schema v6. What this farm would bank selling out now -- money plus its
    # products' `held_value` in coins, the engine's liquidation value (the
    # shaping potential's input) -- on `money`'s signed log1p / 12 scale.
    # Held stock is private, so the opponent row carries the opponent's money
    # alone, the public lower bound on its liquidation; the centralized critic
    # reads the opponent's true proceeds from `opponent_held_value`.
    "liquidation",
    # Schema v6. `money_margin` over the `liquidation` column's coins: this
    # row's signed log1p liquidation minus the other row's, formed in float64,
    # unscaled, so the rows are exact negatives. From the own row it is the
    # margin the seat would lead by if it sold out now and the opponent kept
    # only its bank: a stockpile no longer reads as a deficit.
    "liquidation_margin",
)
_FARM_TOKEN_WIDTHS = {3: 4, 4: 5, 5: 5, 6: 7, 7: 7, 8: 7}
TOWN_TOKEN_FIELDS = (
    "day",
    "hour",
    "progress",
    "remaining",
    "hour_sin",
    "hour_cos",
    *(f"shop_{name}" for name in SHOP_NAMES),  # unlocked instances / 8
    # Schema v5. (1 + the index of this shop's first instance in the town's
    # unlock order) / 8, or 0 while it is locked. Counts alone are a multiset:
    # they cannot say which shop opened first, and demand planning keyed on the
    # opening shops reads that order. With the counts these recover the order
    # of the distinct shops; where a repeat instance fell stays unencoded.
    *(f"shop_{name}_first_unlock" for name in SHOP_NAMES),
)
_TOWN_TOKEN_WIDTHS = {
    3: 6 + len(SHOP_NAMES),
    4: 6 + len(SHOP_NAMES),
    5: 6 + 2 * len(SHOP_NAMES),
    6: 6 + 2 * len(SHOP_NAMES),
    7: 6 + 2 * len(SHOP_NAMES),
    8: 6 + 2 * len(SHOP_NAMES),
}


# Extra economy columns the centralized critic appends from the opponent's
# private state; the actor never sees them. Each is the opponent's own view of
# the matching actor column, which is how rollouts stage it (the paired seat's
# tokens, `rollout._PRODUCT_STOCK_COLUMNS`), and it is schema-sliced like them.
PRODUCT_PRIVATE_FIELDS = (
    "opponent_shed_stock",  # opponent shed count / SHED_CAPACITY
    "opponent_carried_stock",  # summed across opponent units / SHED_CAPACITY
    "opponent_held_value",  # schema v6: the opponent's `held_value`
    "opponent_forecast_price",  # schema v7: the opponent's `forecast_price`
)
_PRODUCT_PRIVATE_WIDTHS = {3: 2, 4: 2, 5: 2, 6: 3, 7: 4, 8: 4}
ANIMAL_PRIVATE_FIELDS = (
    "opponent_shed_stock",
    "opponent_carried_stock",
)
CROP_PRIVATE_FIELDS = ("opponent_seeds_held",)  # opponent seeds / SHED_CAPACITY
# The opponent's `forecast_payback` has no private column: it is its
# `forecast_price`s read through the public yields, which the critic already
# has as `opponent_forecast_price`, and a column after `seeds_held` would break
# the contiguous run the paired-seat staging slices.


@dataclass(frozen=True)
class EconomyTokens:
    """Market, crop, farm-summary, and town/clock tokens for one viewpoint."""

    products: np.ndarray  # [len(PRODUCTS), len(PRODUCT_TOKEN_FIELDS)] float32
    animals: np.ndarray  # [len(ANIMALS), len(ANIMAL_TOKEN_FIELDS)] float32
    crops: np.ndarray  # [len(CROPS), len(CROP_TOKEN_FIELDS)] float32
    farms: np.ndarray  # [2, len(FARM_TOKEN_FIELDS)] float32, own farm first
    town: np.ndarray  # [len(TOWN_TOKEN_FIELDS)] float32


def _schema_prefix(
    fields: tuple[str, ...], widths: dict[int, int], schema_version: int
) -> tuple[str, ...]:
    try:
        return fields[: widths[schema_version]]
    except KeyError:
        raise ValueError(f"unsupported observation schema version {schema_version!r}") from None


def product_token_fields(schema_version: int) -> tuple[str, ...]:
    """The prefix of ``PRODUCT_TOKEN_FIELDS`` a model of this schema consumes."""
    return _schema_prefix(PRODUCT_TOKEN_FIELDS, _PRODUCT_TOKEN_WIDTHS, schema_version)


def animal_token_fields(schema_version: int) -> tuple[str, ...]:
    """The prefix of ``ANIMAL_TOKEN_FIELDS`` a model of this schema consumes."""
    return _schema_prefix(ANIMAL_TOKEN_FIELDS, _ANIMAL_TOKEN_WIDTHS, schema_version)


def crop_token_fields(schema_version: int) -> tuple[str, ...]:
    """The prefix of ``CROP_TOKEN_FIELDS`` a model of this schema consumes."""
    return _schema_prefix(CROP_TOKEN_FIELDS, _CROP_TOKEN_WIDTHS, schema_version)


def farm_token_fields(schema_version: int) -> tuple[str, ...]:
    """The prefix of ``FARM_TOKEN_FIELDS`` a model of this schema consumes."""
    return _schema_prefix(FARM_TOKEN_FIELDS, _FARM_TOKEN_WIDTHS, schema_version)


def town_token_fields(schema_version: int) -> tuple[str, ...]:
    """The prefix of ``TOWN_TOKEN_FIELDS`` a model of this schema consumes."""
    return _schema_prefix(TOWN_TOKEN_FIELDS, _TOWN_TOKEN_WIDTHS, schema_version)


def product_private_fields(schema_version: int) -> tuple[str, ...]:
    """The prefix of ``PRODUCT_PRIVATE_FIELDS`` a critic of this schema consumes."""
    return _schema_prefix(PRODUCT_PRIVATE_FIELDS, _PRODUCT_PRIVATE_WIDTHS, schema_version)


def _signed_log_money(value: float) -> float:
    amount = float(value or 0)
    return float(np.copysign(np.log1p(abs(amount)), amount))


def _money_feature(value: float) -> float:
    # Division is sign-symmetric, so this is bit-identical to scaling inside
    # the copysign, which is how the native tokenizer orders it.
    return _signed_log_money(value) / 12.0


def _carried_items(private: dict) -> dict[str, int]:
    """Each product and animal summed across one seat's unit inventories."""
    return {
        item: sum(int(unit.get(item, 0) or 0) for unit in private.get("inventories") or [])
        for item in (*PRODUCTS, *ANIMALS)
    }


def _held_units(shed: dict, carried: dict[str, int]) -> dict[str, int]:
    """Each product one seat holds, in its shed and its units' hands."""
    return {item: int(shed.get(item, 0) or 0) + carried[item] for item in PRODUCTS}


def _held_proceeds(held: dict[str, int], market: dict) -> dict[str, int]:
    """Each product's exact coins if one seat sold its ``held`` stock now."""
    inventory = market.get("inventory") or {}
    return {
        item: sale_proceeds(
            item, held[item], int(inventory.get(item, MARKET_I0) or 0), market.get("params")
        )
        for item in PRODUCTS
    }


# The outlook reads the default game's rules from `kaggriculture.constants`,
# which a test pins to the official engine; an observation carries no game
# configuration. The native tokenizer reads its own `GameConfig`, which the
# extension only ever builds at these defaults, so the two always agree.
#
# The steps still to act: the engine applies actions through step
# EPISODE_STEPS - 2 and ends the game there, so an arrival counts before this.
_OUTLOOK_END = EPISODE_STEPS - 1
# The last day whose first step still acts.
_LAST_DAY = (_OUTLOOK_END - 1) // TURNS_PER_DAY
# Units of the town's expected draw: a shop still to open is each of the
# SHOP_NAMES with equal chance, so its draw is a whole number of these parts.
_SHOP_CHOICES = len(SHOP_NAMES)
# Parts of a unit each product loses to one sell event of one unopened shop.
_UNOPENED_SHOP_DRAW = {
    item: sum(
        (2 if len(products) == 1 else 1) * (item in products) for products in SHOP_PRODUCTS.values()
    )
    for item in PRODUCTS
}


@dataclass(frozen=True)
class MarketOutlook:
    """Public supply and the town's expected demand, per product (`PRODUCT_TOKEN_FIELDS`)."""

    supply_soon: tuple[dict[str, int], dict[str, int]]  # own farm, then the other's
    supply_to_end: tuple[dict[str, int], dict[str, int]]
    draw_soon: dict[str, int]  # in 1/_SHOP_CHOICES parts of a unit
    draw_to_end: dict[str, int]
    feeds_to_end: int  # WHEAT both farms' animals eat (`_feeds_by_the_end`)


def _tile_arrivals(
    tile: dict, step: int, *, harvest_by_day: int | None = None
) -> Iterator[tuple[str, int, int]]:
    """Each harvest ``(item, units, at)`` of ``tile`` under nominal care.

    Mirrors the engine's yield rules (kaggriculture.py `_apply_unit_action`
    WATER and HARVEST, `_daily_refresh_plants`, `_daily_refresh_animals`) with
    CROP_* and ANIMAL_* constants, for a farm that waters and feeds every tile
    daily and harvests each one as soon as its yield stops growing, from
    ``step`` on (``at`` may pass the end of the game). Several units may
    stand on a tile and the engine applies their actions in turn, so a
    harvest can follow the day's watering in the same step:

    - a single-yield crop's `yield_units` grows by one (two while fertilized,
      `fertilized_until_day >= day`) on each watered day whose age is in
      [ceil(max_yield_day / 2), max_yield_day], up to its max yield; it is
      harvested from `first_yield_day` on, on the last day it grew, and past
      its lifespan it is harvested at once; given ``harvest_by_day``, it
      stops growing after that day, so one still growing is harvested then;
    - an ongoing crop's standing yield is harvested now, and each of its
      max-yield productions lands at the start of day planted_day +
      first_yield_day + k * interval (two units if fertilized the day before);
    - an animal's standing yield is harvested now, and each production day
      placed_day + first_yield_day + k * interval adds one unit plus the care
      bonus it has pending (today's care joins it after tomorrow's
      production), up to its max held; each animal also gives one FERTILIZER
      a day, and one now if it is still available.
    """
    day = step // TURNS_PER_DAY
    stock = int(tile.get("yield_units", 0) or 0)
    fertilized_until = int(tile.get("fertilized_until_day", -1))
    if tile.get("kind") == "PLANT":
        crop = tile["crop"]
        planted = int(tile["planted_day"])
        if crop in ONGOING_CROPS:
            yield crop, stock, step
            for production in range(CROP_MAX_YIELD[crop]):
                at_day = (
                    planted + CROP_FIRST_YIELD_DAY[crop] + production * CROP_YIELD_INTERVAL[crop]
                )
                if at_day > day:
                    units = 2 if fertilized_until >= at_day - 1 else 1
                    yield crop, units, at_day * TURNS_PER_DAY
            return
        grown = max(day, planted + CROP_FIRST_YIELD_DAY[crop])
        first_watering = max(
            day + bool(tile.get("watered_today")),
            planted + (CROP_MAX_YIELD_DAY[crop] + 1) // 2,
        )
        last_watering = planted + CROP_MAX_YIELD_DAY[crop]
        if harvest_by_day is not None:
            last_watering = min(last_watering, harvest_by_day)
        for watering in range(first_watering, last_watering + 1):
            if stock < CROP_MAX_YIELD[crop]:
                bonus = 2 if fertilized_until >= watering else 1
                stock = min(CROP_MAX_YIELD[crop], stock + bonus)
                grown = max(grown, watering)
        yield crop, stock, max(step, grown * TURNS_PER_DAY)
        return
    animal = tile.get("animal")
    if animal is None:
        return
    product = ANIMAL_PRODUCT[animal]
    yield product, stock, step
    yield "FERTILIZER", int(bool(tile.get("fertilizer_available"))), step
    pending = int(tile.get("pending_care_bonus", 0) or 0)
    # A production at the start of a day lands at that day's first step.
    for at_day in range(day + 1, _LAST_DAY + 1):
        since_first = at_day - int(tile["placed_day"]) - ANIMAL_FIRST_YIELD_DAY[animal]
        if since_first >= 0 and since_first % ANIMAL_YIELD_INTERVAL[animal] == 0:
            units = min(ANIMAL_MAX_HELD[animal], 1 + pending)
            yield product, units, at_day * TURNS_PER_DAY
            pending = 0
        if at_day == day + 1 and tile.get("cared_today"):
            pending += 1
        yield "FERTILIZER", 1, at_day * TURNS_PER_DAY


def _sells_by_the_end(at: int, shed_steps: int) -> bool:
    """Whether a harvest at step ``at``, ``shed_steps`` moves from the shed, can still sell.

    Markets clear after units act (kaggriculture.py `interpreter`), so the
    harvesting unit sells on the step it walks up and drops the harvest in
    the shed (`_apply_unit_action` DROP), or the day's end drops it there
    (`_drop_inventories_to_shed`) to sell on the next day's first step.
    """
    walked = at + shed_steps + 1
    return min(walked, (at // TURNS_PER_DAY + 1) * TURNS_PER_DAY) < _OUTLOOK_END


def _feeds_by_the_end(tile: dict, step: int) -> int:
    """WHEAT nominal care feeds an animal: one each day before the last acting day.

    A feed on that day keeps no production (`_tile_arrivals`); one already
    given today is spent.
    """
    days = _LAST_DAY - step // TURNS_PER_DAY
    return max(0, days - bool(tile.get("fed_today")))


def _farm_supply(farm: dict, step: int) -> tuple[dict[str, int], dict[str, int], int]:
    """The farm's nominal-care yields that sell (`_tile_arrivals`), soon and to the end.

    Also the WHEAT its animals eat meanwhile (`_feeds_by_the_end`).
    """
    soon = dict.fromkeys(PRODUCTS, 0)
    to_end = dict.fromkeys(PRODUCTS, 0)
    feeds = 0
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            for item, units, at in _tile_arrivals(tile, step):
                if _sells_by_the_end(at, int(_SHED_STEPS[y, x])):
                    to_end[item] += units
                    if at < step + OUTLOOK_SOON_STEPS:
                        soon[item] += units
            if tile.get("animal"):
                feeds += _feeds_by_the_end(tile, step)
    return soon, to_end, feeds


def _started_now(step: int) -> dict[str, dict]:
    """A fresh tile of each crop sown and each animal placed at ``step``.

    The engine's `_new_plant` and `_new_animal`, keeping the fields
    `_tile_arrivals` reads.
    """
    day = step // TURNS_PER_DAY
    tiles = {
        crop: {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": day,
            "watered_today": False,
            "yield_units": 0 if crop in ONGOING_CROPS else 1,
            "fertilized_until_day": -1,
        }
        for crop in CROPS
    }
    for animal in ANIMALS:
        tiles[animal] = {
            "kind": ANIMAL_STRUCTURE[animal],
            "animal": animal,
            "placed_day": day,
            "yield_units": 0,
            "cared_today": False,
            "fertilizer_available": False,
            "pending_care_bonus": 0,
        }
    return tiles


def _yield_by_the_end(tile: dict, step: int) -> dict[str, int]:
    """Units ``tile``, beside the shed, yields and sells under nominal care (`_tile_arrivals`).

    A crop still growing on the last acting day is harvested then with what
    it holds.
    """
    units = dict.fromkeys(PRODUCTS, 0)
    for item, count, at in _tile_arrivals(tile, step, harvest_by_day=_LAST_DAY):
        if _sells_by_the_end(at, 0):
            units[item] += count
    return units


def _started_now_yields(step: int) -> dict[str, dict[str, int]]:
    """Units each crop sown and each animal placed at ``step`` yields before the game ends."""
    return {name: _yield_by_the_end(tile, step) for name, tile in _started_now(step).items()}


def _paybacks(
    yields: dict[str, dict[str, int]], step: int, prices: dict[str, int]
) -> dict[str, float]:
    """log1p of each started crop's or animal's yield value over its cost at ``prices``.

    The cost is the seed, or the animal plus the fewest WHEAT (the engine's
    FEED) that keep it from escaping through the daily refreshes from today up
    to but not including the last acting day's. `_daily_refresh_animals`
    produces whether or not the animal was fed, and a fresh one earns no care
    bonus for feeding to spend, so one feed every _DECAY_LETHAL refreshes
    yields as much as daily feeding. FERTILIZER an animal gives counts at its
    price like its product. `math.log1p` is the C library's, like Rust's
    `f64::ln_1p`.
    """
    refreshes = max(0, _LAST_DAY - step // TURNS_PER_DAY)
    feeds = refreshes // int(_DECAY_LETHAL)
    paybacks = {}
    for name, units in yields.items():
        value = sum(count * prices[item] for item, count in units.items())
        cost = ANIMAL_COST[name] + feeds * prices["WHEAT"] if name in ANIMALS else SEED_COST[name]
        paybacks[name] = math.log1p(value / cost)
    return paybacks


def _sell_events(interval: int, start: int, stop: int) -> int:
    """Steps in [start, stop) divisible by ``interval``: the town's sell events."""
    return max(0, -(-stop // interval) - -(-start // interval))


def _town_draw(shops: list[str], step: int, stop: int) -> dict[str, int]:
    """Units the town is expected to take in [step, stop), in 1/_SHOP_CHOICES parts.

    Mirrors the engine's `_town_consume`, and `_end_of_day`'s shop draw: the
    open instances sell at every TOWN_SHOP_SELL_INTERVAL step, the town
    center at every TOWN_CENTER_SELL_INTERVAL step, and from each day
    divisible by TOWN_SHOP_UNLOCK_INTERVAL one more instance sells too, until
    MAX_SHOP_INSTANCES are open, as each shop with equal chance.
    """
    draw = dict.fromkeys(PRODUCTS, 0)
    shop_events = _sell_events(TOWN_SHOP_SELL_INTERVAL, step, stop)
    for shop in shops:
        products = SHOP_PRODUCTS[shop]
        for item in products:
            draw[item] += _SHOP_CHOICES * (2 if len(products) == 1 else 1) * shop_events
    center_events = _sell_events(TOWN_CENTER_SELL_INTERVAL, step, stop)
    for item in TOWN_CENTER_PRODUCTS:
        draw[item] += _SHOP_CHOICES * center_events
    opened = len(shops)
    for day in range(step // TURNS_PER_DAY + 1, -(-stop // TURNS_PER_DAY)):
        if opened >= MAX_SHOP_INSTANCES:
            break
        if day % TOWN_SHOP_UNLOCK_INTERVAL == 0:
            opened += 1
            events = _sell_events(TOWN_SHOP_SELL_INTERVAL, day * TURNS_PER_DAY, stop)
            for item in PRODUCTS:
                draw[item] += _UNOPENED_SHOP_DRAW[item] * events
    return draw


def market_outlook(observation: dict) -> MarketOutlook:
    """The seat's view of every product's public supply and expected demand."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    day = int(observation.get("day", 0) or 0)
    step = int(observation.get("step", day * TURNS_PER_DAY + int(observation.get("hour", 0) or 0)))
    own_soon, own_to_end, own_feeds = _farm_supply(farms[player], step)
    other_soon, other_to_end, other_feeds = _farm_supply(farms[1 - player], step)
    shops = (observation.get("town") or {}).get("unlocked_shops") or []
    return MarketOutlook(
        supply_soon=(own_soon, other_soon),
        supply_to_end=(own_to_end, other_to_end),
        draw_soon=_town_draw(shops, step, min(step + OUTLOOK_SOON_STEPS, _OUTLOOK_END)),
        draw_to_end=_town_draw(shops, step, _OUTLOOK_END),
        feeds_to_end=own_feeds + other_feeds,
    )


def _forecast_prices(outlook: MarketOutlook, held: dict[str, int], market: dict) -> dict[str, int]:
    """Each product's quote once ``held`` and all supply sell, the animals eat, and the town draws.

    The sales come first and restock the market only while above the price
    floor (`restocked_inventory`); the WHEAT the animals eat, bought or kept
    back from sale, then leaves it. The inventory is formed in exact
    1/_SHOP_CHOICES parts and rounded half up to a unit before the engine's
    price curve reads it.
    """
    inventory = market.get("inventory") or {}
    prices = {}
    for item in PRODUCTS:
        sold = held[item] + outlook.supply_to_end[0][item] + outlook.supply_to_end[1][item]
        units = restocked_inventory(
            item, sold, int(inventory.get(item, MARKET_I0) or 0), market.get("params")
        )
        if item == "WHEAT":
            units -= outlook.feeds_to_end
        parts = _SHOP_CHOICES * units - outlook.draw_to_end[item]
        rounded = (parts + _SHOP_CHOICES // 2) // _SHOP_CHOICES
        prices[item] = market_price(item, rounded, market.get("params"))
    return prices


def tokenize_economy(observation: dict, outlook: MarketOutlook | None = None) -> EconomyTokens:
    """Tokenize market, crop, farm-summary, and town state for the actor.

    ``outlook`` is the observation's `market_outlook`, when already computed.
    """
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
    carried = _carried_items(private)
    held = _held_units(shed, carried)
    proceeds = _held_proceeds(held, market)
    outlook = outlook or market_outlook(observation)
    forecast = _forecast_prices(outlook, held, market)
    quotes = {item: int(prices.get(item, BASE_PRICE[item]) or 0) for item in PRODUCTS}
    day = int(observation.get("day", 0) or 0)
    step = int(observation.get("step", day * TURNS_PER_DAY + int(observation.get("hour", 0) or 0)))
    yields = _started_now_yields(step)
    payback = _paybacks(yields, step, quotes)
    forecast_payback = _paybacks(yields, step, forecast)
    max_base_price = float(max(BASE_PRICE.values()))
    draw_scale = float(_SHOP_CHOICES * SHED_CAPACITY)

    products = np.asarray(
        [
            (
                (float(inventory.get(item, MARKET_I0) or 0) - MARKET_I0) / 500.0,
                float(prices.get(item, BASE_PRICE[item]) or 0) / (2.0 * BASE_PRICE[item]),
                BASE_PRICE[item] / max_base_price,
                float(shed.get(item, 0) or 0) / SHED_CAPACITY,
                carried[item] / SHED_CAPACITY,
                proceeds[item] / HELD_VALUE_SCALE,
                forecast[item] / (2.0 * BASE_PRICE[item]),
                outlook.supply_soon[0][item] / SHED_CAPACITY,
                outlook.supply_to_end[0][item] / SHED_CAPACITY,
                outlook.supply_soon[1][item] / SHED_CAPACITY,
                outlook.supply_to_end[1][item] / SHED_CAPACITY,
                outlook.draw_soon[item] / draw_scale,
                outlook.draw_to_end[item] / draw_scale,
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
                payback[animal],
                forecast_payback[animal],
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
                payback[crop],
                forecast_payback[crop],
            )
            for crop in CROPS
        ],
        dtype=np.float32,
    )
    # The seat's own stock is the only held stock it observes (see `liquidation`).
    liquidation = {
        player: float(farms[player].get("money", 0) or 0) + sum(proceeds.values()),
        1 - player: float(farms[1 - player].get("money", 0) or 0),
    }
    farm_rows = []
    for farm_index in (player, 1 - player):
        farm = farms[farm_index]
        other = farms[1 - farm_index]
        farm_rows.append(
            (
                _money_feature(farm.get("money", 0)),
                len(farm.get("unlocked_quadrants") or []) / 4.0,
                len(farm.get("hands") or []) / float(MAX_UNITS - 1),
                float(farm.get("hires_today", 0) or 0) / float(MAX_UNITS - 1),
                _signed_log_money(farm.get("money", 0)) - _signed_log_money(other.get("money", 0)),
                _money_feature(liquidation[farm_index]),
                _signed_log_money(liquidation[farm_index])
                - _signed_log_money(liquidation[1 - farm_index]),
            )
        )
    shops = (observation.get("town") or {}).get("unlocked_shops") or []
    town = np.concatenate(
        (
            clock_features(observation),
            np.asarray([shops.count(name) / 8.0 for name in SHOP_NAMES], dtype=np.float32),
            np.asarray(
                [(shops.index(name) + 1) / 8.0 if name in shops else 0.0 for name in SHOP_NAMES],
                dtype=np.float32,
            ),
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
    observation: dict,
    opponent_private: dict,
    outlook: MarketOutlook | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Critic-only columns from the opponent's private shed, hands, and seeds.

    ``observation`` is the seat's own; the opponent values its held stock
    against the same market and forecasts it against the same public outlook,
    ``outlook`` when already computed.
    """
    market = observation.get("market") or {}
    shed = opponent_private.get("shed") or {}
    seeds = opponent_private.get("seeds") or {}
    carried = _carried_items(opponent_private)
    held = _held_units(shed, carried)
    proceeds = _held_proceeds(held, market)
    forecast = _forecast_prices(outlook or market_outlook(observation), held, market)
    products = np.asarray(
        [
            (
                float(shed.get(item, 0) or 0) / SHED_CAPACITY,
                carried[item] / SHED_CAPACITY,
                proceeds[item] / HELD_VALUE_SCALE,
                forecast[item] / (2.0 * BASE_PRICE[item]),
            )
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
    critic_products: np.ndarray | None  # [len(PRODUCTS), len(PRODUCT_PRIVATE_FIELDS)] float16
    critic_animals: np.ndarray | None  # [len(ANIMALS), len(ANIMAL_PRIVATE_FIELDS)] float16
    critic_crops: np.ndarray | None  # [len(CROPS), len(CROP_PRIVATE_FIELDS)] float16
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
    outlook = market_outlook(observation)
    economy = tokenize_economy(observation, outlook)

    critic_products = critic_animals = critic_crops = None
    opponent_categorical = opponent_continuous = opponent_active = None
    if opponent_private is not None:
        critic_products, critic_animals, critic_crops = opponent_economy_columns(
            observation, opponent_private, outlook
        )
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
    "DEFAULT_OBSERVATION_SCHEMA_VERSION",
    "FARM_IDENTITIES",
    "FARM_TOKEN_FIELDS",
    "HELD_VALUE_SCALE",
    "N_TILE_CATEGORICAL",
    "N_TILE_CONTINUOUS",
    "N_UNIT_CATEGORICAL",
    "N_UNIT_CONTINUOUS",
    "OBSERVATION_SCHEMA_VERSION",
    "OUTLOOK_SOON_STEPS",
    "PRODUCT_PRIVATE_FIELDS",
    "PRODUCT_TOKEN_FIELDS",
    "QUADRANT_COUNT",
    "SUPPORTED_OBSERVATION_SCHEMA_VERSIONS",
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
    "MarketOutlook",
    "StructuredObservation",
    "TileTokens",
    "UnitTokens",
    "animal_token_fields",
    "clock_features",
    "crop_token_fields",
    "encode_structured_observation",
    "farm_token_fields",
    "market_outlook",
    "opponent_economy_columns",
    "product_private_fields",
    "product_token_fields",
    "tokenize_economy",
    "tokenize_farm_tiles",
    "tokenize_units",
    "town_token_fields",
]
