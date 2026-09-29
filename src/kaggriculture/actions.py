"""Masked factorized action space and exact Kaggriculture action compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from kaggriculture.constants import (
    ANIMAL_COST,
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    LAND_PRICES,
    MARKET_I0,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    ONGOING_CROPS,
    PRICE_FLOOR,
    PRIVATE_ITEMS,
    PRODUCTS,
    QUANTITY_BINS,
    SEED_COST,
    SHED_CAPACITY,
    fibonacci_hire_cost,
    market_price,
    shed_access_tiles,
)


class UnitAction(IntEnum):
    PASS = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    DROP = 5
    PICKUP_WHEAT_1 = 6
    PICKUP_WHEAT_2 = 7
    PICKUP_WHEAT_3 = 8
    PICKUP_WHEAT_4 = 9
    PICKUP_WHEAT_5 = 10
    PICKUP_WHEAT_6 = 11
    PICKUP_WHEAT_7 = 12
    PICKUP_WHEAT_8 = 13
    PICKUP_WHEAT_9 = 14
    PICKUP_WHEAT_10 = 15
    PICKUP_WHEAT_11 = 16
    PICKUP_WHEAT_12 = 17
    PICKUP_WHEAT_13 = 18
    PICKUP_WHEAT_14 = 19
    PICKUP_WHEAT_15 = 20
    PICKUP_WHEAT_16 = 21
    PICKUP_FERTILIZER_1 = 22
    PICKUP_FERTILIZER_2 = 23
    PICKUP_FERTILIZER_3 = 24
    PICKUP_FERTILIZER_4 = 25
    PICKUP_FERTILIZER_5 = 26
    PICKUP_FERTILIZER_6 = 27
    PICKUP_FERTILIZER_7 = 28
    PICKUP_FERTILIZER_8 = 29
    PICKUP_GOOSE_1 = 30
    PICKUP_GOOSE_2 = 31
    PICKUP_GOOSE_3 = 32
    PICKUP_GOOSE_4 = 33
    PICKUP_COW_1 = 34
    PICKUP_COW_2 = 35
    PICKUP_COW_3 = 36
    PICKUP_COW_4 = 37
    PICKUP_SHEEP_1 = 38
    PICKUP_SHEEP_2 = 39
    PICKUP_SHEEP_3 = 40
    PICKUP_SHEEP_4 = 41
    PLACE_GOOSE = 42
    PLACE_COW = 43
    PLACE_SHEEP = 44
    PLANT_WHEAT = 45
    PLANT_CARROT = 46
    PLANT_TOMATO = 47
    PLANT_STRAWBERRY = 48
    PLANT_MELON = 49
    WATER = 50
    HARVEST = 51
    FERTILIZE = 52
    DIG = 53
    BUILD_COOP = 54
    BUILD_PASTURE = 55
    FEED = 56
    COLLECT_FERTILIZER = 57
    CARE = 58
    PLACE_WHEAT = 59
    PLACE_CARROT = 60
    PLACE_TOMATO = 61
    PLACE_STRAWBERRY = 62
    PLACE_MELON = 63
    PLACE_EGG = 64
    PLACE_MILK = 65
    PLACE_WOOL = 66
    PLACE_FERTILIZER = 67

    # The original unsuffixed names remain readable aliases for the largest
    # transfer, while masks expose every smaller coordination-friendly choice.
    PICKUP_WHEAT = PICKUP_WHEAT_16
    PICKUP_FERTILIZER = PICKUP_FERTILIZER_8
    PICKUP_GOOSE = PICKUP_GOOSE_4
    PICKUP_COW = PICKUP_COW_4
    PICKUP_SHEEP = PICKUP_SHEEP_4


class MarketKind(IntEnum):
    STOP = 0
    HIRE = 1
    BUY_LAND = 2
    BUY_SEED_WHEAT = 3
    BUY_SEED_CARROT = 4
    BUY_SEED_TOMATO = 5
    BUY_SEED_STRAWBERRY = 6
    BUY_SEED_MELON = 7
    BUY_PRODUCT_WHEAT = 8
    BUY_PRODUCT_FERTILIZER = 9
    BUY_ANIMAL_GOOSE = 10
    BUY_ANIMAL_COW = 11
    BUY_ANIMAL_SHEEP = 12
    SELL_WHEAT = 13
    SELL_CARROT = 14
    SELL_TOMATO = 15
    SELL_STRAWBERRY = 16
    SELL_MELON = 17
    SELL_EGG = 18
    SELL_MILK = 19
    SELL_WOOL = 20
    SELL_FERTILIZER = 21


N_UNIT_ACTIONS = len(UnitAction)
N_MARKET_KINDS = len(MarketKind)
N_QUANTITIES = len(QUANTITY_BINS)

_PICKUP_SPEC = {
    **{UnitAction[f"PICKUP_WHEAT_{quantity}"]: ("WHEAT", quantity) for quantity in range(1, 17)},
    **{
        UnitAction[f"PICKUP_FERTILIZER_{quantity}"]: ("FERTILIZER", quantity)
        for quantity in range(1, 9)
    },
    **{
        UnitAction[f"PICKUP_{animal}_{quantity}"]: (animal, quantity)
        for animal in ("GOOSE", "COW", "SHEEP")
        for quantity in (1, 2, 3, 4)
    },
}
_PLACE_ANIMAL = {
    UnitAction.PLACE_GOOSE: "GOOSE",
    UnitAction.PLACE_COW: "COW",
    UnitAction.PLACE_SHEEP: "SHEEP",
}
_PLACE_PRODUCT = {UnitAction[f"PLACE_{item}"]: item for item in PRODUCTS}

_PLANT_CROP = {
    UnitAction.PLANT_WHEAT: "WHEAT",
    UnitAction.PLANT_CARROT: "CARROT",
    UnitAction.PLANT_TOMATO: "TOMATO",
    UnitAction.PLANT_STRAWBERRY: "STRAWBERRY",
    UnitAction.PLANT_MELON: "MELON",
}
_MOVE_COMMAND = {
    UnitAction.NORTH: "NORTH",
    UnitAction.SOUTH: "SOUTH",
    UnitAction.EAST: "EAST",
    UnitAction.WEST: "WEST",
}
_MOVE_DELTA = {
    UnitAction.NORTH: (0, -1),
    UnitAction.SOUTH: (0, 1),
    UnitAction.EAST: (1, 0),
    UnitAction.WEST: (-1, 0),
}

_BUY_SEED = {
    MarketKind.BUY_SEED_WHEAT: "WHEAT",
    MarketKind.BUY_SEED_CARROT: "CARROT",
    MarketKind.BUY_SEED_TOMATO: "TOMATO",
    MarketKind.BUY_SEED_STRAWBERRY: "STRAWBERRY",
    MarketKind.BUY_SEED_MELON: "MELON",
}
_BUY_PRODUCT = {
    MarketKind.BUY_PRODUCT_WHEAT: "WHEAT",
    MarketKind.BUY_PRODUCT_FERTILIZER: "FERTILIZER",
}
_BUY_ANIMAL = {
    MarketKind.BUY_ANIMAL_GOOSE: "GOOSE",
    MarketKind.BUY_ANIMAL_COW: "COW",
    MarketKind.BUY_ANIMAL_SHEEP: "SHEEP",
}
_SELL_PRODUCT = {
    MarketKind(MarketKind.SELL_WHEAT + index): product for index, product in enumerate(PRODUCTS)
}
QUANTIFIED_MARKET_KINDS = frozenset((*_BUY_SEED, *_BUY_PRODUCT, *_BUY_ANIMAL, *_SELL_PRODUCT))


@dataclass
class MarketLedger:
    money: float
    shed: dict[str, int]
    hires: int
    extra_land: int
    inventory: dict[str, int]
    seeds: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_observation(
        cls,
        observation: dict[str, Any],
        *,
        shed: dict[str, int] | None = None,
        seeds: dict[str, int] | None = None,
    ) -> MarketLedger:
        """Initialize market state, retaining an optional post-unit shed ledger."""
        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms") or []
        farm = farms[player] if player < len(farms) else {}
        if shed is None:
            snapshot_shed = (observation.get("private") or {}).get("shed") or {}
            shed = {item: int(value or 0) for item, value in snapshot_shed.items()}
        inventory = (observation.get("market") or {}).get("inventory") or {}
        return cls(
            money=float(farm.get("money", 0) or 0),
            shed=shed,
            hires=int(farm.get("hires_today", 0) or 0),
            extra_land=max(0, len(farm.get("unlocked_quadrants") or []) - 1),
            inventory={item: int(inventory.get(item, MARKET_I0)) for item in PRODUCTS},
            seeds=dict((observation.get("private") or {}).get("seeds") or {})
            if seeds is None
            else dict(seeds),
        )


def _unit_position(farm: dict[str, Any], unit_index: int) -> tuple[int, int] | None:
    if unit_index == 0:
        raw = farm.get("farmer")
    else:
        hands = farm.get("hands") or []
        raw = hands[unit_index - 1] if unit_index - 1 < len(hands) else None
    return None if raw is None else (int(raw[0]), int(raw[1]))


def _unit_inventory(private: dict[str, Any], unit_index: int) -> dict[str, int]:
    inventories = private.get("inventories") or []
    return inventories[unit_index] if unit_index < len(inventories) else {}


def copy_tile_grid(tiles: list[list[Any]]) -> list[list[Any]]:
    """Copy the scalar/dict tile grid without general-purpose deepcopy overhead."""
    return [[dict(tile) if isinstance(tile, dict) else tile for tile in row] for row in tiles]


def unit_action_mask(
    observation: dict[str, Any],
    unit_index: int,
    remaining_seeds: dict[str, int] | None = None,
    remaining_shed: dict[str, int] | None = None,
    tiles_override: list[list[Any]] | None = None,
) -> np.ndarray:
    """Return actions that can have an effect for one currently active unit."""
    mask = np.zeros(N_UNIT_ACTIONS, dtype=np.bool_)
    mask[UnitAction.PASS] = True
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if player >= len(farms):
        return mask
    farm = farms[player]
    position = _unit_position(farm, unit_index)
    if position is None:
        return mask
    private = observation.get("private") or {}
    inventory = _unit_inventory(private, unit_index)
    x, y = position
    tiles = tiles_override if tiles_override is not None else farm.get("tiles") or []
    board_size = len(tiles) or BOARD_SIZE

    for action, (dx, dy) in _MOVE_DELTA.items():
        mask[action] = 0 <= x + dx < board_size and 0 <= y + dy < board_size

    at_shed = position in shed_access_tiles(board_size)
    if at_shed:
        shed = remaining_shed if remaining_shed is not None else private.get("shed") or {}
        shed_room = SHED_CAPACITY - sum(int(value or 0) for value in shed.values())
        mask[UnitAction.DROP] = any(int(value or 0) > 0 for value in inventory.values())

        # Our policy only asks for pickups it can fill completely. The engine
        # would clamp a short one instead of refusing it (kaggriculture.py:357),
        # so this is a deliberate narrowing of our own action space and not a
        # model of the rules -- `Game::unit_action_valid` carries the engine's
        # version for opponents that submit a raw dict. Widening it is a
        # capability change that invalidates every trained artifact, because a
        # masked action never receives gradient: admitting partial pickups took
        # the cloned policy from 136,425 median dollars against `starter` to 10.
        for action, (item, quantity) in _PICKUP_SPEC.items():
            mask[action] = int(shed.get(item, 0) or 0) >= quantity
        for action, animal in _PLACE_ANIMAL.items():
            mask[action] = shed_room > 0 and int(inventory.get(animal, 0) or 0) > 0
        for action, item in _PLACE_PRODUCT.items():
            mask[action] = shed_room > 0 and int(inventory.get(item, 0) or 0) > 0

    tile = tiles[y][x]
    if tile == "LOCKED":
        return mask

    if tile is None:
        seeds = remaining_seeds if remaining_seeds is not None else private.get("seeds") or {}
        for action, crop in _PLANT_CROP.items():
            mask[action] = int(seeds.get(crop, 0) or 0) > 0
        mask[UnitAction.BUILD_COOP] = True
        mask[UnitAction.BUILD_PASTURE] = True
        return mask

    if not isinstance(tile, dict):
        return mask
    kind = tile.get("kind")
    if kind == "WEED":
        mask[UnitAction.DIG] = True
    elif kind == "PLANT":
        mask[UnitAction.WATER] = not bool(tile.get("watered_today", False))
        crop = tile.get("crop")
        day = int(observation.get("day", 0) or 0)
        age = day - int(tile.get("planted_day", day) or 0)
        mask[UnitAction.HARVEST] = (
            crop in CROP_FIRST_YIELD_DAY
            and age >= CROP_FIRST_YIELD_DAY[crop]
            and int(tile.get("yield_units", 0) or 0) > 0
        )
        mask[UnitAction.FERTILIZE] = (
            int(inventory.get("FERTILIZER", 0) or 0) > 0
            and int(tile.get("fertilized_until_day", -1) or -1) < day + 2
        )
        mask[UnitAction.DIG] = True
    elif kind in {"COOP", "PASTURE"}:
        if "animal" not in tile:
            mask[UnitAction.DIG] = True
            for action, candidate in _PLACE_ANIMAL.items():
                mask[action] |= (
                    ANIMAL_STRUCTURE[candidate] == kind
                    and int(inventory.get(candidate, 0) or 0) > 0
                )
        else:
            mask[UnitAction.HARVEST] = int(tile.get("yield_units", 0) or 0) > 0
            mask[UnitAction.FEED] = (
                not bool(tile.get("fed_today", False)) and int(inventory.get("WHEAT", 0) or 0) > 0
            )
            mask[UnitAction.CARE] = not bool(tile.get("cared_today", False))
            mask[UnitAction.COLLECT_FERTILIZER] = bool(tile.get("fertilizer_available", False))
    return mask


def all_unit_action_masks(observation: dict[str, Any]) -> np.ndarray:
    """Build independent snapshot masks without reserving resources between units."""
    masks = np.zeros((MAX_UNITS, N_UNIT_ACTIONS), dtype=np.bool_)
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    n_units = 0
    if player < len(farms):
        n_units = min(MAX_UNITS, 1 + len(farms[player].get("hands") or []))
    for index in range(n_units):
        masks[index] = unit_action_mask(observation, index)
    masks[n_units:, UnitAction.PASS] = True
    return masks


def market_kind_mask(observation: dict[str, Any]) -> np.ndarray:
    """Return exact order-type legality for the observation snapshot, without a prefix."""
    return _ledger_kind_mask(observation, MarketLedger.from_observation(observation))


def quantity_mask(observation: dict[str, Any], kind_value: int) -> np.ndarray:
    """Return snapshot quantity bins using cumulative per-unit market quotes."""
    return _ledger_quantity_mask(
        observation, MarketKind(kind_value), MarketLedger.from_observation(observation)
    )


def _ledger_kind_mask(
    observation: dict[str, Any],
    ledger: MarketLedger,
) -> np.ndarray:
    mask = np.zeros(N_MARKET_KINDS, dtype=np.bool_)
    mask[MarketKind.STOP] = True
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if player >= len(farms):
        return mask
    farm = farms[player]
    mask[MarketKind.HIRE] = len(farm.get("hands") or []) + max(
        0, ledger.hires - int(farm.get("hires_today", 0) or 0)
    ) < MAX_UNITS - 1 and ledger.money >= fibonacci_hire_cost(ledger.hires)
    mask[MarketKind.BUY_LAND] = (
        ledger.extra_land < len(LAND_PRICES) and ledger.money >= LAND_PRICES[ledger.extra_land]
    )
    room = SHED_CAPACITY - sum(ledger.shed.values())
    for raw_kind, crop in _BUY_SEED.items():
        mask[raw_kind] = ledger.money >= SEED_COST[crop]
    market_params = (observation.get("market") or {}).get("params")
    for kind, item in _BUY_PRODUCT.items():
        quote = market_price(item, ledger.inventory[item] - 1, market_params)
        mask[kind] = room > 0 and ledger.money >= quote
    for kind, animal in _BUY_ANIMAL.items():
        mask[kind] = room > 0 and ledger.money >= ANIMAL_COST[animal]
    for kind, product in _SELL_PRODUCT.items():
        mask[kind] = ledger.shed.get(product, 0) > 0
    return mask


def _ledger_quantity_mask(
    observation: dict[str, Any],
    kind: MarketKind,
    ledger: MarketLedger,
) -> np.ndarray:
    if kind not in QUANTIFIED_MARKET_KINDS:
        mask = np.zeros(N_QUANTITIES, dtype=np.bool_)
        mask[0] = True
        return mask
    room = SHED_CAPACITY - sum(ledger.shed.values())
    if kind in _SELL_PRODUCT:
        maximum = ledger.shed.get(_SELL_PRODUCT[kind], 0)
    elif kind in _BUY_SEED:
        maximum = int(ledger.money // SEED_COST[_BUY_SEED[kind]])
    elif kind in _BUY_ANIMAL:
        maximum = min(room, int(ledger.money // ANIMAL_COST[_BUY_ANIMAL[kind]]))
    else:
        item = _BUY_PRODUCT[kind]
        market_params = (observation.get("market") or {}).get("params")
        balance = ledger.money
        inventory = ledger.inventory[item]
        maximum = 0
        while maximum < room:
            quote = market_price(item, inventory - 1, market_params)
            if balance < quote:
                break
            balance -= quote
            inventory -= 1
            maximum += 1
    return np.asarray([quantity <= maximum for quantity in QUANTITY_BINS], dtype=np.bool_)


def _apply_ledger_order(
    observation: dict[str, Any],
    kind: MarketKind,
    quantity: int,
    ledger: MarketLedger,
) -> None:
    market_params = (observation.get("market") or {}).get("params")
    if kind == MarketKind.HIRE:
        ledger.money -= fibonacci_hire_cost(ledger.hires)
        ledger.hires += 1
    elif kind == MarketKind.BUY_LAND:
        ledger.money -= LAND_PRICES[ledger.extra_land]
        ledger.extra_land += 1
    elif kind in _BUY_SEED:
        crop = _BUY_SEED[kind]
        ledger.money -= SEED_COST[crop] * quantity
        ledger.seeds[crop] = ledger.seeds.get(crop, 0) + quantity
    elif kind in _BUY_PRODUCT:
        item = _BUY_PRODUCT[kind]
        for _ in range(quantity):
            quote = market_price(item, ledger.inventory[item] - 1, market_params)
            if ledger.money < quote or sum(ledger.shed.values()) >= SHED_CAPACITY:
                break
            ledger.money -= quote
            ledger.shed[item] = ledger.shed.get(item, 0) + 1
            ledger.inventory[item] -= 1
    elif kind in _BUY_ANIMAL:
        animal = _BUY_ANIMAL[kind]
        ledger.money -= ANIMAL_COST[animal] * quantity
        ledger.shed[animal] = ledger.shed.get(animal, 0) + quantity
    elif kind in _SELL_PRODUCT:
        item = _SELL_PRODUCT[kind]
        for _ in range(quantity):
            if ledger.shed.get(item, 0) <= 0:
                break
            quote = market_price(item, ledger.inventory[item], market_params)
            ledger.shed[item] -= 1
            ledger.money += quote
            if quote > PRICE_FLOOR:
                ledger.inventory[item] += 1


def unit_action_command(
    observation: dict[str, Any],
    unit_index: int,
    action_value: int,
    remaining_shed: dict[str, int] | None = None,
) -> list[Any]:
    action = UnitAction(action_value)
    if action == UnitAction.PASS:
        return ["PASS"]
    if action in _MOVE_COMMAND:
        return [_MOVE_COMMAND[action]]
    if action == UnitAction.DROP:
        return ["DROP"]
    if action in _PICKUP_SPEC:
        item, quantity = _PICKUP_SPEC[action]
        shed = (
            remaining_shed
            if remaining_shed is not None
            else (observation.get("private") or {}).get("shed") or {}
        )
        return ["PICKUP", item, min(quantity, int(shed.get(item, 0) or 0))]
    if action in _PLACE_ANIMAL:
        return ["PLACE", _PLACE_ANIMAL[action]]
    if action in _PLACE_PRODUCT:
        item = _PLACE_PRODUCT[action]
        inventory = _unit_inventory(observation.get("private") or {}, unit_index)
        held = max(0, int(inventory.get(item, 0) or 0))
        shed = (
            remaining_shed
            if remaining_shed is not None
            else (observation.get("private") or {}).get("shed") or {}
        )
        room = max(0, SHED_CAPACITY - sum(int(value or 0) for value in shed.values()))
        return ["PLACE", item, min(held, room)]

    if action in _PLANT_CROP:
        return ["PLANT", _PLANT_CROP[action]]
    return [action.name]


def apply_unit_shed_effect(
    observation: dict[str, Any],
    unit_index: int,
    action_value: int,
    shed: dict[str, int],
    tiles_override: list[list[Any]] | None = None,
) -> None:
    """Update a shed ledger using the engine's sequential unit-action semantics."""
    action = UnitAction(action_value)
    if action in _PICKUP_SPEC:
        item, quantity = _PICKUP_SPEC[action]
        available = int(shed.get(item, 0) or 0)
        shed[item] = available - min(quantity, available)
        return
    if action in _PLACE_ANIMAL:
        player = int(observation.get("player", 0) or 0)
        farm = (observation.get("farms") or [])[player]
        position = _unit_position(farm, unit_index)
        if position is None:
            return
        x, y = position
        tiles = tiles_override if tiles_override is not None else farm.get("tiles") or []
        animal = _PLACE_ANIMAL[action]
        tile = tiles[y][x]
        installs_animal = (
            isinstance(tile, dict)
            and tile.get("kind") == ANIMAL_STRUCTURE[animal]
            and "animal" not in tile
        )
        if installs_animal or position not in shed_access_tiles(len(tiles) or BOARD_SIZE):
            return
        inventory = _unit_inventory(observation.get("private") or {}, unit_index)
        room = max(0, SHED_CAPACITY - sum(int(value or 0) for value in shed.values()))
        if room > 0 and int(inventory.get(animal, 0) or 0) > 0:
            shed[animal] = int(shed.get(animal, 0) or 0) + 1
        return
    if action in _PLACE_PRODUCT:
        player = int(observation.get("player", 0) or 0)
        farm = (observation.get("farms") or [])[player]
        position = _unit_position(farm, unit_index)
        if position is None or position not in shed_access_tiles(
            len(tiles_override if tiles_override is not None else farm.get("tiles") or [])
            or BOARD_SIZE
        ):
            return
        item = _PLACE_PRODUCT[action]
        inventory = _unit_inventory(observation.get("private") or {}, unit_index)
        room = max(0, SHED_CAPACITY - sum(int(value or 0) for value in shed.values()))
        quantity = min(room, max(0, int(inventory.get(item, 0) or 0)))
        if quantity > 0:
            shed[item] = int(shed.get(item, 0) or 0) + quantity
        return
    if action != UnitAction.DROP:
        return
    inventory = _unit_inventory(observation.get("private") or {}, unit_index)
    for item, raw_quantity in inventory.items():
        room = max(0, SHED_CAPACITY - sum(shed.values()))
        quantity = min(room, max(0, int(raw_quantity or 0)))
        if quantity > 0:
            shed[item] = shed.get(item, 0) + quantity


def apply_unit_tile_effect(
    observation: dict[str, Any],
    unit_index: int,
    action_value: int,
    tiles: list[list[Any]],
) -> None:
    """Update a tile ledger for interactions by later units in the same turn."""
    action = UnitAction(action_value)
    player = int(observation.get("player", 0) or 0)
    farm = (observation.get("farms") or [])[player]
    position = _unit_position(farm, unit_index)
    if position is None:
        return
    x, y = position
    tile = tiles[y][x]
    day = int(observation.get("day", 0) or 0)
    if action in _PLANT_CROP and tile is None:
        crop = _PLANT_CROP[action]
        tiles[y][x] = {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": day,
            "watered_today": False,
            "consecutive_unwatered": 1,
            "yield_units": 0 if crop in ONGOING_CROPS else 1,
            "fertilized_until_day": -1,
        }
        return
    if action == UnitAction.BUILD_COOP and tile is None:
        tiles[y][x] = {"kind": "COOP"}
        return
    if action == UnitAction.BUILD_PASTURE and tile is None:
        tiles[y][x] = {"kind": "PASTURE"}
        return
    if action == UnitAction.DIG and tile is not None:
        if not (isinstance(tile, dict) and "animal" in tile):
            tiles[y][x] = None
        return
    if action in _PLACE_ANIMAL and isinstance(tile, dict):
        animal = _PLACE_ANIMAL[action]
        if tile.get("kind") == ANIMAL_STRUCTURE[animal] and "animal" not in tile:
            tiles[y][x] = {
                "kind": ANIMAL_STRUCTURE[animal],
                "animal": animal,
                "placed_day": day,
                "yield_units": 0,
                "consecutive_unfed": 0,
                "fed_today": False,
                "cared_today": False,
                "fertilizer_available": False,
                "pending_care_bonus": 0,
            }
        return
    if not isinstance(tile, dict):
        return
    if action == UnitAction.WATER and tile.get("kind") == "PLANT":
        if tile.get("watered_today", False):
            return
        tile["watered_today"] = True
        crop = tile["crop"]
        if crop not in ONGOING_CROPS:
            age = day - int(tile.get("planted_day", day))
            window_start = (CROP_MAX_YIELD_DAY[crop] + 1) // 2
            if window_start <= age <= CROP_MAX_YIELD_DAY[crop]:
                bonus = 2 if int(tile.get("fertilized_until_day", -1)) >= day else 1
                tile["yield_units"] = min(
                    CROP_MAX_YIELD[crop], int(tile.get("yield_units", 0)) + bonus
                )
    elif action == UnitAction.HARVEST and int(tile.get("yield_units", 0) or 0) > 0:
        if tile.get("kind") == "PLANT" and tile.get("crop") not in ONGOING_CROPS:
            tiles[y][x] = None
        else:
            tile["yield_units"] = 0
    elif action == UnitAction.FERTILIZE and tile.get("kind") == "PLANT":
        tile["fertilized_until_day"] = max(int(tile.get("fertilized_until_day", -1)), day + 2)
    elif action == UnitAction.FEED and "animal" in tile:
        tile["fed_today"] = True
    elif action == UnitAction.CARE and "animal" in tile:
        tile["cared_today"] = True
    elif action == UnitAction.COLLECT_FERTILIZER and "animal" in tile:
        tile["fertilizer_available"] = False


def market_order(kind_value: int, quantity_value: int) -> list[Any] | None:
    kind = MarketKind(kind_value)
    if kind == MarketKind.STOP:
        return None
    if kind == MarketKind.HIRE:
        return ["HIRE"]
    if kind == MarketKind.BUY_LAND:
        return ["BUY_LAND"]
    quantity = QUANTITY_BINS[quantity_value]
    if kind in _BUY_SEED:
        return ["BUY_SEED", _BUY_SEED[kind], quantity]
    if kind in _BUY_PRODUCT:
        return ["BUY_PRODUCT", _BUY_PRODUCT[kind], quantity]
    if kind in _BUY_ANIMAL:
        return ["BUY_ANIMAL", _BUY_ANIMAL[kind], quantity]
    return ["SELL", _SELL_PRODUCT[kind], quantity]


def compile_action(
    observation: dict[str, Any],
    unit_actions: np.ndarray,
    market_kinds: np.ndarray,
    market_quantities: np.ndarray,
) -> dict[str, Any]:
    """Compile sampled factors to one engine action, preserving atomic seed validity."""
    player = int(observation.get("player", 0) or 0)
    farm = (observation.get("farms") or [])[player]
    n_units = min(MAX_UNITS, 1 + len(farm.get("hands") or []))
    remaining_seeds = dict((observation.get("private") or {}).get("seeds") or {})
    remaining_shed = dict((observation.get("private") or {}).get("shed") or {})
    tiles = copy_tile_grid(farm.get("tiles") or [])
    commands: list[list[Any]] = []
    for index in range(n_units):
        selected = UnitAction(int(unit_actions[index]))
        mask = unit_action_mask(observation, index, remaining_seeds, remaining_shed, tiles)
        if not mask[selected]:
            selected = UnitAction.PASS
        crop = _PLANT_CROP.get(selected)
        if crop is not None:
            remaining_seeds[crop] = int(remaining_seeds.get(crop, 0) or 0) - 1
        commands.append(unit_action_command(observation, index, selected, remaining_shed))
        apply_unit_shed_effect(observation, index, selected, remaining_shed, tiles)
        apply_unit_tile_effect(observation, index, selected, tiles)

    orders = []
    for kind_value, quantity_value in zip(
        market_kinds[:MAX_MARKET_ORDERS],
        market_quantities[:MAX_MARKET_ORDERS],
        strict=True,
    ):
        order = market_order(int(kind_value), int(quantity_value))
        if order is None:
            break
        orders.append(order)
    return {"farmer": commands[0], "hands": commands[1:], "market": orders}


def active_unit_count(observation: dict[str, Any]) -> int:
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if player >= len(farms):
        return 0
    return min(MAX_UNITS, 1 + len(farms[player].get("hands") or []))


def private_item_vector(mapping: dict[str, Any]) -> list[float]:
    """Stable item ordering used by callers that need compact inventory state."""
    return [float(mapping.get(item, 0) or 0) for item in PRIVATE_ITEMS]
