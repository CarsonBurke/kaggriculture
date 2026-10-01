"""Game constants shared by training, inference, and tests."""

from __future__ import annotations

import math

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
PRIVATE_ITEMS = PRODUCTS + ANIMALS
SHOP_NAMES = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)

SEED_COST = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}
CROP_FIRST_YIELD_DAY = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}
CROP_MAX_YIELD_DAY = {
    "WHEAT": 4,
    "CARROT": 3,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 12,
}
CROP_MAX_YIELD = {
    "WHEAT": 6,
    "CARROT": 4,
    "TOMATO": 4,
    "STRAWBERRY": 4,
    "MELON": 6,
}
# Days between an ongoing crop's productions (the engine's CROPS "interval");
# zero for the crops that yield once.
CROP_YIELD_INTERVAL = {
    "WHEAT": 0,
    "CARROT": 0,
    "TOMATO": 1,
    "STRAWBERRY": 2,
    "MELON": 0,
}
ONGOING_CROPS = frozenset(("TOMATO", "STRAWBERRY"))
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_FIRST_YIELD_DAY = {"GOOSE": 4, "COW": 8, "SHEEP": 6}
ANIMAL_YIELD_INTERVAL = {"GOOSE": 1, "COW": 2, "SHEEP": 3}
ANIMAL_MAX_HELD = {"GOOSE": 4, "COW": 6, "SHEEP": 6}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
BASE_PRICE = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}
MARKET_I0 = 10_000
PRICE_FLOOR = 1
MARKET_PARAMS = {
    "WHEAT": {
        "base": 25,
        "I0": MARKET_I0,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "I0": MARKET_I0,
        "T": 450,
        "below_func": "hinge",
        "below_target": 1.00,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "I0": MARKET_I0,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "I0": MARKET_I0,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "I0": MARKET_I0,
        "T": 332,
        "below_func": "hinge",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "I0": MARKET_I0,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "I0": MARKET_I0,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}
LAND_PRICES = (1000, 2000, 4000)

# The town (the engine's SHOPS, `_town_consume` and `_end_of_day`, at the
# default configuration). Every `TOWN_SHOP_SELL_INTERVAL` steps each unlocked
# shop instance takes one unit of each product it sells -- two when it sells
# only one -- from the market; every `TOWN_CENTER_SELL_INTERVAL` steps the town
# center takes one of every product but fertilizer. On each day divisible by
# `TOWN_SHOP_UNLOCK_INTERVAL`, until `MAX_SHOP_INSTANCES` are open, one more
# instance opens, drawn uniformly (with replacement) from the shops.
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "YARN_STORE": ("WOOL",),
}
TOWN_CENTER_PRODUCTS = tuple(item for item in PRODUCTS if item != "FERTILIZER")
TOWN_SHOP_SELL_INTERVAL = 4
TOWN_CENTER_SELL_INTERVAL = 24
TOWN_SHOP_UNLOCK_INTERVAL = 3
MAX_SHOP_INSTANCES = 8

STARTING_MONEY = 3_000
DEFAULT_REWARD_GAMMA = 1.0
DEFAULT_REWARD_MODE = "terminal-outcome"

BOARD_SIZE = 10
TURNS_PER_DAY = 24
EPISODE_STEPS = 720
SHED_CAPACITY = 100
MAX_UNITS = 16
MAX_MARKET_ORDERS = 10
MAX_MARKET_QUANTITY = 100
QUANTITY_BINS = tuple(range(1, MAX_MARKET_QUANTITY + 1))

FARMER_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}


HINGE_GAIN = 8.0


def _market_shape(function: str, value: float, scale: float | None = None) -> float:
    value = max(0.0, value)
    if function == "linear":
        return value
    if function == "sq":
        return value * value
    if function == "sqrt":
        return math.sqrt(value)
    if function == "log":
        return math.log1p(value)
    if function == "log10":
        return math.log10(1.0 + value)
    if function == "hinge":
        if scale is None or scale <= 0.0:
            return value
        unit = value / scale
        return unit + HINGE_GAIN * max(0.0, unit - 1.0) ** 2
    return value


def market_price(
    item: str,
    inventory: int,
    params: dict[str, dict[str, int | float | str]] | None = None,
) -> int:
    """Reproduce the engine's inventory-dependent per-unit market quote."""
    defaults = MARKET_PARAMS[item]
    configured = (params or {}).get(item, {})
    pricing = defaults | configured
    base = float(pricing["base"])
    initial_inventory = int(pricing["I0"])
    scale = float(pricing["T"])
    if inventory < initial_inventory:
        function = str(pricing["below_func"])
        target = float(pricing["below_target"])
        amplitude = target * base / _market_shape(function, scale, scale)
        price = base + amplitude * _market_shape(function, initial_inventory - inventory, scale)
    else:
        function = str(pricing["above_func"])
        target = float(pricing["above_target"])
        amplitude = target * base / _market_shape(function, scale, scale)
        price = base - amplitude * _market_shape(function, inventory - initial_inventory, scale)
    return max(PRICE_FLOOR, round(price))


def sale_proceeds(
    item: str,
    units: int,
    inventory: int,
    params: dict[str, dict[str, int | float | str]] | None = None,
) -> int:
    """Exact coins from selling ``units`` of ``item`` one at a time into the market.

    Mirrors the engine's sell arithmetic: each unit quotes at the current
    market inventory, and a sale restocks the market only while the quote sits
    above the price floor -- so once a quote reaches the floor, every remaining
    unit sells at it.
    """
    proceeds = 0
    for sold in range(units):
        price = market_price(item, inventory, params)
        if price <= PRICE_FLOOR:
            return proceeds + (units - sold) * price
        proceeds += price
        inventory += 1
    return proceeds


def restocked_inventory(
    item: str,
    units: int,
    inventory: int,
    params: dict[str, dict[str, int | float | str]] | None = None,
) -> int:
    """The market inventory after selling ``units`` of ``item`` one at a time.

    The engine's sell arithmetic, as in `sale_proceeds`: a sale restocks the
    market only while its quote sits above the price floor. Quotes never rise
    with inventory, so the first floored sale is found by bisection.
    """
    restocking, floored = 0, units
    while restocking < floored:
        middle = (restocking + floored) // 2
        if market_price(item, inventory + middle, params) <= PRICE_FLOOR:
            floored = middle
        else:
            restocking = middle + 1
    return inventory + floored


def fibonacci_hire_cost(hires_today: int) -> int:
    """Return the engine's 1, 1, 2, 3, 5, ... daily hire cost."""
    a, b = 1, 1
    for _ in range(max(0, hires_today)):
        a, b = b, a + b
    return a


def shed_access_tiles(board_size: int = BOARD_SIZE) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
