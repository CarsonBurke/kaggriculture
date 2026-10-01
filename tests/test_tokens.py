from __future__ import annotations

import itertools
import math
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official

from kaggriculture.actions import UnitAction, apply_unit_shed_effect
from kaggriculture.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
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
    sale_proceeds,
    shed_access_tiles,
)
from kaggriculture.encoding import encode_observation, liquidation_value
from kaggriculture.tokens import (
    ANIMAL_TOKEN_FIELDS,
    CROP_TOKEN_FIELDS,
    FARM_IDENTITIES,
    FARM_TOKEN_FIELDS,
    HELD_VALUE_SCALE,
    N_TILE_CATEGORICAL,
    N_TILE_CONTINUOUS,
    OBSERVATION_SCHEMA_VERSION,
    OUTLOOK_SOON_STEPS,
    PRODUCT_PRIVATE_FIELDS,
    PRODUCT_TOKEN_FIELDS,
    QUADRANT_COUNT,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TILE_CONTINUOUS_FIELDS,
    TILE_COUNT,
    TILE_KIND_INDEX,
    TILE_KINDS,
    TILE_OCCUPANT_INDEX,
    TILE_OCCUPANTS,
    TOWN_TOKEN_FIELDS,
    UNIT_TILE_GATHERS,
    animal_token_fields,
    clock_features,
    crop_token_fields,
    encode_structured_observation,
    farm_token_fields,
    market_outlook,
    product_private_fields,
    product_token_fields,
    tokenize_economy,
    tokenize_farm_tiles,
    tokenize_units,
    town_token_fields,
)

_FIELD = {name: index for index, name in enumerate(TILE_CONTINUOUS_FIELDS)}


def _field(tokens, x: int, y: int, name: str) -> float:
    return float(tokens.continuous[y * BOARD_SIZE + x, _FIELD[name]])


def _blank_farm(tile: object, x: int = 3, y: int = 3) -> dict:
    tiles = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    tiles[y][x] = tile
    return {"tiles": tiles}


def test_tile_token_shapes_dtypes_and_static_geometry() -> None:
    tokens = tokenize_farm_tiles({"tiles": []}, day=0, step=0, opponent=False)

    assert tokens.categorical.shape == (TILE_COUNT, N_TILE_CATEGORICAL)
    assert tokens.continuous.shape == (TILE_COUNT, N_TILE_CONTINUOUS)
    assert tokens.categorical.dtype == np.int64
    assert tokens.continuous.dtype == np.float32
    # An absent grid tokenizes as fully locked.
    assert (tokens.categorical[:, 0] == TILE_KIND_INDEX["LOCKED"]).all()

    for x, y in shed_access_tiles(BOARD_SIZE):
        assert _field(tokens, x, y, "shed_access") == 1.0
        assert _field(tokens, x, y, "shed_distance") == 0.0
    # Corners are the farthest tiles from shed access and sit on the edge.
    for x, y in ((0, 0), (9, 0), (0, 9), (9, 9)):
        assert _field(tokens, x, y, "edge") == 1.0
        assert _field(tokens, x, y, "corner") == 1.0
        assert _field(tokens, x, y, "shed_distance") == 1.0
    assert _field(tokens, 5, 0, "edge") == 1.0
    assert _field(tokens, 5, 0, "corner") == 0.0
    # Quadrants split the board at its midlines: token order is row-major.
    quadrants = tokens.categorical[:, 5].reshape(BOARD_SIZE, BOARD_SIZE)
    assert quadrants[0, 0] == 0 and quadrants[0, 9] == 1
    assert quadrants[9, 0] == 2 and quadrants[9, 9] == 3


def test_plant_tile_features_follow_engine_mechanics() -> None:
    farm = _blank_farm(
        {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 2,
            "yield_units": 3,
            "watered_today": True,
            "consecutive_unwatered": 1,
            "fertilized_until_day": 5,
            "max_lifespan_step": 130,
        }
    )

    tokens = tokenize_farm_tiles(farm, day=4, step=126, opponent=False)

    index = 3 * BOARD_SIZE + 3
    assert tokens.categorical[index, 0] == TILE_KIND_INDEX["PLANT"]
    assert tokens.categorical[index, 1] == TILE_OCCUPANT_INDEX["WHEAT"]
    assert _field(tokens, 3, 3, "yield_fraction") == 3 / 6  # CROP_MAX_YIELD WHEAT
    assert _field(tokens, 3, 3, "maturity_fraction") == 1.0  # age 2 >= first yield day 2
    assert _field(tokens, 3, 3, "watered_today") == 1.0
    assert _field(tokens, 3, 3, "fertilizer_remaining") == pytest.approx((5 - 4 + 1) / 3)
    assert _field(tokens, 3, 3, "decay_pressure") == 1 / 2  # dies at 2 consecutive
    assert _field(tokens, 3, 3, "lifespan_remaining") == pytest.approx((130 - 126) / 96)
    assert _field(tokens, 3, 3, "lifespan_expired") == 0.0
    assert _field(tokens, 3, 3, "harvest_ready") == 1.0


def test_expired_plant_lifespan_decay_parity() -> None:
    tile = {
        "kind": "PLANT",
        "crop": "CARROT",
        "planted_day": 0,
        "yield_units": 2,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "fertilized_until_day": -1,
        "max_lifespan_step": 100,
    }

    # Engine removes one yield unit at max_lifespan_step and every 2 steps after.
    on_tick = tokenize_farm_tiles(_blank_farm(tile), day=4, step=102, opponent=False)
    off_tick = tokenize_farm_tiles(_blank_farm(tile), day=4, step=103, opponent=False)

    assert _field(on_tick, 3, 3, "lifespan_expired") == 1.0
    assert _field(on_tick, 3, 3, "lifespan_decay_tick") == 1.0
    assert _field(on_tick, 3, 3, "lifespan_remaining") == 0.0
    assert _field(off_tick, 3, 3, "lifespan_expired") == 1.0
    assert _field(off_tick, 3, 3, "lifespan_decay_tick") == 0.0


def test_animal_tile_features_follow_engine_mechanics() -> None:
    farm = _blank_farm(
        {
            "kind": "COOP",
            "animal": "GOOSE",
            "placed_day": 6,
            "yield_units": 2,
            "fed_today": True,
            "cared_today": False,
            "fertilizer_available": True,
            "pending_care_bonus": 2,
            "consecutive_unfed": 0,
        },
        x=7,
        y=2,
    )

    tokens = tokenize_farm_tiles(farm, day=8, step=200, opponent=True)

    index = 2 * BOARD_SIZE + 7
    assert tokens.categorical[index, 0] == TILE_KIND_INDEX["COOP"]
    assert tokens.categorical[index, 1] == TILE_OCCUPANT_INDEX["GOOSE"]
    assert tokens.categorical[index, 2] == FARM_IDENTITIES.index("OPPONENT")
    assert _field(tokens, 7, 2, "yield_fraction") == 2 / 4  # GOOSE max_held 4
    assert _field(tokens, 7, 2, "maturity_fraction") == 1 / 2  # age 2 of first yield 4
    assert _field(tokens, 7, 2, "fed_today") == 1.0
    assert _field(tokens, 7, 2, "cared_today") == 0.0
    assert _field(tokens, 7, 2, "fertilizer_available") == 1.0
    assert _field(tokens, 7, 2, "pending_care_bonus") == pytest.approx(2 / 5)
    assert _field(tokens, 7, 2, "harvest_ready") == 1.0
    # An empty structure keeps its kind but has no occupant features.
    empty = tokenize_farm_tiles(
        _blank_farm({"kind": "PASTURE", "animal": None}), day=8, step=200, opponent=False
    )
    empty_index = 3 * BOARD_SIZE + 3
    assert empty.categorical[empty_index, 0] == TILE_KIND_INDEX["PASTURE"]
    assert empty.categorical[empty_index, 1] == TILE_OCCUPANT_INDEX["NONE"]
    assert _field(empty, 3, 3, "harvest_ready") == 0.0


def test_farm_identity_is_the_only_viewpoint_difference() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 11})
    state = environment.reset(2)
    farm = state[0].observation["farms"][0]

    own = tokenize_farm_tiles(farm, day=0, step=0, opponent=False)
    other = tokenize_farm_tiles(farm, day=0, step=0, opponent=True)

    assert (own.categorical[:, 2] == 0).all()
    assert (other.categorical[:, 2] == 1).all()
    unchanged = [column for column in range(N_TILE_CATEGORICAL) if column != 2]
    assert (own.categorical[:, unchanged] == other.categorical[:, unchanged]).all()
    assert (own.continuous == other.continuous).all()


def test_real_episode_tokens_stay_bounded_and_match_encoder_kinds() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 120, "seed": 7})
    environment.run(["starter", "starter"])

    kind_channels = {
        "LOCKED": 0,
        "EMPTY": 1,
        "WEED": 2,
        "COOP": 8,
        "PASTURE": 9,
    }
    seen_kinds: set[int] = set()
    seen_occupants: set[int] = set()
    for step_state in environment.steps[::TURNS_PER_DAY]:
        observation = step_state[0].observation
        day = int(observation["day"])
        step = int(observation["step"])
        encoded = encode_observation(observation, step_state[1].observation["private"])
        for farm_index, opponent in ((0, False), (1, True)):
            farm = observation["farms"][farm_index]
            tokens = tokenize_farm_tiles(farm, day, step, opponent=opponent)

            assert np.isfinite(tokens.continuous).all()
            assert (tokens.continuous >= 0.0).all() and (tokens.continuous <= 1.0).all()
            assert (tokens.categorical[:, 0] < len(TILE_KINDS)).all()
            assert (tokens.categorical[:, 1] < len(TILE_OCCUPANTS)).all()
            assert (tokens.categorical[:, 5] < QUADRANT_COUNT).all()
            seen_kinds.update(np.unique(tokens.categorical[:, 0]).tolist())
            seen_occupants.update(np.unique(tokens.categorical[:, 1]).tolist())

            if opponent:
                continue
            # The convolutional encoder's one-hot planes are independent
            # authority on every tile's kind for the acting player's farm.
            kinds = tokens.categorical[:, 0].reshape(BOARD_SIZE, BOARD_SIZE)
            board = encoded.board.astype(np.float32)
            for name, channel in kind_channels.items():
                np.testing.assert_array_equal(
                    kinds == TILE_KIND_INDEX[name],
                    board[channel] == 1.0,
                    err_msg=f"kind {name} disagrees with encoder channel {channel}",
                )
            np.testing.assert_array_equal(
                kinds == TILE_KIND_INDEX["PLANT"],
                board[3:8].sum(axis=0) == 1.0,
                err_msg="PLANT kind disagrees with encoder crop channels",
            )
    # The starter route must exercise real content, not just empty boards.
    assert TILE_KIND_INDEX["PLANT"] in seen_kinds
    assert len(seen_occupants) > 1


def test_unit_tokens_follow_execution_order_and_gather_local_tiles() -> None:
    farm = {"farmer": (5, 4), "hands": [(0, 0), (9, 9)]}
    private = {
        "inventories": [
            {"WHEAT": 2, "GOOSE": 1},
            {},
            {"MELON": 40},
        ]
    }

    tokens = tokenize_units(farm, private)

    assert tokens.active.tolist() == [True] * 3 + [False] * (MAX_UNITS - 3)
    assert tokens.categorical[0].tolist() == [0, 0, 4, 5]  # farmer at (x=5, y=4)
    assert tokens.categorical[1].tolist() == [1, 1, 0, 0]
    assert tokens.categorical[2].tolist() == [1, 2, 9, 9]
    assert tokens.positions[0].tolist() == [5, 4]

    wheat = PRIVATE_ITEMS.index("WHEAT")
    goose = PRIVATE_ITEMS.index("GOOSE")
    melon = PRIVATE_ITEMS.index("MELON")
    assert tokens.continuous[0, wheat] == pytest.approx(2 / 32)
    assert tokens.continuous[0, goose] == pytest.approx(1 / 32)
    assert tokens.continuous[0, len(PRIVATE_ITEMS)] == pytest.approx(3 / 32)
    # Held counts are exact, not clipped: 40 melons exceed the 32-unit scale.
    assert tokens.continuous[2, melon] == pytest.approx(40 / 32)
    # The farmer stands on a shed-access tile; the corner hand does not.
    assert tokens.continuous[0, len(PRIVATE_ITEMS) + 1] == 1.0
    assert tokens.continuous[1, len(PRIVATE_ITEMS) + 1] == 0.0

    # Gather indices: HERE, then NSEW, row-major into the 100 tile tokens.
    here = 4 * BOARD_SIZE + 5
    assert tokens.tile_gather[0].tolist() == [
        here,
        here - BOARD_SIZE,
        here + BOARD_SIZE,
        here + 1,
        here - 1,
    ]
    assert tokens.tile_gather_valid[0].all()
    # The (0, 0) hand has no NORTH or WEST neighbor.
    north = UNIT_TILE_GATHERS.index("NORTH")
    west = UNIT_TILE_GATHERS.index("WEST")
    assert not tokens.tile_gather_valid[1, north]
    assert not tokens.tile_gather_valid[1, west]
    assert tokens.tile_gather_valid[1, UNIT_TILE_GATHERS.index("HERE")]
    # Inactive slots are fully zeroed and excluded via the active mask.
    assert not tokens.categorical[3:].any()
    assert not tokens.continuous[3:].any()
    assert not tokens.tile_gather_valid[3:].any()


@pytest.mark.parametrize("player", [0, 1])
def test_inventory_order_distinguishes_drop_successors_without_leaking_private_state(
    player,
) -> None:
    observation = {
        "player": player,
        "farms": [
            {"tiles": [], "farmer": [4, 4], "hands": []},
            {"tiles": [], "farmer": [4, 4], "hands": []},
        ],
        "private": {"shed": {"WHEAT": 99}, "inventories": [{"WHEAT": 1, "MILK": 1}]},
    }
    reversed_private = {"shed": {"WHEAT": 99}, "inventories": [{"MILK": 1, "WHEAT": 1}]}
    baseline = encode_structured_observation(observation, observation["private"])
    reversed_own = {**observation, "private": reversed_private}
    own = encode_structured_observation(reversed_own, observation["private"])
    hidden = encode_structured_observation(observation, reversed_private)
    base_width = len(PRIVATE_ITEMS) + 2
    np.testing.assert_array_equal(
        own.unit_continuous[:, :base_width], baseline.unit_continuous[:, :base_width]
    )
    assert not np.array_equal(own.unit_continuous, baseline.unit_continuous)
    assert not np.array_equal(hidden.opponent_unit_continuous, baseline.opponent_unit_continuous)
    for name in baseline.__dataclass_fields__:
        if not name.startswith(("critic_", "opponent_")):
            np.testing.assert_array_equal(getattr(hidden, name), getattr(baseline, name))
    wheat, milk = (PRIVATE_ITEMS.index(item) for item in ("WHEAT", "MILK"))
    assert baseline.unit_continuous[0, base_width + wheat] == 1 / 32
    assert baseline.unit_continuous[0, base_width + milk] == 2 / 32
    first_shed, second_shed = {"WHEAT": 99}, {"WHEAT": 99}
    apply_unit_shed_effect(observation, 0, UnitAction.DROP, first_shed)
    apply_unit_shed_effect(reversed_own, 0, UnitAction.DROP, second_shed)
    assert first_shed == {"WHEAT": 100}
    assert second_shed == {"WHEAT": 99, "MILK": 1}


def test_economy_tokens_match_engine_market_state() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 60, "seed": 5})
    environment.run(["starter", "starter"])
    for step_state in environment.steps[:: TURNS_PER_DAY // 2]:
        for seat in (0, 1):
            observation = step_state[seat].observation
            tokens = tokenize_economy(observation)

            assert tokens.products.shape == (len(PRODUCTS), len(PRODUCT_TOKEN_FIELDS))
            assert np.isfinite(tokens.products).all()
            assert np.isfinite(tokens.crops).all()
            assert np.isfinite(tokens.farms).all()
            assert np.isfinite(tokens.town).all()
            # Crop mechanics columns are static truths.
            assert tokens.crops[:, 0].max() == 1.0  # STRAWBERRY has max seed cost
            assert tokens.crops[:, 5].sum() == 2.0  # TOMATO and STRAWBERRY ongoing

            market = observation["market"]
            price_field = PRODUCT_TOKEN_FIELDS.index("price")
            for index, item in enumerate(PRODUCTS):
                expected = float(market["prices"][item]) / (2.0 * BASE_PRICE[item])
                assert tokens.products[index, price_field] == pytest.approx(expected, rel=1e-6)

            # Own farm is always the first summary token regardless of seat.
            own_money = observation["farms"][observation["player"]]["money"]
            other_money = observation["farms"][1 - observation["player"]]["money"]
            sign = 1.0 if own_money >= other_money else -1.0
            if own_money != other_money:
                assert sign * (tokens.farms[0, 0] - tokens.farms[1, 0]) > 0


def _signed_log(amount: float) -> float:
    return float(np.copysign(np.log1p(abs(amount)), amount))


def _legacy_v3_farm_rows(observation: dict) -> np.ndarray:
    """The schema-v3 farm tokenizer, verbatim, before the v4 margin existed."""
    player = int(observation.get("player", 0) or 0)
    rows = []
    for farm_index in (player, 1 - player):
        farm = observation["farms"][farm_index]
        amount = float(farm.get("money", 0) or 0)
        rows.append(
            (
                float(np.copysign(np.log1p(abs(amount)) / 12.0, amount)),
                len(farm.get("unlocked_quadrants") or []) / 4.0,
                len(farm.get("hands") or []) / float(MAX_UNITS - 1),
                float(farm.get("hires_today", 0) or 0) / float(MAX_UNITS - 1),
            )
        )
    return np.asarray(rows, dtype=np.float32)


def test_farm_token_schemas_are_prefixes_of_the_emitted_layout() -> None:
    assert {3, 4, 5, 6, 7, 8} == SUPPORTED_OBSERVATION_SCHEMA_VERSIONS
    assert farm_token_fields(OBSERVATION_SCHEMA_VERSION) == FARM_TOKEN_FIELDS
    assert farm_token_fields(8) == farm_token_fields(7) == farm_token_fields(6)
    assert farm_token_fields(6) == (
        *farm_token_fields(5),
        "liquidation",
        "liquidation_margin",
    )
    assert farm_token_fields(5) == farm_token_fields(4)
    assert farm_token_fields(4) == (*farm_token_fields(3), "money_margin")
    assert farm_token_fields(3) == ("money", "unlocked_quadrants", "hands", "hires_today")
    for version in (2, 9):
        with pytest.raises(ValueError, match="unsupported observation schema"):
            farm_token_fields(version)


def test_town_token_schemas_are_prefixes_of_the_emitted_layout() -> None:
    assert town_token_fields(OBSERVATION_SCHEMA_VERSION) == TOWN_TOKEN_FIELDS
    legacy = (
        "day",
        "hour",
        "progress",
        "remaining",
        "hour_sin",
        "hour_cos",
        *(f"shop_{name}" for name in SHOP_NAMES),
    )
    assert town_token_fields(3) == town_token_fields(4) == legacy
    assert town_token_fields(5) == (
        *legacy,
        *(f"shop_{name}_first_unlock" for name in SHOP_NAMES),
    )
    assert town_token_fields(8) == town_token_fields(7) == town_token_fields(6)
    assert town_token_fields(6) == town_token_fields(5)
    for version in (2, 9):
        with pytest.raises(ValueError, match="unsupported observation schema"):
            town_token_fields(version)


def test_product_token_schemas_are_prefixes_of_the_emitted_layout() -> None:
    assert product_token_fields(OBSERVATION_SCHEMA_VERSION) == PRODUCT_TOKEN_FIELDS
    legacy = ("market_inventory", "price", "base_price", "shed_stock", "carried_stock")
    for version in (3, 4, 5):
        assert product_token_fields(version) == legacy
        assert product_private_fields(version) == (
            "opponent_shed_stock",
            "opponent_carried_stock",
        )
    assert product_token_fields(6) == (*legacy, "held_value")
    assert product_private_fields(6) == (*product_private_fields(5), "opponent_held_value")
    assert product_token_fields(7) == (
        *product_token_fields(6),
        "forecast_price",
        "supply_soon",
        "supply_to_end",
        "opponent_supply_soon",
        "opponent_supply_to_end",
        "town_draw_soon",
        "town_draw_to_end",
    )
    assert product_private_fields(7) == (
        *product_private_fields(6),
        "opponent_forecast_price",
    )
    assert product_token_fields(8) == product_token_fields(7)
    assert product_private_fields(8) == product_private_fields(7)
    assert product_private_fields(OBSERVATION_SCHEMA_VERSION) == PRODUCT_PRIVATE_FIELDS
    for version in (2, 9):
        with pytest.raises(ValueError, match="unsupported observation schema"):
            product_token_fields(version)
        with pytest.raises(ValueError, match="unsupported observation schema"):
            product_private_fields(version)


def test_crop_and_animal_token_schemas_are_prefixes_of_the_emitted_layout() -> None:
    assert crop_token_fields(OBSERVATION_SCHEMA_VERSION) == CROP_TOKEN_FIELDS
    assert animal_token_fields(OBSERVATION_SCHEMA_VERSION) == ANIMAL_TOKEN_FIELDS
    crops = ("seed_cost", "seeds_held", "first_yield_day", "max_yield_day", "max_yield", "ongoing")
    animals = ("purchase_price", "shed_stock", "carried_stock")
    for version in (3, 4, 5, 6, 7):
        assert crop_token_fields(version) == crops
        assert animal_token_fields(version) == animals
    assert crop_token_fields(8) == (*crops, "payback", "forecast_payback")
    assert animal_token_fields(8) == (*animals, "payback", "forecast_payback")
    for version in (2, 9):
        with pytest.raises(ValueError, match="unsupported observation schema"):
            crop_token_fields(version)
        with pytest.raises(ValueError, match="unsupported observation schema"):
            animal_token_fields(version)


def _town_shops(town: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Per-shop (count, first-unlock rank) columns of one town token."""
    column = {name: index for index, name in enumerate(TOWN_TOKEN_FIELDS)}
    return (
        {name: float(town[column[f"shop_{name}"]]) for name in SHOP_NAMES},
        {name: float(town[column[f"shop_{name}_first_unlock"]]) for name in SHOP_NAMES},
    )


def test_town_first_unlock_ranks_distinct_shops_in_unlock_order() -> None:
    def observation(shops: list[str]) -> dict:
        return {"farms": [{}, {}], "town": {"unlocked_shops": shops}}

    def town(shops: list[str]) -> np.ndarray:
        return tokenize_economy(observation(shops)).town

    # A repeat instance adds to its shop's count but keeps its first rank.
    counts, ranks = _town_shops(town(["PIZZA_SHOP", "BAKERY", "PIZZA_SHOP"]))
    assert counts == {**dict.fromkeys(SHOP_NAMES, 0.0), "PIZZA_SHOP": 0.25, "BAKERY": 0.125}
    assert ranks == {**dict.fromkeys(SHOP_NAMES, 0.0), "PIZZA_SHOP": 0.125, "BAKERY": 0.25}
    # The swapped opening: identical counts, so only the ranks separate them.
    swapped_counts, swapped_ranks = _town_shops(town(["BAKERY", "PIZZA_SHOP", "PIZZA_SHOP"]))
    assert swapped_counts == counts
    assert swapped_ranks == {**ranks, "PIZZA_SHOP": 0.25, "BAKERY": 0.125}
    # Eight unlocks fill the ranks to exactly one, and fp16 staging is exact.
    every = list(reversed(SHOP_NAMES))
    _, full = _town_shops(town(every))
    assert full == {name: (every.index(name) + 1) / 8 for name in SHOP_NAMES}
    staged = encode_structured_observation(observation(every)).town
    assert staged.tobytes() == town(every).astype(np.float16).tobytes()
    assert not town([])[len(town_token_fields(4)) :].any()


@pytest.mark.parametrize(
    "own_money,other_money",
    [
        (80_000, 76_000),  # a close late game: 0.0513
        (80_000, 80_001),  # one coin at a late-game bank
        (4_000, 40_000),  # a 10x blowout: -2.3
        (0, 1_000_000_000),
        (250, 250),
        (0, 0),
        (-30, 12),  # signed log keeps a debt below every bank
    ],
)
def test_money_margin_is_the_float64_signed_log_ratio_and_swaps_with_the_seat(
    own_money, other_money
) -> None:
    margin = FARM_TOKEN_FIELDS.index("money_margin")
    observation = {
        "farms": [
            {"money": own_money, "unlocked_quadrants": [0], "hands": [[1, 1]]},
            {"money": other_money, "unlocked_quadrants": [], "hands": [], "hires_today": 2},
        ],
    }
    seat_zero = tokenize_economy({**observation, "player": 0}).farms
    seat_one = tokenize_economy({**observation, "player": 1}).farms

    expected = np.float32(_signed_log(own_money) - _signed_log(other_money))
    assert seat_zero.dtype == np.float32
    assert seat_zero[0, margin] == expected
    # Own row first: each row is that farm minus the other, so rows negate
    # exactly and a seat swap is a row swap of every column.
    assert seat_zero[1, margin] == -expected
    np.testing.assert_array_equal(seat_one, seat_zero[::-1])
    if own_money != other_money:
        assert np.sign(seat_zero[0, margin]) == np.sign(own_money - other_money)
    # Rollout staging is float16, rounded from the same float32 value.
    staged = encode_structured_observation({**observation, "player": 0}).farms
    assert staged[0, margin] == np.float16(expected)
    assert staged[1, margin] == -np.float16(expected)


def test_v3_farm_columns_are_unchanged_by_the_v4_margin() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 80, "seed": 11})
    environment.run(["starter", "starter"])
    v3_width = len(farm_token_fields(3))
    for step_state in environment.steps[::7]:
        for seat in (0, 1):
            observation = step_state[seat].observation
            farms = tokenize_economy(observation).farms
            legacy = _legacy_v3_farm_rows(observation)
            assert farms[:, :v3_width].tobytes() == legacy.tobytes()
            staged = encode_structured_observation(observation).farms
            assert staged[:, :v3_width].tobytes() == legacy.astype(np.float16).tobytes()


def _legacy_v5_product_rows(observation: dict) -> np.ndarray:
    """The schema-v5 product tokenizer, verbatim, before the v6 held value existed."""
    private = observation.get("private") or {}
    market = observation.get("market") or {}
    inventory = market.get("inventory") or {}
    prices = market.get("prices") or {}
    shed = private.get("shed") or {}
    carried = {
        item: sum(int(unit.get(item, 0) or 0) for unit in private.get("inventories") or [])
        for item in PRODUCTS
    }
    max_base_price = float(max(BASE_PRICE.values()))
    return np.asarray(
        [
            (
                (float(inventory.get(item, 10_000) or 0) - 10_000) / 500.0,
                float(prices.get(item, BASE_PRICE[item]) or 0) / (2.0 * BASE_PRICE[item]),
                BASE_PRICE[item] / max_base_price,
                float(shed.get(item, 0) or 0) / SHED_CAPACITY,
                carried[item] / SHED_CAPACITY,
            )
            for item in PRODUCTS
        ],
        dtype=np.float32,
    )


def test_v5_columns_are_unchanged_by_the_v6_liquidation() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 240, "seed": 11})
    environment.run(["starter", "starter"])
    v5_products = len(product_token_fields(5))
    v5_farms = len(farm_token_fields(5))
    held = 0
    for step_state in environment.steps[::3]:
        for seat in (0, 1):
            observation = step_state[seat].observation
            economy = tokenize_economy(observation)
            legacy = _legacy_v5_product_rows(observation)
            assert economy.products[:, :v5_products].tobytes() == legacy.tobytes()
            staged = encode_structured_observation(observation)
            assert staged.products[:, :v5_products].tobytes() == (
                legacy.astype(np.float16).tobytes()
            )
            # Every v6 column is exactly what it replaces: the bank's own terms.
            own_value = liquidation_value(observation, seat)
            held += own_value != observation["farms"][seat]["money"]
            farms = tokenize_economy(observation).farms
            margin = FARM_TOKEN_FIELDS.index("money_margin")
            liquidation = FARM_TOKEN_FIELDS.index("liquidation")
            assert farms[0, liquidation] == np.float32(_signed_log(own_value) / 12.0)
            assert farms[1, liquidation] == farms[1, 0]
            if own_value == observation["farms"][seat]["money"]:
                assert farms[:, v5_farms:].tobytes() == farms[:, [0, margin]].tobytes()
    # The starter holds harvests between sales, so the new columns were live.
    assert held > 10


def test_held_value_and_liquidation_price_only_the_stock_a_seat_can_see() -> None:
    held_value = PRODUCT_TOKEN_FIELDS.index("held_value")
    liquidation = FARM_TOKEN_FIELDS.index("liquidation")
    margin = FARM_TOKEN_FIELDS.index("liquidation_margin")
    # MELON quotes walk from 31 onto the price floor partway through the stock.
    floor = next(level for level in range(10_000, 11_000) if market_price("MELON", level) == 1)
    observation = {
        "player": 0,
        "farms": [{"money": 250, "hands": []}, {"money": 4_000, "hands": []}],
        "market": {"inventory": {"MELON": floor - 10}},
        "private": {"shed": {"WHEAT": 60}, "inventories": [{"MELON": 25}]},
    }
    opponent_private = {"shed": {"WOOL": 90}, "inventories": [{"WOOL": 3}, {"EGG": 7}]}
    tokens = tokenize_economy(observation)

    proceeds = {
        "WHEAT": sale_proceeds("WHEAT", 60, 10_000),
        "MELON": sale_proceeds("MELON", 25, floor - 10),
    }
    assert proceeds["MELON"] == sum(market_price("MELON", floor - 10 + n) for n in range(10)) + 15
    assert proceeds["MELON"] < 25 * market_price("MELON", floor - 10)
    for index, item in enumerate(PRODUCTS):
        expected = proceeds.get(item, 0) / HELD_VALUE_SCALE
        assert tokens.products[index, held_value] == np.float32(expected)
    own = 250 + sum(proceeds.values())
    assert own == liquidation_value(observation, 0)
    assert tokens.farms[0, liquidation] == np.float32(_signed_log(own) / 12.0)
    # The opponent's stock is private: its row values its bank alone.
    assert tokens.farms[1, liquidation] == np.float32(_signed_log(4_000) / 12.0)
    expected_margin = np.float32(_signed_log(own) - _signed_log(4_000))
    assert tokens.farms[0, margin] == expected_margin
    assert tokens.farms[1, margin] == -expected_margin
    assert expected_margin > tokens.farms[0, FARM_TOKEN_FIELDS.index("money_margin")]

    # The critic's private column is the opponent's own held value, priced
    # against the shared market; the actor's tokens never move with it.
    staged = encode_structured_observation(observation, opponent_private)
    blind = encode_structured_observation(observation, {"shed": {}, "inventories": []})
    for name in ("products", "animals", "crops", "farms", "town"):
        np.testing.assert_array_equal(getattr(staged, name), getattr(blind, name))
    opponent_held = PRODUCT_PRIVATE_FIELDS.index("opponent_held_value")
    for index, item in enumerate(PRODUCTS):
        units = {"WOOL": 93, "EGG": 7}.get(item, 0)
        expected = sale_proceeds(item, units, 10_000) / HELD_VALUE_SCALE
        assert staged.critic_products[index, opponent_held] == np.float16(np.float32(expected))
    opponent_view = {
        **observation,
        "player": 1,
        "private": opponent_private,
    }
    np.testing.assert_array_equal(
        staged.critic_products[:, opponent_held],
        encode_structured_observation(opponent_view).products[:, held_value],
    )


def test_clock_features_are_bounded_and_phase_consistent() -> None:
    features = clock_features({"day": 5, "hour": 6, "step": 126})

    assert features.dtype == np.float32
    assert features.shape == (6,)
    assert np.isclose(features[4] ** 2 + features[5] ** 2, 1.0)
    assert np.isclose(features[2] + features[3], 1.0)
    start = clock_features({"day": 0, "hour": 0, "step": 0})
    assert start[2] == 0.0 and start[3] == 1.0


def test_animal_stock_and_public_units_preserve_actor_critic_information_boundary() -> None:
    observation = {
        "player": 0,
        "farms": [
            {"tiles": [], "farmer": [4, 4], "hands": []},
            {"tiles": [], "farmer": [5, 5], "hands": [[3, 3], [3, 3]]},
        ],
        "private": {"shed": {}, "inventories": [{}]},
    }
    hidden = {"shed": {}, "inventories": [{}, {}, {}]}
    baseline = encode_structured_observation(observation, hidden)
    own = deepcopy(observation)
    own["private"]["shed"]["GOOSE"] = 3
    own["private"]["inventories"][0]["COW"] = 2
    changed = encode_structured_observation(own, hidden)
    assert changed.animals.shape == (len(ANIMALS), len(ANIMAL_TOKEN_FIELDS))
    assert changed.animals[ANIMALS.index("GOOSE"), 1] == np.float16(3 / SHED_CAPACITY)
    assert changed.animals[ANIMALS.index("COW"), 2] == np.float16(2 / SHED_CAPACITY)
    assert not np.array_equal(changed.animals, baseline.animals)
    np.testing.assert_array_equal(changed.products, baseline.products)
    np.testing.assert_array_equal(
        changed.animals[:, 0],
        np.asarray(
            [ANIMAL_COST[item] / max(ANIMAL_COST.values()) for item in ANIMALS], dtype=np.float16
        ),
    )

    moved = deepcopy(observation)
    moved["farms"][1]["farmer"] = [5, 4]
    moved["farms"][1]["hands"][0] = [3, 4]
    public = encode_structured_observation(moved, hidden)
    assert not np.array_equal(public.tile_continuous, baseline.tile_continuous)
    tiles = baseline.tile_continuous[TILE_COUNT:]
    assert tiles[5 * BOARD_SIZE + 5, _FIELD["farmer_present"]] == 1
    assert tiles[3 * BOARD_SIZE + 3, _FIELD["hand_count"]] == np.float16(2 / (MAX_UNITS - 1))
    assert public.tile_continuous[
        TILE_COUNT + 3 * BOARD_SIZE + 3, _FIELD["hand_count"]
    ] == np.float16(1 / (MAX_UNITS - 1))

    hidden["shed"]["GOOSE"] = 5
    hidden["inventories"][1]["SHEEP"] = 2
    private_changed = encode_structured_observation(observation, hidden)
    for name in baseline.__dataclass_fields__:
        if not name.startswith(("critic_", "opponent_")):
            np.testing.assert_array_equal(getattr(private_changed, name), getattr(baseline, name))
    assert private_changed.critic_animals[ANIMALS.index("GOOSE"), 0] == np.float16(
        5 / SHED_CAPACITY
    )
    assert private_changed.critic_animals[ANIMALS.index("SHEEP"), 1] == np.float16(
        2 / SHED_CAPACITY
    )


def test_native_town_tokens_match_python_through_every_shop_unlock() -> None:
    """Both tokenizers agree as a real town unlocks repeats out of name order."""
    import json

    from kaggriculture.constants import MAX_MARKET_ORDERS
    from kaggriculture.rust_env import load_native
    from kaggriculture.script_opponents import shaped_observation
    from kaggriculture.structured import StructuredInputs

    # Seed 0 opens YARN_STORE, BAKERY and unlocks both again later.
    environment = load_native().BatchEnv(np.asarray([0], dtype=np.uint64))
    units = np.zeros((1, 2, MAX_UNITS), dtype=np.uint8)
    orders = np.zeros((1, 2, MAX_MARKET_ORDERS), dtype=np.uint8)
    for step in range(EPISODE_STEPS - 1):
        # The town changes only at day boundaries, and the first hour of each
        # day is the first observation carrying a new unlock.
        if step % TURNS_PER_DAY == 0:
            snapshot = json.loads(environment.snapshot_json(0))
            native = environment.structured()
            for seat in (0, 1):
                python = encode_structured_observation(shaped_observation(snapshot, seat))
                for name in StructuredInputs._fields:
                    np.testing.assert_array_equal(
                        getattr(python, name),
                        np.asarray(native[name])[seat],
                        err_msg=f"Python/native divergence at step {step}: {name}",
                    )
        environment.step_factors(units, orders, orders)
    shops = snapshot["town"]["unlocked_shops"]
    # The seed exercises what the ranks exist for: a repeated shop, and an
    # unlock order the per-shop counts (in name order) cannot express.
    assert len(set(shops)) < len(shops)
    assert list(dict.fromkeys(shops)) != sorted(set(shops))


def test_native_liquidation_tokens_match_python_while_v27_trades() -> None:
    """Both tokenizers price held stock identically, and so do the critic columns.

    Two scripted-v27 seats hold harvests and livestock produce in volume, so the
    held values walk the price curve; the rollout's paired-seat critic columns
    must equal Python's valuation of the opponent's private stock.
    """
    import json

    from kaggriculture.rollout import _PRODUCT_STOCK_COLUMNS
    from kaggriculture.rust_env import load_native
    from kaggriculture.script_opponents import shaped_observation

    environment = load_native().BatchEnv(np.asarray([29], dtype=np.uint64))
    scripted = np.full(2, 4, dtype=np.uint8)  # BUILTIN_AGENT_ORDER[3], scripted-v27
    held_value = PRODUCT_TOKEN_FIELDS.index("held_value")
    impacted = 0
    for step in range(EPISODE_STEPS - 1):
        if step % 5 == 0:
            snapshot = json.loads(environment.snapshot_json(0))
            native = environment.structured()
            for seat in (0, 1):
                observation = shaped_observation(snapshot, seat)
                python = encode_structured_observation(observation, snapshot["privates"][1 - seat])
                # Economy tokens only: a snapshot's unit inventories do not keep
                # insertion order, which the unit tokens' carried ranks read.
                for name in ("products", "animals", "crops", "farms", "town"):
                    np.testing.assert_array_equal(
                        getattr(python, name),
                        np.asarray(native[name])[seat],
                        err_msg=f"Python/native divergence at step {step}: {name}",
                    )
                np.testing.assert_array_equal(
                    python.critic_products,
                    np.asarray(native["products"])[1 - seat][:, _PRODUCT_STOCK_COLUMNS],
                    err_msg=f"critic product divergence at step {step}",
                )
                inventory = observation["market"]["inventory"]
                for index, item in enumerate(PRODUCTS):
                    units = round(
                        float(python.products[index, 3] + python.products[index, 4]) * SHED_CAPACITY
                    )
                    quoted = units * market_price(item, int(inventory[item]))
                    worth = float(python.products[index, held_value]) * HELD_VALUE_SCALE
                    impacted += units > 1 and worth < 0.99 * quoted
        actions = environment.builtin_actions(scripted)
        environment.step_factors(
            *(
                actions[name].reshape(1, 2, -1)
                for name in ("unit_actions", "market_kinds", "market_quantities")
            )
        )
    # The stock was large enough, often enough, for price impact to matter.
    assert impacted > 20


def test_outlook_constants_match_the_official_engine() -> None:
    """The yield and town rules `market_outlook` mirrors are the engine's own."""
    for crop in CROPS:
        rules = official.CROPS[crop]
        assert CROP_FIRST_YIELD_DAY[crop] == rules["first_yield_day"]
        assert CROP_MAX_YIELD_DAY[crop] == rules["max_yield_day"]
        assert CROP_MAX_YIELD[crop] == rules["max_yield"]
        assert CROP_YIELD_INTERVAL[crop] == rules["interval"]
        assert (crop in ONGOING_CROPS) == rules["ongoing"]
    for animal in ANIMALS:
        rules = official.ANIMALS[animal]
        assert ANIMAL_FIRST_YIELD_DAY[animal] == rules["first_yield_day"]
        assert ANIMAL_YIELD_INTERVAL[animal] == rules["interval"]
        assert ANIMAL_MAX_HELD[animal] == rules["max_held"]
        assert ANIMAL_PRODUCT[animal] == rules["product"]
    shops = {name: tuple(official.SHOPS[name]) for name in sorted(official.SHOPS)}
    assert shops == SHOP_PRODUCTS
    assert tuple(shops) == SHOP_NAMES
    assert tuple(official.TOWN_CENTER_PRODUCTS) == TOWN_CENTER_PRODUCTS
    assert official.MAX_SHOP_INSTANCES == MAX_SHOP_INSTANCES
    configuration = make("kaggriculture").configuration
    assert configuration.townShopSellInterval == TOWN_SHOP_SELL_INTERVAL
    assert configuration.townCenterSellInterval == TOWN_CENTER_SELL_INTERVAL
    assert configuration.townShopUnlockInterval == TOWN_SHOP_UNLOCK_INTERVAL
    assert configuration.episodeSteps == EPISODE_STEPS
    assert configuration.turnsPerDay == TURNS_PER_DAY


def _supply_observation(farm: dict, step: int) -> dict:
    empty = {"tiles": [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]}
    return {
        "player": 0,
        "farms": [farm, empty],
        "step": step,
        "day": step // TURNS_PER_DAY,
        "hour": step % TURNS_PER_DAY,
    }


def _engine_step(farm: dict, step: int, tile_actions) -> list[tuple[str, int, int]]:
    """Run one step of the official engine on ``farm`` and return what units picked up.

    Each pickup is ``(item, units, shed_steps)``, the last the moves from its
    tile to the nearest shed-access tile.

    A unit stands on every occupied tile in turn and takes ``tile_actions``'
    ops there, handed the WHEAT or FERTILIZER an op consumes; then the plants
    decay and, on the day's last step, the engine's daily refresh runs.
    """
    day = step // TURNS_PER_DAY
    supplied = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}
    picked_up = []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            farm["farmer"] = [x, y]
            private = official._new_private()
            inventory = official._farmer_inventory(private, 0)
            for op in tile_actions(tile, day):
                if op in supplied:
                    official._inv_add(inventory, supplied[op])
                official._apply_unit_action(
                    farm, private, 0, [op], BOARD_SIZE, day, TURNS_PER_DAY, SHED_CAPACITY
                )
                if op in supplied:
                    official._inv_take(inventory, supplied[op])
            shed_steps = min(
                abs(x - ax) + abs(y - ay) for ax, ay in shed_access_tiles(len(farm["tiles"]))
            )
            picked_up.extend((item, units, shed_steps) for item, units in inventory.items())
    official._decay_plants(farm, step)
    if (step + 1) % TURNS_PER_DAY == 0:
        official._daily_refresh_plants(farm, day, TURNS_PER_DAY)
        official._daily_refresh_animals(farm, day)
    return picked_up


def _sold_in_time(step: int, shed_steps: int) -> bool:
    """Whether a pickup at ``step`` sells on a step that acts.

    Its unit walks to the shed and drops it, the market clearing that step,
    or the day's end drops it there to sell on the next day's first step.
    """
    dropped = (step // TURNS_PER_DAY + 1) * TURNS_PER_DAY
    return min(step + shed_steps + 1, dropped) < EPISODE_STEPS - 1


def _nominal_care(tile: object, day: int) -> tuple[str, ...]:
    """The care `_farm_supply` assumes: water and feed daily, harvest once grown.

    Whether a crop is grown is judged after the watering the step's ops
    begin with: another unit on the tile can harvest right after it.
    """
    if not isinstance(tile, dict):
        return ()
    if tile.get("animal"):
        return ("FEED", "HARVEST", "COLLECT_FERTILIZER")
    if tile.get("kind") != "PLANT":
        return ()
    crop = tile["crop"]
    age = day - tile["planted_day"]
    stock = tile["yield_units"]
    growing = (
        crop not in ONGOING_CROPS
        and not tile["watered_today"]
        and (CROP_MAX_YIELD_DAY[crop] + 1) // 2 <= age <= CROP_MAX_YIELD_DAY[crop]
    )
    if growing:
        bonus = 2 if tile["fertilized_until_day"] >= day else 1
        stock = min(CROP_MAX_YIELD[crop], stock + bonus)
    grown = (
        crop in ONGOING_CROPS or stock >= CROP_MAX_YIELD[crop] or age >= CROP_MAX_YIELD_DAY[crop]
    )
    return ("WATER", "HARVEST") if grown else ("WATER",)


def _nominal_supply(farm: dict, step: int) -> tuple[dict[str, int], dict[str, int], int]:
    """What the engine's nominal care of ``farm`` from ``step`` sells, soon and to the end.

    Also the WHEAT it feeds before the last acting day, when a feed can still
    keep a production.
    """
    last_day = (EPISODE_STEPS - 2) // TURNS_PER_DAY
    nominal = deepcopy(farm)
    soon = dict.fromkeys(PRODUCTS, 0)
    to_end = dict.fromkeys(PRODUCTS, 0)
    feeds = 0

    def care(tile: object, day: int) -> tuple[str, ...]:
        nonlocal feeds
        ops = _nominal_care(tile, day)
        feeds += "FEED" in ops and not tile["fed_today"] and day < last_day
        return ops

    for now in range(step, EPISODE_STEPS - 1):
        for item, units, shed_steps in _engine_step(nominal, now, care):
            if _sold_in_time(now, shed_steps):
                to_end[item] += units
                if now < step + OUTLOOK_SOON_STEPS:
                    soon[item] += units
    return soon, to_end, feeds


@pytest.mark.parametrize("seed", [3, 11])
def test_farm_supply_is_the_official_engines_yield_under_nominal_care(seed: int) -> None:
    """Every product's soon and to-end supply equals the engine's own harvests.

    A random history (sparse watering, feeding, care, fertilizer, and late
    harvests, so tiles decay, die, escape, and bank care) runs on the official
    engine's action and daily-refresh functions. From checkpoints across the
    game, the engine then plays nominal care to the last acting step; what the
    units pick up, and when, is the supply `market_outlook` must predict.
    """
    rng = random.Random(seed)
    farm = {"tiles": [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]}

    def history(tile: object, day: int) -> list[str]:
        if not isinstance(tile, dict):
            return []
        chances = (
            {"FEED": 0.15, "CARE": 0.05, "HARVEST": 0.03, "COLLECT_FERTILIZER": 0.05}
            if tile.get("animal")
            else {"FERTILIZE": 0.01, "WATER": 0.15, "HARVEST": 0.03}
        )
        return [op for op, chance in chances.items() if rng.random() < chance]

    # Mid-day steps, day starts (whose soon window ends on a production), and
    # the last steps, where the end of the game cuts the supply short.
    checkpoints = {
        *range(0, EPISODE_STEPS - 1, 29),
        *range(0, EPISODE_STEPS - 1, 7 * TURNS_PER_DAY),
        *range(EPISODE_STEPS - 31, EPISODE_STEPS - 1, 5),
        EPISODE_STEPS - 2,
    }
    seen = dict.fromkeys(("pending_care_bonus", "cared_today", "fertilizer_available"), 0)
    seen |= {"fertilized_ongoing": 0, "watered_today": 0, "expired": 0}
    supplied = dict.fromkeys(PRODUCTS, 0)
    for step in range(EPISODE_STEPS - 1):
        for row in farm["tiles"]:
            for x, tile in enumerate(row):
                if tile is None and rng.random() < 0.004:
                    species = rng.choice((*CROPS, *ANIMALS))
                    day = step // TURNS_PER_DAY
                    row[x] = (
                        official._new_animal(species, day)
                        if species in ANIMALS
                        else official._new_plant(species, day, TURNS_PER_DAY)
                    )
        if step in checkpoints:
            outlook = market_outlook(_supply_observation(farm, step))
            tiles = [tile for row in farm["tiles"] for tile in row if isinstance(tile, dict)]
            for name in ("pending_care_bonus", "cared_today", "fertilizer_available"):
                seen[name] += sum(bool(tile.get(name)) for tile in tiles if tile.get("animal"))
            plants = [tile for tile in tiles if tile.get("kind") == "PLANT"]
            seen["watered_today"] += sum(tile["watered_today"] for tile in plants)
            seen["fertilized_ongoing"] += sum(
                tile["crop"] in ONGOING_CROPS and tile["fertilized_until_day"] >= step // 24
                for tile in plants
            )
            seen["expired"] += sum(0 <= tile["max_lifespan_step"] <= step for tile in plants)
            soon, to_end, feeds = _nominal_supply(farm, step)
            assert outlook.supply_soon[0] == soon, f"soon supply at step {step}"
            assert outlook.supply_to_end[0] == to_end, f"supply to the end at step {step}"
            assert outlook.supply_to_end[1] == dict.fromkeys(PRODUCTS, 0)
            assert outlook.feeds_to_end == feeds, f"feeds at step {step}"
            for item in PRODUCTS:
                supplied[item] += to_end[item]
        _engine_step(farm, step, history)
    # The history reached every product and every tile state the model reads.
    assert min(supplied.values()) > 0, supplied
    assert min(seen.values()) > 0, seen


def _started_on_the_engine(
    name: str, step: int, *, feed_daily: bool = False
) -> tuple[dict[str, int], int]:
    """What the engine's own fresh ``name``, started at ``step``, yields by the end, and its feeds.

    Nominal care (`_nominal_care`) on the official engine, but an animal is
    fed only while a later day still acts, and unless ``feed_daily`` only
    when the day's refresh would otherwise let it escape; a crop still
    standing on the last acting day is harvested then, after its watering.
    """
    last_day = (EPISODE_STEPS - 2) // TURNS_PER_DAY
    day = step // TURNS_PER_DAY
    tile = (
        official._new_animal(name, day)
        if name in ANIMALS
        else official._new_plant(name, day, TURNS_PER_DAY)
    )
    farm = {"tiles": [[tile]]}
    feeds = 0

    def care(tile: object, day: int) -> list[str]:
        nonlocal feeds
        ops = list(_nominal_care(tile, day))
        if "FEED" in ops:
            if day == last_day or not (feed_daily or tile["consecutive_unfed"]):
                ops.remove("FEED")
            elif not tile["fed_today"]:
                feeds += 1
        if day == last_day and "HARVEST" not in ops and ops:
            ops.append("HARVEST")
        return ops

    harvested = dict.fromkeys(PRODUCTS, 0)
    for now in range(step, EPISODE_STEPS - 1):
        for item, units, _ in _engine_step(farm, now, care):
            # Sown beside the shed.
            if _sold_in_time(now, 0):
                harvested[item] += units
    return harvested, feeds


def test_paybacks_value_the_official_engines_yield_of_a_fresh_start_over_its_cost() -> None:
    """Each crop and animal token's v8 paybacks, against the engine's own harvests.

    From steps across the game and every step of its last three days, a
    crop sown or an animal placed then plays out on the official engine;
    its harvests at the quoted and at the forecast prices, over the seed or
    the animal and the fewest WHEAT that kept it, give the paybacks exactly.
    """
    prices = {item: BASE_PRICE[item] + 7 * index for index, item in enumerate(PRODUCTS)}
    steps = {
        *range(0, EPISODE_STEPS - 1, 11),
        *range(EPISODE_STEPS - 1 - 3 * TURNS_PER_DAY, EPISODE_STEPS - 1),
    }
    harvests: dict[str, set[int]] = {name: set() for name in (*CROPS, *ANIMALS)}
    forecast_price = PRODUCT_TOKEN_FIELDS.index("forecast_price")
    for step in sorted(steps):
        observation = {
            "player": 0,
            "step": step,
            "day": step // TURNS_PER_DAY,
            "hour": step % TURNS_PER_DAY,
            "farms": [_blank_farm(None), _blank_farm(None)],
            "market": {"prices": prices},
            "town": {"unlocked_shops": ["BAKERY"]},
        }
        tokens = tokenize_economy(observation)
        forecast = {
            item: round(float(tokens.products[index, forecast_price]) * 2 * BASE_PRICE[item])
            for index, item in enumerate(PRODUCTS)
        }
        assert forecast != prices
        for name in (*CROPS, *ANIMALS):
            harvested, feeds = _started_on_the_engine(name, step)
            if name in ANIMALS:
                # Feeding only to stave off escape yields what daily feeding
                # does, at about half the WHEAT.
                daily, daily_feeds = _started_on_the_engine(name, step, feed_daily=True)
                assert harvested == daily, (name, step)
                assert feeds == daily_feeds // 2, (name, step)
                row, fields = tokens.animals[ANIMALS.index(name)], ANIMAL_TOKEN_FIELDS
            else:
                row, fields = tokens.crops[CROPS.index(name)], CROP_TOKEN_FIELDS
                assert feeds == 0
            for column, quotes in (("payback", prices), ("forecast_payback", forecast)):
                value = sum(units * quotes[item] for item, units in harvested.items())
                cost = (
                    ANIMAL_COST[name] + feeds * quotes["WHEAT"]
                    if name in ANIMALS
                    else SEED_COST[name]
                )
                expected = np.float32(math.log1p(value / cost))
                assert row[fields.index(column)] == expected, (name, step, column)
            harvests[name].add(sum(harvested.values()))
    # The end of the game cut every start short, down to nothing, and the
    # single-yield crops still growing on its last day to part of their yield.
    for name, seen in harvests.items():
        assert 0 in seen and len(seen) >= 2, (name, seen)
    for crop in ("WHEAT", "CARROT"):
        assert len(harvests[crop]) >= 3, (crop, harvests[crop])


@pytest.mark.parametrize(
    ("step", "planted_day", "yield_units", "position", "sold"),
    [
        # Watered to its cap on day 12 and harvested at once, on the soon
        # window's last step.
        (24 * 12 - 47, 2, 3, (3, 3), (6, 6)),
        # Capped by the last acting step's watering: harvested, but too late
        # to sell.
        (EPISODE_STEPS - 2, 19, 5, (3, 3), (0, 0)),
        # Capped on the last day, then carried from the far corner in time...
        (EPISODE_STEPS - 24, 19, 5, (0, 0), (6, 6)),
        # ...or not, unless harvested beside the shed.
        (EPISODE_STEPS - 8, 19, 5, (0, 0), (0, 0)),
        (EPISODE_STEPS - 8, 19, 5, (4, 4), (6, 6)),
    ],
)
def test_a_melon_watered_to_its_cap_sells_as_the_engine_sells_it(
    step: int, planted_day: int, yield_units: int, position: tuple[int, int], sold: tuple[int, int]
) -> None:
    melon = official._new_plant("MELON", planted_day, TURNS_PER_DAY)
    melon["yield_units"] = yield_units
    farm = _blank_farm(melon, *position)
    outlook = market_outlook(_supply_observation(farm, step))
    soon, to_end, _ = _nominal_supply(farm, step)
    assert (soon["MELON"], to_end["MELON"]) == sold
    assert outlook.supply_soon[0] == soon
    assert outlook.supply_to_end[0] == to_end


def _consumed_by_the_town(
    shops: list[str], step: int, stop: int, unlocks: tuple[str, ...]
) -> tuple[dict[str, int], int]:
    """The units the engine's `_town_consume` takes in [step, stop), and the unlocks used.

    ``unlocks`` are the shops the town draws, in order, on its unlock days.
    """
    market = official._new_market()
    town = {"unlocked_shops": list(shops)}
    state = [SimpleNamespace(observation=SimpleNamespace(market=market, town=town))]
    environment = SimpleNamespace(configuration=make("kaggriculture").configuration)
    drawn = iter(unlocks)
    for now in range(step, stop):
        # `_end_of_day` opens the shop before the new day's first step sells.
        unlocking = (
            now > step
            and now % TURNS_PER_DAY == 0
            and now // TURNS_PER_DAY % TOWN_SHOP_UNLOCK_INTERVAL == 0
        )
        if unlocking and len(town["unlocked_shops"]) < MAX_SHOP_INSTANCES:
            town["unlocked_shops"].append(next(drawn))
        official._town_consume(environment, state, now)
    consumed = {item: MARKET_I0 - market["inventory"][item] for item in PRODUCTS}
    return consumed, len(town["unlocked_shops"]) - len(shops)


@pytest.mark.parametrize(
    ("shops", "step"),
    [
        (["PET_CAFE"] * 6, 0),
        (["BAKERY", "PET_CAFE"] * 3, 61),
        (["YARN_STORE"] * 7, 1),
        (["BRUNCH_SPOT"], 620),
        (["SMOOTHIE_SHOP"] * 4 + ["FARMERS_MARKET"] * 4, 575),
        (["PET_CAFE"], EPISODE_STEPS - 4),
        ([], EPISODE_STEPS - 2),
    ],
)
def test_town_draw_is_the_mean_engine_consumption_over_every_unlock(
    shops: list[str], step: int
) -> None:
    """Each draw, in eighths, is the engine's consumption averaged over the unlocks.

    Every sequence of equally likely shops the town could open in the window
    runs through the engine's `_town_consume`.
    """
    observation = {
        "player": 0,
        "farms": [{}, {}],
        "step": step,
        "day": step // TURNS_PER_DAY,
        "town": {"unlocked_shops": shops},
    }
    outlook = market_outlook(observation)
    windows = {
        min(step + OUTLOOK_SOON_STEPS, EPISODE_STEPS - 1): outlook.draw_soon,
        EPISODE_STEPS - 1: outlook.draw_to_end,
    }
    for stop, draw in windows.items():
        _, unlocks = _consumed_by_the_town(shops, step, stop, SHOP_NAMES * MAX_SHOP_INSTANCES)
        total = dict.fromkeys(PRODUCTS, 0)
        for drawn in itertools.product(SHOP_NAMES, repeat=unlocks):
            consumed, _ = _consumed_by_the_town(shops, step, stop, drawn)
            for item in PRODUCTS:
                total[item] += consumed[item]
        # `total` sums len(SHOP_NAMES) ** unlocks draws; `draw` is their mean
        # in 1 / len(SHOP_NAMES) parts.
        expected = {
            item: total[item] * len(SHOP_NAMES) // len(SHOP_NAMES) ** unlocks for item in PRODUCTS
        }
        assert all(
            total[item] * len(SHOP_NAMES) % len(SHOP_NAMES) ** unlocks == 0 for item in PRODUCTS
        )
        assert draw == expected, f"draw in [{step}, {stop}) over {unlocks} unlocks"


def test_town_draw_is_what_a_real_town_takes_between_unlocks() -> None:
    """Where no shop opens, the draw is exactly the fall of the market's inventory.

    Both seats pass, so only the town moves the market.
    """
    environment = make("kaggriculture", configuration={"seed": 3})
    environment.run(["pass", "pass"])
    observations = [state[0].observation for state in environment.steps]
    assert len(observations) == EPISODE_STEPS
    exact = 0
    for step in range(EPISODE_STEPS - 1):
        outlook = market_outlook(observations[step])
        shops = observations[step]["town"]["unlocked_shops"]
        windows = {
            min(step + OUTLOOK_SOON_STEPS, EPISODE_STEPS - 1): outlook.draw_soon,
            EPISODE_STEPS - 1: outlook.draw_to_end,
        }
        for stop, draw in windows.items():
            if observations[stop]["town"]["unlocked_shops"] != shops:
                continue
            before = observations[step]["market"]["inventory"]
            after = observations[stop]["market"]["inventory"]
            fall = {item: (before[item] - after[item]) * len(SHOP_NAMES) for item in PRODUCTS}
            assert draw == fall, f"draw in [{step}, {stop})"
            exact += 1
    # Soon windows between unlocks, and every window once all eight are open.
    assert exact > EPISODE_STEPS // 2


def _legacy_v6_columns(observation: dict, opponent_private: dict) -> tuple[np.ndarray, np.ndarray]:
    """The schema-v6 product rows and critic product columns, verbatim, before v7."""
    market = observation.get("market") or {}
    inventory = market.get("inventory") or {}

    def proceeds(private: dict) -> list[float]:
        shed = private.get("shed") or {}
        carried = {
            item: sum(int(unit.get(item, 0) or 0) for unit in private.get("inventories") or [])
            for item in PRODUCTS
        }
        return [
            sale_proceeds(
                item,
                int(shed.get(item, 0) or 0) + carried[item],
                int(inventory.get(item, MARKET_I0) or 0),
                market.get("params"),
            )
            / HELD_VALUE_SCALE
            for item in PRODUCTS
        ]

    own = np.column_stack(
        (_legacy_v5_product_rows(observation), np.asarray(proceeds(observation["private"])))
    ).astype(np.float32)
    shed = opponent_private.get("shed") or {}
    critic = np.asarray(
        [
            (
                float(shed.get(item, 0) or 0) / SHED_CAPACITY,
                sum(int(unit.get(item, 0) or 0) for unit in opponent_private["inventories"])
                / SHED_CAPACITY,
                value,
            )
            for item, value in zip(PRODUCTS, proceeds(opponent_private), strict=True)
        ],
        dtype=np.float32,
    )
    return own, critic


def test_v6_columns_are_unchanged_by_the_v7_outlook() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 240, "seed": 11})
    environment.run(["starter", "starter"])
    v6_products = len(product_token_fields(6))
    v6_private = len(product_private_fields(6))
    supplied = 0
    for step_state in environment.steps[::3]:
        for seat in (0, 1):
            observation = step_state[seat].observation
            opponent_private = step_state[1 - seat].observation.private
            products, critic = _legacy_v6_columns(observation, opponent_private)
            economy = tokenize_economy(observation)
            assert economy.products[:, :v6_products].tobytes() == products.tobytes()
            staged = encode_structured_observation(observation, opponent_private)
            assert staged.products[:, :v6_products].tobytes() == (
                products.astype(np.float16).tobytes()
            )
            assert staged.critic_products[:, :v6_private].tobytes() == (
                critic.astype(np.float16).tobytes()
            )
            supplied += bool(economy.products[:, v6_products:].any())
    # The starter plants, so the v7 columns were live beside the v6 ones.
    assert supplied > 100


def _legacy_v7_crop_and_animal_rows(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    """The schema-v7 crop and animal rows, verbatim, before v8."""
    private = observation["private"]
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    carried = {
        item: sum(int(unit.get(item, 0) or 0) for unit in private.get("inventories") or [])
        for item in ANIMALS
    }
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
    return crops, animals


def test_v7_columns_are_unchanged_by_the_v8_paybacks() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 240, "seed": 11})
    environment.run(["starter", "starter"])
    v7_crops = len(crop_token_fields(7))
    v7_animals = len(animal_token_fields(7))
    for step_state in environment.steps[::3]:
        for seat in (0, 1):
            observation = step_state[seat].observation
            crops, animals = _legacy_v7_crop_and_animal_rows(observation)
            economy = tokenize_economy(observation)
            assert economy.crops[:, :v7_crops].tobytes() == crops.tobytes()
            assert economy.animals[:, :v7_animals].tobytes() == animals.tobytes()
            staged = encode_structured_observation(
                observation, step_state[1 - seat].observation.private
            )
            assert staged.crops[:, :v7_crops].tobytes() == crops.astype(np.float16).tobytes()
            assert staged.animals[:, :v7_animals].tobytes() == (
                animals.astype(np.float16).tobytes()
            )
            # The run covers the first ten days, where every start pays back.
            assert economy.crops[:, v7_crops:].all()
            assert economy.animals[:, v7_animals:].all()


def test_forecast_price_quotes_the_market_once_supply_sells_and_the_town_draws() -> None:
    tomato = {
        "kind": "PLANT",
        "crop": "TOMATO",
        "planted_day": 20,
        "yield_units": 1,
        "watered_today": True,
        "fertilized_until_day": 29,
        "max_lifespan_step": -1,
    }
    goose = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 3,
        "yield_units": 2,
        "fed_today": True,
        "cared_today": True,
        "fertilizer_available": True,
        "pending_care_bonus": 1,
    }
    observation = {
        "player": 0,
        "step": 650,
        "day": 27,
        "hour": 2,
        "farms": [_blank_farm(tomato), _blank_farm(goose, x=6, y=1)],
        "market": {"inventory": {"TOMATO": 10_060, "EGG": 9_950, "STRAWBERRY": 10_050}},
        "private": {
            "shed": {"TOMATO": 5, "MELON": 3, "STRAWBERRY": 30},
            "inventories": [{"EGG": 4}],
        },
        "town": {"unlocked_shops": ["PIZZA_SHOP", "BAKERY"] * 4},
    }
    opponent_private = {"shed": {"EGG": 30}, "inventories": [{}]}
    forecast = PRODUCT_TOKEN_FIELDS.index("forecast_price")
    tokens = tokenize_economy(observation)
    outlook = market_outlook(observation)
    # The standing tomato, then the pickings of days 28 and 29 (fertilized the
    # day before; day 30 is past the end), and on the other farm the goose's
    # standing eggs, then day 28's with the pending care bonus and day 29's
    # with today's care.
    assert outlook.supply_to_end[0]["TOMATO"] == 1 + 2 + 2
    assert outlook.supply_to_end[1]["EGG"] == 2 + (1 + 1) + (1 + 1)
    assert outlook.supply_to_end[1]["FERTILIZER"] == 1 + 2
    # Fed today, the goose eats once more, on day 28; day 29's feed keeps no
    # production.
    assert outlook.feeds_to_end == 1
    floored = False
    for index, item in enumerate(PRODUCTS):
        held = {"TOMATO": 5, "MELON": 3, "EGG": 4, "STRAWBERRY": 30}.get(item, 0)
        inventory = observation["market"]["inventory"].get(item, MARKET_I0)
        supply = outlook.supply_to_end[0][item] + outlook.supply_to_end[1][item]
        # Sales stop restocking the market once its quote is at the floor. A
        # log curve (WHEAT, EGG) reaches it only near 1e12 units, so search by
        # doubling and bisection rather than one unit at a time.
        high = max(inventory, 1)
        while market_price(item, high) > 1:
            high *= 2
        low = max(inventory, high // 2)
        while low < high:
            middle = (low + high) // 2
            low, high = (low, middle) if market_price(item, middle) <= 1 else (middle + 1, high)
        floor = low
        floored |= inventory + held + supply > floor
        level = min(inventory + held + supply, floor) - outlook.feeds_to_end * (item == "WHEAT")
        # Every shop instance is open, so the town's draw is certain units.
        assert outlook.draw_to_end[item] % len(SHOP_NAMES) == 0
        level -= outlook.draw_to_end[item] // len(SHOP_NAMES)
        expected = market_price(item, level) / (2.0 * BASE_PRICE[item])
        assert tokens.products[index, forecast] == np.float32(expected), item
        for column, value in (
            ("supply_soon", outlook.supply_soon[0][item] / SHED_CAPACITY),
            ("supply_to_end", outlook.supply_to_end[0][item] / SHED_CAPACITY),
            ("opponent_supply_soon", outlook.supply_soon[1][item] / SHED_CAPACITY),
            ("opponent_supply_to_end", outlook.supply_to_end[1][item] / SHED_CAPACITY),
            ("town_draw_soon", outlook.draw_soon[item] / (len(SHOP_NAMES) * SHED_CAPACITY)),
            ("town_draw_to_end", outlook.draw_to_end[item] / (len(SHOP_NAMES) * SHED_CAPACITY)),
        ):
            assert tokens.products[index, PRODUCT_TOKEN_FIELDS.index(column)] == np.float32(value)
    # The strawberries sold past the floor.
    assert floored

    # The opponent's forecast sells its own stock instead; the critic reads it
    # as that seat's own column, and the actor's tokens never move with it.
    staged = encode_structured_observation(observation, opponent_private)
    blind = encode_structured_observation(observation, {"shed": {}, "inventories": [{}]})
    for name in ("products", "animals", "crops", "farms", "town"):
        np.testing.assert_array_equal(getattr(staged, name), getattr(blind, name))
    opponent_view = {
        **observation,
        "player": 1,
        "private": {**opponent_private, "seeds": {}},
    }
    opponent_forecast = staged.critic_products[
        :, PRODUCT_PRIVATE_FIELDS.index("opponent_forecast_price")
    ]
    np.testing.assert_array_equal(
        opponent_forecast, encode_structured_observation(opponent_view).products[:, forecast]
    )
    # Its thirty eggs to this seat's four forecast a lower egg price.
    egg = PRODUCTS.index("EGG")
    assert opponent_forecast[egg] < staged.products[egg, forecast]


def test_forecast_price_at_the_last_step_is_the_quote_after_selling_what_is_held() -> None:
    # No shop sells at the last acting step, which is off the sell interval.
    step = EPISODE_STEPS - 2
    assert step % TOWN_SHOP_SELL_INTERVAL and step % TOWN_CENTER_SELL_INTERVAL
    observation = {
        "player": 1,
        "step": step,
        "day": step // TURNS_PER_DAY,
        "hour": step % TURNS_PER_DAY,
        "farms": [{}, {}],
        "market": {"inventory": {"WOOL": 10_040}},
        "private": {"shed": {"WOOL": 12}, "inventories": [{"WOOL": 2}, {"MILK": 1}]},
        "town": {"unlocked_shops": ["YARN_STORE"] * 8},
    }
    tokens = tokenize_economy(observation)
    forecast = PRODUCT_TOKEN_FIELDS.index("forecast_price")
    for index, item in enumerate(PRODUCTS):
        held = {"WOOL": 14, "MILK": 1}.get(item, 0)
        level = observation["market"]["inventory"].get(item, MARKET_I0) + held
        expected = market_price(item, level) / (2.0 * BASE_PRICE[item])
        assert tokens.products[index, forecast] == np.float32(expected), item
    assert not tokens.products[:, forecast + 1 :].any()


def _tender(observation: dict, rotation: int) -> tuple[list[int], list[tuple[int, int]]]:
    """A farmer that grows every crop and keeps geese: its unit and market orders.

    Built to put every tile state the outlook reads on a real board: it keeps
    seeds, geese, feed and fertilizer in stock, fertilizes ongoing crops,
    cares for its geese, and waters and harvests on a greedy walk, so crops
    are sometimes left to wait, dry out, or decay.
    """
    from kaggriculture.actions import MarketKind

    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    shed, seeds = private["shed"], private["seeds"]
    carried = private["inventories"][0]
    day = observation["day"]
    tiles = farm["tiles"]
    x, y = farm["farmer"]
    coops = {(0, 0), (1, 0)}
    geese = sum(isinstance(tiles[j][i], dict) and "animal" in tiles[j][i] for i, j in coops)

    orders = []
    money = farm["money"]
    for offset in range(len(CROPS)):
        crop = CROPS[(rotation + offset) % len(CROPS)]
        if seeds[crop] == 0 and money > 500:
            orders.append((MarketKind[f"BUY_SEED_{crop}"], 1))
    owned = geese + shed["GOOSE"] + carried.get("GOOSE", 0)
    if owned < len(coops) and money > 1_000:
        orders.append((MarketKind.BUY_ANIMAL_GOOSE, 0))
    if shed["WHEAT"] < 4:
        orders.append((MarketKind.BUY_PRODUCT_WHEAT, 3))
    if shed["FERTILIZER"] < 2 and day % 3 == rotation % 3:
        orders.append((MarketKind.BUY_PRODUCT_FERTILIZER, 1))
    for item in ("TOMATO", "STRAWBERRY", "MELON", "CARROT", "EGG"):
        if shed[item] > 6:
            orders.append((MarketKind[f"SELL_{item}"], shed[item] - 1))

    def work(i: int, j: int) -> UnitAction | None:
        tile = tiles[j][i]
        if (i, j) == (4, 4):
            if carried.get("WHEAT", 0) < 2 and shed["WHEAT"] >= 2:
                return UnitAction.PICKUP_WHEAT_2
            if not carried.get("FERTILIZER") and shed["FERTILIZER"]:
                return UnitAction.PICKUP_FERTILIZER_1
            if shed["GOOSE"] and not carried.get("GOOSE") and geese < len(coops):
                return UnitAction.PICKUP_GOOSE_1
            if any(carried.get(item) for item in PRODUCTS if item not in ("WHEAT", "FERTILIZER")):
                return UnitAction.DROP
            return None
        if tile is None:
            if (i, j) in coops:
                return UnitAction.BUILD_COOP
            for offset in range(len(CROPS)):
                crop = CROPS[(rotation + i + 2 * j + offset) % len(CROPS)]
                if seeds[crop]:
                    return UnitAction[f"PLANT_{crop}"]
            return None
        if not isinstance(tile, dict):
            return None
        if tile.get("kind") == "PLANT":
            crop = tile["crop"]
            age = day - tile["planted_day"]
            if not tile["watered_today"] and (i + j + day) % 7:
                return UnitAction.WATER
            ripe = age >= CROP_FIRST_YIELD_DAY[crop] and tile["yield_units"] > 0
            if ripe and (crop in ONGOING_CROPS or age >= CROP_MAX_YIELD_DAY[crop] + (i % 2)):
                return UnitAction.HARVEST
            fertilizable = crop in ONGOING_CROPS or age < CROP_MAX_YIELD_DAY[crop]
            if fertilizable and tile["fertilized_until_day"] < day and carried.get("FERTILIZER"):
                return UnitAction.FERTILIZE
            return None
        if tile.get("kind") == "WEED":
            return UnitAction.DIG
        if not tile.get("animal"):
            return UnitAction.PLACE_GOOSE if carried.get("GOOSE") else None
        if not tile["fed_today"] and carried.get("WHEAT"):
            return UnitAction.FEED
        if tile["yield_units"] > 1:
            return UnitAction.HARVEST
        if tile["fertilizer_available"]:
            return UnitAction.COLLECT_FERTILIZER
        if not tile["cared_today"] and day % 2:
            return UnitAction.CARE
        return None

    here = work(x, y)
    if here is not None:
        return [int(here)], orders
    # Feeding and restocking come first, so the geese live to bank care.
    urgent = (UnitAction.FEED, UnitAction.PICKUP_WHEAT_2, UnitAction.PLACE_GOOSE)
    targets = [
        (work(i, j) not in urgent, abs(i - x) + abs(j - y), i, j)
        for j in range(BOARD_SIZE // 2)
        for i in range(BOARD_SIZE // 2)
        if (i, j) != (x, y) and work(i, j) is not None
    ]
    if not targets:
        return [int(UnitAction.PASS)], orders
    _, _, i, j = min(targets)
    if i != x:
        move = UnitAction.EAST if i > x else UnitAction.WEST
    else:
        move = UnitAction.SOUTH if j > y else UnitAction.NORTH
    return [int(move)], orders


def test_native_outlook_tokens_match_python_on_every_crop_and_animal() -> None:
    """Both tokenizers agree on the v7 outlook and v8 paybacks as two farms grow everything.

    Scripted-v27 grows no tomato, carrot, or goose; two `_tender` farms grow
    all of them, fertilize and care, so every branch of the yield model and
    the town's unlocks are read through both tokenizers, and the paybacks
    through every quote and forecast the game passes.
    """
    import json

    from kaggriculture.constants import MAX_MARKET_ORDERS
    from kaggriculture.rollout import (
        _ANIMAL_STOCK_COLUMNS,
        _CROP_SEED_COLUMNS,
        _PRODUCT_STOCK_COLUMNS,
    )
    from kaggriculture.rust_env import load_native
    from kaggriculture.script_opponents import shaped_observation

    environment = load_native().BatchEnv(np.asarray([17], dtype=np.uint64))
    outlook = slice(PRODUCT_TOKEN_FIELDS.index("forecast_price"), len(PRODUCT_TOKEN_FIELDS))
    occupants: set[str] = set()
    live = np.zeros((len(PRODUCTS), outlook.stop - outlook.start), dtype=bool)
    states = dict.fromkeys(("fertilized_ongoing", "pending_care", "cared", "expired"), 0)
    payback = {
        name: slice(fields.index("payback"), len(fields))
        for name, fields in (("crops", CROP_TOKEN_FIELDS), ("animals", ANIMAL_TOKEN_FIELDS))
    }
    paybacks: dict[str, set[tuple[float, ...]]] = {name: set() for name in payback}
    for step in range(EPISODE_STEPS - 1):
        snapshot = json.loads(environment.snapshot_json(0))
        units = np.zeros((1, 2, MAX_UNITS), dtype=np.uint8)
        kinds = np.zeros((1, 2, MAX_MARKET_ORDERS), dtype=np.uint8)
        quantities = np.zeros((1, 2, MAX_MARKET_ORDERS), dtype=np.uint8)
        native = environment.structured() if step % 3 == 0 else None
        for seat in (0, 1):
            observation = shaped_observation(snapshot, seat)
            unit_actions, orders = _tender(observation, rotation=2 * seat)
            units[0, seat, : len(unit_actions)] = unit_actions
            for slot, (kind, quantity) in enumerate(orders[:MAX_MARKET_ORDERS]):
                kinds[0, seat, slot] = kind
                quantities[0, seat, slot] = min(quantity, 99)
            if native is None:
                continue
            python = encode_structured_observation(observation, snapshot["privates"][1 - seat])
            for name in ("products", "animals", "crops", "farms", "town"):
                np.testing.assert_array_equal(
                    getattr(python, name),
                    np.asarray(native[name])[seat],
                    err_msg=f"Python/native divergence at step {step}: {name}",
                )
            for name, columns in (
                ("products", _PRODUCT_STOCK_COLUMNS),
                ("animals", _ANIMAL_STOCK_COLUMNS),
                ("crops", _CROP_SEED_COLUMNS),
            ):
                np.testing.assert_array_equal(
                    getattr(python, f"critic_{name}"),
                    np.asarray(native[name])[1 - seat][:, columns],
                    err_msg=f"critic {name} divergence at step {step}",
                )
            for name, columns in payback.items():
                paybacks[name].add(tuple(getattr(python, name)[:, columns].ravel().tolist()))
            live |= python.products[:, outlook] != 0
            for row in observation["farms"][seat]["tiles"]:
                for tile in row:
                    if not isinstance(tile, dict):
                        continue
                    occupants.add(tile.get("animal") or tile.get("crop") or tile["kind"])
                    ongoing = tile.get("crop") in ONGOING_CROPS
                    fertilized = tile.get("fertilized_until_day", -1) >= observation["day"]
                    states["fertilized_ongoing"] += ongoing and fertilized
                    states["pending_care"] += bool(tile.get("pending_care_bonus"))
                    states["cared"] += bool(tile.get("cared_today"))
                    states["expired"] += 0 <= tile.get("max_lifespan_step", -1) <= step
        environment.step_factors(units, kinds, quantities)
    assert occupants >= {*CROPS, "GOOSE"}, occupants
    assert min(states.values()) > 0, states
    # Every product's forecast, supply, and draw columns were exercised, but
    # for supply of what these farms never keep and the draw of fertilizer,
    # which the town never takes.
    expected = np.ones_like(live)
    columns = PRODUCT_TOKEN_FIELDS[outlook]
    for item in ("MILK", "WOOL"):
        for column, name in enumerate(columns):
            expected[PRODUCTS.index(item), column] = "supply" not in name
    for column, name in enumerate(columns):
        if name.startswith("town_draw"):
            expected[PRODUCTS.index("FERTILIZER"), column] = False
    np.testing.assert_array_equal(live, expected)
    # Paybacks moved with the quotes all game and reached zero by its end.
    for name, seen in paybacks.items():
        assert len(seen) > EPISODE_STEPS // 6, (name, len(seen))
        assert (0.0,) * len(next(iter(seen))) in seen, name
