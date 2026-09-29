from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from kaggle_environments import make

from kaggriculture.actions import UnitAction, apply_unit_shed_effect
from kaggriculture.constants import (
    ANIMAL_COST,
    ANIMALS,
    BASE_PRICE,
    BOARD_SIZE,
    EPISODE_STEPS,
    MAX_UNITS,
    PRIVATE_ITEMS,
    PRODUCTS,
    SHED_CAPACITY,
    SHOP_NAMES,
    TURNS_PER_DAY,
    market_price,
    sale_proceeds,
    shed_access_tiles,
)
from kaggriculture.encoding import encode_observation, liquidation_value
from kaggriculture.tokens import (
    ANIMAL_TOKEN_FIELDS,
    FARM_IDENTITIES,
    FARM_TOKEN_FIELDS,
    HELD_VALUE_SCALE,
    N_TILE_CATEGORICAL,
    N_TILE_CONTINUOUS,
    OBSERVATION_SCHEMA_VERSION,
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
    clock_features,
    encode_structured_observation,
    farm_token_fields,
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
    assert {3, 4, 5, 6} == SUPPORTED_OBSERVATION_SCHEMA_VERSIONS
    assert farm_token_fields(OBSERVATION_SCHEMA_VERSION) == FARM_TOKEN_FIELDS
    assert farm_token_fields(6) == (
        *farm_token_fields(5),
        "liquidation",
        "liquidation_margin",
    )
    assert farm_token_fields(5) == farm_token_fields(4)
    assert farm_token_fields(4) == (*farm_token_fields(3), "money_margin")
    assert farm_token_fields(3) == ("money", "unlocked_quadrants", "hands", "hires_today")
    for version in (2, 7):
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
    assert town_token_fields(6) == town_token_fields(5) == (
        *legacy,
        *(f"shop_{name}_first_unlock" for name in SHOP_NAMES),
    )
    for version in (2, 7):
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
    assert product_private_fields(OBSERVATION_SCHEMA_VERSION) == PRODUCT_PRIVATE_FIELDS
    for version in (2, 7):
        with pytest.raises(ValueError, match="unsupported observation schema"):
            product_token_fields(version)
        with pytest.raises(ValueError, match="unsupported observation schema"):
            product_private_fields(version)


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
