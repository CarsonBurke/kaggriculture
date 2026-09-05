from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from kaggle_environments import make

from kaggriculture.constants import (
    ANIMAL_COST,
    ANIMALS,
    BASE_PRICE,
    BOARD_SIZE,
    MAX_UNITS,
    PRIVATE_ITEMS,
    PRODUCTS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    shed_access_tiles,
)
from kaggriculture.encoding import encode_observation
from kaggriculture.tokens import (
    ANIMAL_TOKEN_FIELDS,
    FARM_IDENTITIES,
    N_TILE_CATEGORICAL,
    N_TILE_CONTINUOUS,
    PRODUCT_TOKEN_FIELDS,
    QUADRANT_COUNT,
    TILE_CONTINUOUS_FIELDS,
    TILE_COUNT,
    TILE_KIND_INDEX,
    TILE_KINDS,
    TILE_OCCUPANT_INDEX,
    TILE_OCCUPANTS,
    UNIT_TILE_GATHERS,
    clock_features,
    encode_structured_observation,
    tokenize_economy,
    tokenize_farm_tiles,
    tokenize_units,
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
