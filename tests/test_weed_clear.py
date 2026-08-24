from __future__ import annotations

from kaggriculture.inference import clear_standing_weeds


def _obs(*, farmer, hands=(), tiles) -> dict:
    return {
        "player": 0,
        "farms": [
            {"farmer": list(farmer), "hands": [list(pos) for pos in hands], "tiles": tiles},
            {"farmer": [0, 0], "hands": [], "tiles": tiles},
        ],
    }


def _empty_board() -> list[list[object | None]]:
    return [[None for _ in range(10)] for _ in range(10)]


def test_farmer_on_weed_is_forced_to_dig() -> None:
    tiles = _empty_board()
    tiles[4][4] = {"kind": "WEED"}
    action = {"farmer": ["WEST"], "hands": [], "market": []}

    cleared = clear_standing_weeds(_obs(farmer=(4, 4), tiles=tiles), action)

    assert cleared["farmer"] == ["DIG"]
    assert cleared["market"] == []


def test_hand_on_weed_is_forced_to_dig_without_touching_clear_units() -> None:
    tiles = _empty_board()
    tiles[1][2] = {"kind": "WEED"}
    action = {
        "farmer": ["WATER"],
        "hands": [["EAST"], ["PLANT", "WHEAT"]],
        "market": [["HIRE"]],
    }

    cleared = clear_standing_weeds(
        _obs(farmer=(0, 0), hands=((2, 1), (5, 5)), tiles=tiles),
        action,
    )

    assert cleared["farmer"] == ["WATER"]
    assert cleared["hands"] == [["DIG"], ["PLANT", "WHEAT"]]
    assert cleared["market"] == [["HIRE"]]


def test_empty_and_plant_tiles_are_left_alone() -> None:
    tiles = _empty_board()
    tiles[0][0] = {"kind": "PLANT", "crop": "WHEAT"}
    action = {"farmer": ["HARVEST"], "hands": [["NORTH"]], "market": []}

    cleared = clear_standing_weeds(
        _obs(farmer=(0, 0), hands=((1, 0),), tiles=tiles),
        action,
    )

    assert cleared["farmer"] == ["HARVEST"]
    assert cleared["hands"] == [["NORTH"]]
