from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar

import pytest


def _script():
    path = Path(__file__).parents[1] / "scripts" / "audit_agent_behavior.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_audit_agent_behavior", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_farm_summary_counts_the_route_the_agent_ended_holding() -> None:
    module = _script()
    tiles = [
        [
            {"kind": "PLANT", "crop": "WHEAT"},
            {"kind": "PLANT", "crop": "MELON"},
            {"kind": "ANIMAL", "animal": "COW"},
        ],
        [
            {"kind": "ANIMAL", "animal": "COW"},
            {"kind": "PASTURE"},
            {"kind": "SOIL"},
            None,
        ],
    ]

    summary = module._final_farm_summary(
        {
            "money": 145_000,
            "unlocked_quadrants": ["NW", "NE", "SW"],
            "hands": [{}, {}, {}],
            "tiles": tiles,
        }
    )

    assert summary["money"] == 145_000.0
    assert summary["unlocked_quadrants"] == 3
    assert summary["units"] == 4
    assert summary["animal_tiles"] == {"GOOSE": 0, "COW": 2, "SHEEP": 0}
    assert summary["planted_tiles"]["WHEAT"] == 1
    assert summary["planted_tiles"]["MELON"] == 1
    assert summary["planted_tiles"]["CARROT"] == 0
    assert summary["structure_tiles"] == {"PASTURE": 1}


def test_action_counts_separate_the_seats_and_keep_order_operands() -> None:
    """Two market kinds that share a verb must not collapse into one count."""
    module = _script()
    steps = [
        [
            {
                "action": {
                    "farmer": ["PLANT", "WHEAT"],
                    "hands": [["PASS"]],
                    "market": [["BUY_LAND"], ["BUY_ANIMAL", "COW", 2], ["SELL", "MILK", 5]],
                }
            },
            {"action": {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}},
        ],
        [
            {"action": {"farmer": ["WATER"], "hands": [["PLANT", "MELON"]], "market": []}},
            {"action": None},
        ],
    ]

    first = module._episode_action_counts(steps, 0)
    second = module._episode_action_counts(steps, 1)

    assert first["market_orders"] == {"BUY_ANIMAL_COW": 1, "BUY_LAND": 1, "SELL_MILK": 1}
    assert first["unit_commands"] == {"PASS": 1, "PLANT": 2, "WATER": 1}
    assert second["market_orders"] == {"HIRE": 1}
    assert second["unit_commands"] == {"PASS": 1}


def test_summary_aggregates_every_audited_seat() -> None:
    module = _script()
    episodes = [
        {
            "seed": 0,
            "seats": [
                {
                    "money": 100.0,
                    "unlocked_quadrants": 2,
                    "units": 4,
                    "animal_tiles": {"COW": 3},
                    "planted_tiles": {"WHEAT": 5},
                    "market_orders": {"BUY_LAND": 1},
                    "unit_commands": {"PLANT": 5},
                },
                {
                    "money": 300.0,
                    "unlocked_quadrants": 4,
                    "units": 2,
                    "animal_tiles": {"COW": 1},
                    "planted_tiles": {"WHEAT": 1},
                    "market_orders": {"BUY_LAND": 3},
                    "unit_commands": {"PLANT": 1},
                },
            ],
        }
    ]

    summary = module.summarize(episodes)

    assert summary["audited_seats"] == 2
    assert (summary["mean_bank"], summary["median_bank"]) == (200.0, 200.0)
    assert (summary["min_bank"], summary["max_bank"]) == (100.0, 300.0)
    assert summary["mean_unlocked_quadrants"] == 3.0
    assert summary["mean_units"] == 3.0
    assert summary["mean_animal_tiles"] == 2.0
    assert summary["mean_planted_tiles"] == 3.0
    assert summary["market_orders_per_seat"] == {"BUY_LAND": 2.0}
    assert summary["unit_commands_per_seat"] == {"PLANT": 3.0}

    with pytest.raises(ValueError, match="no seat records"):
        module.summarize([{"seed": 0, "seats": []}])


def test_mirror_audit_covers_both_seats_and_an_external_one_only_ours(monkeypatch) -> None:
    module = _script()
    played: list[list[object]] = []

    class FakeEnvironment:
        steps: ClassVar[list] = [
            [
                {
                    "observation": {"farms": [{"money": 1}, {"money": 2}]},
                    "action": {"farmer": ["PASS"], "hands": [], "market": []},
                },
                {"action": {"farmer": ["PASS"], "hands": [], "market": []}},
            ]
        ]

        def run(self, players):
            played.append(players)

    import sys
    import types

    fake = types.ModuleType("kaggle_environments")
    fake.make = lambda *args, **kwargs: FakeEnvironment()
    monkeypatch.setitem(sys.modules, "kaggle_environments", fake)

    mirror = module.audit_episode(
        object(),
        seed=0,
        mode="deterministic",
        episode_steps=720,
        temperature=1.0,
        opponent=None,
        candidate_seat=0,
    )
    assert [record["seat"] for record in mirror["seats"]] == [0, 1]
    assert all(callable(player) for player in played[-1])

    for candidate_seat in (0, 1):
        versus = module.audit_episode(
            object(),
            seed=0,
            mode="deterministic",
            episode_steps=720,
            temperature=1.0,
            opponent="/tmp/v27.py",
            candidate_seat=candidate_seat,
        )
        assert [record["seat"] for record in versus["seats"]] == [candidate_seat]
        assert played[-1][1 - candidate_seat] == "/tmp/v27.py"
        assert callable(played[-1][candidate_seat])
