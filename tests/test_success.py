from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggriculture.success import (
    PRIMARY_OPPONENT,
    IterationSuccess,
    MemberSuccess,
    OpponentProbe,
    canonical_opponent,
    latest_valid,
    load_external_journal,
    peak,
    rank_runs,
)


def _probe(
    opponent: str, money: float, score: float, *, games: int = 8, completed: int | None = None
) -> OpponentProbe:
    done = games if completed is None else completed
    return OpponentProbe(
        opponent=opponent,
        money_mean=money,
        opponent_money_mean=70_000.0,
        score_rate=score,
        games=games,
        completed_games=done,
    )


def _member(
    iteration: int,
    agent: int,
    v27_money: float,
    v27_score: float = 1.0,
    starter: float = 150_000.0,
) -> MemberSuccess:
    return MemberSuccess(
        iteration=iteration,
        agent=agent,
        probes={
            "starter": _probe("starter", starter, 1.0),
            PRIMARY_OPPONENT: _probe(PRIMARY_OPPONENT, v27_money, v27_score),
        },
    )


def _write_journal(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(
    iteration: int,
    opponent: str,
    money: float,
    score: float,
    *,
    agent: int | None = 0,
    games: int = 8,
    completed: int | None = None,
) -> dict[str, object]:
    return {
        "event": "external_eval",
        "iteration": iteration,
        "agent": agent,
        "opponent": opponent,
        "money_mean": money,
        "opponent_money_mean": 70_000.0,
        "score_rate": score,
        "games": games,
        "completed_games": games if completed is None else completed,
    }


def test_aliases_collapse_onto_the_panel() -> None:
    assert canonical_opponent("public-v27") == PRIMARY_OPPONENT
    assert canonical_opponent("scripted-v27") == PRIMARY_OPPONENT
    assert canonical_opponent("public-v16") == "v16"
    assert canonical_opponent("pass") is None


def test_incomplete_v27_probe_is_not_valid() -> None:
    member = MemberSuccess(
        iteration=10,
        agent=0,
        probes={PRIMARY_OPPONENT: _probe(PRIMARY_OPPONENT, 80_000.0, 1.0, completed=7)},
    )
    assert not member.valid


def test_missing_v27_probe_is_not_valid() -> None:
    member = MemberSuccess(
        iteration=10,
        agent=0,
        probes={"starter": _probe("starter", 150_000.0, 1.0)},
    )
    assert not member.valid


def test_population_success_is_best_member_v27_money() -> None:
    snapshot = IterationSuccess(
        iteration=50,
        members=(
            _member(50, 0, 81_000.0),
            _member(50, 1, 95_000.0, v27_score=1.0),
            _member(50, 2, 67_000.0, v27_score=0.625),
            _member(50, 3, 12_000.0, v27_score=0.375),
        ),
    )
    assert snapshot.best.agent == 1
    assert snapshot.best.v27_money == 95_000.0
    assert snapshot.worst.agent == 3
    assert snapshot.ranking_key[0] == 95_000.0
    assert snapshot.ranking_key[2] == 12_000.0


def test_saturated_starter_score_does_not_outrank_higher_v27_bank() -> None:
    weak = IterationSuccess(iteration=40, members=(_member(40, 0, 20_000.0, starter=150_000.0),))
    strong = IterationSuccess(iteration=40, members=(_member(40, 0, 82_000.0, starter=140_000.0),))
    assert strong.ranking_key > weak.ranking_key


def test_journal_skips_off_panel_rows_and_keeps_null_agent_as_zero(tmp_path: Path) -> None:
    journal = _write_journal(
        tmp_path / "metrics-external.jsonl",
        [
            _row(10, "starter", 155_000.0, 1.0, agent=None),
            _row(10, "public-v27", 83_000.0, 1.0, agent=None),
            _row(10, "pass", 3_000.0, 1.0, agent=None),
        ],
    )
    snapshots = load_external_journal(journal)
    assert len(snapshots) == 1
    member = snapshots[0].members[0]
    assert member.agent == 0
    assert member.v27_money == 83_000.0
    assert "pass" not in member.probes


def test_latest_valid_ignores_a_truncated_tick(tmp_path: Path) -> None:
    journal = _write_journal(
        tmp_path / "metrics-external.jsonl",
        [
            _row(10, "public-v27", 80_000.0, 1.0),
            _row(20, "public-v27", 90_000.0, 1.0, completed=3),
        ],
    )
    snapshots = load_external_journal(journal)
    assert latest_valid(snapshots).iteration == 10
    assert peak(snapshots).best.v27_money == 80_000.0


def test_external_journal_preserves_complete_rows_before_a_torn_tail(tmp_path: Path) -> None:
    journal = _write_journal(
        tmp_path / "metrics-external.jsonl",
        [_row(10, "public-v27", 80_000.0, 1.0)],
    )
    with journal.open("a", encoding="utf-8") as stream:
        stream.write('{"iteration": 20, "opponent":')

    snapshots = load_external_journal(journal)

    assert latest_valid(snapshots).iteration == 10


def test_peak_can_precede_latest(tmp_path: Path) -> None:
    journal = _write_journal(
        tmp_path / "metrics-external.jsonl",
        [
            _row(10, "public-v27", 90_000.0, 1.0),
            _row(20, "public-v27", 40_000.0, 0.25),
        ],
    )
    snapshots = load_external_journal(journal)
    assert peak(snapshots).iteration == 10
    assert latest_valid(snapshots).iteration == 20


def test_rank_runs_orders_by_latest_v27_money(tmp_path: Path) -> None:
    alive = _write_journal(
        tmp_path / "alive.jsonl",
        [_row(50, "public-v27", 82_000.0, 0.875), _row(50, "starter", 150_000.0, 1.0)],
    )
    dead = _write_journal(
        tmp_path / "dead.jsonl",
        [_row(80, "public-v27", 1_000.0, 0.0), _row(80, "starter", 20_000.0, 1.0)],
    )
    ranked = rank_runs([dead, alive])
    assert Path(ranked[0]["journal"]).name == "alive.jsonl"
    assert ranked[0]["latest"]["success"] == 82_000.0
    assert ranked[1]["latest"]["success"] == 1_000.0


def test_empty_journal_has_no_success(tmp_path: Path) -> None:
    journal = _write_journal(tmp_path / "empty.jsonl", [])
    with pytest.raises(ValueError, match="no valid v27 probe"):
        latest_valid(load_external_journal(journal))
