"""Success metric for RL runs: absolute bank against a frozen outside panel.

In-league score rate is 0.5 by symmetry whenever two members of the same
population play each other, whether both banks hold 150,000 or 3,000.
Starter and public-v16 score rates saturate (every healthy clone wins) while
the economy can still collapse. The signal that actually moved when a run
lived or died is mean bank against public-v27.

A probe is valid only when every scheduled game completed with a finite bank.
Incomplete rows do not drop out of the mean — they invalidate the tick.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kaggriculture.telemetry import read_jsonl_snapshot

PRIMARY_OPPONENT = "v27"
FLOOR_OPPONENT = "starter"
PANEL_OPPONENTS = (FLOOR_OPPONENT, "v16", PRIMARY_OPPONENT)

_OPPONENT_ALIASES = {
    "starter": FLOOR_OPPONENT,
    "public-v16": "v16",
    "v16": "v16",
    "public-v27": PRIMARY_OPPONENT,
    "v27": PRIMARY_OPPONENT,
    "scripted-v27": PRIMARY_OPPONENT,
}


def canonical_opponent(label: str) -> str | None:
    """Map a journal opponent label onto the success panel, or None if off-panel."""
    return _OPPONENT_ALIASES.get(label)


@dataclass(frozen=True)
class OpponentProbe:
    """One member's completed probe against one frozen opponent at one iteration."""

    opponent: str
    money_mean: float
    opponent_money_mean: float
    score_rate: float
    games: int
    completed_games: int

    @property
    def valid(self) -> bool:
        return (
            self.games > 0
            and self.completed_games == self.games
            and math.isfinite(self.money_mean)
            and math.isfinite(self.opponent_money_mean)
            and math.isfinite(self.score_rate)
        )


@dataclass(frozen=True)
class MemberSuccess:
    """One population member (or the single learner) at one external-eval tick."""

    iteration: int
    agent: int
    probes: dict[str, OpponentProbe]

    @property
    def valid(self) -> bool:
        probe = self.probes.get(PRIMARY_OPPONENT)
        return probe is not None and probe.valid

    @property
    def v27_money(self) -> float:
        return self.probes[PRIMARY_OPPONENT].money_mean

    @property
    def v27_score(self) -> float:
        return self.probes[PRIMARY_OPPONENT].score_rate

    @property
    def starter_money(self) -> float | None:
        probe = self.probes.get(FLOOR_OPPONENT)
        return probe.money_mean if probe is not None and probe.valid else None

    @property
    def ranking_key(self) -> tuple[float, float, float]:
        """Higher is better. Starter bank is a floor diagnostic, not the objective."""
        starter = self.starter_money
        return (
            self.v27_money,
            self.v27_score,
            -math.inf if starter is None else starter,
        )


@dataclass(frozen=True)
class IterationSuccess:
    """Every member probed at one iteration, with best/worst for the run tick."""

    iteration: int
    members: tuple[MemberSuccess, ...]
    complete: bool = True

    @property
    def valid(self) -> bool:
        return self.complete and bool(self.members) and all(member.valid for member in self.members)

    @property
    def valid_members(self) -> tuple[MemberSuccess, ...]:
        # A population tick is one indivisible comparison. Selecting from the
        # members that happened to land before a crash biases success toward a
        # partial population and disguises the missing work.
        return self.members if self.valid else ()

    @property
    def best(self) -> MemberSuccess:
        valid = self.valid_members
        if not valid:
            raise ValueError(f"iteration {self.iteration} is not a complete valid v27 probe")
        return max(valid, key=lambda member: member.ranking_key)

    @property
    def worst(self) -> MemberSuccess:
        valid = self.valid_members
        if not valid:
            raise ValueError(f"iteration {self.iteration} is not a complete valid v27 probe")
        return min(valid, key=lambda member: member.ranking_key)

    @property
    def ranking_key(self) -> tuple[float, float, float, float, int]:
        best = self.best
        worst = self.worst
        return (
            best.v27_money,
            best.v27_score,
            worst.v27_money,
            -math.inf if best.starter_money is None else best.starter_money,
            best.iteration,
        )


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def probe_from_record(record: Mapping[str, Any]) -> OpponentProbe | None:
    """Build a panel probe from one `external_eval` journal row, or skip it."""
    if record.get("event") != "external_eval":
        return None
    opponent = canonical_opponent(str(record.get("opponent", "")))
    if opponent is None:
        return None
    money = _finite(record.get("money_mean"))
    opponent_money = _finite(record.get("opponent_money_mean"))
    score = _finite(record.get("score_rate"))
    games = int(record.get("games") or 0)
    completed = int(record.get("completed_games") or 0)
    if money is None or opponent_money is None or score is None:
        return OpponentProbe(
            opponent=opponent,
            money_mean=float("nan"),
            opponent_money_mean=float("nan"),
            score_rate=float("nan"),
            games=games,
            completed_games=completed,
        )
    return OpponentProbe(
        opponent=opponent,
        money_mean=money,
        opponent_money_mean=opponent_money,
        score_rate=score,
        games=games,
        completed_games=completed,
    )


def _external_record_valid(record: Mapping[str, Any]) -> bool:
    games = record.get("games")
    return (
        type(games) is int
        and games > 0
        and record.get("completed_games") == games
        and _finite(record.get("money_mean")) is not None
        and _finite(record.get("opponent_money_mean")) is not None
        and _finite(record.get("score_rate")) is not None
    )


def load_external_journal(path: Path) -> tuple[IterationSuccess, ...]:
    """Fold only atomically completed probe matrices into success snapshots."""
    records = read_jsonl_snapshot(path).records
    rows: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    completions: dict[int, Mapping[str, Any]] = {}
    completion_rows: dict[int, dict[tuple[int, int, str], Mapping[str, Any]]] = {}
    iterations: set[int] = set()
    for record in records:
        event = record.get("event")
        if event not in {"external_eval", "external_eval_complete"}:
            continue
        try:
            iteration = int(record["iteration"])
        except (KeyError, TypeError, ValueError):
            continue
        iterations.add(iteration)
        if event == "external_eval_complete":
            completions[iteration] = record
            # Bind the marker to exactly the last-wins rows visible when it was
            # appended. A later interrupted retry or orphan cannot rewrite an
            # already committed matrix through the shared journal.
            completion_rows[iteration] = {
                key: row for key, row in rows.items() if key[0] == iteration
            }
            continue
        agent = record.get("agent")
        try:
            member = 0 if agent is None else int(agent)
        except (TypeError, ValueError):
            continue
        label = str(record.get("opponent", ""))
        rows[(iteration, member, label)] = record

    snapshots: list[IterationSuccess] = []
    for iteration in sorted(iterations):
        completion = completions.get(iteration)
        selected_rows = completion_rows.get(iteration, rows)
        expected_members: list[int] = []
        expected_opponents: list[str] = []
        marker_valid = False
        if completion is not None:
            raw_members = completion.get("members")
            raw_opponents = completion.get("opponents")
            if (
                isinstance(raw_members, list)
                and isinstance(raw_opponents, list)
                and all(
                    member is None or (type(member) is int and member >= 0)
                    for member in raw_members
                )
                and all(type(opponent) is str and opponent for opponent in raw_opponents)
            ):
                expected_members = [0 if member is None else member for member in raw_members]
                expected_opponents = list(raw_opponents)
                marker_valid = bool(
                    expected_members
                    and expected_opponents
                    and len(set(expected_members)) == len(expected_members)
                    and len(set(expected_opponents)) == len(expected_opponents)
                    and completion.get("records") == len(expected_members) * len(expected_opponents)
                )
        artifact = completion.get("artifact") if marker_valid and completion is not None else None
        digest = (
            completion.get("artifact_sha256") if marker_valid and completion is not None else None
        )
        marker_valid = bool(
            marker_valid
            and type(artifact) is str
            and artifact
            and type(digest) is str
            and digest
            and all(
                (row := selected_rows.get((iteration, member, opponent))) is not None
                and row.get("artifact") == artifact
                and row.get("artifact_sha256") == digest
                and _external_record_valid(row)
                for member in expected_members
                for opponent in expected_opponents
            )
        )

        member_ids = (
            expected_members
            if marker_valid
            else sorted({member for tick, member, _opponent in rows if tick == iteration})
        )
        members: list[MemberSuccess] = []
        for member in member_ids:
            probes: dict[str, OpponentProbe] = {}
            labels = (
                expected_opponents
                if marker_valid
                else [
                    label
                    for tick, row_member, label in rows
                    if (tick, row_member) == (iteration, member)
                ]
            )
            for label in labels:
                record = selected_rows.get((iteration, member, label))
                if record is None:
                    continue
                probe = probe_from_record(record)
                if probe is not None:
                    probes[probe.opponent] = probe
            members.append(MemberSuccess(iteration=iteration, agent=member, probes=probes))
        snapshots.append(
            IterationSuccess(
                iteration=iteration,
                members=tuple(sorted(members, key=lambda member: member.agent)),
                complete=marker_valid,
            )
        )
    return tuple(snapshots)


def latest_valid(snapshots: Sequence[IterationSuccess]) -> IterationSuccess:
    if not snapshots:
        raise ValueError("journal has no valid v27 probe")
    latest = snapshots[-1]
    if not latest.valid:
        raise ValueError(
            f"latest external evaluation tick {latest.iteration} is incomplete or invalid"
        )
    return latest


def peak(snapshots: Sequence[IterationSuccess]) -> IterationSuccess:
    valid = [snapshot for snapshot in snapshots if snapshot.valid]
    if not valid:
        raise ValueError("journal has no valid v27 probe")
    return max(valid, key=lambda snapshot: snapshot.ranking_key)


def summarize_run(path: Path) -> dict[str, Any]:
    """Compact comparable record for one training run's external journal."""
    snapshots = load_external_journal(path)
    latest = latest_valid(snapshots)
    best = peak(snapshots)
    return {
        "journal": str(path),
        "ticks": len(snapshots),
        "valid_ticks": sum(1 for snapshot in snapshots if snapshot.valid),
        "latest": _snapshot_record(latest),
        "peak": _snapshot_record(best),
    }


def _member_record(member: MemberSuccess) -> dict[str, Any]:
    record: dict[str, Any] = {
        "agent": member.agent,
        "valid": member.valid,
        "v27_money": member.v27_money if member.valid else None,
        "v27_score": member.v27_score if member.valid else None,
        "starter_money": member.starter_money,
    }
    v16 = member.probes.get("v16")
    if v16 is not None and v16.valid:
        record["v16_money"] = v16.money_mean
        record["v16_score"] = v16.score_rate
    return record


def _snapshot_record(snapshot: IterationSuccess) -> dict[str, Any]:
    best = snapshot.best
    worst = snapshot.worst
    return {
        "iteration": snapshot.iteration,
        "best_agent": best.agent,
        "worst_agent": worst.agent,
        "success": best.v27_money,
        "v27_score": best.v27_score,
        "worst_v27_money": worst.v27_money,
        "starter_money": best.starter_money,
        "members": [_member_record(member) for member in snapshot.members],
    }


def rank_runs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Rank run journals by latest valid success; peak is reported, not ranked."""
    summaries = [summarize_run(path) for path in paths]
    return sorted(
        summaries,
        key=lambda row: (
            float(row["latest"]["success"]),
            float(row["latest"]["v27_score"]),
            float(row["latest"]["worst_v27_money"]),
            -math.inf
            if row["latest"]["starter_money"] is None
            else float(row["latest"]["starter_money"]),
        ),
        reverse=True,
    )


def snapshot_as_dict(snapshot: IterationSuccess) -> dict[str, Any]:
    """JSON-ready snapshot, including the ranking key's public fields."""
    return asdict(snapshot) | _snapshot_record(snapshot)
