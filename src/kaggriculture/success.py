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

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

    @property
    def valid_members(self) -> tuple[MemberSuccess, ...]:
        return tuple(member for member in self.members if member.valid)

    @property
    def best(self) -> MemberSuccess:
        valid = self.valid_members
        if not valid:
            raise ValueError(f"iteration {self.iteration} has no valid v27 probe")
        return max(valid, key=lambda member: member.ranking_key)

    @property
    def worst(self) -> MemberSuccess:
        valid = self.valid_members
        if not valid:
            raise ValueError(f"iteration {self.iteration} has no valid v27 probe")
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


def load_external_journal(path: Path) -> tuple[IterationSuccess, ...]:
    """Fold `metrics-external.jsonl` into per-iteration success snapshots."""
    grouped: dict[tuple[int, int], dict[str, OpponentProbe]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            probe = probe_from_record(record)
            if probe is None:
                continue
            iteration = int(record["iteration"])
            agent = record.get("agent")
            member = 0 if agent is None else int(agent)
            grouped.setdefault((iteration, member), {})[probe.opponent] = probe
    iterations: dict[int, list[MemberSuccess]] = {}
    for (iteration, member), probes in grouped.items():
        iterations.setdefault(iteration, []).append(
            MemberSuccess(iteration=iteration, agent=member, probes=probes)
        )
    return tuple(
        IterationSuccess(
            iteration=iteration,
            members=tuple(sorted(members, key=lambda member: member.agent)),
        )
        for iteration, members in sorted(iterations.items())
    )


def latest_valid(snapshots: Sequence[IterationSuccess]) -> IterationSuccess:
    valid = [snapshot for snapshot in snapshots if snapshot.valid_members]
    if not valid:
        raise ValueError("journal has no valid v27 probe")
    return valid[-1]


def peak(snapshots: Sequence[IterationSuccess]) -> IterationSuccess:
    valid = [snapshot for snapshot in snapshots if snapshot.valid_members]
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
        "valid_ticks": sum(1 for snapshot in snapshots if snapshot.valid_members),
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
