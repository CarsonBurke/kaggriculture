"""Group-relative own-bank advantage.

Every reward mode scores a symmetric margin. In mirror self-play both seats run
one policy, so a gain both copies share cancels out of the margin; against an
opponent the learner always loses to, the outcome advantage is about zero. This
stream scores the learner's own final bank instead, against the other games in
the wave played against the same opponent, from the same seat and, where the
wave was collected that way, on the same map seed.

The baseline for one game is the mean bank of the *other* games in its group.
Their actions are independent of this one's given the group, so subtracting it
leaves the policy gradient unbiased (GRPO's leave-one-out estimator).

The native engine's initial state ignores the seed, which drives only the daily
weed draws, yet the seed dominates the bank of a near-deterministic clone. In 64
self-play games of the v16 clone in blocks of four per seed (entropy 0.007
nats), per-seat bank std was 36k over all games against 19-25k within a seed:
an intraclass correlation of 0.57-0.73. Several seeds' four copies finished within
a few hundred coins of one another. Grouping by seed therefore removes most of
the baseline's noise, and the deviation left is what the policy's own sampling
did.

The bank is terminal and the actor runs at gamma 1, lambda 1, so the
trajectory's one standardized deviation is the return-to-go of every state in
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Trajectories that no bank group owns: rows left out by the caller, and the
#: lone game against an opponent from one seat, which has no one to compare with.
UNGROUPED = -1


@dataclass(frozen=True)
class BankAdvantage:
    """Per-trajectory standardized leave-one-out deviations, zero when ungrouped."""

    z: np.ndarray
    own_bank_mean: float
    deviation_std: float
    scale: float


def opponent_bank_groups(
    opponents: np.ndarray, seats: np.ndarray, maps: np.ndarray | None = None
) -> np.ndarray:
    """Group id per trajectory by (opponent, learner seat, map), else `UNGROUPED`.

    `opponents` names each row's opponent with any nonnegative id, self-play
    included as one opponent; rows marked `UNGROUPED` stay out. `maps` splits
    an opponent's games further, by map seed where the wave repeated seeds, and
    defaults to one map for every row. A group of one has no leave-one-out
    baseline, so its row is left ungrouped too.
    """
    opponents = np.asarray(opponents, dtype=np.int64)
    seats = np.asarray(seats, dtype=np.int64)
    maps = np.zeros_like(opponents) if maps is None else np.asarray(maps, dtype=np.int64)
    if not opponents.shape == seats.shape == maps.shape or opponents.ndim != 1:
        raise ValueError("opponents, seats and maps must be matching vectors")
    if (opponents < UNGROUPED).any():
        raise ValueError("opponent ids must be nonnegative or UNGROUPED")
    groups = np.full(opponents.shape, UNGROUPED, dtype=np.int64)
    rows = np.flatnonzero(opponents != UNGROUPED)
    if not rows.size:
        return groups
    keys = np.stack((opponents[rows], seats[rows], maps[rows]), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    paired = counts[inverse] >= 2
    groups[rows[paired]] = inverse[paired]
    return groups


def own_bank_advantages(
    final_money: np.ndarray, groups: np.ndarray, *, scale_floor: float
) -> BankAdvantage:
    """Standardize each grouped trajectory's bank against its group's leave-one-out mean.

    The deviation is `x_i - mean_{j != i} x_j`, which equals
    `K / (K - 1) * (x_i - mean_group)`. It is divided by the uncentered RMS of
    the deviations over all grouped trajectories, which is their std because
    each group's deviations sum to zero. `scale_floor`, in money, keeps a wave
    of near-identical copies from turning a few coins of difference into unit
    advantages.
    """
    final_money = np.asarray(final_money, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if final_money.shape != groups.shape or final_money.ndim != 1:
        raise ValueError("final money and bank groups must be matching vectors")
    if not np.isfinite(final_money).all():
        raise ValueError("final money must be finite")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("bank advantage scale floor must be finite and positive")
    z = np.zeros(final_money.shape, dtype=np.float32)
    grouped = groups != UNGROUPED
    if not grouped.any():
        return BankAdvantage(z=z, own_bank_mean=0.0, deviation_std=0.0, scale=scale_floor)
    money = final_money[grouped]
    _, members = np.unique(groups[grouped], return_inverse=True)
    members = members.reshape(-1)
    counts = np.bincount(members)
    if (counts < 2).any():
        raise ValueError("every bank group needs at least two trajectories")
    totals = np.bincount(members, weights=money)
    size = counts[members].astype(np.float64)
    deviations = money - (totals[members] - money) / (size - 1.0)
    deviation_std = float(np.sqrt(np.mean(deviations**2)))
    scale = max(deviation_std, scale_floor)
    z[grouped] = deviations / scale
    return BankAdvantage(
        z=z,
        own_bank_mean=float(money.mean()),
        deviation_std=deviation_std,
        scale=scale,
    )
