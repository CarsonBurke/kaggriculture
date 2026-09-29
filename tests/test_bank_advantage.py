from __future__ import annotations

import numpy as np
import pytest

from kaggriculture.bank_advantage import UNGROUPED, opponent_bank_groups, own_bank_advantages


def test_groups_are_one_opponent_from_one_seat() -> None:
    # Two self-play games (opponent 0, both seats), then league games against
    # lanes 1 and 2 from alternating seats, then a row the caller left out.
    opponents = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, UNGROUPED])
    seats = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)

    groups = opponent_bank_groups(opponents, seats)

    # Self-play seat 0 and seat 1 are separate groups of two.
    assert groups[0] == groups[2] and groups[1] == groups[3]
    # Lane 1 has two seat-0 games and lane 2 two seat-1 games; each lane's lone
    # game from the other seat has no one to compare with.
    assert groups[4] == groups[6] and groups[7] == groups[9]
    assert (groups[[5, 8, 10]] == UNGROUPED).all()
    assert len({groups[0], groups[1], groups[4], groups[7]}) == 4


def test_maps_split_an_opponents_games_by_seed() -> None:
    # Four seat-0 self-play games on two seeds, and a league lane whose games all
    # have map 0: the seeds split the self-play block and leave the lane whole.
    opponents = np.asarray([0, 0, 0, 0, 1, 1])
    seats = np.zeros(6, dtype=np.int8)
    maps = np.asarray([7, 7, 8, 8, 0, 0])

    groups = opponent_bank_groups(opponents, seats, maps)

    assert groups[0] == groups[1] and groups[2] == groups[3] and groups[4] == groups[5]
    assert len({groups[0], groups[2], groups[4]}) == 3
    # Without maps the self-play block is one group.
    assert len(set(opponent_bank_groups(opponents, seats)[:4])) == 1


def test_nothing_to_group_and_malformed_opponents() -> None:
    seats = np.zeros(3, dtype=np.int8)
    assert (opponent_bank_groups(np.full(3, UNGROUPED), seats) == UNGROUPED).all()
    assert (opponent_bank_groups(np.asarray([0, 1, 2]), seats) == UNGROUPED).all()
    with pytest.raises(ValueError, match="matching vectors"):
        opponent_bank_groups(np.zeros(2, dtype=np.int64), seats)
    with pytest.raises(ValueError, match="nonnegative"):
        opponent_bank_groups(np.asarray([0, 0, -2]), seats)


def test_leave_one_out_deviations_match_the_definition_and_are_standardized() -> None:
    money = np.asarray([10_000.0, 14_000.0, 9_000.0, 7_000.0, 2_000.0, 50_000.0, 55_000.0])
    groups = np.asarray([0, 0, 0, 1, 1, UNGROUPED, UNGROUPED])

    bank = own_bank_advantages(money, groups, scale_floor=1.0)

    deviations = np.asarray(
        [
            money[i] - np.mean([money[j] for j in range(5) if groups[j] == groups[i] and j != i])
            for i in range(5)
        ]
    )
    np.testing.assert_allclose(bank.z[:5] * bank.scale, deviations, rtol=1e-6)
    # Each group's leave-one-out deviations sum to zero, so their RMS is their std.
    assert bank.deviation_std == pytest.approx(float(deviations.std()), rel=1e-12)
    assert bank.scale == pytest.approx(bank.deviation_std)
    assert float(np.sqrt(np.mean(bank.z[:5].astype(np.float64) ** 2))) == pytest.approx(1.0)
    # Ungrouped rows carry no bank advantage and do not enter its statistics.
    assert (bank.z[5:] == 0.0).all()
    assert bank.own_bank_mean == pytest.approx(money[:5].mean())


def test_the_floor_keeps_near_identical_copies_from_becoming_unit_advantages() -> None:
    money = np.asarray([90_000.0, 90_010.0, 90_000.0, 90_000.0])
    groups = np.asarray([0, 0, 1, 1])

    bank = own_bank_advantages(money, groups, scale_floor=1_000.0)

    assert bank.deviation_std == pytest.approx(np.sqrt(50.0))
    assert bank.scale == 1_000.0
    np.testing.assert_allclose(bank.z, [-0.01, 0.01, 0.0, 0.0], atol=1e-7)


def test_identical_copies_and_ungrouped_waves_carry_no_advantage() -> None:
    same = own_bank_advantages(np.full(4, 5_000.0), np.asarray([0, 0, 1, 1]), scale_floor=1.0)
    assert (same.z == 0.0).all() and same.deviation_std == 0.0
    none = own_bank_advantages(np.ones(3), np.full(3, UNGROUPED), scale_floor=1.0)
    assert (none.z == 0.0).all()


@pytest.mark.parametrize(
    ("money", "groups", "floor", "message"),
    [
        (np.ones(3), np.zeros(2, dtype=np.int64), 1.0, "matching vectors"),
        (np.asarray([1.0, np.nan]), np.asarray([0, 0]), 1.0, "finite"),
        (np.ones(2), np.asarray([0, 0]), 0.0, "floor"),
        (np.ones(3), np.asarray([0, 0, 1]), 1.0, "at least two"),
    ],
)
def test_malformed_inputs_are_refused(money, groups, floor, message) -> None:
    with pytest.raises(ValueError, match=message):
        own_bank_advantages(money, groups, scale_floor=floor)
