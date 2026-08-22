"""Per-member state orientations for population league play.

Every population member sees the farm through its own fixed spatial orientation
of the board. The transforms are exactly the symmetries of the square grid that
map grid tiles to grid tiles: identity, horizontal mirror, vertical mirror, and
the 180-degree rotation. A member's observation is the true world rendered
through its orientation, and its movement actions are interpreted back through
the same map, so every member keeps acting legally in the real environment while
receiving genuinely different state streams -- four members at four orientations
cannot collapse into duplicate receivers of one shared encoding.

The mapping is closed over three surfaces:

1. Board features are flipped spatially ([channel, y, x] layout).
2. Unit position features (and raw positions) transform like points.
3. Movement actions permute: what the oriented view calls EAST executes as the
   real-world direction the orientation maps it to. Market heads have no
   spatial semantics and are never permuted.

Masks and sampled actions cross between the two spaces through the same
permutation, so a mask column always describes the action whose logit occupies
that column, whichever space storage uses.
"""

from __future__ import annotations

import enum

import numpy as np

from kaggriculture.actions import N_UNIT_ACTIONS, UnitAction
from kaggriculture.constants import BOARD_SIZE


class Orientation(enum.IntEnum):
    """One grid symmetry, named by what it does to the real world on screen.

    MIRROR_X flips the x axis (columns): the viewed board shows the real
    board's columns reversed, so the viewed EAST direction executes as real
    WEST. MIRROR_Y flips rows, so viewed NORTH executes as real SOUTH.
    ROTATE_180 applies both flips.
    """

    IDENTITY = 0
    MIRROR_X = 1
    MIRROR_Y = 2
    ROTATE_180 = 3


#: Member i of a population trains and plays under MEMBER_ORIENTATIONS[i % 4].
MEMBER_ORIENTATIONS = (
    Orientation.IDENTITY,
    Orientation.MIRROR_X,
    Orientation.MIRROR_Y,
    Orientation.ROTATE_180,
)


def member_orientation(member_index: int) -> Orientation:
    """The fixed orientation assigned to one population member index."""
    if member_index < 0:
        raise ValueError("member index cannot be negative")
    return MEMBER_ORIENTATIONS[member_index % len(MEMBER_ORIENTATIONS)]


def row_orientations(member_indices: np.ndarray) -> np.ndarray:
    """Per-row orientation codes for a wave's agent column."""
    members = np.asarray(member_indices, dtype=np.int64)
    if members.size and int(members.min()) < 0:
        raise ValueError("member indices cannot be negative")
    table = np.asarray([int(value) for value in MEMBER_ORIENTATIONS], dtype=np.int8)
    return table[members % len(MEMBER_ORIENTATIONS)]


def movement_permutation(orientation: Orientation) -> np.ndarray:
    """Oriented action index -> real action index, identity off movement.

    Column j of a real-space mask describes real action j. Placing
    ``oriented_logits[perm]`` into real-indexed columns puts each oriented
    logit under exactly the real mask column of the action it executes, so
    sampling against real masks yields real actions directly.
    """
    permutation = np.arange(N_UNIT_ACTIONS, dtype=np.int64)
    swaps: tuple[tuple[UnitAction, UnitAction], ...] = ()
    if orientation in (Orientation.MIRROR_X, Orientation.ROTATE_180):
        swaps += ((UnitAction.EAST, UnitAction.WEST),)
    if orientation in (Orientation.MIRROR_Y, Orientation.ROTATE_180):
        swaps += ((UnitAction.NORTH, UnitAction.SOUTH),)
    for left, right in swaps:
        permutation[left], permutation[right] = permutation[right], permutation[left]
    return permutation


def inverse_permutation(permutation: np.ndarray) -> np.ndarray:
    """The index map that undoes ``permutation``."""
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=permutation.dtype)
    return inverse


def flip_board(board: np.ndarray, orientation: Orientation) -> np.ndarray:
    """Flip the trailing [y, x] axes of board features in place."""
    if orientation == Orientation.IDENTITY:
        return board
    if orientation in (Orientation.MIRROR_X, Orientation.ROTATE_180):
        board[...] = board[..., ::-1]
    if orientation in (Orientation.MIRROR_Y, Orientation.ROTATE_180):
        board[...] = board[..., ::-1, :]
    return board


def orient_boards(board: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Apply one orientation per leading row of board features, in place."""
    for code in np.unique(orientations):
        orientation = Orientation(int(code))
        if orientation == Orientation.IDENTITY:
            continue
        rows = np.flatnonzero(orientations == code)
        flip_board(board[rows], orientation)
    return board


def orient_unit_features(
    units: np.ndarray, positions: np.ndarray, orientations: np.ndarray
) -> None:
    """Transform unit x/y columns and raw positions per row, in place.

    ``units`` columns 3 and 4 carry x and y scaled by ``1 / (BOARD_SIZE - 1)``;
    ``positions`` carries raw integer (x, y) pairs. A mirrored view maps a
    point's coordinate to ``BOARD_SIZE - 1 - coordinate`` on the flipped axis.
    """
    for code in np.unique(orientations):
        orientation = Orientation(int(code))
        if orientation == Orientation.IDENTITY:
            continue
        rows = np.flatnonzero(orientations == code)
        mirror_x = orientation in (Orientation.MIRROR_X, Orientation.ROTATE_180)
        mirror_y = orientation in (Orientation.MIRROR_Y, Orientation.ROTATE_180)
        if mirror_x:
            units[rows, :, 3] = 1.0 - units[rows, :, 3]
            positions[rows, :, 0] = (BOARD_SIZE - 1) - positions[rows, :, 0]
        if mirror_y:
            units[rows, :, 4] = 1.0 - units[rows, :, 4]
            positions[rows, :, 1] = (BOARD_SIZE - 1) - positions[rows, :, 1]


def orient_unit_logits(unit_logits: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Restripe oriented movement logits into real-action columns, in place.

    The actor forward consumes oriented features and therefore produces
    oriented action logits. The native sampler masks and samples against
    real-action columns, so each row's movement columns are restriped by its
    orientation's permutation; the market columns pass through untouched.
    """
    for code in np.unique(orientations):
        orientation = Orientation(int(code))
        if orientation == Orientation.IDENTITY:
            continue
        rows = np.flatnonzero(orientations == code)
        unit_logits[rows] = unit_logits[rows][:, :, movement_permutation(orientation)]
    return unit_logits


def orient_unit_actions(unit_actions: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Map sampled real actions into oriented action indices for storage."""
    oriented = np.array(unit_actions, dtype=unit_actions.dtype, copy=True)
    for code in np.unique(orientations):
        orientation = Orientation(int(code))
        if orientation == Orientation.IDENTITY:
            continue
        rows = np.flatnonzero(orientations == code)
        inverse = inverse_permutation(movement_permutation(orientation))
        oriented[rows] = inverse[unit_actions[rows]]
    return oriented


def orient_unit_masks(unit_masks: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Restripe real-action mask columns into oriented action columns."""
    oriented = np.array(unit_masks, dtype=unit_masks.dtype, copy=True)
    for code in np.unique(orientations):
        orientation = Orientation(int(code))
        if orientation == Orientation.IDENTITY:
            continue
        rows = np.flatnonzero(orientations == code)
        oriented[rows] = oriented[rows][:, :, movement_permutation(orientation)]
    return oriented
