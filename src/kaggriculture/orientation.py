"""Per-game board symmetries for training invariance.

The farm is a square. Four grid-preserving symmetries — identity, horizontal
mirror, vertical mirror, and 180-degree rotation — produce four views of the
same game. Training assigns one symmetry per *game*, not per population
member: both seats of a game share the frame, and games cycle through all
four so every learner sees every rendering. A member-locked assignment
left three of four warm-started clones playing a board their BC weights
had never seen, and they went broke on the first wave.

The mapping is closed over three surfaces:

1. Board features are flipped spatially ([channel, y, x] layout).
2. Unit position features (and raw positions) transform like points.
3. Movement actions permute: what the oriented view calls EAST executes as the
   real-world direction the orientation maps it to. Market heads have no
   spatial semantics and are never permuted.

Masks and sampled actions cross between the two spaces through the same
permutation, so a mask column always describes the action whose logit occupies
that column, whichever space storage uses. Evaluation and inference stay on
the identity frame — the real competition board.
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


#: The four grid symmetries, in cycle order. Game *g* trains under
#: ``ORIENTATION_CYCLE[g % 4]``. Both seats of that game share the code.
ORIENTATION_CYCLE = (
    Orientation.IDENTITY,
    Orientation.MIRROR_X,
    Orientation.MIRROR_Y,
    Orientation.ROTATE_180,
)

#: Historical alias: the cycle used to be locked to member index.
MEMBER_ORIENTATIONS = ORIENTATION_CYCLE


def member_orientation(member_index: int) -> Orientation:
    """The *i*-th frame in the four-symmetry cycle.

    Kept because older checkpoints stamped this code next to member *i*'s
    weights. New collection ignores member index: use ``game_orientation``.
    """
    if member_index < 0:
        raise ValueError("member index cannot be negative")
    return ORIENTATION_CYCLE[member_index % len(ORIENTATION_CYCLE)]


def game_orientation(game_index: int) -> Orientation:
    """The board symmetry assigned to one game, cycling all four frames."""
    if game_index < 0:
        raise ValueError("game index cannot be negative")
    return ORIENTATION_CYCLE[game_index % len(ORIENTATION_CYCLE)]


def row_orientations(indices: np.ndarray) -> np.ndarray:
    """Map non-negative indices onto the four-symmetry cycle.

    Used as a compact source of all four codes in tests. Collection uses
    ``seat_orientations``, which is this cycle over games, repeated per seat.
    """
    values = np.asarray(indices, dtype=np.int64)
    if values.size and int(values.min()) < 0:
        raise ValueError("orientation indices cannot be negative")
    table = np.asarray([int(value) for value in ORIENTATION_CYCLE], dtype=np.int8)
    return table[values % len(ORIENTATION_CYCLE)]


def game_orientations(game_count: int) -> np.ndarray:
    """Per-game orientation codes, cycling identity, mirror-x, mirror-y, rotate-180."""
    if game_count < 0:
        raise ValueError("game count cannot be negative")
    return row_orientations(np.arange(game_count, dtype=np.int64))


def seat_orientations(game_count: int) -> np.ndarray:
    """Per-row codes for a native wave: both seats of a game share one frame."""
    return np.repeat(game_orientations(game_count), 2)


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
        board[rows] = flip_board(board[rows], orientation)
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


def apply_state_orientations(arrays: dict[str, np.ndarray], codes: np.ndarray) -> None:
    """Flip encoded board and unit features in place to match ``codes``."""
    orient_boards(arrays["board"], codes)
    orient_unit_features(arrays["units"], arrays["unit_positions"], codes)


def augment_demonstration_rows(arrays: dict[str, np.ndarray], codes: np.ndarray) -> None:
    """Flip encoded demonstration rows and remapped unit targets in place.

    Structured encodings have no flip. A non-identity code against a batch
    that lacks a board is a silent no-op for market-only fields and a bug
    for unit targets, so it is refused.
    """
    if "board" not in arrays:
        if (np.asarray(codes) != int(Orientation.IDENTITY)).any():
            raise ValueError("structured demonstrations have no orientation mapping")
        return
    apply_state_orientations(arrays, codes)
    arrays["unit_actions"] = orient_unit_actions(arrays["unit_actions"], codes)
    arrays["unit_masks"] = orient_unit_masks(arrays["unit_masks"], codes)
