#!/usr/bin/env python3
"""Prove the native sampler already selects per-row quantity heads out of an N-agent stack.

Stage 0(a) of `POPULATION_LEAGUE_PLAN.md`.  The population wave replaces one
learner plus frozen lanes with N concurrent learners, and it reaches the engine
through exactly one changed argument: `quantity_kind_gate`/`quantity_values`/
`quantity_bias` become a stack of N heads instead of `(actor, *opponents)`, and
`head_ids` carries values in `0..N-1` instead of 0 on every learner row
(`collect_mixed_play_rust`, `src/kaggriculture/rollout.py:1142` and `:1155`; the
population caller submitting exactly that is `collect_population_play_rust`,
`rollout.py:1471-1472`).  The plan's stated expected
Rust diff for that is *none*.  That claim is load-bearing -- every later stage is
scheduled behind it -- and it is the kind of claim that is easy to believe and
easy to be wrong about, because the head slice is computed by hand from
`head_ids[row]` (`rust/kagg_env/src/python.rs:527-538`): three offsets into three
flat slices, with `rank` read from a different argument.  Off-by-one arithmetic
there does not crash and does not look wrong.  It silently samples one agent's
market quantities from another agent's head, which in the population wave means
the behaviour policy that acted is not the policy the update replays, and PPO's
importance ratio is quietly measuring the wrong thing.

So this probe does not assert that the code looks right.  It asserts that a
4-head stack with mixed per-row `head_ids` reproduces, bit for bit, separate
single-head calls made with the same states, the same logits, the same draws, the
same temperatures and the same determinism flags, over a whole segment of an
episode.  Five checks, all exact, all over every array `sample_buffers` returns:

  * `per_game_heads` -- one 4-head stack, `head_ids` constant within a game and
    covering all four heads across games, against four separate width-1 calls.
    Because both seats of a game share a head, that game's whole trajectory is a
    function of that one head, so parity holds for the entire segment rather
    than only the first step, and the states compared are real reached states
    rather than the opening position.  This is the plan's sentence, verbatim.
  * `pair_heads_duplicated` -- the layout the population wave actually submits:
    `head_ids[2g]`/`head_ids[2g+1]` are the two members of an ordered pairing, so
    the two seats of one game read *different* lanes.  Comparing that against a
    width-1 call needs the four lanes to hold four copies of one head, which
    makes every row's sample independent of which lane it read and so keeps the
    trajectory shared for the whole segment.  What it proves is what a width-1
    call cannot: a wide stack plus per-seat-mixed `head_ids` adds nothing.
  * `pair_heads_permuted` -- the same per-seat-mixed layout with four genuinely
    different heads, run twice: once with the stack in order, once with the lanes
    permuted and `head_ids` remapped through that permutation's *inverse*, which
    is the mapping that names the lane now holding the head a row wants.  It is
    the only check that pins the offset arithmetic under the layout the wave
    actually submits, where the two seats of one game read two *different* heads:
    `per_game_heads` also catches a stride error, but only across games, and
    `pair_heads_duplicated` cannot catch one at all, since identical lanes read
    the same head whatever offset is used.
  * `pair_heads_padded` -- an 8-lane stack whose trailing four lanes hold noise
    no row references.  `collect_mixed_play_rust` already pads lanes to a common
    width so a captured CUDA graph survives a changing mix (`rollout.py:1179`),
    and the population wave keeps doing so; an unreferenced lane must not be
    readable from any row.
  * `pair_heads_first_step` -- the one check that pins selection to the *row*
    rather than the game, and the reason the four above are not sufficient on
    their own.  Every one of them keeps a game's two seats on a single head,
    because that is what holds the trajectory comparable for a whole segment; the
    consequence is that an engine resolving one head per game -- reading
    `head_ids[row - row % PLAYERS]`, so seat 1 silently acts from its opponent's
    head -- satisfies all four, and satisfies every mutation below that only
    changes head *content*.  In the population wave that fault would give half of
    all rows a behaviour policy the update does not replay, which is exactly the
    defect this probe exists to exclude.  It is closed at step 0, where every run
    holds the identical state and each row's own head is therefore checkable
    against that head's width-1 call, row by row, with a game's two seats on two
    different heads.  Depth 1 by necessity: the native API can reset an
    environment to a seed but not clone or restore one, so a deep version would
    have to replay every prefix and cost O(steps^2).  Selection is state-blind
    arithmetic, so one step of it is the whole of it; game-shaped outputs are
    excluded here because both seats' actions determine them and here they differ.
    Its resolution, measured against the emulated collapse rather than assumed:
    11 of 12 affected rows are flagged at 12 games and 23 of 24 at 24 games, with
    no false positive on the honest run.  The row that escapes is one whose
    opening mask left no quantity to choose, so no head could have changed it --
    which is why this check is stated per row: a uniform indexing fault touches
    every seat-1 row and cannot hide behind one of them.

Why a width-1 call is the right thing to compare against, and not merely a
convenient one: every reference run submits `head_ids` of all zeros, so it reads
at offset zero on all three arrays.  Offset zero is what every candidate stride
formula agrees on.  The references are therefore independent of the arithmetic
under test, and any error in it can only move the *mixed* run -- which is what
makes a zero here evidence rather than two implementations agreeing on one bug.

The negative control matters as much as the checks.  Four heads that happened to
sample identically would satisfy every equality above while proving nothing, so
`discrimination` counts, for each of the six unordered head pairs, how many
sampled market-quantity slots the two width-1 runs disagree on, and the run fails
if any pair falls below `--min-discrimination`.  That subsumes a separate coverage
gate rather than needing one, because a pair cannot disagree on a slot the mask
never opened; `quantity_decisions` is still reported, since the reader is owed the
size of the factor space the heads were consulted over.

What that number says about this probe's reach, measured rather than assumed:
active quantity slots scale with the *width* of the wave and not its depth -- 303
at 12 games, 902 at 24, 1,938 at 48, but only 352 by step 719 against 303 by step
96.  The synthetic policy driving these rows earns nothing, so once the opening
money is spent no further order is affordable, and depth buys deeper states rather
than more market decisions.  The head is therefore exercised on early- and
mid-game market states and not on a wealthy agent trading in large quantities.
That bounds the probe honestly and does not weaken its claim: what is under test
is which slice of parameters a row reads, and an offset is right or wrong
independently of whether the bin it selects is 3 or 97.

`discrimination` bounds all four checks at once, which is not the same as showing
each one individually could have failed, so every check is also re-run against
the specific fault it guards -- head 0's rows read against head 1's call, a stack
of identical lanes read against a different head, a permuted stack whose
`head_ids` were left unremapped, and rows pointed at the padding lanes.  Each
mutation must diverge; one that still compares equal is reported as a failure of
this probe, because a check satisfied by construction is worse than a missing one.
The unremapped variant is the mutation chosen for the permutation check because it
is a genuine fault at every N: the sharper direction error, `head_ids` mapped
through the permutation rather than its inverse, does not exist at N = 2, where
the only permutation of two lanes is its own inverse.

The verdict is **bit-exact, no tolerance**, and that is structural rather than
lucky.  Each row's sample is a sequential scalar computation over its own slices.
`sample_and_step_into` runs two rayon blocks, one across rows (`python.rs:492-559`)
and one across games (`:562-574`), and both are indexed rather than reduced: every
element is written by exactly one iteration from its own index, so no sum or
maximum can accumulate in a varying order and `potential_cache` is per game.  A
tolerance here would therefore hide a real defect instead of absorbing
floating-point noise, so the gate is exact equality of the underlying bits --
floats are compared as `uint32`, which also refuses `-0.0` against `0.0` and a
differing NaN payload.  The game-parallel block is also what carries the
independence `per_game_heads` rests on: one game's step cannot see another's.

Deliberately not proven.  Not that the *Python* side assembles the stack
correctly: `_quantity_heads` and the vmapped ensemble are Stage 0(b)'s subject
(`scripts/audit_replay_parity.py`), and this probe feeds the engine synthetic
random heads precisely so that a defect in either side cannot cancel a defect in
the other.  Not the cost of the wider stack either; that is Stage 0(c).  And not
the pairing schedule -- the ordered pairs are re-derived locally from
`itertools.permutations` rather than imported from `rollout.population_pairings`,
because Stage 0 must be runnable before Stage 2 exists and a probe that shares a
helper with the code it is clearing cannot fault that helper.

One class a single run cannot close, named because `--rank` looks like a knob
nobody needs to turn.  All three offsets are linear in `rank`, so an offset
computed against the literal 32 rather than the passed argument is numerically
identical at the default and invisible to every check here.  A second run at any
other rank closes it, since the shapes are all derived from the argument and
nothing else changes; 16 and 32 both return the same verdict.

The probe never builds: `load_native(build=False)` measures the artifact training
already runs, since a probe whose evidence is "no Rust change is needed" must not
itself have compiled one.

Run from the repository root:
  OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python \
    scripts/probe_population_heads.py --output artifacts/probes/population-heads.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import EPISODE_STEPS, MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.provenance import source_identity
from kaggriculture.rust_env import load_native

PLAYERS = 2
# One transition fewer than the configured episode: the last one ends the game.
MAX_TRANSITIONS = EPISODE_STEPS - 1
# `sample_buffers` keys whose leading axis is the game, not the factor row.
GAME_OUTPUTS = frozenset(
    {
        "rewards",
        "final_money",
        "previous_potentials",
        "potentials",
        "terminal_utilities",
        "dones",
    }
)
# Temperatures cycled across rows so the per-row temperature array is exercised
# rather than left at the scalar the current self-play path passes.
ROW_TEMPERATURES = (1.0, 0.75, 1.25, 1.0)
# Every fifth row samples by argmax, which is the setting most sensitive to a
# mis-selected head: it reports the head's mode instead of a draw from it.
DETERMINISTIC_STRIDE = 5
# `python.rs:458-461`, the guard that makes an over-wide `head_ids` an error
# rather than a read of whatever follows the stack in memory.
HEAD_RANGE_ERROR = "head_ids contains an out-of-range head"
# Separates the unreferenced padding lanes from the live heads in the same run.
PADDING_SEED_MASK = 0xBADFEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="population size N, which is the height of the quantity-head stack",
    )
    parser.add_argument(
        "--games",
        type=int,
        help=(
            "games in the wave; must be a multiple of both N and N*(N-1). "
            "Defaults to 2*N*(N-1), so every ordered pairing gets two games and "
            "no lane is over-represented in the compared rows"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=96,
        help=(
            "transitions to compare; buys state depth, not market coverage -- "
            "measured at 12 games, active quantity slots reach 303 by step 96 and "
            "only 352 by step 719, because a synthetic policy earns nothing and "
            "cannot afford further orders. Width is the coverage axis instead"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0, help="first game seed")
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="quantity-head rank; the production ModelConfig default",
    )
    parser.add_argument(
        "--draw-seed",
        type=int,
        default=20260818,
        help="seed for the logits and uniform draws, replayed identically for every run",
    )
    parser.add_argument(
        "--head-seed",
        type=int,
        default=7,
        help="seed for the synthetic quantity heads",
    )
    parser.add_argument(
        "--min-discrimination",
        type=int,
        default=64,
        help=(
            "sampled market-quantity slots each pair of heads must disagree on; "
            "below this the parity checks are satisfiable by heads that do nothing. "
            "The default is a floor with margin, not a target: the closest pair "
            "measures 591 slots on the default wave"
        ),
    )
    parser.add_argument("--debug-build", action="store_true")
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    return parser.parse_args()


def _row_temperatures(rows: int) -> np.ndarray:
    cycle = np.array(ROW_TEMPERATURES, dtype=np.float32)
    return np.resize(cycle, rows)


def _row_deterministic(rows: int) -> np.ndarray:
    flags = np.zeros(rows, dtype=np.bool_)
    flags[::DETERMINISTIC_STRIDE] = True
    return flags


def _head_stack(heads: int, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic quantity heads with the exact shapes the sampler validates.

    Scaled to be genuinely different policies rather than perturbations of one:
    the bias term alone shifts the per-kind quantity logits by order 1.5 nats, so
    two lanes disagree on the sampled bin often enough for the discrimination
    control below to have power. Random rather than loaded from a checkpoint on
    purpose -- a defect in how Python assembles real heads must not be able to
    cancel a defect in how Rust indexes them.
    """
    generator = np.random.default_rng(seed)
    return (
        generator.normal(0.0, 1.0, (heads, N_MARKET_KINDS, rank)).astype(np.float32),
        generator.normal(0.0, 1.0, (heads, N_QUANTITIES, rank)).astype(np.float32),
        generator.normal(0.0, 1.5, (heads, N_MARKET_KINDS, N_QUANTITIES)).astype(np.float32),
    )


def _lane(stack: tuple[np.ndarray, ...], lanes: np.ndarray) -> tuple[np.ndarray, ...]:
    """Restack the heads in `lanes` order, C-contiguous as the sampler requires."""
    return tuple(np.ascontiguousarray(array[lanes]) for array in stack)


def _padded_stack(
    stack: tuple[np.ndarray, ...], args: argparse.Namespace
) -> tuple[np.ndarray, ...]:
    """The population's heads with an equal number of unrelated lanes appended.

    Different heads, not zeros: a zeroed lane samples uniformly over what the
    mask allows, which is a plausible enough answer to be mistaken for a correct
    one if a row ever read it.
    """
    padding = _head_stack(args.heads, args.rank, args.head_seed ^ PADDING_SEED_MASK)
    return tuple(np.concatenate([live, noise]) for live, noise in zip(stack, padding, strict=True))


def _game_head_ids(heads: int, games: int) -> np.ndarray:
    """One head per game, both seats, cycling so every head owns games.

    Both seats sharing a head is what makes the whole segment comparable against
    a width-1 call: the game's entire trajectory then depends on that head alone.
    """
    return np.repeat(np.arange(games, dtype=np.uint16) % heads, PLAYERS)


def _pair_head_ids(heads: int, games: int) -> np.ndarray:
    """The population wave's own layout: row 2g and 2g+1 hold an ordered pairing.

    Ordered pairs are re-derived here rather than imported from
    `rollout.population_pairings` so Stage 0 does not depend on Stage 2.  Each of
    the `heads * (heads - 1)` pairs appears equally often, so no lane is
    over-represented in the rows this probe compares.
    """
    pairs = np.array(list(itertools.permutations(range(heads), 2)), dtype=np.uint16)
    return np.tile(pairs, (games // len(pairs), 1)).reshape(-1)


def _seat_collapsed(head_ids: np.ndarray) -> np.ndarray:
    """`head_ids` as an engine would see them if it read one head per game.

    Handing a correct engine these ids reproduces exactly what an engine indexing
    `head_ids[row - row % PLAYERS]` would do with the real ones, so it emulates
    that fault without touching Rust. Seat 0 is unaffected; seat 1 inherits its
    game-mate's head, which is the defect every segment-long check is blind to.
    """
    return np.repeat(head_ids[::PLAYERS], PLAYERS)


def _play(
    module: Any,
    stack: tuple[np.ndarray, ...],
    head_ids: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, np.ndarray]]:
    """Play one wave and return copies of every sampler output, per step.

    The draw generator is re-seeded from `--draw-seed` on every call, so two runs
    receive byte-identical logits, contexts and uniform draws by construction
    rather than by a stored buffer that could be mutated between runs. Any
    difference between two returned trajectories is therefore attributable to the
    head stack and `head_ids` alone.
    """
    rows = args.games * PLAYERS
    kind_gate, quantity_values, quantity_bias = stack
    temperatures = _row_temperatures(rows)
    deterministic = _row_deterministic(rows)
    # No built-in lane exists in the population wave; 0 means "sample from the network".
    builtin_agents = np.zeros(rows, dtype=np.uint8)
    seeds = np.arange(args.seed_start, args.seed_start + args.games, dtype=np.uint64)
    environment = module.BatchEnv(seeds)
    sampled = environment.sample_buffers()
    generator = np.random.default_rng(args.draw_seed)
    trajectory: list[dict[str, np.ndarray]] = []
    for _ in range(args.steps):
        unit_logits = generator.normal(0.0, 2.0, (rows, MAX_UNITS, N_UNIT_ACTIONS)).astype(
            np.float32
        )
        kind_logits = generator.normal(0.0, 2.0, (rows, MAX_MARKET_ORDERS, N_MARKET_KINDS)).astype(
            np.float32
        )
        context = generator.normal(0.0, 1.0, (rows, MAX_MARKET_ORDERS, args.rank)).astype(
            np.float32
        )
        unit_draws = generator.random((rows, MAX_UNITS), dtype=np.float32)
        kind_draws = generator.random((rows, MAX_MARKET_ORDERS), dtype=np.float32)
        quantity_draws = generator.random((rows, MAX_MARKET_ORDERS), dtype=np.float32)
        environment.sample_and_step_into(
            unit_logits,
            kind_logits,
            context,
            kind_gate,
            quantity_values,
            quantity_bias,
            head_ids,
            unit_draws,
            kind_draws,
            quantity_draws,
            deterministic,
            temperatures,
            builtin_agents,
            sampled,
        )
        trajectory.append({name: np.array(value) for name, value in sampled.items()})
    return trajectory


def _verify_output_axes(module: Any) -> dict[str, int]:
    """Refuse to compare until the engine confirms which axis each output leads with.

    Every comparison below selects either rows or games by `GAME_OUTPUTS`, so an
    output that changed axes would quietly compare the wrong slice -- inventing a
    divergence, or hiding one. The layout is not this probe's to pin, so it is
    read back from `sample_buffers` rather than trusted. One game makes the two
    axes distinguishable, since a row axis is then 2 and a game axis 1.
    """
    sampled = module.BatchEnv(np.zeros(1, dtype=np.uint64)).sample_buffers()
    leading = {name: int(np.asarray(value).shape[0]) for name, value in sampled.items()}
    misread = {
        name: axis
        for name, axis in leading.items()
        if axis != (1 if name in GAME_OUTPUTS else PLAYERS)
    }
    if misread:
        raise SystemExit(
            f"GAME_OUTPUTS no longer classifies the native sampler's buffers: {misread}"
        )
    return leading


def _bits(values: np.ndarray) -> np.ndarray:
    """The array's raw bits, so float equality refuses -0.0 and NaN payloads."""
    return values.view(np.uint32) if values.dtype == np.float32 else values


def _compare(
    name: str,
    left: list[dict[str, np.ndarray]],
    right: list[dict[str, np.ndarray]],
    *,
    row_index: np.ndarray,
    game_index: np.ndarray,
    detail: dict[str, Any],
) -> dict[str, Any]:
    """Count every element on which two trajectories differ bit for bit.

    The counts carry their own denominator: a zero divergence count is evidence
    only next to the number of elements it was zero out of.
    """
    divergent: dict[str, int] = {}
    first: dict[str, Any] | None = None
    compared = 0
    worst = 0.0
    for step, (before, after) in enumerate(zip(left, right, strict=True)):
        for output, values in before.items():
            index = game_index if output in GAME_OUTPUTS else row_index
            selected, other = values[index], after[output][index]
            compared += selected.size
            differing = int(np.count_nonzero(_bits(selected) != _bits(other)))
            if not differing:
                continue
            divergent[output] = divergent.get(output, 0) + differing
            # The first one names where to start reading; the rest are its wake.
            if first is None:
                first = {"step": step, "output": output, "elements": differing}
            if selected.dtype == np.float32:
                gap = np.abs(selected.astype(np.float64) - other.astype(np.float64))
                worst = max(worst, float(gap[np.isfinite(gap)].max(initial=0.0)))
    return {
        "check": name,
        "compared_elements": compared,
        "divergent_elements": sum(divergent.values()),
        "divergent_outputs": dict(sorted(divergent.items())),
        "first_divergence": first,
        "max_absolute_difference": worst,
        **detail,
    }


def _discrimination(
    singles: list[list[dict[str, np.ndarray]]], row_index: np.ndarray
) -> list[dict[str, Any]]:
    """How far apart the heads actually are, as sampled market-quantity slots.

    This is the negative control every equality above rests on. The quantity
    factor is the only one the head feeds directly, so it is counted on its own;
    measured, the market *kinds* diverge too, because the factors are masked
    sequentially and a quantity already committed narrows what the next order
    slot may be -- so a mis-selected head is if anything easier to see than this
    number alone suggests.
    """
    records = []
    for left, right in itertools.combinations(range(len(singles)), 2):
        differing = sum(
            int(
                np.count_nonzero(
                    before["market_quantities"][row_index] != after["market_quantities"][row_index]
                )
            )
            for before, after in zip(singles[left], singles[right], strict=True)
        )
        records.append({"heads": [left, right], "differing_quantity_slots": differing})
    return records


def _quantity_decisions(trajectory: list[dict[str, np.ndarray]]) -> int:
    """Market-quantity slots the mask left to choose, so the head was consulted."""
    return sum(int(np.count_nonzero(step["market_quantity_active"])) for step in trajectory)


def _rejects_out_of_range_head(
    module: Any, stack: tuple[np.ndarray, ...], args: argparse.Namespace
) -> dict[str, Any]:
    """The sampler must refuse a `head_ids` value the stack cannot supply.

    Part of the same claim: an N-agent stack is safe to submit only if the engine
    validates the width it was given rather than reading past it, which is what
    makes an indexing mistake in the caller a raised error instead of a silent
    read of another agent's head.

    Matched on the message and not merely on `ValueError`, because
    `sample_and_step_into` raises that for every shape mismatch, every
    non-contiguous input and every out-of-range draw, and all of those are
    validated *before* the head range. Accepting any `ValueError` would report
    this check green without the head-range guard having run at all.
    """
    head_ids = _pair_head_ids(args.heads, args.games).copy()
    head_ids[-1] = np.uint16(args.heads)
    # One step is enough: validation precedes sampling.
    try:
        _play(module, stack, head_ids, argparse.Namespace(**{**vars(args), "steps": 1}))
    except ValueError as error:
        raised, message = type(error).__name__, str(error)
    else:
        raised, message = None, None
    return {
        "check": "rejects_out_of_range_head",
        "raised": raised,
        "message": message,
        "matched": message is not None and HEAD_RANGE_ERROR in message,
    }


def _power(
    module: Any,
    stack: tuple[np.ndarray, ...],
    trajectories: dict[str, list[dict[str, np.ndarray]]],
    singles: list[list[dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Re-run each check with the fault it exists to catch, and require it to fail.

    A zero above is only evidence if a non-zero was reachable. Each mutation here
    is an engine fault its check must catch, emulated by feeding a correct engine
    the inputs a faulty one would have effectively used: read the wrong lane,
    permute the stack without remapping `head_ids`, point a row at a padding lane,
    or resolve one head per game instead of one per row. A mutation that still
    compares equal means the check is satisfied by construction and its zero says
    nothing, so it is a failure of this probe rather than of the engine.

    `pair_heads_duplicated` deliberately has no entry. Its lanes are identical
    copies, so no in-range lane-index error is observable through it by
    construction, and any substituted reference would only re-measure
    `discrimination`. What it rules out is a stack's *width*, or a nonzero
    `head_ids`, changing anything other than which head is read -- a property with
    no reachable counterexample to inject, which is stated here rather than
    dressed up as a fifth measurement.
    """
    rows = args.games * PLAYERS
    all_rows, all_games = np.arange(rows, dtype=np.int64), np.arange(args.games, dtype=np.int64)
    pair_ids = _pair_head_ids(args.heads, args.games)
    game_ids = _game_head_ids(args.heads, args.games)
    # Head 0's rows read against head 1: `--heads` is validated at 2 or more, so
    # this is always a head those rows provably did not use.
    other = 1
    permutation = np.roll(np.arange(args.heads), 1)
    padded = _padded_stack(stack, args)
    # The collapse fault only has to be visible at the one step the check reads.
    first_step = argparse.Namespace(**{**vars(args), "steps": 1})
    return [
        _compare(
            "per_game_heads",
            trajectories["per_game"],
            singles[other],
            row_index=np.flatnonzero(game_ids == 0).astype(np.int64),
            game_index=np.flatnonzero(game_ids[::PLAYERS] == 0).astype(np.int64),
            detail={"mutation": f"head 0's rows read against head {other}'s single-head call"},
        ),
        _compare(
            "pair_heads_permuted",
            trajectories["pair"],
            _play(module, _lane(stack, permutation), pair_ids, args),
            row_index=all_rows,
            game_index=all_games,
            detail={"mutation": "the stack permuted with head_ids left unremapped"},
        ),
        _compare(
            "pair_heads_padded",
            trajectories["pair"],
            _play(module, padded, (pair_ids + args.heads).astype(np.uint16), args),
            row_index=all_rows,
            game_index=all_games,
            detail={"mutation": "head_ids pointed at the padding lanes"},
        ),
        _compare(
            "pair_heads_first_step",
            _play(module, stack, _seat_collapsed(pair_ids), first_step),
            singles[0][:1],
            row_index=np.flatnonzero(pair_ids == 0).astype(np.int64),
            game_index=np.empty(0, dtype=np.int64),
            detail={
                "mutation": "one head resolved per game, so seat 1 inherits its game-mate's",
                "diverging_rows": "seat 1's only; the collapse leaves seat 0's head alone",
            },
        ),
    ]


def _failures(
    comparisons: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
    discrimination: list[dict[str, Any]],
    rejection: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    """Every reason this probe does not clear its gates."""
    failures = [
        f"{record['check']}: {record['divergent_elements']} of {record['compared_elements']} "
        f"elements diverge in {sorted(record['divergent_outputs'])}"
        for record in comparisons
        if record["divergent_elements"]
    ]
    failures += [
        f"heads {record['heads']} sample the same quantity in all but "
        f"{record['differing_quantity_slots']} slots, below --min-discrimination "
        f"{args.min_discrimination}, so the parity checks above are vacuous"
        for record in discrimination
        if record["differing_quantity_slots"] < args.min_discrimination
    ]
    failures += [
        f"{record['check']} still compares equal over {record['compared_elements']} elements "
        f"with the fault it guards injected ({record['mutation']}), so its zero above is "
        "satisfied by construction and proves nothing"
        for record in mutations
        if not record["divergent_elements"]
    ]
    if not rejection["matched"]:
        failures.append(
            "an out-of-range head id did not reach the head-range guard: expected a ValueError "
            f"saying {HEAD_RANGE_ERROR!r}, got {rejection['raised']} {rejection['message']!r}"
        )
    return failures


def main() -> None:
    args = parse_args()
    if args.heads < 2:
        raise ValueError("--heads must be at least 2 for a population to have distinct pairings")
    ordered_pairs = args.heads * (args.heads - 1)
    if args.games is None:
        args.games = 2 * ordered_pairs
    if args.games < 1 or args.games % args.heads or args.games % ordered_pairs:
        raise ValueError(f"--games must be a positive multiple of {args.heads} and {ordered_pairs}")
    if not 1 <= args.steps <= MAX_TRANSITIONS:
        raise ValueError(f"--steps must be between 1 and {MAX_TRANSITIONS}")
    if args.rank < 1:
        raise ValueError("--rank must be positive")

    # The probe never builds: the artifact under test is the one training already
    # runs, and evidence that "no Rust change is needed" must not have compiled one.
    module = load_native(build=False, release=not args.debug_build)
    output_axes = _verify_output_axes(module)
    rows = args.games * PLAYERS
    all_rows = np.arange(rows, dtype=np.int64)
    all_games = np.arange(args.games, dtype=np.int64)
    stack = _head_stack(args.heads, args.rank, args.head_seed)
    zero_ids = np.zeros(rows, dtype=np.uint16)
    game_ids = _game_head_ids(args.heads, args.games)
    pair_ids = _pair_head_ids(args.heads, args.games)

    comparisons: list[dict[str, Any]] = []
    # One width-1 call per head, reused by three checks: it is the same run by
    # construction, since the seeds, the draw seed and every per-row input match.
    singles = [
        _play(module, _lane(stack, np.array([head])), zero_ids, args) for head in range(args.heads)
    ]

    per_game = _play(module, stack, game_ids, args)
    for head in range(args.heads):
        comparisons.append(
            _compare(
                "per_game_heads",
                per_game,
                singles[head],
                row_index=np.flatnonzero(game_ids == head).astype(np.int64),
                game_index=np.flatnonzero(game_ids[::PLAYERS] == head).astype(np.int64),
                detail={"head": head, "head_ids": "constant within a game, cycles across games"},
            )
        )

    duplicated: dict[str, list[dict[str, np.ndarray]]] = {}
    for head in range(args.heads):
        duplicated[f"duplicated_{head}"] = _play(
            module, _lane(stack, np.full(args.heads, head)), pair_ids, args
        )
        comparisons.append(
            _compare(
                "pair_heads_duplicated",
                duplicated[f"duplicated_{head}"],
                singles[head],
                row_index=all_rows,
                game_index=all_games,
                detail={"head": head, "head_ids": "ordered pairings, one lane per seat"},
            )
        )

    # The reference the remaining checks are read against: the population layout
    # on the stack in order, run once.
    pair_reference = _play(module, stack, pair_ids, args)

    # Per-row selection, which no segment-long check above can see. All of them
    # keep a game's two seats on one head so the trajectory stays shared, and an
    # engine that read `head_ids` once per game instead of once per row would
    # satisfy every one of them while seat 1 acted from its opponent's head. At
    # step 0 every run holds the identical state, so each row's own head is
    # checkable directly against that head's width-1 call. Game-shaped outputs
    # are excluded by an empty game index: both seats' actions determine them, and
    # here the two seats genuinely differ.
    for head in range(args.heads):
        comparisons.append(
            _compare(
                "pair_heads_first_step",
                pair_reference[:1],
                singles[head][:1],
                row_index=np.flatnonzero(pair_ids == head).astype(np.int64),
                game_index=np.empty(0, dtype=np.int64),
                detail={
                    "head": head,
                    "head_ids": "ordered pairings; a game's two seats read different heads",
                    "excludes": "game-shaped outputs, which both seats' actions determine",
                },
            )
        )

    # A rotation rather than a reversal: it relocates every lane whatever N is,
    # where a reversal leaves the middle lane of an odd stack at its own offset.
    # A row wanting head h must name the lane that now holds it, which is the
    # inverse permutation; `permutation[pair_ids]` would send it to a different
    # head entirely. That substitution is a genuine fault only from N = 3, since
    # the sole permutation of two lanes is its own inverse, so the power section
    # injects the unremapped ids instead and stays falsifiable at every N.
    permutation = np.roll(np.arange(args.heads), 1)
    inverse = np.argsort(permutation)
    comparisons.append(
        _compare(
            "pair_heads_permuted",
            pair_reference,
            _play(module, _lane(stack, permutation), inverse[pair_ids].astype(np.uint16), args),
            row_index=all_rows,
            game_index=all_games,
            detail={"permutation": permutation.tolist(), "head_ids": "ordered pairings, remapped"},
        )
    )

    padded = _padded_stack(stack, args)
    comparisons.append(
        _compare(
            "pair_heads_padded",
            pair_reference,
            _play(module, padded, pair_ids, args),
            row_index=all_rows,
            game_index=all_games,
            detail={"lanes": len(padded[0]), "referenced_lanes": args.heads},
        )
    )

    trajectories = {"per_game": per_game, "pair": pair_reference, **duplicated}
    mutations = _power(module, stack, trajectories, singles, args)

    discrimination = _discrimination(singles, all_rows)
    decisions = _quantity_decisions(per_game)
    rejection = _rejects_out_of_range_head(module, stack, args)
    failures = _failures(comparisons, mutations, discrimination, rejection, args)

    for record in comparisons:
        print(
            f"{record['check']}: {record['divergent_elements']} divergences of "
            f"{record['compared_elements']} elements"
        )
    weakest_mutation = min(record["divergent_elements"] for record in mutations)
    weakest_pair = min(record["differing_quantity_slots"] for record in discrimination)
    print(
        f"\npower: the least visible injected fault still diverges in {weakest_mutation} elements"
        f"\ndiscrimination: the closest pair of heads samples a different market quantity in "
        f"{weakest_pair} slots\nquantity decisions reached: {decisions}\n"
        f"out-of-range head id: {rejection['raised'] or 'accepted'}"
    )

    if args.output is not None:
        report = {
            "event": "population_head_probe",
            "module": getattr(module, "__file__", None),
            "heads": args.heads,
            "games": args.games,
            "steps": args.steps,
            "seed_start": args.seed_start,
            "seeds": [args.seed_start, args.seed_start + args.games - 1],
            "rank": args.rank,
            "draw_seed": args.draw_seed,
            "head_seed": args.head_seed,
            "row_temperatures": list(ROW_TEMPERATURES),
            "deterministic_stride": DETERMINISTIC_STRIDE,
            "game_head_ids": game_ids.tolist(),
            "pair_head_ids": pair_ids.tolist(),
            "min_discrimination": args.min_discrimination,
            "quantity_decisions": decisions,
            "output_axes": output_axes,
            "comparisons": comparisons,
            "discrimination": discrimination,
            "mutations": mutations,
            "out_of_range_head": rejection,
            "tolerance": "none; every comparison is exact equality of the underlying bits",
            "verdict": "no rust change required" if not failures else "rust change required",
            "failures": failures,
            "source_identity": source_identity(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    if failures:
        raise SystemExit("population head parity failed:\n  " + "\n  ".join(failures))
    print(f"\nverdict: the native sampler needs no Rust change for {args.heads}-agent head stacks")


if __name__ == "__main__":
    main()
