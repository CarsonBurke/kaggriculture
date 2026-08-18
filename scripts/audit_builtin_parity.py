#!/usr/bin/env python3
"""Differentially gate the native built-in agents against the kaggle_environments reference.

Each built-in reference agent now exists twice: once as the Python function in
`kaggle_environments.envs.kaggriculture.kaggriculture`, which is what the
leaderboard and every external evaluation actually plays, and once as
`Game::builtin_action` in the batched Rust wave, which is what training plays
once the built-ins enter the league.  Two implementations of one policy is a
divergence waiting to happen, and a divergence here hides well: a native
`starter` that waters one turn early still looks like a plausible opponent,
still produces gradients and still reports a score -- it just is not the agent
the leaderboard grades against, so every conclusion drawn from beating it is
about a different game.  This is the gate that keeps the second implementation
honest.

Proven exactly, at every step of the episode, for every game in the wave:

  * `pass` -- the submitted action is PASS on every active unit slot with an
    empty market queue.
  * `starter` -- the submitted action is *identical* to `starter_agent`'s return
    value.  `starter_agent` is a pure function of the state, so identical means
    identical, not close.
  * `random` -- the reachable set is identical, which holds for every draw of
    the reference's RNG and so is exact even though the rates are not: the
    farmer command is one of the seven work ops or a PLANT of a crop the seat
    holds seed for, every hand command is one of the seven ops, and the market
    queue is empty or exactly one single-unit BUY_SEED of an affordable crop.
  * all three -- the non-stepping `BatchEnv.builtin_actions` query and the
    action `sample_and_step_into` actually submits agree; a built-in row reports
    exactly zero log-probability and zero entropy; and the network-sampled row
    beside it in the same wave still reports positive entropy, so the built-in
    lane cannot be silently swallowing its neighbour.

Proven only distributionally, and why: `random_agent` draws from an unseeded
`random.Random()`.  There is no seed to match, so exact parity is not merely
unimplemented, it is undefined -- the reference does not reproduce itself.  The
native agent is seeded from the game seed instead, deliberately, so that
training runs reproduce.  What stays checkable is that both draw from the same
distribution, so the rates the reference hard-codes are gated against their
exact values (0.1 for BUY_SEED given an affordable crop, 0.3 for PLANT given a
held seed, 1/7 per farmer op given no PLANT, 1/7 per hand op) and the crop
choices against the exact uniform-over-what-was-available expectation.  The
Python reference is measured in lockstep on the identical states and gated the
same way, because a harness that has mis-read the reference and a native agent
that has mis-implemented it look the same from one side only.

The tolerance is a standard-normal score on each count rather than a hand-picked
epsilon: `--sigma` (default 5) bounds every statistic at five standard errors,
so across the ~140 statistics gated in a full run the family-wise false-alarm
rate stays below 1e-4.  The flip side, stated because it is the real limit of
this gate: five standard errors on n samples cannot see a rate error below
5*sqrt(p*(1-p)/n), which for the BUY_SEED rate on the default wave is about 0.6
points, or 6% relative.  A subtler bias needs a larger `--games`; the report
carries the implied tolerance next to every statistic so the next reader does
not have to re-derive it.

Deliberately not proven.  Not that the two engines agree: the reference agent is
asked about the *native* state rather than a state replayed through the official
Python engine, because an agent asked about a state it did not act from cannot
give an attributable answer.  Engine parity is a separate standing gate
(`scripts/check_rust_parity.py`, `rust/kagg_env/tests/parity_oracle.py`), and
this script depends on it.  Not the *effect* of the submitted action either: the
native built-ins submit unlegalised actions exactly as the reference does and
the engine no-ops the illegal ones internally, so what is compared is what was
asked for, which is where a policy divergence lives.

Run from the repository root after a release build:
  PYTHONPATH=src .venv/bin/python scripts/audit_builtin_parity.py \
    --output reports/builtin_parity.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as REFERENCE_CROPS
from kaggle_environments.envs.kaggriculture.kaggriculture import agents as REFERENCE_AGENTS

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    UnitAction,
    active_unit_count,
    market_order,
)
from kaggriculture.constants import (
    EPISODE_STEPS,
    MAX_MARKET_ORDERS,
    MAX_MARKET_QUANTITY,
    MAX_UNITS,
    SHED_CAPACITY,
)
from kaggriculture.provenance import source_identity
from kaggriculture.rust_env import load_native

PLAYERS = 2
# The codes `BuiltinAgent::from_code` accepts; 0 means "sample from the network".
BUILTIN_CODES = {"pass": 1, "random": 2, "starter": 3}
SCENARIOS = ("on_policy", "explore")
SOURCES = ("native", "reference")
# kaggriculture.py:1023 -- the only ops `random_agent` ever emits for a unit.
FARMER_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PASS")
PLANT = "PLANT"
# kaggriculture.py:1028 and :1032, the two rates the reference hard-codes.
BUY_SEED_PROBABILITY = 0.1
PLANT_PROBABILITY = 0.3
UNIFORM_OP_PROBABILITY = 1.0 / len(FARMER_OPS)
# Expected events needed in both tails before a Gaussian score means anything;
# the textbook normal-approximation threshold for a binomial count.
MIN_NORMAL_COUNT = 5.0
# One transition fewer than the configured episode: the last one ends the game.
MAX_TRANSITIONS = EPISODE_STEPS - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        default=",".join(BUILTIN_CODES),
        help="comma-separated built-ins to gate",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help=(
            "on_policy drives the seat under test with the built-in itself; "
            "explore drives both seats from the network so the built-in is "
            "asked about states its own policy never reaches"
        ),
    )
    # Wide by default: the distributional half of this gate resolves a rate
    # error of 5*sqrt(p*(1-p)/n), so the sample size *is* the sensitivity.
    parser.add_argument("--games", type=int, default=32, help="games in the wave")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=MAX_TRANSITIONS)
    parser.add_argument(
        "--draw-seed",
        type=int,
        default=20260812,
        help="seed for the uniform draws that feed the network-sampled rows",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="standard errors allowed on every distributional statistic",
    )
    parser.add_argument(
        "--min-expected",
        type=float,
        default=100.0,
        help="expected count below which a statistic is reported as underpowered and fails",
    )
    parser.add_argument(
        "--max-divergences",
        type=int,
        default=8,
        help="exact divergences to record before abandoning a scenario",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=4,
        help="whole episodes per pairing to replay through the official engine; 0 skips",
    )
    parser.add_argument("--debug-build", action="store_true")
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    return parser.parse_args()


def _observation(snapshot: dict[str, Any], player: int) -> dict[str, Any]:
    """Shape the native state into the observation the reference agents read.

    Native rather than replayed through the official engine on purpose: the
    reference must be asked about the exact state the native built-in acted
    from, or a disagreement cannot be attributed to either side.
    """
    return {
        "player": player,
        "step": snapshot["step"],
        "day": snapshot["day"],
        "hour": snapshot["hour"],
        "farms": snapshot["farms"],
        "private": snapshot["privates"][player],
        "market": snapshot["market"],
        "town": snapshot["town"],
    }


def _affordable_crops(observation: dict[str, Any]) -> tuple[str, ...]:
    """The crops `random_agent` may buy, taken from the reference's own table.

    Reading seed costs out of `kaggle_environments` rather than out of
    `kaggriculture.constants` keeps both sides of this differential test sourced
    from the side they describe, so a drift between the two tables shows up as a
    seed-cost bug somewhere else rather than as a phantom parity failure here.
    """
    money = observation["farms"][observation["player"]]["money"]
    return tuple(crop for crop in REFERENCE_CROPS if REFERENCE_CROPS[crop]["seed"] <= money)


def _available_seeds(observation: dict[str, Any]) -> tuple[str, ...]:
    seeds = observation["private"].get("seeds") or {}
    return tuple(crop for crop in REFERENCE_CROPS if (seeds.get(crop) or 0) > 0)


def _unit_action_value(command: list[Any]) -> int:
    """Map one engine unit command back onto its factor value.

    Raises rather than defaulting to PASS: a reference command this harness
    cannot encode is a reference it has stopped modelling, and folding it into
    PASS would report parity for a comparison that never happened.
    """
    if not command:
        raise ValueError("reference agent emitted an empty unit command")
    if command[0] == PLANT:
        return int(UnitAction[f"PLANT_{command[1]}"])
    if len(command) != 1:
        raise ValueError(f"reference agent emitted unencodable unit command {command!r}")
    return int(UnitAction[command[0]])


def _reference_unit_values(action: dict[str, Any], units: int) -> list[int]:
    """The factor value the reference's dict asks each active unit slot to take.

    The engine applies `hands` positionally and never reaches a unit the list
    does not mention (kaggriculture.py:924), so a reference returning
    `"hands": []` is asking every hand to do nothing, which is PASS.  Padding is
    that rule rather than a convenience, and it is the only normalization
    applied to either side.
    """
    commands = [action["farmer"], *action["hands"]]
    if len(commands) > units:
        raise ValueError(
            f"reference emitted {len(commands)} unit commands for {units} representable units"
        )
    values = [_unit_action_value(command) for command in commands]
    return values + [int(UnitAction.PASS)] * (units - len(values))


def _unit_ops(values: list[int]) -> list[tuple[str, str | None]]:
    """Split factor values into (op, crop) so PLANT_X collapses to one category."""
    ops: list[tuple[str, str | None]] = []
    for value in values:
        name = UnitAction(value).name
        if name.startswith(f"{PLANT}_"):
            ops.append((PLANT, name.removeprefix(f"{PLANT}_")))
        else:
            ops.append((name, None))
    return ops


def _orders(kinds: np.ndarray, quantities: np.ndarray) -> list[list[Any]]:
    """Decode one row's market factors into the orders it submits.

    Stops at the first STOP exactly as `compile_action` does, which is what
    makes the padding behind it don't-care on both sides of the comparison.
    """
    orders: list[list[Any]] = []
    for kind, quantity in zip(
        kinds[:MAX_MARKET_ORDERS], quantities[:MAX_MARKET_ORDERS], strict=True
    ):
        order = market_order(int(kind), int(quantity))
        if order is None:
            break
        orders.append(order)
    return orders


def _exact_difference(
    native_values: list[int],
    native_orders: list[list[Any]],
    reference: dict[str, Any],
    units: int,
) -> str | None:
    """Describe the first way a deterministic built-in differs from the reference."""
    expected = _reference_unit_values(reference, units)
    if native_values != expected:
        index = next(
            slot
            for slot, (left, right) in enumerate(zip(native_values, expected, strict=True))
            if left != right
        )
        return (
            f"unit {index}: native={UnitAction(native_values[index]).name} "
            f"reference={UnitAction(expected[index]).name}"
        )
    expected_orders = reference["market"][:MAX_MARKET_ORDERS]
    if native_orders != expected_orders:
        return f"market: native={native_orders!r} reference={expected_orders!r}"
    return None


def _support_violation(
    ops: list[tuple[str, str | None]],
    orders: list[list[Any]],
    affordable: tuple[str, ...],
    available: tuple[str, ...],
) -> str | None:
    """Describe the first way an action leaves `random_agent`'s reachable set."""
    op, crop = ops[0]
    if crop is not None:
        if crop not in available:
            return f"farmer planted {crop} holding seed for {sorted(available)}"
    elif op not in FARMER_OPS:
        return f"farmer op {op} outside {list(FARMER_OPS)}"
    for slot, (hand_op, hand_crop) in enumerate(ops[1:], start=1):
        if hand_crop is not None:
            return f"hand {slot} planted {hand_crop}; the reference never plants with a hand"
        if hand_op not in FARMER_OPS:
            return f"hand {slot} op {hand_op} outside {list(FARMER_OPS)}"
    if not orders:
        return None
    if len(orders) != 1:
        return f"market queue {orders!r} holds more than the reference's single order"
    order = orders[0]
    if order[0] != "BUY_SEED" or order[2] != 1:
        return f"market order {order!r} is not a single-unit BUY_SEED"
    if order[1] not in affordable:
        return f"market order bought {order[1]}, affordable {sorted(affordable)}"
    return None


class _Tally:
    """Counts of everything `random_agent`'s specification pins down, per source."""

    def __init__(self) -> None:
        self.farmer_ops: Counter[str] = Counter()
        self.hand_ops: Counter[str] = Counter()
        self.hand_slots = 0
        self.farmer_samples = 0
        self.plant_opportunities = 0
        self.plants = 0
        self.buy_opportunities = 0
        self.buys = 0
        self.buy_crops: Counter[str] = Counter()
        self.plant_crops: Counter[str] = Counter()
        # Every event offers a different option set -- what is affordable, what
        # seed is held -- so the expected per-crop count is a sum of differing
        # probabilities rather than n/5, with the matching sum of p*(1-p).
        self.buy_crop_mean = dict.fromkeys(REFERENCE_CROPS, 0.0)
        self.buy_crop_variance = dict.fromkeys(REFERENCE_CROPS, 0.0)
        self.plant_crop_mean = dict.fromkeys(REFERENCE_CROPS, 0.0)
        self.plant_crop_variance = dict.fromkeys(REFERENCE_CROPS, 0.0)

    def observe(
        self,
        ops: list[tuple[str, str | None]],
        orders: list[list[Any]],
        affordable: tuple[str, ...],
        available: tuple[str, ...],
    ) -> None:
        op, crop = ops[0]
        self.farmer_samples += 1
        if crop is None:
            self.farmer_ops[op] += 1
        else:
            self.farmer_ops[PLANT] += 1
            self.plants += 1
            self.plant_crops[crop] += 1
            _accumulate_uniform(self.plant_crop_mean, self.plant_crop_variance, available)
        for hand_op, _ in ops[1:]:
            self.hand_ops[hand_op] += 1
        self.hand_slots += len(ops) - 1
        if available:
            self.plant_opportunities += 1
        if affordable:
            self.buy_opportunities += 1
        if orders:
            self.buys += 1
            self.buy_crops[orders[0][1]] += 1
            _accumulate_uniform(self.buy_crop_mean, self.buy_crop_variance, affordable)


def _accumulate_uniform(
    mean: dict[str, float],
    variance: dict[str, float],
    options: tuple[str, ...],
) -> None:
    """Fold one uniform-over-`options` draw into a Poisson-binomial expectation."""
    probability = 1.0 / len(options)
    for option in options:
        mean[option] += probability
        variance[option] += probability * (1.0 - probability)


def _statistic(
    name: str,
    count: int,
    trials: int,
    mean: float,
    variance: float,
    sigma: float,
) -> dict[str, Any]:
    """One gated count with the score and the absolute tolerance it implies."""
    deviation = math.sqrt(variance)
    return {
        "statistic": name,
        "trials": trials,
        "count": count,
        "rate": count / trials if trials else 0.0,
        "expected_count": mean,
        "expected_rate": mean / trials if trials else 0.0,
        "z": (count - mean) / deviation if deviation > 0.0 else 0.0,
        "rate_tolerance": sigma * deviation / trials if trials and deviation > 0.0 else 0.0,
    }


def _rate_statistic(
    name: str,
    count: int,
    trials: int,
    probability: float,
    sigma: float,
) -> dict[str, Any]:
    """A count against a rate the reference states outright."""
    return _statistic(
        name,
        count,
        trials,
        trials * probability,
        trials * probability * (1.0 - probability),
        sigma,
    )


def _exact_statistics(tally: _Tally, sigma: float) -> list[dict[str, Any]]:
    """Gate one source's counts against the reference's stated probabilities."""
    # The reference reaches its uniform-over-seven branch exactly when it did
    # not plant (kaggriculture.py:1032), so that is the denominator: PLANT is
    # not one of the seven, and conditioning on it makes 1/7 exact rather than
    # an approximation that drifts with how much seed the seat happens to hold.
    non_plant = tally.farmer_samples - tally.plants
    statistics = [
        _rate_statistic(
            "buy_seed_rate", tally.buys, tally.buy_opportunities, BUY_SEED_PROBABILITY, sigma
        ),
        _rate_statistic(
            "plant_rate", tally.plants, tally.plant_opportunities, PLANT_PROBABILITY, sigma
        ),
    ]
    statistics += [
        _rate_statistic(
            f"farmer_op[{op}]", tally.farmer_ops[op], non_plant, UNIFORM_OP_PROBABILITY, sigma
        )
        for op in FARMER_OPS
    ]
    statistics += [
        _rate_statistic(
            f"hand_op[{op}]", tally.hand_ops[op], tally.hand_slots, UNIFORM_OP_PROBABILITY, sigma
        )
        for op in FARMER_OPS
    ]
    statistics += [
        _statistic(
            f"buy_crop[{crop}]",
            tally.buy_crops[crop],
            tally.buys,
            tally.buy_crop_mean[crop],
            tally.buy_crop_variance[crop],
            sigma,
        )
        for crop in REFERENCE_CROPS
    ]
    statistics += [
        _statistic(
            f"plant_crop[{crop}]",
            tally.plant_crops[crop],
            tally.plants,
            tally.plant_crop_mean[crop],
            tally.plant_crop_variance[crop],
            sigma,
        )
        for crop in REFERENCE_CROPS
    ]
    return statistics


def _paired_statistic(
    name: str,
    native_count: int,
    reference_count: int,
    trials: int,
    sigma: float,
) -> dict[str, Any]:
    """Score the native count against the reference's own count on the same states.

    The two share a denominator because both agents were asked about the
    identical state sequence, so the difference of counts has variance
    2*n*p*(1-p) around zero and needs no independent-sample correction.
    """
    pooled = (native_count + reference_count) / (2 * trials) if trials else 0.0
    variance = 2 * trials * pooled * (1.0 - pooled)
    deviation = math.sqrt(variance)
    return {
        "statistic": name,
        "trials": trials,
        "count": native_count,
        "reference_count": reference_count,
        "rate": native_count / trials if trials else 0.0,
        "expected_rate": reference_count / trials if trials else 0.0,
        "expected_count": float(reference_count),
        "z": (native_count - reference_count) / deviation if deviation > 0.0 else 0.0,
        "rate_tolerance": sigma * deviation / trials if trials and deviation > 0.0 else 0.0,
    }


def _paired_statistics(native: _Tally, reference: _Tally, sigma: float) -> list[dict[str, Any]]:
    """Compare the two sources on the denominators the shared states fix."""
    for field in ("farmer_samples", "hand_slots", "buy_opportunities", "plant_opportunities"):
        if getattr(native, field) != getattr(reference, field):
            raise RuntimeError(
                f"lockstep broke: {field} was {getattr(native, field)} for the native agent "
                f"and {getattr(reference, field)} for the reference"
            )
    statistics = [
        _paired_statistic(
            "buy_seed_rate", native.buys, reference.buys, native.buy_opportunities, sigma
        ),
        _paired_statistic(
            "plant_rate", native.plants, reference.plants, native.plant_opportunities, sigma
        ),
    ]
    statistics += [
        _paired_statistic(
            f"farmer_op[{op}]",
            native.farmer_ops[op],
            reference.farmer_ops[op],
            native.farmer_samples,
            sigma,
        )
        for op in (*FARMER_OPS, PLANT)
    ]
    statistics += [
        _paired_statistic(
            f"hand_op[{op}]", native.hand_ops[op], reference.hand_ops[op], native.hand_slots, sigma
        )
        for op in FARMER_OPS
    ]
    return statistics


def _gate(statistics: list[dict[str, Any]], sigma: float) -> list[str]:
    """Reasons these statistics diverge, skipping the ones a Gaussian cannot score.

    Below five expected events in either tail the normal score is the wrong
    reference distribution, so gating on it would invent failures rather than
    find them. Those statistics are still reported, and `_power_failures`
    refuses a run that leaves any of them unscored everywhere, so silence here
    cannot be mistaken for evidence.
    """
    failures = []
    for record in statistics:
        expected = record["expected_count"]
        if min(expected, record["trials"] - expected) < MIN_NORMAL_COUNT:
            continue
        if abs(record["z"]) > sigma:
            failures.append(
                f"{record['statistic']} diverged: rate {record['rate']:.4f} against "
                f"{record['expected_rate']:.4f} over {record['trials']} trials, "
                f"z={record['z']:.2f} beyond {sigma}"
            )
    return failures


def _zero_inputs(games: int) -> dict[str, np.ndarray]:
    """Sampling inputs that turn the network path into a uniform legal draw.

    Zero logits and a rank-one zero quantity head make every network-sampled row
    uniform over whatever the native mask leaves legal, which is the co-player
    this audit wants: it hires hands, unlocks quadrants, buys animals and plants
    every crop, so the built-in under test is interrogated about states its own
    narrow policy would never visit.
    """
    rows = games * PLAYERS
    return {
        "unit_logits": np.zeros((rows, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.float32),
        "market_kind_logits": np.zeros((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.float32),
        "market_quantity_context": np.zeros((rows, MAX_MARKET_ORDERS, 1), dtype=np.float32),
        "quantity_kind_gate": np.zeros((1, N_MARKET_KINDS, 1), dtype=np.float32),
        "quantity_values": np.zeros((1, N_QUANTITIES, 1), dtype=np.float32),
        "quantity_bias": np.zeros((1, N_MARKET_KINDS, N_QUANTITIES), dtype=np.float32),
        "head_ids": np.zeros(rows, dtype=np.uint16),
        "unit_draws": np.zeros((rows, MAX_UNITS), dtype=np.float32),
        "market_kind_draws": np.zeros((rows, MAX_MARKET_ORDERS), dtype=np.float32),
        "market_quantity_draws": np.zeros((rows, MAX_MARKET_ORDERS), dtype=np.float32),
        "deterministic_rows": np.zeros(rows, dtype=np.bool_),
        "temperatures": np.ones(rows, dtype=np.float32),
    }


def _measure(
    module: Any,
    agent: str,
    scenario: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Play one wave and compare the built-in against the reference at every step."""
    seeds = np.arange(args.seed_start, args.seed_start + args.games, dtype=np.uint64)
    environment = module.BatchEnv(seeds)
    sampled = environment.sample_buffers()
    inputs = _zero_inputs(args.games)
    generator = np.random.default_rng(args.draw_seed)
    rows = args.games * PLAYERS

    # Alternate the seat under test across games. A binding that ignored the seat
    # index, or that walked rows player-major, would read the wrong farm for half
    # the wave and could not survive a single deterministic step.
    seats = np.arange(args.games) % PLAYERS
    builtin_rows = np.arange(args.games) * PLAYERS + seats
    network_rows = np.setdiff1d(np.arange(rows), builtin_rows)
    codes = np.zeros(rows, dtype=np.uint8)
    codes[builtin_rows] = BUILTIN_CODES[agent]
    # `explore` steps with no built-in anywhere, so the wave wanders through the
    # whole state space while `builtin_actions` reports what the built-in would
    # have done; `on_policy` hands the seat to the built-in, which is how the
    # league will actually use it and the only way to reach the stepping path.
    step_codes = codes if scenario == "on_policy" else np.zeros(rows, dtype=np.uint8)

    reference_agent = REFERENCE_AGENTS[agent]
    tallies = {source: _Tally() for source in SOURCES}
    divergences: list[dict[str, Any]] = []
    comparisons = 0
    builtin_logprob_nonzero = 0
    builtin_entropy_nonzero = 0
    network_entropy_total = 0.0
    transitions = 0

    for step in range(args.steps):
        snapshots = [json.loads(environment.snapshot_json(game)) for game in range(args.games)]
        queried = environment.builtin_actions(codes)
        for name in ("unit_draws", "market_kind_draws", "market_quantity_draws"):
            inputs[name][...] = generator.random(inputs[name].shape, dtype=np.float32)
        environment.sample_and_step_into(
            inputs["unit_logits"],
            inputs["market_kind_logits"],
            inputs["market_quantity_context"],
            inputs["quantity_kind_gate"],
            inputs["quantity_values"],
            inputs["quantity_bias"],
            inputs["head_ids"],
            inputs["unit_draws"],
            inputs["market_kind_draws"],
            inputs["market_quantity_draws"],
            inputs["deterministic_rows"],
            inputs["temperatures"],
            step_codes,
            sampled,
        )
        transitions += 1
        stepped_units = np.asarray(sampled["unit_actions"])
        stepped_kinds = np.asarray(sampled["market_kinds"])
        stepped_quantities = np.asarray(sampled["market_quantities"])
        if scenario == "on_policy":
            for field in ("unit_logprobs", "market_kind_logprobs", "market_quantity_logprobs"):
                builtin_logprob_nonzero += int(
                    np.count_nonzero(np.asarray(sampled[field])[builtin_rows])
                )
            entropy = np.asarray(sampled["entropy"])
            builtin_entropy_nonzero += int(np.count_nonzero(entropy[builtin_rows]))
            network_entropy_total += float(entropy[network_rows].sum())

        for game in range(args.games):
            seat = int(seats[game])
            row = game * PLAYERS + seat
            observation = _observation(snapshots[game], seat)
            units = active_unit_count(observation)
            reference = reference_agent(observation)
            native_values = [int(value) for value in queried["unit_actions"][row][:units]]
            native_orders = _orders(queried["market_kinds"][row], queried["market_quantities"][row])
            context = {
                "step": step,
                "game": int(seeds[game]),
                "seat": seat,
                "day": observation["day"],
                "money": observation["farms"][seat]["money"],
                "farmer": observation["farms"][seat]["farmer"],
                "hands": len(observation["farms"][seat]["hands"]),
                "seeds": observation["private"].get("seeds"),
                "shed": observation["private"].get("shed"),
            }
            if scenario == "on_policy":
                # Two code paths, one policy: the query the league admission
                # logic will read and the action the wave actually submitted
                # have to be the same action, including for the seeded random
                # agent, whose draw is a function of seed, step and seat.
                submitted_values = [int(value) for value in stepped_units[row][:units]]
                submitted_orders = _orders(stepped_kinds[row], stepped_quantities[row])
                if submitted_values != native_values or submitted_orders != native_orders:
                    divergences.append(
                        {
                            **context,
                            "kind": "query_vs_step",
                            "detail": (
                                f"queried units={native_values} orders={native_orders!r}; "
                                f"submitted units={submitted_values} orders={submitted_orders!r}"
                            ),
                        }
                    )
            if agent == "random":
                affordable = _affordable_crops(observation)
                available = _available_seeds(observation)
                actions = {
                    "native": (_unit_ops(native_values), native_orders),
                    "reference": (
                        _unit_ops(_reference_unit_values(reference, units)),
                        reference["market"][:MAX_MARKET_ORDERS],
                    ),
                }
                for source, (ops, orders) in actions.items():
                    violation = _support_violation(ops, orders, affordable, available)
                    if violation is not None:
                        divergences.append(
                            {**context, "kind": f"{source}_support", "detail": violation}
                        )
                        continue
                    tallies[source].observe(ops, orders, affordable, available)
            else:
                comparisons += 1
                difference = _exact_difference(native_values, native_orders, reference, units)
                if difference is not None:
                    divergences.append({**context, "kind": "action", "detail": difference})

        if len(divergences) >= args.max_divergences:
            break

    dones = np.asarray(sampled["dones"], dtype=np.bool_)
    record: dict[str, Any] = {
        "event": "measurement",
        "agent": agent,
        "scenario": scenario,
        "games": args.games,
        "seed_start": args.seed_start,
        "transitions": transitions,
        "samples": transitions * args.games,
        "exact_comparisons": comparisons,
        "abandoned": len(divergences) >= args.max_divergences,
        "terminated": bool(dones.all()),
        "divergences": divergences,
    }
    if scenario == "on_policy":
        record["builtin_logprob_nonzero"] = builtin_logprob_nonzero
        record["builtin_entropy_nonzero"] = builtin_entropy_nonzero
        record["network_mean_entropy"] = network_entropy_total / max(
            transitions * len(network_rows), 1
        )
    if agent == "random":
        record["hand_slots"] = tallies["native"].hand_slots
        record["statistics"] = {
            **{source: _exact_statistics(tallies[source], args.sigma) for source in SOURCES},
            "paired": _paired_statistics(tallies["native"], tallies["reference"], args.sigma),
        }
    return record


def _episode_outcomes(module: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Compare whole native episodes against `kaggle_environments` by final money.

    The per-step comparison above is strictly stronger, but it runs through this
    harness's own observation shaping, so it is the one check here that could in
    principle share a blind spot with the code it audits. This one has no glue at
    all: it plays the pairing natively and then plays the same seed through the
    official engine's own `run`, with the official engine picking its own agents,
    and demands the same money. It is only available for the deterministic
    built-ins, since an unseeded reference cannot reproduce an outcome.
    """
    if not args.episodes:
        return []
    # The asymmetric pairings pin the seat index down; `starter` against itself
    # puts both agents into the same market, which is a different price path; and
    # `pass` against itself must not move money at all, which is the cheapest way
    # to catch a native `pass` that leaked a market order.
    pairings = (("starter", "pass"), ("pass", "starter"), ("starter", "starter"), ("pass", "pass"))
    seeds = np.arange(args.seed_start, args.seed_start + args.episodes, dtype=np.uint64)
    outcomes = []
    for pairing in pairings:
        environment = module.BatchEnv(seeds)
        sampled = environment.sample_buffers()
        inputs = _zero_inputs(args.episodes)
        codes = np.tile(
            np.array([BUILTIN_CODES[name] for name in pairing], dtype=np.uint8), args.episodes
        )
        for _ in range(MAX_TRANSITIONS):
            environment.sample_and_step_into(
                inputs["unit_logits"],
                inputs["market_kind_logits"],
                inputs["market_quantity_context"],
                inputs["quantity_kind_gate"],
                inputs["quantity_values"],
                inputs["quantity_bias"],
                inputs["head_ids"],
                inputs["unit_draws"],
                inputs["market_kind_draws"],
                inputs["market_quantity_draws"],
                inputs["deterministic_rows"],
                inputs["temperatures"],
                codes,
                sampled,
            )
        native_money = np.asarray(sampled["final_money"], dtype=np.float64)
        for index, seed in enumerate(seeds):
            official = make(
                "kaggriculture",
                configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
                debug=False,
            )
            official.run(list(pairing))
            outcomes.append(
                {
                    "pairing": list(pairing),
                    "seed": int(seed),
                    "native_money": [float(value) for value in native_money[index]],
                    "official_money": [
                        float(farm["money"]) for farm in official.state[0].observation.farms
                    ],
                }
            )
    return outcomes


def _coverage_failures(records: list[dict[str, Any]]) -> list[str]:
    """Refuse a run whose measurements never reached the branches they claim to gate."""
    failures = []
    for record in records:
        if record["agent"] != "random" and not record["exact_comparisons"]:
            failures.append(f"{record['agent']}/{record['scenario']} compared no actions at all")
        if record["scenario"] == "on_policy" and record["network_mean_entropy"] <= 0.0:
            failures.append(
                f"{record['agent']}/on_policy left the network-sampled rows at zero entropy, "
                "so the built-in lane is not isolated from its neighbour"
            )
    return failures


def _power_failures(records: list[dict[str, Any]], min_expected: float) -> list[str]:
    """Refuse a run that never gathered enough events to score a statistic anywhere.

    Aggregated over the run rather than demanded of each scenario, because the
    scenarios reach different branches on purpose: `explore` is the only one that
    ever gives the seat hands, and `on_policy` the only one that leaves it
    solvent enough to buy seed at all -- uniform random play spends the farm
    broke inside a day. Requiring both of each would fail a correct
    implementation, and requiring neither would let a branch go unmeasured.
    """
    totals: Counter[tuple[str, str]] = Counter()
    for record in records:
        for block, statistics in record.get("statistics", {}).items():
            for statistic in statistics:
                totals[block, statistic["statistic"]] += statistic["expected_count"]
    return [
        f"{block} {name} saw {total:.1f} expected events across the whole run, below "
        f"{min_expected}; add the explore scenario or raise --games"
        for (block, name), total in sorted(totals.items())
        if total < min_expected
    ]


def _failures(
    records: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    """Every reason this audit does not clear its gates."""
    failures = []
    for record in records:
        label = f"{record['agent']}/{record['scenario']}"
        for divergence in record["divergences"]:
            failures.append(
                f"{label} {divergence['kind']} divergence at step {divergence['step']} "
                f"game {divergence['game']} seat {divergence['seat']}: {divergence['detail']}"
            )
        if record["abandoned"]:
            failures.append(f"{label} abandoned after {args.max_divergences} divergences")
        if record["transitions"] == args.steps == MAX_TRANSITIONS and not record["terminated"]:
            failures.append(f"{label} did not terminate at the competition horizon")
        if record.get("builtin_logprob_nonzero"):
            failures.append(
                f"{label} wrote {record['builtin_logprob_nonzero']} non-zero log-probabilities "
                "on built-in rows, which would enter the update as real likelihoods"
            )
        if record.get("builtin_entropy_nonzero"):
            failures.append(
                f"{label} wrote {record['builtin_entropy_nonzero']} non-zero entropies "
                "on built-in rows"
            )
        for block, statistics in record.get("statistics", {}).items():
            failures += [f"{label} {block}: {reason}" for reason in _gate(statistics, args.sigma)]
    for episode in episodes:
        if episode["native_money"] != episode["official_money"]:
            failures.append(
                f"{'+'.join(episode['pairing'])} on seed {episode['seed']} ended at "
                f"{episode['native_money']} natively and {episode['official_money']} in the "
                "official engine"
            )
    if not episodes and args.episodes:
        failures.append("no whole-episode outcome was compared against the official engine")
    return failures + _coverage_failures(records) + _power_failures(records, args.min_expected)


def main() -> None:
    args = parse_args()
    if args.games < 1:
        raise ValueError("--games must be positive")
    if not 1 <= args.steps <= MAX_TRANSITIONS:
        raise ValueError(f"--steps must be between 1 and {MAX_TRANSITIONS}")
    if args.sigma <= 0.0:
        raise ValueError("--sigma must be positive")
    if args.max_divergences < 1:
        raise ValueError("--max-divergences must be positive")
    if args.episodes < 0:
        raise ValueError("--episodes cannot be negative")
    agents = [name.strip() for name in args.agents.split(",") if name.strip()]
    scenarios = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    unknown = sorted({*agents} - {*BUILTIN_CODES}) + sorted({*scenarios} - {*SCENARIOS})
    if unknown:
        raise ValueError(f"unknown agents or scenarios: {unknown}")
    # `starter_agent` sells its whole carrot shed in one order, so the shed has
    # to fit in a single quantity bin or the native agent would have to clamp
    # and diverge from the reference. Both bounds are tunable, so this is
    # checked rather than assumed.
    if SHED_CAPACITY > MAX_MARKET_QUANTITY:
        raise SystemExit(
            f"shed capacity {SHED_CAPACITY} exceeds the {MAX_MARKET_QUANTITY} quantity bins, "
            "so a full-shed SELL cannot be one order"
        )

    module = load_native(release=not args.debug_build)
    records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    try:
        for agent in agents:
            for scenario in scenarios:
                record = _measure(module, agent, scenario, args)
                records.append(record)
                print(
                    f"{agent}/{scenario}: {record['samples']} samples, "
                    f"{len(record['divergences'])} divergences"
                )
        episodes = _episode_outcomes(module, args)
        for episode in episodes:
            print(
                f"{'+'.join(episode['pairing'])} seed {episode['seed']}: "
                f"native {episode['native_money']} official {episode['official_money']}"
            )
    finally:
        # Whatever was measured before a failure is still the evidence for what
        # failed, so the report is written on every exit path. It carries the
        # source identity because a parity measurement is only evidence about
        # the tree that produced it, like every other gate artifact here.
        if args.output is not None and (records or episodes):
            report = {
                "event": "builtin_parity_audit",
                "agents": agents,
                "scenarios": scenarios,
                "games": args.games,
                "seed_start": args.seed_start,
                "steps": args.steps,
                "draw_seed": args.draw_seed,
                "sigma": args.sigma,
                "min_expected": args.min_expected,
                "buy_seed_probability": BUY_SEED_PROBABILITY,
                "plant_probability": PLANT_PROBABILITY,
                "episodes": args.episodes,
                "uniform_op_probability": UNIFORM_OP_PROBABILITY,
                "source_identity": source_identity(),
                "measurements": records,
                "episode_outcomes": episodes,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    failures = _failures(records, episodes, args)
    worst = max(
        (
            abs(statistic["z"])
            for record in records
            for statistics in record.get("statistics", {}).values()
            for statistic in statistics
            if statistic["trials"]
        ),
        default=0.0,
    )
    print(
        f"\nexact parity over {sum(record['exact_comparisons'] for record in records)} "
        f"deterministic actions; worst distributional score {worst:.2f} of {args.sigma}"
    )
    if failures:
        raise SystemExit("built-in parity failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
