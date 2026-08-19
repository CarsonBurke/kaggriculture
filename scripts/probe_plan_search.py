#!/usr/bin/env python3
"""Score edits to the public v27 plan against the unedited agent it must beat.

The public agent is an open-loop 719-step table plus two run-time adaptations
(weed repair and sell-slot reordering).  `scripted-v27` in the engine carries
both, and it ties itself exactly, so taking its emitted action as the base makes
the margin of any edit the honest value of that edit -- zero is a tie with the
strongest agent we can see, and anything positive beats it.

Every candidate is scored on the same seeds as the base, so the comparison is
paired: the engine is bit-exact with `kaggle_environments`, so a margin measured
here is a margin on the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract_v27_script import compile_script

from kaggriculture.actions import MarketKind
from kaggriculture.opponents import PUBLIC_V27_OPPONENT
from kaggriculture.rust_env import load_native

PLAYERS = 2
#: `BuiltinAgent::from_code`: 0 samples from the network, which for a seat with
#: no network is every unit passing, and the rest is `BUILTIN_AGENT_ORDER`.
CODES = {"network": 0, "pass": 1, "random": 2, "starter": 3, "scripted-v27": 4}
#: Market kinds that raise money rather than spend it; the only orders whose
#: quantity an edit can grow without needing cash it may not have.
SELL_KINDS = np.asarray(
    [int(kind) for kind in MarketKind if kind.name.startswith("SELL_")], dtype=np.uint8
)
#: `QUANTITY_BINS` is 1..100, so a bin index is its quantity minus one.
MAX_QUANTITY_BIN = 99

#: A plan edit rewrites seat zero's submitted row in place for one step.
Edit = Callable[[int, np.ndarray, np.ndarray, np.ndarray], None]


class Plan:
    """The public agent's compiled table, indexed by step."""

    def __init__(self, path: Path) -> None:
        rows, self.digest = compile_script(path)
        self.units = np.asarray([row["units"] for row in rows], dtype=np.uint8)
        self.kinds = np.asarray([row["market_kinds"] for row in rows], dtype=np.uint8)
        self.quantities = np.asarray([row["market_quantities"] for row in rows], dtype=np.uint8)
        self.steps = int(self.units.shape[0])


def make_edit(spec: str, plan: Plan) -> Edit:
    """Compile one edit specification into a row rewriter.

    Specifications are `name`, `name:value`, or several joined by `+`, which lets
    a scan build on an already accepted set instead of restarting from the plan.
    Unknown names raise rather than silently scoring the base twice, which would
    read as a worthless edit.
    """

    if "+" in spec:
        parts = [make_edit(piece, plan) for piece in spec.split("+") if piece]

        def composed(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            for part in parts:
                part(step, units, kinds, quantities)

        return composed

    name, _, raw_value = spec.partition(":")
    if name == "base":
        return lambda step, units, kinds, quantities: None
    if name == "pass-step":
        # Idle every unit at these steps, leaving the market row alone: the unit
        # half of the same slack question the cancellation scan asks of orders.
        idle = frozenset(int(part) for part in raw_value.split(",") if part)

        def pass_step(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            if step in idle:
                units[:, 0, :] = np.uint8(0)

        return pass_step
    if name in {"stop-kind", "cap-kind"}:
        # The engine decodes market orders up to the first STOP, so removing one
        # mid-row would silently drop every order behind it. Both edits therefore
        # rewrite the whole row and compact the survivors forward.
        target, _, bound = raw_value.partition("@")
        kind = np.uint8(int(MarketKind[target]))
        cap = np.uint8(int(bound) - 1) if bound else None
        stop = np.uint8(int(MarketKind.STOP))

        def rewrite_kind(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            row_kinds = kinds[:, 0, :]
            row_quantities = quantities[:, 0, :]
            # Slots at or after the first STOP are already inert; treating them as
            # live would promote dead bytes into real orders when compacting.
            active = np.cumprod(row_kinds != stop, axis=1).astype(bool)
            matched = active & (row_kinds == kind)
            if cap is not None:
                np.copyto(row_quantities, np.minimum(row_quantities, cap), where=matched)
                return
            keep = active & ~matched
            order = np.argsort(~keep, axis=1, kind="stable")
            compacted_kinds = np.take_along_axis(row_kinds, order, axis=1)
            compacted_quantities = np.take_along_axis(row_quantities, order, axis=1)
            beyond = np.arange(row_kinds.shape[1])[None, :] >= keep.sum(axis=1)[:, None]
            compacted_kinds[beyond] = stop
            kinds[:, 0, :] = compacted_kinds
            quantities[:, 0, :] = compacted_quantities

        return rewrite_kind
    if name == "raw-market":
        # Replace the adaptation's reordered sell slots with the table's own,
        # shifted by `value` steps: negative acts earlier than the agent does.
        shift = int(raw_value or 0)

        def raw_market(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            source = min(max(step + shift, 0), plan.steps - 1)
            kinds[:, 0, :] = plan.kinds[source]
            quantities[:, 0, :] = plan.quantities[source]

        return raw_market
    if name == "raw-units":
        shift = int(raw_value or 0)

        def raw_units(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            source = min(max(step + shift, 0), plan.steps - 1)
            units[:, 0, :] = plan.units[source]

        return raw_units
    if name == "sell-bin-delta":
        # Sell more or less per order than the agent asks for. The engine clamps
        # a sell to the stock on hand, so a raise is free where stock is short.
        delta = int(raw_value)

        def sell_bin_delta(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            selling = np.isin(kinds[:, 0, :], SELL_KINDS)
            shifted = quantities[:, 0, :].astype(np.int16) + delta
            np.clip(shifted, 0, MAX_QUANTITY_BIN, out=shifted)
            quantities[:, 0, :] = np.where(selling, shifted.astype(np.uint8), quantities[:, 0, :])

        return sell_bin_delta
    if name == "sell-hold-until":
        # Refuse every sale before a step, then sell the accumulated stock at
        # the agent's own schedule: tests whether v27 sells too early.
        until = int(raw_value)

        def sell_hold_until(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            if step >= until:
                return
            selling = np.isin(kinds[:, 0, :], SELL_KINDS)
            kinds[:, 0, :] = np.where(selling, np.uint8(int(MarketKind.STOP)), kinds[:, 0, :])

        return sell_hold_until
    if name == "stop-step":
        # Cancel every market order at these steps. A step whose cancellation
        # costs nothing carries slack; one that costs dollars is load-bearing, so
        # this scan is what tells a search where to spend its candidates.
        targets = frozenset(int(part) for part in raw_value.split(",") if part)

        def stop_step(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            if step in targets:
                kinds[:, 0, :] = np.uint8(int(MarketKind.STOP))

        return stop_step
    if name == "step-bin-delta":
        # Resize only this step's sells, leaving the rest of the plan intact.
        target, _, amount = raw_value.partition("@")
        at_step = int(target)
        delta = int(amount)

        def step_bin_delta(
            step: int, units: np.ndarray, kinds: np.ndarray, quantities: np.ndarray
        ) -> None:
            if step != at_step:
                return
            selling = np.isin(kinds[:, 0, :], SELL_KINDS)
            shifted = quantities[:, 0, :].astype(np.int16) + delta
            np.clip(shifted, 0, MAX_QUANTITY_BIN, out=shifted)
            quantities[:, 0, :] = np.where(selling, shifted.astype(np.uint8), quantities[:, 0, :])

        return step_bin_delta
    raise SystemExit(f"unknown edit {spec!r}")


def play(
    module: Any,
    seeds: np.ndarray,
    *,
    edit: Edit,
    base: str,
    opponent: str,
    steps: int,
) -> np.ndarray:
    """Play one wave and return final money as `(games, PLAYERS)`."""

    games = int(seeds.shape[0])
    environment = module.BatchEnv(np.ascontiguousarray(seeds, dtype=np.uint64))
    codes = np.empty(games * PLAYERS, dtype=np.uint8)
    codes[0::PLAYERS] = CODES[base]
    codes[1::PLAYERS] = CODES[opponent]

    money = np.zeros((games, PLAYERS), dtype=np.float64)
    for step in range(steps):
        # Exactly one call per step: the scripted agent advances its weed repair
        # as a side effect, so a second call inside a step is not a read.
        emitted = environment.builtin_actions(codes)
        units = np.ascontiguousarray(
            np.asarray(emitted["unit_actions"]).reshape(games, PLAYERS, -1)
        )
        kinds = np.ascontiguousarray(
            np.asarray(emitted["market_kinds"]).reshape(games, PLAYERS, -1)
        )
        quantities = np.ascontiguousarray(
            np.asarray(emitted["market_quantities"]).reshape(games, PLAYERS, -1)
        )
        edit(step, units, kinds, quantities)
        result = environment.step_factors(
            units,
            kinds,
            quantities,
            # Both seats submit a raw dict rather than a policy-masked action, so
            # the engine owes them the interpreter's own clamping.
            True,
        )
        money = np.asarray(result["final_money"], dtype=np.float64)
    return money


def summarize(money: np.ndarray) -> dict[str, Any]:
    own = money[:, 0]
    other = money[:, 1]
    total = own + other
    # The engine's own objective, so a summary cannot flatter an edit that only
    # denies the opponent without banking anything itself.
    relative = np.where(total == 0.0, 0.0, (own - other) / np.maximum(total, 1e-9))
    margin = own - other
    return {
        "games": int(own.shape[0]),
        "own_money_mean": float(own.mean()),
        "own_money_median": float(np.median(own)),
        "opponent_money_mean": float(other.mean()),
        "relative_score_mean": float(relative.mean()),
        "win_rate": float((own > other).mean()),
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path(PUBLIC_V27_OPPONENT))
    parser.add_argument("--base", default="scripted-v27", choices=sorted(CODES))
    parser.add_argument("--opponent", default="scripted-v27", choices=sorted(CODES))
    parser.add_argument("--edit", nargs="+", default=["base"])
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=90_001)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = load_native()
    plan = Plan(args.agent)
    if plan.digest != module.V27_SOURCE_SHA256:
        raise SystemExit(
            f"plan digest {plan.digest} does not match the engine table {module.V27_SOURCE_SHA256}"
        )
    if plan.steps != int(module.V27_STEPS):
        raise SystemExit(f"plan has {plan.steps} steps, engine table has {module.V27_STEPS}")

    seeds = np.arange(args.seed_start, args.seed_start + args.games, dtype=np.uint64)
    rows = []
    for spec in args.edit:
        summary = summarize(
            play(
                module,
                seeds,
                edit=make_edit(spec, plan),
                base=args.base,
                opponent=args.opponent,
                steps=plan.steps,
            )
        )
        summary["edit"] = spec
        rows.append(summary)
        print(json.dumps(summary), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "probe": "plan-search",
                    "agent": str(args.agent),
                    "agent_sha256": plan.digest,
                    "base": args.base,
                    "opponent": args.opponent,
                    "steps": plan.steps,
                    "seed_start": args.seed_start,
                    "rows": rows,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
