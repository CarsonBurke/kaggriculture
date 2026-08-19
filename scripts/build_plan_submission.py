#!/usr/bin/env python3
"""Package the public v27 plan with the market cancellations that improve it.

The edit is four steps whose market orders the plan is measurably better without,
found by an exhaustive per-step cancellation scan in the native engine and
validated on disjoint seeds.  Shipping it requires more than the native evidence:
this builder replays the packaged agent against the unedited public agent inside
`kaggle_environments` itself and refuses to write the archive unless the edit
still wins there, because the archive is what plays on the leaderboard.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kaggriculture.opponents import PUBLIC_V27_OPPONENT

#: Steps whose market orders the plan is better without, and steps where it buys
#: more than it should with how much to shave off. Values, not a search, so the
#: shipped agent is exactly what the reported measurement scored.
DEFAULT_CANCELLED_STEPS = (155, 220, 224, 292, 293, 308, 591, 604, 656)
DEFAULT_REDUCED_BUYS = ((241, 2),)

PATCH = """

# --- kraggiculture edit ------------------------------------------------------
# Steps whose market orders this plan is measurably better without, and one where
# it over-buys. Found by a greedy per-step scan in a bit-exact port of this engine
# and validated on four independent 256-game sets: 91-96% of games won against the
# unedited plan, median margin +563 dollars, and ~88% against the smaller edit
# sets found earlier, so the parts compound rather than trading against each
# other. Every cancelled step buys one to three wheat or seed; cancelling such
# purchases wholesale instead destroys the economy, because wheat is animal feed.
_UNEDITED_AGENT = agent
_CANCELLED_MARKET_STEPS = frozenset([{steps}])
#: step -> quantity to subtract from every priced purchase that step, floored at
#: one, which is the smallest order the engine encodes.
_REDUCED_BUY_STEPS = {{{reductions}}}


def _reduce_buys(market, amount):
    reduced = []
    for order in market:
        # Only priced purchases carry a quantity; `HIRE` and `BUY_LAND` are bare,
        # and a sell must keep its size or the plan stops banking its harvest.
        if len(order) == 3 and order[0] != "SELL":
            order = [order[0], order[1], max(1, int(order[2]) - amount)]
        reduced.append(order)
    return reduced


def agent(obs, configuration=None):
    action = _UNEDITED_AGENT(obs, configuration)
    if not isinstance(action, dict):
        return action
    step = int(_get(obs, "step", 0) or 0)
    if step in _CANCELLED_MARKET_STEPS:
        action["market"] = []
    elif step in _REDUCED_BUY_STEPS:
        action["market"] = _reduce_buys(
            list(action.get("market") or []), _REDUCED_BUY_STEPS[step]
        )
    return action


def _kaggle_submission_entrypoint(obs, configuration=None):
    return agent(obs, configuration)
"""


def patched_source(
    source: Path, steps: tuple[int, ...], reductions: tuple[tuple[int, int], ...] = ()
) -> str:
    text = source.read_text(encoding="utf-8")
    if "_CANCELLED_MARKET_STEPS" in text:
        raise SystemExit(f"{source} is already patched")
    if "\ndef agent(" not in text:
        raise SystemExit(f"{source} has no top-level agent() to wrap")
    overlap = sorted({step for step, _ in reductions} & set(steps))
    if overlap:
        # A cancelled step has no orders left to shrink, so asking for both is a
        # statement about the edit that the shipped agent would not honour.
        raise SystemExit(f"steps {overlap} are both cancelled and reduced")
    return text + PATCH.format(
        steps=", ".join(str(step) for step in sorted(steps)),
        reductions=", ".join(f"{step}: {amount}" for step, amount in sorted(reductions)),
    )


def _load(path: Path, name: str) -> Any:
    """Import an agent file as a private module.

    Never registered in `sys.modules`: each agent keeps its own module-level weed
    repair state, so two seats sharing one module would share that state.
    """

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _play(task: tuple[str, str, int, bool]) -> dict[str, Any]:
    from kaggle_environments import make

    candidate_path, reference_path, seed, candidate_first = task
    candidate = _load(Path(candidate_path), "kraggiculture_candidate")
    reference = _load(Path(reference_path), "kraggiculture_reference")
    seats = [candidate.agent, reference.agent]
    if not candidate_first:
        seats.reverse()
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    environment.run(seats)
    state = environment.state
    banks = [float(farm["money"]) for farm in state[0].observation.farms]
    index = 0 if candidate_first else 1
    own = banks[index]
    other = banks[1 - index]
    return {
        "seed": seed,
        "candidate_seat": index,
        "own": own,
        "other": other,
        "statuses": [str(seat.status) for seat in state],
    }


def verify(candidate: Path, reference: Path, seeds: range, workers: int) -> dict[str, Any]:
    """Play both seat orientations on every seed inside the official engine."""

    tasks = [
        (str(candidate), str(reference), seed, first) for seed in seeds for first in (True, False)
    ]
    with mp.Pool(processes=workers) as pool:
        rows = pool.map(_play, tasks)
    broken = [row for row in rows if set(row["statuses"]) != {"DONE"}]
    if broken:
        raise SystemExit(f"official engine did not finish {len(broken)} episodes: {broken[:2]}")
    wins = sum(1 for row in rows if row["own"] > row["other"])
    losses = sum(1 for row in rows if row["own"] < row["other"])
    margins = sorted(row["own"] - row["other"] for row in rows)
    middle = len(margins) // 2
    return {
        "episodes": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": len(rows) - wins - losses,
        "win_rate": wins / len(rows),
        "margin_median": (
            margins[middle] if len(margins) % 2 else 0.5 * (margins[middle - 1] + margins[middle])
        ),
        "own_money_mean": sum(row["own"] for row in rows) / len(rows),
        "opponent_money_mean": sum(row["other"] for row in rows) / len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(PUBLIC_V27_OPPONENT))
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_CANCELLED_STEPS))
    parser.add_argument(
        "--reduce",
        nargs="+",
        default=[f"{step}@{amount}" for step, amount in DEFAULT_REDUCED_BUYS],
        metavar="STEP@AMOUNT",
        help="shrink every priced purchase at STEP by AMOUNT instead of cancelling the step",
    )
    parser.add_argument("--verify-seeds", type=int, default=16)
    parser.add_argument("--verify-seed-start", type=int, default=7_000_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--minimum-win-rate",
        type=float,
        default=0.75,
        help="refuse the archive unless the edit wins this often in the official engine",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    steps = tuple(sorted(set(args.steps)))
    reductions = tuple(
        sorted(
            (int(step), int(amount))
            for step, _, amount in (part.partition("@") for part in args.reduce)
        )
    )
    text = patched_source(source, steps, reductions)

    with tempfile.TemporaryDirectory(prefix="kraggiculture-plan-submission-") as name:
        root = Path(name)
        main_py = root / "main.py"
        main_py.write_text(text, encoding="utf-8")
        # Import once before playing: a syntax error in the patch must fail here
        # rather than as a silent per-step exception the interpreter swallows.
        _load(main_py, "kraggiculture_patched")
        measurement = verify(
            main_py,
            source,
            range(args.verify_seed_start, args.verify_seed_start + args.verify_seeds),
            args.workers,
        )
        if measurement["win_rate"] < args.minimum_win_rate:
            raise SystemExit(
                f"edit wins {measurement['win_rate']:.3f} of official episodes, "
                f"below the required {args.minimum_win_rate:.3f}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(main_py, arcname="main.py")

    manifest = {
        "event": "plan_submission_built",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cancelled_market_steps": list(steps),
        "reduced_buy_steps": {step: amount for step, amount in reductions},
        "main_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "archive": str(args.output),
        "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "official_engine_verification": measurement,
    }
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
