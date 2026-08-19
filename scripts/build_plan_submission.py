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

#: Steps whose market orders the plan is better without. Values, not a search, so
#: the shipped agent is exactly what the reported measurement scored.
DEFAULT_CANCELLED_STEPS = (220, 224, 293, 604)

PATCH = """

# --- kraggiculture edit ------------------------------------------------------
# Four steps whose market orders this plan is measurably better without. Found by
# an exhaustive per-step cancellation scan in a bit-exact port of this engine and
# validated on disjoint seeds: 90-93% of games won against the unedited plan over
# three independent 256-game sets, median margin +259 dollars, and +190 dollars in
# absolute bank against `starter`, `pass`, and `random` alike -- so the orders are
# unprofitable outright, not merely bad in a contested market.
_UNEDITED_AGENT = agent
_CANCELLED_MARKET_STEPS = frozenset([{steps}])


def agent(obs, configuration=None):
    action = _UNEDITED_AGENT(obs, configuration)
    step = int(_get(obs, "step", 0) or 0)
    if step in _CANCELLED_MARKET_STEPS and isinstance(action, dict):
        action["market"] = []
    return action


def _kaggle_submission_entrypoint(obs, configuration=None):
    return agent(obs, configuration)
"""


def patched_source(source: Path, steps: tuple[int, ...]) -> str:
    text = source.read_text(encoding="utf-8")
    if "_CANCELLED_MARKET_STEPS" in text:
        raise SystemExit(f"{source} is already patched")
    if "\ndef agent(" not in text:
        raise SystemExit(f"{source} has no top-level agent() to wrap")
    return text + PATCH.format(steps=", ".join(str(step) for step in sorted(steps)))


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
    parser.add_argument("--verify-seeds", type=int, default=16)
    parser.add_argument("--verify-seed-start", type=int, default=7_000_000)
    parser.add_argument("--workers", type=int, default=4)
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
    text = patched_source(source, steps)

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
        "main_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "archive": str(args.output),
        "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "official_engine_verification": measurement,
    }
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
