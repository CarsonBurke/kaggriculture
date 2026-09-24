#!/usr/bin/env python3
"""Paired official-engine panel for the public v16 market-order rewrite (Stage 0b).

Run one variant/opponent per queued CPU job. Each seed plays both seats. The
unmodified control uses the same wrapper/import path as the rewritten variants.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from kaggriculture.demonstrations import canonicalize_market_orders
from kaggriculture.opponents import normalize_opponent
from kaggriculture.provenance import file_sha256, source_identity


def _load_agent(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def _play(task: tuple[str, str, str, int, int]) -> dict[str, Any]:
    from kaggle_environments import make

    teacher_path, opponent_spec, variant, seed, seat = task
    teacher = _load_agent(teacher_path, f"teacher_{seed}_{seat}")
    opponent = (
        opponent_spec
        if opponent_spec in {"starter", "pass", "random"}
        else _load_agent(opponent_spec, f"opponent_{seed}_{seat}")
    )
    changed = 0
    old_orders = 0
    new_orders = 0

    def candidate(observation: dict[str, Any], configuration: Any = None) -> Any:
        nonlocal changed, old_orders, new_orders
        action = teacher(observation)
        if not isinstance(action, dict):
            return action
        market = list(action.get("market") or [])
        old_orders += len(market)
        if variant == "control":
            new_orders += len(market)
            return action
        canonical = canonicalize_market_orders(
            observation,
            market,
            sell_order="fixed" if variant == "hire_last" else variant,
            hire_last=variant == "hire_last",
        )
        new_orders += len(canonical)
        changed += canonical != market
        return {**action, "market": canonical}

    players = [candidate, opponent] if seat == 0 else [opponent, candidate]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=False)
    environment.run(players)
    terminal = environment.state
    statuses = [str(player.status) for player in terminal]
    banks = [float(farm["money"]) for farm in terminal[0].observation.farms]
    return {
        "seed": seed,
        "seat": seat,
        "own": banks[seat],
        "other": banks[1 - seat],
        "statuses": statuses,
        "changed_turns": changed,
        "old_orders": old_orders,
        "new_orders": new_orders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("control", "fixed", "impact", "hire_last"), required=True
    )
    parser.add_argument("--opponent", choices=("public-v27", "public-v16"), required=True)
    parser.add_argument("--seed-start", type=int, default=9_000_000)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, teacher_path = normalize_opponent("public-v16")
    _, opponent_spec = normalize_opponent(args.opponent)
    tasks = [
        (teacher_path, opponent_spec, args.variant, seed, seat)
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for seat in (0, 1)
    ]
    with mp.Pool(processes=args.workers) as pool:
        rows = pool.map(_play, tasks)
    broken = [row for row in rows if row["statuses"] != ["DONE", "DONE"]]
    if broken:
        raise RuntimeError(f"official engine did not finish: {broken[:2]}")
    report = {
        "variant": args.variant,
        "opponent": args.opponent,
        "teacher_sha256": file_sha256(Path(teacher_path)),
        "opponent_sha256": file_sha256(Path(opponent_spec)),
        "kaggle_environments_version": importlib.metadata.version("kaggle_environments"),
        "source_identity": source_identity(),
        "script_sha256": file_sha256(Path(__file__)),
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "seed_list": list(range(args.seed_start, args.seed_start + args.seeds)),
        "rows": rows,
        "score_rate": sum(
            (row["own"] > row["other"]) + 0.5 * (row["own"] == row["other"]) for row in rows
        )
        / len(rows),
        "money_mean": sum(row["own"] for row in rows) / len(rows),
        "margin_mean": sum(row["own"] - row["other"] for row in rows) / len(rows),
        "changed_turns": sum(row["changed_turns"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
