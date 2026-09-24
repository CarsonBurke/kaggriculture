#!/usr/bin/env python3
"""Measure turn-level exact engine equivalence of market canonicalization.

Uses the mirror teacher corpus so both seats' recorded actions are available.
For each changed market list, a branch from the original state applies that
candidate with the opponent's original action; both resulting observations,
statuses and rewards must match. The baseline advances on original actions.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from kaggle_environments import make

from kaggriculture.demonstrations import canonicalize_market_orders
from kaggriculture.provenance import file_sha256, source_identity


def _raw(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()).decode("utf-8"))


def _branch(environment: Any) -> Any:
    clone = copy.copy(environment)
    clone.state = copy.deepcopy(environment.state)
    # The interpreter derives its step index from len(steps), so preserve the
    # history length while replacing only the mutable current state.
    clone.steps = [*environment.steps[:-1], clone.state]
    clone.logs = environment.logs.copy()
    return clone


def _signature(states: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "observation": dict(state.observation),
            "reward": state.reward,
            "status": str(state.status),
        }
        for state in states
    ]


def audit_seed(source: Path, records: dict[int, dict[str, Any]], variant: str) -> dict[str, Any]:
    raws = {seat: _raw(source / record["file"]) for seat, record in records.items()}
    actions = [raws[seat]["actions"] for seat in (0, 1)]
    steps = len(actions[0])
    if len(actions[1]) != steps:
        raise ValueError("seat action streams have different lengths")
    seed = int(records[0]["seed"])
    environment = make("kaggriculture", configuration={"seed": seed, "episodeSteps": steps + 1})
    environment.reset(2)
    changed = exact = 0
    changed_by_seat = [0, 0]
    exact_by_seat = [0, 0]
    examples = []
    for index in range(steps):
        for seat in (0, 1):
            observed = dict(environment.state[seat].observation)
            observed.setdefault("step", index)
            if observed != raws[seat]["observations"][index]["observation"]:
                raise RuntimeError(f"seed {seed} seat {seat} baseline diverged at step {index}")
        originals = [actions[seat][index] for seat in (0, 1)]
        candidates = []
        for seat in (0, 1):
            original = originals[seat]
            canonical = canonicalize_market_orders(
                raws[seat]["observations"][index]["observation"],
                list(original.get("market") or []),
                sell_order="fixed" if variant == "hire_last" else variant,
                hire_last=variant == "hire_last",
            )
            candidates.append({**original, "market": canonical})

        branches = {}
        for seat in (0, 1):
            if candidates[seat]["market"] == originals[seat].get("market"):
                continue
            changed += 1
            changed_by_seat[seat] += 1
            branch = _branch(environment)
            trial = originals.copy()
            trial[seat] = candidates[seat]
            branches[seat] = _signature(branch.step(trial))
        expected = _signature(environment.step(originals))
        for seat, actual in branches.items():
            if actual == expected:
                exact += 1
                exact_by_seat[seat] += 1
            elif len(examples) < 8:
                examples.append(
                    {
                        "step": index,
                        "seat": seat,
                        "original_market": originals[seat].get("market"),
                        "canonical_market": candidates[seat]["market"],
                        "original_money": expected[seat]["observation"]["farms"][seat]["money"],
                        "canonical_money": actual[seat]["observation"]["farms"][seat]["money"],
                    }
                )
    return {
        "seed": seed,
        "changed": changed,
        "exact": exact,
        "changed_by_seat": changed_by_seat,
        "exact_by_seat": exact_by_seat,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/bc-v16-current-mirror-64"))
    parser.add_argument("--variant", choices=("fixed", "impact", "hire_last"), required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    by_seed: dict[int, dict[int, dict[str, Any]]] = {}
    for record in manifest["episodes"]:
        seed, seat = int(record["seed"]), int(record["seat"])
        if seed >= int(manifest["seed_start"]) + args.seeds:
            continue
        path = source / record["file"]
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"archive digest mismatch: {path}")
        by_seed.setdefault(seed, {})[seat] = record
    if len(by_seed) != args.seeds or any(set(records) != {0, 1} for records in by_seed.values()):
        raise RuntimeError("mirror corpus lacks both recorded seats for requested seeds")
    rows = [audit_seed(source, by_seed[seed], args.variant) for seed in sorted(by_seed)]
    report = {
        "variant": args.variant,
        "source_manifest_sha256": file_sha256(manifest_path),
        "kaggle_environments_version": importlib.metadata.version("kaggle_environments"),
        "source_identity": source_identity(),
        "script_sha256": file_sha256(Path(__file__)),
        "seeds": sorted(by_seed),
        "changed": sum(row["changed"] for row in rows),
        "exact": sum(row["exact"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
