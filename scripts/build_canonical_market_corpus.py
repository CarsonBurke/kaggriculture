#!/usr/bin/env python3
"""Audit and, only on exact replay parity, build an A1 market-canonical BC corpus.

Each recorded teacher action is replayed against the pinned opponent in the
official engine, first unchanged and then with only its market list rewritten.
Every observation must equal the source trajectory at every step. A changed
trajectory is reported and cannot become a training corpus from the original
observations. The path rewrite proposed for A1 is outside this market-only arm.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.demonstrations import (
    canonicalize_market_orders,
    project_demonstration,
    verify_round_trip,
)
from kaggriculture.opponents import normalize_opponent
from kaggriculture.provenance import file_sha256, source_identity


def _load_raw(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()).decode("utf-8"))


def _observation(step: Any, seat: int, index: int) -> dict[str, Any]:
    value = dict(step[seat].observation)
    value.setdefault("step", index)
    return value


def _replay(
    raw: dict[str, Any], seed: int, seat: int, opponent: str, *, actions: list[dict[str, Any]]
) -> tuple[int | None, float, list[str]]:
    from kaggle_environments import make

    turn = 0

    def recorded(_observation: Any, _configuration: Any = None) -> dict[str, Any]:
        nonlocal turn
        action = actions[turn]
        turn += 1
        return action

    players = [recorded, opponent] if seat == 0 else [opponent, recorded]
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": len(actions) + 1, "seed": seed},
        debug=False,
    )
    environment.run(players)
    statuses = [str(player.status) for player in environment.state]
    if len(environment.steps) != len(actions) + 1 or turn != len(actions):
        raise RuntimeError(f"seed {seed} seat {seat}: replay length {turn}/{len(actions)}")
    first_difference = next(
        (
            index
            for index, step in enumerate(environment.steps[:-1])
            if _observation(step, seat, index) != raw["observations"][index]["observation"]
        ),
        None,
    )
    money = float(environment.state[seat].observation.farms[seat]["money"])
    return first_difference, money, statuses


def _projected_archive(raw: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    projections = []
    for index, (record, action) in enumerate(zip(raw["observations"], actions, strict=True)):
        observation = record["observation"]
        try:
            projected = project_demonstration(observation, action)
            verify_round_trip(observation, action, projected)
        except Exception as error:
            raise RuntimeError(f"projection failed at step {index}: {error}") from error
        projections.append(projected)

    names = (
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "unit_masks",
        "market_kind_masks",
        "market_quantity_masks",
        "unit_active",
        "market_active",
        "market_quantity_active",
    )
    updated = {name: np.stack([getattr(row, name) for row in projections]) for name in names}
    updated["raw_json_zlib"] = zlib.compress(
        json.dumps({**raw, "actions": actions}, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        ),
        level=6,
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--sell-order", choices=("fixed", "impact", "hire_last"), default="fixed"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "perturbation" in manifest:
        # Recovery corpora label with the teacher's actions, not the executed ones
        # this rebuild relabels, and replay against the unperturbed teacher diverges.
        raise ValueError(f"{source}: recovery (perturbed) corpora cannot be relabeled")
    opponent_label = manifest["opponent"]["label"]
    _, opponent = normalize_opponent(opponent_label)
    opponent_digest = manifest["opponent"].get("sha256")
    if opponent_digest is not None and file_sha256(Path(opponent)) != opponent_digest:
        raise RuntimeError("opponent file differs from corpus manifest")

    rows: list[dict[str, Any]] = []
    for record in manifest["episodes"]:
        path = source / record["file"]
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"source archive digest mismatch: {path}")
        raw = _load_raw(path)
        seed, seat = int(record["seed"]), int(record["seat"])
        original = raw["actions"]
        canonical = [
            {
                **action,
                "market": canonicalize_market_orders(
                    frame["observation"],
                    list(action.get("market") or []),
                    sell_order="fixed" if args.sell_order == "hire_last" else args.sell_order,
                    hire_last=args.sell_order == "hire_last",
                ),
            }
            for frame, action in zip(raw["observations"], original, strict=True)
        ]
        changed = sum(
            old.get("market") != new["market"] for old, new in zip(original, canonical, strict=True)
        )
        baseline_step, baseline_money, baseline_status = _replay(
            raw, seed, seat, opponent, actions=original
        )
        candidate_step, candidate_money, candidate_status = _replay(
            raw, seed, seat, opponent, actions=canonical
        )
        rows.append(
            {
                "file": record["file"],
                "seed": seed,
                "seat": seat,
                "changed_turns": changed,
                "baseline_first_difference": baseline_step,
                "canonical_first_difference": candidate_step,
                "baseline_money": baseline_money,
                "canonical_money": candidate_money,
                "baseline_statuses": baseline_status,
                "canonical_statuses": candidate_status,
            }
        )

    exact = all(
        row["baseline_first_difference"] is None
        and row["canonical_first_difference"] is None
        and row["baseline_statuses"] == ["DONE", "DONE"]
        and row["canonical_statuses"] == ["DONE", "DONE"]
        for row in rows
    )
    report = {
        "source_manifest_sha256": file_sha256(manifest_path),
        "source": str(source),
        "sell_order": args.sell_order,
        "exact_replay": exact,
        "episodes": len(rows),
        "changed_turns": sum(row["changed_turns"] for row in rows),
        "baseline_mismatches": sum(row["baseline_first_difference"] is not None for row in rows),
        "canonical_mismatches": sum(row["canonical_first_difference"] is not None for row in rows),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))
    if not exact or args.output_dir is None:
        return

    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as name:
        staging = Path(name)
        output_records = []
        for record in manifest["episodes"]:
            raw = _load_raw(source / record["file"])
            actions = [
                {
                    **action,
                    "market": canonicalize_market_orders(
                        frame["observation"],
                        list(action.get("market") or []),
                        sell_order="fixed" if args.sell_order == "hire_last" else args.sell_order,
                        hire_last=args.sell_order == "hire_last",
                    ),
                }
                for frame, action in zip(raw["observations"], raw["actions"], strict=True)
            ]
            file = f"episode-{record['seed']:08d}-seat{record['seat']}.npz"
            np.savez_compressed(staging / file, **_projected_archive(raw, actions))
            output_records.append({**record, "file": file, "sha256": file_sha256(staging / file)})
        output_manifest = {
            **manifest,
            "format_version": 2,
            "episodes": output_records,
            "canonicalization": {
                "scope": "market_only",
                "sell_order": args.sell_order,
                "source_manifest_sha256": report["source_manifest_sha256"],
                "source_identity": source_identity(),
                "replay_report_sha256": file_sha256(args.report),
            },
        }
        (staging / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
        os.rename(staging, destination)


if __name__ == "__main__":
    main()
