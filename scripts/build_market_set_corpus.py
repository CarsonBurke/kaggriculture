#!/usr/bin/env python3
"""Relabel a public-v16 corpus with official effective market fills.

Mirror games have both seats recorded. For other opponents the paired action
stream is reconstructed by rerunning both public agents, with every teacher
observation and action checked against the archive before labels are emitted.
An official debug replay then traces successful per-unit market commits and
projects interface-3 targets from those actual fills.
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
from kaggle_environments import make

from kaggriculture.actions import (
    UnitAction,
    apply_unit_shed_effect,
    apply_unit_tile_effect,
    copy_tile_grid,
)
from kaggriculture.demonstrations import project_demonstration, verify_round_trip
from kaggriculture.market_set import MarketSetOrder, project_effective_market_set
from kaggriculture.market_set_trace import trace_effective_market_run, trace_effective_market_step
from kaggriculture.opponents import normalize_opponent
from kaggriculture.provenance import file_sha256, source_identity


def _load(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        raw = json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()))
        arrays = {key: archive[key] for key in archive.files if key != "raw_json_zlib"}
    return raw, arrays


def _post_unit_shed(observation: dict[str, Any], unit_actions: np.ndarray) -> dict[str, int]:
    player = int(observation["player"])
    farm = observation["farms"][player]
    unit_count = 1 + len(farm.get("hands") or [])
    shed = dict(observation["private"]["shed"])
    tiles = copy_tile_grid(farm["tiles"])
    for index in range(unit_count):
        action = UnitAction(int(unit_actions[index]))
        apply_unit_shed_effect(observation, index, action, shed, tiles)
        apply_unit_tile_effect(observation, index, action, tiles)
    return shed


def _observation(environment: Any, seat: int, step: int) -> dict[str, Any]:
    observed = dict(environment.state[seat].observation)
    observed.setdefault("step", step)
    return observed


def _extract_live_seat(steps: list[Any], seat: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    observations = []
    actions = []
    projections = []
    for step in range(len(steps) - 1):
        observation = dict(steps[step][seat].observation)
        observation.setdefault("step", step)
        action = steps[step + 1][seat].action
        projected = project_demonstration(observation, action)
        verify_round_trip(observation, action, projected)
        observations.append(
            {
                "observation": observation,
                "opponent_private": steps[step][1 - seat].observation.get("private"),
            }
        )
        actions.append(action)
        projections.append(projected)
    fields = (
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
    return (
        {"observations": observations, "actions": actions},
        {name: np.stack([getattr(row, name) for row in projections]) for name in fields},
    )


def _process_seed(
    seed: int,
    paths: dict[int, Path],
    output: Path,
    order: MarketSetOrder,
    *,
    teacher: str | None = None,
    opponent: str | None = None,
    fresh_live: bool = False,
) -> list[dict[str, Any]]:
    raw, old = {}, {}
    seats = tuple(sorted(paths))
    if seats not in ((0,), (1,), (0, 1)):
        raise ValueError(f"seed {seed}: expected one or both recorded seats")
    for seat in seats:
        raw[seat], old[seat] = _load(paths[seat])
    steps = len(raw[seats[0]]["actions"])
    if any(len(raw[seat]["actions"]) != steps for seat in seats):
        raise ValueError(f"seed {seed}: seat action lengths differ")
    if fresh_live:
        if len(seats) != 1 or teacher is None or opponent is None:
            raise ValueError("fresh live extraction needs one teacher seat and both runnable specs")
        seat = seats[0]
        players = [teacher, opponent] if seat == 0 else [opponent, teacher]
        live = make(
            "kaggriculture", configuration={"seed": seed, "episodeSteps": steps + 1}, debug=True
        )
        effective_by_step, live_steps = trace_effective_market_run(live, players)
        raw[seat], old[seat] = _extract_live_seat(live_steps, seat)
        paired_actions = [
            [live_steps[step + 1][player].action for player in (0, 1)] for step in range(steps)
        ]
    elif len(seats) == 2:
        paired_actions = [[raw[seat]["actions"][step] for seat in (0, 1)] for step in range(steps)]
    else:
        if teacher is None or opponent is None:
            raise ValueError("single-seat replay needs teacher and opponent runnable specs")
        seat = seats[0]
        players = [teacher, opponent] if seat == 0 else [opponent, teacher]
        reference = make("kaggriculture", configuration={"seed": seed, "episodeSteps": steps + 1})
        reference.run(players)
        paired_actions = []
        for step in range(steps):
            observed = dict(reference.steps[step][seat].observation)
            observed.setdefault("step", step)
            if observed != raw[seat]["observations"][step]["observation"]:
                raise RuntimeError(
                    f"seed {seed} seat {seat}: paired opponent replay diverged at {step}"
                )
            actions = [reference.steps[step + 1][player].action for player in (0, 1)]
            if actions[seat] != raw[seat]["actions"][step]:
                raise RuntimeError(
                    f"seed {seed} seat {seat}: paired teacher action diverged at {step}"
                )
            paired_actions.append(actions)
    if not fresh_live:
        environment = make(
            "kaggriculture", configuration={"seed": seed, "episodeSteps": steps + 1}, debug=True
        )
        environment.reset(2)
    values: dict[int, list[np.ndarray]] = {seat: [] for seat in seats}
    masks: dict[int, list[np.ndarray]] = {seat: [] for seat in seats}
    active: dict[int, list[np.ndarray]] = {seat: [] for seat in seats}
    canonical: dict[int, list[list[list[Any]]]] = {seat: [] for seat in seats}
    legacy: dict[int, list[Any]] = {seat: [] for seat in seats}
    changed = 0
    for step in range(steps):
        actions = paired_actions[step]
        if fresh_live:
            effective = effective_by_step[step]
        else:
            for seat in seats:
                expected = raw[seat]["observations"][step]["observation"]
                if _observation(environment, seat, step) != expected:
                    raise RuntimeError(f"seed {seed} seat {seat} baseline diverged at step {step}")
            effective, _ = trace_effective_market_step(environment, actions)
        for seat in seats:
            observation = raw[seat]["observations"][step]["observation"]
            post_unit_shed = _post_unit_shed(observation, old[seat]["unit_actions"][step])
            compiled, factors = project_effective_market_set(
                observation, effective[seat], order=order, post_unit_shed=post_unit_shed
            )
            values[seat].append(factors.values)
            masks[seat].append(factors.masks)
            active[seat].append(factors.active)
            canonical[seat].append(compiled)
            canonical_action = {**actions[seat], "market": compiled}
            projected = project_demonstration(observation, canonical_action)
            verify_round_trip(observation, canonical_action, projected)
            if not np.array_equal(projected.unit_actions, old[seat]["unit_actions"][step]):
                raise RuntimeError(f"seed {seed} seat {seat} unit projection changed at {step}")
            legacy[seat].append(projected)
            changed += compiled != actions[seat].get("market", [])
    records = []
    for seat in seats:
        arrays = dict(old[seat])
        arrays["market_set_values"] = np.stack(values[seat])
        arrays["market_set_masks"] = np.stack(masks[seat])
        arrays["market_set_active"] = np.stack(active[seat])
        # Old market factors remain for action-conditioned auxiliary consumers,
        # now compiled from the interface-3 choices. They are not policy
        # decisions under interface 3.
        for name in (
            "market_kinds",
            "market_quantities",
            "market_kind_masks",
            "market_quantity_masks",
        ):
            arrays[name] = np.stack([getattr(row, name) for row in legacy[seat]])
        arrays["market_active"] = np.zeros_like(arrays["market_active"])
        arrays["market_quantity_active"] = np.zeros_like(arrays["market_quantity_active"])
        raw[seat]["market_set_canonical_actions"] = canonical[seat]
        arrays["raw_json_zlib"] = zlib.compress(
            json.dumps(raw[seat], separators=(",", ":"), allow_nan=False).encode(), level=6
        )
        name = f"episode-{seed:08d}-seat{seat}.npz"
        np.savez_compressed(output / name, **arrays)
        records.append(
            {
                "file": name,
                "seed": seed,
                "seat": seat,
                "steps": steps,
                "sha256": file_sha256(output / name),
            }
        )
    print(f"seed {seed} seats {seats}: {changed} canonical market turns", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sell-order", choices=("fixed", "impact"), required=True)
    parser.add_argument("--hire-last", action="store_true")
    parser.add_argument(
        "--allow-order-changing-relabel",
        action="store_true",
        help="acknowledge Stage-0b's large teacher regression under current fixed-order compilers",
    )
    parser.add_argument("--seed-count", type=int, default=64)
    args = parser.parse_args()
    if not args.allow_order_changing_relabel:
        parser.error(
            "current market-set compilers change teacher outcomes sharply; "
            "--allow-order-changing-relabel is required for exploratory extraction"
        )
    order = MarketSetOrder(args.sell_order, args.hire_last)
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mirror = manifest["teacher"] == manifest["opponent"]
    _, teacher = normalize_opponent(manifest["teacher"]["label"])
    _, opponent = normalize_opponent(manifest["opponent"]["label"])
    for spec, record in ((teacher, manifest["teacher"]), (opponent, manifest["opponent"])):
        digest = record.get("sha256")
        if digest is not None and file_sha256(Path(spec)) != digest:
            raise ValueError(f"public agent digest changed: {record['label']}")
    by_seed: dict[int, dict[int, Path]] = {}
    for entry in manifest["episodes"]:
        seed = int(entry["seed"])
        if seed >= int(manifest["seed_start"]) + args.seed_count:
            continue
        path = source / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"digest mismatch: {path}")
        by_seed.setdefault(seed, {})[int(entry["seat"])] = path
    if len(by_seed) != args.seed_count or any(set(paths) != {0, 1} for paths in by_seed.values()):
        raise ValueError("source lacks both recorded teacher seats for requested seeds")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as name:
        staging = Path(name)
        records = []
        for seed, paths in sorted(by_seed.items()):
            if mirror:
                records.extend(_process_seed(seed, paths, staging, order))
            else:
                for seat in (0, 1):
                    records.extend(
                        _process_seed(
                            seed,
                            {seat: paths[seat]},
                            staging,
                            order,
                            teacher=teacher,
                            opponent=opponent,
                            fresh_live=manifest["opponent"]["label"] == "random",
                        )
                    )
        result = {
            **{key: value for key, value in manifest.items() if key != "episodes"},
            "format_version": 3,
            "episode_count": len(by_seed),
            "episodes": records,
            "market_set": {"sell_order": args.sell_order, "hire_last": args.hire_last},
            "order_equivalence": "not_guaranteed; Stage-0b showed severe v27 regressions",
            "source_replay_mode": (
                "fresh_live_random" if manifest["opponent"]["label"] == "random" else "verified"
            ),
            "source_manifest_sha256": file_sha256(manifest_path),
            "extractor_source_identity": source_identity(),
        }
        (staging / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        os.rename(staging, output)


if __name__ == "__main__":
    main()
