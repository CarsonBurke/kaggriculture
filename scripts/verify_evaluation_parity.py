#!/usr/bin/env python3
"""Gate lockstep evaluation on exact serial result parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kaggriculture.provenance import source_identity

_RESULT_FIELDS = (
    "candidate_reward",
    "opponent_reward",
    "margin",
    "outcome",
    "candidate_status",
    "opponent_status",
    "environment_done",
    "complete",
    "steps",
    "expected_steps",
    "error",
)
_REPORT_FIELDS = (
    "agent",
    "opponent_provenance",
    "seed_start",
    "seed_count",
    "paired_seats",
    "device",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _games(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for game in report["games"]:
        key = (int(game["seed"]), int(game["candidate_seat"]))
        if key in indexed:
            raise ValueError(f"duplicate game key {key}")
        indexed[key] = game
    return indexed


def main() -> None:
    args = _parse_args()
    left_bytes = args.left.read_bytes()
    right_bytes = args.right.read_bytes()
    left = json.loads(left_bytes)
    right = json.loads(right_bytes)
    left_games = _games(left)
    right_games = _games(right)
    mismatches: list[dict[str, Any]] = []
    for label, report, games in (
        ("serial", left, left_games),
        ("lockstep", right, right_games),
    ):
        if report.get("valid_for_selection") is not True:
            mismatches.append({"field": f"{label}.valid_for_selection", "value": False})
        seed_count = report.get("seed_count")
        expected_games = seed_count * 2 if type(seed_count) is int else None
        if expected_games is None or expected_games < 1 or len(games) != expected_games:
            mismatches.append(
                {
                    "field": f"{label}.games",
                    "expected": expected_games,
                    "actual": len(games),
                }
            )
        incomplete = [
            key
            for key, game in games.items()
            if game.get("complete") is not True or game.get("error") is not None
        ]
        if incomplete:
            mismatches.append({"field": f"{label}.complete_games", "games": incomplete})
        if report.get("workers") != 1:
            mismatches.append(
                {"field": f"{label}.workers", "expected": 1, "actual": report.get("workers")}
            )
    if left.get("batch_size") != 1:
        mismatches.append(
            {"field": "serial.batch_size", "expected": 1, "actual": left.get("batch_size")}
        )
    lockstep_batch_size = right.get("batch_size")
    if type(lockstep_batch_size) is not int or lockstep_batch_size <= 1:
        mismatches.append(
            {
                "field": "lockstep.batch_size",
                "expected": "> 1",
                "actual": lockstep_batch_size,
            }
        )

    left_provenance = left.get("artifact_provenance", {})
    right_provenance = right.get("artifact_provenance", {})
    left_digest = left_provenance.get("sha256")
    right_digest = right_provenance.get("sha256")
    if left_digest != right_digest or left_digest is None:
        mismatches.append({"field": "artifact_sha256", "left": left_digest, "right": right_digest})
    left_source = left_provenance.get("source_identity")
    right_source = right_provenance.get("source_identity")
    if left_source != right_source or left_source is None:
        mismatches.append(
            {"field": "artifact_source_identity", "left": left_source, "right": right_source}
        )
    verifier_source = source_identity()
    if left_source != verifier_source:
        mismatches.append(
            {
                "field": "verifier_source_identity",
                "artifact": left_source,
                "verifier": verifier_source,
            }
        )
    for field in _REPORT_FIELDS:
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value != right_value:
            mismatches.append({"field": field, "left": left_value, "right": right_value})
    if set(left_games) != set(right_games):
        mismatches.append(
            {
                "field": "game_keys",
                "left_only": sorted(set(left_games) - set(right_games)),
                "right_only": sorted(set(right_games) - set(left_games)),
            }
        )
    for key in sorted(set(left_games) & set(right_games)):
        for field in _RESULT_FIELDS:
            left_value = left_games[key].get(field)
            right_value = right_games[key].get(field)
            if left_value != right_value:
                mismatches.append(
                    {"game": key, "field": field, "left": left_value, "right": right_value}
                )

    result = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "artifact_sha256": left_digest,
        "left_report_sha256": hashlib.sha256(left_bytes).hexdigest(),
        "right_report_sha256": hashlib.sha256(right_bytes).hexdigest(),
        "source_identity": verifier_source,
        "device": left.get("device"),
        "serial_batch_size": left.get("batch_size"),
        "lockstep_batch_size": right.get("batch_size"),
        "architecture": left_provenance.get("architecture"),
        "games": len(left_games),
        "compared_fields": list(_RESULT_FIELDS),
        "compared_report_fields": list(_REPORT_FIELDS),
        "mismatches": mismatches,
        "valid": not mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit("evaluation results differ")


if __name__ == "__main__":
    main()
