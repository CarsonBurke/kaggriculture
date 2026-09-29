#!/usr/bin/env python3
"""Validate full-game BC evidence before executing a frozen core PPO command."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def admission(
    reports: dict[str, dict], *, variant: str, artifact: Path, source_digest: str
) -> dict:
    """Bind both decoding panels to the exact clone, then apply adequacy floors."""
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    measurements = {}
    failures = []
    expected_pairs = None
    expected_opponents = None
    for mode in ("argmax", "sampled"):
        report = reports[mode]
        if report.get("complete") is not True or report.get("decoding") != mode:
            raise ValueError(f"{mode}: incomplete or mismatched evaluation")
        if report.get("source_identity", {}).get("sha256") != source_digest:
            raise ValueError(f"{mode}: evaluation used a different source tree")
        if report.get("games_per_opponent", 0) < 256:
            raise ValueError(f"{mode}: fewer than 256 full games per opponent")
        entry = report["artifacts"][variant]
        if entry.get("sha256") != digest or Path(entry["path"]).resolve() != artifact.resolve():
            raise ValueError(f"{mode}: evaluation does not bind this clone")
        if entry.get("source_identity", {}).get("sha256") != source_digest:
            raise ValueError(f"{mode}: clone has a different source identity")
        panels = entry["panels"]
        if not {"starter", "scripted-v27"}.issubset(panels):
            raise ValueError(f"{mode}: missing required admission opponents")
        if expected_opponents is None:
            expected_opponents = set(panels)
        elif set(panels) != expected_opponents:
            raise ValueError(f"{mode}: mismatched admission opponents")
        for opponent in panels:
            rows = panels[opponent]["games"]
            if len(rows) != report["games_per_opponent"]:
                raise ValueError(f"{mode}/{opponent}: incomplete game records")
            pairs = [(row["seed"], row["seat"]) for row in rows]
            if expected_pairs is None:
                expected_pairs = pairs
            if pairs != expected_pairs or len(set(pairs)) != len(pairs):
                raise ValueError(f"{mode}/{opponent}: mismatched or duplicate game records")
            if any(
                row["score"] not in (0.0, 0.5, 1.0)
                or not math.isfinite(row["money"])
                or row["seat"] not in (0, 1)
                for row in rows
            ):
                raise ValueError(f"{mode}/{opponent}: invalid game outcome")
            score = sum(row["score"] for row in rows) / len(rows)
            low_money = sum(row["money"] < 1000 for row in rows) / len(rows)
            measurements[f"{mode}/{opponent}"] = {"score": score, "low_money": low_money}
        starter_floor = 0.9 if mode == "argmax" else 0.75
        if measurements[f"{mode}/starter"]["score"] < starter_floor:
            failures.append(f"{mode} starter score below {starter_floor}")
        if mode == "argmax" and measurements[f"{mode}/scripted-v27"]["score"] < 0.5:
            failures.append("argmax V27 score below 0.5")
        if mode == "sampled":
            for opponent in panels:
                if measurements[f"{mode}/{opponent}"]["low_money"] > 0.25:
                    failures.append(f"sampled {opponent} bankrupt tail above 25%")
    return {
        "variant": variant,
        "artifact_sha256": digest,
        "admitted": not failures,
        "failures": failures,
        "measurements": measurements,
        "purpose": "initialization adequacy, not a claim of architecture superiority",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["source"])
    source_digest = json.loads((source / ".source-identity.json").read_text())["sha256"]
    command = manifest["variants"][args.variant]["ppo"]
    artifact = Path(command[command.index("--init-actor-from") + 1])
    reports = {
        mode: json.loads((args.manifest.parent / f"bc-{mode}.json").read_text())
        for mode in ("argmax", "sampled")
    }
    result = admission(
        reports, variant=args.variant, artifact=artifact, source_digest=source_digest
    )
    output = args.manifest.parent / f"admission-{args.variant}.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["admitted"]:
        raise SystemExit(75)
    # Keep the workload in MLQ's foreground process group and preserve its env.
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
