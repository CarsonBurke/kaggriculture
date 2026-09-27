#!/usr/bin/env python3
"""Evaluate exact actor-wave checkpoints; run this evaluator only through MLQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MILESTONES = (25, 50, 100, 200, 300, 400, 500)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_checkpoints(run: Path, milestones=MILESTONES) -> dict:
    """Select by journal boundaries, never nearest checkpoint or latest endpoint."""
    journal = run / "metrics.jsonl"
    if not journal.exists():
        return {"status": "no journal", "milestones": {str(w): None for w in milestones}}
    rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    iterations = [r["iteration"] for r in rows]
    if iterations != sorted(set(iterations)):
        raise ValueError(f"{run}: duplicate or unordered journal iterations")
    complete_prefix = iterations == list(range(1, len(rows) + 1))
    totals = {"actor_updates": 0, "states": 0, "warmup_iterations": 0, "warmup_states": 0}
    selected = {str(w): None for w in milestones}
    for row in rows:
        states = row.get("states")
        updates = row.get("actor_updates")
        if not isinstance(states, (int, float)) or not isinstance(updates, (int, float)):
            raise ValueError(f"{run}: missing step accounting at {row['iteration']}")
        totals["states"] += states
        totals["actor_updates"] += updates
        if row.get("critic_warmup_active"):
            totals["warmup_iterations"] += 1
            totals["warmup_states"] += states
        wave = row.get("architecture_panel_state", {}).get("actor_waves")
        if wave not in milestones:
            continue
        path = run / f"checkpoint-{row['iteration']:06d}.pt"
        if not path.is_file():
            continue
        if selected[str(wave)] is not None:
            raise ValueError(f"{run}: multiple checkpoints at actor wave {wave}")
        selected[str(wave)] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "iteration": row["iteration"],
            "actor_waves": wave,
            "cumulative": dict(totals) if complete_prefix else None,
            "observed_journal_totals": dict(totals),
            "accounting_complete": complete_prefix,
        }
    return {
        "status": "journal available",
        "journal_sha256": digest(journal),
        "last_iteration": iterations[-1] if rows else None,
        "last_panel_state": rows[-1].get("architecture_panel_state") if rows else None,
        "milestones": selected,
    }


def verify_checkpoint(record: dict, source_digest: str) -> None:
    """Load only selected checkpoints, checking their embedded boundary metadata."""
    import torch

    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"checkpoint changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["iteration"] != record["iteration"]
        or payload["metrics"]["architecture_panel_state"]["actor_waves"] != record["actor_waves"]
        or payload.get("source_identity", {}).get("sha256") != source_digest
    ):
        raise ValueError(f"checkpoint metadata disagrees with journal/source: {path}")
    record["source_sha256"] = source_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=4_730_000)
    parser.add_argument("--games", type=int, default=512)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["source"])
    source_digest = json.loads((source / ".source-identity.json").read_text())["sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, commands in manifest["variants"].items():
        command = commands["ppo"]
        run = Path(command[command.index("--run-dir") + 1])
        records[name] = select_checkpoints(run)
    index = {
        "manifest_sha256": digest(args.manifest),
        "source_sha256": source_digest,
        "milestones": list(MILESTONES),
        "arms": records,
        "comparison": "Exact actor waves; optimizer updates and warmup exposure reported separately",
    }
    (args.output_dir / "selection.json").write_text(json.dumps(index, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{source / 'src'}:{manifest['native_source']}"
    for wave in MILESTONES:
        candidates = {
            n: r["milestones"][str(wave)]
            for n, r in records.items()
            if r["milestones"][str(wave)] is not None
        }
        result = {
            "actor_waves": wave,
            "candidates": candidates,
            "missing_arms": sorted(set(records) - set(candidates)),
            "panels": {},
            "status": "incomplete comparison" if len(candidates) < 2 else "ready",
        }
        for record in candidates.values():
            verify_checkpoint(record, source_digest)
        path = args.output_dir / f"wave-{wave:04d}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        if len(candidates) < 2:
            continue
        for mode in ("argmax", "sampled"):
            output = args.output_dir / f"wave-{wave:04d}-{mode}.json"
            command = [
                sys.executable,
                str(source / "scripts/evaluate_architecture_campaign.py"),
                "--output",
                str(output),
                "--seed-start",
                str(args.seed_start),
                "--games",
                str(args.games),
                "--decoding",
                mode,
            ]
            for name, record in candidates.items():
                command += ["--artifact", f"{name}={record['path']}"]
            subprocess.run(command, env=env, check=True)
            report = json.loads(output.read_text())
            if not report.get("complete") or report["source_identity"]["sha256"] != source_digest:
                raise ValueError(f"incomplete or wrong-source report: {output}")
            if report["opponents"] != ["starter", "scripted-v27", "scripted-v16"]:
                raise ValueError(f"expected only the three builtin opponents: {output}")
            for name, record in candidates.items():
                if report["artifacts"][name]["sha256"] != record["sha256"]:
                    raise ValueError(f"evaluation artifact changed: {name}")
            result["panels"][mode] = {
                "report": str(output.resolve()),
                "sha256": digest(output),
                "opponent_summaries": {
                    name: {
                        opponent: panel["summary"]
                        for opponent, panel in report["artifacts"][name]["panels"].items()
                    }
                    for name in candidates
                },
            }
        result["status"] = "evaluated available arms"
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
