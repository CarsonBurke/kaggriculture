#!/usr/bin/env python3
"""Evaluate exact actor-wave checkpoints; run this evaluator only through MLQ.

`--final` adds each arm's clean endpoint, labelled as such: under a wall-clock
budget arms stop at different waves, and the endpoint is what the budget buys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MILESTONES = (25, 50, 100, 200, 300, 400, 500)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_checkpoints(run: Path, milestones=MILESTONES, *, final: bool = False) -> dict:
    """Select by journal boundaries, never nearest checkpoint or latest endpoint.

    The endpoint is selected only when asked for, under its own `final` key: the
    checkpoint written at the journal's last iteration, never a substitute.
    """
    journal = run / "metrics.jsonl"
    labels = [*map(str, milestones), *(("final",) if final else ())]
    if not journal.exists():
        return {"status": "no journal", "milestones": dict.fromkeys(labels)}
    rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    iterations = [r["iteration"] for r in rows]
    if iterations != sorted(set(iterations)):
        raise ValueError(f"{run}: duplicate or unordered journal iterations")
    complete_prefix = iterations == list(range(1, len(rows) + 1))
    totals = {"actor_updates": 0, "states": 0, "warmup_iterations": 0, "warmup_states": 0}
    selected = dict.fromkeys(labels)
    for index, row in enumerate(rows):
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
        endpoint = final and index == len(rows) - 1
        if wave not in milestones and not endpoint:
            continue
        path = run / f"checkpoint-{row['iteration']:06d}.pt"
        if not path.is_file():
            continue
        record = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "iteration": row["iteration"],
            "actor_waves": wave,
            "cumulative": dict(totals) if complete_prefix else None,
            "observed_journal_totals": dict(totals),
            "accounting_complete": complete_prefix,
        }
        if endpoint:
            selected["final"] = record
        if wave not in milestones:
            continue
        if selected[str(wave)] is not None:
            raise ValueError(f"{run}: multiple checkpoints at actor wave {wave}")
        selected[str(wave)] = record
    return {
        "status": "journal available",
        "journal_sha256": digest(journal),
        "last_iteration": iterations[-1] if rows else None,
        "last_panel_state": rows[-1].get("architecture_panel_state") if rows else None,
        "milestones": selected,
    }


def source_digest_of(source: Path) -> str:
    return json.loads((source / ".source-identity.json").read_text())["sha256"]


def training_source_digests(manifest: dict) -> dict[str, str]:
    """The frozen source each arm trained under, checked against its checkpoints.

    An arm may name its own `source`, so a campaign can score a reference run
    trained from an earlier snapshot beside new arms. Every arm is still
    evaluated under the manifest's one source, so each plays through the same
    inference code.
    """
    default = manifest["source"]
    return {
        name: source_digest_of(Path(variant.get("source", default)))
        for name, variant in manifest["variants"].items()
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
    from evaluate_architecture_campaign import OPPONENTS, artifact_argument

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=4_730_000)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument(
        "--milestones",
        type=lambda value: tuple(int(wave) for wave in value.split(",") if wave),
        default=MILESTONES,
        help="comma-separated exact actor waves; empty with --final for endpoints alone",
    )
    parser.add_argument("--final", action="store_true", help="also compare clean endpoints")
    parser.add_argument(
        "--baseline",
        type=artifact_argument,
        help="LABEL=PATH artifact evaluated first at every milestone, the paired reference",
    )
    parser.add_argument(
        "--reference-opponent",
        action="append",
        type=artifact_argument,
        default=[],
        help="LABEL=PATH neural opponent passed through to the panel",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["source"])
    source_digest = source_digest_of(source)
    training_sources = training_source_digests(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, commands in manifest["variants"].items():
        command = commands["ppo"]
        run = Path(command[command.index("--run-dir") + 1])
        records[name] = select_checkpoints(run, args.milestones, final=args.final)
    labels = [*map(str, args.milestones), *(("final",) if args.final else ())]
    baseline = None
    if args.baseline is not None:
        label, path = args.baseline
        if label in records:
            raise ValueError("the baseline label collides with an arm")
        baseline = {"label": label, "path": str(path.resolve()), "sha256": digest(path)}
    references = [(label, str(path.resolve())) for label, path in args.reference_opponent]
    opponents = [*OPPONENTS, *(label for label, _ in references)]
    index = {
        "manifest_sha256": digest(args.manifest),
        "source_sha256": source_digest,
        "training_source_sha256": training_sources,
        "milestones": labels,
        "baseline": baseline,
        "reference_opponents": dict(references),
        "arms": records,
        "comparison": "Exact actor waves; separate optimizer-update and warmup accounting",
    }
    (args.output_dir / "selection.json").write_text(json.dumps(index, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{source / 'src'}:{manifest['native_source']}"
    evaluated = 0
    for milestone in labels:
        candidates = {
            n: r["milestones"][milestone]
            for n, r in records.items()
            if r["milestones"][milestone] is not None
        }
        comparable = len(candidates) + (baseline is not None) >= 2
        result = {
            "milestone": milestone,
            "actor_waves": None if milestone == "final" else int(milestone),
            "candidates": candidates,
            "missing_arms": sorted(set(records) - set(candidates)),
            "panels": {},
            "status": "ready" if comparable else "incomplete comparison",
        }
        for name, record in candidates.items():
            verify_checkpoint(record, training_sources[name])
        stem = "final" if milestone == "final" else f"wave-{int(milestone):04d}"
        path = args.output_dir / f"{stem}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        if not comparable:
            continue
        for mode in ("argmax", "sampled"):
            output = args.output_dir / f"{stem}-{mode}.json"
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
            if baseline is not None:
                command += ["--artifact", f"{baseline['label']}={baseline['path']}"]
            for name, record in candidates.items():
                command += ["--artifact", f"{name}={record['path']}"]
            for label, reference in references:
                command += ["--reference-opponent", f"{label}={reference}"]
            subprocess.run(command, env=env, check=True)
            report = json.loads(output.read_text())
            if not report.get("complete") or report["source_identity"]["sha256"] != source_digest:
                raise ValueError(f"incomplete or wrong-source report: {output}")
            if report["opponents"] != opponents:
                raise ValueError(f"evaluation opponents differ from the frozen panel: {output}")
            expected = dict(
                {n: r["sha256"] for n, r in candidates.items()},
                **({baseline["label"]: baseline["sha256"]} if baseline is not None else {}),
            )
            for name, sha256 in expected.items():
                if report["artifacts"][name]["sha256"] != sha256:
                    raise ValueError(f"evaluation artifact changed: {name}")
            result["panels"][mode] = {
                "report": str(output.resolve()),
                "sha256": digest(output),
                "opponent_summaries": {
                    name: {
                        opponent: panel["summary"]
                        for opponent, panel in report["artifacts"][name]["panels"].items()
                    }
                    for name in expected
                },
            }
        result["status"] = "evaluated available arms"
        path.write_text(json.dumps(result, indent=2) + "\n")
        evaluated += 1
    if not evaluated:
        raise RuntimeError("No milestone has two valid artifacts; comparison remains unresolved")


if __name__ == "__main__":
    main()
