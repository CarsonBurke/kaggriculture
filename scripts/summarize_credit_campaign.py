#!/usr/bin/env python3
"""Collect completed credit/valuation evidence, retaining failures and valid pruned runs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from evaluate_architecture_campaign import OPPONENTS, paired_comparison
from summarize_architecture_campaign import league_summary, read_jsonl

from kaggriculture.provenance import file_sha256


def training_evidence(job: dict, rows: list[dict], evaluated_iteration: int | None) -> dict:
    """Require real actor updates and an aligned checkpoint, including genuine culls."""
    updates = sum(row.get("actor_updates", 0) for row in rows)
    last = rows[-1] if rows else {}
    attempt = job.get("attempts", [{}])[-1] if job.get("attempts") else {}
    pruned = (
        job.get("state") == "failed"
        and attempt.get("exitCode") == 75
        and last.get("autocull_state", {}).get("stale_observations", 0) >= 30
    )
    reasons = []
    if job.get("state") != "succeeded" and not pruned:
        reasons.append("training_not_successful_or_verified_pruned")
    if not rows:
        reasons.append("missing_training_metrics")
    elif not all("actor_updates" in row for row in rows):
        reasons.append("missing_actor_update_counts")
    if not updates:
        reasons.append("no_actor_updates")
    if evaluated_iteration is None:
        reasons.append("missing_completed_evaluation")
    elif evaluated_iteration != last.get("iteration"):
        reasons.append("checkpoint_metrics_mismatch")
    return {
        "training_state": "pruned" if pruned else job.get("state"),
        "actor_updates": updates,
        "last_iteration": last.get("iteration"),
        "evaluated_iteration": evaluated_iteration,
        "valid_completed_learning_evidence": not reasons,
        "reasons": reasons,
        "scope": "One training seed; pruning is a completed partial budget, not convergence",
    }


def read_evaluation(path: Path, mode: str, initial_digest: str) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (
        not value.get("complete")
        or value.get("decoding") != mode
        or value.get("artifacts", {}).get("bc", {}).get("sha256") != initial_digest
    ):
        raise ValueError(f"incomplete, wrong-decoding or unmatched-BC evaluation: {path}")
    return value


def validate_checkpoint_binding(evaluation: dict, checkpoint: Path, source_digest: str) -> None:
    artifact = evaluation["artifacts"]["ppo"]
    if (
        Path(artifact["path"]).resolve() != checkpoint.resolve()
        or artifact["sha256"] != file_sha256(checkpoint)
        or artifact["source_identity"]["sha256"] != source_digest
    ):
        raise ValueError("evaluation checkpoint path, bytes or source do not match the arm")


def panels(evaluation: dict) -> dict:
    return {
        opponent: evaluation["artifacts"]["ppo"]["panels"][opponent]["games"]
        for opponent in OPPONENTS
    }


def compact_evaluation(evaluation: dict) -> dict:
    return {
        "decoding": evaluation["decoding"],
        "artifacts": {
            name: {
                "sha256": artifact["sha256"],
                "iteration": artifact["iteration"],
                "panels": {key: value["summary"] for key, value in artifact["panels"].items()},
            }
            for name, artifact in evaluation["artifacts"].items()
        },
        "paired_to_bc": evaluation["comparisons_to_first"]["ppo"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.campaign.read_text())
    baseline_path = Path(manifest["baseline"])
    if file_sha256(baseline_path) != manifest["baseline_sha256"]:
        raise ValueError("baseline campaign changed after the comparison was queued")
    baseline = json.loads(baseline_path.read_text())
    initial_digest = manifest["initial_actor"]["sha256"]
    if baseline["initial_actor"]["sha256"] != initial_digest:
        raise ValueError("baseline and candidates do not share the same actor initialization")
    directory = args.campaign.parent
    report = {
        "campaign": str(args.campaign.resolve()),
        "campaign_sha256": file_sha256(args.campaign),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "baseline": str(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "plan": manifest["plan"],
        "jobs": {},
        "arms": {},
    }
    sources = {"control": baseline, **dict.fromkeys(manifest["arms"], manifest)}
    full_evaluations = {}
    for arm, source in sources.items():
        control = arm == "control"
        label = "train" if control else f"train-{arm}"
        train = source["jobs"][label]
        command = train["command"]
        run = Path(command[command.index("--run-dir") + 1])
        job = json.loads(
            subprocess.check_output(
                ["mlq", "show", str(train["submission"]["id"]), "--json"], text=True
            )
        )
        report["jobs"][arm] = job
        metrics = run / "metrics.jsonl"
        rows = (
            [row for row in read_jsonl(metrics) if "iteration_seconds" in row]
            if metrics.exists()
            else []
        )
        active = [row for row in rows if row.get("actor_updates", 0)]
        entry = {
            "run": str(run),
            "source_sha256": source["source_identity"]["sha256"],
            "training": {
                "last_iteration": rows[-1]["iteration"] if rows else None,
                "elapsed_hours": rows[-1]["elapsed_hours"] if rows else None,
                "actor_active_waves": len(active),
                "median_actor_active_iteration_seconds": statistics.median(
                    row["iteration_seconds"] for row in active
                )
                if active
                else None,
                "league_last_50": league_summary(rows[-50:]),
            },
            "evaluations": {},
        }
        full_evaluations[arm] = {}
        for mode in ("argmax", "sampled"):
            evaluation_path = (
                baseline_path.parent / f"evaluation-{mode}.json"
                if control
                else directory / f"{arm}-{mode}.json"
            )
            try:
                evaluation = read_evaluation(evaluation_path, mode, initial_digest)
                if evaluation is not None:
                    validate_checkpoint_binding(
                        evaluation, run / "latest.pt", source["source_identity"]["sha256"]
                    )
                    eval_label = f"evaluate-{mode}" if control else f"evaluate-{arm}-{mode}"
                    eval_job = json.loads(
                        subprocess.check_output(
                            [
                                "mlq",
                                "show",
                                str(source["jobs"][eval_label]["submission"]["id"]),
                                "--json",
                            ],
                            text=True,
                        )
                    )
                    if eval_job.get("state") != "succeeded":
                        raise ValueError("evaluation job is not successful")
                    full_evaluations[arm][mode] = evaluation
                    entry["evaluations"][mode] = {
                        **compact_evaluation(evaluation),
                        "evaluation_sha256": file_sha256(evaluation_path),
                    }
            except ValueError as error:
                entry["evaluations"][mode] = {"error": str(error)}
                evaluation = None
            evidence = training_evidence(
                job, rows, evaluation["artifacts"]["ppo"]["iteration"] if evaluation else None
            )
            entry["evaluations"].setdefault(mode, {})["learning_evidence"] = evidence
            reference = full_evaluations["control"].get(mode)
            if not control and evaluation and reference:
                entry["evaluations"][mode]["paired_to_control"] = paired_comparison(
                    panels(evaluation), panels(reference)
                )
                entry["evaluations"][mode]["both_learning_runs_valid"] = (
                    evidence["valid_completed_learning_evidence"]
                    and report["arms"]["control"]["evaluations"][mode]["learning_evidence"][
                        "valid_completed_learning_evidence"
                    ]
                )
        modes = full_evaluations[arm]
        if len(modes) == 2:
            if (
                modes["argmax"]["artifacts"]["ppo"]["sha256"]
                != modes["sampled"]["artifacts"]["ppo"]["sha256"]
            ):
                raise ValueError(f"decoding evaluations used different checkpoints for {arm}")
            entry["sampled_minus_argmax"] = paired_comparison(
                panels(modes["sampled"]), panels(modes["argmax"])
            )
        report["arms"][arm] = entry
    destination = directory / "outcome-summary.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
