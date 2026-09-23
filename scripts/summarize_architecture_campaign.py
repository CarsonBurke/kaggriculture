#!/usr/bin/env python3
"""Summarize completed matched evaluations without running any model."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from evaluate_architecture_campaign import OPPONENTS, paired_comparison

from kaggriculture.provenance import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def league_summary(rows: list[dict]) -> dict:
    """Game-weighted realized difficulty; never confuse it with fixed-panel skill."""
    categories = defaultdict(lambda: {"games": 0, "score_sum": 0.0})
    for row in rows:
        for key, games in row.items():
            if not key.startswith("league_opponent_") or not key.endswith("_games"):
                continue
            prefix = key.removesuffix("_games")
            score = row[f"{prefix}_score_rate"]
            for group in (
                "all",
                "category:" + row[f"{prefix}_category"],
                "role:" + row.get(f"{prefix}_role", "stratified"),
                "opponent:" + prefix.removeprefix("league_opponent_"),
            ):
                categories[group]["games"] += games
                categories[group]["score_sum"] += games * score
    return {
        group: {"games": item["games"], "score_rate": item["score_sum"] / item["games"]}
        for group, item in categories.items()
        if item["games"]
    }


def learning_evidence(entry: dict, *, training_job: dict, evaluation_job: dict) -> dict:
    """Separate completed actor learning from merely evaluating a saved checkpoint."""
    training = entry.get("training")
    evaluation = entry.get("evaluation", {}).get("ppo")
    states = {"training": training_job.get("state"), "evaluation": evaluation_job.get("state")}
    reasons = [
        f"{stage}_job_{state or 'missing'}"
        for stage, state in states.items()
        if state != "succeeded"
    ]
    actor_updates = training.get("actor_updates") if training else None
    if training is None:
        reasons.append("missing_training_metrics")
    elif actor_updates == 0:
        reasons.append("no_actor_updates")
        if training.get("last_critic_warmup_active"):
            reasons.append("actor_not_released_from_critic_warmup_before_stop")
    elif actor_updates is None:
        reasons.append("missing_actor_update_counts")
    if evaluation is None:
        reasons.append("missing_completed_evaluation")
    elif training and evaluation["iteration"] != training["last_iteration"]:
        reasons.append("evaluated_checkpoint_does_not_match_last_training_iteration")
    if actor_updates == 0:
        status = "no_actor_learning"
    elif training is None:
        status = "missing_training_metrics"
    elif any(state != "succeeded" for state in states.values()):
        pending = {None, "queued", "running", "pending", "waiting"}
        status = (
            "incomplete_queue"
            if all(state == "succeeded" or state in pending for state in states.values())
            else "unsuccessful_queue"
        )
    elif evaluation is None:
        status = "missing_completed_evaluation"
    elif reasons:
        status = "inconsistent_training_and_evaluation"
    else:
        status = "evaluated_actor_learning"
    return {
        "status": status,
        "eligible_for_learning_comparison": not reasons,
        "actor_updates_observed": actor_updates,
        "completed_gameplay_evaluation_available": evaluation is not None,
        "job_states": states,
        "reasons": reasons,
        "scope": (
            "A completed gameplay evaluation remains valid even without actor updates; "
            "learning eligibility is not evidence of improvement or training-seed robustness"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--reference-evaluation",
        type=Path,
        help="completed argmax control panel for a separately queued extension arm",
    )
    args = parser.parse_args()
    manifest = json.loads(args.campaign.read_text())
    directory = args.campaign.parent
    report = {
        "campaign": str(args.campaign.resolve()),
        "campaign_sha256": file_sha256(args.campaign),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "jobs": {},
        "arms": {},
    }
    jobs = manifest["jobs"] | manifest.get("supplemental_jobs", {})
    for label, job in jobs.items():
        report["jobs"][label] = json.loads(
            subprocess.check_output(
                ["mlq", "show", str(job["submission"]["id"]), "--json"],
                text=True,
            )
        )
    reference = None
    reference_learning_evidence = None
    if args.reference_evaluation is not None:
        result = json.loads(args.reference_evaluation.read_text())
        if not result.get("complete") or result["decoding"] != "argmax":
            raise ValueError("reference must be a completed argmax evaluation")
        artifact = result["artifacts"]["ppo"]
        reference = {opponent: artifact["panels"][opponent]["games"] for opponent in OPPONENTS}
        report["reference"] = {
            "evaluation": str(args.reference_evaluation.resolve()),
            "actor_sha256": artifact["sha256"],
            "learning_evidence": {
                "status": "unknown_from_evaluation_alone",
                "scope": (
                    "Valid gameplay panel; inspect the control campaign outcome summary "
                    "to establish actor updates and successful training/evaluation jobs"
                ),
            },
        }
    arms = [name.removeprefix("learn-") for name in manifest["jobs"] if name.startswith("learn-")]
    if "control" in arms:
        arms.remove("control")
        arms.insert(0, "control")
    for arm in arms:
        command = manifest["jobs"][f"learn-{arm}"]["command"]
        run = Path(command[command.index("--run-dir") + 1])
        entry = {}
        metrics = run / "metrics.jsonl"
        if metrics.exists():
            rows = [row for row in read_jsonl(metrics) if "iteration_seconds" in row]
            active = [row for row in rows if row.get("actor_updates", 0) > 0]
            if rows:
                entry["training"] = {
                    "last_iteration": rows[-1]["iteration"],
                    "elapsed_hours": rows[-1]["elapsed_hours"],
                    "actor_active_waves": len(active),
                    "actor_updates": (
                        sum(row["actor_updates"] for row in rows)
                        if all("actor_updates" in row for row in rows)
                        else None
                    ),
                    "last_critic_warmup_active": rows[-1].get("critic_warmup_active"),
                    "actor_active_iteration_seconds_median": (
                        statistics.median(row["iteration_seconds"] for row in active)
                        if active
                        else None
                    ),
                    "league_all": league_summary(rows),
                    "league_actor_active": league_summary(active),
                    "league_last_50": league_summary(rows[-50:]),
                }
        benchmark = directory / f"{arm}-benchmark.jsonl"
        if arm == "hardness-league":
            benchmark = directory / "control-benchmark.jsonl"
        if benchmark.exists():
            records = read_jsonl(benchmark)
            entry["benchmark_complete"] = any(
                row.get("event") == "benchmark_complete" and row.get("completed") for row in records
            )
            entry["benchmark"] = [row for row in records if row.get("event") == "batch_summary"]
        evaluation = directory / f"{arm}-evaluation.json"
        if evaluation.exists():
            result = json.loads(evaluation.read_text())
            if not result.get("complete") or result["decoding"] != "argmax":
                raise ValueError(f"invalid primary evaluation: {evaluation}")
            entry["evaluation"] = {
                label: {
                    "iteration": artifact["iteration"],
                    "sha256": artifact["sha256"],
                    "overall_score_rate": artifact["overall_score_rate"],
                    "panels": {key: value["summary"] for key, value in artifact["panels"].items()},
                }
                for label, artifact in result["artifacts"].items()
            }
            panels = {
                opponent: result["artifacts"]["ppo"]["panels"][opponent]["games"]
                for opponent in OPPONENTS
            }
            if arm == "control":
                reference = panels
            elif reference is not None:
                entry["paired_to_control"] = paired_comparison(panels, reference)
            entry["paired_to_bc"] = result["comparisons_to_first"]["ppo"]
        entry["learning_evidence"] = learning_evidence(
            entry,
            training_job=report["jobs"].get(f"learn-{arm}", {}),
            evaluation_job=report["jobs"].get(f"evaluate-{arm}", {}),
        )
        if arm == "control" and "evaluation" in entry:
            reference_learning_evidence = entry["learning_evidence"]
        if "paired_to_control" in entry:
            reference_eligible = (
                reference_learning_evidence["eligible_for_learning_comparison"]
                if reference_learning_evidence is not None
                else None
            )
            candidate_eligible = entry["learning_evidence"]["eligible_for_learning_comparison"]
            entry["paired_to_control_qualification"] = {
                "raw_gameplay_comparison": "valid_matched_panel_comparison",
                "candidate_learning_eligible": candidate_eligible,
                "reference_learning_eligible": reference_eligible,
                "both_arms_learning_eligible": candidate_eligible and reference_eligible is True,
                "scope": (
                    "Raw gameplay differences do not establish comparative learning when "
                    "either arm's learning eligibility is false or unknown"
                ),
            }
        report["arms"][arm] = entry
    for name, filename in (
        ("common_policy_fit", "common-policy-fit.json"),
        ("source_attention", "critic-source-attention.json"),
    ):
        path = directory / filename
        if path.exists():
            report[name] = json.loads(path.read_text())
    output = directory / "outcome-summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)
    for arm, entry in report["arms"].items():
        print(
            json.dumps(
                {
                    "arm": arm,
                    "learning_evidence": entry["learning_evidence"],
                    "comparison_qualification": entry.get("paired_to_control_qualification"),
                    "training": {
                        key: value
                        for key, value in entry.get("training", {}).items()
                        if not key.startswith("league_")
                    },
                    "evaluation": entry.get("evaluation"),
                    "difference": entry.get("paired_to_control", {})
                    .get("panels", {})
                    .get("overall"),
                }
            )
        )


if __name__ == "__main__":
    main()
