#!/usr/bin/env python3
"""Compare architecture runs across both decoding modes and observed learning curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluate_architecture_campaign import OPPONENTS, paired_comparison

from kaggriculture.provenance import file_sha256


def read_training(run: Path) -> dict:
    path = run / "metrics.jsonl"
    milestones = {0: {"actor_updates": 0, "measured_hours": 0.0, "complete_prefix": True}}
    if not path.exists():
        return {
            "actor_updates": 0,
            "panels": [],
            "status": "no_metrics",
            "milestones": milestones,
            "complete_prefix": False,
        }
    journal = path.read_text()
    lines = [line for line in journal.splitlines() if line.strip()]
    rows, partial_tail = [], False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1 or journal.endswith("\n"):
                raise ValueError(f"invalid training journal row: {path}") from None
            partial_tail = True
            break
        if "iteration_seconds" in row:
            rows.append(row)
    updates, panels, measured_seconds, previous = 0, [], 0.0, 0
    complete_prefix = True
    for row in rows:
        iteration, count = row["iteration"], row["actor_updates"]
        if type(iteration) is not int or iteration <= previous:
            raise ValueError(f"duplicate or out-of-order training iteration: {path}")
        if type(count) is not int or count < 0:
            raise ValueError(f"invalid actor update count: {path}")
        complete_prefix = complete_prefix and iteration == previous + 1
        previous = iteration
        updates += count
        for key in ("iteration_seconds", "architecture_panel_seconds"):
            seconds = row.get(key, 0.0)
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"invalid measured training duration: {path}")
            measured_seconds += seconds
        milestone = {
            "actor_updates": updates if complete_prefix else None,
            "measured_hours": measured_seconds / 3600 if complete_prefix else None,
            "complete_prefix": complete_prefix,
        }
        milestones[iteration] = milestone
        if "architecture_panel" in row:
            panels.append({**row["architecture_panel"], "iteration": iteration, **milestone})
    if not rows:
        return {
            "actor_updates": 0,
            "panels": [],
            "status": "no_completed_wave",
            "milestones": milestones,
            "complete_prefix": False,
            "partial_tail": partial_tail,
        }
    last = rows[-1]
    guard = last.get("architecture_panel_state", {})
    return {
        "status": "culled" if guard.get("culled") else "observed_training",
        "iterations": last["iteration"],
        **milestones[last["iteration"]],
        "observed_actor_updates": updates,
        "observed_measured_hours": measured_seconds / 3600,
        "actor_active_waves": sum(row["actor_updates"] > 0 for row in rows)
        if complete_prefix
        else None,
        "last_session_elapsed_hours": last.get("elapsed_hours"),
        "best_panel_checkpoint": guard.get("best_checkpoint"),
        "panels": panels,
        "milestones": milestones,
        "partial_tail": partial_tail,
    }


def read_evaluation(path: Path, mode: str) -> dict:
    result = json.loads(path.read_text())
    if result.get("complete") is not True or result.get("decoding") != mode:
        raise ValueError(f"incomplete or mismatched decoding evaluation: {path}")
    if set(result["artifacts"]) != {"bc", "ppo"}:
        raise ValueError(f"expected BC and PPO evaluations: {path}")
    count, seed_start = result["games_per_opponent"], result["seed_start"]
    if type(count) is not int or count < 2 or count % 2 or type(seed_start) is not int:
        raise ValueError(f"invalid declared development panel: {path}")
    expected_pairs = [(seed, seed % 2) for seed in range(seed_start, seed_start + count)]
    for artifact in result["artifacts"].values():
        if set(artifact["panels"]) != set(OPPONENTS):
            raise ValueError(f"incomplete opponent panel: {path}")
        for panel in artifact["panels"].values():
            games = panel["games"]
            pairs = [(game["seed"], game["seat"]) for game in games]
            if not pairs or len(set(pairs)) != len(pairs):
                raise ValueError(f"empty or duplicated evaluation seed pairs: {path}")
            if pairs != expected_pairs:
                raise ValueError(
                    f"BC/PPO opponent panels use different seed pairs than declared: {path}"
                )
            scores = [game["score"] for game in games]
            if any(type(score) not in (int, float) or score not in (0, 0.5, 1) for score in scores):
                raise ValueError(f"invalid evaluation game score: {path}")
            if not math.isclose(
                panel["summary"]["score_rate"], sum(scores) / len(scores), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"evaluation score summary disagrees with games: {path}")
    return result


def bind_evaluations(entry: dict, command: list[str], *, arm: str) -> None:
    """Bind every decoding panel to immutable bytes and the corresponding budget."""
    evaluations = list(entry["evaluations"].values())
    if not evaluations:
        return
    for field in (
        "seed_start",
        "games_per_opponent",
        "sampling_seed",
        "precision",
        "seat_protocol",
    ):
        if any(result[field] != evaluations[0][field] for result in evaluations[1:]):
            raise ValueError(f"decoding panels disagree on evaluation {field}: {arm}")
    ppo_artifacts = [result["artifacts"]["ppo"] for result in evaluations]
    bc_artifacts = [result["artifacts"]["bc"] for result in evaluations]
    for label, artifacts in (("PPO", ppo_artifacts), ("BC", bc_artifacts)):
        for field in ("sha256", "iteration", "architecture", "model_config", "source_identity"):
            if any(artifact[field] != artifacts[0][field] for artifact in artifacts[1:]):
                raise ValueError(f"decoding panels disagree on {label} {field}: {arm}")
    iteration = ppo_artifacts[0]["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError(f"invalid evaluated iteration: {arm}")
    immutable = Path(entry["run"]) / f"checkpoint-{iteration:06d}.pt"
    if not immutable.is_file() or file_sha256(immutable) != ppo_artifacts[0]["sha256"]:
        raise ValueError(f"evaluated policy does not match declared run checkpoint: {arm}")
    if command.count("--init-actor-from") != 1:
        raise ValueError(f"expected one declared BC initializer: {arm}")
    initial = Path(command[command.index("--init-actor-from") + 1])
    if not initial.is_file() or file_sha256(initial) != bc_artifacts[0]["sha256"]:
        raise ValueError(f"evaluated BC policy does not match training initializer: {arm}")
    milestone = entry["training"]["milestones"].get(iteration)
    if milestone is None:
        raise ValueError(f"evaluated checkpoint has no matching training boundary: {arm}")
    entry["evaluated_checkpoint"] = str(immutable.resolve())
    entry["evaluated_budget"] = {"iteration": iteration, **milestone}


def comparison(report: dict, *, reference: str) -> dict:
    if reference not in report["arms"]:
        raise ValueError(f"reference arm absent: {reference}")
    reference_entry = report["arms"][reference]
    report["reference"] = reference
    for name, entry in report["arms"].items():
        entry["paired_to_reference"] = {}
        for mode, result in entry["evaluations"].items():
            baseline = reference_entry["evaluations"].get(mode)
            if baseline is None or name == reference:
                continue
            entry["paired_to_reference"][mode] = paired_comparison(
                {op: result["artifacts"]["ppo"]["panels"][op]["games"] for op in OPPONENTS},
                {op: baseline["artifacts"]["ppo"]["panels"][op]["games"] for op in OPPONENTS},
            )
    return report


def markdown(report: dict) -> str:
    rows = [
        "These compare runs on common development maps. Scores are win/draw percentages. "
        "A checkpoint score is valid even after a cull; unequal updates and time remain explicit. "
        "Measured hours sum completed waves and fixed panels, "
        "excluding startup/checkpoint overhead. "
        "Missing journal prefixes make cumulative budgets unknown. "
        "One training seed does not establish training-seed robustness.",
        "",
        "| Run | Updates | Measured hours | v27 sampled | v27 argmax | "
        "Sampled gain from BC | Argmax gain from BC | Evaluated iteration | Run status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, entry in report["arms"].items():
        training = entry.get("evaluated_budget", entry["training"])
        scores, gains = {}, {}
        for mode in ("sampled", "argmax"):
            evaluation = entry["evaluations"].get(mode)
            if evaluation is None:
                scores[mode] = gains[mode] = "pending"
                continue
            values = {
                key: artifact["panels"]["scripted-v27"]["summary"]["score_rate"] * 100
                for key, artifact in evaluation["artifacts"].items()
            }
            scores[mode], gains[mode] = (
                f"{values['ppo']:.2f}",
                f"{values['ppo'] - values['bc']:+.2f}",
            )
        hours = training.get("measured_hours")
        hours_text = "—" if hours is None else f"{hours:.2f}"
        updates = training.get("actor_updates")
        updates_text = "unknown" if updates is None else str(updates)
        iteration = entry.get("evaluated_budget", {}).get("iteration", "pending")
        status = entry["training"]["status"]
        if not entry["training"].get("complete_prefix", True):
            status += "; partial journal"
        if entry["training"].get("partial_tail"):
            status += "; incomplete final row"
        if entry.get("requested_iterations", 0) > entry["training"].get("iterations", 0):
            status += "; before target"
        rows.append(
            f"| {name} | {updates_text} | {hours_text} | "
            f"{scores['sampled']} | {scores['argmax']} | {gains['sampled']} | {gains['argmax']} | "
            f"{iteration} | {status} |"
        )
    rows += [
        "",
        "Full opponent panels, paired uncertainty, checkpoint identities and learning curves "
        "are retained in the JSON report. No winner is inferred from missing evaluations.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="+")
    parser.add_argument("--reference", default="component-control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "campaigns": {},
        "arms": {},
        "scope": "Observed cross-run gameplay; no interpolated or synthetic matched-update scores",
    }
    for path in args.campaign:
        manifest = json.loads(path.read_text())
        report["campaigns"][str(path.resolve())] = file_sha256(path)
        for label, job in manifest["jobs"].items():
            if not label.startswith("learn-"):
                continue
            arm = label.removeprefix("learn-")
            if arm in report["arms"]:
                raise ValueError(f"duplicate cross-campaign arm: {arm}")
            command = job["command"]
            run = Path(command[command.index("--run-dir") + 1])
            entry = {
                "run": str(run),
                "training_job": job["submission"]["id"],
                "requested_iterations": int(command[command.index("--iterations") + 1]),
                "source_identity": manifest["source_identity"],
                "training": read_training(run),
                "evaluations": {},
            }
            for mode in ("sampled", "argmax"):
                evaluation = path.parent / f"{arm}-{mode}-evaluation.json"
                if evaluation.exists():
                    entry["evaluations"][mode] = read_evaluation(evaluation, mode)
            bind_evaluations(entry, command, arm=arm)
            report["arms"][arm] = entry
    comparison(report, reference=args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = markdown(report)
    args.output.with_suffix(".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
