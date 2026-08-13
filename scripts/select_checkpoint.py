#!/usr/bin/env python3
"""Evaluate a checkpoint archive on a fixed panel and promote the best model."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluate_checkpoint import _opponent_identity, evaluate, render_json

from kaggriculture.provenance import file_sha256


def _mean_confidence_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def summarize_panel(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a fixed opponent panel at the paired-seed cluster level."""
    if not evaluations:
        raise ValueError("checkpoint panel cannot be empty")
    invalid = [row["opponent_label"] for row in evaluations if not row["valid_for_selection"]]
    if invalid:
        raise ValueError(f"checkpoint has invalid panel evaluations: {invalid}")

    seed_maps = []
    opponent_summaries = []
    for evaluation in evaluations:
        clusters = evaluation["summary"]["seed_cluster_statistics"]
        seed_map = {
            int(row["seed"]): (float(row["score_rate"]), float(row["mean_margin"]))
            for row in clusters
        }
        seed_maps.append(seed_map)
        opponent_summaries.append(
            {
                "opponent": evaluation["opponent"],
                "opponent_label": evaluation["opponent_label"],
                "score_rate": float(evaluation["summary"]["score_rate"]),
                "score_rate_95ci": evaluation["summary"]["score_rate_95ci"],
                "mean_margin": float(evaluation["summary"]["mean_margin"]),
                "margin_95ci": evaluation["summary"]["margin_95ci"],
            }
        )
    seeds = set(seed_maps[0])
    if any(set(seed_map) != seeds for seed_map in seed_maps[1:]):
        raise ValueError("opponent evaluations do not use identical completed seed clusters")
    if not seeds:
        raise ValueError("opponent panel contains no completed seed clusters")

    panel_scores = [
        statistics.fmean(seed_map[seed][0] for seed_map in seed_maps) for seed in sorted(seeds)
    ]
    panel_margins = [
        statistics.fmean(seed_map[seed][1] for seed_map in seed_maps) for seed in sorted(seeds)
    ]
    score_ci = _mean_confidence_interval(panel_scores)
    margin_ci = _mean_confidence_interval(panel_margins)
    return {
        "valid_for_selection": True,
        "opponents": len(evaluations),
        "paired_seed_clusters": len(seeds),
        "panel_score_rate": statistics.fmean(panel_scores),
        "panel_score_rate_95ci": [max(0.0, score_ci[0]), min(1.0, score_ci[1])],
        "panel_mean_margin": statistics.fmean(panel_margins),
        "panel_margin_95ci": list(margin_ci),
        "worst_opponent_score_rate": min(row["score_rate"] for row in opponent_summaries),
        "opponent_summaries": opponent_summaries,
    }


def _ranking_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    panel = candidate["panel"]
    return (
        float(panel["panel_score_rate_95ci"][0]),
        float(panel["worst_opponent_score_rate"]),
        float(panel["panel_margin_95ci"][0]),
        int(candidate["iteration"]),
    )


def _write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_atomic(source: Path, destination: Path, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        actual_digest = file_sha256(temporary)
        if actual_digest != expected_sha256:
            raise ValueError("selected checkpoint changed after evaluation")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if file_sha256(destination) != expected_sha256:
            raise OSError("selected checkpoint digest changed during atomic promotion")
        return actual_digest
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="checkpoint-*.pt")
    parser.add_argument(
        "--opponent",
        action="append",
        dest="opponents",
        help="repeat for a fixed panel; defaults to public v27",
    )
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1 or args.workers < 1 or args.torch_threads < 1:
        raise ValueError("seeds, workers, and torch threads must be positive")
    run_directory = args.run_dir.expanduser().resolve()
    if not run_directory.is_dir():
        raise NotADirectoryError(run_directory)
    checkpoints = sorted(
        path.resolve() for path in run_directory.glob(args.pattern) if path.is_file()
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints match {args.pattern!r} under {run_directory}")
    opponents = args.opponents or ["v27"]
    candidates = []
    started = time.perf_counter()
    for checkpoint in checkpoints:
        evaluations = [
            evaluate(
                SimpleNamespace(
                    artifact=checkpoint,
                    opponent=opponent,
                    seeds=args.seeds,
                    seed_start=args.seed_start,
                    workers=args.workers,
                    torch_threads=args.torch_threads,
                    device=args.device,
                )
            )
            for opponent in opponents
        ]
        provenance = evaluations[0]["artifact_provenance"]
        if any(evaluation["artifact_provenance"] != provenance for evaluation in evaluations[1:]):
            raise ValueError("checkpoint changed between fixed-panel opponent evaluations")
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "iteration": int(provenance["iteration"]),
                "artifact_provenance": provenance,
                "panel": summarize_panel(evaluations),
                "evaluations": evaluations,
            }
        )

    ranked = sorted(candidates, key=_ranking_key, reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    best = ranked[0]
    source_identity = best["artifact_provenance"]["source_identity"]
    if any(
        candidate["artifact_provenance"]["source_identity"] != source_identity
        for candidate in ranked[1:]
    ):
        raise ValueError("candidate checkpoints were produced by different source identities")
    run_provenance = best["artifact_provenance"]["run_provenance"]
    if any(
        candidate["artifact_provenance"].get("run_provenance") != run_provenance
        for candidate in ranked[1:]
    ):
        raise ValueError("candidate checkpoints were produced by different calibrated runs")
    opponent_provenance = {
        evaluation["opponent_label"]: evaluation["opponent_provenance"]
        for evaluation in best["evaluations"]
    }
    expected_opponent_identities = {
        label: _opponent_identity(provenance) for label, provenance in opponent_provenance.items()
    }
    for candidate in ranked:
        identities = {
            evaluation["opponent_label"]: _opponent_identity(evaluation["opponent_provenance"])
            for evaluation in candidate["evaluations"]
        }
        if identities != expected_opponent_identities:
            raise ValueError("fixed-panel opponent bytes changed during checkpoint selection")
    best_output = None
    best_output_sha256 = None
    if args.best_output:
        best_output = args.best_output.expanduser().resolve()
        best_output_sha256 = _copy_atomic(
            Path(best["checkpoint"]),
            best_output,
            best["artifact_provenance"]["sha256"],
        )
    payload = {
        "valid_for_selection": True,
        "run_dir": str(run_directory),
        "checkpoint_pattern": args.pattern,
        "opponents": opponents,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "ranking": "panel score lower 95% bound, worst-opponent score, margin lower bound",
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint": best["checkpoint"],
        "best_iteration": best["iteration"],
        "best_checkpoint_sha256": best["artifact_provenance"]["sha256"],
        "best_output": None if best_output is None else str(best_output),
        "best_output_sha256": best_output_sha256,
        "source_identity": source_identity,
        "run_provenance": run_provenance,
        "opponent_provenance": opponent_provenance,
        "candidates": ranked,
    }
    rendered = render_json(payload)
    print(rendered)
    _write_atomic(args.output, rendered)


if __name__ == "__main__":
    main()
