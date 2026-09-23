#!/usr/bin/env python3
"""Compare frozen actors on a matched native development panel using CUDA BF16."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.critic_diagnostics import terminal_outcomes
from kaggriculture.evaluation import artifact_seed_usage, bounded_mean_interval, seed_protocol
from kaggriculture.provenance import file_sha256, source_identity

OPPONENTS = ("starter", "scripted-v27")


def artifact_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    return label, Path(path)


def panel_rows(rollout: Any, *, seed_start: int, games: int) -> list[dict[str, Any]]:
    """Reject partial or mis-seated panels before computing any scores."""
    seeds = np.arange(seed_start, seed_start + games)
    if (
        rollout.state_count != games * 719
        or np.shape(rollout.valid) != (games, 719)
        or not np.all(rollout.valid)
        or not np.array_equal(rollout.episode_seeds, seeds)
        or not np.array_equal(rollout.seats, seeds % 2)
        or np.shape(rollout.final_money) != (games,)
        or np.shape(rollout.opponent_money) != (games,)
        or not np.isfinite(rollout.final_money).all()
        or not np.isfinite(rollout.opponent_money).all()
    ):
        raise ValueError("incomplete, nonfinite, or mismatched development panel")
    outcomes = terminal_outcomes(rollout)
    return [
        {
            "seed": int(seed),
            "seat": int(seat),
            "money": float(money),
            "opponent_money": float(opposition),
            "terminal_outcome": float(outcome),
            "score": (float(outcome) + 1.0) / 2.0,
        }
        for seed, seat, money, opposition, outcome in zip(
            seeds, rollout.seats, rollout.final_money, rollout.opponent_money, outcomes, strict=True
        )
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    money = [row["money"] for row in rows]
    return {
        "games": len(rows),
        "score_rate": float(np.mean(scores)),
        "score_rate_95ci": list(bounded_mean_interval(scores)),
        "score_interval_method": "hoeffding_independent_seed_clusters",
        "money_mean": float(np.mean(money)),
        "money_median": float(np.median(money)),
        "money_p10": float(np.quantile(money, 0.1)),
        "margin_mean": float(np.mean([row["money"] - row["opponent_money"] for row in rows])),
    }


def paired_comparison(
    candidate: dict[str, list[dict[str, Any]]],
    reference: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Bootstrap whole matched seeds, keeping both opponent outcomes together."""
    if set(candidate) != set(OPPONENTS) or set(reference) != set(OPPONENTS):
        raise ValueError("paired comparison requires the complete opponent panel")
    expected = [(row["seed"], row["seat"]) for row in reference[OPPONENTS[0]]]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("paired comparison requires distinct nonempty seed clusters")
    for panels in (candidate, reference):
        for rows in panels.values():
            if [(row["seed"], row["seat"]) for row in rows] != expected:
                raise ValueError("paired comparison requires identical seeds and seats")
    indices = np.random.default_rng(20260918).integers(
        0, len(expected), size=(10_000, len(expected))
    )
    results: dict[str, Any] = {
        "seed_clusters": len(expected),
        "method": "paired_seed_cluster_percentile_bootstrap",
        "resamples": 10_000,
        "bootstrap_seed": 20260918,
        "confidence": 0.95,
        "scope": "Exploratory game-seed uncertainty; no multiplicity or training-seed correction",
        "panels": {},
    }
    for panel in (*OPPONENTS, "overall"):
        opponents = OPPONENTS if panel == "overall" else (panel,)
        metrics = {}
        for field in ("score", "money"):
            difference = np.mean(
                [
                    np.asarray([row[field] for row in candidate[opponent]])
                    - np.asarray([row[field] for row in reference[opponent]])
                    for opponent in opponents
                ],
                axis=0,
            )
            metrics[field] = {
                "difference": float(difference.mean()),
                "ci95": np.quantile(difference[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            }
        results["panels"][panel] = metrics
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=artifact_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=4_501_000)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--decoding", choices=("argmax", "sampled"), default="argmax")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even count for balanced seats")
    labels = [label for label, _ in args.artifact]
    if len(set(labels)) != len(labels):
        parser.error("artifact labels must be distinct")
    seed_protocol("development", args.seed_start, args.games, usage=[])
    return args


def main() -> None:
    args = parse_args()
    # Imports stay local so panel validation/statistics tests never initialize models.
    import torch

    from kaggriculture.inference import load_actor_artifact
    from kaggriculture.rollout import collect_mixed_play_rust

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("evaluation requires queued CUDA with native BF16 support")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "format_version": 1,
        "source_identity": source_identity(),
        "decoding": args.decoding,
        "precision": "CUDA BF16, Inductor graph",
        "seed_start": args.seed_start,
        "games_per_opponent": args.games,
        "sampling_seed": 20260917,
        "opponents": list(OPPONENTS),
        "seat_protocol": "seed modulo 2; balanced across maps, not both seats per map",
        "primary_metric": "Mean win/draw score, equally weighted over the two opponents",
        "scope": "Native development diagnostic; one training seed, not official finalist evidence",
        "artifacts": {},
        "comparisons_to_first": {},
    }
    reference = None
    for label, path in args.artifact:
        digest = file_sha256(path)
        actor, metadata = load_actor_artifact(path, device="cuda")
        actor.eval().requires_grad_(False)
        protocol = seed_protocol(
            "development", args.seed_start, args.games, usage=artifact_seed_usage(metadata)
        )
        entry: dict[str, Any] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "iteration": metadata.get("iteration", 0),
            "source_identity": metadata["source_identity"],
            "architecture": metadata.get("architecture"),
            "model_config": metadata["model_config"],
            "seed_protocol": protocol,
            "panels": {},
        }
        panels = {}
        for opponent in OPPONENTS:
            rollout = collect_mixed_play_rust(
                actor,
                self_play_games=0,
                league_games=args.games,
                builtin_lanes=(opponent,),
                seed_start=args.seed_start,
                deterministic=args.decoding == "argmax",
                temperature=1.0,
                episode_steps=720,
                reward_mode="terminal-outcome",
                sampling_seed=20260917,
                forward_mode="inductor_graph",
                forward_autocast=True,
            )
            if rollout.learner_stochastic != (args.decoding == "sampled"):
                raise RuntimeError("collector did not honor the requested decoding mode")
            rows = panel_rows(rollout, seed_start=args.seed_start, games=args.games)
            summary = summarize(rows)
            entry["panels"][opponent] = {
                "summary": summary,
                "games": rows,
                "rollout_seconds": rollout.elapsed_seconds,
            }
            panels[opponent] = rows
            print(json.dumps({"artifact": label, "opponent": opponent, **summary}), flush=True)
            del rollout
        if file_sha256(path) != digest:
            raise RuntimeError(f"artifact changed during evaluation: {path}")
        entry["overall_score_rate"] = float(
            np.mean([entry["panels"][opponent]["summary"]["score_rate"] for opponent in OPPONENTS])
        )
        report["artifacts"][label] = entry
        if reference is None:
            reference = panels
            report["reference"] = label
        else:
            report["comparisons_to_first"][label] = paired_comparison(panels, reference)
        del actor, metadata
    report["elapsed_seconds"] = time.perf_counter() - started
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
