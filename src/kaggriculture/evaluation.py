"""Disjoint map domains and finite-sample, paired-seed evaluation evidence.

Intervals are half-open. BC holdout maps count as development exposure too;
only the actual corpus lists identify maps used by a particular actor. Reserved
ranges additionally prevent an evaluation from leaking into future training.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

SEED_PROTOCOL_VERSION = 1
DEVELOPMENT_SEED_START = 4_000_000
SCREENING_SEED_START = 10_000_000
FINALIST_SEED_START = 12_000_000
ONLINE_RL_SEED_START = 20_260_812
SEED_DOMAINS = {
    "bc": (0, 4_000_000),
    "development": (DEVELOPMENT_SEED_START, 8_000_000),
    "screening": (SCREENING_SEED_START, 11_000_000),
    "finalist": (FINALIST_SEED_START, 13_000_000),
    "online_rl": (20_000_000, 2**32),
}
SCORE_CONFIDENCE = {
    "method": "hoeffding",
    "confidence": 0.95,
    "sampling_unit": "independent_seed_cluster",
    "bounds": [0.0, 1.0],
}


def bounded_mean_interval(
    values: Iterable[float], *, lower: float = 0.0, upper: float = 1.0, alpha: float = 0.05
) -> tuple[float, float]:
    """Distribution-free two-sided Hoeffding bound; seats are not replicates."""
    values = list(values)
    if not values or not 0 < alpha < 1 or not lower < upper:
        raise ValueError("confidence interval needs samples, ordered bounds and 0 < alpha < 1")
    if any(not math.isfinite(value) or not lower <= value <= upper for value in values):
        raise ValueError("confidence interval samples must be finite and within bounds")
    mean = statistics.fmean(values)
    radius = (upper - lower) * math.sqrt(math.log(2 / alpha) / (2 * len(values)))
    return max(lower, mean - radius), min(upper, mean + radius)


def validate_seed_interval(
    domain: str, start: int, count: int, *, usage: Iterable[dict[str, Any]] = ()
) -> dict[str, Any]:
    if domain not in SEED_DOMAINS:
        raise ValueError(f"unknown seed domain: {domain!r}")
    if type(start) is not int or type(count) is not int or count < 1:
        raise ValueError("seed interval requires integer start and positive count")
    low, high = SEED_DOMAINS[domain]
    if not low <= start < start + count <= high:
        raise ValueError(f"{domain} seed interval is outside reserved domain [{low}, {high})")
    for row in usage:
        if not isinstance(row, dict):
            raise ValueError("recorded seed usage must be a mapping")
        other, other_start, other_count = row.get("domain"), row.get("start"), row.get("count")
        if (
            other not in SEED_DOMAINS
            or type(other_start) is not int
            or type(other_count) is not int
            or other_count < 1
        ):
            raise ValueError("recorded seed usage has an invalid schema")
        if (other != domain or domain == "finalist") and max(start, other_start) < min(
            start + count, other_start + other_count
        ):
            raise ValueError(f"{domain} seeds overlap recorded {other} usage")
        other_low, other_high = SEED_DOMAINS[other]
        if not other_low <= other_start < other_start + other_count <= other_high:
            raise ValueError(f"recorded {other} usage is outside its reserved domain")
    return {"domain": domain, "start": start, "count": count}


def artifact_seed_usage(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read exact BC exposure and recorded RL/development reservations; no legacy inference."""
    usage = payload.get("seed_usage")
    if not isinstance(usage, list):
        raise ValueError("artifact lacks seed_usage provenance; fresh training is required")
    result = [dict(row) for row in usage]
    bc = payload.get("bc_provenance")
    if bc is not None:
        datasets = bc.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("BC provenance lacks actual corpus seed sets")
        for dataset in datasets:
            for field in ("train_seeds", "holdout_seeds"):
                seeds = dataset.get(field)
                if (
                    not isinstance(seeds, list)
                    or not seeds
                    or any(type(seed) is not int for seed in seeds)
                ):
                    raise ValueError("BC provenance lacks actual train/holdout seed sets")
                result.extend(
                    {"domain": "bc", "start": seed, "count": 1} for seed in sorted(set(seeds))
                )
    initializer = payload.get("initial_actor")
    if initializer is not None:
        if not isinstance(initializer, dict):
            raise ValueError("initial actor provenance must be a mapping")
        members = initializer.get("agents", [initializer])
        if not isinstance(members, list) or not members:
            raise ValueError("initial actor provenance has no members")
        for member in members:
            result.extend(artifact_seed_usage(member))
    # Initializer seed_usage already includes its BC seeds; deduplicate without
    # changing order so repeated export remains a stable evidence binding.
    result = list({(row["domain"], row["start"], row["count"]): row for row in result}.values())
    return result


def seed_protocol(
    domain: str, start: int, count: int, *, usage: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    usage = list(usage)
    interval = validate_seed_interval(domain, start, count, usage=usage)
    return {
        "format_version": SEED_PROTOCOL_VERSION,
        "domains": {name: list(bounds) for name, bounds in SEED_DOMAINS.items()},
        "evaluation": interval,
        "prior_usage": usage,
    }


def validate_seed_protocol(value: object, *, domain: str, start: int, count: int) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("format_version") != SEED_PROTOCOL_VERSION:
        raise ValueError("missing or stale evaluation seed protocol")
    expected = seed_protocol(domain, start, count, usage=value.get("prior_usage", ()))
    if value != expected:
        raise ValueError("evaluation seed protocol does not bind the reported interval")
    return expected


def validate_score_evidence(payload: dict[str, Any]) -> None:
    """Recompute the bound from a complete seed panel instead of trusting a flag."""
    summary = payload.get("summary", {})
    rows = summary.get("seed_cluster_statistics")
    start, count = payload.get("seed_start"), payload.get("seed_count")
    if type(start) is not int or type(count) is not int or count < 1 or not isinstance(rows, list):
        raise ValueError("evaluation lacks a complete seed-cluster panel")
    if len(rows) != count or {row.get("seed") for row in rows} != set(range(start, start + count)):
        raise ValueError("evaluation seed-cluster panel is incomplete or duplicated")
    scores = [row["score_rate"] for row in rows]
    interval = list(bounded_mean_interval(scores))
    if (
        summary.get("score_confidence") != SCORE_CONFIDENCE
        or summary.get("score_rate_95ci") != interval
        or summary.get("score_rate") != statistics.fmean(scores)
        or any(not math.isfinite(row["mean_margin"]) for row in rows)
    ):
        raise ValueError("evaluation has invalid finite-sample score evidence")


def paired_score_comparison(first: dict[int, float], second: dict[int, float]) -> dict[str, Any]:
    if not first or first.keys() != second.keys():
        raise ValueError("paired comparison requires identical nonempty seed clusters")
    differences = [first[seed] - second[seed] for seed in sorted(first)]
    return {
        "seed_clusters": len(differences),
        "mean_score_difference": statistics.fmean(differences),
        "score_difference_95ci": list(bounded_mean_interval(differences, lower=-1.0, upper=1.0)),
        "method": "hoeffding",
        "sampling_unit": "independent_paired_seed_cluster",
    }


def validate_finalist_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection_provenance")
    if not isinstance(selection, dict):
        raise ValueError("finalist requires checkpoint-selection provenance")
    design = selection.get("statistical_selection")
    if (
        not isinstance(design, dict)
        or design.get("protocol") != "archive_screening_then_untouched_finalist"
        or type(design.get("candidate_count")) is not int
        or design["candidate_count"] < 1
        or design.get("finalist_required") is not True
        or design.get("candidate_frozen_before_finalist") is not True
    ):
        raise ValueError("finalist lacks statistical selection provenance")
    screening = validate_seed_protocol(
        selection.get("seed_protocol"),
        domain="screening",
        start=selection.get("screening_seed_start"),
        count=selection.get("screening_seed_count"),
    )
    finalist = validate_seed_protocol(
        payload.get("seed_protocol"),
        domain="finalist",
        start=payload.get("seed_start"),
        count=payload.get("seed_count"),
    )
    exposure = payload.get("artifact_provenance", {}).get("seed_usage")
    if not isinstance(exposure, list):
        raise ValueError("finalist artifact lacks seed usage provenance")
    required = exposure + screening["prior_usage"] + [screening["evaluation"]]
    if finalist["prior_usage"] != required:
        raise ValueError("finalist seed protocol omits recorded training or screening exposure")
    validate_score_evidence(payload)
    return finalist
