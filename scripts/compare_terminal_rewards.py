#!/usr/bin/env python3
"""Compare terminal utility shapes on saved games and explicit synthetic lotteries.

This runs no policy or environment. Re-scoring fixed outcomes cannot predict the
policies learned under another reward; lotteries expose preference differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

STARTING_BANK = 3000.0


def rewards(own: float, other: float) -> dict[str, float]:
    if not all(math.isfinite(x) and x >= 0 for x in (own, other)):
        raise ValueError("terminal banks must be finite and nonnegative")
    margin = (own - other) / (own + other + 2 * STARTING_BANK)
    log_ratio = math.log(own + STARTING_BANK) - math.log(other + STARTING_BANK)
    outcome = float((own > other) - (own < other))
    return {
        "win_loss": outcome,
        "terminal_bank": margin,
        "paused_run_soft": math.tanh(margin / 0.01),
        "log_ratio_starting_bank": log_ratio,
        "log_ratio_one": math.log1p(own) - math.log1p(other),
        "bounded_log_ratio": math.tanh(log_ratio),
        "signed_log1p_log_ratio": math.copysign(math.log1p(abs(log_ratio)), log_ratio),
        "signed_sqrt_log_ratio": math.copysign(math.sqrt(abs(log_ratio)), log_ratio),
        "near_sign_log_ratio": math.tanh(log_ratio / 0.02),
        "win_loss_plus_001_bounded_margin": outcome + 0.01 * math.tanh(log_ratio / 0.02),
    }


def summarize(pairs: list[tuple[float, float]]) -> dict:
    values = [rewards(*pair) for pair in pairs]
    return {
        "games": len(pairs),
        "score_rate": statistics.fmean((r["win_loss"] + 1) / 2 for r in values),
        "mean_money": statistics.fmean(x for x, _ in pairs),
        "rewards": {
            name: {
                "mean": statistics.fmean(row[name] for row in values),
                "standard_deviation": statistics.pstdev(row[name] for row in values),
                "outside_existing_value_support_fraction": statistics.fmean(
                    abs(row[name]) > 2.2 for row in values
                ),
            }
            for name in values[0]
        },
        "paused_run_saturated_fraction_abs_gt_099": statistics.fmean(
            abs(row["paused_run_soft"]) > 0.99 for row in values
        ),
    }


def lotteries() -> dict:
    # Exact probabilities, not fitted to measured trajectories. Both policies
    # below sometimes win/lose; this isolates expected utility from win rate.
    examples = {
        "frequent_narrow_wins": [(110000.0, 100000.0)] * 90 + [(90000.0, 100000.0)] * 10,
        "less_frequent_large_wins": [(200000.0, 100000.0)] * 60 + [(90000.0, 100000.0)] * 40,
        "certain_narrow_win": [(101000.0, 100000.0)] * 100,
        "certain_one_unit_win": [(100001.0, 100000.0)] * 100,
        "rare_loss_large_wins": [(200000.0, 100000.0)] * 99 + [(0.0, 100000.0)],
    }
    rng = random.Random(20260927)
    shocks = [rng.gauss(0, 1) for _ in range(10000)]
    # Positive banks with controlled expected log ratio and volatility. Reuse
    # each shock across alternatives for a paired sensitivity simulation.
    for mean in (0.01, 0.05, 0.15):
        for sigma in (0.02, 0.1, 0.3):
            examples[f"lognormal_mean{mean}_sigma{sigma}"] = [
                (100000 * math.exp(mean + sigma * z), 100000.0) for z in shocks
            ]
    return {name: summarize(pairs) for name, pairs in examples.items()}


def score_preference_audit(arms: dict[str, dict]) -> dict:
    """Audit fixed-outcome preferences against match score, not margin utility.

    These are descriptive rankings, not estimates of reward-dependent learning
    or a fitted Elo rating. Tied utility maximizers retain their score range.
    """
    best_score = max(arm["score_rate"] for arm in arms.values())
    result = {}
    for reward in next(iter(arms.values()))["rewards"]:
        best_utility = max(arm["rewards"][reward]["mean"] for arm in arms.values())
        selected = [
            name for name, arm in arms.items() if arm["rewards"][reward]["mean"] == best_utility
        ]
        scores = [arms[name]["score_rate"] for name in selected]
        result[reward] = {
            "utility_maximizers": selected,
            "match_score_range": [min(scores), max(scores)],
            "worst_score_regret_against_available_arms": best_score - min(scores),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = []
    for path in args.report:
        report = json.loads(path.read_text())
        if report.get("complete") is not True:
            raise ValueError(f"incomplete evaluation: {path}")
        arms = {}
        for name, artifact in report["artifacts"].items():
            arms[name] = {
                opponent: summarize([(r["money"], r["opponent_money"]) for r in p["games"]])
                for opponent, p in artifact["panels"].items()
            }
        comparisons.append(
            {
                "report": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "decoding": report["decoding"],
                "arms": arms,
                "match_score_preference_audit_by_opponent": {
                    opponent: score_preference_audit(
                        {
                            name: panels[opponent]
                            for name, panels in arms.items()
                            if opponent in panels
                        }
                    )
                    for opponent in sorted({op for panels in arms.values() for op in panels})
                },
            }
        )
    synthetic = lotteries()
    result = {
        "selection_objective": "Expected match score: P(win) + 0.5 * P(draw)",
        "scope": "No training or model execution; fixed-outcome utilities and synthetic lotteries",
        "interpretation": (
            "log(% margin) interpreted as log(1 + fractional lead), i.e. log bank ratio"
        ),
        "regularization": (
            "starting-bank pseudocount isolates reward shape from zero-bank behavior; "
            "+1 shown separately"
        ),
        "reference_rewards": {
            "main_default": "terminal-outcome: win_loss",
            "paused_promising_run": "terminal-soft-outcome: paused_run_soft",
        },
        "identity": "paused_run_soft = tanh(tanh(log((own+3000)/(other+3000))/2)/0.01)",
        "limits": [
            "Re-scoring cannot estimate the return of a policy trained on another reward",
            "Synthetic outcome distributions are assumptions, not an environment model",
            "Monotone utilities preserve individual-game order, not expected-policy order",
            "Unbounded log rewards require revisiting the current critic support and loss scales",
        ],
        "relative_lead_examples_other_bank_100000": {
            str(lead): rewards(100000 * (1 + lead), 100000)
            for lead in (-0.5, -0.1, -0.01, 0, 0.01, 0.05, 0.1, 1)
        },
        "saved_game_comparisons": comparisons,
        "synthetic_lotteries": synthetic,
        "synthetic_match_score_preference_audit": score_preference_audit(synthetic),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
