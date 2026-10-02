#!/usr/bin/env python3
"""Chart a PPO checkpoint's league score against every archived predecessor.

The evidence is the league's own matchup record (`league_matchup_evidence`): the
learner's decayed win/draw/loss tally against each snapshot, as the hardness
selector saw it. Snapshots are pooled by era because a single snapshot's
decayed record is often only five to eight games.

    uv run --extra train --with matplotlib --with scipy \\
        python scripts/plot_league_evidence.py --checkpoint runs/ppo/latest.pt \\
        --output results/league-3379
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import beta as beta_dist

THEMES = {
    "light": {
        "surface": "#ffffff",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": "#2a78d6",
        "snapshot": "#9ec5f4",
        "band": "#f3f2ef",
    },
    "dark": {
        "surface": "#0d1117",
        "text": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#262c36",
        "series": "#3987e5",
        "snapshot": "#1c5cab",
        "band": "#161b22",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path prefix; writes <prefix>.json, <prefix>-light.png and <prefix>-dark.png",
    )
    parser.add_argument(
        "--eras",
        type=int,
        nargs="+",
        default=[0, 1000, 2000, 2536, 3000, 3200, 3300],
        help="left edges of the pooled eras, in training iterations",
    )
    parser.add_argument(
        "--run-boundary",
        type=int,
        default=2536,
        help="first iteration of the resumed run, shaded on the chart; 0 disables it",
    )
    return parser.parse_args()


def load_evidence(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    evidence = [
        {"snapshot": int(key), **record}
        for key, record in sorted(payload["league_matchup_evidence"].items())
        if key.isdigit()
    ]
    return {"learner_iteration": int(payload["iteration"]), "evidence": evidence}


def beta_summary(score: float, games: float) -> tuple[float, float, float]:
    """Posterior mean and 90% interval under the selector's Beta(1, 1) prior."""
    a, b = 1.0 + score, 1.0 + games - score
    return a / (a + b), beta_dist.ppf(0.05, a, b), beta_dist.ppf(0.95, a, b)


def plot(data: dict, eras: list[int], run_boundary: int, output: Path) -> list[dict]:
    learner = data["learner_iteration"]
    rows = data["evidence"]
    x = np.array([row["snapshot"] for row in rows])
    score = np.array([row["score_sum"] for row in rows])
    games = np.array([row["games"] for row in rows])
    single = (1.0 + score) / (2.0 + games)

    pooled = []
    for left, right in zip(eras, [*eras[1:], learner], strict=True):
        members = (x >= left) & (x < right)
        mean, low, high = beta_summary(score[members].sum(), games[members].sum())
        pooled.append(
            {
                "from": left,
                "to": right,
                "snapshots": int(members.sum()),
                "games": float(games[members].sum()),
                "score": mean,
                "low": low,
                "high": high,
            }
        )

    for mode, theme in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
        fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
        fig.patch.set_facecolor(theme["surface"])
        ax.set_facecolor(theme["surface"])

        if run_boundary:
            ax.axvspan(run_boundary - 0.5, learner, color=theme["band"], zorder=0, lw=0)
            ax.text(run_boundary + 10, 0.32, "resumed run", color=theme["muted"], fontsize=9)
            ax.text(15, 0.32, "first run", color=theme["muted"], fontsize=9)
        ax.axhline(0.5, color=theme["muted"], lw=1, ls=(0, (4, 3)), zorder=1)

        ax.scatter(x, single, s=10, color=theme["snapshot"], linewidth=0, zorder=2)
        for era in pooled:
            span = [era["from"], era["to"]]
            ax.fill_between(
                span, era["low"], era["high"], color=theme["series"], alpha=0.18, lw=0, zorder=3
            )
            ax.hlines(era["score"], *span, color=theme["series"], lw=2.5, zorder=4)
            ax.text(
                sum(span) / 2,
                era["high"] + 0.015,
                f"{era['score']:.0%}",
                color=theme["text"],
                fontsize=9,
                ha="center",
                va="bottom",
                zorder=5,
            )

        ax.set_xlim(-40, learner + 40)
        ax.set_ylim(0.3, 1.0)
        ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ax.set_yticks(ticks, [f"{tick:.0%}" for tick in ticks])
        ax.set_xlabel("Opponent snapshot (training iteration)", color=theme["muted"])
        ax.set_ylabel(f"Iteration {learner}'s score", color=theme["muted"])
        ax.grid(axis="y", color=theme["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])
        ax.tick_params(colors=theme["muted"], length=0)

        fig.suptitle(
            f"Iteration {learner} against its {len(x)} archived predecessors",
            x=0.065,
            ha="left",
            color=theme["text"],
            fontsize=13,
            fontweight="bold",
        )
        ax.set_title(
            "Score counts a draw as ½. Bars pool each era with a 90% interval; "
            "dots are single snapshots.",
            loc="left",
            color=theme["muted"],
            fontsize=10,
            pad=8,
        )
        fig.tight_layout()
        fig.savefig(f"{output}-{mode}.png", facecolor=theme["surface"])
        plt.close(fig)
    return pooled


def main() -> None:
    args = parse_args()
    data = load_evidence(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pooled = plot(data, args.eras, args.run_boundary, args.output)
    Path(f"{args.output}.json").write_text(json.dumps({**data, "eras": pooled}, indent=1) + "\n")
    for era in pooled:
        print(
            f"{era['from']:>5}-{era['to']:<5} {era['snapshots']:>3} snapshots "
            f"{era['score']:.1%} [{era['low']:.1%}, {era['high']:.1%}]"
        )


if __name__ == "__main__":
    main()
