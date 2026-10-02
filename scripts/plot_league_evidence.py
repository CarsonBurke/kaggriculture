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
        "series": "#9ec5f4",
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
        "--window",
        type=int,
        default=8,
        help="snapshots on each side pooled into the trend line",
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


def opponent_score(score: np.ndarray, games: np.ndarray, members: np.ndarray) -> dict:
    """The snapshots' pooled score against the learner, from the learner's record."""
    mean, low, high = beta_summary(score[members].sum(), games[members].sum())
    return {
        "snapshots": int(members.sum()),
        "games": float(games[members].sum()),
        "score": 1.0 - mean,
        "low": 1.0 - high,
        "high": 1.0 - low,
    }


def plot(data: dict, window: int, output: Path) -> dict:
    learner = data["learner_iteration"]
    rows = data["evidence"]
    iteration = np.array([row["snapshot"] for row in rows])
    score = np.array([row["score_sum"] for row in rows])
    games = np.array([row["games"] for row in rows])
    single = 1.0 - (1.0 + score) / (2.0 + games)
    index = np.arange(len(rows))

    # The archive keeps spacing that doubles with age, so plotting snapshots in
    # order puts recent ones at full resolution and old ones roughly log-spaced.
    trend = [
        {
            "snapshot": int(iteration[position]),
            **opponent_score(score, games, np.abs(index - position) <= window),
        }
        for position in index
    ]
    old = opponent_score(score, games, iteration < learner - 1000)
    recent = opponent_score(score, games, iteration >= learner - 100)

    for mode, theme in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
        fig.patch.set_facecolor(theme["surface"])
        ax.set_facecolor(theme["surface"])

        ax.bar(index, single, width=0.8, color=theme["snapshot"], linewidth=0, zorder=2)
        ax.fill_between(
            index,
            [point["low"] for point in trend],
            [point["high"] for point in trend],
            color=theme["series"],
            alpha=0.12,
            lw=0,
            zorder=3,
        )
        ax.plot(index, [point["score"] for point in trend], color=theme["series"], lw=2, zorder=4)
        ax.axhline(0.5, color=theme["muted"], lw=1, ls=(0, (4, 3)), zorder=5)

        older = iteration < learner - 1000
        newer = iteration >= learner - 100
        for members, pooled, label, align in (
            (older, old, "snapshots over 1,000 iterations older", "left"),
            (newer, recent, "the last 100 iterations", "right"),
        ):
            span = index[members]
            ax.hlines(0.635, span.min(), span.max(), color=theme["text"], lw=1)
            ax.text(
                span.min() if align == "left" else span.max(),
                0.65,
                f"{pooled['score']:.0%} for {label}",
                ha=align,
                color=theme["text"],
                fontsize=10,
                fontweight="bold",
            )

        labelled = [0, 2500, 3000, 3200, 3300, 3350, learner]
        positions = [int(np.argmin(np.abs(iteration - target))) for target in labelled]
        ax.set_xticks(positions, [f"{iteration[position]:,}" for position in positions])
        ax.set_xlim(-1, len(rows))
        ax.set_ylim(0, 0.7)
        yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ax.set_yticks(yticks, [f"{tick:.0%}" for tick in yticks])
        ax.set_xlabel(
            "Predecessor snapshot (training iteration, archive order)", color=theme["muted"]
        )
        ax.set_ylabel(f"Win rate against {learner}", color=theme["muted"])
        ax.grid(axis="y", color=theme["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])
        ax.tick_params(colors=theme["muted"], length=0)

        fig.suptitle(
            f"How often each predecessor beats the final policy (iteration {learner})",
            x=0.065,
            ha="left",
            color=theme["text"],
            fontsize=13,
            fontweight="bold",
        )
        ax.set_title(
            "A draw counts as half a win. Each bar is one of the 239 archived snapshots; "
            "the line pools neighbours with a 90% interval.",
            loc="left",
            color=theme["muted"],
            fontsize=9.5,
            pad=8,
        )
        fig.tight_layout()
        fig.savefig(f"{output}-{mode}.png", facecolor=theme["surface"])
        plt.close(fig)
    return {"older_than_1000": old, "last_100": recent, "trend": trend}


def main() -> None:
    args = parse_args()
    data = load_evidence(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = plot(data, args.window, args.output)
    Path(f"{args.output}.json").write_text(json.dumps({**data, **summary}, indent=1) + "\n")
    for name in ("older_than_1000", "last_100"):
        pooled = summary[name]
        print(
            f"{name}: {pooled['snapshots']} snapshots {pooled['score']:.1%} "
            f"[{pooled['low']:.1%}, {pooled['high']:.1%}]"
        )


if __name__ == "__main__":
    main()
