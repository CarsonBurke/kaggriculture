#!/usr/bin/env python3
"""Measure what an entropy bonus buys against the built-in reference agents.

The actor this pipeline warm-starts from is near-deterministic: 0.1395 nats per
active component at the first actor-active iteration of the cancelled run, which
is 3.4% of the unit head's ln(59) ceiling and the entropy of a two-way choice
taken 96.9% one way. It loses every game to `starter`, a 28-line carrot loop,
ending on 92 money against its ~3480 -- so it does not merely fail to out-farm
`starter`, it destroys 2908 of the 3000 it is given, where `pass` keeps all 3000
by doing nothing at all.

Nothing in the shipped objective opposes that collapse. `policy_loss` is the
clipped surrogate alone, and sampling temperature cannot substitute because
`collect_self_play` rejects any learner temperature other than 1.0 to keep the
replay-parity contract. DAPO's Clip-Higher, which this pipeline does implement,
only lets a *sampled* low-probability action's probability grow -- it preserves
exploration and cannot restore it, and at 96.9% top-action mass there is nothing
left to preserve.

So the coefficient is a real decision and this measures it rather than guessing.
Each candidate runs the same short training loop from the same warm checkpoint --
the iteration-40 state, whose critic has its own 40 warmup iterations and whose
actor is still byte-identical to the clone -- against a league of the native
built-ins, and reports three things per iteration:

  * whether exploration actually returns, as mean entropy per active component;
  * whether it converts into play, as score rate and money against each
    built-in, which is the objective and not a proxy for it;
  * what it costs the trust region, as `max_approx_kl` and the share of the
    epoch's minibatches that completed before the KL gate latched.

What this deliberately does NOT measure: the long-run effect. A dozen iterations
cannot say whether a coefficient that helps here anneals correctly over 500, and
entropy that rises without the score rate following is a policy paying for noise.
Read the money column first; entropy is the mechanism, not the goal.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.league import load_actor_snapshot
from kaggriculture.opponents import BUILTIN_OPPONENTS
from kaggriculture.ppo import PpoConfig, make_optimizers, update_ppo
from kaggriculture.production import (
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_ROLLOUT_BFLOAT16,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_UPDATE_COMPILE_MODE,
    production_ppo_config,
)
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import collect_mixed_play_rust, slice_trajectories

#: Update metrics worth a column. `actor_updates` is the one that says whether
#: the trust region let the epoch finish, which every other number depends on.
REPORTED = (
    "entropy",
    "actor_updates",
    "max_approx_kl",
    "first_minibatch_approx_kl",
    "clip_fraction",
    "policy_loss",
    "critic_fit_explained_variance_last_epoch",
)


def _lane_statistics(
    league: Any, assignments: np.ndarray, lanes: list[str]
) -> dict[str, dict[str, float]]:
    """Score rate, margin and absolute money for each built-in lane.

    Absolute money is reported beside the relative score because the two answer
    different questions here and only the pair is diagnostic: the competition
    scores a scale-free relative bank, but the failure being fixed is capital
    destruction, and `money` is the only column that distinguishes a policy that
    has learned to out-earn `starter` from one that has merely learned to stop
    setting its own money on fire.
    """
    margins = league.final_money - league.opponent_money
    outcomes = (margins > 0).astype(np.float32) - (margins < 0).astype(np.float32)
    statistics: dict[str, dict[str, float]] = {}
    for index, lane in enumerate(lanes):
        selected = assignments == index
        games = int(selected.sum())
        if not games:
            continue
        statistics[lane] = {
            "games": float(games),
            "score_rate": float(((outcomes[selected] + 1.0) / 2.0).mean()),
            "mean_margin": float(margins[selected].mean()),
            "money": float(league.final_money[selected].mean()),
            "opponent_money": float(league.opponent_money[selected].mean()),
        }
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--coefficients",
        type=lambda value: [float(part) for part in value.split(",")],
        default=[0.0, 0.003, 0.01, 0.03],
        help="entropy coefficients to compare; the first should be 0.0 as the control",
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--games", type=int, default=PRODUCTION_SELF_PLAY_GAMES)
    parser.add_argument("--league-games", type=int, default=PRODUCTION_LEAGUE_GAMES)
    parser.add_argument(
        "--builtins",
        type=lambda value: [part for part in value.split(",") if part],
        default=sorted(BUILTIN_OPPONENTS),
        help="built-in lanes to fill the league with",
    )
    parser.add_argument(
        "--league-dir",
        type=Path,
        default=None,
        help="optional snapshot directory; omitted runs built-in lanes only",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.builtins) - BUILTIN_OPPONENTS)
    if unknown:
        raise SystemExit(f"unknown built-in opponents: {', '.join(unknown)}")
    if not args.builtins:
        raise SystemExit("the probe measures play against built-ins, so at least one is required")

    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    entry = resolve_architecture(state["architecture"])
    model_config = entry.config_class(**state["model_config"])
    actor = entry.actor_class(model_config).to(device)
    critic = entry.critic_class(model_config).to(device)
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    schedule = dict(production_ppo_config(update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE))

    opponents = []
    if args.league_dir is not None:
        opponents = [
            load_actor_snapshot(path, expected_model_config=model_config, device=device).eval()
            for path in sorted(args.league_dir.glob("*.pt"))
        ]
    # Built-in lanes trail the frozen networks in the lane index space, exactly as
    # `collect_mixed_play_rust` orders them, so the assignment below indexes both
    # strata with one arange.
    lanes = [f"snapshot_{index}" for index in range(len(opponents))] + list(args.builtins)
    assignments = np.arange(args.league_games) % len(lanes)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": int(state["iteration"]),
        "coefficients": args.coefficients,
        "iterations": args.iterations,
        "games": args.games,
        "league_games": args.league_games,
        "lanes": lanes,
        "seed": args.seed,
        "shipped_actor_learning_rate": float(schedule["actor_learning_rate"]),
        "target_kl": float(schedule["target_kl"]),
    }

    sweep: list[dict[str, Any]] = []
    for coefficient in args.coefficients:
        candidate_actor = copy.deepcopy(actor)
        candidate_critic = copy.deepcopy(critic)
        config = PpoConfig(**{**schedule, "entropy_coefficient": coefficient})
        actor_optimizer, critic_optimizer = make_optimizers(
            candidate_actor, candidate_critic, config
        )
        # Both optimizers are restored so every candidate starts from the same
        # Adam moments the checkpoint reached, rather than from a cold second
        # moment that would make the first steps of each candidate incomparable.
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        history: list[dict[str, Any]] = []
        for iteration in range(args.iterations):
            candidate_actor.eval()
            # Each iteration draws a fresh wave, and every candidate draws the
            # same waves in the same order: the seed advances with the iteration
            # and not with the coefficient, so a difference between candidates is
            # the coefficient rather than the games they happened to be dealt.
            seed = args.seed + iteration * (args.games + args.league_games) * 2
            rollout = collect_mixed_play_rust(
                candidate_actor,
                tuple(opponents),
                self_play_games=args.games,
                league_games=args.league_games,
                opponent_indices=assignments,
                builtin_lanes=tuple(args.builtins),
                seed_start=seed,
                episode_steps=PRODUCTION_EPISODE_STEPS,
                sampling_seed=seed,
                forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
                forward_autocast=PRODUCTION_ROLLOUT_BFLOAT16,
            )
            candidate_actor.train()
            metrics = update_ppo(
                candidate_actor,
                candidate_critic,
                actor_optimizer,
                critic_optimizer,
                rollout,
                config,
                generator=np.random.default_rng(seed),
            )
            league_part = slice_trajectories(rollout, args.games * 2, rollout.trajectories)
            row: dict[str, Any] = {
                "entropy_coefficient": coefficient,
                "iteration": iteration,
            }
            row.update({name: float(metrics[name]) for name in REPORTED if name in metrics})
            row["lanes"] = _lane_statistics(league_part, assignments, lanes)
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        sweep.append({"entropy_coefficient": coefficient, "history": history})

    report["sweep"] = sweep
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
