#!/usr/bin/env python3
"""Compare PPO schedules over several iterations against the built-in agents.

A single update cannot answer a schedule question. Whether a learning rate, a
trust region or an entropy coefficient is right shows up as the trajectory of
play over iterations -- entropy that returns and then converts into money, or an
epoch that keeps completing -- and neither is visible in one step's metrics.

So each candidate here runs the same short training loop from the same warm
checkpoint, drawing the same waves in the same order, and reports per iteration:

  * whether the epoch survives its trust region, as `actor_updates` against
    `actor_minibatches_intended` and the KL that stopped it;
  * whether exploration is present, as mean entropy per active component;
  * whether either converts into play, as score rate AND absolute money against
    each built-in lane, which is the objective rather than a proxy for it.

Money is reported beside the relative score because only the pair is diagnostic.
The competition scores a scale-free relative bank, but the failure this measures
is capital destruction: the warm-start clone ends on 92 money against `starter`'s
~3480, having burned 2908 of the 3000 it was given, where `pass` keeps all 3000
by doing nothing. A policy that has learned to stop setting its own money on fire
and one that has learned to out-earn `starter` look the same in score rate and
completely different in money.

Two questions this instrument was built to answer, in order:

  1. The entropy coefficient. DAPO's Clip-Higher is implemented here, and it
     only lets a *sampled* low-probability action's probability grow: at 0.1395
     nats per active component -- 3.4% of the unit head's ln(59) ceiling, a
     two-way choice taken 96.9% one way -- there is nothing left for it to act
     on. It preserves exploration and cannot restore it. Sampling temperature
     cannot substitute, since `rollout.py` rejects any learner temperature other
     than 1.0 to hold the replay-parity contract, which is what makes the entropy
     bonus a term in the objective rather than a sampling knob.
  2. The trust region. `target_kl` was calibrated on a self-play-only wave, where
     the actor had almost no gradient. Against opponents it cannot beat, the same
     bound stops the epoch after a fraction of its minibatches, and below
     `MINIMUM_ACTOR_EPOCH_FRACTION` the run aborts by design.

What this deliberately does NOT measure: the long run. A dozen iterations cannot
say whether a setting that helps here still holds over 500, and a candidate whose
entropy rises without money following is paying for noise.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.league import load_actor_snapshot
from kaggriculture.opponents import BUILTIN_OPPONENTS
from kaggriculture.ppo import (
    PpoConfig,
    make_optimizers,
    make_structured_dynamics_optimizer,
    update_ppo,
)
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
from kaggriculture.structured_dynamics import StructuredCriticDynamics, StructuredDynamics
from kaggriculture.training import checkpoint_agent_states, require_checkpoint_format

#: Update metrics worth a column. The pair `actor_updates` /
#: `actor_minibatches_intended` is the one that says whether the trust region
#: let the epoch finish, which every other number here depends on.
REPORTED = (
    "entropy",
    "actor_updates",
    "actor_minibatches_intended",
    "max_approx_kl",
    "first_minibatch_approx_kl",
    "kl_early_stop",
    "clip_fraction",
    "policy_loss",
    "actor_gradient_norm",
    "critic_fit_explained_variance_last_epoch",
)


def _optimizer_uses_normuon(value: object, name: str) -> bool:
    """Read the optimizer family from a checkpoint state without guessing."""
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} state is incomplete")
    groups = value.get("param_groups")
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(group, Mapping) for group in groups)
    ):
        raise ValueError(f"checkpoint {name} parameter groups are incomplete")
    group_kinds = ["kind" in group for group in groups]
    if any(group_kinds) != all(group_kinds):
        raise ValueError(f"checkpoint {name} mixes optimizer parameter-group formats")
    return all(group_kinds)


def _lane_statistics(
    league: Any, assignments: np.ndarray, lanes: list[str]
) -> dict[str, dict[str, float]]:
    """Score rate, margin and absolute money for each league lane."""
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


def _label(overrides: dict[str, Any]) -> str:
    """Name a candidate by what it changes, so a report row is self-describing."""

    def rendered(value: Any) -> str:
        # `:g` is for the numeric knobs this began as; a knob whose value is a
        # name -- `optimizer` -- is not formattable that way and crashed here.
        return f"{value:g}" if isinstance(value, (int, float)) else str(value)

    return ",".join(f"{name}={rendered(value)}" for name, value in sorted(overrides.items())) or (
        "shipped"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--configs",
        type=json.loads,
        required=True,
        help="JSON list of PpoConfig overrides, e.g. '[{}, {\"target_kl\": 0.05}]'",
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--games", type=int, default=PRODUCTION_SELF_PLAY_GAMES)
    parser.add_argument("--league-games", type=int, default=PRODUCTION_LEAGUE_GAMES)
    parser.add_argument(
        "--builtins",
        type=lambda value: [part for part in value.split(",") if part],
        default=sorted(BUILTIN_OPPONENTS),
    )
    parser.add_argument(
        "--league-dir",
        type=Path,
        default=None,
        help="snapshot directory; omitted runs built-in lanes only",
    )
    parser.add_argument(
        "--snapshot-lanes",
        type=int,
        default=0,
        help=(
            "frozen snapshot lanes to run beside the built-ins. Production admits "
            "2 active and 2 historical against at most 3 built-ins, so 4 here "
            "reproduces the share of the wave a built-in actually receives"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.builtins) - BUILTIN_OPPONENTS)
    if unknown:
        raise SystemExit(f"unknown built-in opponents: {', '.join(unknown)}")
    if not args.builtins:
        raise SystemExit("the probe measures play against built-ins, so at least one is required")
    if args.snapshot_lanes and args.league_dir is None:
        raise SystemExit("--snapshot-lanes needs --league-dir to load them from")

    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    require_checkpoint_format(state)
    member_state = checkpoint_agent_states(state)[0]
    required_auxiliary_state = {
        "structured_dynamics",
        "structured_dynamics_optimizer",
        "structured_critic_dynamics",
        "structured_critic_dynamics_optimizer",
    }
    missing_auxiliary_state = sorted(required_auxiliary_state - member_state.keys())
    if "structured_auxiliary_rng" not in state:
        missing_auxiliary_state.append("structured_auxiliary_rng")
    if missing_auxiliary_state:
        raise ValueError(
            "checkpoint is missing production structured auxiliary recovery state: "
            + ", ".join(missing_auxiliary_state)
        )

    gate_record = state.get("structured_gate_state")
    if (
        not isinstance(gate_record, Mapping)
        or gate_record.get("version") != 2
        or set(gate_record) != {"version", "agents"}
    ):
        raise ValueError("checkpoint structured predictor gate state is not version 2")
    gate_agents = gate_record["agents"]
    if not isinstance(gate_agents, list) or not gate_agents:
        raise ValueError("checkpoint has no structured predictor gate state for member zero")
    gate_member = gate_agents[0]
    if not isinstance(gate_member, Mapping) or set(gate_member) != {"actor", "critic"}:
        raise ValueError("checkpoint member-zero structured predictor gate state is incomplete")
    enabled: dict[str, bool] = {}
    for kind in ("actor", "critic"):
        gate = gate_member[kind]
        if not isinstance(gate, Mapping) or not isinstance(gate.get("enabled"), bool):
            raise ValueError(f"checkpoint member-zero structured {kind} gate is incomplete")
        enabled[kind] = gate["enabled"]

    saved_auxiliary_rng = copy.deepcopy(state["structured_auxiliary_rng"])
    recovered_auxiliary_generator = np.random.default_rng()
    try:
        recovered_auxiliary_generator.bit_generator.state = copy.deepcopy(saved_auxiliary_rng)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint structured auxiliary RNG state is invalid") from error
    saved_auxiliary_rng = copy.deepcopy(recovered_auxiliary_generator.bit_generator.state)

    optimizer_state_names = (
        "actor_optimizer",
        "critic_optimizer",
        "structured_dynamics_optimizer",
        "structured_critic_dynamics_optimizer",
    )
    saved_optimizer_families = {
        _optimizer_uses_normuon(member_state[name], name) for name in optimizer_state_names
    }
    if len(saved_optimizer_families) != 1:
        raise ValueError("checkpoint member-zero optimizer families are inconsistent")
    saved_is_normuon = saved_optimizer_families.pop()

    entry = resolve_architecture(state["architecture"])
    model_config = entry.build_config(state["model_config"])
    actor = entry.actor_class(model_config).to(device)
    critic = entry.critic_class(model_config).to(device)
    actor.load_state_dict(member_state["actor"])
    critic.load_state_dict(member_state["critic"])
    schedule = dict(production_ppo_config(update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE))
    production_config = PpoConfig(**schedule)
    if not (
        production_config.structured_actor_auxiliary_active
        and production_config.structured_critic_auxiliary_active
    ):
        raise RuntimeError("production PPO config must enable actor and critic NextLat")

    opponents = []
    if args.snapshot_lanes:
        available = sorted((args.league_dir or Path()).glob("*.pt"))
        if len(available) < args.snapshot_lanes:
            raise SystemExit(f"need {args.snapshot_lanes} snapshots, found {len(available)}")
        # The two most recent stand in for production's `active` stratum and the
        # oldest for its log-age `historical` draw, which at a warm start is the
        # behaviour-cloned policy the run departs from.
        chosen = [available[-1], available[-2], available[0], available[len(available) // 2]]
        opponents = [
            load_actor_snapshot(path, expected_model_config=model_config, device=device).eval()
            for path in chosen[: args.snapshot_lanes]
        ]
    # Built-in lanes trail the frozen networks in the lane index space, exactly as
    # `collect_mixed_play_rust` assigns them, so one arange indexes both strata.
    lanes = [f"snapshot_{index}" for index in range(len(opponents))] + list(args.builtins)
    assignments = np.arange(args.league_games) % len(lanes)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": int(state["iteration"]),
        "configs": args.configs,
        "iterations": args.iterations,
        "games": args.games,
        "league_games": args.league_games,
        "lanes": lanes,
        "seed": args.seed,
        "shipped": {
            "actor_learning_rate": float(schedule["actor_learning_rate"]),
            "target_kl": float(schedule["target_kl"]),
        },
    }

    sweep: list[dict[str, Any]] = []
    for overrides in args.configs:
        label = _label(overrides)
        candidate_actor = copy.deepcopy(actor)
        candidate_critic = copy.deepcopy(critic)
        config = PpoConfig(**{**schedule, **overrides})
        if not (
            config.structured_actor_auxiliary_active and config.structured_critic_auxiliary_active
        ):
            raise ValueError(f"schedule candidate {label!r} must keep actor and critic NextLat")

        candidate_dynamics = StructuredDynamics(model_config).to(device)
        candidate_critic_dynamics = StructuredCriticDynamics(model_config).to(device)
        candidate_dynamics.load_state_dict(member_state["structured_dynamics"])
        candidate_critic_dynamics.load_state_dict(member_state["structured_critic_dynamics"])
        actor_optimizer, critic_optimizer = make_optimizers(
            candidate_actor, candidate_critic, config
        )
        dynamics_optimizer = make_structured_dynamics_optimizer(candidate_dynamics, config)
        critic_dynamics_optimizer = make_structured_dynamics_optimizer(
            candidate_critic_dynamics, config
        )
        auxiliary_generator = np.random.default_rng()
        auxiliary_generator.bit_generator.state = copy.deepcopy(saved_auxiliary_rng)

        # All optimizer moments are inherited when their representation is
        # compatible. Switching optimizer families cold-starts all four
        # optimizers while retaining every network's checkpoint weights.
        restored = saved_is_normuon == (config.optimizer == "normuon")
        if restored:
            actor_optimizer.load_state_dict(member_state["actor_optimizer"])
            critic_optimizer.load_state_dict(member_state["critic_optimizer"])
            dynamics_optimizer.load_state_dict(member_state["structured_dynamics_optimizer"])
            critic_dynamics_optimizer.load_state_dict(
                member_state["structured_critic_dynamics_optimizer"]
            )
        # Restoring param groups also restores their checkpoint rates. Re-stamp
        # the swept actor and predictor rates, and restart their short-probe
        # warmup clocks, so candidate overrides reach every intended step.
        for optimizer, base_rate in (
            (actor_optimizer, config.actor_learning_rate),
            (dynamics_optimizer, config.resolved_structured_learning_rate),
            (critic_dynamics_optimizer, config.resolved_structured_learning_rate),
        ):
            for group in optimizer.param_groups:
                rate = base_rate
                if group.get("kind") == "adam":
                    rate *= config.adam_learning_rate_ratio
                group["base_lr"] = rate
                group["warmup_step"] = 0
                group["lr"] = rate
        history: list[dict[str, Any]] = []
        for iteration in range(args.iterations):
            candidate_actor.eval()
            # Every candidate draws the same waves in the same order: the seed
            # advances with the iteration and not with the candidate, so a
            # difference between them is the schedule rather than the games each
            # happened to be dealt.
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
                structured_dynamics=candidate_dynamics,
                structured_dynamics_optimizer=dynamics_optimizer,
                structured_actor_auxiliary=enabled["actor"],
                structured_critic_dynamics=candidate_critic_dynamics,
                structured_critic_dynamics_optimizer=critic_dynamics_optimizer,
                structured_critic_auxiliary=enabled["critic"],
                auxiliary_generator=auxiliary_generator,
            )
            league_part = slice_trajectories(rollout, args.games * 2, rollout.trajectories)
            row: dict[str, Any] = {"config": label, "iteration": iteration}
            row.update({name: float(metrics[name]) for name in REPORTED if name in metrics})
            intended = float(metrics.get("actor_minibatches_intended") or 0.0)
            row["epoch_fraction"] = (
                float(metrics["actor_updates"]) / intended if intended > 0 else float("nan")
            )
            row["lanes"] = _lane_statistics(league_part, assignments, lanes)
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        sweep.append(
            {
                "config": label,
                "overrides": overrides,
                "inherited_optimizer_moments": restored,
                "history": history,
            }
        )

    report["sweep"] = sweep
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
