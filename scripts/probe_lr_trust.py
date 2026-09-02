"""Find the largest actor learning rate whose full epoch fits inside target_kl.

The trust region at 0.03 stops the BC-warm-started actor after 1-5 of 113
minibatches because a full epoch at 2.5e-4 moves the policy to a worst-minibatch
KL of 0.174. The decision taken is to keep the bound binding and bring the step
down to fit it, and the size of that reduction is a measurement rather than an
assumption: KL is only approximately linear in the step, and the acceptance test
is discrete -- either the epoch completes all 113 updates or the gate latches.

Sweeps candidate learning rates on one wave, each from the same checkpoint
weights and optimizer state, with the production bound in force. The answer is
the largest rate reporting `actor_updates == minibatches_per_epoch` and
`kl_early_stop == 0`, with margin left over on `max_approx_kl`.

`--games 160` is deliberate: 160 self-play games are 320 trajectories of 719
steps, which is 230,080 states and 113 minibatches at 2048 -- the exact shape of
the production league-mixed wave, so the step count under test is the step count
that ships. Only opponent identity differs, which moves the advantages but not
the number of sequential updates the bound has to survive.
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
from kaggriculture.ppo import PpoConfig, make_optimizers, update_ppo, update_replay_parity
from kaggriculture.production import (
    PRODUCTION_ROLLOUT_BFLOAT16,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_TEMPERATURE,
    PRODUCTION_UPDATE_COMPILE_MODE,
    production_ppo_config,
)
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import collect_mixed_play_rust, collect_self_play_rust

REPORTED = (
    "actor_updates",
    "kl_early_stop",
    "approx_kl",
    "max_approx_kl",
    "first_minibatch_approx_kl",
    "clip_fraction",
    "entropy",
    "actor_gradient_norm",
    "policy_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260812)
    # The self-play-only wave has production's exact state count -- 320
    # trajectories of 719 -- but not its opponent composition. A league wave
    # replaces 64 of those current-policy rows with 64 rows played against
    # frozen snapshots, whose state distribution is what the KL is measured on.
    # Pass `--league-games` to sweep the shipped mixture.
    parser.add_argument("--league-games", type=int, default=0)
    parser.add_argument("--league-dir", type=Path, default=None)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[2.5e-4, 1.0e-4, 5.0e-5, 3.0e-5, 1.5e-5],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    entry = resolve_architecture(state["architecture"])
    model_config = entry.config_class(**state["model_config"])
    actor = entry.actor_class(model_config).to(device)
    critic = entry.critic_class(model_config).to(device)
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    schedule = dict(production_ppo_config(update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE))

    actor.eval()
    if args.league_games:
        snapshots = sorted((args.league_dir or args.checkpoint.parent / "league").glob("*.pt"))
        if len(snapshots) < 4:
            raise SystemExit(f"need at least 4 league snapshots, found {len(snapshots)}")
        # The four the shipped selector would hold: the two most recent as
        # "active" opponents and two older ones as historical. Every one of them
        # decodes at the learner's own temperature, because production no longer
        # splits active sampling from argmax history, and a probe of the trust
        # region has to sample from the wave the trust region is measured on.
        chosen = [snapshots[-1], snapshots[-2], snapshots[len(snapshots) // 2], snapshots[0]]
        opponents = [
            load_actor_snapshot(path, expected_model_config=model_config, device=device).eval()
            for path in chosen
        ]
        rollout = collect_mixed_play_rust(
            actor,
            opponents,
            self_play_games=args.games,
            league_games=args.league_games,
            opponent_indices=np.arange(args.league_games) % len(opponents),
            seed_start=args.seed,
            temperature=PRODUCTION_TEMPERATURE,
            opponent_temperature=PRODUCTION_TEMPERATURE,
            sampling_seed=args.seed,
            forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
            forward_autocast=PRODUCTION_ROLLOUT_BFLOAT16,
        )
        wave = f"mixed({args.games} self-play + {args.league_games} league)"
    else:
        rollout = collect_self_play_rust(
            actor,
            games=args.games,
            seed_start=args.seed,
            sampling_seed=args.seed,
            forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
            forward_autocast=PRODUCTION_ROLLOUT_BFLOAT16,
        )
        wave = f"self-play({args.games})"
    states = int(rollout.unit_actions.shape[0] * rollout.unit_actions.shape[1])
    minibatch_size = int(schedule["minibatch_size"])
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": int(state["iteration"]),
        "games": args.games,
        "wave": wave,
        "seed": args.seed,
        "states": states,
        "minibatch_size": minibatch_size,
        "minibatches_per_epoch": -(-states // minibatch_size),
        "target_kl": float(schedule["target_kl"]),
        "shipped_actor_learning_rate": float(schedule["actor_learning_rate"]),
    }

    parity = update_replay_parity(
        actor,
        rollout,
        minibatch_size=minibatch_size,
        compile_mode=str(schedule["update_compile_mode"]),
        autocast_enabled=bool(schedule["use_bfloat16"]),
    )
    report["parity"] = {k: float(v) for k, v in parity.items()}

    sweep: list[dict[str, Any]] = []
    for rate in args.learning_rates:
        candidate_actor = copy.deepcopy(actor)
        candidate_critic = copy.deepcopy(critic)
        # lr_warmup_steps is zeroed so the measured rate is the rate applied on
        # every one of the 113 steps. With warmup left on, the early steps run
        # below the candidate and the epoch would clear a bound the shipped
        # schedule then exceeds once warmup finishes -- which is exactly how the
        # cancelled run cleared its first post-warmup iterations at 7 updates
        # and fell to 1 as the rate ramped.
        config = PpoConfig(**{**schedule, "actor_learning_rate": rate, "lr_warmup_steps": 0})
        actor_optimizer, critic_optimizer = make_optimizers(
            candidate_actor, candidate_critic, config
        )
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        # `load_state_dict` restores `param_groups`, and `_optimizer_step` reads
        # the rate from `group["base_lr"]` in preference to its argument, so
        # restoring the checkpoint puts the checkpoint's 2.5e-4 back and the
        # swept rate never reaches a single step. Re-stamping the groups after
        # the restore is what makes this sweep measure its own variable. The
        # first attempt at this probe omitted it and reported five identical
        # configurations as though they were five learning rates.
        for group in actor_optimizer.param_groups:
            group["base_lr"] = rate
            group["warmup_step"] = 0
            group["lr"] = rate
        metrics = update_ppo(
            candidate_actor,
            candidate_critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            config,
            generator=np.random.default_rng(args.seed),
        )
        row = {"actor_learning_rate": rate}
        row.update({k: float(metrics[k]) for k in REPORTED if k in metrics})
        row["completed_epoch"] = bool(
            int(metrics["actor_updates"]) == report["minibatches_per_epoch"]
        )
        row["kl_margin"] = report["target_kl"] / max(float(metrics["max_approx_kl"]), 1e-30)
        sweep.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    report["sweep"] = sweep
    admissible = [row for row in sweep if row["completed_epoch"]]
    report["largest_completing_rate"] = (
        max(row["actor_learning_rate"] for row in admissible) if admissible else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
