#!/usr/bin/env python3
"""Fit a lejepa and an entity critic on one frozen lejepa-policy wave; offline, not PPO evidence.

The question is where a `lejepa` run's low critic explained variance comes from:
the critic -- its tower reads the actor's detached belief through a few private
rounds -- or the data, since a near-deterministic policy may leave outcomes that
no critic can predict. Both critics are fitted from fresh initialization on the
same frozen wave, the same training rows and the same minibatch order, and read
back on held-out physical seeds after every epoch:

* `lejepa`: `LejepaCritic` as the trainer builds it for this actor, attached to
  the frozen actor's backbone by `registry.pair_towers`.
* `entity`: `EntityCritic` with its own encoder, on the actor's entity geometry
  with the entity baseline's critic options (`--entity-reference`).

A `lejepa` actor cannot sit in `update_ppo`'s actor slot for this. Without its
LeJEPA objective the update refuses a backbone no optimizer owns, and with it the
update hands the actor's belief to the critic as a fifth argument that
`EntityCritic` does not take. The slot therefore holds an inert entity actor.
With `actor_epochs=0` and no predictor the update never steps it, and nothing
the critic fits depends on it: Monte Carlo targets (gamma and critic GAE lambda
of one) read only rewards. Its one use is the discarded warm compile of actor
graphs a frozen wave always runs. The `lejepa` critic then encodes through its
attached backbone itself, the pass `LejepaCritic.encode_belief` runs whenever it
is not handed the belief the production update would pass: the same module at
the same weights on the same public inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.critic_diagnostics import critic_replay_arrays, terminal_outcomes
from kaggriculture.evaluation import artifact_seed_usage, seed_protocol
from kaggriculture.provenance import file_sha256, source_identity

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from fit_common_policy_critics import (  # noqa: E402
    HORIZON,
    LEAGUE_GAMES,
    SELF_PLAY_GAMES,
    TRAJECTORIES,
    fit_statistics,
    split_seed_rows,
)

#: The entity baseline whose critic options the `entity` variant takes.
ENTITY_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "runs/structural-gae-20260918/component-control/ppo/config.json"
)


def entity_critic_config(lejepa_config: Any, reference_model: dict[str, Any]) -> Any:
    """The actor's entity geometry under the reference run's critic-only options.

    Every non-critic field must already agree with the reference, so the tower
    is the baseline's critic and not a resized one; only the fields a critic
    alone reads are taken from the reference, which are exactly the ones the
    lejepa family sets differently (`critic_readout_ffn`, at the time of writing).
    """
    from kaggriculture.entity import EntityConfig
    from kaggriculture.modelargs import CRITIC_ONLY_MODEL_FIELDS

    names = {field.name for field in fields(EntityConfig)}
    values = {name: getattr(lejepa_config, name) for name in names}
    mismatched = sorted(
        name
        for name in names - CRITIC_ONLY_MODEL_FIELDS
        if reference_model.get(name) != values[name]
    )
    if mismatched:
        raise ValueError(
            "entity reference geometry differs from the actor's: " + ", ".join(mismatched)
        )
    values |= {name: reference_model[name] for name in names & CRITIC_ONLY_MODEL_FIELDS}
    return EntityConfig(**values)


def wave_summary(rollout: Any, outcomes: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Per-lane outcome spread and behavior entropy: the data side of the question."""
    components = (
        rollout.unit_active.reshape(rollout.trajectories, -1).sum(axis=1)
        + rollout.market_active.reshape(rollout.trajectories, -1).sum(axis=1)
        + rollout.market_quantity_active.reshape(rollout.trajectories, -1).sum(axis=1)
    )
    result = {}
    for group in ("all", *sorted(set(groups.tolist()))):
        chosen = np.arange(groups.size) if group == "all" else np.flatnonzero(groups == group)
        target = outcomes[chosen]
        result[group] = {
            "trajectories": int(chosen.size),
            "outcome_mean": float(np.mean(target)),
            "outcome_variance": float(np.var(target)),
            "wins": int(np.sum(target == 1.0)),
            "draws": int(np.sum(target == 0.0)),
            "losses": int(np.sum(target == -1.0)),
            "mean_entropy": float(
                rollout.entropy_sums[chosen].sum() / max(1, int(components[chosen].sum()))
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity-reference", type=Path, default=ENTITY_REFERENCE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=540)
    parser.add_argument("--seed-start", type=int, default=4_502_000)
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("epochs and max-seconds must be positive")
    seed_protocol("development", args.seed_start, SELF_PLAY_GAMES + LEAGUE_GAMES, usage=[])
    return args


def main() -> None:
    args = parse_args()
    import torch

    from kaggriculture.entity import EntityActor, EntityCritic
    from kaggriculture.inference import load_actor_artifact
    from kaggriculture.lejepa_model import LejepaActor, LejepaCritic
    from kaggriculture.ppo import PpoConfig, make_optimizers, replay_behavior_values, update_ppo
    from kaggriculture.production import production_ppo_config
    from kaggriculture.registry import pair_towers
    from kaggriculture.rollout import collect_mixed_play_rust
    from kaggriculture.training import checkpoint_agent_states

    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("lejepa critic fit requires queued CUDA with native BF16")
    started = time.perf_counter()
    torch.set_num_threads(1)
    actor_digest = file_sha256(args.actor)
    reference_digest = file_sha256(args.entity_reference)
    reference_model = json.loads(args.entity_reference.read_text(encoding="utf-8"))["model"]
    # A BC artifact or a single-learner PPO checkpoint; both carry the actor state
    # and model configuration `load_actor_artifact` reads.
    actor, metadata = load_actor_artifact(args.actor, device="cuda")
    if not isinstance(actor, LejepaActor):
        raise ValueError("this diagnostic requires a lejepa actor")
    # Frozen outright: the `lejepa` critic encodes through this actor's backbone,
    # and no value gradient may accumulate on it even where it is never stepped.
    actor.eval().requires_grad_(False)
    initial_actor = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    entity_config = entity_critic_config(actor.config, reference_model)
    protocol = seed_protocol(
        "development",
        args.seed_start,
        SELF_PLAY_GAMES + LEAGUE_GAMES,
        usage=artifact_seed_usage(metadata),
    )
    # The trainer's default update compile mode, which `lejepa-anchored-20260922`
    # recorded. Compilation changes kernels, not the fitted objective.
    config = PpoConfig(**production_ppo_config(update_compile_mode=PpoConfig.update_compile_mode))
    if config.gamma != 1.0 or config.critic_gae_lambda != 1.0 or config.critic_epochs != 1:
        raise ValueError("lejepa critic fit requires Monte Carlo critic targets and one epoch")
    if config.structured_actor_auxiliary_active or config.structured_critic_auxiliary_active:
        # Each would make the inert actor slot, or a predictor, part of the fit.
        raise ValueError("lejepa critic fit requires every actor-side objective off")
    assignments = np.arange(LEAGUE_GAMES) % 3
    rollout = collect_mixed_play_rust(
        actor,
        opponents=(actor,),
        self_play_games=SELF_PLAY_GAMES,
        league_games=LEAGUE_GAMES,
        opponent_indices=assignments,
        builtin_lanes=("starter", "scripted-v27"),
        seed_start=args.seed_start,
        deterministic=False,
        temperature=1.0,
        opponent_temperature=1.0,
        episode_steps=720,
        reward_mode="terminal-outcome",
        sampling_seed=20260919,
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    if rollout.state_count != TRAJECTORIES * HORIZON or not rollout.valid.all():
        raise RuntimeError("lejepa critic diagnostic requires the full production-size wave")
    outcomes = terminal_outcomes(rollout)
    train_rows, holdout_rows = split_seed_rows(rollout.episode_seeds)
    groups = np.asarray(
        ["self_play"] * (2 * SELF_PLAY_GAMES)
        + [
            # Lane zero plays the frozen actor itself, not a BC snapshot.
            str(np.asarray(["frozen_self", "starter", "scripted-v27"])[lane])
            for lane in assignments
        ]
    )
    report: dict[str, Any] = {
        "format_version": 1,
        "source_identity": source_identity(),
        "scope": "Offline critic fit on one frozen-policy wave; not online learning evidence",
        "actor": {
            "path": str(args.actor.resolve()),
            "sha256": actor_digest,
            "iteration": int(metadata.get("iteration", 0)),
            "source_identity": metadata["source_identity"],
            "model_config": metadata["model_config"],
        },
        "entity_reference": {
            "path": str(args.entity_reference.resolve()),
            "sha256": reference_digest,
            "model_config": reference_model,
        },
        "actor_slot": (
            "Inert fresh EntityActor on the actor's entity geometry; actor_epochs=0 never "
            "steps it and Monte Carlo critic targets never read it"
        ),
        "seed_protocol": protocol,
        "ppo": asdict(config),
        "epochs_requested": args.epochs,
        "max_seconds": args.max_seconds,
        "initialization_seed": 20260920,
        "sampling_seed": 20260919,
        "wave": {
            "self_play_games": SELF_PLAY_GAMES,
            "league_games": LEAGUE_GAMES,
            "states": rollout.state_count,
            "rollout_seconds": rollout.elapsed_seconds,
            "mean_entropy": rollout.mean_entropy,
            "summary": wave_summary(rollout, outcomes, groups),
            "episode_seeds": rollout.episode_seeds.tolist(),
            "seats": rollout.seats.tolist(),
            "opponents": groups.tolist(),
            "outcomes": outcomes.tolist(),
            "train_rows": train_rows.tolist(),
            "holdout_rows": holdout_rows.tolist(),
            "split": "Hold out complete physical seeds divisible by five; never split seats",
        },
        "variants": {},
    }

    def evaluate(critic: Any) -> dict[str, Any]:
        # Stage for this read only and release before update_ppo stages its training tensors.
        arrays = critic_replay_arrays(rollout)
        staged = {
            name: torch.from_numpy(array.reshape((-1, *array.shape[2:]))).to("cuda")
            for name, array in arrays.items()
        }
        predictions = (
            replay_behavior_values(
                critic,
                rollout.architecture,
                staged,
                compile_mode=config.update_compile_mode,
                autocast_enabled=True,
            )
            .cpu()
            .numpy()
            .reshape(TRAJECTORIES, HORIZON)
        )
        return {
            "train": fit_statistics(predictions, outcomes, train_rows, groups),
            "holdout": fit_statistics(predictions, outcomes, holdout_rows, groups),
        }

    def parameter_count(module: Any) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    # The run's own critic, read once on this wave as the reference the fresh
    # fits are compared against. Only a PPO checkpoint carries one, and its tower
    # was trained against this very checkpoint's backbone.
    trained_critic_state = checkpoint_agent_states(metadata)[0].get("critic")
    if trained_critic_state is not None:
        trained = LejepaCritic(actor.config).to("cuda")
        trained.load_state_dict(trained_critic_state)
        pair_towers(actor, trained)
        report["checkpoint_critic"] = evaluate(trained)
        del trained
    else:
        report["checkpoint_critic"] = None
    del trained_critic_state

    # Its geometry is the actor's by `entity_critic_config`'s check; the critic
    # options on that config are fields an actor never reads.
    torch.manual_seed(20260924)
    actor_slot = EntityActor(entity_config).to("cuda")
    initial_actor_slot = {
        key: value.detach().cpu().clone() for key, value in actor_slot.state_dict().items()
    }
    variants = {}
    for label in ("lejepa", "entity"):
        torch.manual_seed(20260920)
        if label == "lejepa":
            critic_config = actor.config
            critic = LejepaCritic(critic_config).to("cuda")
            # The trainer's own pairing: the tower reads this actor's backbone.
            pair_towers(actor, critic)
        else:
            critic_config = entity_config
            critic = EntityCritic(critic_config).to("cuda")
        actor_slot_optimizer, critic_optimizer = make_optimizers(actor_slot, critic, config)
        variants[label] = (critic, actor_slot_optimizer, critic_optimizer)
        report["variants"][label] = {
            "critic_class": type(critic).__name__,
            "model_config": asdict(critic_config),
            "trainable_parameters": parameter_count(critic),
            # The frozen backbone the lejepa tower reads is not the critic's to train.
            "frozen_encoder_parameters": (
                sum(parameter.numel() for parameter in actor.trunk.parameters())
                if label == "lejepa"
                else 0
            ),
            "epochs": [],
        }

    def save_progress(completed_epochs: int) -> None:
        report["epochs_completed"] = completed_epochs
        report["complete"] = completed_epochs == args.epochs and report.get(
            "actor_unchanged", False
        )
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    for label, (critic, *_rest) in variants.items():
        report["variants"][label]["epochs"].append({"epoch": 0, **evaluate(critic)})
    save_progress(0)
    for epoch in range(1, args.epochs + 1):
        if time.perf_counter() - started >= args.max_seconds:
            break
        for label, (critic, actor_slot_optimizer, critic_optimizer) in variants.items():
            metrics = update_ppo(
                actor_slot,
                critic,
                actor_slot_optimizer,
                critic_optimizer,
                rollout,
                config,
                generator=np.random.default_rng(20260922 + epoch),
                actor_epochs=0,
                rows=train_rows,
            )
            record = {"epoch": epoch, "update": metrics, **evaluate(critic)}
            report["variants"][label]["epochs"].append(record)
            holdout = record["holdout"]
            print(
                json.dumps(
                    {
                        "variant": label,
                        "epoch": epoch,
                        **holdout["all"]["all"],
                        "group_r_squared": {
                            group: windows["all"]["r_squared"]
                            for group, windows in holdout.items()
                            if group != "all"
                        },
                    }
                ),
                flush=True,
            )
        save_progress(epoch)
    if any(
        not torch.equal(initial_actor[name], value.detach().cpu())
        for name, value in actor.state_dict().items()
    ):
        raise RuntimeError("critic-only diagnostic mutated the frozen actor")
    if any(
        not torch.equal(initial_actor_slot[name], value.detach().cpu())
        for name, value in actor_slot.state_dict().items()
    ):
        raise RuntimeError("critic-only diagnostic stepped the inert actor slot")
    if file_sha256(args.actor) != actor_digest:
        raise RuntimeError("actor artifact changed during the diagnostic")
    report["actor_unchanged"] = True
    report["stop_reason"] = (
        "epoch_budget" if report["epochs_completed"] == args.epochs else "soft_time_budget"
    )
    save_progress(report["epochs_completed"])


if __name__ == "__main__":
    main()
