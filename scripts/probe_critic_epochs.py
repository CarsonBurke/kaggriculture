#!/usr/bin/env python3
"""Measure whether the last critic epochs of a PPO update generalize or memorize.

The original schedule measured here ran `epochs=1, critic_epochs=4` at
`minibatch_size=2048`; the critic refit was roughly 26 s of a 33.7 s iteration,
about 6.5 s per critic epoch. The hypothesis under test is that the later epochs
buy nothing the run can use. The run's own telemetry raises it:
`critic_fit_explained_variance_first_epoch` against `_last_epoch` measured
0.906 against 0.987, which is the signature of a
regression that keeps improving on the rollout batch it is being fitted to
rather than on states it has not seen.

Neither of those two numbers can settle it, because both are scored on states
the epoch in question is fitting. This instrument holds states out instead.

The holdout is BY GAME, never by state. States along one trajectory are
strongly correlated, so a random state split puts near-duplicates of fitted
states into the holdout and would report memorization as generalization. Games
are split 80/20 on `episode_seeds` -- both self-play seats of a game share one
seed and therefore land on the same side -- the critic is refitted on the fit
games alone under the measured schedule (same `_stage_tensor` staging, same
`_fixed_minibatch_positions` partitioning, same `make_optimizers` construction
and `_optimizer_step` warmup, same bf16 autocast, same `update_compile_mode`,
same `_critic_minibatch_objective`), and after every epoch
explained variance and the optimized distributional loss are measured on both
sides at frozen weights, with a CUDA-synchronized wall clock around each epoch's
gradient steps. Explained variance uses the same four target/residual moments
and `_fit_explained_variance` definition as run telemetry; the probe computes
those moments outside the timed path.

The curve runs past the production schedule so its shape is measured rather
than extrapolated, and over several repeats with different rollout seeds.
With ``--actor`` those repeats use different fresh critic initializations;
with ``--checkpoint`` they clone the same real critic and optimizer state so
the only between-repeat variation is the rollout and game split.

FALSIFICATION. The hypothesis "later epochs are wasted" is falsified if the
holdout explained variance still gained from epoch 2 onwards -- differenced per
repeat, so each curve is its own control -- exceeds the across-repeat spread of
that same difference: those epochs are then buying generalization and their wall
clock is earned. It is supported if the holdout curve goes flat within that
spread while the fit curve keeps climbing, which is memorization. `decision.rule`
states the exact test, and it is the strict direction of the two available: it
scores the whole remaining gain rather than one epoch at a time, so a run of
small real gains that each hide under the noise still fails it.

Only the critic runs here, and that is exact rather than a simplification: the
actor and critic are separate modules with separate optimizers, the value
targets are fixed before the update begins, and so an actor epoch cannot change
what the critic fits.

The default ``--actor`` mode uses a fresh critic with a zero-initialized value
head. That is the pipeline's starting point, but it is biased towards finding
later epochs useful. ``--warmup-iterations`` fits disjoint waves first to
approach the warm regime. ``--checkpoint`` is the preferred steady-state test:
it restores both critic weights and optimizer moments before every repeat and
does not permit synthetic warmup. A value-fit curve is not end-to-end policy
improvement; the report records that caveat explicitly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._dynamo
from torch import Tensor

from kaggriculture.constants import DEFAULT_REWARD_MODE
from kaggriculture.inference import load_actor_artifact
from kaggriculture.ppo import (
    UPDATE_COMPILE_MODES,
    PpoConfig,
    _batch_tensor,
    _cached_update_callable,
    _critic_batch_args,
    _critic_minibatch_fit_terms,
    _critic_minibatch_objective,
    _device_compile_mode,
    _entity_active_batch,
    _fit_explained_variance,
    _fit_moment_mapping,
    _fixed_minibatch_positions,
    _optimizer_step,
    _stage_tensor,
    make_optimizers,
    prepare_advantages,
    replay_behavior_values,
)
from kaggriculture.production import (
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_ROLLOUT_BFLOAT16,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_UPDATE_COMPILE_MODE,
)
from kaggriculture.provenance import source_identity
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import (
    REWARD_MODES,
    ROLLOUT_FORWARD_MODES,
    RolloutBatch,
    allocate_rollout_storage,
    collect_self_play_rust,
)


def _production_critic_minibatches_per_epoch(minibatch_size: int) -> int:
    """Critic minibatches one production epoch runs, from the shipped wave shape.

    (128 self-play games x 2 seats + 64 league games) x 719 stored steps is
    230,080 states, every one of them a valid learner state, and
    `_fixed_minibatch_positions` splits an epoch into
    ceil(states / minibatch_size) minibatches of exactly `minibatch_size` rows
    -- 57 at the current 4096-row ceiling, or 228 critic minibatches over four
    epochs. Only the last one can hold fewer than `minibatch_size` fresh rows,
    and it wraps onto the epoch's leading rows rather than running short, so
    every minibatch is one compiled shape.

    This probe fits self-play states minus a holdout, so its per-epoch seconds
    are reported raw AND rescaled through this count, which is the unit an
    epoch's cost transfers in: the minibatch size is fixed, so an epoch costs its
    minibatch count times a per-minibatch cost.
    """
    trajectories = PRODUCTION_SELF_PLAY_GAMES * 2 + PRODUCTION_LEAGUE_GAMES
    return math.ceil(trajectories * (PRODUCTION_EPISODE_STEPS - 1) / minibatch_size)


#: Rollout fields outside ``states`` that critic replay needs. ``unit_active``
#: is a structured actor input reused by the centralized critic; unit actions
#: supply only the flattened row count used while constructing value targets.
_CRITIC_SHARED_FIELDS = ("unit_actions", "unit_active", "market_active")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _spread(values: list[float]) -> float:
    """Sample standard deviation across repeats; 0.0 when one repeat cannot show one."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _stage_rollout(rollout: RolloutBatch, device: torch.device) -> dict[str, Tensor]:
    """Stage every critic-visible rollout array once, exactly as `update_ppo` does."""
    staged = {name: _stage_tensor(array, device) for name, array in rollout.states.items()}
    staged |= {
        name: _stage_tensor(getattr(rollout, name), device) for name in _CRITIC_SHARED_FIELDS
    }
    return staged


def _stage_value_targets(
    rollout: RolloutBatch,
    critic: torch.nn.Module,
    architecture: str,
    staged: dict[str, Tensor],
    config: PpoConfig,
    *,
    compile_mode: str,
    autocast_enabled: bool,
    device: torch.device,
) -> dict[str, float]:
    """Build and stage the value targets `update_ppo` fits, by its own route.

    Behavior values are replayed from the staged features through the critic at
    unchanged weights, advantages and lambda-return targets come from
    `prepare_advantages`, and the targets are clipped to the critic's support
    the way the update clips them -- the critic regresses on the clipped copy,
    so scoring a fit against the unclipped one would measure the support width.
    """
    behavior_values = (
        replay_behavior_values(
            critic,
            architecture,
            staged,
            compile_mode=compile_mode,
            autocast_enabled=autocast_enabled,
        )
        .cpu()
        .numpy()
        .reshape(rollout.rewards.shape)
    )
    prepared = prepare_advantages(rollout, behavior_values, config)
    support = critic.support.detach().float().cpu().numpy()
    valid_targets = prepared.value_targets[rollout.valid]
    if not np.isfinite(valid_targets).all():
        raise ValueError("value targets must be finite")
    saturated = np.count_nonzero((valid_targets < support[0]) | (valid_targets > support[-1]))
    clipped = np.clip(prepared.value_targets, support[0], support[-1])
    staged["value_targets"] = torch.from_numpy(clipped.reshape(-1)).to(device)
    return {
        "value_target_mean": float(valid_targets.mean()),
        "value_target_std": float(valid_targets.std()),
        "value_target_saturated_fraction": float(saturated) / float(valid_targets.size),
        "behavior_value_mean": float(behavior_values[rollout.valid].mean()),
        "behavior_value_std": float(behavior_values[rollout.valid].std()),
    }


def _split_games(
    rollout: RolloutBatch, holdout_fraction: float, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Partition valid state indices by GAME, never by state.

    `episode_seeds` is the game identity: self-play stores both seats of a game
    as adjacent trajectories carrying the same seed, so grouping on it keeps a
    game's two trajectories -- which observe the same board from opposite sides
    -- on the same side of the split. Splitting on states, or even on
    trajectories, would leak: consecutive states of one 719-step trajectory are
    near-duplicates of each other.
    """
    seeds = np.unique(rollout.episode_seeds)
    holdout_count = round(seeds.size * holdout_fraction)
    if not 1 <= holdout_count < seeds.size:
        raise ValueError("holdout fraction must leave at least one game on each side")
    holdout_seeds = generator.permutation(seeds)[:holdout_count]
    trajectory_holdout = np.isin(rollout.episode_seeds, holdout_seeds)
    # Trajectory-major storage: flat state index is trajectory * horizon + step,
    # so the per-trajectory mask expands by repetition.
    holdout_rows = np.repeat(trajectory_holdout, rollout.horizon)
    flat_valid = rollout.valid.reshape(-1)
    fit_indices = np.flatnonzero(flat_valid & ~holdout_rows)
    holdout_indices = np.flatnonzero(flat_valid & holdout_rows)
    return (
        fit_indices,
        holdout_indices,
        {
            "fit_games": int(seeds.size - holdout_count),
            "holdout_games": holdout_count,
            "fit_states": int(fit_indices.size),
            "holdout_states": int(holdout_indices.size),
            "fit_trajectories": int(np.count_nonzero(~trajectory_holdout)),
            "holdout_trajectories": int(np.count_nonzero(trajectory_holdout)),
        },
    )


def _critic_epoch(
    critic: torch.nn.Module,
    critic_optimizer: torch.optim.Optimizer,
    architecture: str,
    staged: dict[str, Tensor],
    indices: np.ndarray,
    config: PpoConfig,
    *,
    compile_mode: str,
    autocast_enabled: bool,
    generator: np.random.Generator,
    device: torch.device,
) -> tuple[float, int]:
    """Run one critic epoch over `indices` exactly as `update_ppo`'s loop does.

    Returns the CUDA-synchronized wall clock of the epoch's gradient steps and
    the minibatch count, which is the unit the cost transfers in. The non-finite
    handling mirrors the critic-only branch of the update: the fused optimizer's
    own device-side skip, read once at the epoch boundary, so no minibatch pays
    a host synchronization.
    """
    loss_fn = _cached_update_callable(
        critic, "_kaggriculture_update_objective", _critic_minibatch_objective, compile_mode
    )
    gateable = any(group.get("fused", False) for group in critic_optimizer.param_groups)
    nonfinite = torch.zeros((), dtype=torch.float64, device=device)
    shuffled = generator.permutation(indices)
    positions, _counts = _fixed_minibatch_positions(shuffled.size, config.minibatch_size)
    # One staged (batch_count, minibatch_size) index tensor, gathered on the
    # host before the clock starts: the timed loop must not pay a host-to-device
    # copy per minibatch.
    batch_indices = torch.from_numpy(shuffled[positions]).to(device=device)
    was_training = critic.training
    critic.train()
    _synchronize(device)
    started = time.perf_counter()
    for batch in batch_indices:
        critic_args = _critic_batch_args(architecture, staged, batch)
        value_targets = _batch_tensor(staged["value_targets"], batch, torch.float32)
        critic_optimizer.zero_grad(set_to_none=True)
        value_loss = loss_fn(
            critic,
            value_targets,
            autocast_enabled,
            *critic_args,
            entity_active=(
                _entity_active_batch(staged, batch)
                if getattr(critic.config, "per_entity_critic", False)
                else None
            ),
        )
        value_loss.backward()
        if gateable:
            critic_skip = (~torch.isfinite(value_loss.detach())).float()
            nonfinite += critic_skip
        else:
            critic_skip = None
            if not math.isfinite(float(value_loss.detach())):
                raise FloatingPointError("non-finite critic loss")
        _optimizer_step(
            critic_optimizer,
            config.critic_learning_rate,
            config.lr_warmup_steps,
            found_inf=critic_skip,
        )
    _synchronize(device)
    seconds = time.perf_counter() - started
    critic.train(was_training)
    if nonfinite.item():
        raise FloatingPointError("non-finite critic loss")
    return seconds, len(batch_indices)


@torch.no_grad()
def _evaluate(
    critic: torch.nn.Module,
    architecture: str,
    staged: dict[str, Tensor],
    indices: np.ndarray,
    config: PpoConfig,
    *,
    autocast_enabled: bool,
    device: torch.device,
) -> dict[str, float]:
    """Score the frozen critic on one state set: repo-definition EV plus its loss.

    Uncompiled deliberately. Dynamo specializes on the grad context, so routing
    this through the update's compiled entry would add a second cache entry per
    minibatch shape on the same code object the gradient path already fills,
    against a per-code-object limit of eight that a `fullgraph=True` region
    overruns as a hard failure. This pass is off the timed path, runs the same
    `_critic_minibatch_fit_terms` under the same autocast, and the
    Inductor-versus-eager difference on an explained variance is far below the
    gains being resolved.
    """
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    loss_total = torch.zeros((), dtype=torch.float64, device=device)
    states = int(indices.size)
    positions, counts = _fixed_minibatch_positions(states, config.minibatch_size)
    was_training = critic.training
    critic.eval()
    try:
        for row, count in zip(positions, counts, strict=True):
            # Trimmed to the fresh rows: the final minibatch wraps onto the
            # epoch's leading states, and an explained variance must score every
            # held-out state exactly once.
            batch = torch.from_numpy(indices[row[: int(count)]]).to(device=device)
            critic_args = _critic_batch_args(architecture, staged, batch)
            value_targets = _batch_tensor(staged["value_targets"], batch, torch.float32)
            value_loss, moments = _critic_minibatch_fit_terms(
                critic,
                value_targets,
                autocast_enabled,
                *critic_args,
                entity_active=(
                    _entity_active_batch(staged, batch)
                    if getattr(critic.config, "per_entity_critic", False)
                    else None
                ),
            )
            sums += moments
            loss_total += value_loss.double() * batch.numel()
    finally:
        critic.train(was_training)
    return {
        "explained_variance": _fit_explained_variance(_fit_moment_mapping(sums), states),
        "loss": float(loss_total) / max(1, states),
        "states": states,
    }


def _collect(
    actor: torch.nn.Module,
    args: argparse.Namespace,
    config: PpoConfig,
    seed_start: int,
    arena: dict[str, np.ndarray],
) -> RolloutBatch:
    return collect_self_play_rust(
        actor,
        games=args.games,
        seed_start=seed_start,
        episode_steps=args.episode_steps,
        gamma=config.gamma,
        reward_mode=args.reward_mode,
        sampling_seed=seed_start ^ 0x5EED,
        forward_mode=args.rollout_forward_mode,
        forward_autocast=not args.no_rollout_bfloat16,
        storage=arena,
    )


def _warm_critic(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    critic_optimizer: torch.optim.Optimizer,
    architecture: str,
    args: argparse.Namespace,
    config: PpoConfig,
    *,
    repeat_seed: int,
    arena: dict[str, np.ndarray],
    compile_mode: str,
    autocast_enabled: bool,
    generator: np.random.Generator,
    device: torch.device,
) -> list[dict[str, float]]:
    """Fit the critic on disjoint waves before the measured one.

    Each warmup iteration is a whole production critic refit on its own fresh
    rollout: replay behavior values, build targets, fit `--warmup-epochs` epochs
    over every valid state. Nothing from these waves is measured or held out --
    their only purpose is to move the critic off initialization, so the measured
    curve can be read at a warmer operating point than a from-scratch one.
    """
    records: list[dict[str, float]] = []
    for iteration in range(args.warmup_iterations):
        rollout = _collect(actor, args, config, repeat_seed + 1024 * (iteration + 1), arena)
        staged = _stage_rollout(rollout, device)
        target_stats = _stage_value_targets(
            rollout,
            critic,
            architecture,
            staged,
            config,
            compile_mode=compile_mode,
            autocast_enabled=autocast_enabled,
            device=device,
        )
        indices = np.flatnonzero(rollout.valid.reshape(-1))
        for _ in range(args.warmup_epochs):
            _critic_epoch(
                critic,
                critic_optimizer,
                architecture,
                staged,
                indices,
                config,
                compile_mode=compile_mode,
                autocast_enabled=autocast_enabled,
                generator=generator,
                device=device,
            )
        scored = _evaluate(
            critic,
            architecture,
            staged,
            indices,
            config,
            autocast_enabled=autocast_enabled,
            device=device,
        )
        records.append({"iteration": iteration, **target_stats, **scored})
        del staged
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def _build_critic(
    actor: torch.nn.Module,
    architecture: str,
    model_config: dict[str, Any],
    config: PpoConfig,
    checkpoint_state: dict[str, Any] | None,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Construct one isolated critic/optimizer pair, optionally from a checkpoint."""
    critic = resolve_architecture(architecture).build_critic(model_config).to(device)
    _, critic_optimizer = make_optimizers(actor, critic, config)
    if checkpoint_state is not None:
        critic.load_state_dict(checkpoint_state["critic"], strict=True)
        # Optimizer.load_state_dict may retain same-device tensor objects. A
        # private copy keeps one repeat's in-place moments from changing the
        # immutable starting point used by every later repeat.
        critic_optimizer.load_state_dict(copy.deepcopy(checkpoint_state["critic_optimizer"]))
    return critic, critic_optimizer


def _run_repeat(
    actor: torch.nn.Module,
    architecture: str,
    model_config: dict[str, Any],
    args: argparse.Namespace,
    config: PpoConfig,
    checkpoint_state: dict[str, Any] | None,
    *,
    repeat: int,
    arena: dict[str, np.ndarray],
    compile_mode: str,
    autocast_enabled: bool,
    device: torch.device,
) -> dict[str, Any]:
    """One independent rollout curve from an identical critic starting state."""
    repeat_seed = args.base_seed + repeat * 65536
    # A fresh Dynamo state per repeat. The update path compiles with
    # dynamic=False and guards on the critic instance, so without this the
    # entries from earlier repeats accumulate against the per-code-object limit.
    torch._dynamo.reset()
    torch.manual_seed(repeat_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(repeat_seed)
        torch.cuda.empty_cache()
    generator = np.random.default_rng(repeat_seed ^ 0xC0FFEE)
    critic, critic_optimizer = _build_critic(
        actor, architecture, model_config, config, checkpoint_state, device
    )
    warmup = _warm_critic(
        actor,
        critic,
        critic_optimizer,
        architecture,
        args,
        config,
        repeat_seed=repeat_seed,
        arena=arena,
        compile_mode=compile_mode,
        autocast_enabled=autocast_enabled,
        generator=generator,
        device=device,
    )

    rollout = _collect(actor, args, config, repeat_seed, arena)
    staged = _stage_rollout(rollout, device)
    target_stats = _stage_value_targets(
        rollout,
        critic,
        architecture,
        staged,
        config,
        compile_mode=compile_mode,
        autocast_enabled=autocast_enabled,
        device=device,
    )
    fit_indices, holdout_indices, split = _split_games(rollout, args.holdout_fraction, generator)

    def scored(epoch: int, seconds: float | None, minibatches: int) -> dict[str, Any]:
        fit = _evaluate(
            critic,
            architecture,
            staged,
            fit_indices,
            config,
            autocast_enabled=autocast_enabled,
            device=device,
        )
        holdout = _evaluate(
            critic,
            architecture,
            staged,
            holdout_indices,
            config,
            autocast_enabled=autocast_enabled,
            device=device,
        )
        return {
            "epoch": epoch,
            "fit_explained_variance": fit["explained_variance"],
            "holdout_explained_variance": holdout["explained_variance"],
            "fit_loss": fit["loss"],
            "holdout_loss": holdout["loss"],
            "memorization_gap": fit["explained_variance"] - holdout["explained_variance"],
            "seconds": seconds,
            "minibatches": minibatches,
            # The first epoch of a repeat pays Inductor compilation, which is a
            # one-off per process rather than a per-epoch cost of the schedule.
            "includes_compilation": epoch == 1,
        }

    # Epoch zero is the critic before any gradient step of this wave, so epoch
    # one's marginal gain is defined on the same footing as every later one.
    epochs = [scored(0, None, 0)]
    for epoch in range(1, args.critic_epochs + 1):
        seconds, minibatches = _critic_epoch(
            critic,
            critic_optimizer,
            architecture,
            staged,
            fit_indices,
            config,
            compile_mode=compile_mode,
            autocast_enabled=autocast_enabled,
            generator=generator,
            device=device,
        )
        epochs.append(scored(epoch, seconds, minibatches))
        print(f"repeat {repeat} {json.dumps(epochs[-1], sort_keys=True)}", flush=True)

    record = {
        "repeat": repeat,
        "seed": repeat_seed,
        "rollout_seconds": rollout.elapsed_seconds,
        "rollout_valid_states": int(rollout.state_count),
        "split": split,
        "targets": target_stats,
        "warmup": warmup,
        "epochs": epochs,
    }
    # The repeat's staged rollout, critic and optimizer state are released with
    # this frame; the next repeat empties the allocator cache before it collects.
    return record


def _per_epoch_table(records: list[dict[str, Any]], epochs: int) -> list[dict[str, Any]]:
    """Aggregate the repeats into one row per epoch index, with marginal gains."""
    table: list[dict[str, Any]] = []
    for epoch in range(epochs + 1):
        holdout = [record["epochs"][epoch]["holdout_explained_variance"] for record in records]
        fit = [record["epochs"][epoch]["fit_explained_variance"] for record in records]
        holdout_loss = [record["epochs"][epoch]["holdout_loss"] for record in records]
        fit_loss = [record["epochs"][epoch]["fit_loss"] for record in records]
        row: dict[str, Any] = {
            "epoch": epoch,
            "holdout_explained_variance_mean": statistics.fmean(holdout),
            "holdout_explained_variance_spread": _spread(holdout),
            "holdout_explained_variance": holdout,
            "fit_explained_variance_mean": statistics.fmean(fit),
            "fit_explained_variance_spread": _spread(fit),
            "fit_explained_variance": fit,
            "holdout_loss_mean": statistics.fmean(holdout_loss),
            "fit_loss_mean": statistics.fmean(fit_loss),
            "memorization_gap_mean": statistics.fmean(fit) - statistics.fmean(holdout),
        }
        if epoch:
            seconds = [record["epochs"][epoch]["seconds"] for record in records]
            minibatches = records[0]["epochs"][epoch]["minibatches"]
            # Paired per repeat: each repeat is its own curve, so the gain's
            # noise is the spread of the differences, not of the levels.
            gains = [
                record["epochs"][epoch]["holdout_explained_variance"]
                - record["epochs"][epoch - 1]["holdout_explained_variance"]
                for record in records
            ]
            gain_mean = statistics.fmean(gains)
            gain_spread = _spread(gains)
            row |= {
                "seconds_mean": statistics.fmean(seconds),
                "seconds_median": statistics.median(seconds),
                "seconds": seconds,
                "minibatches": minibatches,
                "seconds_per_minibatch_median": statistics.median(seconds) / max(1, minibatches),
                "marginal_holdout_explained_variance_mean": gain_mean,
                "marginal_holdout_explained_variance_spread": gain_spread,
                "marginal_holdout_explained_variance": gains,
                "marginal_within_noise": abs(gain_mean) <= gain_spread,
                "includes_compilation": records[0]["epochs"][epoch]["includes_compilation"],
            }
        table.append(row)
    return table


def _decision(
    records: list[dict[str, Any]],
    table: list[dict[str, Any]],
    epochs: int,
    production_minibatches: int,
) -> dict[str, Any]:
    """Smallest epoch count that leaves no detectable holdout gain behind it.

    The plateau is decided on the gain REMAINING after a candidate count -- the
    holdout explained variance at the last measured epoch minus the candidate's,
    paired per repeat -- rather than on one epoch's marginal gain at a time. A
    per-epoch test is not usable as the decision rule at this repeat count: a
    gain that is pure noise exceeds its own three-sample standard deviation
    something like a third of the time, so a single fluke late in the curve would
    veto a plateau the rest of the curve supports. The remaining gain is also
    exactly the quantity the decision spends wall clock on, and it is the
    stricter test of the two, since a run of small real gains sums into it while
    each one alone hides under the noise. Per-epoch marginal gains and their own
    noise flags stay in `per_epoch`, which is where the curve's shape is read.

    Per-epoch seconds are rescaled to the production schedule's minibatch count,
    because this probe fits self-play states minus a holdout while production
    fits a league-mixed wave whole. Steady-state per-minibatch cost is taken as
    the median over epochs that did not pay compilation.
    """
    remaining: list[dict[str, Any]] = []
    for candidate in range(epochs + 1):
        gains = [
            record["epochs"][epochs]["holdout_explained_variance"]
            - record["epochs"][candidate]["holdout_explained_variance"]
            for record in records
        ]
        mean = statistics.fmean(gains)
        spread = _spread(gains)
        remaining.append(
            {
                "epochs": candidate,
                "holdout_gain_mean": mean,
                "holdout_gain_spread": spread,
                "holdout_gain": gains,
                "within_noise": abs(mean) <= spread,
            }
        )
    # The last candidate is trivially within noise (its gain is identically
    # zero), so this scan always terminates at or before the measured epochs.
    plateau = next(row["epochs"] for row in remaining if row["within_noise"])
    steady = statistics.median(
        [
            row["seconds_per_minibatch_median"]
            for row in table[1:]
            if not row["includes_compilation"]
        ]
        or [table[1]["seconds_per_minibatch_median"]]
    )
    production_epoch_seconds = steady * production_minibatches
    gain_mean = remaining[plateau]["holdout_gain_mean"]
    gain_spread = remaining[plateau]["holdout_gain_spread"]
    cost = production_epoch_seconds * (epochs - plateau)
    if plateau == epochs:
        # Every epoch measured still earned its keep, so the curve's plateau --
        # if it has one -- is outside the range this run can speak about.
        headline = (
            f"holdout explained variance had not plateaued within {epochs} epochs; "
            f"the last epoch still gained "
            f"{table[epochs]['marginal_holdout_explained_variance_mean']:+.4f} +/- "
            f"{table[epochs]['marginal_holdout_explained_variance_spread']:.4f} "
            f"for {production_epoch_seconds:.1f} s/iteration"
        )
    else:
        headline = (
            f"holdout explained variance plateaus after {plateau} epochs; "
            f"epochs {plateau + 1}..{epochs} cost {cost:.1f} s/iteration and gain "
            f"{gain_mean:+.4f} +/- {gain_spread:.4f}"
        )
    return {
        "rule": (
            "the plateau is the smallest epoch count N whose REMAINING holdout explained "
            "variance gain -- epoch N to the last measured epoch, differenced per repeat -- "
            "satisfies |mean across repeats| <= the across-repeat sample standard deviation "
            "of those same paired differences; per-epoch marginal gains carry the same "
            "one-sigma flag in per_epoch[].marginal_within_noise"
        ),
        "plateau_epochs": plateau,
        "remaining_gain_scan": remaining,
        "production_seconds_per_critic_epoch": production_epoch_seconds,
        "production_critic_minibatches_per_epoch": production_minibatches,
        "steady_seconds_per_minibatch": steady,
        "epochs_beyond_plateau": epochs - plateau,
        "seconds_beyond_plateau": cost,
        "holdout_gain_beyond_plateau_mean": gain_mean,
        "holdout_gain_beyond_plateau_spread": gain_spread,
        "headline": headline,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--actor", type=Path)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="single-learner checkpoint whose critic and optimizer start every repeat",
    )
    parser.add_argument("--games", type=int, default=PRODUCTION_SELF_PLAY_GAMES)
    parser.add_argument("--episode-steps", type=int, default=PRODUCTION_EPISODE_STEPS)
    parser.add_argument("--critic-epochs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="defaults to the checkpoint schedule, or 2048 with --actor",
    )
    parser.add_argument(
        "--update-compile-mode",
        choices=UPDATE_COMPILE_MODES,
        default=PRODUCTION_UPDATE_COMPILE_MODE,
    )
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument(
        "--rollout-forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        default=PRODUCTION_ROLLOUT_FORWARD_MODE,
    )
    parser.add_argument(
        "--no-rollout-bfloat16", action="store_true", default=not PRODUCTION_ROLLOUT_BFLOAT16
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=0,
        help="disjoint waves fitted before the measured one; incompatible with --checkpoint",
    )
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument(
        "--production-minibatches-per-epoch",
        type=int,
        default=None,
        help="critic minibatches one production epoch runs; derived from the shipped wave",
    )
    parser.add_argument("--base-seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    """Load and validate the single-learner state required by checkpoint mode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload.get("agents"), list):
        raise ValueError("--checkpoint requires a single-learner checkpoint, not a population")
    missing = [key for key in ("critic", "critic_optimizer", "ppo_config") if key not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing required state: {', '.join(missing)}")
    training_data = payload.get("training_data_config")
    if not isinstance(training_data, dict) or training_data.get("reward_mode") not in REWARD_MODES:
        raise ValueError("checkpoint is missing a valid recorded reward mode")
    return payload


def main() -> None:
    args = parse_args()
    if args.critic_epochs < 2:
        raise SystemExit("a marginal per-epoch gain needs at least two epochs")
    if args.repeats < 2:
        raise SystemExit("one curve cannot separate a plateau from noise; use --repeats 2 or more")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must lie strictly between zero and one")
    if args.warmup_iterations < 0 or args.warmup_epochs < 1:
        raise SystemExit("warmup iterations cannot be negative and warmup epochs must be positive")
    if args.checkpoint is not None and args.warmup_iterations:
        raise SystemExit(
            "--checkpoint already supplies warm state and cannot use --warmup-iterations"
        )

    checkpoint_state = (
        _load_checkpoint_state(args.checkpoint) if args.checkpoint is not None else None
    )
    args.reward_mode = (
        checkpoint_state["training_data_config"]["reward_mode"]
        if checkpoint_state is not None
        else DEFAULT_REWARD_MODE
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("this probe measures the CUDA update path and needs a GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    source_path = args.checkpoint if args.checkpoint is not None else args.actor
    assert source_path is not None
    actor, payload = load_actor_artifact(source_path, device=device)
    actor.eval()
    architecture = resolve_architecture(payload)
    schedule = dict(checkpoint_state["ppo_config"]) if checkpoint_state is not None else {}
    minibatch_size = args.minibatch_size or int(schedule.get("minibatch_size", 2048))
    schedule.update(
        epochs=1,
        critic_epochs=args.critic_epochs,
        minibatch_size=minibatch_size,
        use_bfloat16=not args.no_bfloat16,
        update_compile_mode=args.update_compile_mode,
    )
    config = PpoConfig(**schedule)
    autocast_enabled = config.use_bfloat16 and device.type == "cuda"
    compile_mode = _device_compile_mode(config.update_compile_mode, device)
    arena = allocate_rollout_storage(
        architecture.name,
        trajectories=args.games * 2,
        horizon=args.episode_steps - 1,
        pin_memory=True,
    )

    records = [
        _run_repeat(
            actor,
            architecture.name,
            payload["model_config"],
            args,
            config,
            checkpoint_state,
            repeat=repeat,
            arena=arena,
            compile_mode=compile_mode,
            autocast_enabled=autocast_enabled,
            device=device,
        )
        for repeat in range(args.repeats)
    ]
    production_minibatches = (
        args.production_minibatches_per_epoch
        or _production_critic_minibatches_per_epoch(config.minibatch_size)
    )
    table = _per_epoch_table(records, args.critic_epochs)
    decision = _decision(records, table, args.critic_epochs, production_minibatches)
    checkpoint_mode = checkpoint_state is not None
    caveats = [
        "One operating point: the loaded actor's rollout distribution at temperature 1.0.",
        "Self-play collection only, not the production league-mixed wave.",
        "A value-fit curve is not end-to-end policy improvement; fewer critic epochs could "
        "still change learning through the advantages of later iterations.",
        "Per-epoch seconds are measured on this probe's smaller fit set and rescaled to the "
        "production minibatch count; the first epoch of each repeat also pays compilation.",
    ]
    if not checkpoint_mode:
        caveats.insert(
            0,
            "The critic is fresh per repeat, not a warm mid-run critic; this favours later "
            "epochs looking useful.",
        )
    report = {
        "actor": str(source_path),
        "checkpoint": str(args.checkpoint) if checkpoint_mode else None,
        "checkpoint_iteration": (
            int(checkpoint_state.get("iteration", 0)) if checkpoint_state is not None else None
        ),
        "architecture": architecture.name,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "source_identity": source_identity()["sha256"],
        "configuration": {
            "games": args.games,
            "episode_steps": args.episode_steps,
            "reward_mode": args.reward_mode,
            "gamma": config.gamma,
            "critic_epochs": args.critic_epochs,
            "repeats": args.repeats,
            "holdout_fraction": args.holdout_fraction,
            "holdout_split_by": "game (episode_seeds), never by state",
            "minibatch_size": config.minibatch_size,
            "update_compile_mode": config.update_compile_mode,
            "use_bfloat16": config.use_bfloat16,
            "rollout_forward_mode": args.rollout_forward_mode,
            "rollout_bfloat16": not args.no_rollout_bfloat16,
            "warmup_iterations": args.warmup_iterations,
            "warmup_epochs": args.warmup_epochs,
            "base_seed": args.base_seed,
            "critic_initialization": (
                "checkpoint weights and optimizer restored identically per repeat"
                if checkpoint_mode
                else "fresh per repeat (zero-initialized value head)"
            ),
            "critic_optimizer_restored": checkpoint_mode,
        },
        "caveats": caveats,
        "repeats": records,
        "per_epoch": table,
        "decision": decision,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
