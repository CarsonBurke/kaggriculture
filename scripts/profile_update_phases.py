#!/usr/bin/env python3
"""Attribute PPO update time to phases and, at production shape, to kernels.

Two measurements from one collected production wave:

1. Whole-update phase wall times (`update_*_seconds`) from two `update_ppo`
   calls: the first pays compilation, the second is steady state.
2. Isolated sections at exact production minibatch shape -- gathers, compiled
   actor and critic forward/backward, gradient norms, optimizer steps, value
   replay, and the joint NextLat trunk forward plus predictor step -- each
   timed with CUDA events and, with `--kernels`, profiled to a per-kernel table.

Kernel attribution is confined to one section at a time, so the profiler never
sees more than a few thousand events.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from kaggriculture.inference import load_actor_artifact
from kaggriculture.model import parameter_count
from kaggriculture.ppo import (
    PpoConfig,
    _actor_batch_args,
    _actor_minibatch_terms,
    _batch_tensor,
    _cached_update_callable,
    _critic_batch_args,
    _critic_minibatch_fit_terms,
    _optimizer_step,
    _replayed_value_chunk,
    _stage_tensor,
    _structured_auxiliary_terms,
    _structured_critic_auxiliary_terms,
    make_optimizers,
    make_structured_dynamics_optimizer,
    update_ppo,
)
from kaggriculture.production import (
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_TEMPERATURE,
    production_model_config,
    production_ppo_config,
)
from kaggriculture.provenance import source_identity
from kaggriculture.registry import STRUCTURED, resolve_architecture
from kaggriculture.rollout import allocate_rollout_storage, collect_mixed_play_rust
from kaggriculture.structured import StructuredConfig
from kaggriculture.structured_dynamics import StructuredCriticDynamics, StructuredDynamics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=PRODUCTION_SELF_PLAY_GAMES)
    parser.add_argument("--league-games", type=int, default=PRODUCTION_LEAGUE_GAMES)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--init-actor-from", type=Path, default=None)
    parser.add_argument("--update-compile-mode", default="default")
    parser.add_argument("--updates", type=int, default=2, help="update_ppo calls to time")
    parser.add_argument("--kernels", action="store_true", help="emit per-section kernel tables")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _synchronize() -> None:
    torch.cuda.synchronize()


def measure(fn: Callable[[], Any], *, warmup: int = 2, repeats: int = 5) -> dict[str, float]:
    """Median CUDA-event and wall milliseconds of `fn` after warmup."""
    for _ in range(warmup):
        fn()
    _synchronize()
    device_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _synchronize()
        wall_started = time.perf_counter()
        start.record()
        fn()
        end.record()
        _synchronize()
        wall_ms.append((time.perf_counter() - wall_started) * 1000.0)
        device_ms.append(start.elapsed_time(end))
    return {"device_ms": statistics.median(device_ms), "wall_ms": statistics.median(wall_ms)}


def kernel_table(fn: Callable[[], Any], top: int) -> dict[str, Any]:
    """Top kernels by device time for one execution of `fn`."""
    fn()
    _synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        fn()
        _synchronize()
    events = [
        event
        for event in prof.key_averages()
        if event.self_device_time_total > 0 and event.device_type.name != "CPU"
    ]
    events.sort(key=lambda event: event.self_device_time_total, reverse=True)
    total = sum(event.self_device_time_total for event in events)
    launches = sum(event.count for event in events)
    return {
        "device_total_ms": total / 1000.0,
        "kernel_launches": launches,
        "kernels": [
            {
                "name": event.key[:120],
                "count": event.count,
                "total_ms": event.self_device_time_total / 1000.0,
                "share": event.self_device_time_total / max(total, 1.0),
            }
            for event in events[:top]
        ],
    }


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    architecture = resolve_architecture(STRUCTURED)
    model_config = StructuredConfig(**production_model_config())
    ppo_config = PpoConfig(
        **cast(dict[str, Any], production_ppo_config(update_compile_mode=args.update_compile_mode))
    )
    if args.init_actor_from is None:
        actor = architecture.actor_class(model_config).to(device)
    else:
        actor, _ = load_actor_artifact(args.init_actor_from, device=device)
    critic = architecture.critic_class(model_config).to(device)
    dynamics = StructuredDynamics(model_config).to(device)
    critic_dynamics = StructuredCriticDynamics(model_config).to(device)
    frozen_state = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, ppo_config)
    dynamics_optimizer = make_structured_dynamics_optimizer(dynamics, ppo_config)
    critic_dynamics_optimizer = make_structured_dynamics_optimizer(critic_dynamics, ppo_config)
    generator = np.random.default_rng(args.seed)
    auxiliary_generator = np.random.default_rng(args.seed + 2)

    opponents = [
        architecture.actor_class(model_config).to(device)
        for _ in range(PRODUCTION_LEAGUE_ACTIVE_OPPONENTS + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS)
    ]
    for opponent in opponents:
        opponent.load_state_dict(frozen_state)
        opponent.requires_grad_(False)
    assignments = np.arange(args.league_games, dtype=np.int64) % len(opponents)
    generator.shuffle(assignments)
    arena = allocate_rollout_storage(
        architecture.name,
        args.games * 2 + args.league_games,
        PRODUCTION_EPISODE_STEPS - 1,
        pin_memory=True,
    )
    _synchronize()
    rollout_started = time.perf_counter()
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=args.games,
        league_games=args.league_games,
        opponent_indices=assignments,
        seed_start=args.seed,
        episode_steps=PRODUCTION_EPISODE_STEPS,
        temperature=PRODUCTION_TEMPERATURE,
        opponent_temperature=PRODUCTION_TEMPERATURE,
        sampling_seed=int(generator.integers(0, np.iinfo(np.int64).max)),
        forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
        forward_autocast=True,
        storage=arena,
    )
    _synchronize()
    report: dict[str, Any] = {
        "source_digest": source_identity()["sha256"],
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "games": args.games,
        "league_games": args.league_games,
        "learner_states": rollout.state_count,
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "rollout_seconds": time.perf_counter() - rollout_started,
        "updates": [],
    }
    del opponents

    phase_keys = (
        "update_staging_seconds",
        "update_behavior_replay_seconds",
        "update_advantage_seconds",
        "update_minibatch_seconds",
        "update_finalize_seconds",
        "actor_updates",
        "updates",
    )
    for index in range(args.updates):
        torch.cuda.reset_peak_memory_stats(device)
        _synchronize()
        started = time.perf_counter()
        metrics = update_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            ppo_config,
            generator=generator,
            structured_dynamics=dynamics,
            structured_dynamics_optimizer=dynamics_optimizer,
            structured_critic_dynamics=critic_dynamics,
            structured_critic_dynamics_optimizer=critic_dynamics_optimizer,
            auxiliary_generator=auxiliary_generator,
        )
        _synchronize()
        record: dict[str, Any] = {key: metrics[key] for key in phase_keys if key in metrics}
        record["update_seconds"] = time.perf_counter() - started
        record["peak_cuda_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        record["phase"] = "cold" if index == 0 else "steady"
        report["updates"].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    # Isolated sections at exact production minibatch shape.
    staged = {name: _stage_tensor(array, device) for name, array in rollout.states.items()}
    staged |= {
        name: _stage_tensor(getattr(rollout, name), device)
        for name in (
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "unit_masks",
            "market_kind_masks",
            "market_quantity_masks",
            "unit_active",
            "market_active",
            "market_quantity_active",
            "old_unit_logprobs",
            "old_market_kind_logprobs",
            "old_market_quantity_logprobs",
        )
    }
    flat_valid = rollout.valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    minibatch = valid_indices.size // -(-valid_indices.size // ppo_config.minibatch_size)
    host_indices = generator.permutation(valid_indices)[:minibatch]
    indices = torch.from_numpy(host_indices).to(device)
    staged["advantages"] = torch.zeros(flat_valid.size, device=device)
    staged["value_targets"] = torch.zeros(flat_valid.size, device=device)
    autocast_enabled = ppo_config.use_bfloat16
    mode = args.update_compile_mode
    actor_terms = _cached_update_callable(
        actor, "_kaggriculture_update_terms", _actor_minibatch_terms, mode
    )
    critic_terms = _cached_update_callable(
        critic, "_kaggriculture_update_fit_terms", _critic_minibatch_fit_terms, mode
    )
    replay = _cached_update_callable(
        critic, "_kaggriculture_value_replay", _replayed_value_chunk, mode
    )
    actor.eval()
    critic.train()

    def gather_actor():
        return _actor_batch_args(STRUCTURED, staged, indices)

    def gather_critic():
        return _critic_batch_args(STRUCTURED, staged, indices, actor_args=gather_actor())

    def gather_surrogate():
        return (
            _batch_tensor(staged["unit_actions"], indices, torch.long),
            _batch_tensor(staged["market_kinds"], indices, torch.long),
            _batch_tensor(staged["market_quantities"], indices, torch.long),
            _batch_tensor(staged["unit_masks"], indices, torch.bool),
            _batch_tensor(staged["market_kind_masks"], indices, torch.bool),
            _batch_tensor(staged["market_quantity_masks"], indices, torch.bool),
            _batch_tensor(staged["unit_active"], indices, torch.float32),
            _batch_tensor(staged["market_active"], indices, torch.float32),
            _batch_tensor(staged["market_quantity_active"], indices, torch.float32),
            _batch_tensor(staged["old_unit_logprobs"], indices, torch.float32),
            _batch_tensor(staged["old_market_kind_logprobs"], indices, torch.float32),
            _batch_tensor(staged["old_market_quantity_logprobs"], indices, torch.float32),
            _batch_tensor(staged["advantages"], indices, torch.float32),
        )

    actor_args = gather_actor()
    critic_args = gather_critic()
    surrogate = gather_surrogate()
    component_count = max(1, int(sum(component.sum() for component in surrogate[6:9])))
    value_targets = _batch_tensor(staged["value_targets"], indices, torch.float32)

    def actor_forward():
        return actor_terms(
            actor,
            *surrogate,
            ppo_config.clip_low,
            ppo_config.clip_high,
            autocast_enabled,
            *actor_args,
        )

    def actor_forward_backward():
        actor_optimizer.zero_grad(set_to_none=True)
        policy_sum, _entropy, _kl, _clipped = actor_forward()
        if not policy_sum.requires_grad:
            raise RuntimeError(
                "actor objective lost its graph: "
                f"grad_enabled={torch.is_grad_enabled()} "
                f"params_requiring_grad={sum(p.requires_grad for p in actor.parameters())} "
                f"inference={policy_sum.is_inference()} training={actor.training}"
            )
        (-policy_sum / component_count).backward()

    def actor_norm():
        return torch.nn.utils.get_total_norm(
            [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
        )

    def actor_step():
        _optimizer_step(actor_optimizer, ppo_config.actor_learning_rate, ppo_config.lr_warmup_steps)

    def critic_forward():
        return critic_terms(critic, value_targets, autocast_enabled, *critic_args)

    def critic_forward_backward():
        critic_optimizer.zero_grad(set_to_none=True)
        loss, _moments = critic_forward()
        loss.backward()

    def critic_norm():
        return torch.nn.utils.get_total_norm(
            [parameter.grad for parameter in critic.parameters() if parameter.grad is not None]
        )

    def critic_step():
        _optimizer_step(
            critic_optimizer, ppo_config.critic_learning_rate, ppo_config.lr_warmup_steps
        )

    replay_args = _critic_batch_args(STRUCTURED, staged, slice(0, 4096))

    @torch.inference_mode()
    def replay_chunk():
        was_training = critic.training
        critic.eval()
        try:
            return replay(critic, autocast_enabled, *replay_args)
        finally:
            critic.train(was_training)

    # Joint NextLat: the same PPO minibatch, one trunk pass, then p_ψ.
    batch_indices = indices
    report["predictor_batch"] = {"rows": int(host_indices.size)}
    (belief_inputs,) = actor_args
    critic_belief_args = critic_args

    @torch.no_grad()
    def actor_source_forward_eager():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            return actor.forward_with_belief(belief_inputs)

    @torch.no_grad()
    def critic_source_forward_eager():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            return critic.forward_with_belief(*critic_belief_args)

    # The predictor sections pass `model_grad=False`, and that path takes its
    # source belief under `torch.no_grad()` (`ppo.py:2574-2576`), so no
    # gradient reaches the actor or critic no matter what `requires_grad` says.
    # An earlier version froze both models here instead, while the sections were
    # still being defined -- before any of them ran -- which left
    # `actor_forward_backward` with a graphless surrogate and killed the
    # profiler at its fifth section.

    dynamics.train()
    critic_dynamics.train()

    def actor_predictor_batch():
        dynamics_optimizer.zero_grad(set_to_none=True)
        loss, _terms = _structured_auxiliary_terms(
            actor,
            dynamics,
            staged,
            batch_indices,
            steps_per_trajectory=rollout.valid.shape[1],
            config=ppo_config,
            autocast_enabled=autocast_enabled,
            model_grad=False,
            complete_windows=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), ppo_config.nextlat_max_gradient_norm)
        _optimizer_step(
            dynamics_optimizer,
            ppo_config.resolved_structured_learning_rate,
            ppo_config.lr_warmup_steps,
        )

    def critic_predictor_batch():
        critic_dynamics_optimizer.zero_grad(set_to_none=True)
        loss, _terms = _structured_critic_auxiliary_terms(
            critic,
            critic_dynamics,
            staged,
            batch_indices,
            steps_per_trajectory=rollout.valid.shape[1],
            config=ppo_config,
            autocast_enabled=autocast_enabled,
            model_grad=False,
            complete_windows=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            critic_dynamics.parameters(), ppo_config.nextlat_max_gradient_norm
        )
        _optimizer_step(
            critic_dynamics_optimizer,
            ppo_config.resolved_structured_critic_learning_rate,
            ppo_config.lr_warmup_steps,
        )

    sections: dict[str, Callable[[], Any]] = {
        "gather_actor": gather_actor,
        "gather_critic_extra": gather_critic,
        "gather_surrogate": gather_surrogate,
        "actor_forward": actor_forward,
        "actor_forward_backward": actor_forward_backward,
        "actor_gradient_norm": actor_norm,
        "actor_optimizer_step": actor_step,
        "critic_forward": critic_forward,
        "critic_forward_backward": critic_forward_backward,
        "critic_gradient_norm": critic_norm,
        "critic_optimizer_step": critic_step,
        "value_replay_chunk_4096": replay_chunk,
        "actor_source_forward_eager": actor_source_forward_eager,
        "critic_source_forward_eager": critic_source_forward_eager,
        "actor_predictor_batch": actor_predictor_batch,
        "critic_predictor_batch": critic_predictor_batch,
    }
    # The optimizer sections need populated gradients; forward/backward sections
    # leave them in place, and the steps below only move weights negligibly.
    # Sections are ordered so gradients exist before the norm and step sections.
    results: dict[str, Any] = {}
    for name, fn in sections.items():
        timing = measure(fn)
        results[name] = timing
        print(json.dumps({"section": name, **timing}), flush=True)
    if args.kernels:
        for name in (
            "actor_forward_backward",
            "critic_forward_backward",
            "actor_optimizer_step",
            "critic_optimizer_step",
            "value_replay_chunk_4096",
            "actor_source_forward_eager",
            "critic_source_forward_eager",
            "actor_predictor_batch",
            "critic_predictor_batch",
        ):
            table = kernel_table(sections[name], args.top)
            results[name]["kernels"] = table
            print(json.dumps({"section": name, "kernel_table": table}), flush=True)
    report["sections"] = results
    report["minibatch_rows"] = minibatch
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"event": "done", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
