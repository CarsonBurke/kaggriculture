#!/usr/bin/env python3
"""Historical NextLat architecture benchmark; run ONLY inside an MLQ allocation.

The fixed-state experiment includes a real 719-step production mixed rollout,
then compares identical 6400-row PPO inputs. It is NOT an iteration-throughput
claim: the report also emits the two six-repeat whole-iteration commands needed
to measure collection, optimizer, replay, and league overhead together.
Critic NextLat remains explicitly enabled for this experiment; this is not the
current production auxiliary recipe.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from kaggriculture.entity import EntityConfig
from kaggriculture.inference import load_actor_artifact
from kaggriculture.model import parameter_count
from kaggriculture.modelargs import model_config_arguments
from kaggriculture.policy import mask_logits
from kaggriculture.ppo import (
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    UPDATE_REPLAY_TAIL_LOGPROB,
    PpoConfig,
    _actor_batch_args,
    _actor_minibatch_terms,
    _critic_batch_args,
    _replayed_component_logprobs,
    _structured_critic_auxiliary_terms,
    _structured_critic_minibatch_fit_terms,
)
from kaggriculture.production import (
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_TEMPERATURE,
    production_ppo_config,
)
from kaggriculture.provenance import file_sha256, source_identity
from kaggriculture.registry import (
    ENTITY_ATTENTION,
    STRUCTURED,
    pair_towers,
    resolve_architecture,
)
from kaggriculture.rollout import collect_mixed_play_rust
from kaggriculture.structured import StructuredConfig, StructuredCriticBelief
from kaggriculture.structured_dynamics import StructuredCriticDynamics, structured_horizon_plan

FACTOR_NAMES = (
    "unit_actions",
    "market_kinds",
    "market_quantities",
    "unit_masks",
    "market_kind_masks",
    "market_quantity_masks",
    "unit_active",
    "market_active",
    "market_quantity_active",
)


def historical_ppo_config() -> PpoConfig:
    """Retain the critic-NextLat workload after production disables it.

    The rest of the schedule is production's current one for entity-attention.
    """
    return PpoConfig(
        **{
            **production_ppo_config(update_compile_mode="default", architecture=ENTITY_ATTENTION),
            "structured_latent_coefficient": 0.0,
            "structured_decision_coefficient": 0.0,
            "structured_critic_latent_coefficient": 1.0,
            "structured_critic_value_coefficient": 1.0,
            "structured_critic_horizon": 1,
            "structured_critic_gradient_balance": False,
        }
    )


def legacy_config() -> StructuredConfig:
    """Freeze the actual previous production family, not changing defaults."""
    return StructuredConfig(
        observation_schema_version=3,
        model_dim=80,
        attention_heads=4,
        attention_kv_heads=2,
        ffn_multiplier=2,
        farm_blocks=2,
        opponent_latents=8,
        latents=32,
        core_layers=8,
        quantity_rank=32,
        global_refresh_layers=(),
        global_refresh_context="none",
        input_reinject_layers=(1, 2, 3, 4, 5, 6, 7, 8),
        core_skip_source=3,
        core_skip_target=6,
        zero_init_branches=False,
        mudd_lite=True,
        fuse_market_decoder=True,
        fuse_unit_decoder=False,
        split_clock_token=False,
        global_modulation=True,
        fused_mlp=False,
        critic_core_layers=0,
        critic_latents=0,
        critic_state_read=False,
        per_entity_critic=False,
        actor_opponent_farm=True,
        value_atoms=255,
        value_min=-2.2,
        value_max=2.2,
        value_sigma_ratio=3.0,
        scalar_value=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity-actor", type=Path)
    parser.add_argument("--legacy-actor", type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--core-layers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--deadline-seconds", type=int, default=1440)
    parser.add_argument("--commands-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1 or args.deadline_seconds <= 0:
        parser.error("need >=3 repeats, >=1 warmup, and a positive deadline")
    return args


def iteration_command(family, config, artifact, output, seed):
    command = [
        sys.executable,
        "scripts/benchmark_ppo_iteration.py",
        "--architecture",
        family,
        "--games",
        str(PRODUCTION_SELF_PLAY_GAMES),
        "--league-games",
        str(PRODUCTION_LEAGUE_GAMES),
        "--league-opponents",
        str(PRODUCTION_LEAGUE_ACTIVE_OPPONENTS + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS),
        "--repeats",
        "6",
        "--seed",
        str(seed),
        "--reward-mode",
        "terminal-outcome",
        "--auxiliary-mode",
        "enabled",
        "--structured-latent-coefficient",
        "0.0",
        "--structured-decision-coefficient",
        "0.0",
        "--structured-critic-latent-coefficient",
        "1.0",
        "--structured-critic-value-coefficient",
        "1.0",
        "--update-compile-mode",
        "default",
        "--rollout-forward-mode",
        PRODUCTION_ROLLOUT_FORWARD_MODE,
        "--rollout-bfloat16",
        "--output",
        str(output),
        *model_config_arguments(resolve_architecture(family), config.to_dict()),
    ]
    if artifact is not None:
        command.extend(("--init-actor-from", str(artifact)))
    return command


def measure(call, *, warmup, repeats, deadline):
    """Cold wall includes compile; warm samples expose device AND synchronized wall."""

    def once():
        if time.monotonic() >= deadline:
            raise TimeoutError("benchmark deadline reached; no partial report is successful")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        result = call()
        end.record()
        torch.cuda.synchronize()
        return result, {
            "cuda_ms": start.elapsed_time(end),
            "wall_ms": (time.perf_counter() - wall) * 1000,
        }

    torch.cuda.reset_peak_memory_stats()
    result, cold = once()
    cold_peak = torch.cuda.max_memory_allocated()
    del result
    for _ in range(warmup):
        result, _ = once()
        del result
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        result, timing = once()
        if isinstance(result, torch.Tensor) and not torch.isfinite(result).all():
            raise RuntimeError("non-finite measured objective")
        del result
        samples.append(timing)
    return {
        "cold_compile_and_first_execution": cold,
        "cold_peak_allocated_bytes": cold_peak,
        "warm_samples": samples,
        "warm_median_cuda_ms": statistics.median(x["cuda_ms"] for x in samples),
        "warm_median_wall_ms": statistics.median(x["wall_ms"] for x in samples),
        "warm_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "warm_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def build_actor(family, config, artifact):
    architecture = resolve_architecture(family)
    if artifact is None:
        return architecture.actor_class(config).cuda().eval()
    actor, payload = load_actor_artifact(artifact, device="cuda")
    if resolve_architecture(payload).name != family:
        raise ValueError(f"{artifact}: wrong family for {family}; no cross-family loading")
    # BC readout support is metadata only for actors, unlike its trunk configuration.
    ignored = {"value_atoms", "value_min", "value_max", "value_sigma_ratio"}
    actual = {k: v for k, v in actor.config.to_dict().items() if k not in ignored}
    expected = {k: v for k, v in config.to_dict().items() if k not in ignored}
    if actual != expected:
        raise ValueError(f"{artifact}: actor configuration differs from benchmark contract")
    return actor


def collect_corpus(actor, seed):
    # Identical frozen weights, deterministic lane assignment, real legal native sampling.
    count = PRODUCTION_LEAGUE_ACTIVE_OPPONENTS + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS
    opponents = [type(actor)(actor.config).cuda().eval() for _ in range(count)]
    for opponent in opponents:
        opponent.load_state_dict(actor.state_dict())
        opponent.requires_grad_(False)
    assignments = np.arange(PRODUCTION_LEAGUE_GAMES, dtype=np.int64) % count
    np.random.default_rng(seed).shuffle(assignments)
    torch.cuda.synchronize()
    started = time.perf_counter()
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=PRODUCTION_SELF_PLAY_GAMES,
        league_games=PRODUCTION_LEAGUE_GAMES,
        opponent_indices=assignments,
        seed_start=seed,
        episode_steps=PRODUCTION_EPISODE_STEPS,
        temperature=PRODUCTION_TEMPERATURE,
        opponent_temperature=PRODUCTION_TEMPERATURE,
        sampling_seed=seed + 1,
        forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
        forward_autocast=True,
        reward_mode="terminal-outcome",
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if rollout.valid.shape[1] != 719 or not rollout.valid.all():
        raise RuntimeError("corpus must contain complete native 719-step trajectories")
    return rollout, seconds


def matched_batch(rollout, seed, ppo):
    # Sample complete contiguous 16-state runs throughout the full trajectories;
    # preserve original IDs for NextLat rather than joining unrelated episodes.
    rows = 6400
    run = 16
    generator = np.random.default_rng(seed)
    # Duplicates would alias ancestry in the plan. Choose disjoint runs instead.
    available = np.arange(rollout.valid.shape[0] * (719 // run))
    chosen = generator.choice(available, rows // run, replace=False)
    indices = (
        (chosen // (719 // run))[:, None] * 719
        + (chosen % (719 // run))[:, None] * run
        + np.arange(run)
    ).reshape(-1)
    host = {}
    for name, array in {**rollout.states, **{n: getattr(rollout, n) for n in FACTOR_NAMES}}.items():
        host[name] = np.ascontiguousarray(array.reshape((-1, *array.shape[2:]))[indices])
    # True Monte Carlo targets from the same native terminal-outcome episodes.
    returns = np.zeros_like(rollout.rewards, dtype=np.float32)
    carry = np.zeros(rollout.valid.shape[0], dtype=np.float32)
    for step in range(718, -1, -1):
        carry = rollout.rewards[:, step] + ppo.gamma * carry
        returns[:, step] = carry
    host["value_targets"] = np.ascontiguousarray(returns.reshape(-1)[indices])
    host["advantages"] = host["value_targets"].copy()
    host["advantages"] -= host["advantages"].mean()
    host["advantages"] /= max(float(host["advantages"].std()), 1e-8)
    digest = hashlib.sha256()
    for name, array in sorted(host.items()):
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    staged = {name: torch.from_numpy(array).cuda() for name, array in host.items()}
    plan = structured_horizon_plan(indices // 719, indices % 719, 1)
    plan = type(plan)(*(tensor.cuda() for tensor in plan))
    return (
        staged,
        plan,
        {
            "rows": rows,
            "run_length": run,
            "sha256": digest.hexdigest(),
            "original_indices": indices.tolist(),
            "advantages": "normalized Monte Carlo terminal returns",
            "value_targets": "discounted native terminal-outcome Monte Carlo",
        },
    )


def family_benchmark(family, config, artifact, staged, plan, args, deadline):
    torch.manual_seed(args.seed)
    actor = build_actor(family, config, artifact)
    critic = resolve_architecture(family).critic_class(config).cuda().train()
    pair_towers(actor, critic)
    predictor = StructuredCriticDynamics(config).cuda().train()
    ppo = historical_ppo_config()
    indices = torch.arange(6400, device="cuda")
    actor_args = _actor_batch_args(family, staged, indices)
    critic_args = _critic_batch_args(family, staged, indices, actor_args=actor_args)
    factors = tuple(
        staged[name].long() if index < 3 else staged[name].bool()
        for index, name in enumerate(FACTOR_NAMES)
    )
    timings = {}
    timing_args = dict(warmup=args.warmup, repeats=args.repeats, deadline=deadline)

    replay = torch.compile(_replayed_component_logprobs, fullgraph=True, mode="default")
    with torch.no_grad():
        timings["behavior_replay_6400"] = measure(
            lambda: replay(actor, *factors[:6], True, *actor_args), **timing_args
        )
        old = tuple(t.detach().clone() for t in replay(actor, *factors[:6], True, *actor_args)[:3])
    normalizer = (
        sum(int(x.sum()) for x in factors[6:9])
        if ppo.policy_loss_reduction == "components"
        else 6400
    )

    def actor_loss():
        terms = _actor_minibatch_terms(
            actor,
            *factors,
            *old,
            staged["advantages"],
            ppo.clip_low,
            ppo.clip_high,
            True,
            *actor_args,
            policy_ratio_scope=ppo.policy_ratio_scope,
        )
        return -terms[0] / normalizer

    def critic_loss():
        pack = _structured_critic_minibatch_fit_terms(
            critic, staged["value_targets"], True, *critic_args
        )
        auxiliary, _ = _structured_critic_auxiliary_terms(
            critic,
            predictor,
            staged,
            indices,
            steps_per_trajectory=719,
            config=ppo,
            autocast_enabled=True,
            model_grad=True,
            complete_windows=False,
            belief=StructuredCriticBelief(*pack[2:]),
            plan=plan,
        )
        return pack[0] + auxiliary

    # BF16 eager is ONLY a correctness reference; never included as a performance fallback.
    compiled_actor = torch.compile(actor_loss, fullgraph=True, mode="default")
    compiled_critic = torch.compile(critic_loss, fullgraph=True, mode="default")
    parity = {}
    for label, eager, compiled, modules in (
        ("actor_ppo", actor_loss, compiled_actor, (actor,)),
        ("critic_hl_plus_nextlat", critic_loss, compiled_critic, (critic, predictor)),
    ):

        def backward(modules=modules, compiled=compiled):
            for module in modules:
                module.zero_grad(set_to_none=True)
            loss = compiled()
            loss.backward()
            return loss.detach()

        timings[label + "_forward_backward"] = measure(backward, **timing_args)
        compiled_value = compiled().detach()
        eager_value = eager().detach()
        error = float((compiled_value - eager_value).abs())
        parity[label] = {
            "compiled": float(compiled_value),
            "eager_bf16": float(eager_value),
            "absolute_error": error,
        }
        torch.testing.assert_close(compiled_value, eager_value, rtol=2e-2, atol=2e-3)
        for module in modules:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError(f"{label}: missing/nonfinite compiled gradients")
        for module in modules:
            module.zero_grad(set_to_none=True)

    # Learner wave: 128*2+64=320; frozen active states: 64 (ensemble padding
    # overhead belongs to the emitted whole-iteration benchmark).
    # Match the collector's FP32 master actor under BF16 autocast.
    for rows in (320, 64):
        inputs = type(actor_args[0])(*(x[:rows] for x in actor_args[0]))
        forward = torch.compile(actor, fullgraph=True, mode="default")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            timings[f"rollout_actor_forward_{rows}"] = measure(
                lambda forward=forward, inputs=inputs: forward(inputs), **timing_args
            )
            expected = actor(inputs)
            actual = forward(inputs)
            errors = [
                float((left.float() - right.float()).abs().max())
                for left, right in zip(actual, expected, strict=True)
            ]
            # Raw logits include inactive and illegal choices, and additive
            # offsets do not change a categorical policy. Gate the consumer's
            # legal distributions, retaining raw differences as diagnostics.
            distributions = {}
            for name, left, right, mask, active in zip(
                ("unit", "kind", "quantity"),
                (
                    expected.unit_logits,
                    expected.market_kind_logits,
                    actor.quantity_logits(
                        expected.market_quantity_context, factors[1][:rows], factors[5][:rows]
                    ),
                ),
                (
                    actual.unit_logits,
                    actual.market_kind_logits,
                    actor.quantity_logits(
                        actual.market_quantity_context, factors[1][:rows], factors[5][:rows]
                    ),
                ),
                factors[3:6],
                factors[6:9],
                strict=True,
            ):
                left_log = mask_logits(left, mask[:rows]).log_softmax(-1)
                right_log = mask_logits(right, mask[:rows]).log_softmax(-1)
                active = active[:rows]
                delta = left_log - right_log
                probability = left_log.exp()
                kl = (probability * delta).sum(-1)[active]
                tail = (probability * (delta.abs() > UPDATE_REPLAY_TAIL_LOGPROB)).sum(-1)[active]
                mean_kl = float(kl.mean()) if kl.numel() else 0.0
                tail_mass = float(tail.mean()) if tail.numel() else 0.0
                distributions[name] = {"kl": mean_kl, "tail_probability_mass": tail_mass}
                if not np.isfinite(mean_kl) or mean_kl > MAX_UPDATE_REPLAY_KL:
                    raise RuntimeError(f"{family}/{rows}/{name}: policy KL {mean_kl}")
                if not np.isfinite(tail_mass) or tail_mass > MAX_UPDATE_REPLAY_TAIL_FRACTION:
                    raise RuntimeError(f"{family}/{rows}/{name}: policy tail mass {tail_mass}")
            parity[f"rollout_{rows}"] = {
                "max_absolute_error_by_output": errors,
                "legal_policy_distributions": distributions,
            }
    return {
        "family": family,
        "model_config": config.to_dict(),
        "ppo_config": asdict(ppo),
        "artifact_sha256": file_sha256(artifact) if artifact else None,
        "parameters": {
            "actor": parameter_count(actor),
            "critic": parameter_count(critic),
            "critic_nextlat": parameter_count(predictor),
        },
        "timings": timings,
        "compiled_parity": parity,
    }


def main():
    args = parse_args()
    configs = {
        STRUCTURED: legacy_config(),
        ENTITY_ATTENTION: EntityConfig(model_dim=args.model_dim, core_layers=args.core_layers),
    }
    artifacts = {STRUCTURED: args.legacy_actor, ENTITY_ATTENTION: args.entity_actor}
    commands = {
        family: iteration_command(
            family,
            config,
            artifacts[family],
            args.output.with_name(f"{family}-whole-iteration.jsonl"),
            args.seed,
        )
        for family, config in configs.items()
    }
    report = {
        "status": "commands_only" if args.commands_only else "running",
        "whole_iteration_commands": {
            k: {"argv": v, "shell": shlex.join(v)} for k, v in commands.items()
        },
        "scope": (
            "historical critic-NextLat-on fixed-state matched forward/backward; "
            "not the current production auxiliary recipe; "
            "whole-iteration pair required for throughput claims"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    if args.commands_only:
        save()
        print(json.dumps(report, indent=2))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; no CPU, FP32, or eager performance fallback")
    deadline = time.monotonic() + args.deadline_seconds
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    gpu_identity = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    report.update(
        source_identity=source_identity(),
        seed=args.seed,
        hardware={
            "gpu": props.name,
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
            "multiprocessors": props.multi_processor_count,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nvidia_smi": gpu_identity,
        },
        execution={
            "precision": "BF16; FP32 master weights and quantity heads",
            "compile": "inductor default/fullgraph",
            "attention": "architecture-selected supported backend (no forced flash)",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "deadline_seconds": args.deadline_seconds,
        },
    )
    save()
    try:
        collector = build_actor(ENTITY_ATTENTION, configs[ENTITY_ATTENTION], args.entity_actor)
        rollout, seconds = collect_corpus(collector, args.seed)
        report["native_corpus_rollout"] = {
            "seconds_including_cold_compile": seconds,
            "self_play_games": PRODUCTION_SELF_PLAY_GAMES,
            "league_games": PRODUCTION_LEAGUE_GAMES,
            "steps": 719,
            "states": rollout.state_count,
            "collector_family": ENTITY_ATTENTION,
            "artifact_sha256": file_sha256(args.entity_actor) if args.entity_actor else None,
        }
        ppo = historical_ppo_config()
        staged, plan, report["matched_inputs"] = matched_batch(rollout, args.seed, ppo)
        del collector, rollout
        gc.collect()
        torch.cuda.empty_cache()
        report["families"] = []
        for family, config in configs.items():
            report["families"].append(
                family_benchmark(family, config, artifacts[family], staged, plan, args, deadline)
            )
            save()
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
