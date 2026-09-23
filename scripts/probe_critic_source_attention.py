#!/usr/bin/env python3
"""Measure trained critic source attention on a frozen-policy development wave."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.critic_diagnostics import critic_replay_arrays
from kaggriculture.evaluation import artifact_seed_usage, seed_protocol
from kaggriculture.provenance import file_sha256, source_identity
from kaggriculture.tokens import TILE_COUNT


def attention_summary(probabilities: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Separate attention preference from the prior induced by context group size."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if (
        probabilities.ndim != 3
        or valid.shape != (probabilities.shape[0], probabilities.shape[2])
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-5)
        or np.any(np.where(valid[:, None, :], 0, probabilities) != 0)
    ):
        raise ValueError("attention probabilities must be normalized over valid context tokens")
    entities = MAX_UNITS + MAX_MARKET_ORDERS
    private_start = valid.shape[1] - MAX_UNITS
    economy_start = entities + 2 * TILE_COUNT
    if private_start <= economy_start:
        raise ValueError("attention context does not contain entity and source memory groups")
    slices = {
        "states": slice(0, entities),
        "own_farm": slice(entities, entities + TILE_COUNT),
        "opponent_farm": slice(entities + TILE_COUNT, economy_start),
        "economy": slice(economy_start, private_start),
        "private_units": slice(private_start, valid.shape[1]),
    }
    totals = valid.sum(axis=-1)
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy = -(probabilities * log_probabilities).sum(axis=-1)
    result: dict[str, Any] = {
        "states": probabilities.shape[0],
        "heads": probabilities.shape[1],
        "valid_tokens_mean": float(totals.mean()),
        "entropy_mean_nats": float(entropy.mean()),
        "entropy_per_head_nats": entropy.mean(axis=0).tolist(),
        "entropy_fraction_of_uniform_mean": float((entropy / np.log(totals[:, None])).mean()),
        "groups": {},
    }
    for name, selection in slices.items():
        counts = valid[:, selection].sum(axis=-1)
        mass = probabilities[:, :, selection].sum(axis=-1)
        prior = counts / totals
        present = counts > 0
        result["groups"][name] = {
            "valid_tokens_mean": float(counts.mean()),
            "attention_mass_mean": float(mass.mean()),
            "attention_mass_per_head": mass.mean(axis=0).tolist(),
            "uniform_count_prior_mass_mean": float(prior.mean()),
            "present_state_fraction": float(present.mean()),
            "attention_per_valid_token_mean_when_present": (
                float((mass[present] / counts[present, None]).mean()) if present.any() else None
            ),
            "mass_over_count_prior_mean_when_present": (
                float((mass[present] / prior[present, None]).mean()) if present.any() else None
            ),
        }
    return result


def require_entity_source_pool(config) -> None:
    if (
        not getattr(config, "critic_source_read", False)
        or getattr(config, "critic_architecture", "entity") != "entity"
    ):
        raise ValueError("attention probe requires an entity critic with source read enabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True, help="fixed BC rollout policy")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained source-read critic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=4_503_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    from kaggriculture.entity import EntityCritic
    from kaggriculture.inference import load_actor_artifact
    from kaggriculture.model import policy_compile_options
    from kaggriculture.ppo import _critic_batch_args
    from kaggriculture.registry import resolve_architecture
    from kaggriculture.rollout import collect_mixed_play_rust
    from kaggriculture.training import checkpoint_agent_states, require_checkpoint_format

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("attention probe requires queued CUDA with native BF16")
    started = time.perf_counter()
    torch.set_num_threads(1)
    actor_digest, checkpoint_digest = file_sha256(args.actor), file_sha256(args.checkpoint)
    actor, actor_metadata = load_actor_artifact(args.actor, device="cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    require_checkpoint_format(payload)
    config = resolve_architecture(payload).build_config(payload["model_config"])
    require_entity_source_pool(config)
    members = checkpoint_agent_states(payload)
    if len(members) != 1:
        raise ValueError("attention probe requires a single-learner checkpoint")
    critic = EntityCritic(config).to("cuda")
    critic.load_state_dict(members[0]["critic"], strict=True)
    critic.eval().requires_grad_(False)
    protocol = seed_protocol(
        "development",
        args.seed_start,
        192,
        usage=artifact_seed_usage(actor_metadata) + artifact_seed_usage(payload),
    )
    rollout = collect_mixed_play_rust(
        actor,
        opponents=(actor,),
        self_play_games=128,
        league_games=64,
        opponent_indices=np.arange(64) % 3,
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
    if rollout.state_count != 320 * 719 or not rollout.valid.all():
        raise RuntimeError("attention probe requires the full frozen-policy rollout")
    eligible = np.flatnonzero(
        (rollout.valid & (rollout.episode_seeds[:, None] % 5 == 0)).reshape(-1)
    )
    positions = eligible[np.linspace(0, len(eligible) - 1, 512, dtype=np.int64)]
    staged = {
        name: torch.from_numpy(array.reshape((-1, *array.shape[2:]))[positions]).to("cuda")
        for name, array in critic_replay_arrays(rollout).items()
    }
    critic_args = _critic_batch_args(rollout.architecture, staged, slice(None))
    captured = {}

    def capture_pool(attention: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        queries, context = inputs
        batch, query_tokens, _ = queries.shape
        query = (
            attention.query(queries)
            .view(batch, query_tokens, attention.heads, attention.head_dim)
            .transpose(1, 2)
        )
        key = (
            attention.key_value(context)
            .view(batch, context.shape[1], 2, attention.kv_heads, attention.head_dim)[:, :, 0]
            .transpose(1, 2)
        )
        # Match the production SDPA boundary, then reconstruct probabilities
        # in FP32 without the outer autocast rounding matmul back to BF16.
        query = attention.query_norm(query).bfloat16()
        key = (
            attention.key_norm(key)
            .bfloat16()
            .repeat_interleave(attention.heads // attention.kv_heads, dim=1)
        )
        with torch.autocast("cuda", enabled=False):
            logits = (query.float() @ key.float().transpose(-1, -2)) * attention.head_dim**-0.5
        valid = kwargs["context_valid"]
        captured["probabilities"] = logits.masked_fill(
            ~valid[:, None, None, :], float("-inf")
        ).softmax(dim=-1)[:, :, 0]
        captured["valid"] = valid

    handle = critic.pool_attention.register_forward_pre_hook(capture_pool, with_kwargs=True)

    def probe(*inputs: Any) -> tuple[Any, Any]:
        critic(*inputs)
        return captured["probabilities"], captured["valid"]

    compile_options = policy_compile_options("reduce-overhead")
    try:
        compiled = torch.compile(
            probe,
            fullgraph=True,
            dynamic=False,
            options=compile_options,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            probabilities, valid = compiled(*critic_args)
        summary = attention_summary(probabilities.float().cpu().numpy(), valid.cpu().numpy())
    finally:
        handle.remove()
    if file_sha256(args.actor) != actor_digest or file_sha256(args.checkpoint) != checkpoint_digest:
        raise RuntimeError("input artifact changed during attention probe")
    report = {
        "complete": True,
        "source_identity": source_identity(),
        "diagnostic_script_sha256": file_sha256(Path(__file__)),
        "actor": {"path": str(args.actor.resolve()), "sha256": actor_digest},
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_digest,
            "source_identity": payload["source_identity"],
            "model_config": payload["model_config"],
        },
        "seed_protocol": protocol,
        "sampling_seed": 20260919,
        "precision": "Compiled CUDA BF16 model; FP32 normalized QK probability reconstruction",
        "compile_options": compile_options,
        "scope": "Attention allocation diagnostic, not causal attribution or gameplay evidence",
        "probe_positions": [
            {
                "seed": int(rollout.episode_seeds[position // 719]),
                "seat": int(rollout.seats[position // 719]),
                "step": int(position % 719),
            }
            for position in positions
        ],
        "attention": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
