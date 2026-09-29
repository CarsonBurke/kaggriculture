#!/usr/bin/env python3
"""Gate one complete native interface-3 rollout before BC/PPO campaigns."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from kaggriculture.entity import EntityActor, EntityConfig
from kaggriculture.rollout import collect_mixed_play_rust


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--forward-mode", default="eager")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--league-games", type=int, default=0)
    parser.add_argument(
        "--production-model",
        action="store_true",
        help="the entity-attention family at its full default width, as production ran it "
        "before adopting lejepa",
    )
    parser.add_argument("--seed-start", type=int, default=7_310_000)
    args = parser.parse_args()
    if args.games < 0 or args.league_games < 0 or args.games + args.league_games < 1:
        parser.error("at least one nonnegative game count must be positive")
    config = (
        EntityConfig(action_interface=3, observation_schema_version=4)
        if args.production_model
        else EntityConfig(
            action_interface=3,
            model_dim=32,
            attention_heads=4,
            attention_kv_heads=2,
            farm_blocks=1,
            core_layers=1,
        )
    )
    actor = EntityActor(config).to(args.device)
    result = collect_mixed_play_rust(
        actor,
        opponents=(actor,) if args.league_games else (),
        self_play_games=args.games,
        league_games=args.league_games,
        seed_start=args.seed_start,
        deterministic=False,
        reward_mode="terminal-outcome",
        forward_mode=args.forward_mode,
        forward_autocast=args.device == "cuda",
    )
    if result.market_set_values is None or result.market_set_masks is None:
        raise AssertionError("native rollout omitted market-set decisions")
    if result.market_set_active is None or result.old_market_set_logprobs is None:
        raise AssertionError("native rollout omitted market-set probabilities")
    values = result.market_set_values.astype(np.int64)
    selected_legal = np.take_along_axis(result.market_set_masks, values[..., None], -1)
    if not selected_legal.all():
        raise AssertionError("native sampler selected an illegal market-set value")
    if not np.isfinite(result.old_market_set_logprobs).all():
        raise AssertionError("nonfinite market-set behavior likelihood")
    if not result.learner_stochastic or result.horizon != 719:
        raise AssertionError("native rollout has the wrong behavior contract")
    print(
        json.dumps(
            {
                "self_play_games": args.games,
                "league_games": args.league_games,
                "trajectories": result.trajectories,
                "horizon": result.horizon,
                "market_set_decisions": int(result.market_set_active.sum()),
                "emitted_market_slots": int(result.market_active.sum()),
                "mean_final_money": float(result.final_money.mean()),
                "max_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
