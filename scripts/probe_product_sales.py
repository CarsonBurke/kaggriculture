#!/usr/bin/env python3
"""Count legal product sell orders in full native games from a frozen actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kaggriculture import rollout as rollout_module
from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.constants import QUANTITY_BINS
from kaggriculture.inference import load_actor_artifact
from kaggriculture.provenance import file_sha256
from kaggriculture.rollout import collect_self_play_rust


def action_diagnostics(rollout: rollout_module.RolloutBatch) -> dict[str, object]:
    """Count selected actions and cap-bin opportunities on valid actor rows."""
    active_market = rollout.valid[..., None] & rollout.market_active
    active_units = rollout.valid[..., None] & rollout.unit_active
    hires = active_market & (rollout.market_kinds == int(MarketKind.HIRE))
    hires_by_step = hires.sum(axis=(0, 2))

    pickup_caps = {
        "wheat_16": UnitAction.PICKUP_WHEAT_16,
        "fertilizer_8": UnitAction.PICKUP_FERTILIZER_8,
        "goose_4": UnitAction.PICKUP_GOOSE_4,
        "cow_4": UnitAction.PICKUP_COW_4,
        "sheep_4": UnitAction.PICKUP_SHEEP_4,
    }
    return {
        "hire": {
            "orders": int(hires.sum()),
            "by_step": {str(step): int(count) for step, count in enumerate(hires_by_step) if count},
            "terminal_step_718": int(hires_by_step[718]) if len(hires_by_step) > 718 else 0,
        },
        "pickup_caps": {
            name: {
                "selected": int((active_units & (rollout.unit_actions == int(action))).sum()),
                "legal_opportunities": int(
                    (active_units & rollout.unit_masks[..., int(action)]).sum()
                ),
            }
            for name, action in pickup_caps.items()
        },
        "place_wheat": int(
            (active_units & (rollout.unit_actions == int(UnitAction.PLACE_WHEAT))).sum()
        ),
        "feed": int((active_units & (rollout.unit_actions == int(UnitAction.FEED))).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=6_000_000)
    parser.add_argument("--sampling-seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("product-sales probe requires a BF16-capable CUDA device")
    source_root = Path.cwd().resolve()
    rollout_source = Path(rollout_module.__file__).resolve()
    if not rollout_source.is_relative_to(source_root / "src"):
        raise RuntimeError(f"native rollout imported from {rollout_source}, outside {source_root}")

    actor, _ = load_actor_artifact(args.artifact, torch.device("cuda"))
    rollout = collect_self_play_rust(
        actor,
        games=args.games,
        seed_start=args.seed_start,
        sampling_seed=args.sampling_seed,
        temperature=1.0,
        forward_mode="inductor_graph",
        forward_autocast=True,
    )
    active = rollout.valid[..., None] & rollout.market_active
    bins = np.asarray(QUANTITY_BINS)
    sales = {}
    for kind in MarketKind:
        if not kind.name.startswith("SELL_"):
            continue
        selected = active & (rollout.market_kinds == int(kind))
        sales[kind.name.removeprefix("SELL_")] = {
            "orders": int(selected.sum()),
            "units": int(bins[rollout.market_quantities[selected].astype(np.intp)].sum()),
            "trajectories_with_sale": int(selected.any(axis=(1, 2)).sum()),
        }
    report = {
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": file_sha256(args.artifact),
        "games": args.games,
        "trajectories": rollout.trajectories,
        "seed_start": args.seed_start,
        "sampling_seed": args.sampling_seed,
        "mode": "sampled_temperature_1_native_self_play",
        "source_root": str(source_root),
        "sales": sales,
        "action_diagnostics": action_diagnostics(rollout),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
