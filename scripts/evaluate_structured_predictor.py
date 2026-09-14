#!/usr/bin/env python3
"""Evaluate a checkpointed structured predictor on a fixed fresh self-play panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from kaggriculture.actor_dynamics import ActorDynamics
from kaggriculture.inference import load_actor_artifact
from kaggriculture.ppo import (
    _STRUCTURED_AUXILIARY_METRICS,
    PpoConfig,
    _fixed_minibatch_positions,
    _stage_tensor,
    _structured_auxiliary_horizon,
    _structured_auxiliary_terms,
    _structured_transition_order,
)
from kaggriculture.provenance import file_sha256, source_identity
from kaggriculture.rollout import collect_self_play_rust
from kaggriculture.structured import StructuredActor
from kaggriculture.training import checkpoint_agent_states, require_checkpoint_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--agent", type=int)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=31_000)
    parser.add_argument("--sampling-seed", type=int, default=20_260_815)
    parser.add_argument("--control-seed", type=int, default=20_260_815)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--forward-mode", default="inductor")
    parser.add_argument(
        "--forward-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _agent_index(payload: dict[str, Any], requested: int | None) -> int:
    states = checkpoint_agent_states(payload)
    if len(states) == 1:
        if requested is not None:
            raise ValueError("single-learner checkpoint does not accept --agent")
        return 0
    if requested is None or not 0 <= requested < len(states):
        raise ValueError("population checkpoint requires a valid --agent")
    return requested


def _staged_rollout(rollout: Any, device: torch.device) -> dict[str, Tensor]:
    staged = {name: _stage_tensor(array, device) for name, array in rollout.states.items()}
    staged |= {
        "unit_actions": _stage_tensor(rollout.unit_actions, device),
        "market_kinds": _stage_tensor(rollout.market_kinds, device),
        "market_quantities": _stage_tensor(rollout.market_quantities, device),
        "unit_masks": _stage_tensor(rollout.unit_masks, device),
        "market_kind_masks": _stage_tensor(rollout.market_kind_masks, device),
        "market_quantity_masks": _stage_tensor(rollout.market_quantity_masks, device),
        "unit_active": _stage_tensor(rollout.unit_active, device),
        "market_active": _stage_tensor(rollout.market_active, device),
        "market_quantity_active": _stage_tensor(rollout.market_quantity_active, device),
    }
    return staged


def _measure(
    actor: StructuredActor,
    dynamics: ActorDynamics,
    staged: dict[str, Tensor],
    windows: np.ndarray,
    *,
    steps_per_trajectory: int,
    config: PpoConfig,
    batch_size: int,
    autocast_enabled: bool,
) -> dict[str, float]:
    device = next(actor.parameters()).device
    windows_per_batch = max(1, batch_size // windows.shape[1])
    totals = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in _STRUCTURED_AUXILIARY_METRICS
    }
    loss_total = torch.zeros((), device=device, dtype=torch.float64)
    measured = 0

    actor.eval()
    dynamics.eval()
    with torch.inference_mode():
        positions, counts = _fixed_minibatch_positions(windows.shape[0], windows_per_batch)
        for row, count in zip(positions, counts, strict=True):
            # The final row wraps onto the epoch's leading windows so the
            # training partition is one shape; a mean over windows must not
            # weight those repeats twice, so it is trimmed to the fresh rows.
            selected = windows[row[: int(count)]]
            indices = torch.from_numpy(selected.reshape(-1)).to(device=device)
            loss, terms = _structured_auxiliary_terms(
                actor,
                dynamics,
                staged,
                indices,
                steps_per_trajectory=steps_per_trajectory,
                config=config,
                autocast_enabled=autocast_enabled,
                model_grad=False,
                complete_windows=True,
            )
            weight = selected.shape[0]
            loss_total += loss.detach().double() * weight
            for name in _STRUCTURED_AUXILIARY_METRICS:
                totals[name] += getattr(terms, name).detach().double() * weight
            measured += weight
    result = {name: float(total / measured) for name, total in totals.items()}
    result["combined"] = float(loss_total / measured)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.games < 1 or args.batch_size < 1:
        raise ValueError("games and batch size must be positive")
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_checkpoint_format(payload)
    agent = _agent_index(payload, args.agent)
    actor_module, _ = load_actor_artifact(
        checkpoint_path,
        device=device,
        agent=None if len(checkpoint_agent_states(payload)) == 1 else agent,
    )
    if not isinstance(actor_module, StructuredActor):
        raise ValueError("predictor evaluation requires a structured actor")
    actor = actor_module
    config = PpoConfig(**payload["ppo_config"])
    state = checkpoint_agent_states(payload)[agent]
    if "structured_dynamics" not in state:
        raise ValueError("checkpoint has no structured dynamics predictor")
    trained = ActorDynamics(actor.config).to(device)
    trained.load_state_dict(state["structured_dynamics"], strict=True)

    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.control_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.control_seed)
        control = ActorDynamics(actor.config).to(device)

    rollout = collect_self_play_rust(
        actor,
        games=args.games,
        seed_start=args.seed_start,
        episode_steps=720,
        temperature=1.0,
        sampling_seed=args.sampling_seed,
        forward_mode=args.forward_mode,
        forward_autocast=args.forward_bfloat16,
    )
    windows = _structured_transition_order(
        rollout.valid,
        None,
        _structured_auxiliary_horizon(config),
        np.random.default_rng(0),
    )
    staged = _staged_rollout(rollout, device)
    measure_args = {
        "steps_per_trajectory": rollout.valid.shape[1],
        "config": config,
        "batch_size": args.batch_size,
        "autocast_enabled": args.forward_bfloat16 and device.type == "cuda",
    }
    trained_metrics = _measure(actor, trained, staged, windows, **measure_args)
    control_metrics = _measure(actor, control, staged, windows, **measure_args)
    ratios = {
        name: (trained_metrics[name] / value if value else 0.0)
        for name, value in control_metrics.items()
    }
    report = {
        "format_version": 1,
        "command": sys.argv,
        "evaluation_source_identity": source_identity(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "iteration": int(payload["iteration"]),
            "source_identity": payload["source_identity"],
            "agent": None if len(checkpoint_agent_states(payload)) == 1 else agent,
        },
        "panel": {
            "games": args.games,
            "seed_start": args.seed_start,
            "sampling_seed": args.sampling_seed,
            "control_seed": args.control_seed,
            "windows": int(windows.shape[0]),
            "batch_size": args.batch_size,
            "forward_mode": args.forward_mode,
            "forward_bfloat16": args.forward_bfloat16,
            "device": str(device),
            "torch": torch.__version__,
        },
        "trained": trained_metrics,
        "fixed_random_control": control_metrics,
        "trained_over_random": ratios,
    }
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
