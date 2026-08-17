#!/usr/bin/env python3
"""Prove CUDA Flash dispatch, autocast dtypes, and finite transformer gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from kaggriculture.constants import MAX_UNITS
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
)
from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    ModelConfig,
    ReluSquaredFeedForward,
    parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _inputs(batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    board = torch.randn(batch_size, BOARD_CHANNELS, 10, 10, device=device)
    global_features = torch.randn(batch_size, GLOBAL_FEATURES, device=device)
    critic_features = torch.randn(batch_size, CRITIC_FEATURES, device=device)
    units = torch.zeros(batch_size, MAX_UNITS, UNIT_FEATURES, device=device)
    units[:, :4] = torch.randn(batch_size, 4, UNIT_FEATURES, device=device)
    units[:, :4, 0] = 1.0
    positions = torch.randint(0, 10, (batch_size, MAX_UNITS, 2), device=device)
    return board, global_features, critic_features, units, positions


def _gradient_issues(module: torch.nn.Module) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
    return missing, nonfinite


def _autocast_residual_dtypes(config: ModelConfig, device: torch.device) -> dict[str, str]:
    """Record the dtype each hot activation actually carries under bf16 autocast.

    Autocast's fp32 cast list is device-specific, and on CUDA it contains `pow`
    but not `mul`. Spelling ReLU-squared as `x.square()` therefore upcast the
    FFN's widest activation to fp32 and doubled its traffic, while the CPU
    autocast policy this repo's test suite runs under shows no such thing. This
    check is the only place that discrepancy is observable, so it belongs here.
    """
    block = ReluSquaredFeedForward(config).to(device)
    hidden = torch.randn(8, 16, config.model_dim, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        projected = block.input(hidden)
        return {
            "projection": str(projected.dtype),
            "activation": str(block.activation(projected).dtype),
            "output": str(block(hidden).dtype),
        }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA transformer validation requires an available CUDA device")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    config = ModelConfig()
    actor = FarmActor(config).to(device)
    critic = DistributionalCritic(config).to(device)
    board, global_features, critic_features, units, positions = _inputs(args.batch_size, device)

    with torch.inference_mode():
        actor(board, global_features, units, positions)
        torch.cuda.synchronize()
        with profile(activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA)) as trace:
            actor(board, global_features, units, positions)
            torch.cuda.synchronize()
    event_names = sorted({event.key for event in trace.key_averages()})
    flash_events = [name for name in event_names if "flash_attention" in name.lower()]
    if not flash_events:
        raise RuntimeError("scaled-dot-product attention did not dispatch a CUDA Flash kernel")

    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    actor_output = actor(board, global_features, units, positions)
    market_kinds = (
        torch.arange(
            actor_output.market_quantity_context.shape[1],
            device=device,
        )
        .unsqueeze(0)
        .expand(args.batch_size, -1)
    )
    quantity_logits = actor.quantity_logits(
        actor_output.market_quantity_context,
        market_kinds,
    )
    critic_logits = critic(board, critic_features)
    loss = sum(tensor.float().square().mean() for tensor in actor_output)
    loss = loss + quantity_logits.float().square().mean() + critic_logits.float().square().mean()
    loss.backward()
    actor_missing, actor_nonfinite = _gradient_issues(actor)
    critic_missing, critic_nonfinite = _gradient_issues(critic)
    if actor_missing or actor_nonfinite or critic_missing or critic_nonfinite:
        details = {
            "actor_missing": actor_missing,
            "actor_nonfinite": actor_nonfinite,
            "critic_missing": critic_missing,
            "critic_nonfinite": critic_nonfinite,
        }
        raise FloatingPointError(
            "transformer backward produced invalid gradients: "
            + json.dumps(details, sort_keys=True)
        )

    autocast_dtypes = _autocast_residual_dtypes(config, device)
    if autocast_dtypes["activation"] != "torch.bfloat16":
        raise TypeError(
            "the ReLU-squared activation left the autocast compute dtype: "
            + json.dumps(autocast_dtypes, sort_keys=True)
        )

    result = {
        "batch_size": args.batch_size,
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "device": torch.cuda.get_device_name(device),
        "flash_events": flash_events,
        "autocast_dtypes": autocast_dtypes,
        "finite_backward": True,
        "model": config.to_dict(),
        "torch": torch.__version__,
    }
    rendered = json.dumps(result, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
