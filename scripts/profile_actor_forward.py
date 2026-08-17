"""Attribute one rollout-shaped actor forward to modules and CUDA kernels.

The stage profiler shows the actor forward is roughly two thirds of collection
wall clock, at a throughput far below what the parameter count implies. That
gap is either the spatial trunk, the entity transformer, or per-kernel overhead,
and those call for different fixes, so this splits the forward three ways.

Two views are reported. The module view wraps the forward's own top-level
sections in `record_function` scopes and sums the CUDA time attributed to each,
answering which part of the network to work on. The kernel view lists the
costliest individual kernels with launch counts, answering whether the cost is
bandwidth in a few large kernels or overhead spread across many small ones.

Both fp32 and bfloat16 autocast are profiled, because collection currently runs
fp32 while the update runs bf16 and the split may differ between them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from kaggriculture.model import FarmActor
from kaggriculture.rollout import _native_encoded_wave
from kaggriculture.rust_env import load_native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=112)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-kernels", type=int, default=14)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _instrument(actor: FarmActor) -> None:
    """Wrap the forward's top-level sections in profiler scopes."""
    spatial, transformer = actor.spatial, actor.transformer

    def scoped(name: str, module: torch.nn.Module):
        original = module.forward

        def forward(*args, **kwargs):
            with record_function(name):
                return original(*args, **kwargs)

        module.forward = forward

    scoped("section::spatial_trunk", spatial)
    scoped("section::entity_transformer", transformer)
    blocks = [
        *transformer.encoder,
        transformer.bottleneck,
        *transformer.decoder,
    ]
    for index, block in enumerate(blocks):
        scoped(f"block::{index:02d}", block)
    for name in ("input", "encoder", "down", "bottleneck", "up_projection", "decoder", "output"):
        scoped(f"cnn::{name}", getattr(spatial, name))


def _profile(actor: FarmActor, inputs, autocast: bool, args) -> dict:
    device = torch.device(args.device)
    for _ in range(args.warmup):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            actor(*inputs)
    torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(args.iters):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                actor(*inputs)
        torch.cuda.synchronize(device)

    scopes = ("section::", "block::", "cnn::")
    sections: dict[str, float] = {}
    for event in prof.key_averages():
        if event.key.startswith(scopes):
            sections[event.key] = event.device_time_total / args.iters / 1e3

    kernels = []
    total_device = 0.0
    for event in prof.key_averages():
        if event.device_time_total <= 0 or event.key.startswith(scopes):
            continue
        if event.self_device_time_total <= 0:
            continue
        total_device += event.self_device_time_total
        kernels.append(
            {
                "name": event.key,
                "milliseconds": event.self_device_time_total / args.iters / 1e3,
                "launches_per_forward": event.count / args.iters,
            }
        )
    kernels.sort(key=lambda entry: -entry["milliseconds"])
    return {
        "section_milliseconds": sections,
        "total_device_milliseconds": total_device / args.iters / 1e3,
        "kernel_launches_per_forward": sum(entry["launches_per_forward"] for entry in kernels),
        "top_kernels": kernels[: args.top_kernels],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    seeds = np.arange(args.seed, args.seed + args.games, dtype=np.uint64)
    environment = load_native().BatchEnv(seeds)
    actor = FarmActor().to(device).eval()
    wave = _native_encoded_wave(environment, device)
    wave.refresh(environment)
    wave.copy_to_device()
    _instrument(actor)

    report = {"games": args.games, "rows": args.games * 2, "iters": args.iters}
    with torch.inference_mode():
        inputs = wave.inputs()
        for label, autocast in (("float32", False), ("bfloat16", True)):
            report[label] = _profile(actor, inputs, autocast, args)

    report["autocast_device_speedup"] = (
        report["float32"]["total_device_milliseconds"]
        / report["bfloat16"]["total_device_milliseconds"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
