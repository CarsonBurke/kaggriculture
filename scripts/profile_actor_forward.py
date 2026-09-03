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

FP32 and BF16 autocast are always profiled with one scope per forward. The
optional native-BF16 replica quantifies the cost of repeated autocast conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from kaggriculture.model import FarmActor
from kaggriculture.modelargs import add_model_config_arguments, model_config_from_args
from kaggriculture.registry import ARCHITECTURES, CONV_ENTITY, resolve_architecture
from kaggriculture.rollout import _native_wave
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=112)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=CONV_ENTITY,
    )
    add_model_config_arguments(parser)
    parser.add_argument("--top-kernels", type=int, default=14)
    parser.add_argument(
        "--native-bfloat16",
        action="store_true",
        help="also profile a replica whose floating parameters and inputs are bfloat16",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _instrument(actor: FarmActor | StructuredActor) -> None:
    """Wrap the forward's top-level sections in profiler scopes."""

    def scoped(name: str, module: torch.nn.Module) -> None:
        original = module.forward

        def forward(*args, **kwargs):
            with record_function(name):
                return original(*args, **kwargs)

        module.forward = forward

    if isinstance(actor, StructuredActor):
        trunk = actor.trunk
        for name in ("tiles", "units", "economy"):
            scoped(f"tokenizer::{name}", getattr(trunk, name))
        for index, block in enumerate(trunk.farm_local):
            scoped(f"farm_block::{index:02d}", block)
        scoped("section::opponent_summary", trunk.opponent_summary)
        scoped("section::latent_read", trunk.latent_read)
        for index, block in enumerate(trunk.core):
            scoped(f"core_block::{index:02d}", block)
        for layer, block in actor.trunk.global_refresh.items():
            scoped(f"refresh_block::{layer}", block)
        for name in (
            "unit_decoder",
            "unit_local_decoder",
            "market_decoder",
            "market_economy_decoder",
        ):
            scoped(f"decoder::{name}", getattr(actor, name))
        return

    spatial, transformer = actor.spatial, actor.transformer
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


def _profile(actor: FarmActor | StructuredActor, inputs, autocast: bool, args) -> dict:
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

    scopes = (
        "section::",
        "block::",
        "cnn::",
        "tokenizer::",
        "refresh_block::",
        "farm_block::",
        "core_block::",
        "decoder::",
    )
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
    architecture = resolve_architecture(args.architecture)
    config = model_config_from_args(architecture, args)
    actor = architecture.build_actor(config.to_dict()).to(device).eval()
    wave = _native_wave(args.architecture, environment, device)
    wave.refresh(environment)
    wave.copy_to_device()
    _instrument(actor)

    report = {
        "architecture": args.architecture,
        "model_config": config.to_dict(),
        "games": args.games,
        "rows": args.games * 2,
        "iters": args.iters,
    }
    with torch.inference_mode():
        inputs = wave.inputs()
        for label, autocast in (("float32", False), ("bfloat16", True)):
            report[label] = _profile(actor, inputs, autocast, args)
        if args.native_bfloat16:
            native_actor = (
                architecture.build_actor(config.to_dict())
                .to(device=device, dtype=torch.bfloat16)
                .eval()
            )
            native_actor.load_state_dict(actor.state_dict())
            native_inputs = torch.utils._pytree.tree_map_only(
                torch.Tensor,
                lambda tensor: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor,
                inputs,
            )
            _instrument(native_actor)
            report["native_bfloat16"] = _profile(native_actor, native_inputs, False, args)

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
