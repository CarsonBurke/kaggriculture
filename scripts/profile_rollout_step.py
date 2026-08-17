"""Attribute one self-play rollout step to its individual stages.

The iteration benchmark reports a single rollout number, which is enough to
rank configurations but not to decide what to fix. This walks the exact stage
sequence `collect_native_rollout` runs -- Rust encode, host-to-device upload,
actor forward, device-to-host readback, host draws, Rust sample-and-step, and
trajectory bookkeeping -- and times each one in isolation.

Attribution requires a device barrier between stages, which serializes work the
real loop leaves overlapping, so the summed stage budget runs slightly above a
measured unsynchronized step. Both numbers are reported: the stages say where
the time is, the unsynchronized total says how much there is to win.

The forward is additionally timed under bfloat16 autocast. Collection currently
runs the actor in fp32 while the update runs it in bf16, and the gap between
those two timings is what switching would be worth before any parity argument.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.model import FarmActor
from kaggriculture.rollout import (
    ROLLOUT_FORWARD_MODES,
    _categorical_draws,
    _native_encoded_wave,
    _packed_outputs_to_host,
    _PackedTransfer,
    _quantity_heads,
    _rollout_model_forward,
)
from kaggriculture.rust_env import load_native

STAGES = (
    "rust_encode",
    "host_to_device",
    "actor_forward",
    "device_to_host",
    "host_draws",
    "rust_sample_step",
    "bookkeeping",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=112)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=719)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-mode", choices=ROLLOUT_FORWARD_MODES, default="inductor")
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    seeds = np.arange(args.seed, args.seed + args.games, dtype=np.uint64)
    environment = load_native().BatchEnv(seeds)
    rows = args.games * 2

    actor = FarmActor().to(device).eval()
    wave = _native_encoded_wave(environment, device)
    sampled = environment.sample_buffers()
    kind_gate, quantity_values, quantity_bias = _quantity_heads((actor,))
    head_ids = np.zeros(rows, dtype=np.uint16)
    deterministic_rows = np.zeros(rows, dtype=np.bool_)
    temperatures = np.ones(rows, dtype=np.float32)
    generator = np.random.default_rng(args.seed)

    samples: dict[str, list[float]] = {name: [] for name in STAGES}
    samples["unsynchronized_step"] = []
    packed_host: _PackedTransfer | None = None

    def timed(name: str, record: bool, work: Any) -> Any:
        if not record:
            return work()
        _synchronize(device)
        started = time.perf_counter()
        result = work()
        _synchronize(device)
        samples[name].append((time.perf_counter() - started) * 1e3)
        return result

    autocast_forward = args.bfloat16 and device.type == "cuda"

    def run_actor() -> Any:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_forward):
            return _rollout_model_forward(actor, *wave.inputs(), mode=args.forward_mode)

    # Compilation happens on the first call, so the warmup below has to be long
    # enough to absorb it or the first recorded sample would time the compiler.
    with torch.inference_mode():
        total_started = 0.0
        for step in range(args.warmup + args.steps):
            record = step >= args.warmup
            # An unsynchronized step measured on its own iteration, so the
            # barriers used for attribution never inflate the total.
            measure_total = record and step % 2 == 0
            if measure_total:
                _synchronize(device)
                total_started = time.perf_counter()
            stage = record and not measure_total

            timed("rust_encode", stage, lambda: wave.refresh(environment))
            timed("host_to_device", stage, wave.copy_to_device)
            output = timed("actor_forward", stage, run_actor)
            host_outputs, packed_host = timed(
                "device_to_host",
                stage,
                lambda captured=output, buffer=packed_host: _packed_outputs_to_host(
                    (captured,), buffer
                ),
            )
            host = host_outputs[0]
            draws = timed("host_draws", stage, lambda: _categorical_draws(generator, rows))

            def sample_step(logits=host, uniforms=draws) -> None:
                environment.sample_and_step_into(
                    logits.unit_logits,
                    logits.market_kind_logits,
                    logits.market_quantity_context,
                    kind_gate,
                    quantity_values,
                    quantity_bias,
                    head_ids,
                    *uniforms,
                    deterministic_rows,
                    temperatures,
                    sampled,
                )

            timed("rust_sample_step", stage, sample_step)

            def bookkeeping() -> None:
                np.asarray(sampled["shaped_rewards"], dtype=np.float32).reshape(-1)
                counts = (
                    np.asarray(sampled["unit_active"]).sum(axis=1)
                    + np.asarray(sampled["market_active"]).sum(axis=1)
                    + np.asarray(sampled["market_quantity_active"]).sum(axis=1)
                )
                _ = np.asarray(sampled["entropy"]) * counts
                _ = np.asarray(sampled["dones"], dtype=np.bool_).any()
                _ = np.asarray(sampled["final_money"], dtype=np.float32)

            timed("bookkeeping", stage, bookkeeping)
            if measure_total:
                _synchronize(device)
                samples["unsynchronized_step"].append((time.perf_counter() - total_started) * 1e3)

    report = {
        "games": args.games,
        "rows": rows,
        "steps": args.steps,
        "horizon": args.horizon,
        "device": str(device),
        "forward_mode": args.forward_mode,
        "bfloat16": args.bfloat16,
        "stage_milliseconds_median": {
            name: statistics.median(values) for name, values in samples.items() if values
        },
    }
    stages = report["stage_milliseconds_median"]
    budget = sum(stages[name] for name in STAGES)
    report["stage_budget_milliseconds"] = budget
    report["stage_fractions"] = {name: stages[name] / budget for name in STAGES}
    report["projected_rollout_seconds"] = stages["unsynchronized_step"] * args.horizon / 1e3
    report["stage_budget_rollout_seconds"] = budget * args.horizon / 1e3

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
