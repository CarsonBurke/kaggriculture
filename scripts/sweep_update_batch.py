"""Measure the joint minibatch-size x compile-mode cost surface of the update phase.

`scripts/profile_update_backends.py` swept the compile-mode axis alone, at
production's 2048-row minibatch. It left the second axis untouched, and the
profiling that motivated it says that axis is where the remaining cost sits: a
single actor forward issues 859 kernels in fp32 and 1133 in bf16 whose summed
device time is 4.9 ms against a 12.94 ms measured wall clock, so the update path
is substantially per-launch bound rather than per-FLOP bound. Fewer, larger
minibatches issue proportionally fewer launches for the same data. The two axes
interact through memory: `reduce-overhead` reserves 2.9 GiB less than `default`
at the same speed, so it may admit a minibatch the default mode cannot fit.

Total work is held at the production schedule in every cell -- the same rollout
state count, `epochs` actor passes and `critic_epochs` critic passes -- so the
minibatch COUNT falls as the size rises and the cells are directly comparable.
Counts come from `_balanced_minibatch_slices`, production's own partitioner, so
a nominal 4096 runs 37 near-equal batches of 4040 rather than 36 full ones plus
a short tail. Both the nominal and the effective size are reported, and the
schedule runs every minibatch at the effective size, which leaves the processed
state count short of the requested total by at most one row per minibatch.

Reported per cell: median steady wall clock for the whole schedule, peak
reserved and allocated bytes, first-call compile seconds kept strictly separate
from the steady clock, and -- from `torch.profiler` with CUDA activity -- the
kernel count and summed device time of one actor and one critic minibatch, so
the launch-count hypothesis is measured rather than assumed. From those come
seconds per 1000 rollout states and the launch-bound gap ratio, wall clock over
summed device time, which is the number that says whether a smaller launch
count bought anything. A cell that runs out of memory is recorded as a result
with `"oom": true`, not raised: on a 32 GiB card a size that does not fit is
itself a finding.

`max-autotune` is excluded by default. Its win at 2048 is already known to be a
net loss over a 500-iteration run -- 8.0 minutes saved against 8.7 minutes
compiling -- and its 522 s compile time would dominate a grid.

THIS MEASURES SPEED ONLY, AND SPEED IS NOT THE WHOLE DECISION. Minibatch size
is not a free performance knob, because it changes optimization at the same
time as it changes cost. At 2048 the actor takes 73 gradient steps per epoch; at
4096 it takes 37 larger ones. That changes the granularity at which PPO's
per-minibatch KL trust region is enforced (`target_kl` 0.03 against a realized
`approx_kl` median of 2.3e-3), changes how many optimizer steps the
learning-rate warmup (`lr_warmup_steps` 32) sees before reaching the base rate,
and changes the gradient noise each step is taken under. A wall-clock win
measured here is therefore not automatically adoptable; adopting one needs a
learning run, which this script does not perform.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from kaggriculture.actions import N_QUANTITIES
from kaggriculture.model import (
    BOARD_CHANNELS,
    BOARD_SIZE,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    N_MARKET_KINDS,
    N_UNIT_ACTIONS,
    UNIT_FEATURES,
    DistributionalCritic,
    FarmActor,
)
from kaggriculture.ppo import (
    UPDATE_COMPILE_MODES,
    PpoConfig,
    _actor_minibatch_terms,
    _balanced_minibatch_slices,
    _cached_update_callable,
    _critic_minibatch_loss,
    _device_compile_mode,
    _optimizer_step,
    make_optimizers,
)

#: 73 * 2048, the state count that reproduces production's observed schedule of
#: 73 actor and 292 critic minibatches at the production minibatch size. The
#: real per-iteration count is the number of valid states in a wave, which lands
#: within one minibatch of this; pinning it here keeps the 2048 cell directly
#: comparable with the single-axis measurements in `_cached_update_callable`.
PRODUCTION_STATES = 149_504

#: Device-side Kineto categories. `kernel` is a launched CUDA kernel; the memory
#: operations are device work too and belong in the summed device time, but not
#: in a kernel count.
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")

#: Host-side categories through which work is submitted. Kernels reach the
#: device through both the runtime API (`cudaLaunchKernel`) and, for Inductor's
#: Triton kernels, the driver API (`cuLaunchKernel`), so a matcher naming one
#: family undercounts by most of the total. Rather than pin a name list that
#: drifts with the CUDA and Kineto versions, the launch count is every event in
#: these categories whose name contains `Launch`, and the per-name histogram
#: goes into the report so the composition is auditable instead of assumed.
SUBMISSION_CATEGORIES = ("cuda_runtime", "cuda_driver")


def _staged_minibatch(rows: int, device: torch.device) -> dict[str, Any]:
    """One production-shaped minibatch, resident and reused across repeats.

    Constructed exactly as `profile_update_backends._minibatch` does, down to
    the seed, so a 2048-row cell here is comparable with that harness.
    """
    generator = torch.Generator(device=device.type).manual_seed(20260817)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=device, generator=generator)

    unit_masks = rand(rows, MAX_UNITS, N_UNIT_ACTIONS) > 0.2
    kind_masks = rand(rows, MAX_MARKET_ORDERS, N_MARKET_KINDS) > 0.2
    quantity_masks = rand(rows, MAX_MARKET_ORDERS, N_QUANTITIES) > 0.2
    # Every categorical needs at least one legal action or the log-softmax over
    # its masked logits is -inf and the surrogate is not finite.
    unit_masks[..., 0] = True
    kind_masks[..., 0] = True
    quantity_masks[..., 0] = True
    return {
        "unit_actions": torch.zeros(rows, MAX_UNITS, dtype=torch.long, device=device),
        "market_kinds": torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.long, device=device),
        "market_quantities": torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.long, device=device),
        "unit_masks": unit_masks,
        "kind_masks": kind_masks,
        "quantity_masks": quantity_masks,
        "unit_active": (rand(rows, MAX_UNITS) > 0.3).float(),
        "kind_active": (rand(rows, MAX_MARKET_ORDERS) > 0.5).float(),
        "quantity_active": (rand(rows, MAX_MARKET_ORDERS) > 0.5).float(),
        "old_unit": torch.full((rows, MAX_UNITS), -2.0, device=device),
        "old_kind": torch.full((rows, MAX_MARKET_ORDERS), -1.5, device=device),
        "old_quantity": torch.full((rows, MAX_MARKET_ORDERS), -1.5, device=device),
        "advantages": torch.randn(rows, device=device, generator=generator),
        "value_targets": torch.randn(rows, device=device, generator=generator) * 0.3,
        "board": torch.randn(
            rows, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device, generator=generator
        ),
        "global_features": torch.randn(rows, GLOBAL_FEATURES, device=device, generator=generator),
        "units": rand(rows, MAX_UNITS, UNIT_FEATURES),
        "unit_positions": torch.randint(
            0, BOARD_SIZE, (rows, MAX_UNITS, 2), device=device, generator=generator
        ),
        "critic_features": torch.randn(rows, CRITIC_FEATURES, device=device, generator=generator),
    }


def _cell_schedule(
    states: int, nominal_size: int, epochs: int, critic_epochs: int
) -> dict[str, int]:
    """Production's minibatch partition for one nominal size, as plain counts."""
    slices = _balanced_minibatch_slices(states, nominal_size)
    effective = min(item.stop - item.start for item in slices)
    batches = len(slices)
    return {
        "minibatch_size": nominal_size,
        "effective_minibatch_size": effective,
        "batches_per_epoch": batches,
        "actor_minibatches": batches * epochs,
        "critic_minibatches": batches * critic_epochs,
        "states_processed": batches * effective,
    }


def _key_average_totals(profiler: Any) -> dict[str, Any]:
    """Cross-check the trace counts against the profiler's own aggregation.

    The Kineto trace schema and the `FunctionEventAvg` attribute names have
    drifted independently across releases, so neither is trusted alone; a
    disagreement between the two is visible in the JSON rather than silent.
    """
    try:
        from torch.autograd import DeviceType

        rows = [row for row in profiler.key_averages() if row.device_type == DeviceType.CUDA]
        return {
            "key_average_kernel_count": sum(int(row.count) for row in rows),
            "key_average_device_ms": sum(float(row.self_device_time_total) for row in rows) / 1e3,
        }
    except Exception as error:
        return {"key_average_error": repr(error)}


def _profile_minibatch(step: Any, device: torch.device) -> dict[str, Any]:
    """Kernel count and summed device time for one minibatch of the given kind."""
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        step()
        torch.cuda.synchronize(device)
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text()).get("traceEvents", [])
    counts = dict.fromkeys(DEVICE_CATEGORIES, 0)
    device_microseconds = 0.0
    launch_names: dict[str, int] = {}
    for event in events:
        category = event.get("cat")
        if category in counts:
            counts[category] += 1
            device_microseconds += float(event.get("dur", 0.0))
            continue
        name = str(event.get("name", ""))
        if category in SUBMISSION_CATEGORIES and "Launch" in name:
            launch_names[name] = launch_names.get(name, 0) + 1
    return {
        "kernel_count": counts["kernel"],
        "device_op_count": sum(counts.values()),
        "device_time_ms": device_microseconds / 1e3,
        "launch_api_calls": sum(launch_names.values()),
        "launch_api_names": launch_names,
        **_key_average_totals(profiler),
    }


def _run_cell(
    mode: str,
    schedule: dict[str, int],
    config: PpoConfig,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Time and profile one minibatch-size x compile-mode cell."""
    torch._dynamo.reset()
    torch.manual_seed(4242)
    rows = schedule["effective_minibatch_size"]
    actor_minibatches = schedule["actor_minibatches"]
    critic_minibatches = schedule["critic_minibatches"]
    batch = _staged_minibatch(rows, device)
    actor = FarmActor().to(device)
    critic = DistributionalCritic().to(device)
    actor.train()
    critic.train()
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    resolved = _device_compile_mode(mode, device)
    actor_terms = _cached_update_callable(
        actor, "_sweep_actor_terms", _actor_minibatch_terms, resolved
    )
    critic_loss = _cached_update_callable(
        critic, "_sweep_critic_loss", _critic_minibatch_loss, resolved
    )
    autocast = config.use_bfloat16
    actor_args = (batch["board"], batch["global_features"], batch["units"], batch["unit_positions"])
    critic_args = (batch["board"], batch["critic_features"])

    def actor_step() -> None:
        actor_optimizer.zero_grad(set_to_none=True)
        policy_sum, _entropy_sum, _kl_sum, _clipped = actor_terms(
            actor,
            batch["unit_actions"],
            batch["market_kinds"],
            batch["market_quantities"],
            batch["unit_masks"],
            batch["kind_masks"],
            batch["quantity_masks"],
            batch["unit_active"],
            batch["kind_active"],
            batch["quantity_active"],
            batch["old_unit"],
            batch["old_kind"],
            batch["old_quantity"],
            batch["advantages"],
            config.clip_low,
            config.clip_high,
            autocast,
            *actor_args,
        )
        (-policy_sum / rows).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_gradient_norm)
        _optimizer_step(actor_optimizer, config.actor_learning_rate, config.lr_warmup_steps)

    def critic_step() -> None:
        critic_optimizer.zero_grad(set_to_none=True)
        value_loss, _predicted = critic_loss(critic, batch["value_targets"], autocast, *critic_args)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_gradient_norm)
        _optimizer_step(critic_optimizer, config.critic_learning_rate, config.lr_warmup_steps)

    def one_schedule() -> None:
        # The actor participates only in the first `epochs` of the critic's
        # epochs, and the two graphs interleave inside those, exactly as the
        # minibatch loop in `update_ppo` runs them.
        for index in range(critic_minibatches):
            if index < actor_minibatches:
                actor_step()
            critic_step()

    # First call of each graph: the compile cost, kept out of the steady clock.
    compile_started = time.perf_counter()
    actor_step()
    critic_step()
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - compile_started

    # Whole discarded schedules, so CUDA graph trees finish warming up and the
    # caching allocator reaches its steady footprint before anything is read.
    for _ in range(args.warmup_repeats):
        one_schedule()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        one_schedule()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    # Profiled last: Kineto adds host overhead, and it must not land inside a
    # timed repeat.
    actor_profile = _profile_minibatch(actor_step, device)
    critic_profile = _profile_minibatch(critic_step, device)

    wall = statistics.median(samples)
    device_seconds = (
        actor_minibatches * actor_profile["device_time_ms"]
        + critic_minibatches * critic_profile["device_time_ms"]
    ) / 1e3
    return {
        "oom": False,
        "update_seconds_median": wall,
        "update_seconds_min": min(samples),
        "update_seconds": samples,
        "compile_seconds": compile_seconds,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_allocated / 1024**3,
        "peak_reserved_gib": peak_reserved / 1024**3,
        "actor_minibatch_profile": actor_profile,
        "critic_minibatch_profile": critic_profile,
        "schedule_kernel_count": (
            actor_minibatches * actor_profile["kernel_count"]
            + critic_minibatches * critic_profile["kernel_count"]
        ),
        "schedule_launch_api_calls": (
            actor_minibatches * actor_profile["launch_api_calls"]
            + critic_minibatches * critic_profile["launch_api_calls"]
        ),
        "schedule_device_seconds": device_seconds,
        "launch_bound_gap_ratio": wall / device_seconds if device_seconds > 0 else None,
        "seconds_per_1000_states": wall * 1e3 / schedule["states_processed"],
    }


def _release(device: torch.device) -> None:
    """Return a finished or failed cell's memory before the next one starts."""
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _is_out_of_memory(message: str) -> bool:
    lowered = message.lower()
    return "out of memory" in lowered or "outofmemoryerror" in lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--minibatch-sizes", default="2048,4096,8192,16384")
    parser.add_argument("--compile-modes", default="default,reduce-overhead")
    parser.add_argument("--states", type=int, default=PRODUCTION_STATES)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the update phase is only meaningful on the accelerator")
    torch.set_float32_matmul_precision("high")

    sizes = [int(item) for item in args.minibatch_sizes.split(",") if item]
    modes = [item for item in args.compile_modes.split(",") if item]
    for mode in modes:
        if mode not in UPDATE_COMPILE_MODES:
            raise ValueError(f"unknown update compile mode {mode!r}")

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "states": args.states,
        "epochs": args.epochs,
        "critic_epochs": args.critic_epochs,
        "repeats": args.repeats,
        "warmup_repeats": args.warmup_repeats,
        "use_bfloat16": PpoConfig().use_bfloat16,
        "cells": {},
    }
    for size in sizes:
        config = PpoConfig(
            epochs=args.epochs, critic_epochs=args.critic_epochs, minibatch_size=size
        )
        schedule = _cell_schedule(args.states, size, args.epochs, args.critic_epochs)
        for mode in modes:
            label = f"{size}/{mode}"
            print(f"--- {label} {json.dumps(schedule, sort_keys=True)}", flush=True)
            try:
                cell = _run_cell(mode, schedule, config, device, args)
            except Exception as error:
                message = repr(error)
                cell = {"oom": _is_out_of_memory(message), "error": message[:4000]}
            _release(device)
            cell.update(schedule)
            cell["compile_mode"] = mode
            report["cells"][label] = cell
            if cell.get("error"):
                print(f"{label:24s} FAILED oom={cell['oom']} {cell['error'][:200]}", flush=True)
            else:
                print(
                    f"{label:24s} update={cell['update_seconds_median']:7.3f} s"
                    f"  peak_resv={cell['peak_reserved_gib']:6.2f} GiB"
                    f"  gap={cell['launch_bound_gap_ratio']:5.2f}x"
                    f"  kernels={cell['schedule_kernel_count']:,}"
                    f"  compile={cell['compile_seconds']:7.2f} s",
                    flush=True,
                )

    print("\nminibatch(effective) x mode -> wall / peak reserved / launch-bound gap")
    for label, cell in report["cells"].items():
        if cell.get("error"):
            print(f"  {label:24s} oom={cell['oom']}")
            continue
        print(
            f"  {label:24s} n={cell['effective_minibatch_size']:6d}"
            f"  batches={cell['actor_minibatches']:4d}/{cell['critic_minibatches']:4d}"
            f"  {cell['update_seconds_median']:7.3f} s"
            f"  {cell['peak_reserved_gib']:6.2f} GiB"
            f"  {cell['launch_bound_gap_ratio']:5.2f}x"
            f"  {cell['seconds_per_1000_states']:6.4f} s/1k"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
