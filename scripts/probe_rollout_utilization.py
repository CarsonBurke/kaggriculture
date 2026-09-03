"""Attribute the production rollout's GPU idle time.

The iteration benchmark reports one rollout number and `nvidia-smi` reports a
duty cycle averaged over a sampling window that is longer than most of our
kernels. Neither says whether the device is idle because the host cannot launch
fast enough or because the host is off doing Rust work. This unions the
the device time the profiler attributes to CUDA kernels against the wall time
of the same wave. Two shard streams mean overlapped kernels are counted twice,
so the reported busy fraction is an upper bound: the device cannot have been
busier than that, which is what makes a low number decisive.

The native simulator only accepts the competition horizon, so this is the
full 720-step production wave -- no shortened proxy. Only GPU activities are
recorded, which keeps the trace to the kernels themselves instead of the far
larger set of host-side operator events. Busy time is reported per CUDA stream
as well as unioned, because the collector runs two shards on two side streams
and the gap between those two numbers is exactly how much the pipelining wins.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from kaggriculture.production import PRODUCTION_ROLLOUT_FORWARD_MODE
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import (
    ROLLOUT_FORWARD_MODES,
    allocate_rollout_storage,
    collect_mixed_play_rust,
)
from kaggriculture.structured import StructuredConfig


@contextmanager
def _stall_guard(seconds: float, what: str) -> Iterator[None]:
    """Abort with every thread's stack if one rollout takes longer than `seconds`.

    A compiled collection can wedge rather than fail -- the shard threads park
    on a futex and the process sits at a few percent CPU until the queue's time
    limit kills it, holding the GPU lease the whole time and leaving nothing
    behind to read. `mlq`'s own limit cannot do better than that because it can
    only send a signal from outside. This can: the stall is bounded by the work
    itself rather than by the whole run, and the dump names the line each thread
    is stuck on, which is the one thing a hang has to tell us. Elevated
    permissions would be needed to attach a sampling profiler to a queued job
    after the fact, so arming the dump in advance is also the only way to get it.
    """
    if seconds <= 0:
        yield
        return
    print(f"stall guard armed: {what} has {seconds:.0f}s", file=sys.stderr, flush=True)
    faulthandler.dump_traceback_later(seconds, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _interval_report(prof: object, output: Path) -> dict[str, float]:
    """Union the kernel intervals and describe the gaps between them.

    Separated from the aggregate path because it is the expensive one: it needs
    every event's absolute timestamp, which only the chrome trace carries.
    """
    trace_path = output.with_suffix(".trace.json")
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())
    events = events["traceEvents"] if isinstance(events, dict) else events
    spans: list[tuple[float, float]] = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        start = float(event["ts"])
        spans.append((start, start + float(event.get("dur", 0.0))))
    del events
    spans.sort()
    busy = 0.0
    gaps: list[float] = []
    span_start, span_end = spans[0]
    for start, end in spans[1:]:
        if start > span_end:
            busy += span_end - span_start
            gaps.append(start - span_end)
            span_start, span_end = start, end
        else:
            span_end = max(span_end, end)
    busy += span_end - span_start
    gaps.sort()
    small = [gap for gap in gaps if gap < 100.0]
    return {
        "unioned_busy_seconds": busy / 1e6,
        "gap_count": len(gaps),
        "gap_total_seconds": sum(gaps) / 1e6,
        "gap_median_microseconds": gaps[len(gaps) // 2] if gaps else 0.0,
        "sub_100us_gap_seconds": sum(small) / 1e6,
        "sub_100us_gap_count": len(small),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-play-games", type=int, default=128)
    parser.add_argument("--league-games", type=int, default=64)
    parser.add_argument("--league-opponents", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument(
        "--forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        default=PRODUCTION_ROLLOUT_FORWARD_MODE,
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "attribute device time per kernel. Off by default: a 720-step wave "
            "produces ~1.2M CUDA events and aggregating them has been "
            "OOM-killed on a loaded host. Wall time per step needs none of it."
        ),
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=180.0,
        help="seconds one rollout may take before dumping every thread and aborting",
    )
    parser.add_argument(
        "--compile-timeout",
        type=float,
        default=600.0,
        help="extra seconds the first warmup wave gets for compiling and recording",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gaps",
        action="store_true",
        help=(
            "Also union the kernel intervals and report the gap distribution. This is what "
            "identifies launch overhead as distinct from host-side work, but it exports and "
            "reparses a chrome trace of every kernel event -- roughly 2.1 GB and about 20 GiB "
            "of resident memory for a 720-step wave, which has been OOM-killed on this "
            "workstation when other jobs were resident. Off by default."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    architecture = resolve_architecture("structured")
    config = StructuredConfig()
    actor = architecture.actor_class(config).to(device)
    frozen = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    opponents = []
    for _ in range(args.league_opponents):
        opponent = architecture.actor_class(config).to(device)
        opponent.load_state_dict(frozen)
        opponent.requires_grad_(False)
        opponents.append(opponent)
    generator = np.random.default_rng(args.seed)
    assignments = np.arange(args.league_games, dtype=np.int64) % len(opponents)
    generator.shuffle(assignments)

    arena = allocate_rollout_storage(
        architecture.name,
        args.self_play_games * 2 + args.league_games,
        args.episode_steps - 1,
        pin_memory=True,
    )

    def one_rollout(seed_start: int):
        return collect_mixed_play_rust(
            actor,
            opponents,
            self_play_games=args.self_play_games,
            league_games=args.league_games,
            opponent_indices=assignments,
            seed_start=seed_start,
            episode_steps=args.episode_steps,
            temperature=1.0,
            opponent_temperature=1.0,
            sampling_seed=int(generator.integers(0, np.iinfo(np.int64).max)),
            forward_mode=args.forward_mode,
            forward_autocast=True,
            storage=arena,
        )

    # Warm up: allocator, cuDNN autotune, and any compilation happen off-trace.
    # The first wave carries every compile and, under a capturing mode, every
    # graph recording, so it gets the compile budget on top of the stall budget.
    for warm in range(2):
        with _stall_guard(
            args.stall_timeout + (args.compile_timeout if not warm else 0.0),
            f"warmup rollout {warm}",
        ):
            one_rollout(args.seed + warm * 10_000)
    torch.cuda.synchronize()
    gc.collect()

    # Wall time is the measurement; kernel attribution is an optional extra that
    # costs gigabytes of host memory to produce. Keeping the timed region
    # identical in both cases means the two can be compared to each other, and
    # the CUDA events bracket the same work either way.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    prof = None
    with ExitStack() as stack:
        if args.profile:
            prof = stack.enter_context(
                profile(
                    activities=[ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_stack=False,
                    profile_memory=False,
                )
            )
        torch.cuda.synchronize()
        start_event.record()
        with _stall_guard(args.stall_timeout, "timed rollout"):
            one_rollout(args.seed + 999_000)
        end_event.record()
        torch.cuda.synchronize()
    wall_seconds = start_event.elapsed_time(end_event) / 1e3

    # A 720-step wave produces enough kernel events that exporting and reparsing
    # a chrome trace OOM-killed this probe once.  aggregates in
    # place instead, which costs nothing and answers the question: it sums each
    # kernel's device time, so with two shard streams that sum counts overlapped
    # work twice and is therefore an UPPER BOUND on how busy the device was.
    # A low upper bound is decisive -- the GPU cannot have been busier than it.
    gap_report: dict[str, float] = {}
    busy = 0.0
    launches = 0
    by_kernel: dict[str, tuple[int, float]] = {}
    if prof is not None:
        if args.gaps:
            gap_report = _interval_report(prof, args.output)
        totals = prof.key_averages()
        rows = []
        for entry in totals:
            device_seconds = getattr(entry, "device_time_total", 0.0) / 1e6
            if device_seconds <= 0.0:
                continue
            busy += device_seconds
            launches += entry.count
            rows.append((entry.key, entry.count, device_seconds))
        rows.sort(key=lambda row: -row[2])
        by_kernel = {name: (count, seconds * 1e6) for name, count, seconds in rows}

    # The attribution keys are absent rather than zeroed when the profiler did
    # not run: a reported busy fraction of 0.0 would read as a measurement, and
    # this probe's whole point is that a low busy fraction is decisive evidence.
    report: dict[str, object] = {
        "forward_mode": args.forward_mode,
        "episode_steps": args.episode_steps,
        "physical_games": args.self_play_games + args.league_games,
        "profiled": bool(args.profile),
        "wall_seconds": wall_seconds,
        "wall_milliseconds_per_step": wall_seconds * 1e3 / args.episode_steps,
    }
    if prof is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    report |= {
        **gap_report,
        "gpu_busy_seconds_upper_bound": busy,
        "gpu_busy_fraction_upper_bound": busy / wall_seconds if wall_seconds else 0.0,
        "gpu_idle_seconds_lower_bound": wall_seconds - busy,
        "kernel_launches": launches,
        "launches_per_step": launches / args.episode_steps,
        "mean_kernel_microseconds": (busy * 1e6 / launches) if launches else 0.0,
        "top_kernels": [
            {
                "name": name,
                "count": count,
                "total_milliseconds": total / 1e3,
                "share_of_busy": (total / 1e6) / busy if busy else 0.0,
            }
            for name, (count, total) in list(by_kernel.items())[:20]
        ],
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "top_kernels"}, indent=2))
    print("\ntop kernels by device time:")
    for entry in report["top_kernels"][:12]:
        print(
            f"  {entry['total_milliseconds']:9.2f} ms  {entry['share_of_busy']:6.1%}  "
            f"x{entry['count']:<7d} {entry['name'][:78]}"
        )


if __name__ == "__main__":
    main()
