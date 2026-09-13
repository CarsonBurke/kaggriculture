"""Measure the rollout-shaped actor forward across execution backends.

Collection spends roughly two thirds of its wall clock in the actor forward, and
a kernel profile shows around 860 launches whose summed device time is far below
the measured forward. That gap is either CPU launch cost, which a CUDA graph
removes, or per-kernel GPU dispatch latency, which it does not. The rollout used
to compile with the `cudagraphs` backend and gained almost nothing, so the two
explanations had to be separated directly.

Each backend is timed on the identical persistent input buffers the real
collector uses, in fp32 and under bfloat16 autocast:

- `eager`: no compilation, which is `--rollout-forward-mode eager`.
- `cudagraphs`: the only backend the retired `--compile-rollout` boolean could
  select.
- `manual_graph`: an explicit `torch.cuda.CUDAGraph` capture and replay, which
  is the floor for launch-overhead removal with no compiler involvement.
- `inductor_reduce_overhead`: Inductor fusion plus its own graph capture, which
  reduces kernel count rather than only launch cost.

`manual_graph` beating `cudagraphs` means the compiled path is not actually
replaying a graph. Both landing near `eager` means the forward is limited by
kernel latency and only fusion or a smaller network will help.

That is what happened, and it is why the collection backend is now a named mode
rather than a flag. The isolated historical ranking remains useful for backend
diagnosis, but production now selects the collector-owned whole-wave `graph`
mode from end-to-end mixed-rollout evidence. Native BF16 replicas are measured
separately by `profile_actor_forward.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.model import FarmActor, policy_compile_options
from kaggriculture.rollout import _native_encoded_wave
from kaggriculture.rust_env import load_native

BACKENDS = ("eager", "cudagraphs", "manual_graph", "inductor_reduce_overhead")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=112)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=719)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _time(run: Callable[[], Any], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def _build(backend: str, actor: FarmActor, inputs, autocast: bool) -> Callable[[], Any]:
    def call() -> Any:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return actor(*inputs)

    if backend == "eager":
        return call
    if backend == "cudagraphs":
        compiled = torch.compile(actor.forward, backend="cudagraphs", fullgraph=True, dynamic=False)

        def run_compiled() -> Any:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                return compiled(*inputs)

        return run_compiled
    if backend == "inductor_reduce_overhead":
        compiled = torch.compile(
            actor.forward,
            options=policy_compile_options("reduce-overhead"),
            fullgraph=True,
            dynamic=False,
        )

        def run_inductor() -> Any:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                return compiled(*inputs)

        return run_inductor

    # Manual capture. Warm up on a side stream first so cuBLAS/cuDNN finish
    # their own lazy allocations outside the capture, then record once.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph.replay


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    seeds = np.arange(args.seed, args.seed + args.games, dtype=np.uint64)
    environment = load_native().BatchEnv(seeds)
    actor = FarmActor().to("cuda").eval()
    wave = _native_encoded_wave(environment, torch.device("cuda"))
    wave.refresh(environment)
    wave.copy_to_device()
    inputs = wave.inputs()

    report: dict[str, Any] = {"games": args.games, "rows": args.games * 2, "iters": args.iters}
    results: dict[str, dict[str, float]] = {}
    for precision, autocast in (("float32", False), ("bfloat16", True)):
        results[precision] = {}
        for backend in BACKENDS:
            torch._dynamo.reset()
            try:
                with torch.inference_mode(backend != "manual_graph"):
                    run = _build(backend, actor, inputs, autocast)
                    results[precision][backend] = _time(run, args.iters, args.warmup)
            except Exception as error:  # a backend failing is itself a result
                results[precision][backend] = float("nan")
                report[f"error::{precision}::{backend}"] = f"{type(error).__name__}: {error}"
    report["forward_milliseconds_median"] = results
    report["projected_forward_seconds"] = {
        precision: {backend: value * args.horizon / 1e3 for backend, value in backends.items()}
        for precision, backends in results.items()
    }
    baseline = results["float32"]["eager"]
    report["speedup_over_eager_fp32"] = {
        precision: {backend: baseline / value for backend, value in backends.items()}
        for precision, backends in results.items()
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
