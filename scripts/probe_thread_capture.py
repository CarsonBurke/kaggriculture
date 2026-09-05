"""Ask whether two shard threads can each own and replay a CUDA graph.

The rollout is launch-bound: 1,669 kernel launches a step against 4.7 ms of
actual device work, so the lever is collapsing launches, and the way to collapse
them is capture. Inductor's `reduce-overhead` is the off-the-shelf route and it
does not survive this collector. `torch._inductor.cudagraph_trees` holds a tree
manager per *thread* but keys generations off a process-global counter, so one
shard's `cudagraph_mark_step_begin` retires the other shard's live outputs; the
guard in `rollout._cuda_graph_generation` fixes that specific race, and the mode
then wedges somewhere else with every thread parked on a futex. Two full-wave
probes cost twenty-five minutes of an idle exclusive GPU lease between them and
returned no stack, which is too expensive a way to keep asking.

So ask the underlying question directly instead, with no compiler and no
simulator in the way: two threads, two side streams, one `torch.cuda.CUDAGraph`
each over the real actor forward at the real production shapes, captured under
the `torch.inference_mode` and bf16 autocast the collector actually runs in,
then replayed concurrently. That isolates the CUDA-level feasibility of the
design from `cudagraph_trees`' bookkeeping, and it runs in seconds.

A pass means an explicit per-shard graph is a viable replacement for the mode
that keeps hanging, and reports what one replay costs against one eager
forward. A failure here means capture cannot serve two shards at all and the
launch count has to come down some other way -- which is worth knowing for the
price of one short job rather than another twenty-five minute stall.

Inputs are zero-filled at the shapes `allocate_rollout_storage` defines rather
than encoded from a live wave. The question is mechanical -- does capture hold,
and does replay reproduce the eager result from the same buffers -- and the
equivalence check compares graph against eager on identical inputs, so it is
just as sharp on zeros. Nothing here is evidence about throughput of a real
collection; `probe_rollout_utilization.py` remains the measurement.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import threading
import time
from pathlib import Path

import torch

from kaggriculture.constants import ANIMALS
from kaggriculture.registry import resolve_architecture
from kaggriculture.structured import StructuredConfig, StructuredInputs
from kaggriculture.tokens import ANIMAL_TOKEN_FIELDS, N_TILE_CONTINUOUS

#: Field dtype and trailing shape, as `allocate_rollout_storage("structured")`
#: lays them out. The categorical and flag groups arrive from the simulator
#: narrow and are widened on upload, so the model sees int64 and bool.
_FIELDS: tuple[tuple[str, torch.dtype, tuple[int, ...]], ...] = (
    ("tile_categorical", torch.int64, (200, 6)),
    ("tile_continuous", torch.float32, (200, N_TILE_CONTINUOUS)),
    ("unit_categorical", torch.int64, (16, 4)),
    ("unit_continuous", torch.float32, (16, 14)),
    ("unit_active", torch.bool, (16,)),
    ("unit_tile_gather", torch.int64, (16, 5)),
    ("unit_tile_gather_valid", torch.bool, (16, 5)),
    ("products", torch.float32, (9, 5)),
    ("animals", torch.float32, (len(ANIMALS), len(ANIMAL_TOKEN_FIELDS))),
    ("crops", torch.float32, (5, 6)),
    ("farms", torch.float32, (2, 4)),
    ("town", torch.float32, (14,)),
)


def _static_inputs(rows: int, device: torch.device) -> StructuredInputs:
    """Persistent zero-filled device buffers at one shard's wave shape.

    Zero is a valid index for every categorical embedding and a valid own-farm
    tile for the unit gather, so nothing here can index out of range. Capture
    binds these addresses, which is why they are allocated once and reused.
    """
    fields = {
        # Flags are set rather than cleared: an all-inactive wave masks every
        # unit out and the row softmax over an all-masked row is a NaN, which
        # would make the eager-against-graph comparison below vacuous.
        name: (
            torch.ones((rows, *shape), dtype=dtype, device=device)
            if dtype is torch.bool
            else torch.zeros((rows, *shape), dtype=dtype, device=device)
        )
        for name, dtype, shape in _FIELDS
    }
    return StructuredInputs(**fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=192)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--stall-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    faulthandler.dump_traceback_later(args.stall_timeout, exit=True)
    device = torch.device("cuda")
    torch.manual_seed(20260902)

    architecture = resolve_architecture("structured")
    config = StructuredConfig(
        model_dim=80,
        attention_heads=4,
        attention_kv_heads=2,
        ffn_multiplier=2,
        global_modulation=True,
        fuse_market_decoder=True,
    )
    actors = []
    for _ in range(args.shards):
        actor = architecture.actor_class(config).to(device)
        actor.eval()
        actor.requires_grad_(False)
        actors.append(actor)

    inputs = [_static_inputs(args.rows, device) for _ in range(args.shards)]
    streams = [torch.cuda.Stream(device=device) for _ in range(args.shards)]
    graphs: list[torch.cuda.CUDAGraph] = []
    captured: list[tuple[torch.Tensor, ...]] = []
    expected: list[tuple[torch.Tensor, ...]] = []
    # Capture is a once-per-process event and switches the caching allocator to
    # a private pool, so it is serialised here exactly as production would
    # serialise it. Only replay has to be concurrent, and replay is the part
    # that runs 720 times a collection.
    capture_lock = threading.Lock()
    failures: list[str] = []
    barrier = threading.Barrier(args.shards)

    def forward(shard: int) -> tuple[torch.Tensor, ...]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            return tuple(actors[shard](inputs[shard]))

    def prepare(shard: int) -> None:
        with torch.inference_mode(), torch.cuda.device(device), capture_lock:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    reference = forward(shard)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            expected.append(tuple(tensor.clone().float() for tensor in reference))

            graph = torch.cuda.CUDAGraph()
            # `thread_local` is what lets a peer shard keep issuing on its own
            # stream while this one captures; the default aborts the capture.
            with torch.cuda.graph(graph, stream=streams[shard], capture_error_mode="thread_local"):
                output = forward(shard)
            graphs.append(graph)
            captured.append(tuple(output))

    def replay(shard: int) -> float:
        with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(streams[shard]):
            graphs[shard].replay()
            torch.cuda.current_stream().synchronize()
            barrier.wait()
            started = time.perf_counter()
            for _ in range(args.replays):
                graphs[shard].replay()
            torch.cuda.current_stream().synchronize()
            return time.perf_counter() - started

    def eager(shard: int) -> float:
        with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(streams[shard]):
            forward(shard)
            torch.cuda.current_stream().synchronize()
            barrier.wait()
            started = time.perf_counter()
            for _ in range(args.replays):
                forward(shard)
            torch.cuda.current_stream().synchronize()
            return time.perf_counter() - started

    def run(target, label: str) -> list[float]:
        results: list[float | None] = [None] * args.shards

        def one(shard: int) -> None:
            try:
                results[shard] = target(shard)
            except Exception as error:
                failures.append(f"{label} shard {shard}: {type(error).__name__}: {error}")

        threads = [threading.Thread(target=one, args=(shard,)) for shard in range(args.shards)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return [value for value in results if value is not None]

    # Capture serially first: a deadlock in concurrent replay is the question,
    # and it cannot be told apart from a deadlock in concurrent capture if both
    # happen at once.
    for shard in range(args.shards):
        prepare(shard)

    eager_seconds = run(eager, "eager")
    replay_seconds = run(replay, "replay")

    drift = []
    for shard in range(args.shards):
        for reference, produced in zip(expected[shard], captured[shard], strict=True):
            drift.append(float((reference - produced.float()).abs().max().item()))

    faulthandler.cancel_dump_traceback_later()
    report = {
        "rows": args.rows,
        "shards": args.shards,
        "replays": args.replays,
        "eager_milliseconds_per_forward": (
            1e3 * max(eager_seconds) / args.replays if eager_seconds else None
        ),
        "graph_milliseconds_per_replay": (
            1e3 * max(replay_seconds) / args.replays if replay_seconds else None
        ),
        "max_absolute_drift": max(drift) if drift else None,
        "failures": failures,
    }
    if report["eager_milliseconds_per_forward"] and report["graph_milliseconds_per_replay"]:
        report["speedup"] = (
            report["eager_milliseconds_per_forward"] / report["graph_milliseconds_per_replay"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
