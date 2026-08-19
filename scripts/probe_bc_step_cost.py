#!/usr/bin/env python3
"""Where a behaviour-cloning step's wall clock goes, and what the spare VRAM buys.

The shipped trainer runs 407 steps of 2048 rows per epoch in ~66 seconds, which
is 162 ms per step for a 1.42M-parameter model whose forward measures 4.9 ms of
kernel time. Something other than arithmetic owns that step, and this probe
measures which part, at production shapes, on the real corpus.

Three candidates, separated rather than bundled:

* **Host gather.** The corpus lives in host memory and a minibatch is a fancy
  index into it. At batch 2048 that copies ~25 MiB of scattered rows.
* **The transfer.** `_batch` moves those rows with ``non_blocking=True``, which
  is a NO-OP on unpageable memory: the staged corpus is not pinned, so every
  step's copy is synchronous and serialises against compute. This probe measures
  the same transfer out of a pinned staging buffer to price that.
* **Launch overhead.** If the step is launch-bound rather than FLOP-bound, then
  doubling the batch nearly halves the epoch, because the per-step costs are
  paid once instead of twice. The sweep over batch size is what shows it, and
  peak VRAM per batch is reported beside it so the ceiling is a measurement
  rather than a guess.

Reported per cell: seconds per step split into gather / transfer / forward /
backward / optimizer, rows per second, and peak device memory. The interesting
number is rows per second: it is the one that decides an epoch's length.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from kaggriculture.ppo import PpoConfig, make_optimizers
from kaggriculture.registry import resolve_architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--seeds-per-dataset",
        type=int,
        default=16,
        help="seeds staged per corpus; the probe times steps, so it needs rows, not breadth",
    )
    parser.add_argument("--architecture", default="entity-cnn")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(1024, 2048, 4096, 8192),
        help="the shipped value is 2048; the sweep is what prices the spare VRAM",
    )
    parser.add_argument("--steps", type=int, default=24, help="timed steps per cell after warmup")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _stage_pinned(staged: dict[str, torch.Tensor], rows: int) -> dict[str, torch.Tensor]:
    """A pinned host buffer wide enough for one minibatch of every field.

    Pinning the whole corpus is not an option -- it is tens of GiB -- so the
    comparison is the one the rollout path already uses: gather into a pinned
    arena of exactly one batch, then transfer out of that.
    """
    return {
        name: torch.empty((rows, *value.shape[1:]), dtype=value.dtype).pin_memory()
        for name, value in staged.items()
    }


def _time_cell(
    trainer: Any,
    architecture: str,
    tensors: Any,
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    steps: int,
    warmup: int,
    device: torch.device,
    pinned: dict[str, torch.Tensor] | None,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(0)
    totals = {"gather": 0.0, "transfer": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + steps):
        indices = torch.randint(0, tensors.rows, (batch_size,), generator=generator)
        measure = step >= warmup

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        host = {
            name: trainer._batch_tensor(value, indices) for name, value in tensors.staged.items()
        }
        gathered = time.perf_counter()

        if pinned is not None:
            for name, value in host.items():
                pinned[name][: value.shape[0]].copy_(value)
            host = {name: value[:batch_size] for name, value in pinned.items()}
        rows = {name: value.to(device, non_blocking=True) for name, value in host.items()}
        torch.cuda.synchronize(device)
        transferred = time.perf_counter()

        whole = slice(None)
        actor_args = trainer._actor_batch_args(architecture, rows, whole)
        factors = {
            "unit_actions": trainer._batch_tensor(rows["unit_actions"], whole, torch.long),
            "market_kinds": trainer._batch_tensor(rows["market_kinds"], whole, torch.long),
            "market_quantities": trainer._batch_tensor(
                rows["market_quantities"], whole, torch.long
            ),
            "unit_masks": rows["unit_masks"],
            "market_kind_masks": rows["market_kind_masks"],
            "market_quantity_masks": rows["market_quantity_masks"],
            "unit_active": rows["unit_active"],
            "market_active": rows["market_active"],
            "market_quantity_active": rows["market_quantity_active"],
        }
        # The shipped trainer enables bf16 autocast on CUDA unconditionally
        # (`train_bc.py:776`), so timing it disabled would price a configuration
        # nobody runs.
        loss = trainer._clone_loss(actor, actor_args, factors, True)
        torch.cuda.synchronize(device)
        forward = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize(device)
        backward = time.perf_counter()

        optimizer.step()
        torch.cuda.synchronize(device)
        stepped = time.perf_counter()

        if measure:
            totals["gather"] += gathered - started
            totals["transfer"] += transferred - gathered
            totals["forward"] += forward - transferred
            totals["backward"] += backward - forward
            totals["optimizer"] += stepped - backward

    cell = {name: value / steps for name, value in totals.items()}
    cell["seconds_per_step"] = sum(cell.values())
    cell["rows_per_second"] = batch_size / cell["seconds_per_step"]
    cell["peak_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    cell["batch_size"] = batch_size
    cell["pinned"] = pinned is not None
    return cell


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this probe measures a CUDA configuration and needs a GPU")
    device = torch.device("cuda")

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "kaggriculture_train_bc", Path(__file__).resolve().parent / "train_bc.py"
    )
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)

    architecture = resolve_architecture(args.architecture)
    config = architecture.config_class()
    train_split, _, _ = trainer.load_dataset(
        args.dataset,
        architecture=args.architecture,
        holdout_seeds=1,
        seeds_per_dataset=args.seeds_per_dataset,
        encode_workers=2,
    )
    actor = architecture.actor_class(config).to(device)
    optimizer, _ = make_optimizers(actor, architecture.critic_class(config).to(device), PpoConfig())

    cells: list[dict[str, float]] = []
    for batch_size in args.batch_sizes:
        if batch_size > train_split.rows:
            continue
        for pinned_arena in (None, _stage_pinned(train_split.staged, batch_size)):
            try:
                cell = _time_cell(
                    trainer,
                    args.architecture,
                    train_split,
                    actor,
                    optimizer,
                    batch_size=batch_size,
                    steps=args.steps,
                    warmup=args.warmup,
                    device=device,
                    pinned=pinned_arena,
                )
            except torch.cuda.OutOfMemoryError:
                cell = {"batch_size": batch_size, "pinned": pinned_arena is not None, "oom": True}
                torch.cuda.empty_cache()
            cells.append(cell)
            print(json.dumps(cell, sort_keys=True), flush=True)
            del pinned_arena

    report = {
        "probe": "bc-step-cost",
        "command": sys.argv,
        "rows_staged": train_split.rows,
        "architecture": args.architecture,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
