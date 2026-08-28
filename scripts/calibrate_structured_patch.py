#!/usr/bin/env python3
"""Calibrate structured patch loss against clone trunk-gradient scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from train_bc import (
    _balanced_minibatch_slices,
    _batch,
    _clone_and_structured_loss,
    _run_blocks,
    _run_epoch_order,
    load_dataset,
)

from kaggriculture.modelargs import add_model_config_arguments, model_config_from_args
from kaggriculture.provenance import source_identity
from kaggriculture.registry import STRUCTURED, resolve_architecture
from kaggriculture.structured import StructuredActor
from kaggriculture.structured_dynamics import StructuredDynamics

_CANDIDATES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
_TARGET_RATIO = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-seeds", type=int, default=8)
    parser.add_argument("--seeds-per-dataset", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--run-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encode-workers", type=int, default=2)
    add_model_config_arguments(parser)
    return parser.parse_args()


def _gradient_norm(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...], *, retain_graph: bool
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = sum(
        (gradient.float().square().sum() for gradient in gradients if gradient is not None),
        start=torch.zeros((), device=loss.device),
    )
    return float(squared.sqrt())


def main() -> None:
    args = parse_args()
    if args.run_length <= 1:
        raise ValueError("patch calibration needs --run-length above one")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {args.output}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    architecture = resolve_architecture(STRUCTURED)
    config = model_config_from_args(architecture, args)
    train_split, _, datasets = load_dataset(
        args.dataset,
        architecture=STRUCTURED,
        holdout_seeds=args.holdout_seeds,
        encode_workers=args.encode_workers,
        seeds_per_dataset=args.seeds_per_dataset,
    )
    starts, lengths = _run_blocks(train_split.staged["episode_index"], args.run_length)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    order = _run_epoch_order(starts, lengths, generator)
    first_slice = _balanced_minibatch_slices(train_split.rows, args.batch_size)[0]
    batch_indices = order[first_slice]
    actor_args, factors = _batch(STRUCTURED, train_split, batch_indices, device)

    actor = StructuredActor(config).to(device)
    dynamics = StructuredDynamics(config).to(device)
    terms = _clone_and_structured_loss(
        actor,
        dynamics,
        actor_args,
        factors,
        device.type == "cuda",
        0,
        1,
        False,
        False,
        False,
    )
    trunk_parameters = tuple(
        parameter for parameter in actor.trunk.parameters() if parameter.requires_grad
    )
    clone_norm = _gradient_norm(terms.clone, trunk_parameters, retain_graph=True)
    patch_norm = _gradient_norm(terms.patch, trunk_parameters, retain_graph=False)
    if not math.isfinite(clone_norm) or clone_norm <= 0:
        raise RuntimeError(f"invalid clone trunk-gradient norm: {clone_norm}")
    if not math.isfinite(patch_norm) or patch_norm <= 0:
        raise RuntimeError(f"invalid patch trunk-gradient norm: {patch_norm}")

    ratios = {
        str(coefficient): coefficient * patch_norm / clone_norm for coefficient in _CANDIDATES
    }
    eligible = [
        coefficient for coefficient in _CANDIDATES if 0.1 <= ratios[str(coefficient)] <= 0.3
    ]
    if not eligible:
        raise RuntimeError(f"no calibrated coefficient reaches the required 10-30% range: {ratios}")
    selected = min(eligible, key=lambda coefficient: abs(ratios[str(coefficient)] - _TARGET_RATIO))
    index_bytes = np.ascontiguousarray(batch_indices.numpy()).tobytes()
    payload = {
        "source_identity": source_identity(),
        "datasets": datasets,
        "model_config": config.to_dict(),
        "seed": args.seed,
        "run_length": args.run_length,
        "batch_size": int(batch_indices.numel()),
        "batch_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "batch_index_min": int(batch_indices.min()),
        "batch_index_max": int(batch_indices.max()),
        "clone_trunk_gradient_norm": clone_norm,
        "patch_trunk_gradient_norm": patch_norm,
        "candidate_weighted_gradient_ratios": ratios,
        "selected_patch_coefficient": selected,
        "selected_weighted_gradient_ratio": ratios[str(selected)],
        "raw_patch_loss": float(terms.patch.detach()),
        "raw_patch_all_loss": float(terms.patch_all.detach()),
        "raw_patch_changed_loss": float(terms.patch_changed.detach()),
        "raw_patch_unchanged_loss": float(terms.patch_unchanged.detach()),
        "eligible_pairs": float(terms.eligible.detach()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
