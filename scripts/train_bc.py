#!/usr/bin/env python3
"""Behavior-clone an actor on a projected demonstration dataset.

Trains a registered actor family with the exact factored masked likelihood
VAPO optimizes — `component_selected_logprobs` over teacher-forced masks —
with demonstrated selections as targets. The stored raw observations are
retokenized per architecture, and batches are staged through the same
helpers as the VAPO update, so every family clones the same projected
episodes with identical likelihood semantics. Whole seeds are held out
(steps within an episode are nearly duplicates), the best-holdout weights
are kept, and the output is a standard architecture-tagged actor artifact
that `CheckpointAgent`, `evaluate_checkpoint.py`, and RL warm-starting all
consume directly.

GPU work: queue through mlq.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.encoding import encode_observation
from kaggriculture.inference import ACTOR_ARTIFACT_FORMAT_VERSION
from kaggriculture.model import FarmActor
from kaggriculture.modelargs import add_model_config_arguments, model_config_from_args
from kaggriculture.policy import component_logprobs, component_selected_logprobs
from kaggriculture.provenance import source_identity
from kaggriculture.registry import (
    ARCHITECTURES,
    CONV_ENTITY,
    STRUCTURED,
    architecture_of_config,
    resolve_architecture,
)
from kaggriculture.structured import StructuredActor
from kaggriculture.telemetry import TensorboardMirror
from kaggriculture.tokens import encode_structured_observation
from kaggriculture.training import write_checkpoint
from kaggriculture.vapo import _actor_batch_args, _balanced_minibatch_slices, _batch_tensor

SUPPORTED_DATASET_FORMAT_VERSIONS = frozenset((1,))

_FACTOR_FIELDS = (
    "unit_actions",
    "market_kinds",
    "market_quantities",
    "unit_masks",
    "market_kind_masks",
    "market_quantity_masks",
    "unit_active",
    "market_active",
    "market_quantity_active",
)

# Actor-input state fields per family; the structured critic-only extras are
# irrelevant here because behavior cloning trains the actor alone.
_STRUCTURED_STATE_FIELDS = (
    "tile_categorical",
    "tile_continuous",
    "unit_categorical",
    "unit_continuous",
    "unit_tile_gather",
    "unit_tile_gather_valid",
    "products",
    "crops",
    "farms",
    "town",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="extract_bc_dataset output dir")
    parser.add_argument("--output", type=Path, required=True, help="run directory to create")
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=CONV_ENTITY,
        help="actor family to clone; the same episodes are retokenized per family",
    )
    add_model_config_arguments(parser)
    parser.add_argument(
        "--holdout-seeds", type=int, default=12, help="highest N seeds held out entirely"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help=(
            "passes over the corpus; an epoch is a pass, not a fixed step count, "
            "so a larger corpus needs fewer of them, not more"
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="epochs without holdout improvement before stopping"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--encode-workers", type=int, default=8, help="processes for observation encoding"
    )
    return parser.parse_args()


@dataclass
class DemonstrationTensors:
    """Whole-dataset tensors staged in host memory.

    Storage dtypes mirror the rollout staging path (fp16 features, int8/bool
    factors); minibatches cast to compute dtypes through the same batching
    helpers as the VAPO update.

    The corpus stays on the host and only the minibatch crosses to the
    accelerator. Staged on the device instead, dataset size and batch size
    compete for the same memory: the conv-entity encoding is ~10 MiB per
    episode-seat, so a thousand-seat corpus reserves ~10 GiB before a single
    activation is allocated, and the clone's memory ceiling becomes a limit on
    how much data it may learn from rather than on how wide a batch it may
    take. The gather and transfer cost a few percent of a step whose forward
    and backward dominate.
    """

    staged: dict[str, torch.Tensor]
    # Active components per row, kept on the host. The clone loss is a mean
    # over active components, so an epoch average must weight by that count;
    # reducing the device masks per minibatch would sync the accelerator once
    # per optimizer step for a value that never changes.
    row_components: np.ndarray

    @property
    def rows(self) -> int:
        return self.staged["unit_actions"].shape[0]


def _encode_episode_file(path_text: str, architecture_name: str) -> dict[str, np.ndarray]:
    """Encode one episode-seat archive's raw observations into model inputs."""
    with np.load(path_text) as archive:
        raw = json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()))
        arrays = {name: archive[name] for name in _FACTOR_FIELDS}
    if architecture_name == CONV_ENTITY:
        encoded = [
            encode_observation(entry["observation"], entry["opponent_private"])
            for entry in raw["observations"]
        ]
        arrays["board"] = np.stack([row.board for row in encoded]).astype(np.float16)
        arrays["global_features"] = np.stack([row.global_features for row in encoded]).astype(
            np.float16
        )
        arrays["units"] = np.stack([row.units for row in encoded]).astype(np.float16)
        arrays["unit_positions"] = np.stack([row.unit_positions for row in encoded]).astype(np.int8)
    elif architecture_name == STRUCTURED:
        encoded = [
            encode_structured_observation(entry["observation"], entry["opponent_private"])
            for entry in raw["observations"]
        ]
        for name in _STRUCTURED_STATE_FIELDS:
            arrays[name] = np.stack([getattr(row, name) for row in encoded])
    else:
        raise ValueError(f"architecture {architecture_name!r} has no demonstration tokenizer")
    encoding_active = np.stack([row.unit_active for row in encoded]).astype(bool)
    if not np.array_equal(encoding_active, arrays["unit_active"].astype(bool)):
        raise ValueError(f"{path_text}: projection and encoding disagree on active units")
    return arrays


def _validate_targets_satisfy_masks(arrays: dict[str, np.ndarray], name: str) -> None:
    for factor, mask, active in (
        ("unit_actions", "unit_masks", "unit_active"),
        ("market_kinds", "market_kind_masks", "market_active"),
        ("market_quantities", "market_quantity_masks", "market_quantity_active"),
    ):
        selected = np.take_along_axis(
            arrays[mask], arrays[factor][..., None].astype(np.int64), axis=-1
        )[..., 0]
        if not selected[arrays[active].astype(bool)].all():
            raise ValueError(f"{name}: demonstrated {factor} violate their own masks")


def load_dataset(
    dataset_dir: Path,
    *,
    architecture: str,
    holdout_seeds: int,
    encode_workers: int,
) -> tuple[DemonstrationTensors, DemonstrationTensors, dict[str, Any]]:
    """Load, encode, and stage the dataset; returns (train, holdout, manifest)."""
    # Reject an unknown family before paying for the encode, not inside a
    # worker process after every episode has been tokenized.
    resolve_architecture(architecture)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("format_version")
    if version not in SUPPORTED_DATASET_FORMAT_VERSIONS:
        raise ValueError(f"unsupported dataset format: {version}")
    episodes = manifest["episodes"]
    if not episodes:
        raise ValueError("dataset manifest lists no episodes")
    seeds = sorted({int(entry["seed"]) for entry in episodes})
    if not 0 < holdout_seeds < len(seeds):
        raise ValueError(
            f"holdout of {holdout_seeds} seeds needs 1..{len(seeds) - 1} with {len(seeds)} seeds"
        )
    held_out = set(seeds[-holdout_seeds:])

    paths = [str(dataset_dir / entry["file"]) for entry in episodes]
    encode = partial(_encode_episode_file, architecture_name=architecture)
    if encode_workers > 1:
        with ProcessPoolExecutor(max_workers=encode_workers) as pool:
            encoded = list(pool.map(encode, paths, chunksize=1))
    else:
        encoded = [encode(path) for path in paths]

    splits: dict[bool, list[dict[str, np.ndarray]]] = {False: [], True: []}
    for entry, arrays in zip(episodes, encoded, strict=True):
        _validate_targets_satisfy_masks(arrays, entry["file"])
        splits[int(entry["seed"]) in held_out].append(arrays)

    def stage(members: list[dict[str, np.ndarray]]) -> DemonstrationTensors:
        stacked = {
            name: np.concatenate([arrays[name] for arrays in members]) for name in members[0]
        }
        components = sum(
            stacked[name].astype(np.float64).sum(axis=1)
            for name in ("unit_active", "market_active", "market_quantity_active")
        )
        return DemonstrationTensors(
            staged={name: torch.from_numpy(value) for name, value in stacked.items()},
            row_components=components,
        )

    return stage(splits[False]), stage(splits[True]), manifest


def _batch(
    architecture: str,
    tensors: DemonstrationTensors,
    indices: torch.Tensor | slice,
    device: torch.device,
) -> tuple[tuple[Any, ...], dict[str, torch.Tensor]]:
    """One minibatch of actor forward arguments plus teacher-forced factors.

    The gather runs on the host, where the corpus lives, and moves the narrow
    storage dtypes; the widening casts to the compute dtypes then run on the
    accelerator through the same helpers the VAPO update uses, so the bus
    carries fp16 and int8 rather than the fp32 and int64 they become.
    """
    rows = {
        name: _batch_tensor(value, indices).to(device, non_blocking=True)
        for name, value in tensors.staged.items()
    }
    whole = slice(None)
    actor_args = _actor_batch_args(architecture, rows, whole)
    factors = {
        "unit_actions": _batch_tensor(rows["unit_actions"], whole, torch.long),
        "market_kinds": _batch_tensor(rows["market_kinds"], whole, torch.long),
        "market_quantities": _batch_tensor(rows["market_quantities"], whole, torch.long),
        "unit_masks": rows["unit_masks"],
        "market_kind_masks": rows["market_kind_masks"],
        "market_quantity_masks": rows["market_quantity_masks"],
        "unit_active": rows["unit_active"],
        "market_active": rows["market_active"],
        "market_quantity_active": rows["market_quantity_active"],
    }
    return actor_args, factors


def _masked_mean(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return (values * active).sum() / active.sum().clamp(min=1)


def _clone_loss(
    actor: FarmActor | StructuredActor,
    actor_args: tuple[Any, ...],
    factors: dict[str, torch.Tensor],
    autocast: bool,
) -> torch.Tensor:
    """Negative mean log-likelihood over active factored components."""
    with torch.autocast(
        device_type=factors["unit_actions"].device.type, dtype=torch.bfloat16, enabled=autocast
    ):
        output = actor(*actor_args)
        unit_logprob, kind_logprob, quantity_logprob = component_selected_logprobs(
            output,
            actor.quantity_logits(output.market_quantity_context, factors["market_kinds"]),
            factors["unit_actions"],
            factors["market_kinds"],
            factors["market_quantities"],
            factors["unit_masks"],
            factors["market_kind_masks"],
            factors["market_quantity_masks"],
            validate_masks=False,
        )
    active = torch.cat(
        (factors["unit_active"], factors["market_active"], factors["market_quantity_active"]),
        dim=1,
    )
    logprobs = torch.cat((unit_logprob, kind_logprob, quantity_logprob), dim=1)
    return -_masked_mean(logprobs, active)


@torch.no_grad()
def evaluate(
    architecture: str,
    actor: FarmActor | StructuredActor,
    tensors: DemonstrationTensors,
    *,
    batch_size: int,
    device: torch.device,
    autocast: bool,
) -> dict[str, float]:
    """Per-head masked NLL, top-1 accuracy, and entropy on one split."""
    actor.eval()
    sums = {name: 0.0 for name in ("unit", "kind", "quantity")}
    hits = dict(sums)
    entropies = dict(sums)
    counts = dict(sums)
    for start in range(0, tensors.rows, batch_size):
        indices = slice(start, min(start + batch_size, tensors.rows))
        actor_args, factors = _batch(architecture, tensors, indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            output = actor(*actor_args)
            quantity_logits = actor.quantity_logits(
                output.market_quantity_context, factors["market_kinds"]
            )
            logprobs_entropies = component_logprobs(
                output,
                quantity_logits,
                factors["unit_actions"],
                factors["market_kinds"],
                factors["market_quantities"],
                factors["unit_masks"],
                factors["market_kind_masks"],
                factors["market_quantity_masks"],
                validate_masks=False,
            )
        heads = {
            "unit": (
                logprobs_entropies[0],
                logprobs_entropies[3],
                output.unit_logits,
                factors["unit_masks"],
                factors["unit_actions"],
                factors["unit_active"],
            ),
            "kind": (
                logprobs_entropies[1],
                logprobs_entropies[4],
                output.market_kind_logits,
                factors["market_kind_masks"],
                factors["market_kinds"],
                factors["market_active"],
            ),
            "quantity": (
                logprobs_entropies[2],
                logprobs_entropies[5],
                quantity_logits,
                factors["market_quantity_masks"],
                factors["market_quantities"],
                factors["market_quantity_active"],
            ),
        }
        for name, (logprob, entropy, logits, mask, target, active) in heads.items():
            greedy = logits.float().masked_fill(~mask, -torch.inf).argmax(dim=-1)
            sums[name] += float(-(logprob * active).sum())
            hits[name] += float(((greedy == target) & active).sum())
            entropies[name] += float((entropy * active).sum())
            counts[name] += float(active.sum())
    metrics: dict[str, float] = {}
    for name in sums:
        count = max(counts[name], 1.0)
        metrics[f"{name}_nll"] = sums[name] / count
        metrics[f"{name}_accuracy"] = hits[name] / count
        metrics[f"{name}_entropy"] = entropies[name] / count
    metrics["nll"] = sum(sums.values()) / max(sum(counts.values()), 1.0)
    return metrics


def _artifact_payload(
    architecture: str,
    actor: FarmActor | StructuredActor,
    config: Any,
    metrics: dict[str, float],
    bc_provenance: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
        "architecture": architecture,
        "model_config": config.to_dict(),
        "actor": {name: value.cpu() for name, value in actor.state_dict().items()},
        "iteration": 0,
        "metrics": metrics,
        # Bound once at launch, like every other entry point: re-hashing the
        # tree per improving epoch would tag the weights with a source that
        # may have changed since they were trained.
        "source_identity": identity,
        "run_provenance": None,
        "bc_provenance": bc_provenance,
    }


def train(
    *,
    dataset_dir: Path,
    output_dir: Path,
    architecture: str,
    config: Any,
    holdout_seeds: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    encode_workers: int,
) -> dict[str, float]:
    """Run the full clone; returns the best holdout metrics."""
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    if architecture_of_config(config).name != architecture:
        raise ValueError(
            f"{type(config).__name__} does not configure the {architecture} architecture"
        )
    artifact_path = output_dir / "bc-actor.pt"
    metrics_path = output_dir / "metrics.jsonl"
    # A second clone into a populated directory would overwrite an artifact
    # that may be better than anything this run produces, and truncate the
    # journal that is the only record of how it was produced.
    existing = [path for path in (artifact_path, metrics_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to clone into a directory that already holds "
            f"{', '.join(path.name for path in existing)}: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    train_split, holdout_split, manifest = load_dataset(
        dataset_dir,
        architecture=architecture,
        holdout_seeds=holdout_seeds,
        encode_workers=encode_workers,
    )
    manifest_digest = hashlib.sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest()
    print(
        f"dataset: {train_split.rows} train rows, {holdout_split.rows} holdout rows "
        f"({holdout_seeds} held-out seeds)",
        flush=True,
    )

    actor = resolve_architecture(architecture).actor_class(config).to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(train_split.rows / batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    autocast = device.type == "cuda"
    bc_provenance = {
        "dataset_dir": str(dataset_dir.resolve()),
        "manifest_sha256": manifest_digest,
        "teacher": manifest["teacher"],
        "opponent": manifest["opponent"],
        "architecture": architecture,
        "holdout_seeds": holdout_seeds,
        "command": sys.argv,
    }
    identity = source_identity()

    best = math.inf
    best_metrics: dict[str, float] = {}
    stale = 0
    # The journal stays the durable append-only record the mirror rebuilds
    # from, and TensorBoard is written live beside it, exactly as VAPO
    # training does. A clone that only journals is one nobody watches: its
    # curves appear after a conversion step that has to be remembered.
    writer = TensorboardMirror(metrics_path, output_dir / "tensorboard")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(epochs):
            actor.train()
            started = time.perf_counter()
            # The permutation indexes host storage and also selects the epoch
            # weights from precomputed host-side counts, which needs the same
            # order.
            order = torch.randperm(train_split.rows, generator=generator)
            shuffled_components = train_split.row_components[order.numpy()]
            # The applied learning rate, captured before the first step of this
            # epoch advances the cosine schedule past it.
            applied_learning_rate = optimizer.param_groups[0]["lr"]
            epoch_loss = 0.0
            epoch_components = 0.0
            for indices in _balanced_minibatch_slices(train_split.rows, batch_size):
                actor_args, factors = _batch(architecture, train_split, order[indices], device)
                loss = _clone_loss(actor, actor_args, factors, autocast)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite clone loss in epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
                # The loss is a mean over active components, so the epoch
                # average must weight by that same count, exactly as the VAPO
                # update aggregates its per-minibatch losses.
                components = float(shuffled_components[indices].sum())
                epoch_loss += float(loss.detach()) * components
                epoch_components += components
            holdout = evaluate(
                architecture,
                actor,
                holdout_split,
                batch_size=batch_size,
                device=device,
                autocast=autocast,
            )
            record = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(epoch_components, 1.0),
                "learning_rate": applied_learning_rate,
                "seconds": time.perf_counter() - started,
                **{f"holdout_{name}": value for name, value in holdout.items()},
            }
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            writer.record(record)
            print(
                f"epoch {epoch}: train {record['train_loss']:.4f} "
                f"holdout {holdout['nll']:.4f} "
                f"acc unit {holdout['unit_accuracy']:.3f} "
                f"kind {holdout['kind_accuracy']:.3f} "
                f"quantity {holdout['quantity_accuracy']:.3f}",
                flush=True,
            )
            if holdout["nll"] < best:
                best, best_metrics, stale = holdout["nll"], holdout, 0
                # Atomic: a kill mid-save must not destroy the best artifact so
                # far, which the rerun guard would then refuse to replace.
                write_checkpoint(
                    artifact_path,
                    _artifact_payload(
                        architecture, actor, config, holdout, bc_provenance, identity
                    ),
                )
            else:
                stale += 1
                if stale >= patience:
                    print(f"stopping: no holdout improvement in {patience} epochs", flush=True)
                    break
    writer.close()
    if not best_metrics:
        raise RuntimeError("training produced no holdout evaluation")
    print(f"best holdout nll {best:.4f}; artifact at {artifact_path}", flush=True)
    return best_metrics


def main() -> None:
    args = parse_args()
    train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        architecture=args.architecture,
        config=model_config_from_args(resolve_architecture(args.architecture), args),
        holdout_seeds=args.holdout_seeds,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=torch.device(args.device),
        encode_workers=args.encode_workers,
    )


if __name__ == "__main__":
    main()
