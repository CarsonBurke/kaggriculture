#!/usr/bin/env python3
"""Behavior-clone an actor on a projected demonstration dataset.

Trains a registered actor family with the exact factored masked likelihood
PPO optimizes — `component_selected_logprobs` over teacher-forced masks —
with demonstrated selections as targets. The stored raw observations are
retokenized per architecture, and batches are staged through the same
helpers as the PPO update, so every family clones the same projected
episodes with identical likelihood semantics. Whole seeds are held out
(steps within an episode are nearly duplicates), the best-holdout weights
are kept, and the output is a standard architecture-tagged actor artifact
that `CheckpointAgent`, `evaluate_checkpoint.py`, and RL warm-starting all
consume directly.

Several dataset directories are merged into one corpus, each holding out its
own highest seeds. A clone trained on one opponent alone earned 92 money
against `starter` while earning ~28.9k against a copy of itself, having
learned to gate farming on "opponent is rich" — constant in that corpus, so
the demonstrations have to vary it.

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
from collections.abc import Sequence
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
from kaggriculture.optim import NorMuon, route_parameters
from kaggriculture.policy import component_logprobs, component_selected_logprobs
from kaggriculture.ppo import _actor_batch_args, _balanced_minibatch_slices, _batch_tensor
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
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "one or more extract_bc_dataset output dirs, merged into one corpus; mixing "
            "opponents is what keeps the clone from latching onto a spurious trigger, as a "
            "single-opponent corpus made 'opponent is rich' a constant it could gate farming on"
        ),
    )
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
        "--seeds-per-dataset",
        type=int,
        default=None,
        help=(
            "lowest N seeds to take from each corpus; the whole corpus is staged in "
            "host memory at ~10 MiB per episode-seat, so this is what keeps a wide "
            "mixture affordable. Breadth beats depth here: an uncapped single-opponent "
            "clone reached 99.996% accuracy and still could not act off its own regime"
        ),
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
    # Both rate and decay changed UNITS when this moved off AdamW, so both are
    # renamed: a stale invocation now fails at argparse instead of silently
    # training a tenth as fast with a hundredth of the intended decay.
    parser.add_argument(
        "--matrix-learning-rate",
        type=float,
        default=3e-3,
        help=(
            "NorMuon rate: the fraction of itself a hidden matrix moves per step. "
            "About ten times the Adam rate it replaced, because an Adam step is "
            "per-element and moves a matrix roughly lr*sqrt(fan_in) of itself"
        ),
    )
    parser.add_argument(
        "--adam-learning-rate-ratio",
        type=float,
        default=0.35,
        help=(
            "rate for the gains, biases, embeddings and logit heads NorMuon does "
            "not take, as a multiple of the matrix rate; the reference's own "
            "0.008/0.023"
        ),
    )
    parser.add_argument(
        "--matrix-weight-decay",
        type=float,
        default=1.2,
        help=(
            "cautious decay on hidden matrices; quadratic in the rate, as in the "
            "reference, which is why the coefficient is above one"
        ),
    )
    parser.add_argument(
        "--adam-weight-decay",
        type=float,
        default=0.005,
        help="cautious decay on the parameters Adam takes",
    )
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
    helpers as the PPO update.

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


def _stage_split(members: list[dict[str, np.ndarray]]) -> DemonstrationTensors:
    """Concatenate one split's episodes into whole-corpus arrays.

    Filling a preallocated array and releasing each episode as it is copied,
    rather than `np.concatenate`, is what makes a mixture affordable: the encoding
    is ~10 MiB per episode-seat, and concatenating holds the parts and the whole at
    once, so a 2,048-seat corpus peaked near 40 GiB and was killed where 20 GiB of
    steady state fits. `members` is consumed, and row order is preserved so a
    corpus stages identically however it was built.
    """
    rows = sum(member["unit_actions"].shape[0] for member in members)
    template = members[0]
    stacked = {
        name: np.empty((rows, *value.shape[1:]), dtype=value.dtype)
        for name, value in template.items()
    }
    components = np.zeros(rows, dtype=np.float64)
    offset = 0
    for position, member in enumerate(members):
        span = member["unit_actions"].shape[0]
        for name, value in member.items():
            stacked[name][offset : offset + span] = value
        components[offset : offset + span] = sum(
            member[name].astype(np.float64).sum(axis=1)
            for name in ("unit_active", "market_active", "market_quantity_active")
        )
        offset += span
        members[position] = {}
    return DemonstrationTensors(
        staged={name: torch.from_numpy(value) for name, value in stacked.items()},
        row_components=components,
    )


def load_dataset(
    dataset_dirs: Sequence[Path],
    *,
    architecture: str,
    holdout_seeds: int,
    encode_workers: int,
    seeds_per_dataset: int | None = None,
) -> tuple[DemonstrationTensors, DemonstrationTensors, list[dict[str, Any]]]:
    """Load, encode, and stage a mixture of datasets; returns (train, holdout, records).

    Several directories are merged into one corpus because a single-opponent
    corpus teaches the wrong precondition: the clone of `public-v27` trained
    on v27-vs-v27 games alone earned 92 money against `starter` while earning
    ~28.9k against a copy of itself, having latched onto "opponent is rich" —
    a constant in that corpus — as a condition for farming at all.

    `seeds_per_dataset` caps how many seeds each directory contributes. The whole
    corpus is staged in host memory at ~10 MiB per episode-seat, so it is the knob
    that trades breadth against that ceiling -- and breadth is what wins: the
    uncapped clone reached 99.996% accuracy on the distribution it saw and still
    could not act off it, so a fifth of four opponents beats all of one.
    """
    # Reject an unknown family before paying for the encode, not inside a
    # worker process after every episode has been tokenized.
    resolve_architecture(architecture)
    if not dataset_dirs:
        raise ValueError("no dataset directories given")
    manifests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for directory in dataset_dirs:
        raw = (directory / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        version = manifest.get("format_version")
        if version not in SUPPORTED_DATASET_FORMAT_VERSIONS:
            raise ValueError(f"{directory}: unsupported dataset format: {version}")
        if not manifest["episodes"]:
            raise ValueError(f"{directory}: dataset manifest lists no episodes")
        # A step means one environment decision at a fixed horizon; mixing
        # horizons would silently change what a cloned step is.
        if manifests and manifest["episode_steps"] != manifests[0]["episode_steps"]:
            raise ValueError(
                f"{directory}: episode_steps {manifest['episode_steps']} disagrees with "
                f"{manifests[0]['episode_steps']} from {dataset_dirs[0]}"
            )
        manifests.append(manifest)
        records.append(
            {
                "path": str(directory.resolve()),
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "teacher": manifest["teacher"],
                "opponent": manifest["opponent"],
                "episodes": len(manifest["episodes"]),
            }
        )

    # Seeds are only unique within one extraction: each run has its own
    # `--seed-start` but nothing forbids overlap, so the split key carries the
    # dataset the episode came from. Each corpus also holds out its own
    # highest seeds; a holdout taken from the globally highest keys would
    # measure one opponent while training on the mixture.
    held_out: set[tuple[int, int]] = set()
    entries: list[tuple[int, Path, dict[str, Any]]] = []
    for index, (directory, manifest) in enumerate(zip(dataset_dirs, manifests, strict=True)):
        episodes = manifest["episodes"]
        seeds = sorted({int(entry["seed"]) for entry in episodes})
        if seeds_per_dataset is not None:
            # The lowest seeds, so the corpus a cap selects is a prefix of the one
            # it would have used uncapped and does not move when a directory is
            # extended. The holdout still comes off the top of what is kept.
            seeds = seeds[:seeds_per_dataset]
            kept = set(seeds)
            episodes = [entry for entry in episodes if int(entry["seed"]) in kept]
            records[index]["episodes"] = len(episodes)
            records[index]["seeds_kept"] = len(seeds)
        if not 0 < holdout_seeds < len(seeds):
            raise ValueError(
                f"{directory}: holdout of {holdout_seeds} seeds needs "
                f"1..{len(seeds) - 1} with {len(seeds)} seeds"
            )
        held_out.update((index, seed) for seed in seeds[-holdout_seeds:])
        entries.extend((index, directory, entry) for entry in episodes)

    paths = [str(directory / entry["file"]) for _, directory, entry in entries]
    encode = partial(_encode_episode_file, architecture_name=architecture)
    if encode_workers > 1:
        with ProcessPoolExecutor(max_workers=encode_workers) as pool:
            encoded = list(pool.map(encode, paths, chunksize=1))
    else:
        encoded = [encode(path) for path in paths]

    splits: dict[bool, list[dict[str, np.ndarray]]] = {False: [], True: []}
    for (index, directory, entry), arrays in zip(entries, encoded, strict=True):
        _validate_targets_satisfy_masks(arrays, str(directory / entry["file"]))
        splits[(index, int(entry["seed"])) in held_out].append(arrays)
    # The split lists alias the same dicts, so dropping this one only frees the
    # list itself -- but it is what lets `stage` below release each episode as it
    # copies it, instead of the corpus being reachable from two places at once.
    encoded.clear()

    return _stage_split(splits[False]), _stage_split(splits[True]), records


def _batch(
    architecture: str,
    tensors: DemonstrationTensors,
    indices: torch.Tensor | slice,
    device: torch.device,
) -> tuple[tuple[Any, ...], dict[str, torch.Tensor]]:
    """One minibatch of actor forward arguments plus teacher-forced factors.

    The gather runs on the host, where the corpus lives, and moves the narrow
    storage dtypes; the widening casts to the compute dtypes then run on the
    accelerator through the same helpers the PPO update uses, so the bus
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


# `modded-nanogpt`'s pretraining schedule, transcribed from
# `TrainingSchedule.get_lr` (train_gpt.py:1968-1976) and `get_muon_momentum`
# (:1995-2005). Two departures from what this file used to do: the rate holds
# flat and then decays LINEARLY to a floor rather than following a cosine to
# zero, and the Nesterov coefficient is scheduled at all.
COOLDOWN_FRACTION = 0.60
FINAL_RATE_FRACTION = 0.15
MOMENTUM_WARMUP_STEPS = 300
MOMENTUM_COOLDOWN_STEPS = 50
MOMENTUM_MINIMUM = 0.85
MOMENTUM_MAXIMUM = 0.95


def _rate_fraction(step: int, total_steps: int) -> float:
    """The reference's trapezoid: flat, then linear to a floor, never to zero."""
    cooldown_start = int(total_steps * (1.0 - COOLDOWN_FRACTION))
    if step < cooldown_start:
        return 1.0
    progress = min(1.0, (step - cooldown_start) / max(total_steps - cooldown_start, 1))
    return (1.0 - progress) + FINAL_RATE_FRACTION * progress


def _momentum_at(step: int, total_steps: int) -> float:
    """Nesterov coefficient warmed up, held, then cooled back down.

    The reference's 300 and 50 steps are absolute counts on a run of tens of
    thousands of steps. A clone is far shorter, so both are capped as fractions
    of the run: an uncapped 300-step warmup could otherwise span the whole of
    training and never reach the coefficient it is warming up to.
    """
    warmup = min(MOMENTUM_WARMUP_STEPS, max(total_steps // 4, 1))
    cooldown = min(MOMENTUM_COOLDOWN_STEPS, max(total_steps // 20, 1))
    span = MOMENTUM_MAXIMUM - MOMENTUM_MINIMUM
    if step < warmup:
        return MOMENTUM_MINIMUM + span * (step / warmup)
    cooldown_start = total_steps - cooldown
    if step >= cooldown_start:
        return MOMENTUM_MAXIMUM - span * min(1.0, (step - cooldown_start) / cooldown)
    return MOMENTUM_MAXIMUM


def _apply_schedule(optimizer: NorMuon, step: int, total_steps: int) -> None:
    """Set this step's rate and Nesterov coefficient on every group."""
    fraction = _rate_fraction(step, total_steps)
    momentum = _momentum_at(step, total_steps)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * fraction
        if group["kind"] == "normuon":
            group["momentum"] = momentum


def train(
    *,
    dataset_dirs: Sequence[Path],
    output_dir: Path,
    architecture: str,
    config: Any,
    holdout_seeds: int,
    epochs: int,
    patience: int,
    batch_size: int,
    matrix_learning_rate: float,
    matrix_weight_decay: float,
    adam_learning_rate_ratio: float,
    adam_weight_decay: float,
    seed: int,
    device: torch.device,
    encode_workers: int,
    seeds_per_dataset: int | None = None,
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

    train_split, holdout_split, datasets = load_dataset(
        dataset_dirs,
        architecture=architecture,
        holdout_seeds=holdout_seeds,
        encode_workers=encode_workers,
        seeds_per_dataset=seeds_per_dataset,
    )
    print(
        f"dataset: {len(datasets)} corpora, {train_split.rows} train rows, "
        f"{holdout_split.rows} holdout rows ({holdout_seeds} held-out seeds each)",
        flush=True,
    )

    actor = resolve_architecture(architecture).actor_class(config).to(device)
    # Pretraining, so the reference's full recipe applies: spectrally normalized
    # matrix steps, Adam on the gains, biases and heads, and cautious decay on
    # both halves. The PPO update deliberately runs the same optimizer with
    # decay at zero; a clone has no trust region to keep.
    matrices, vectors = route_parameters(actor)
    optimizer = NorMuon(
        matrices,
        vectors,
        learning_rate=matrix_learning_rate,
        adam_learning_rate=matrix_learning_rate * adam_learning_rate_ratio,
        weight_decay=matrix_weight_decay,
        adam_weight_decay=adam_weight_decay,
    )
    steps_per_epoch = math.ceil(train_split.rows / batch_size)
    total_steps = max(epochs * steps_per_epoch, 1)
    step_index = 0
    autocast = device.type == "cuda"
    bc_provenance: dict[str, Any] = {
        "datasets": datasets,
        "architecture": architecture,
        "holdout_seeds": holdout_seeds,
        "command": sys.argv,
    }
    # A scalar teacher survives only when the mixture agrees on one, because a
    # warm start is tagged with this label; the opponent deliberately has no
    # scalar at all, since varying it across corpora is the point of mixing.
    teachers = {json.dumps(record["teacher"], sort_keys=True) for record in datasets}
    if len(teachers) == 1:
        bc_provenance["teacher"] = datasets[0]["teacher"]
    identity = source_identity()

    best = math.inf
    best_metrics: dict[str, float] = {}
    stale = 0
    # The journal stays the durable append-only record the mirror rebuilds
    # from, and TensorBoard is written live beside it, exactly as PPO
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
            # The rate and coefficient this epoch opens with, read from the
            # schedule rather than from the optimizer, whose groups still hold
            # the previous epoch's last step until the first step below sets it.
            applied_learning_rate = matrix_learning_rate * _rate_fraction(step_index, total_steps)
            applied_momentum = _momentum_at(step_index, total_steps)
            epoch_loss = 0.0
            epoch_components = 0.0
            for indices in _balanced_minibatch_slices(train_split.rows, batch_size):
                actor_args, factors = _batch(architecture, train_split, order[indices], device)
                loss = _clone_loss(actor, actor_args, factors, autocast)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite clone loss in epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _apply_schedule(optimizer, step_index, total_steps)
                optimizer.step()
                step_index += 1
                # The loss is a mean over active components, so the epoch
                # average must weight by that same count, exactly as the PPO
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
                "momentum": applied_momentum,
                "seconds": time.perf_counter() - started,
                **{f"holdout_{name}": value for name, value in holdout.items()},
            }
            # A diverged epoch must fail here rather than be written as the
            # bare `NaN` token, which is not JSON and which every downstream
            # reader would either reject or silently accept as a real loss.
            metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
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
        dataset_dirs=args.dataset,
        output_dir=args.output,
        architecture=args.architecture,
        config=model_config_from_args(resolve_architecture(args.architecture), args),
        holdout_seeds=args.holdout_seeds,
        seeds_per_dataset=args.seeds_per_dataset,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        matrix_learning_rate=args.matrix_learning_rate,
        matrix_weight_decay=args.matrix_weight_decay,
        adam_learning_rate_ratio=args.adam_learning_rate_ratio,
        adam_weight_decay=args.adam_weight_decay,
        seed=args.seed,
        device=torch.device(args.device),
        encode_workers=args.encode_workers,
    )


if __name__ == "__main__":
    main()
