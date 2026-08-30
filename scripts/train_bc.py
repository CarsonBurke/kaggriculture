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
import os
import sys
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from kaggriculture.encoding import encode_observation
from kaggriculture.inference import ACTOR_ARTIFACT_FORMAT_VERSION
from kaggriculture.latent_dynamics import (
    DecodeContext,
    DecodeHeads,
    DecodeMasks,
    LatentDynamics,
    belief_spread,
    latent_horizon_loss,
)
from kaggriculture.model import ActorOutput, FarmActor
from kaggriculture.modelargs import add_model_config_arguments, model_config_from_args
from kaggriculture.optim import NorMuon, route_parameters
from kaggriculture.orientation import ORIENTATION_CYCLE, augment_demonstration_rows
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
from kaggriculture.structured import (
    StructuredActor,
    StructuredBelief,
    StructuredInputs,
    refresh_fused_mlp_fp8,
)
from kaggriculture.structured_dynamics import StructuredDynamics, structured_horizon_loss
from kaggriculture.telemetry import TensorboardMirror
from kaggriculture.tokens import encode_structured_observation
from kaggriculture.training import write_checkpoint

SUPPORTED_DATASET_FORMAT_VERSIONS = frozenset((1,))
BC_ENCODING_CACHE_FORMAT_VERSION = 1
_ENCODING_SOURCE_FILES = frozenset(
    {
        "src/kaggriculture/actions.py",
        "src/kaggriculture/constants.py",
        "src/kaggriculture/encoding.py",
        "src/kaggriculture/tokens.py",
    }
)

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
        "--encoded-cache",
        type=Path,
        default=Path("data/.bc-encoded-cache"),
        help=(
            "architecture- and tokenizer-bound derived episode cache; pass an empty "
            "path through the programmatic API to disable it"
        ),
    )
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
            "clone reached 99.996%% accuracy and still could not act off its own regime"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        # 12 is a budget, not a plateau. The plateau reading was measured on the
        # v27 corpora (holdout 0.00188 at epoch 9, 0.00182 at 10) and the v16
        # corpora refute it: there holdout fell 0.000516 -> 0.000133 over epochs
        # 10-19, a 4x gain, while unit accuracy sat at 0.99996 the whole way. What
        # justifies the cap is that those gains arrive WITH THE DECAY, and the
        # rate schedule is a fraction of this number rather than a fixed step
        # count: at `--epochs 12` the final epoch runs at 1.13e-3, where a
        # 20-epoch schedule is still at 3.31e-3 on the same epoch. A 12-epoch run
        # is therefore not the first 12 epochs of a 20-epoch run -- it is the same
        # trapezoid compressed, tail included. The residual NLL it gives up is
        # confidence on decisions that were already correct.
        help=(
            "passes over the corpus; an epoch is a pass, not a fixed step count, "
            "so a larger corpus needs fewer of them, not more"
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="epochs without holdout improvement before stopping"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--run-length",
        type=int,
        default=1,
        help=(
            "rows per contiguous run: each minibatch is built from blocks of this "
            "many consecutive steps of one episode-seat, so an auxiliary objective "
            "over (step, step+1) pairs has pairs to work with. The default of 1 is "
            "the independent-row shuffle exactly -- same generator draw, same order "
            "-- so batches only change when an A/B raises it"
        ),
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help=(
            "torch.compile mode for the clone step, or 'none' for eager. Measured "
            "on three full 12-epoch arms: 'default' and "
            "'max-autotune-no-cudagraphs' reach the SAME steady state (2.03x and "
            "2.01x per epoch) but cost 39s and 473s to compile, so autotuning "
            "buys 0.6%% of throughput for 12x its own benefit and only breaks even "
            "past 16 epochs. Cudagraphs modes are refused separately: an epoch's "
            "last minibatch is a short tail, so the shape varies"
        ),
    )
    parser.add_argument(
        "--latent-dynamics-coefficient",
        type=float,
        default=0.0,
        help=(
            "weight on NextLat's SmoothL1 next-latent regression (the reference's "
            "lambda_mse, 1.0-3.0 in its shipped configs). Zero runs the plain "
            "clone and never builds the dynamics model"
        ),
    )
    parser.add_argument(
        "--latent-decode-coefficient",
        type=float,
        default=0.0,
        help=(
            "weight on the decode KL (the reference's lambda_kl, 0.1-1.0). This is "
            "the term that makes the latent decision-relevant rather than merely "
            "self-predictable; zero skips it rather than multiplying it by zero"
        ),
    )
    parser.add_argument(
        "--latent-horizon",
        type=int,
        default=1,
        help=(
            "steps to unroll the dynamics model (the reference's mtp_horizon, 1-8). "
            "Each extra step needs one more consecutive row, so it needs "
            "--run-length above it to have pairs to consume"
        ),
    )
    parser.add_argument(
        "--structured-latent-coefficient",
        type=float,
        default=0.0,
        help=(
            "weight on the reference-normalized SmoothL1 over policy-read "
            "structured decision tokens"
        ),
    )
    parser.add_argument(
        "--structured-decision-coefficient",
        type=float,
        default=0.0,
        help="weight on structured future-decision decode KL; zero removes the predictor",
    )
    parser.add_argument(
        "--structured-patch-coefficient",
        type=float,
        default=0.0,
        help="weight on normalized future own-patch feature L1",
    )
    parser.add_argument(
        "--structured-economy-coefficient",
        type=float,
        default=0.0,
        help="weight on normalized future economy-entity feature L1",
    )
    parser.add_argument(
        "--structured-opponent-summary-coefficient",
        type=float,
        default=0.0,
        help="weight on normalized future opponent-summary feature L1",
    )
    parser.add_argument(
        "--structured-opponent-patch-coefficient",
        type=float,
        default=0.0,
        help="weight on normalized future opponent-patch feature L1",
    )
    parser.add_argument(
        "--structured-decision-horizon",
        type=int,
        default=2,
        help="recursive steps for structured decision KL",
    )
    parser.add_argument(
        "--structured-patch-horizon",
        type=int,
        default=1,
        help="recursive steps for structured patch and state feature prediction",
    )
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
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="global gradient-norm clip; matches the NextLat pretraining recipe",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--encode-workers",
        type=int,
        default=2,
        help=(
            "processes for observation encoding; each worker is pinned to one "
            "host thread so this is the actual core count, not cores times BLAS"
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="intra-op threads for the parent and each encode worker",
    )
    return parser.parse_args()


def _pin_host_threads(threads: int = 1) -> None:
    """Stop BLAS/PyTorch from multiplying across encode workers.

    A worker that inherits the default intra-op pool (one thread per core)
    turns `--encode-workers N` into N times cores runnable threads.
    """
    threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


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

    Two int32 row-metadata columns travel with the features: `episode_index`,
    a dense index over staged episode-seats in staging order, and `step`, the
    step number inside that episode-seat. They are what lets a consumer pair
    row j with row j+1 -- eligible exactly when the episode index matches and
    the step advances by one -- without trusting a batch's provenance.
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


def _encoding_cache_schema(architecture: str) -> str:
    """Digest only sources that can change staged observation tensors."""

    identity = source_identity()
    files = identity["files"]
    missing = sorted(_ENCODING_SOURCE_FILES - files.keys())
    if missing:
        raise RuntimeError(f"source identity is missing encoding inputs: {missing}")
    payload = {
        "format_version": BC_ENCODING_CACHE_FORMAT_VERSION,
        "architecture": architecture,
        "files": {name: files[name] for name in sorted(_ENCODING_SOURCE_FILES)},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cached_episode_arrays(path: Path, expected: set[str]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError(
                f"{path}: encoded cache fields {sorted(archive.files)} disagree with "
                f"expected {sorted(expected)}"
            )
        return {name: archive[name] for name in archive.files}


def _encode_episode_file(
    path_text: str,
    cache_text: str | None = None,
    *,
    architecture_name: str,
) -> dict[str, np.ndarray]:
    """Encode one episode-seat archive's raw observations into model inputs."""

    state_fields = (
        {"board", "global_features", "units", "unit_positions"}
        if architecture_name == CONV_ENTITY
        else set(_STRUCTURED_STATE_FIELDS)
    )
    expected = set(_FACTOR_FIELDS) | state_fields
    cache = None if cache_text is None else Path(cache_text)
    if cache is not None:
        cached = _cached_episode_arrays(cache, expected)
        if cached is not None:
            return cached

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

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez(temporary, **arrays)
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
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
    # Pairing metadata, derived from row order rather than read from a field:
    # `extract_episode` walks `range(episode_steps - 1)` and stacks in that
    # order, so row j of an archive is step j of that seat, and the copy below
    # preserves it. One member is one episode-seat, so its staging position is
    # the dense episode index.
    episode_index = np.empty(rows, dtype=np.int32)
    step = np.empty(rows, dtype=np.int32)
    offset = 0
    for position, member in enumerate(members):
        span = member["unit_actions"].shape[0]
        for name, value in member.items():
            expected = (span, *stacked[name].shape[1:])
            if value.shape != expected:
                raise ValueError(
                    f"{name} has shape {value.shape}, expected {expected}; "
                    "re-extract every dataset after an action-space change"
                )
            stacked[name][offset : offset + span] = value

        components[offset : offset + span] = sum(
            member[name].astype(np.float64).sum(axis=1)
            for name in ("unit_active", "market_active", "market_quantity_active")
        )
        episode_index[offset : offset + span] = position
        step[offset : offset + span] = np.arange(span, dtype=np.int32)
        offset += span
        members[position] = {}
    stacked["episode_index"] = episode_index
    stacked["step"] = step
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
    torch_threads: int = 1,
    seeds_per_dataset: int | None = None,
    encoded_cache: Path | None = None,
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
    cache_paths: list[str | None]
    if encoded_cache is None:
        cache_paths = [None] * len(entries)
    else:
        schema = _encoding_cache_schema(architecture)
        cache_root = encoded_cache.expanduser().resolve() / schema
        cache_paths = [
            str(cache_root / records[index]["manifest_sha256"] / Path(entry["file"]).name)
            for index, _, entry in entries
        ]
        for record in records:
            record["encoded_cache_schema"] = schema
    encode = partial(_encode_episode_file, architecture_name=architecture)
    if encode_workers < 1:
        raise ValueError("encode workers must be positive")
    workers = min(encode_workers, len(paths))
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_pin_host_threads,
            initargs=(torch_threads,),
        ) as pool:
            encoded = list(pool.map(encode, paths, cache_paths, chunksize=1))
    else:
        encoded = [encode(path, cache) for path, cache in zip(paths, cache_paths, strict=True)]

    splits: dict[bool, list[dict[str, np.ndarray]]] = {False: [], True: []}
    for (index, directory, entry), arrays in zip(entries, encoded, strict=True):
        _validate_targets_satisfy_masks(arrays, str(directory / entry["file"]))
        splits[(index, int(entry["seed"])) in held_out].append(arrays)
    # The split lists alias the same dicts, so dropping this one only frees the
    # list itself -- but it is what lets `stage` below release each episode as it
    # copies it, instead of the corpus being reachable from two places at once.
    encoded.clear()
    return _stage_split(splits[False]), _stage_split(splits[True]), records


def _orient_host_batch(rows: dict[str, torch.Tensor], rng: np.random.Generator) -> None:
    """Apply one sampled symmetry per episode-seat, in place on the host gather.

    Holdout stays identity so the reported clone score is the real board.
    Training cycles all four frames so a warm-started member is not seeing
    a flipped farm for the first time in self-play.
    """
    if "board" not in rows:
        return
    episode = rows["episode_index"].detach().cpu().numpy()
    codes = np.zeros(episode.shape[0], dtype=np.int8)
    for ep in np.unique(episode):
        codes[episode == ep] = int(rng.integers(0, len(ORIENTATION_CYCLE)))
    if not np.any(codes):
        return
    arrays = {
        name: rows[name].detach().cpu().numpy().copy()
        for name in ("board", "units", "unit_positions", "unit_actions", "unit_masks")
    }
    augment_demonstration_rows(arrays, codes)
    for name, array in arrays.items():
        rows[name] = torch.from_numpy(np.ascontiguousarray(array))


def _batch(
    architecture: str,
    tensors: DemonstrationTensors,
    indices: torch.Tensor | slice,
    device: torch.device,
    orientation_rng: np.random.Generator | None = None,
) -> tuple[tuple[Any, ...], dict[str, torch.Tensor]]:
    """One minibatch of actor forward arguments plus teacher-forced factors.

    The gather runs on the host, where the corpus lives, and moves the narrow
    storage dtypes; the widening casts to the compute dtypes then run on the
    accelerator through the same helpers the PPO update uses, so the bus
    carries fp16 and int8 rather than the fp32 and int64 they become.
    """
    rows = {name: _batch_tensor(value, indices) for name, value in tensors.staged.items()}
    if orientation_rng is not None:
        _orient_host_batch(rows, orientation_rng)
    rows = {name: value.to(device, non_blocking=True) for name, value in rows.items()}
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
    # The auxiliary needs to know which rows are consecutive steps of one
    # episode-seat. Carried as int64 on the device rather than recomputed from
    # the host order, so the pairing a step trains on is the pairing that step's
    # rows actually have.
    factors["episode_index"] = _batch_tensor(rows["episode_index"], whole, torch.long)
    factors["step"] = _batch_tensor(rows["step"], whole, torch.long)
    return actor_args, factors


def _masked_mean(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return (values * active).sum() / active.sum().clamp(min=1)


def _clone_loss_from_output(
    actor: FarmActor | StructuredActor,
    output: ActorOutput,
    factors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """The clone objective given a forward that already ran.

    Split out so the auxiliary path can reuse ONE trunk pass: computing the
    belief and the logits separately would double the most expensive part of the
    step to save nothing.
    """
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
        return _clone_loss_from_output(actor, output, factors)


class LatentTerms(NamedTuple):
    """The clone loss and the auxiliary terms from one trunk pass."""

    clone: torch.Tensor
    dynamics: torch.Tensor
    decode: torch.Tensor
    unit_half: torch.Tensor
    market_half: torch.Tensor
    eligible: torch.Tensor
    cosine: torch.Tensor
    dispersion: torch.Tensor


def _clone_and_latent_loss(
    actor: FarmActor,
    dynamics: LatentDynamics,
    actor_args: tuple[Any, ...],
    factors: dict[str, torch.Tensor],
    autocast: bool,
    horizon: int,
    decode_coefficient: float,
) -> LatentTerms:
    """The clone objective and NextLat's two terms, sharing one forward.

    The belief is the tensor the policy heads read, so the decode term scores it
    through those same heads with their weights detached -- the gradient reaches
    the prediction and stops, which is what stops the degenerate solution of
    flattening the policy until every decode agrees.

    `decode_coefficient` at zero skips the decode entirely rather than
    multiplying it by zero. It is the more expensive of the two terms and the
    reference ships `lambda_kl = 0` configurations, so paying for it unweighted
    would be a pure waste.
    """
    with torch.autocast(
        device_type=factors["unit_actions"].device.type, dtype=torch.bfloat16, enabled=autocast
    ):
        belief_output = actor.forward_with_belief(*actor_args)
        clone = _clone_loss_from_output(actor, belief_output.output, factors)
    belief = belief_output.belief
    decode = (
        DecodeContext(
            heads=DecodeHeads.from_actor(actor),
            masks=DecodeMasks(
                unit_masks=factors["unit_masks"],
                market_kind_masks=factors["market_kind_masks"],
                market_quantity_masks=factors["market_quantity_masks"],
                unit_active=factors["unit_active"],
                market_active=factors["market_active"],
                market_quantity_active=factors["market_quantity_active"],
                market_kinds=factors["market_kinds"],
            ),
        )
        if decode_coefficient
        else None
    )
    horizon_loss = latent_horizon_loss(
        dynamics,
        belief,
        factors["unit_actions"],
        factors["market_kinds"],
        factors["market_quantities"],
        factors["episode_index"],
        factors["step"],
        horizon=horizon,
        decode=decode,
    )
    # The halves come back from the unroll rather than from a second prediction:
    # they are reported for attribution, not optimized separately, and the two
    # are on different scales because the actor norms the market half and not the
    # unit half, so a pooled SmoothL1 leans market and a play difference would
    # otherwise be unattributable.
    spread = belief_spread(belief)
    return LatentTerms(
        clone=clone,
        dynamics=horizon_loss.dynamics,
        decode=horizon_loss.decode,
        unit_half=horizon_loss.unit_half,
        market_half=horizon_loss.market_half,
        eligible=horizon_loss.eligible,
        cosine=spread.cosine_similarity,
        dispersion=spread.dispersion,
    )


class StructuredCloneTerms(NamedTuple):
    """Clone loss and typed structured-auxiliary diagnostics from one trunk pass."""

    clone: torch.Tensor
    latent: torch.Tensor
    decision: torch.Tensor
    decision_one: torch.Tensor
    decision_final: torch.Tensor
    decision_unit: torch.Tensor
    decision_market_kind: torch.Tensor
    decision_market_quantity: torch.Tensor
    patch: torch.Tensor
    patch_one: torch.Tensor
    patch_final: torch.Tensor
    patch_all: torch.Tensor
    patch_changed: torch.Tensor
    patch_unchanged: torch.Tensor
    economy: torch.Tensor
    opponent_summary: torch.Tensor
    opponent_patches: torch.Tensor
    eligible: torch.Tensor
    residual_ratio: torch.Tensor
    residual_own_patches: torch.Tensor
    residual_opponent_patches: torch.Tensor
    residual_opponent_summary: torch.Tensor
    residual_economy_entities: torch.Tensor
    residual_central_latents: torch.Tensor
    residual_unit_decisions: torch.Tensor
    residual_market_decisions: torch.Tensor


def _clone_and_structured_loss(
    actor: StructuredActor,
    dynamics: StructuredDynamics,
    actor_args: tuple[Any, ...],
    factors: dict[str, torch.Tensor],
    autocast: bool,
    decision_horizon: int,
    latent_horizon: int,
    patch_horizon: int,
    economy_active: bool,
    opponent_summary_active: bool,
    opponent_patches_active: bool,
) -> StructuredCloneTerms:
    """Clone and typed future-feature losses from one structured actor forward."""
    inputs = actor_args[0]
    if not isinstance(inputs, StructuredInputs):
        raise TypeError("structured auxiliary requires StructuredInputs")
    with torch.autocast(
        device_type=factors["unit_actions"].device.type,
        dtype=torch.bfloat16,
        enabled=autocast,
    ):
        output, belief = actor.forward_with_belief(inputs)
        clone = _clone_loss_from_output(actor, output, factors)
        decode = (
            DecodeContext(
                heads=DecodeHeads.from_actor(actor),
                masks=DecodeMasks(
                    unit_masks=factors["unit_masks"],
                    market_kind_masks=factors["market_kind_masks"],
                    market_quantity_masks=factors["market_quantity_masks"],
                    unit_active=factors["unit_active"],
                    market_active=factors["market_active"],
                    market_quantity_active=factors["market_quantity_active"],
                    market_kinds=factors["market_kinds"],
                ),
            )
            if decision_horizon
            else None
        )
        auxiliary = structured_horizon_loss(
            dynamics,
            belief,
            inputs,
            factors,
            decode=decode,
            decision_horizon=decision_horizon,
            latent_horizon=latent_horizon,
            patch_horizon=patch_horizon,
            economy_active=economy_active,
            opponent_summary_active=opponent_summary_active,
            opponent_patches_active=opponent_patches_active,
        )
    return StructuredCloneTerms(clone, *auxiliary)


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


@torch.no_grad()
def _structured_belief_diagnostics(
    actor: StructuredActor,
    tensors: DemonstrationTensors,
    *,
    batch_size: int,
    device: torch.device,
    autocast: bool,
) -> dict[str, float]:
    """Collapse diagnostics on one fixed holdout batch for each typed belief family."""
    rows = min(batch_size, tensors.rows)
    actor_args, _ = _batch(STRUCTURED, tensors, slice(0, rows), device)
    inputs = actor_args[0]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
        _, belief = actor.forward_with_belief(inputs)

    diagnostics: dict[str, float] = {}
    for name, value in zip(StructuredBelief._fields, belief, strict=True):
        flat = value.detach().float().flatten(0, 1)
        centered = flat - flat.mean(dim=0, keepdim=True)
        variance = centered.square().mean()
        singular = torch.linalg.svdvals(centered)
        spectrum = singular.square()
        probabilities = spectrum / spectrum.sum().clamp_min(1e-12)
        effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        spread = belief_spread(value)
        prefix = f"structured_{name}"
        diagnostics[f"{prefix}_variance"] = float(variance)
        diagnostics[f"{prefix}_effective_rank"] = float(effective_rank)
        diagnostics[f"{prefix}_cosine"] = float(spread.cosine_similarity)
        diagnostics[f"{prefix}_dispersion"] = float(spread.dispersion)
    return diagnostics


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


# Inductor's own list, read rather than restated, so a torch upgrade that adds or
# drops a mode cannot leave a stale allowlist behind. `none` is ours and means
# eager.
COMPILE_MODES = tuple(sorted(torch._inductor.list_mode_options()))

# Journalled per-step means when the auxiliary is on. `latent_steps` is the
# divisor and is popped before the record is written rather than shipped as a
# field nobody reads.
_LATENT_FIELDS = (
    "latent_dynamics",
    "latent_decode",
    "latent_unit_half",
    "latent_market_half",
    "latent_eligible",
    "belief_cosine",
    "belief_dispersion",
    "latent_steps",
)
_STRUCTURED_FIELDS = (
    "structured_latent",
    "structured_decision",
    "structured_decision_one",
    "structured_decision_final",
    "structured_decision_unit",
    "structured_decision_market_kind",
    "structured_decision_market_quantity",
    "structured_patch",
    "structured_patch_one",
    "structured_patch_final",
    "structured_patch_all",
    "structured_patch_changed",
    "structured_patch_unchanged",
    "structured_economy",
    "structured_opponent_summary",
    "structured_opponent_patches",
    "structured_eligible",
    "structured_residual_ratio",
    "structured_residual_own_patches",
    "structured_residual_opponent_patches",
    "structured_residual_opponent_summary",
    "structured_residual_economy_entities",
    "structured_residual_central_latents",
    "structured_residual_unit_decisions",
    "structured_residual_market_decisions",
    "structured_steps",
)


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


def _run_blocks(episode_index: torch.Tensor, run_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut the corpus into contiguous same-episode row blocks, in staging order.

    A block is up to `run_length` consecutive rows of one episode-seat, and rows
    within it are consecutive steps, because staging preserves the archive's row
    order and an archive is written step 0 first. A block never crosses into the
    next episode-seat, so no pair a consumer forms from adjacent rows straddles
    two games.

    An episode's last block is kept short rather than dropped. A drop would
    delete the same rows every epoch -- the tail of every episode, which is where
    the late-game behaviour lives -- and it would cost real data: the shipped
    720-step horizon stages 719 rows per seat, so a run length of 64 leaves a
    15-row tail on every one of them, and an epoch would stop being a full pass
    over the corpus.
    """
    if run_length < 1:
        raise ValueError("run length must be positive")
    rows = int(episode_index.shape[0])
    if rows < 1:
        raise ValueError("cannot build runs over an empty corpus")
    positions = torch.arange(rows)
    opens = torch.ones(rows, dtype=torch.bool)
    opens[1:] = episode_index[1:] != episode_index[:-1]
    episode_starts = positions[opens]
    episode_lengths = torch.diff(torch.cat((episode_starts, positions.new_tensor([rows]))))
    within = positions - torch.repeat_interleave(episode_starts, episode_lengths)
    # Every episode's first row opens a block, so consecutive starts are never
    # more than one episode apart and the gaps between them are the lengths.
    starts = positions[within % run_length == 0]
    return starts, torch.diff(torch.cat((starts, starts.new_tensor([rows]))))


def _run_epoch_order(
    starts: torch.Tensor, lengths: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """One epoch's row order: every block once, blocks shuffled, rows within in step order.

    A permutation of the blocks is a permutation of the rows, so an epoch stays a
    full pass with every row appearing exactly once. At a run length of one the
    blocks are the rows and this reduces to `torch.randperm(rows, generator=...)`
    -- the identical single generator draw, hence the identical order -- so the
    default is the sampler it replaces rather than a lookalike of it.
    """
    order = torch.randperm(int(starts.shape[0]), generator=generator)
    shuffled_starts, shuffled_lengths = starts[order], lengths[order]
    offsets = torch.cumsum(shuffled_lengths, 0) - shuffled_lengths
    rows = int(lengths.sum())
    return torch.repeat_interleave(shuffled_starts - offsets, shuffled_lengths) + torch.arange(rows)


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
    run_length: int = 1,
    encoded_cache: Path | None = None,
    compile_mode: str = "none",
    latent_dynamics_coefficient: float = 0.0,
    latent_decode_coefficient: float = 0.0,
    latent_horizon: int = 1,
    structured_latent_coefficient: float = 0.0,
    structured_decision_coefficient: float = 0.0,
    structured_patch_coefficient: float = 0.0,
    structured_economy_coefficient: float = 0.0,
    structured_opponent_summary_coefficient: float = 0.0,
    structured_opponent_patch_coefficient: float = 0.0,
    structured_decision_horizon: int = 2,
    structured_patch_horizon: int = 1,
    matrix_learning_rate: float,
    matrix_weight_decay: float,
    adam_learning_rate_ratio: float,
    adam_weight_decay: float,
    gradient_clip: float = 1.0,
    seed: int,
    device: torch.device,
    encode_workers: int,
    torch_threads: int = 1,
    seeds_per_dataset: int | None = None,
) -> dict[str, float]:
    """Run the full clone; returns the best holdout metrics."""
    if torch_threads < 1:
        raise ValueError("torch threads must be positive")
    _pin_host_threads(torch_threads)
    if epochs < 1 or patience < 1 or batch_size < 1 or run_length < 1:
        raise ValueError("epochs, patience, batch size, and run length must be positive")
    if gradient_clip <= 0:
        raise ValueError("gradient clip must be positive")
    # Checked before the corpus is staged, which takes minutes: a typo'd mode
    # otherwise surfaces at the first minibatch, after the wait. Inductor owns
    # the list, so this cannot drift from what torch actually accepts.
    if compile_mode != "none" and compile_mode not in COMPILE_MODES:
        raise ValueError(
            f"unknown compile mode {compile_mode!r}; expected 'none' or one of {COMPILE_MODES}"
        )
    # A captured graph is bound to one set of shapes, and an epoch's last
    # minibatch is a short tail, so a cudagraphs mode would recapture per shape
    # or fail outright. Read from inductor's config for the mode rather than
    # matched on the mode's name, which only happens to say so today.
    if compile_mode != "none" and torch._inductor.list_mode_options(compile_mode).get(
        "triton.cudagraphs"
    ):
        raise ValueError(
            f"compile mode {compile_mode!r} enables cudagraphs, which cannot capture "
            "the short last minibatch of an epoch"
        )
    if architecture_of_config(config).name != architecture:
        raise ValueError(
            f"{type(config).__name__} does not configure the {architecture} architecture"
        )
    # The auxiliary regresses row j onto row j+1, so it needs blocks longer than
    # one row to have any pair at all, and one more row per extra horizon step.
    # Refused rather than silently trained on an all-false eligibility mask,
    # which would report a loss of exactly zero and look converged.
    if latent_horizon < 1:
        raise ValueError("latent horizon must be at least one step")
    entity_coefficients = (latent_dynamics_coefficient, latent_decode_coefficient)
    structured_coefficients = (
        structured_latent_coefficient,
        structured_decision_coefficient,
        structured_patch_coefficient,
        structured_economy_coefficient,
        structured_opponent_summary_coefficient,
        structured_opponent_patch_coefficient,
    )
    if not all(
        math.isfinite(value) and value >= 0
        for value in (*entity_coefficients, *structured_coefficients)
    ):
        raise ValueError("latent coefficients must be finite and nonnegative")
    entity_active = any(entity_coefficients)
    structured_active = any(structured_coefficients)
    if entity_active and architecture == STRUCTURED:
        raise ValueError("structured actors require the typed structured auxiliary")
    if structured_active and architecture != STRUCTURED:
        raise ValueError("structured auxiliary coefficients require --architecture structured")
    if entity_active and structured_active:
        raise ValueError("entity and structured auxiliaries cannot be active together")
    if (structured_latent_coefficient or structured_decision_coefficient) and (
        structured_decision_horizon < 1
    ):
        raise ValueError("structured latent horizon must be positive when NextLat is active")
    state_active = any(structured_coefficients[2:])
    if state_active and structured_patch_horizon < 1:
        raise ValueError(
            "structured patch horizon must be positive when feature prediction is active"
        )
    entity_horizon = latent_horizon if entity_active else 0
    decision_horizon = (
        structured_decision_horizon
        if structured_latent_coefficient or structured_decision_coefficient
        else 0
    )
    patch_horizon = structured_patch_horizon if state_active else 0
    auxiliary_horizon = max(entity_horizon, decision_horizon, patch_horizon)
    if auxiliary_horizon and run_length <= auxiliary_horizon:
        raise ValueError(
            f"an auxiliary horizon of {auxiliary_horizon} needs --run-length above it; "
            f"got {run_length}, which yields no eligible pair"
        )
    if auxiliary_horizon and batch_size <= auxiliary_horizon:
        raise ValueError(
            f"an auxiliary horizon of {auxiliary_horizon} needs --batch-size above it; "
            f"got {batch_size}, so every minibatch has no eligible pair"
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
    orientation_rng = np.random.default_rng(seed)

    train_split, holdout_split, datasets = load_dataset(
        dataset_dirs,
        architecture=architecture,
        holdout_seeds=holdout_seeds,
        encode_workers=encode_workers,
        torch_threads=torch_threads,
        seeds_per_dataset=seeds_per_dataset,
        encoded_cache=encoded_cache,
    )
    if auxiliary_horizon:
        minibatches = _balanced_minibatch_slices(train_split.rows, batch_size)
        minimum_minibatch = min(batch.stop - batch.start for batch in minibatches)
        if minimum_minibatch <= auxiliary_horizon:
            raise ValueError(
                f"an auxiliary horizon of {auxiliary_horizon} needs every balanced "
                f"minibatch above it; the smallest minibatch has {minimum_minibatch} rows"
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
    # p_psi trains alongside the policy and under the same recipe: it is a
    # matrix-and-vector module like any other, and giving it a second optimizer
    # would mean a second schedule nobody chose.
    dynamics: LatentDynamics | StructuredDynamics | None
    if entity_active:
        dynamics = LatentDynamics(config.model_dim).to(device)
    elif structured_active:
        dynamics = StructuredDynamics(config).to(device)
    else:
        dynamics = None
    refresh_fused_mlp_fp8(actor, bootstrap_down=True)
    if dynamics is not None:
        refresh_fused_mlp_fp8(dynamics, bootstrap_down=True)
    matrices, vectors = route_parameters(actor)
    if dynamics is not None:
        extra_matrices, extra_vectors = route_parameters(dynamics)
        matrices, vectors = matrices + extra_matrices, vectors + extra_vectors
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
    # Fixed for the whole run: the blocks depend on the corpus and the run
    # length, and only their order is redrawn per epoch.
    run_starts, run_lengths = _run_blocks(train_split.staged["episode_index"], run_length)
    step_index = 0
    autocast = device.type == "cuda"
    # Measured on three full 12-epoch arms, not on a step benchmark. Fusing the
    # clone step is 2.03x per epoch (15.96s -> 7.88s) and cuts peak VRAM
    # 12.21 -> 7.99 GiB, because the intermediates stop being materialized: a
    # 96-dim, 127-token transformer is bandwidth-bound, which is where fusion
    # pays. `max-autotune-no-cudagraphs` reaches the SAME steady state (2.01x)
    # and is nonetheless the wrong choice -- it spends 473s compiling against
    # `default`'s 39s, so it breaks even only past 16 production epochs and made
    # the 12-epoch run measured here 3x SLOWER end to end. A per-step probe
    # cannot see that, because warmup hides exactly the cost that decides it.
    #
    # Compiled and eager are not bit-identical: inductor reorders reductions,
    # which moves a bf16 accumulation by 0.097% in global gradient L2 at
    # unchanged direction, while fp32 agrees to 2e-05
    # (`artifacts/probes/compile-parity.json`). Behaviourally they agree --
    # final holdout NLL 0.001574 eager against 0.001158 compiled, with unit
    # accuracy 0.99982 against 0.99986 -- so the residual is confidence on
    # decisions that were already right. Recorded in provenance below so an
    # artifact is never silently compared against one trained the other way.
    if dynamics is None:
        step_loss = _clone_loss
    elif isinstance(dynamics, StructuredDynamics):
        step_loss = _clone_and_structured_loss
    else:
        step_loss = _clone_and_latent_loss
    clone_loss: Any = (
        step_loss if compile_mode == "none" else torch.compile(step_loss, mode=compile_mode)
    )
    bc_provenance: dict[str, Any] = {
        "datasets": datasets,
        "architecture": architecture,
        "holdout_seeds": holdout_seeds,
        # How the batches were built, so an artifact is not silently comparable
        # to one trained with a different sampler.
        "batch_size": batch_size,
        "run_length": run_length,
        "compile_mode": compile_mode,
        "latent_dynamics_coefficient": latent_dynamics_coefficient,
        "latent_decode_coefficient": latent_decode_coefficient,
        "latent_horizon": latent_horizon,
        "structured_latent_coefficient": structured_latent_coefficient,
        "structured_decision_coefficient": structured_decision_coefficient,
        "structured_patch_coefficient": structured_patch_coefficient,
        "structured_economy_coefficient": structured_economy_coefficient,
        "structured_opponent_summary_coefficient": structured_opponent_summary_coefficient,
        "structured_opponent_patch_coefficient": structured_opponent_patch_coefficient,
        "structured_decision_horizon": decision_horizon,
        "structured_patch_horizon": patch_horizon,
        "gradient_clip": gradient_clip,
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
            # The order indexes host storage and also selects the epoch weights
            # from precomputed host-side counts, which needs the same order.
            order = _run_epoch_order(run_starts, run_lengths, generator)
            shuffled_components = train_split.row_components[order.numpy()]
            # The rate and coefficient this epoch opens with, read from the
            # schedule rather than from the optimizer, whose groups still hold
            # the previous epoch's last step until the first step below sets it.
            applied_learning_rate = matrix_learning_rate * _rate_fraction(step_index, total_steps)
            applied_momentum = _momentum_at(step_index, total_steps)
            epoch_loss = 0.0
            epoch_components = 0.0
            diagnostic_fields = _STRUCTURED_FIELDS if structured_active else _LATENT_FIELDS
            diagnostic_sums = dict.fromkeys(diagnostic_fields, 0.0)
            for indices in _balanced_minibatch_slices(train_split.rows, batch_size):
                actor_args, factors = _batch(
                    architecture,
                    train_split,
                    order[indices],
                    device,
                    orientation_rng=orientation_rng,
                )
                terms: Any = 0

                if dynamics is None:
                    loss = clone_loss(actor, actor_args, factors, autocast)
                elif isinstance(dynamics, StructuredDynamics):
                    terms = clone_loss(
                        actor,
                        dynamics,
                        actor_args,
                        factors,
                        autocast,
                        decision_horizon if structured_decision_coefficient else 0,
                        decision_horizon if structured_latent_coefficient else 0,
                        patch_horizon,
                        bool(structured_economy_coefficient),
                        bool(structured_opponent_summary_coefficient),
                        bool(structured_opponent_patch_coefficient),
                    )
                    loss = (
                        terms.clone
                        + structured_latent_coefficient * terms.latent
                        + structured_decision_coefficient * terms.decision
                        + structured_patch_coefficient * terms.patch
                        + structured_economy_coefficient * terms.economy
                        + structured_opponent_summary_coefficient * terms.opponent_summary
                        + structured_opponent_patch_coefficient * terms.opponent_patches
                    )
                else:
                    terms = clone_loss(
                        actor,
                        dynamics,
                        actor_args,
                        factors,
                        autocast,
                        latent_horizon,
                        latent_decode_coefficient,
                    )
                    loss = (
                        terms.clone
                        + latent_dynamics_coefficient * terms.dynamics
                        + latent_decode_coefficient * terms.decode
                    )
                if not torch.isfinite(loss):
                    if isinstance(terms, StructuredCloneTerms):
                        values = {
                            "clone": float(terms.clone.detach()),
                            "latent": float(terms.latent.detach()),
                            "decision": float(terms.decision.detach()),
                        }
                    else:
                        values = {"combined": float(loss.detach())}
                    raise FloatingPointError(
                        f"non-finite training loss in epoch {epoch}, step {step_index}: {values}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    matrices + vectors,
                    gradient_clip,
                    error_if_nonfinite=True,
                )
                _apply_schedule(optimizer, step_index, total_steps)
                optimizer.step()
                bootstrap_fp8_down = step_index < 16
                refresh_fused_mlp_fp8(actor, bootstrap_down=bootstrap_fp8_down)
                if dynamics is not None:
                    refresh_fused_mlp_fp8(
                        dynamics,
                        bootstrap_down=bootstrap_fp8_down,
                    )
                step_index += 1
                # The loss is a mean over active components, so the epoch
                # average must weight by that same count, exactly as the PPO
                # update aggregates its per-minibatch losses.
                components = float(shuffled_components[indices].sum())
                # The CLONE term is what the journal's `train_loss` has always
                # meant, and it stays comparable across arms only if the
                # auxiliary is excluded from it. The combined objective is not a
                # likelihood and averaging it under that name would make an A/B
                # unreadable.
                clone_term = loss if dynamics is None else terms.clone
                epoch_loss += float(clone_term.detach()) * components
                epoch_components += components
                if isinstance(dynamics, StructuredDynamics):
                    diagnostic_sums["structured_latent"] += float(terms.latent.detach())
                    diagnostic_sums["structured_decision"] += float(terms.decision.detach())
                    diagnostic_sums["structured_decision_one"] += float(terms.decision_one.detach())
                    diagnostic_sums["structured_decision_final"] += float(
                        terms.decision_final.detach()
                    )
                    diagnostic_sums["structured_decision_unit"] += float(
                        terms.decision_unit.detach()
                    )
                    diagnostic_sums["structured_decision_market_kind"] += float(
                        terms.decision_market_kind.detach()
                    )
                    diagnostic_sums["structured_decision_market_quantity"] += float(
                        terms.decision_market_quantity.detach()
                    )
                    diagnostic_sums["structured_patch"] += float(terms.patch.detach())
                    diagnostic_sums["structured_patch_one"] += float(terms.patch_one.detach())
                    diagnostic_sums["structured_patch_final"] += float(terms.patch_final.detach())
                    diagnostic_sums["structured_patch_all"] += float(terms.patch_all.detach())
                    diagnostic_sums["structured_patch_changed"] += float(
                        terms.patch_changed.detach()
                    )
                    diagnostic_sums["structured_patch_unchanged"] += float(
                        terms.patch_unchanged.detach()
                    )
                    diagnostic_sums["structured_economy"] += float(terms.economy.detach())
                    diagnostic_sums["structured_opponent_summary"] += float(
                        terms.opponent_summary.detach()
                    )
                    diagnostic_sums["structured_opponent_patches"] += float(
                        terms.opponent_patches.detach()
                    )
                    diagnostic_sums["structured_eligible"] += float(terms.eligible.detach())
                    diagnostic_sums["structured_residual_ratio"] += float(
                        terms.residual_ratio.detach()
                    )
                    diagnostic_sums["structured_residual_own_patches"] += float(
                        terms.residual_own_patches.detach()
                    )
                    diagnostic_sums["structured_residual_opponent_patches"] += float(
                        terms.residual_opponent_patches.detach()
                    )
                    diagnostic_sums["structured_residual_opponent_summary"] += float(
                        terms.residual_opponent_summary.detach()
                    )
                    diagnostic_sums["structured_residual_economy_entities"] += float(
                        terms.residual_economy_entities.detach()
                    )
                    diagnostic_sums["structured_residual_central_latents"] += float(
                        terms.residual_central_latents.detach()
                    )
                    diagnostic_sums["structured_residual_unit_decisions"] += float(
                        terms.residual_unit_decisions.detach()
                    )
                    diagnostic_sums["structured_residual_market_decisions"] += float(
                        terms.residual_market_decisions.detach()
                    )
                    diagnostic_sums["structured_steps"] += 1.0
                elif dynamics is not None:
                    diagnostic_sums["latent_dynamics"] += float(terms.dynamics.detach())
                    diagnostic_sums["latent_decode"] += float(terms.decode.detach())
                    diagnostic_sums["latent_unit_half"] += float(terms.unit_half.detach())
                    diagnostic_sums["latent_market_half"] += float(terms.market_half.detach())
                    diagnostic_sums["latent_eligible"] += float(terms.eligible.sum())
                    diagnostic_sums["belief_cosine"] += float(terms.cosine)
                    diagnostic_sums["belief_dispersion"] += float(terms.dispersion)
                    diagnostic_sums["latent_steps"] += 1.0
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
            if architecture == STRUCTURED:
                if not isinstance(actor, StructuredActor):
                    raise TypeError("structured architecture resolved a non-structured actor")
                record.update(
                    _structured_belief_diagnostics(
                        actor,
                        holdout_split,
                        batch_size=batch_size,
                        device=device,
                        autocast=autocast,
                    )
                )
            # Per-step means, so an arm's numbers are comparable across corpora
            # and batch sizes. Absent entirely on a plain clone rather than
            # written as zeros, which would read as a measured collapse.
            step_key = "structured_steps" if structured_active else "latent_steps"
            steps = diagnostic_sums.pop(step_key)
            if steps:
                record.update({name: value / steps for name, value in diagnostic_sums.items()})
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
        encoded_cache=args.encoded_cache,
        run_length=args.run_length,
        compile_mode=args.compile_mode,
        latent_dynamics_coefficient=args.latent_dynamics_coefficient,
        latent_decode_coefficient=args.latent_decode_coefficient,
        latent_horizon=args.latent_horizon,
        structured_latent_coefficient=args.structured_latent_coefficient,
        structured_decision_coefficient=args.structured_decision_coefficient,
        structured_patch_coefficient=args.structured_patch_coefficient,
        structured_economy_coefficient=args.structured_economy_coefficient,
        structured_opponent_summary_coefficient=args.structured_opponent_summary_coefficient,
        structured_opponent_patch_coefficient=args.structured_opponent_patch_coefficient,
        structured_decision_horizon=args.structured_decision_horizon,
        structured_patch_horizon=args.structured_patch_horizon,
        matrix_learning_rate=args.matrix_learning_rate,
        matrix_weight_decay=args.matrix_weight_decay,
        adam_learning_rate_ratio=args.adam_learning_rate_ratio,
        adam_weight_decay=args.adam_weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        device=torch.device(args.device),
        encode_workers=args.encode_workers,
        torch_threads=args.torch_threads,
    )


if __name__ == "__main__":
    main()
