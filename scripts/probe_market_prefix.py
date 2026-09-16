#!/usr/bin/env python3
"""Probe whether causal market prefixes improve a frozen structured BC actor.

This is an experiment-only read path.  It encodes every state once with a
frozen actor, then fits equally-sized residual decoders to either the real
preceding market orders or deterministic within-corpus/step permutations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    QUANTIFIED_MARKET_KINDS,
    MarketKind,
    MarketLedger,
    _apply_ledger_order,
    _ledger_kind_mask,
    _ledger_quantity_mask,
    apply_unit_shed_effect,
    apply_unit_tile_effect,
    copy_tile_grid,
)
from kaggriculture.constants import (
    MAX_MARKET_ORDERS,
    QUANTITY_BINS,
)
from kaggriculture.entity import EntityActor
from kaggriculture.inference import load_actor_artifact
from kaggriculture.registry import STRUCTURED, architecture_of
from kaggriculture.structured import StructuredActor

# The sibling trainer owns the dataset encoding/staging contract.  Make its
# directory importable both for direct CLI execution and importlib-based tests.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import train_bc  # noqa: E402


@dataclass(frozen=True)
class SeedSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class PrefixBatch:
    kinds: np.ndarray
    quantities: np.ndarray
    source_rows: np.ndarray


@dataclass
class LoadedSplit:
    tensors: train_bc.DemonstrationTensors
    episodes: list[dict[str, Any]]
    observations: list[dict[str, Any]] | None = None


@dataclass
class CachedSplit:
    decisions: Tensor
    base_kind_logits: Tensor
    base_quantity_context: Tensor
    unit_actions: Tensor
    market_kinds: Tensor
    market_quantities: Tensor
    market_kind_masks: Tensor
    market_quantity_masks: Tensor
    market_active: Tensor
    market_quantity_active: Tensor
    corpus_index: Tensor
    seed: Tensor
    episode_index: Tensor
    step: Tensor
    episodes: list[dict[str, Any]]
    observations: list[dict[str, Any]] | None = None

    @property
    def rows(self) -> int:
        return int(self.market_kinds.shape[0])


@dataclass(frozen=True)
class TeacherOutputs:
    kind_nll: np.ndarray
    quantity_nll: np.ndarray
    kind_correct: np.ndarray
    quantity_correct: np.ndarray
    kind_prediction: np.ndarray
    quantity_prediction: np.ndarray


def split_seed_sets(
    seeds: Sequence[int],
    *,
    holdout_seeds: int,
    validation_holdout_seeds: int,
    seeds_per_dataset: int | None = None,
) -> SeedSplit:
    """Return disjoint whole-seed train/validation/test partitions.

    The lowest capped seeds are retained, matching ``train_bc.load_dataset``;
    the highest retained seeds are held out, with validation preceding the
    untouched highest-seed test partition.
    """

    ordered = sorted({int(seed) for seed in seeds})
    if seeds_per_dataset is not None:
        if seeds_per_dataset <= 0:
            raise ValueError("seeds_per_dataset must be positive when provided")
        ordered = ordered[:seeds_per_dataset]
    if not ordered:
        raise ValueError("cannot split an empty seed set")
    if not 0 < holdout_seeds < len(ordered):
        raise ValueError(
            f"holdout of {holdout_seeds} seeds needs 1..{len(ordered) - 1} "
            f"with {len(ordered)} seeds"
        )
    if not 0 < validation_holdout_seeds < holdout_seeds:
        raise ValueError("validation_holdout_seeds must be positive and smaller than holdout_seeds")
    boundary = len(ordered) - holdout_seeds
    held_out = ordered[boundary:]
    split = SeedSplit(
        train=tuple(ordered[:boundary]),
        validation=tuple(held_out[:validation_holdout_seeds]),
        test=tuple(held_out[validation_holdout_seeds:]),
    )
    if (
        (set(split.train) & set(split.validation))
        or (set(split.train) & set(split.test))
        or (set(split.validation) & set(split.test))
    ):
        raise AssertionError("seed partitions overlap")
    return split


def _control_donors_for_slot(
    corpus_index: np.ndarray,
    steps: np.ndarray,
    active_history: np.ndarray,
    episode_seeds: np.ndarray,
    slot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose cross-seed donors using only activity revealed before this slot."""

    rows = len(corpus_index)
    episode_seeds = np.asarray(episode_seeds)
    if episode_seeds.shape != (rows,):
        raise ValueError("episode_seeds must have one value per row")
    if not 0 < slot < active_history.shape[1]:
        raise ValueError("control donor slot must have a preceding action")
    groups: dict[tuple[int, int, tuple[bool, ...]], list[int]] = defaultdict(list)
    for row, key in enumerate(zip(corpus_index.tolist(), steps.tolist(), strict=True)):
        # active[slot] is determined by whether an earlier prefix action stopped;
        # active after slot is deliberately excluded because it reveals targets.
        causal_signature = tuple(bool(value) for value in active_history[row, : slot + 1])
        groups[(int(key[0]), int(key[1]), causal_signature)].append(row)

    donors = np.arange(rows, dtype=np.int64)
    for key in sorted(groups):
        members = np.asarray(groups[key], dtype=np.int64)
        unique_seeds, counts = np.unique(episode_seeds[members], return_counts=True)
        if unique_seeds.size < 2 or int(counts.max()) * 2 > members.size:
            continue
        randomized = rng.permutation(members)
        order = randomized[np.argsort(episode_seeds[randomized], kind="stable")]
        shifted = np.roll(order, -int(counts.max()))
        donor_by_member = dict(zip(order.tolist(), shifted.tolist(), strict=True))
        candidate = np.asarray([donor_by_member[int(row)] for row in members], dtype=np.int64)
        if np.any(episode_seeds[candidate] == episode_seeds[members]):
            raise AssertionError("cross-seed donor construction failed")
        donors[members] = candidate
    return donors


def construct_prefixes(
    kinds: np.ndarray,
    quantities: np.ndarray,
    corpus_index: np.ndarray,
    steps: np.ndarray,
    market_active: np.ndarray,
    episode_seeds: np.ndarray,
    *,
    shuffled: bool,
    seed: int,
) -> PrefixBatch:
    """Build one coherent causal action prefix for each independently scored slot.

    Target slot ``s`` receives BOS followed by one donor's actions ``0:s``.
    The donor is selected within corpus/step and the active history causally
    known before ``s``, never using the current or later target.  Donors cross
    seeds whenever a bijective matching exists; source row -1 denotes slot 0.
    """

    kinds = np.asarray(kinds)
    quantities = np.asarray(quantities)
    corpus_index = np.asarray(corpus_index)
    steps = np.asarray(steps)
    market_active = np.asarray(market_active, dtype=bool)
    if kinds.ndim != 2 or quantities.shape != kinds.shape:
        raise ValueError("market kinds and quantities must be equal rank-2 arrays")
    rows, slots = kinds.shape
    if rows == 0 or slots == 0:
        raise ValueError("cannot construct prefixes from empty actions")
    if corpus_index.shape != (rows,) or steps.shape != (rows,):
        raise ValueError("corpus_index and steps must have one value per row")
    if market_active.shape != (rows, slots):
        raise ValueError("market_active must match the market action shape")
    if np.any((kinds < 0) | (kinds >= N_MARKET_KINDS)):
        raise ValueError("market kind target is out of range")
    if np.any((quantities < 0) | (quantities >= N_QUANTITIES)):
        raise ValueError("market quantity target is out of range")
    prefix_shape = (rows, slots, slots)
    prefix_kinds = np.full(prefix_shape, N_MARKET_KINDS, dtype=np.int16)
    prefix_quantities = np.full(prefix_shape, N_QUANTITIES, dtype=np.int16)
    sources = np.full((rows, slots), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    own_rows = np.arange(rows, dtype=np.int64)
    for target_slot in range(1, slots):
        donors = (
            _control_donors_for_slot(
                corpus_index,
                steps,
                market_active,
                episode_seeds,
                target_slot,
                rng,
            )
            if shuffled
            else own_rows
        )
        sources[:, target_slot] = donors
        prefix_kinds[:, target_slot, 1 : target_slot + 1] = kinds[donors, :target_slot]
        prefix_quantities[:, target_slot, 1 : target_slot + 1] = quantities[donors, :target_slot]
    return PrefixBatch(prefix_kinds, prefix_quantities, sources)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_raw_observations(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "raw_json_zlib" not in archive.files:
            raise ValueError(f"{path}: archive has no raw_json_zlib for ledger evaluation")
        raw = json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()))
    observations = [entry["observation"] for entry in raw.get("observations", [])]
    if len(observations) != expected_rows:
        raise ValueError(
            f"{path}: {len(observations)} raw observations disagree with {expected_rows} rows"
        )
    return observations


def load_probe_dataset(
    dataset_dirs: Sequence[Path],
    *,
    holdout_seeds: int,
    validation_holdout_seeds: int,
    seeds_per_dataset: int | None,
    encoded_cache: Path | None,
    encode_workers: int,
    torch_threads: int,
) -> tuple[dict[str, LoadedSplit], list[dict[str, Any]]]:
    """Load three whole-episode splits through train_bc's encoder and stager."""

    if not dataset_dirs:
        raise ValueError("no dataset directories given")
    if encode_workers < 1:
        raise ValueError("encode_workers must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")

    records: list[dict[str, Any]] = []
    entries: list[tuple[str, int, Path, dict[str, Any]]] = []
    episode_steps: int | None = None
    for corpus, raw_directory in enumerate(dataset_dirs):
        directory = raw_directory.expanduser().resolve()
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"{directory}: missing manifest.json")
        raw_manifest = manifest_path.read_bytes()
        try:
            manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc
        if manifest.get("format_version") not in train_bc.SUPPORTED_DATASET_FORMAT_VERSIONS:
            raise ValueError(
                f"{directory}: unsupported dataset format: {manifest.get('format_version')}"
            )
        manifest_episodes = manifest.get("episodes")
        if not isinstance(manifest_episodes, list) or not manifest_episodes:
            raise ValueError(f"{directory}: dataset manifest lists no episodes")
        current_steps = int(manifest.get("episode_steps", 0))
        if current_steps <= 1:
            raise ValueError(f"{directory}: invalid episode_steps {current_steps}")
        if episode_steps is not None and current_steps != episode_steps:
            raise ValueError(
                f"{directory}: episode_steps {current_steps} disagrees with {episode_steps}"
            )
        episode_steps = current_steps
        seeds = [int(entry["seed"]) for entry in manifest_episodes]
        seed_split = split_seed_sets(
            seeds,
            holdout_seeds=holdout_seeds,
            validation_holdout_seeds=validation_holdout_seeds,
            seeds_per_dataset=seeds_per_dataset,
        )
        partition_by_seed = {
            seed: partition
            for partition, values in (
                ("train", seed_split.train),
                ("validation", seed_split.validation),
                ("test", seed_split.test),
            )
            for seed in values
        }
        kept = set(partition_by_seed)
        kept_entries = [entry for entry in manifest_episodes if int(entry["seed"]) in kept]
        for entry in kept_entries:
            file_path = directory / str(entry["file"])
            if not file_path.is_file():
                raise ValueError(f"{file_path}: episode file does not exist")
            entries.append((partition_by_seed[int(entry["seed"])], corpus, directory, entry))
        records.append(
            {
                "index": corpus,
                "path": str(directory),
                "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                "teacher": manifest.get("teacher"),
                "opponent": manifest.get("opponent"),
                "episodes_manifest": len(manifest_episodes),
                "episodes_kept": len(kept_entries),
                "seeds": asdict(seed_split),
            }
        )

    schema = train_bc._encoding_cache_schema(STRUCTURED) if encoded_cache is not None else None
    paths = [str(directory / str(entry["file"])) for _, _, directory, entry in entries]
    cache_paths = [
        None
        if encoded_cache is None
        else str(
            encoded_cache.expanduser().resolve()
            / str(schema)
            / records[corpus]["manifest_sha256"]
            / Path(str(entry["file"])).name
        )
        for _, corpus, _, entry in entries
    ]
    encode = partial(train_bc._encode_episode_file, architecture_name=STRUCTURED)
    workers = min(encode_workers, len(paths))
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=train_bc._pin_host_threads,
            initargs=(torch_threads,),
        ) as pool:
            encoded = list(pool.map(encode, paths, cache_paths, chunksize=1))
    else:
        encoded = [encode(path, cache) for path, cache in zip(paths, cache_paths, strict=True)]

    members: dict[str, list[dict[str, np.ndarray]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    episode_records: dict[str, list[dict[str, Any]]] = {key: [] for key in members}
    test_observations: list[dict[str, Any]] = []
    for (partition, corpus, directory, entry), arrays in zip(entries, encoded, strict=True):
        file_path = directory / str(entry["file"])
        train_bc._validate_targets_satisfy_masks(arrays, str(file_path))
        rows = int(arrays["market_kinds"].shape[0])
        arrays["corpus_index"] = np.full(rows, corpus, dtype=np.int16)
        arrays["seed"] = np.full(rows, int(entry["seed"]), dtype=np.int64)
        members[partition].append(arrays)
        episode_records[partition].append(
            {
                "corpus_index": corpus,
                "seed": int(entry["seed"]),
                "file": str(entry["file"]),
                "seat": entry.get("seat"),
                "rows": rows,
            }
        )
        if partition == "test":
            test_observations.extend(_read_raw_observations(file_path, rows))
    encoded.clear()

    loaded: dict[str, LoadedSplit] = {}
    for partition in ("train", "validation", "test"):
        if not members[partition]:
            raise ValueError(f"{partition} partition contains no episodes")
        staged = train_bc._stage_split(members[partition])
        if staged.rows == 0:
            raise ValueError(f"{partition} partition contains no rows")
        loaded[partition] = LoadedSplit(
            tensors=staged,
            episodes=episode_records[partition],
            observations=test_observations if partition == "test" else None,
        )
    for record in records:
        record["encoded_cache_schema"] = schema
    return loaded, records


class ResidualMarketDecoder(nn.Module):
    """A tiny shared recurrent decoder that only adds residual actor logits."""

    quantity_kind_gate: Tensor
    quantity_values: Tensor
    quantity_bias: Tensor

    def __init__(self, actor: StructuredActor | EntityActor) -> None:
        super().__init__()
        width = int(actor.config.model_dim)
        quantity_rank = int(actor.config.quantity_rank)
        embedding = max(8, width // 4)
        self.width = width
        self.quantity_rank = quantity_rank
        self.kind_embedding = nn.Embedding(N_MARKET_KINDS + 1, embedding)
        self.quantity_embedding = nn.Embedding(N_QUANTITIES + 1, embedding)
        self.cell = nn.GRUCell(width + 2 * embedding, width)
        self.kind_residual = nn.Linear(width, N_MARKET_KINDS)
        self.quantity_residual = nn.Linear(width, quantity_rank)
        nn.init.zeros_(self.kind_residual.weight)
        nn.init.zeros_(self.kind_residual.bias)
        nn.init.zeros_(self.quantity_residual.weight)
        nn.init.zeros_(self.quantity_residual.bias)
        self.register_buffer(
            "quantity_kind_gate", actor.market_quantity_kind_gate.weight.detach().float().clone()
        )
        self.register_buffer(
            "quantity_values", actor.market_quantity_value.weight.detach().float().clone()
        )
        self.register_buffer("quantity_bias", actor.market_quantity_bias.detach().float().clone())

    def forward(
        self,
        decisions: Tensor,
        base_kind_logits: Tensor,
        base_quantity_context: Tensor,
        prefix_kinds: Tensor,
        prefix_quantities: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if decisions.ndim != 3 or decisions.shape[1] != MAX_MARKET_ORDERS:
            raise ValueError("decisions must be [batch, market slots, width]")
        batch, slots, _ = decisions.shape
        expected_prefix = (batch, slots, slots)
        if prefix_kinds.shape != expected_prefix or prefix_quantities.shape != expected_prefix:
            raise ValueError(f"prefix tensors must have shape {expected_prefix}")
        hidden = decisions.new_zeros((batch, slots, self.width), dtype=torch.float32)
        target_slots = torch.arange(slots, device=decisions.device)[None, :, None]
        decisions = decisions.float()
        for prefix_slot in range(slots):
            decision = decisions[:, prefix_slot, None].expand(-1, slots, -1)
            cell_input = torch.cat(
                (
                    decision,
                    self.kind_embedding(prefix_kinds[:, :, prefix_slot]),
                    self.quantity_embedding(prefix_quantities[:, :, prefix_slot]),
                ),
                dim=-1,
            )
            updated = self.cell(
                cell_input.reshape(batch * slots, -1),
                hidden.reshape(batch * slots, self.width),
            ).view(batch, slots, self.width)
            hidden = torch.where(target_slots >= prefix_slot, updated, hidden)
        return (
            base_kind_logits.float() + self.kind_residual(hidden),
            base_quantity_context.float() + self.quantity_residual(hidden),
        )

    def decode_slot(
        self,
        decisions: Tensor,
        base_kind_logits: Tensor,
        base_quantity_context: Tensor,
        prefix_kinds: Tensor,
        prefix_quantities: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score one slot from a reset hidden state and one coherent prefix."""

        prefix_steps = prefix_kinds.shape[1]
        if prefix_quantities.shape != prefix_kinds.shape:
            raise ValueError("kind and quantity prefixes must have equal shapes")
        if decisions.shape[1] < prefix_steps:
            raise ValueError("decision sequence is shorter than the action prefix")
        hidden = decisions.new_zeros((decisions.shape[0], self.width), dtype=torch.float32)
        for prefix_slot in range(prefix_steps):
            cell_input = torch.cat(
                (
                    decisions[:, prefix_slot].float(),
                    self.kind_embedding(prefix_kinds[:, prefix_slot]),
                    self.quantity_embedding(prefix_quantities[:, prefix_slot]),
                ),
                dim=-1,
            )
            hidden = self.cell(cell_input, hidden)
        return (
            base_kind_logits.float() + self.kind_residual(hidden),
            base_quantity_context.float() + self.quantity_residual(hidden),
        )

    def quantity_logits(self, context: Tensor, kinds: Tensor) -> Tensor:
        features = context * (1.0 + self.quantity_kind_gate[kinds])
        return features @ self.quantity_values.T + self.quantity_bias[kinds]


@torch.inference_mode()
def cache_actor_features(
    actor: StructuredActor | EntityActor,
    split: LoadedSplit,
    *,
    batch_size: int,
    device: torch.device,
    autocast: bool,
) -> CachedSplit:
    if split.tensors.rows == 0:
        raise ValueError("cannot cache an empty split")
    cached: dict[str, list[Tensor]] = defaultdict(list)
    actor.eval()
    actor.requires_grad_(False)
    for start in range(0, split.tensors.rows, batch_size):
        stop = min(split.tensors.rows, start + batch_size)
        actor_args, _ = train_bc._batch(STRUCTURED, split.tensors, slice(start, stop), device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast,
        ):
            output, belief = actor.forward_with_belief(actor_args[0])
        cached["decisions"].append(belief.market_decisions.detach().to("cpu"))
        cached["base_kind_logits"].append(output.market_kind_logits.detach().to("cpu"))
        cached["base_quantity_context"].append(output.market_quantity_context.detach().to("cpu"))
    fields = {
        name: split.tensors.staged[name]
        for name in (
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "market_kind_masks",
            "market_quantity_masks",
            "market_active",
            "market_quantity_active",
            "corpus_index",
            "seed",
            "episode_index",
            "step",
        )
    }
    return CachedSplit(
        decisions=torch.cat(cached["decisions"]),
        base_kind_logits=torch.cat(cached["base_kind_logits"]),
        base_quantity_context=torch.cat(cached["base_quantity_context"]),
        episodes=split.episodes,
        observations=split.observations,
        **fields,
    )


def _to_device(
    tensor: Tensor,
    indices: Tensor | slice,
    device: torch.device,
    dtype=None,
) -> Tensor:
    selected = tensor[indices]
    if dtype is not None:
        selected = selected.to(dtype=dtype)
    return selected.to(device, non_blocking=True)


def _masked_nll(logits: Tensor, masks: Tensor, targets: Tensor) -> Tensor:
    if logits.shape != masks.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} != {masks.shape}")
    if not bool(masks.any(dim=-1).all()):
        raise ValueError("every categorical decision needs a valid action")
    if not bool(masks.gather(-1, targets.unsqueeze(-1)).all()):
        raise ValueError("target violates its teacher-forced mask")
    masked = logits.float().masked_fill(~masks.bool(), torch.finfo(torch.float32).min)
    return -F.log_softmax(masked, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _decoder_batch(
    model: ResidualMarketDecoder | None,
    data: CachedSplit,
    prefixes: PrefixBatch | None,
    indices: Tensor | slice,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    base_kind = _to_device(data.base_kind_logits, indices, device, torch.float32)
    base_quantity = _to_device(data.base_quantity_context, indices, device, torch.float32)
    if model is None:
        return base_kind, base_quantity
    if prefixes is None:
        raise ValueError("a residual decoder requires prefixes")
    prefix_kinds = torch.as_tensor(prefixes.kinds[indices], device=device, dtype=torch.long)
    prefix_quantities = torch.as_tensor(
        prefixes.quantities[indices], device=device, dtype=torch.long
    )
    return model(
        _to_device(data.decisions, indices, device, torch.float32),
        base_kind,
        base_quantity,
        prefix_kinds,
        prefix_quantities,
    )


def train_paired_decoders(
    true_model: ResidualMarketDecoder,
    shuffled_model: ResidualMarketDecoder,
    data: CachedSplit,
    true_prefixes: PrefixBatch,
    shuffled_prefixes: PrefixBatch,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    if data.rows == 0:
        raise ValueError("cannot train on an empty split")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid decoder training settings")
    true_model.to(device).train()
    shuffled_model.to(device).train()
    optimizers = (
        torch.optim.AdamW(true_model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        torch.optim.AdamW(shuffled_model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        order = torch.randperm(data.rows, generator=generator)
        sums = [0.0, 0.0]
        counts = [0, 0]
        for start in range(0, data.rows, batch_size):
            indices = order[start : start + batch_size]
            kinds = _to_device(data.market_kinds, indices, device, torch.long)
            quantities = _to_device(data.market_quantities, indices, device, torch.long)
            kind_masks = _to_device(data.market_kind_masks, indices, device).bool()
            quantity_masks = _to_device(data.market_quantity_masks, indices, device).bool()
            kind_active = _to_device(data.market_active, indices, device).bool()
            quantity_active = _to_device(data.market_quantity_active, indices, device).bool()
            active_count = int(kind_active.sum() + quantity_active.sum())
            if active_count == 0:
                continue
            for arm, (model, optimizer, prefixes) in enumerate(
                zip(
                    (true_model, shuffled_model),
                    optimizers,
                    (true_prefixes, shuffled_prefixes),
                    strict=True,
                )
            ):
                optimizer.zero_grad(set_to_none=True)
                kind_logits, quantity_context = _decoder_batch(
                    model, data, prefixes, indices, device
                )
                quantity_logits = model.quantity_logits(quantity_context, kinds)
                kind_nll = _masked_nll(kind_logits, kind_masks, kinds)
                quantity_nll = _masked_nll(quantity_logits, quantity_masks, quantities)
                loss_sum = (kind_nll * kind_active).sum() + (quantity_nll * quantity_active).sum()
                loss = loss_sum / active_count
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite training loss in arm {arm}")
                loss.backward()
                optimizer.step()
                sums[arm] += float(loss_sum.detach())
                counts[arm] += active_count
        if min(counts) == 0:
            raise ValueError("training split has no active market decisions")
        history.append(
            {
                "epoch": epoch + 1,
                "true_nll": sums[0] / counts[0],
                "shuffled_nll": sums[1] / counts[1],
            }
        )
    return history


@torch.inference_mode()
def teacher_forced_outputs(
    model: ResidualMarketDecoder | None,
    data: CachedSplit,
    prefixes: PrefixBatch | None,
    *,
    batch_size: int,
    device: torch.device,
) -> TeacherOutputs:
    if data.rows == 0:
        raise ValueError("cannot evaluate an empty split")
    if model is not None:
        model.to(device).eval()
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, data.rows, batch_size):
        indices = slice(start, min(data.rows, start + batch_size))
        kinds = _to_device(data.market_kinds, indices, device, torch.long)
        quantities = _to_device(data.market_quantities, indices, device, torch.long)
        kind_masks = _to_device(data.market_kind_masks, indices, device).bool()
        quantity_masks = _to_device(data.market_quantity_masks, indices, device).bool()
        kind_logits, quantity_context = _decoder_batch(model, data, prefixes, indices, device)
        if model is None:
            raise_if = None
            # The frozen actor's exact factored quantity head is supplied by the caller's
            # decoder-compatible buffer holder; baseline evaluation is routed separately.
            del raise_if
            raise ValueError("baseline evaluation requires a quantity-head reference")
        quantity_logits = model.quantity_logits(quantity_context, kinds)
        masked_kind = kind_logits.masked_fill(~kind_masks, torch.finfo(torch.float32).min)
        masked_quantity = quantity_logits.masked_fill(
            ~quantity_masks, torch.finfo(torch.float32).min
        )
        values = {
            "kind_nll": _masked_nll(kind_logits, kind_masks, kinds),
            "quantity_nll": _masked_nll(quantity_logits, quantity_masks, quantities),
            "kind_correct": masked_kind.argmax(-1).eq(kinds),
            "quantity_correct": masked_quantity.argmax(-1).eq(quantities),
            "kind_prediction": masked_kind.argmax(-1),
            "quantity_prediction": masked_quantity.argmax(-1),
        }
        for name, value in values.items():
            collected[name].append(value.cpu().numpy())
    return TeacherOutputs(**{name: np.concatenate(parts) for name, parts in collected.items()})


@torch.inference_mode()
def baseline_teacher_forced_outputs(
    quantity_head: ResidualMarketDecoder,
    data: CachedSplit,
    *,
    batch_size: int,
    device: torch.device,
) -> TeacherOutputs:
    """Evaluate the untouched actor logits using frozen quantity-head buffers."""

    quantity_head.to(device).eval()
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, data.rows, batch_size):
        indices = slice(start, min(data.rows, start + batch_size))
        kinds = _to_device(data.market_kinds, indices, device, torch.long)
        quantities = _to_device(data.market_quantities, indices, device, torch.long)
        kind_masks = _to_device(data.market_kind_masks, indices, device).bool()
        quantity_masks = _to_device(data.market_quantity_masks, indices, device).bool()
        kind_logits = _to_device(data.base_kind_logits, indices, device, torch.float32)
        quantity_context = _to_device(data.base_quantity_context, indices, device, torch.float32)
        quantity_logits = quantity_head.quantity_logits(quantity_context, kinds)
        masked_kind = kind_logits.masked_fill(~kind_masks, torch.finfo(torch.float32).min)
        masked_quantity = quantity_logits.masked_fill(
            ~quantity_masks, torch.finfo(torch.float32).min
        )
        values = {
            "kind_nll": _masked_nll(kind_logits, kind_masks, kinds),
            "quantity_nll": _masked_nll(quantity_logits, quantity_masks, quantities),
            "kind_correct": masked_kind.argmax(-1).eq(kinds),
            "quantity_correct": masked_quantity.argmax(-1).eq(quantities),
            "kind_prediction": masked_kind.argmax(-1),
            "quantity_prediction": masked_quantity.argmax(-1),
        }
        for name, value in values.items():
            collected[name].append(value.cpu().numpy())
    return TeacherOutputs(**{name: np.concatenate(parts) for name, parts in collected.items()})


def _head_metrics(nll: np.ndarray, correct: np.ndarray, active: np.ndarray) -> dict[str, Any]:
    active = np.asarray(active, dtype=bool)
    count = int(active.sum())
    if count == 0:
        return {"count": 0, "nll": None, "accuracy": None}
    return {
        "count": count,
        "nll": float(np.asarray(nll)[active].mean(dtype=np.float64)),
        "accuracy": float(np.asarray(correct)[active].mean(dtype=np.float64)),
    }


def summarize_teacher_outputs(
    outputs: TeacherOutputs,
    data: CachedSplit,
    corpus_names: Sequence[str],
) -> dict[str, Any]:
    kind_active = data.market_active.numpy().astype(bool)
    quantity_active = data.market_quantity_active.numpy().astype(bool)
    slots = np.arange(kind_active.shape[1])[None, :]

    def summary(row_filter: np.ndarray, slot_start: int = 0) -> dict[str, Any]:
        slot_filter = slots >= slot_start
        ka = kind_active & row_filter[:, None] & slot_filter
        qa = quantity_active & row_filter[:, None] & slot_filter
        kind = _head_metrics(outputs.kind_nll, outputs.kind_correct, ka)
        quantity = _head_metrics(outputs.quantity_nll, outputs.quantity_correct, qa)
        count = kind["count"] + quantity["count"]
        if count:
            total_nll = (
                float((outputs.kind_nll * ka).sum(dtype=np.float64))
                + float((outputs.quantity_nll * qa).sum(dtype=np.float64))
            ) / count
            total_accuracy = (
                float((outputs.kind_correct * ka).sum(dtype=np.float64))
                + float((outputs.quantity_correct * qa).sum(dtype=np.float64))
            ) / count
        else:
            total_nll = total_accuracy = None
        return {
            "total": {"count": count, "nll": total_nll, "accuracy": total_accuracy},
            "kind": kind,
            "quantity": quantity,
        }

    all_rows = np.ones(data.rows, dtype=bool)
    # Exact-slot reports, rather than suffixes.
    by_slot = {}
    for slot in range(kind_active.shape[1]):
        ka = kind_active & (slots == slot)
        qa = quantity_active & (slots == slot)
        kind = _head_metrics(outputs.kind_nll, outputs.kind_correct, ka)
        quantity = _head_metrics(outputs.quantity_nll, outputs.quantity_correct, qa)
        count = kind["count"] + quantity["count"]
        total = {
            "count": count,
            "nll": None
            if not count
            else (
                float((outputs.kind_nll * ka).sum(dtype=np.float64))
                + float((outputs.quantity_nll * qa).sum(dtype=np.float64))
            )
            / count,
            "accuracy": None
            if not count
            else (
                float((outputs.kind_correct * ka).sum(dtype=np.float64))
                + float((outputs.quantity_correct * qa).sum(dtype=np.float64))
            )
            / count,
        }
        by_slot[str(slot)] = {"total": total, "kind": kind, "quantity": quantity}
    corpus_values = data.corpus_index.numpy()
    return {
        "all_slots": summary(all_rows),
        "prefix_effect_slots_1_9": summary(all_rows, slot_start=1),
        "slot_0_negative_control": by_slot["0"],
        "by_slot": by_slot,
        "by_corpus": {
            str(corpus_names[index]): summary(corpus_values == index)
            for index in range(len(corpus_names))
        },
    }


def paired_episode_summary(
    true_nll: np.ndarray,
    control_nll: np.ndarray,
    episode_ids: np.ndarray,
    active: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
    group_name: str = "episode",
) -> dict[str, Any]:
    """Aggregate paired true-minus-control NLL by an untouched whole group."""

    true_nll = np.asarray(true_nll, dtype=np.float64)
    control_nll = np.asarray(control_nll, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    episode_ids = np.asarray(episode_ids)
    if true_nll.shape != control_nll.shape or true_nll.shape != active.shape:
        raise ValueError("paired losses and active mask must have identical shapes")
    if true_nll.shape[0] != episode_ids.shape[0]:
        raise ValueError("episode_ids must have one value per loss row")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    deltas: list[float] = []
    true_means: list[float] = []
    control_means: list[float] = []
    for episode in np.unique(episode_ids):
        rows = episode_ids == episode
        mask = active[rows]
        if not mask.any():
            continue
        selected_true = true_nll[rows][mask]
        selected_control = control_nll[rows][mask]
        true_mean = float(selected_true.mean())
        control_mean = float(selected_control.mean())
        true_means.append(true_mean)
        control_means.append(control_mean)
        deltas.append(true_mean - control_mean)
    if not deltas:
        raise ValueError(f"paired aggregation has no active {group_name}s")
    delta_array = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(delta_array), size=(bootstrap_samples, len(delta_array)))
    bootstrapped = delta_array[draw].mean(axis=1)
    delta = float(delta_array.mean())
    control_mean = float(np.mean(control_means))
    return {
        f"{group_name}s": len(deltas),
        f"true_{group_name}_mean_nll": float(np.mean(true_means)),
        f"shuffled_{group_name}_mean_nll": control_mean,
        "nll_true_minus_shuffled": delta,
        "nll_improvement_shuffled_minus_true": -delta,
        "relative_nll_improvement": None if control_mean == 0 else -delta / control_mean,
        "bootstrap_95_ci_true_minus_shuffled": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
        "bootstrap_samples": bootstrap_samples,
    }


def _comparison_metrics(
    true_outputs: TeacherOutputs,
    shuffled_outputs: TeacherOutputs,
    baseline_outputs: TeacherOutputs,
    data: CachedSplit,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    kind_active = data.market_active.numpy().astype(bool)
    quantity_active = data.market_quantity_active.numpy().astype(bool)
    prefix_slots = np.arange(kind_active.shape[1])[None, :] >= 1
    ka = kind_active & prefix_slots
    qa = quantity_active & prefix_slots
    combined_true = np.concatenate((true_outputs.kind_nll, true_outputs.quantity_nll), axis=1)
    combined_shuffled = np.concatenate(
        (shuffled_outputs.kind_nll, shuffled_outputs.quantity_nll), axis=1
    )
    combined_active = np.concatenate((ka, qa), axis=1)
    paired_episode = paired_episode_summary(
        combined_true,
        combined_shuffled,
        data.episode_index.numpy(),
        combined_active,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    seed_ids = np.asarray(
        [
            f"{int(corpus)}:{int(episode_seed)}"
            for corpus, episode_seed in zip(
                data.corpus_index.numpy(), data.seed.numpy(), strict=True
            )
        ]
    )
    paired_seed = paired_episode_summary(
        combined_true,
        combined_shuffled,
        seed_ids,
        combined_active,
        seed=seed + 1,
        bootstrap_samples=bootstrap_samples,
        group_name="seed",
    )

    def flat_correct(outputs: TeacherOutputs) -> np.ndarray:
        return np.concatenate((outputs.kind_correct[ka], outputs.quantity_correct[qa]))

    def delta(
        true_values: np.ndarray,
        shuffled_values: np.ndarray,
        active: np.ndarray,
    ) -> float | None:
        if not active.any():
            return None
        return float(true_values[active].mean() - shuffled_values[active].mean())

    true_correct = flat_correct(true_outputs)
    shuffled_correct = flat_correct(shuffled_outputs)
    baseline_correct = flat_correct(baseline_outputs)
    if baseline_correct.size == 0:
        raise ValueError("test split has no active prefix-slot decisions")
    baseline_errors = ~baseline_correct
    total_true_nll = np.concatenate((true_outputs.kind_nll[ka], true_outputs.quantity_nll[qa]))
    total_shuffled_nll = np.concatenate(
        (shuffled_outputs.kind_nll[ka], shuffled_outputs.quantity_nll[qa])
    )
    slot_zero_kind = kind_active & ~prefix_slots
    slot_zero_quantity = quantity_active & ~prefix_slots
    return {
        "primary_paired_seed": paired_seed,
        "paired_episode_seat": paired_episode,
        "prefix_slots_1_9_true_minus_shuffled": {
            "total": {
                "nll": float(total_true_nll.mean() - total_shuffled_nll.mean()),
                "accuracy": float(true_correct.mean() - shuffled_correct.mean()),
            },
            "kind": {
                "nll": delta(true_outputs.kind_nll, shuffled_outputs.kind_nll, ka),
                "accuracy": delta(true_outputs.kind_correct, shuffled_outputs.kind_correct, ka),
            },
            "quantity": {
                "nll": delta(true_outputs.quantity_nll, shuffled_outputs.quantity_nll, qa),
                "accuracy": delta(
                    true_outputs.quantity_correct,
                    shuffled_outputs.quantity_correct,
                    qa,
                ),
            },
        },
        "slot_0_true_minus_shuffled": {
            "kind_nll": delta(true_outputs.kind_nll, shuffled_outputs.kind_nll, slot_zero_kind),
            "kind_accuracy": delta(
                true_outputs.kind_correct,
                shuffled_outputs.kind_correct,
                slot_zero_kind,
            ),
            "quantity_nll": delta(
                true_outputs.quantity_nll,
                shuffled_outputs.quantity_nll,
                slot_zero_quantity,
            ),
            "quantity_accuracy": delta(
                true_outputs.quantity_correct,
                shuffled_outputs.quantity_correct,
                slot_zero_quantity,
            ),
        },
        "baseline_error_correction_rate": None
        if not baseline_errors.any()
        else float(true_correct[baseline_errors].mean()),
        "previously_correct_breakage_rate": None
        if not baseline_correct.any()
        else float((~true_correct[baseline_correct]).mean()),
    }


def _initialize_ledgers(data: CachedSplit) -> list[MarketLedger]:
    if data.observations is None or len(data.observations) != data.rows:
        raise ValueError("free-running evaluation requires one raw observation per row")
    unit_actions = data.unit_actions.numpy()
    ledgers: list[MarketLedger] = []
    for row, observation in enumerate(data.observations):
        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms") or []
        if player >= len(farms):
            raise ValueError(f"row {row}: observation has no acting farm")
        farm = farms[player]
        remaining_shed = dict((observation.get("private") or {}).get("shed") or {})
        unit_tiles = copy_tile_grid(farm.get("tiles") or [])
        for unit_index, raw_action in enumerate(unit_actions[row]):
            apply_unit_shed_effect(
                observation,
                unit_index,
                int(raw_action),
                remaining_shed,
                unit_tiles,
            )
            apply_unit_tile_effect(
                observation,
                unit_index,
                int(raw_action),
                unit_tiles,
            )
        ledgers.append(MarketLedger.from_observation(observation, shed=remaining_shed))
    return ledgers


@torch.inference_mode()
def free_running_decode(
    model: ResidualMarketDecoder | None,
    quantity_head: ResidualMarketDecoder,
    data: CachedSplit,
    *,
    shuffled: bool,
    shuffle_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Greedily decode all slots with live MarketLedger masks and prefixes."""

    if data.rows == 0:
        raise ValueError("cannot free-run an empty split")
    rows, slots = data.market_kinds.shape
    ledgers = _initialize_ledgers(data)
    observations = data.observations
    assert observations is not None
    still_active = np.ones(rows, dtype=bool)
    kinds = np.zeros((rows, slots), dtype=np.int64)
    quantities = np.zeros((rows, slots), dtype=np.int64)
    legal = True
    finite = True

    base_kind = data.base_kind_logits.to(device, dtype=torch.float32)
    base_quantity = data.base_quantity_context.to(device, dtype=torch.float32)
    decisions = data.decisions.to(device, dtype=torch.float32)
    quantity_head.to(device).eval()
    if model is not None:
        model.to(device).eval()
    rng = np.random.default_rng(shuffle_seed)
    generated_active_history = np.zeros((rows, slots), dtype=bool)
    own_rows = np.arange(rows, dtype=np.int64)
    for slot in range(slots):
        generated_active_history[:, slot] = still_active
        if model is None:
            slot_kind_logits = base_kind[:, slot]
            slot_quantity_context = base_quantity[:, slot]
        else:
            donors = (
                _control_donors_for_slot(
                    data.corpus_index.numpy(),
                    data.step.numpy(),
                    generated_active_history,
                    data.seed.numpy(),
                    slot,
                    rng,
                )
                if shuffled and slot > 0
                else own_rows
            )
            prefix_kinds = np.full((rows, slot + 1), N_MARKET_KINDS, dtype=np.int64)
            prefix_quantities = np.full((rows, slot + 1), N_QUANTITIES, dtype=np.int64)
            if slot > 0:
                prefix_kinds[:, 1:] = kinds[donors, :slot]
                prefix_quantities[:, 1:] = quantities[donors, :slot]
            slot_kind_logits, slot_quantity_context = model.decode_slot(
                decisions[:, : slot + 1],
                base_kind[:, slot],
                base_quantity[:, slot],
                torch.as_tensor(prefix_kinds, device=device),
                torch.as_tensor(prefix_quantities, device=device),
            )

        kind_masks = np.zeros((rows, N_MARKET_KINDS), dtype=bool)
        for row, observation in enumerate(observations):
            if still_active[row]:
                kind_masks[row] = _ledger_kind_mask(observation, ledgers[row])
            else:
                kind_masks[row, int(MarketKind.STOP)] = True
        kind_logits_np = slot_kind_logits.float().cpu().numpy()
        finite &= bool(np.isfinite(kind_logits_np).all())
        chosen_kinds = np.where(kind_masks, kind_logits_np, -np.inf).argmax(axis=1)
        legal &= bool(kind_masks[np.arange(rows), chosen_kinds].all())
        kinds[:, slot] = chosen_kinds

        quantity_masks = np.zeros((rows, N_QUANTITIES), dtype=bool)
        quantity_active = np.zeros(rows, dtype=bool)
        for row, (observation, raw_kind) in enumerate(zip(observations, chosen_kinds, strict=True)):
            kind = MarketKind(int(raw_kind))
            if not still_active[row] or kind == MarketKind.STOP:
                quantity_masks[row, 0] = True
                still_active[row] = False
            else:
                quantity_masks[row] = _ledger_quantity_mask(observation, kind, ledgers[row])
                quantity_active[row] = kind in QUANTIFIED_MARKET_KINDS
        kind_tensor = torch.as_tensor(chosen_kinds, device=device, dtype=torch.long)
        quantity_logits = quantity_head.quantity_logits(slot_quantity_context, kind_tensor)
        quantity_logits_np = quantity_logits.float().cpu().numpy()
        finite &= bool(np.isfinite(quantity_logits_np).all())
        chosen_quantities = np.where(quantity_masks, quantity_logits_np, -np.inf).argmax(axis=1)
        legal &= bool(quantity_masks[np.arange(rows), chosen_quantities].all())
        quantities[:, slot] = chosen_quantities
        for row, (raw_kind, raw_quantity) in enumerate(
            zip(chosen_kinds, chosen_quantities, strict=True)
        ):
            if not quantity_active[row] and raw_kind == int(MarketKind.STOP):
                continue
            if raw_kind != int(MarketKind.STOP):
                _apply_ledger_order(
                    observations[row],
                    MarketKind(int(raw_kind)),
                    int(QUANTITY_BINS[int(raw_quantity)]),
                    ledgers[row],
                )

    target_kinds = data.market_kinds.numpy()
    target_quantities = data.market_quantities.numpy()
    kind_active = data.market_active.numpy().astype(bool)
    quantity_active_targets = data.market_quantity_active.numpy().astype(bool)
    by_slot = {}
    for slot in range(slots):
        by_slot[str(slot)] = {
            "kind_accuracy": None
            if not kind_active[:, slot].any()
            else float((kinds[:, slot] == target_kinds[:, slot])[kind_active[:, slot]].mean()),
            "quantity_accuracy": None
            if not quantity_active_targets[:, slot].any()
            else float(
                (quantities[:, slot] == target_quantities[:, slot])[
                    quantity_active_targets[:, slot]
                ].mean()
            ),
        }
    queue_match = (kinds == target_kinds).all(axis=1) & (
        (~quantity_active_targets) | (quantities == target_quantities)
    ).all(axis=1)
    return {
        "kinds": kinds,
        "quantities": quantities,
        "queue_match": queue_match,
        "metrics": {
            "all_logits_finite": finite,
            "all_selected_actions_legal": legal,
            "kind_accuracy": None
            if not kind_active.any()
            else float((kinds == target_kinds)[kind_active].mean()),
            "quantity_accuracy": None
            if not quantity_active_targets.any()
            else float((quantities == target_quantities)[quantity_active_targets].mean()),
            "exact_queue_match_rate": float(queue_match.mean()),
            "by_slot": by_slot,
        },
    }


def _free_running_report(result: dict[str, Any]) -> dict[str, Any]:
    return result["metrics"]


def _free_running_comparison(
    true_result: dict[str, Any],
    shuffled_result: dict[str, Any],
    baseline_result: dict[str, Any],
    data: CachedSplit,
) -> dict[str, Any]:
    kind_active = data.market_active.numpy().astype(bool)
    quantity_active = data.market_quantity_active.numpy().astype(bool)
    target_kinds = data.market_kinds.numpy()
    target_quantities = data.market_quantities.numpy()

    def correctness(result: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            (
                (result["kinds"] == target_kinds)[kind_active],
                (result["quantities"] == target_quantities)[quantity_active],
            )
        )

    def relative_error_reduction(true_rate: float, control_rate: float) -> float | None:
        control_error = 1.0 - control_rate
        if control_error == 0:
            return None
        return (control_error - (1.0 - true_rate)) / control_error

    true_correct = correctness(true_result)
    shuffled_correct = correctness(shuffled_result)
    baseline_correct = correctness(baseline_result)

    baseline_errors = ~baseline_correct
    true_queue = np.asarray(true_result["queue_match"], dtype=bool)
    shuffled_queue = np.asarray(shuffled_result["queue_match"], dtype=bool)
    baseline_queue = np.asarray(baseline_result["queue_match"], dtype=bool)
    return {
        "active_decision_accuracy_true_minus_shuffled": float(
            true_correct.mean() - shuffled_correct.mean()
        ),
        "active_decision_baseline_error_correction_rate": None
        if not baseline_errors.any()
        else float(true_correct[baseline_errors].mean()),
        "active_decision_previously_correct_breakage_rate": None
        if not baseline_correct.any()
        else float((~true_correct[baseline_correct]).mean()),
        "exact_queue_match_true_minus_shuffled": float(true_queue.mean() - shuffled_queue.mean()),
        "exact_queue_relative_error_reduction_vs_shuffled": relative_error_reduction(
            float(true_queue.mean()), float(shuffled_queue.mean())
        ),
        "exact_queue_relative_error_reduction_vs_no_prefix_actor": relative_error_reduction(
            float(true_queue.mean()), float(baseline_queue.mean())
        ),
    }


def _prefix_control_report(prefixes: PrefixBatch, rows: int) -> dict[str, Any]:
    sources = prefixes.source_rows[:, 1:]
    fallback = sources == np.arange(rows)[:, None]
    switches = sources[:, 1:] != sources[:, :-1]
    return {
        "unchanged_fallback_prefix_slots": int(fallback.sum()),
        "unchanged_fallback_prefix_slot_fraction": float(fallback.mean()),
        "rows_with_any_fallback": int(fallback.any(axis=1).sum()),
        "donor_switches_between_slots": int(switches.sum()),
        "donor_switch_fraction": float(switches.mean()),
    }


def _count_and_split_report(data: CachedSplit, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": data.rows,
        "episodes": len(data.episodes),
        "market_kind_active": int(data.market_active.sum()),
        "market_quantity_active": int(data.market_quantity_active.sum()),
        "by_corpus": {
            str(record["path"]): {
                "rows": int((data.corpus_index == index).sum()),
                "episodes": sum(int(episode["corpus_index"] == index) for episode in data.episodes),
                "seeds": sorted(
                    {
                        int(episode["seed"])
                        for episode in data.episodes
                        if int(episode["corpus_index"]) == index
                    }
                ),
            }
            for index, record in enumerate(records)
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True, help="structured BC actor artifact")
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="one or more BC corpus directories",
    )
    parser.add_argument(
        "--encoded-cache",
        type=Path,
        default=Path("data/.bc-encoded-cache"),
        help="train_bc-compatible encoded cache root",
    )
    parser.add_argument(
        "--no-encoded-cache",
        action="store_true",
        help="disable reading and writing derived encoded episodes",
    )
    parser.add_argument("--seeds-per-dataset", type=int, default=64)
    parser.add_argument("--holdout-seeds", type=int, default=8)
    parser.add_argument("--validation-holdout-seeds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--encode-workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda", help="torch device, e.g. cuda or cpu")
    parser.add_argument("--output", type=Path, required=True, help="deterministic JSON report path")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> torch.device:
    if not args.actor.is_file():
        raise ValueError(f"actor artifact does not exist: {args.actor}")
    if not args.dataset:
        raise ValueError("at least one dataset is required")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if args.bootstrap_samples <= 0 or args.encode_workers <= 0 or args.torch_threads <= 0:
        raise ValueError("bootstrap_samples, encode_workers, and torch_threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; pass --device cpu")
    return device


@torch.inference_mode()
def _measure_decoder_latency(
    model: ResidualMarketDecoder,
    data: CachedSplit,
    prefixes: PrefixBatch,
    device: torch.device,
) -> float:
    model.eval()
    indices = slice(0, 1)
    decisions = _to_device(data.decisions, indices, device, torch.float32)
    base_kind = _to_device(data.base_kind_logits, indices, device, torch.float32)
    base_quantity = _to_device(data.base_quantity_context, indices, device, torch.float32)
    prefix_kinds = torch.as_tensor(prefixes.kinds[indices], device=device, dtype=torch.long)
    prefix_quantities = torch.as_tensor(
        prefixes.quantities[indices], device=device, dtype=torch.long
    )
    target_kinds = _to_device(data.market_kinds, indices, device, torch.long)

    for _ in range(5):
        _, context = model(
            decisions,
            base_kind,
            base_quantity,
            prefix_kinds,
            prefix_quantities,
        )
        model.quantity_logits(context, target_kinds)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    repeats = 20
    for _ in range(repeats):
        _, context = model(
            decisions,
            base_kind,
            base_quantity,
            prefix_kinds,
            prefix_quantities,
        )
        model.quantity_logits(context, target_kinds)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = validate_args(args)
    train_bc._pin_host_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    shuffle_seed = args.seed + 1 if args.shuffle_seed is None else args.shuffle_seed
    cache_root = None if args.no_encoded_cache else args.encoded_cache
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    loaded, dataset_records = load_probe_dataset(
        args.dataset,
        holdout_seeds=args.holdout_seeds,
        validation_holdout_seeds=args.validation_holdout_seeds,
        seeds_per_dataset=args.seeds_per_dataset,
        encoded_cache=cache_root,
        encode_workers=args.encode_workers,
        torch_threads=args.torch_threads,
    )
    timings["dataset_load_and_encode_seconds"] = time.perf_counter() - started

    actor, artifact = load_actor_artifact(args.actor, device)
    if not architecture_of(actor).structured_inputs:
        raise ValueError("market-prefix probe requires a structured-input actor artifact")
    actor_autocast = device.type == "cuda" and bool(actor.config.fused_mlp)
    initial_model = ResidualMarketDecoder(actor)
    initial_state = copy.deepcopy(initial_model.state_dict())
    true_model = ResidualMarketDecoder(actor)
    shuffled_model = ResidualMarketDecoder(actor)
    true_model.load_state_dict(initial_state)
    shuffled_model.load_state_dict(initial_state)

    started = time.perf_counter()
    cached: dict[str, CachedSplit] = {}
    for partition in ("train", "validation", "test"):
        loaded_split = loaded.pop(partition)
        cached[partition] = cache_actor_features(
            actor,
            loaded_split,
            batch_size=args.batch_size,
            device=device,
            autocast=actor_autocast,
        )
        del loaded_split
    timings["frozen_actor_feature_cache_seconds"] = time.perf_counter() - started
    del loaded, actor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prefixes: dict[str, dict[str, PrefixBatch]] = {}
    for partition, data in cached.items():
        prefix_arguments = (
            data.market_kinds.numpy(),
            data.market_quantities.numpy(),
            data.corpus_index.numpy(),
            data.step.numpy(),
            data.market_active.numpy(),
            data.seed.numpy(),
        )
        prefixes[partition] = {
            "true": construct_prefixes(*prefix_arguments, shuffled=False, seed=shuffle_seed),
            "shuffled": construct_prefixes(*prefix_arguments, shuffled=True, seed=shuffle_seed),
        }

    started = time.perf_counter()
    history = train_paired_decoders(
        true_model,
        shuffled_model,
        cached["train"],
        prefixes["train"]["true"],
        prefixes["train"]["shuffled"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    timings["decoder_training_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    corpus_names = [str(record["path"]) for record in dataset_records]
    teacher_reports: dict[str, Any] = {}
    for partition in ("validation", "test"):
        data = cached[partition]
        outputs = {
            "no_prefix_actor": baseline_teacher_forced_outputs(
                initial_model, data, batch_size=args.batch_size, device=device
            ),
            "true_prefix": teacher_forced_outputs(
                true_model,
                data,
                prefixes[partition]["true"],
                batch_size=args.batch_size,
                device=device,
            ),
            "shuffled_prefix": teacher_forced_outputs(
                shuffled_model,
                data,
                prefixes[partition]["shuffled"],
                batch_size=args.batch_size,
                device=device,
            ),
        }
        teacher_reports[partition] = {
            name: summarize_teacher_outputs(output, data, corpus_names)
            for name, output in outputs.items()
        }
        teacher_reports[partition]["comparison"] = _comparison_metrics(
            outputs["true_prefix"],
            outputs["shuffled_prefix"],
            outputs["no_prefix_actor"],
            data,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    timings["teacher_forced_evaluation_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    test = cached["test"]
    baseline_free = free_running_decode(
        None, initial_model, test, shuffled=False, shuffle_seed=shuffle_seed, device=device
    )
    true_free = free_running_decode(
        true_model,
        true_model,
        test,
        shuffled=False,
        shuffle_seed=shuffle_seed,
        device=device,
    )
    shuffled_free = free_running_decode(
        shuffled_model,
        shuffled_model,
        test,
        shuffled=True,
        shuffle_seed=shuffle_seed,
        device=device,
    )
    true_repeat = free_running_decode(
        true_model,
        true_model,
        test,
        shuffled=False,
        shuffle_seed=shuffle_seed,
        device=device,
    )
    shuffled_repeat = free_running_decode(
        shuffled_model,
        shuffled_model,
        test,
        shuffled=True,
        shuffle_seed=shuffle_seed,
        device=device,
    )
    free_running = {
        "no_prefix_actor": _free_running_report(baseline_free),
        "true_prefix": _free_running_report(true_free),
        "shuffled_prefix": _free_running_report(shuffled_free),
        "true_prefix_repeat_identical": bool(
            np.array_equal(true_free["kinds"], true_repeat["kinds"])
            and np.array_equal(true_free["quantities"], true_repeat["quantities"])
        ),
        "shuffled_prefix_repeat_identical": bool(
            np.array_equal(shuffled_free["kinds"], shuffled_repeat["kinds"])
            and np.array_equal(shuffled_free["quantities"], shuffled_repeat["quantities"])
        ),
        "comparison": _free_running_comparison(true_free, shuffled_free, baseline_free, test),
    }
    timings["free_running_evaluation_seconds"] = time.perf_counter() - started
    latency_ms = _measure_decoder_latency(true_model, test, prefixes["test"]["true"], device)
    timings["total_seconds"] = time.perf_counter() - total_started

    trainable_parameters = sum(parameter.numel() for parameter in true_model.parameters())
    prefix_changed = (
        prefixes["test"]["shuffled"].source_rows[:, 1:] != np.arange(test.rows)[:, None]
    )
    result = {
        "format_version": 1,
        "experiment": "frozen_structured_actor_market_prefix_probe",
        "primary_result": teacher_reports["test"]["comparison"]["primary_paired_seed"],
        "teacher_forced": teacher_reports,
        "free_running_test": free_running,
        "splits": {
            partition: _count_and_split_report(data, dataset_records)
            for partition, data in cached.items()
        },
        "provenance": {
            "actor": {
                "path": str(args.actor.expanduser().resolve()),
                "sha256": _sha256(args.actor),
                "format_version": artifact.get("format_version"),
                "model_config": artifact.get("model_config"),
                "source_identity": artifact.get("source_identity"),
                "run_provenance": artifact.get("run_provenance"),
            },
            "datasets": dataset_records,
        },
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "shuffle_seed": shuffle_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "shuffled_prefix_control_by_split": {
                partition: _prefix_control_report(arm["shuffled"], cached[partition].rows)
                for partition, arm in prefixes.items()
            },
            "frozen_actor_autocast_bfloat16": actor_autocast,
            "device": str(device),
            "encoded_cache": None if cache_root is None else str(cache_root.expanduser().resolve()),
            "shared_initialization": True,
            "common_minibatch_order": True,
            "reset_hidden_per_scored_slot": True,
            "single_donor_coherent_prefix_per_scored_slot": True,
            "fixed_epochs_no_test_selection": True,
            "shuffled_test_prefix_source_fraction_changed": float(prefix_changed.mean()),
        },
        "resource_usage": {
            "trainable_parameters_per_arm": trainable_parameters,
            "decoder_batch1_10_slot_latency_ms": latency_ms,
            "timing": timings,
            "peak_cuda_memory_allocated_bytes": None
            if device.type != "cuda"
            else int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_memory_reserved_bytes": None
            if device.type != "cuda"
            else int(torch.cuda.max_memory_reserved(device)),
        },
        "training_history": history,
    }
    return _jsonable(result)


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"market-prefix probe failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(result["primary_result"], sort_keys=True), flush=True)
    print(f"report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
