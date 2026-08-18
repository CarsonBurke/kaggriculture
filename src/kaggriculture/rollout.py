"""Synchronous self-play rollout collection against the official simulator."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from kaggle_environments import make

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import BOARD_SIZE, CROPS, MAX_MARKET_ORDERS, MAX_UNITS, PRODUCTS
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
    pair_potential,
    shaped_pair_reward,
    terminal_pair_potential,
)
from kaggriculture.model import ActorOutput, FarmActor
from kaggriculture.opponents import BUILTIN_AGENT_ORDER
from kaggriculture.policy import PolicyStep, act_batch
from kaggriculture.registry import CONV_ENTITY, STRUCTURED, architecture_of
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredActor, StructuredInputs
from kaggriculture.tokens import (
    CROP_PRIVATE_FIELDS,
    CROP_TOKEN_FIELDS,
    FARM_TOKEN_FIELDS,
    N_TILE_CATEGORICAL,
    N_TILE_CONTINUOUS,
    N_UNIT_CATEGORICAL,
    N_UNIT_CONTINUOUS,
    PRODUCT_PRIVATE_FIELDS,
    PRODUCT_TOKEN_FIELDS,
    TILE_COUNT,
    TOWN_TOKEN_FIELDS,
    UNIT_TILE_GATHERS,
)

_MAX_FLOAT32_CATEGORICAL_DRAW = np.nextafter(np.float32(1.0), np.float32(0.0))

# Per-row codes for the wave's `builtin_agents` argument: 0 samples the row
# from the network, anything else hands the row to the named engine reference
# agent inside Rust. Mirrors `BuiltinAgent::from_code` in rust/kagg_env.
BUILTIN_AGENT_CODES = {name: code for code, name in enumerate(BUILTIN_AGENT_ORDER, start=1)}


@dataclass(frozen=True)
class RolloutBatch:
    """Behavior rollout: architecture-specific state arrays plus shared factors."""

    architecture: str
    states: dict[str, np.ndarray]
    unit_actions: np.ndarray
    market_kinds: np.ndarray
    market_quantities: np.ndarray
    unit_masks: np.ndarray
    market_kind_masks: np.ndarray
    market_quantity_masks: np.ndarray
    unit_active: np.ndarray
    market_active: np.ndarray
    market_quantity_active: np.ndarray
    old_unit_logprobs: np.ndarray
    old_market_kind_logprobs: np.ndarray
    old_market_quantity_logprobs: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    episode_seeds: np.ndarray
    final_money: np.ndarray
    opponent_money: np.ndarray
    seats: np.ndarray
    entropy_sums: np.ndarray
    elapsed_seconds: float

    @property
    def trajectories(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def state_count(self) -> int:
        return int(self.valid.sum())

    @property
    def mean_entropy(self) -> float:
        """Mean behavior entropy per active policy component."""
        components = int(
            self.unit_active.sum() + self.market_active.sum() + self.market_quantity_active.sum()
        )
        return float(self.entropy_sums.sum() / max(1, components))


def _trajectory_first(values: list[np.ndarray], dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.stack(values, axis=1)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def _observations(states: list[list[Any]]) -> list[dict[str, Any]]:
    return [agent.observation for state in states for agent in state]


def _opponent_privates(states: list[list[Any]]) -> list[dict[str, Any]]:
    out = []
    for state in states:
        out.extend((state[1].observation["private"], state[0].observation["private"]))
    return out


def _state_field_specs(architecture: str) -> dict[str, tuple[tuple[int, ...], type]]:
    """Per-state shape and staging dtype of every architecture state field.

    Structured rollouts persist the exact staging layout of
    ``StructuredObservation`` plus the centralized-critic extras. The extras
    come from the opponent seat's viewpoint, so native collection derives
    them from the paired row's buffers instead of encoding them twice.
    """
    if architecture == CONV_ENTITY:
        return {
            "board": ((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), np.float16),
            "global_features": ((GLOBAL_FEATURES,), np.float16),
            "critic_features": ((CRITIC_FEATURES,), np.float16),
            "units": ((MAX_UNITS, UNIT_FEATURES), np.float16),
            "unit_positions": ((MAX_UNITS, 2), np.int8),
        }
    if architecture == STRUCTURED:
        gathers = len(UNIT_TILE_GATHERS)
        return {
            "tile_categorical": ((2 * TILE_COUNT, N_TILE_CATEGORICAL), np.int8),
            "tile_continuous": ((2 * TILE_COUNT, N_TILE_CONTINUOUS), np.float16),
            "unit_categorical": ((MAX_UNITS, N_UNIT_CATEGORICAL), np.int8),
            "unit_continuous": ((MAX_UNITS, N_UNIT_CONTINUOUS), np.float16),
            "unit_tile_gather": ((MAX_UNITS, gathers), np.int8),
            "unit_tile_gather_valid": ((MAX_UNITS, gathers), np.bool_),
            "products": ((len(PRODUCTS), len(PRODUCT_TOKEN_FIELDS)), np.float16),
            "crops": ((len(CROPS), len(CROP_TOKEN_FIELDS)), np.float16),
            "farms": ((2, len(FARM_TOKEN_FIELDS)), np.float16),
            "town": ((len(TOWN_TOKEN_FIELDS),), np.float16),
            "opponent_unit_categorical": ((MAX_UNITS, N_UNIT_CATEGORICAL), np.int8),
            "opponent_unit_continuous": ((MAX_UNITS, N_UNIT_CONTINUOUS), np.float16),
            "opponent_unit_active": ((MAX_UNITS,), np.bool_),
            "critic_products": ((len(PRODUCTS), len(PRODUCT_PRIVATE_FIELDS)), np.float16),
            "critic_crops": ((len(CROPS), len(CROP_PRIVATE_FIELDS)), np.float16),
        }
    raise ValueError(f"unknown rollout architecture {architecture!r}")


_SHARED_FIELD_SPECS: dict[str, tuple[tuple[int, ...], type]] = {
    "unit_actions": ((MAX_UNITS,), np.int8),
    "market_kinds": ((MAX_MARKET_ORDERS,), np.int8),
    "market_quantities": ((MAX_MARKET_ORDERS,), np.int8),
    "unit_masks": ((MAX_UNITS, N_UNIT_ACTIONS), np.bool_),
    "market_kind_masks": ((MAX_MARKET_ORDERS, N_MARKET_KINDS), np.bool_),
    "market_quantity_masks": ((MAX_MARKET_ORDERS, N_QUANTITIES), np.bool_),
    "unit_active": ((MAX_UNITS,), np.bool_),
    "market_active": ((MAX_MARKET_ORDERS,), np.bool_),
    "market_quantity_active": ((MAX_MARKET_ORDERS,), np.bool_),
    "old_unit_logprobs": ((MAX_UNITS,), np.float32),
    "old_market_kind_logprobs": ((MAX_MARKET_ORDERS,), np.float32),
    "old_market_quantity_logprobs": ((MAX_MARKET_ORDERS,), np.float32),
    "rewards": ((), np.float32),
    "valid": ((), np.bool_),
}

_SHARED_ROLLOUT_FIELDS = tuple(_SHARED_FIELD_SPECS)

# Opponent-viewpoint columns the centralized critic reads from the paired
# row's economy tokens: their own shed/carried product stock and seed counts.
_PRODUCT_STOCK_COLUMNS = slice(
    PRODUCT_TOKEN_FIELDS.index("shed_stock"), PRODUCT_TOKEN_FIELDS.index("carried_stock") + 1
)
_CROP_SEED_COLUMNS = slice(
    CROP_TOKEN_FIELDS.index("seeds_held"), CROP_TOKEN_FIELDS.index("seeds_held") + 1
)


def _rollout_array(batch: RolloutBatch, field: str) -> np.ndarray:
    return batch.states[field] if field in batch.states else getattr(batch, field)


def _new_fields(architecture: str) -> dict[str, list[np.ndarray]]:
    return {name: [] for name in (*_state_field_specs(architecture), *_SHARED_ROLLOUT_FIELDS)}


def _record_policy_step(
    architecture: str, fields: dict[str, list[np.ndarray]], policy_step: PolicyStep
) -> None:
    factors = policy_step.factors
    for name, (_, dtype) in _state_field_specs(architecture).items():
        rows = [getattr(row, name) for row in policy_step.encoded]
        if any(row is None for row in rows):
            raise ValueError("rollout collection requires opponent private state on every row")
        fields[name].append(np.stack(rows).astype(dtype, copy=False))
    fields["unit_actions"].append(factors.unit_actions.astype(np.int8))
    fields["market_kinds"].append(factors.market_kinds.astype(np.int8))
    fields["market_quantities"].append(factors.market_quantities.astype(np.int8))
    fields["unit_masks"].append(factors.unit_masks)
    fields["market_kind_masks"].append(factors.market_kind_masks)
    fields["market_quantity_masks"].append(factors.market_quantity_masks)
    fields["unit_active"].append(factors.unit_active)
    fields["market_active"].append(factors.market_active)
    fields["market_quantity_active"].append(factors.market_quantity_active)
    fields["old_unit_logprobs"].append(factors.unit_logprobs)
    fields["old_market_kind_logprobs"].append(factors.market_kind_logprobs)
    fields["old_market_quantity_logprobs"].append(factors.market_quantity_logprobs)


def _finish_rollout(
    architecture: str,
    fields: dict[str, list[np.ndarray]],
    *,
    episode_seeds: np.ndarray,
    final_money: np.ndarray,
    opponent_money: np.ndarray,
    seats: np.ndarray,
    entropy_sums: np.ndarray,
    started: float,
) -> RolloutBatch:
    return RolloutBatch(
        architecture=architecture,
        states={name: _trajectory_first(fields[name]) for name in _state_field_specs(architecture)},
        **{
            name: _trajectory_first(fields[name], dtype)
            for name, (_, dtype) in _SHARED_FIELD_SPECS.items()
        },
        episode_seeds=episode_seeds,
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        entropy_sums=entropy_sums,
        elapsed_seconds=time.perf_counter() - started,
    )


def _native_field_specs(
    architecture: str, trajectories: int, horizon: int
) -> dict[str, tuple[tuple[int, ...], type]]:
    """Return the trajectory-major shape and dtype of every native rollout field."""
    prefix = (trajectories, horizon)
    return {
        name: ((*prefix, *shape), dtype)
        for name, (shape, dtype) in {
            **_state_field_specs(architecture),
            **_SHARED_FIELD_SPECS,
        }.items()
    }


_TORCH_STORAGE_DTYPES = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.bool_): torch.bool,
}


def allocate_rollout_storage(
    architecture: str, trajectories: int, horizon: int, *, pin_memory: bool = False
) -> dict[str, np.ndarray]:
    """Allocate reusable trajectory-major rollout storage.

    Pinned storage is allocated through page-locked torch tensors and exposed
    as NumPy views, so replay staging can upload the complete rollout to the
    accelerator asynchronously instead of through pageable-memory copies.
    """
    if trajectories < 1 or horizon < 1:
        raise ValueError("rollout storage requires positive trajectories and horizon")
    storage: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in _native_field_specs(architecture, trajectories, horizon).items():
        if pin_memory:
            tensor = torch.empty(
                shape, dtype=_TORCH_STORAGE_DTYPES[np.dtype(dtype)], pin_memory=True
            )
            storage[name] = tensor.numpy()
        else:
            storage[name] = np.empty(shape, dtype=dtype)
    storage["valid"][:] = True
    return storage


def _native_rollout_storage(
    storage: dict[str, np.ndarray] | None, architecture: str, trajectories: int, horizon: int
) -> dict[str, np.ndarray]:
    """Validate caller-provided storage or allocate a fresh full-horizon block."""
    if storage is None:
        return allocate_rollout_storage(architecture, trajectories, horizon)
    specs = _native_field_specs(architecture, trajectories, horizon)
    if set(storage) != set(specs):
        raise ValueError("rollout storage fields do not match the native layout")
    for name, (shape, dtype) in specs.items():
        array = storage[name]
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"rollout storage field {name} has the wrong shape or dtype")
    storage["valid"][:] = True
    return storage


def _quantity_heads(
    actors: tuple[FarmActor | StructuredActor, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the small selected-kind quantity heads once per rollout."""

    def parameter(actor: FarmActor | StructuredActor, name: str) -> np.ndarray:
        value = getattr(actor, name)
        if hasattr(value, "weight"):
            value = value.weight
        return value.detach().float().cpu().numpy()

    return (
        np.ascontiguousarray(
            np.stack([parameter(actor, "market_quantity_kind_gate") for actor in actors])
        ),
        np.ascontiguousarray(
            np.stack([parameter(actor, "market_quantity_value") for actor in actors])
        ),
        np.ascontiguousarray(
            np.stack([parameter(actor, "market_quantity_bias") for actor in actors])
        ),
    )


def _select_inputs(inputs: tuple[Any, ...], rows: torch.Tensor) -> tuple[Any, ...]:
    """Gather rows of a model-argument tuple, recursing into token bundles."""
    return tuple(
        type(entry)(*(tensor.index_select(0, rows) for tensor in entry))
        if isinstance(entry, tuple)
        else entry.index_select(0, rows)
        for entry in inputs
    )


def _lane_view_inputs(
    inputs: tuple[Any, ...], rows: torch.Tensor, lanes: int, width: int
) -> tuple[Any, ...]:
    """Gather ensemble rows and fold them into [lanes, width, ...] shapes."""

    def folded(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(0, rows).view(lanes, width, *tensor.shape[1:])

    return tuple(
        type(entry)(*(folded(tensor) for tensor in entry))
        if isinstance(entry, tuple)
        else folded(entry)
        for entry in inputs
    )


def _leading_tensor(inputs: tuple[Any, ...]) -> torch.Tensor:
    head = inputs[0]
    while isinstance(head, tuple):
        head = head[0]
    return head


@dataclass(frozen=True)
class _NativeEncodedWave:
    arrays: dict[str, np.ndarray]
    host_features: torch.Tensor
    host_positions: torch.Tensor
    staged_features: torch.Tensor
    device_features: torch.Tensor
    board: torch.Tensor
    global_features: torch.Tensor
    critic_features: torch.Tensor
    units: torch.Tensor
    unit_positions: torch.Tensor

    def copy_to_device(self) -> None:
        """Upload the encoded block, then widen it in a separate device kernel.

        The Rust encoder emits fp16 and the model consumes fp32. A single
        cross-dtype `copy_` does not transfer fp32 across the bus: ATen's
        `copy_requires_temporaries` sends a CPU-to-CUDA copy with mismatched
        dtypes down a staged path that allocates a device temporary, transfers at
        the source dtype, then widens on the device. This spells that out with a
        buffer owned by the wave, so the temporary is allocated once at
        construction instead of per step, and both destinations are persistent,
        which keeps the captured rollout graph's input addresses static.

        The two steps are therefore value-identical to the one they replace. The
        matched-dtype transfer is measurably cheaper than the mismatched one on
        this box, but that gap is not yet attributed to a mechanism, so treat the
        split as a clarification whose payoff is still being measured.
        """
        non_blocking = self.device_features.device.type == "cuda"
        self.staged_features.copy_(self.host_features, non_blocking=non_blocking)
        self.device_features.copy_(self.staged_features, non_blocking=non_blocking)
        self.unit_positions.copy_(self.host_positions, non_blocking=non_blocking)

    def refresh(self, environment: Any) -> None:
        environment.encoded_into(self.arrays)

    def inputs(self) -> tuple[torch.Tensor, ...]:
        return (self.board, self.global_features, self.units, self.unit_positions)


def _native_encoded_wave(environment: Any, device: torch.device) -> _NativeEncodedWave:
    """Build reusable Rust output plus packed, optionally pinned model input buffers."""
    arrays = {name: np.asarray(value) for name, value in environment.encoded_buffers().items()}
    feature_names = ("board", "global_features", "critic_features", "units")
    feature_sizes = [arrays[name].size for name in feature_names]
    pin_memory = device.type == "cuda"
    host_features = torch.empty(sum(feature_sizes), dtype=torch.float16, pin_memory=pin_memory)
    staged_features = torch.empty(sum(feature_sizes), dtype=host_features.dtype, device=device)
    device_features = torch.empty(sum(feature_sizes), dtype=torch.float32, device=device)
    device_views: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, size in zip(feature_names, feature_sizes, strict=True):
        shape = arrays[name].shape
        host_view = host_features[cursor : cursor + size].reshape(shape)
        device_views[name] = device_features[cursor : cursor + size].reshape(shape)
        arrays[name] = host_view.numpy()
        cursor += size

    host_positions = torch.empty(
        arrays["unit_positions"].shape,
        dtype=torch.long,
        pin_memory=pin_memory,
    )
    arrays["unit_positions"] = host_positions.numpy()
    unit_positions = torch.empty(host_positions.shape, dtype=torch.long, device=device)
    return _NativeEncodedWave(
        arrays=arrays,
        host_features=host_features,
        host_positions=host_positions,
        staged_features=staged_features,
        device_features=device_features,
        board=device_views["board"],
        global_features=device_views["global_features"],
        critic_features=device_views["critic_features"],
        units=device_views["units"],
        unit_positions=unit_positions,
    )


# Host-to-device transport groups for the structured wave: every group packs
# into one pinned block so a wave costs three asynchronous uploads.
_STRUCTURED_CONTINUOUS_BUFFERS = (
    "tile_continuous",
    "unit_continuous",
    "products",
    "crops",
    "farms",
    "town",
)
_STRUCTURED_CATEGORICAL_BUFFERS = ("tile_categorical", "unit_categorical", "unit_tile_gather")
_STRUCTURED_FLAG_BUFFERS = ("unit_active", "unit_tile_gather_valid")


@dataclass(frozen=True)
class _NativeStructuredWave:
    arrays: dict[str, np.ndarray]
    host_continuous: torch.Tensor
    host_categorical: torch.Tensor
    host_flags: torch.Tensor
    device_continuous: torch.Tensor
    device_categorical: torch.Tensor
    device_flags: torch.Tensor
    device_inputs: StructuredInputs

    def copy_to_device(self) -> None:
        non_blocking = self.device_continuous.device.type == "cuda"
        self.device_continuous.copy_(self.host_continuous, non_blocking=non_blocking)
        self.device_categorical.copy_(self.host_categorical, non_blocking=non_blocking)
        self.device_flags.copy_(self.host_flags, non_blocking=non_blocking)

    def refresh(self, environment: Any) -> None:
        environment.structured_into(self.arrays)

    def inputs(self) -> tuple[StructuredInputs]:
        return (self.device_inputs,)


def _native_structured_wave(environment: Any, device: torch.device) -> _NativeStructuredWave:
    """Build reusable structured Rust output plus packed device token tensors.

    Categorical indices upload straight from their int8 staging bytes into
    int64 embedding-index tensors; continuous features upload from float16
    staging into the float32 the model consumes.
    """
    arrays = {name: np.asarray(value) for name, value in environment.structured_buffers().items()}
    pin_memory = device.type == "cuda"

    def packed(
        names: tuple[str, ...], host_dtype: torch.dtype, device_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        sizes = [arrays[name].size for name in names]
        host = torch.empty(sum(sizes), dtype=host_dtype, pin_memory=pin_memory)
        block = torch.empty(sum(sizes), dtype=device_dtype, device=device)
        views: dict[str, torch.Tensor] = {}
        cursor = 0
        for name, size in zip(names, sizes, strict=True):
            shape = arrays[name].shape
            views[name] = block[cursor : cursor + size].reshape(shape)
            arrays[name] = host[cursor : cursor + size].reshape(shape).numpy()
            cursor += size
        return host, block, views

    host_continuous, device_continuous, views = packed(
        _STRUCTURED_CONTINUOUS_BUFFERS, torch.float16, torch.float32
    )
    host_categorical, device_categorical, categorical_views = packed(
        _STRUCTURED_CATEGORICAL_BUFFERS, torch.int8, torch.int64
    )
    host_flags, device_flags, flag_views = packed(_STRUCTURED_FLAG_BUFFERS, torch.bool, torch.bool)
    views |= categorical_views | flag_views
    return _NativeStructuredWave(
        arrays=arrays,
        host_continuous=host_continuous,
        host_categorical=host_categorical,
        host_flags=host_flags,
        device_continuous=device_continuous,
        device_categorical=device_categorical,
        device_flags=device_flags,
        device_inputs=StructuredInputs(**{name: views[name] for name in StructuredInputs._fields}),
    )


def _native_wave(
    architecture: str, environment: Any, device: torch.device
) -> _NativeEncodedWave | _NativeStructuredWave:
    if architecture == CONV_ENTITY:
        return _native_encoded_wave(environment, device)
    return _native_structured_wave(environment, device)


@dataclass(frozen=True)
class _HostActorOutput:
    unit_logits: np.ndarray
    market_kind_logits: np.ndarray
    market_quantity_context: np.ndarray


@dataclass(frozen=True)
class _PackedTransfer:
    """Persistent staging pair for the single D2H copy of a collection step.

    `device` is a flat fp32 block on the model's device and `host` is its pinned
    mirror. Both are allocated once per collection call and reused by every
    step, so the steady-state transfer path performs no allocation at all.
    """

    device: torch.Tensor
    host: torch.Tensor


def _packed_outputs_to_host(
    outputs: tuple[ActorOutput, ...],
    transfer: _PackedTransfer | None,
) -> tuple[list[_HostActorOutput], _PackedTransfer | None]:
    """Move every policy head into pinned host memory with one device sync.

    This stage measures 2.42 ms of the 17.72 ms collection step (11.9%) and runs
    719 times per iteration. Most of that was not transfer: the previous
    implementation widened each of the three heads with `.float()` and then
    concatenated them, so a step allocated four short-lived device tensors and
    read every logit twice before the D2H copy even started. Under bf16 autocast
    the `.float()` calls are real conversion kernels rather than no-ops.

    A persistent packed device block removes both costs. `copy_` into a slice of
    that block performs the dtype widen as part of the placement, so the separate
    `.float()` disappears and the concatenation has nothing left to do: the heads
    land directly in the memory the D2H reads. The pair is allocated on the first
    step of a collection call and reused by every later step; the element-count
    guard reallocates it should a block ever be carried into a wave of a
    different width.
    """
    if outputs[0].unit_logits.device.type == "cpu":
        return (
            [
                _HostActorOutput(
                    output.unit_logits.float().numpy(),
                    output.market_kind_logits.float().numpy(),
                    output.market_quantity_context.float().numpy(),
                )
                for output in outputs
            ],
            transfer,
        )

    heads = [
        tensor
        for output in outputs
        for tensor in (
            output.unit_logits,
            output.market_kind_logits,
            output.market_quantity_context,
        )
    ]
    total = sum(tensor.numel() for tensor in heads)
    if transfer is None or transfer.device.numel() != total:
        transfer = _PackedTransfer(
            device=torch.empty(total, dtype=torch.float32, device=heads[0].device),
            host=torch.empty(total, dtype=torch.float32, device="cpu", pin_memory=True),
        )
    flat = transfer.host.numpy()

    cursor = 0
    arrays: list[np.ndarray] = []
    for tensor in heads:
        size = tensor.numel()
        shape = tuple(tensor.shape)
        transfer.device[cursor : cursor + size].view(shape).copy_(tensor)
        arrays.append(flat[cursor : cursor + size].reshape(shape))
        cursor += size
    transfer.host.copy_(transfer.device, non_blocking=True)
    # Native sampling consumes the host buffer immediately. This is the
    # single required D2H synchronization for the complete model wave.
    torch.cuda.current_stream(heads[0].device).synchronize()
    host_outputs = [
        _HostActorOutput(*arrays[index : index + 3]) for index in range(0, 3 * len(outputs), 3)
    ]
    return host_outputs, transfer


#: Execution mode for the collection forward. Collection spends roughly two
#: thirds of its wall clock in this one call, so the choice here is the single
#: largest lever on rollout cost -- and the shipped default was measurably the
#: wrong one on both axes it trades off.
#:
#: Isolated forward, median of 60 on the production 224-row wave over identical
#: persistent input buffers (artifacts/benchmarks/backends-*.json):
#:
#:     mode                     fp32       bf16
#:     eager                    4.907 ms   4.396 ms
#:     cudagraphs               5.309 ms   4.268 ms
#:     inductor                 2.720 ms   1.626 ms
#:
#: `cudagraphs` is a pessimization in fp32: slower than not compiling at all,
#: which is why enabling it only ever moved the measured rollout by 3.4%. The
#: forward issues ~859 kernels whose summed device time is well under the
#: measured wall clock, so the cost is per-kernel overhead; removing launch cost
#: alone does not touch it, and Inductor's fusion reduces the kernel count.
#:
#: Every mode perturbs the sampled behavior policy relative to the distribution
#: the update path reconstructs, and `update_replay_parity` gates exactly that.
#: Faster is therefore not automatically admissible -- but here the two axes
#: agree, because the drift is dominated by *systematic* differences between the
#: collection and update paths rather than by rounding noise, and the update path
#: is already Inductor plus bf16. Matching it cancels most of the difference.
#: Shipped gate, 4 waves, production league-mixed path, cloned conv actor
#: (artifacts/benchmarks/parity-*.jsonl; bounds 5e-3 / 2e-4 / 1.1e-1):
#:
#:     configuration     max_kl      tail       first_minibatch_kl   rollout
#:     eager / fp32      1.9089e-3   3.185e-5   4.0598e-2            8.91 s
#:     inductor / bf16   2.2786e-4   0.0        4.9855e-3            5.36 s
#:
#: So the selected configuration is 1.66x faster with 8.4x lower drift. Note
#: `eager` with bf16 alone measures 2.908e-3, worse than fp32: it is the
#: matching that pays, not the precision. The mode stays a named knob the audit
#: must be told rather than a silent default, and the library defaults below
#: preserve the previous behavior so `benchmark_rust_rollout.py` and its
#: recorded drift artifacts stay comparable; the training and calibration
#: entrypoints state the measured decision explicitly.
ROLLOUT_FORWARD_MODES = ("eager", "cudagraphs", "inductor")


def _cached_compiled_forward(model: FarmActor | StructuredActor, mode: str = "cudagraphs") -> Any:
    """Return the cached compiled collection forward for `mode`.

    CUDA convolution and GEMM kernels are not bitwise invariant across eager,
    graph, and fused execution, nor across batch shapes, so any mode here moves
    the behavior policy relative to the update-path replay. That drift is bounded
    semantically rather than bitwise, by `update_replay_parity`.

    `reduce-overhead` is what makes the Inductor mode worth its drift: plain
    Inductor keeps the launch cost this forward is dominated by, so the fused
    kernels alone would not pay. One compiled callable is cached per mode so a
    parity audit can measure several in one process without recompiling.
    """
    if mode not in ROLLOUT_FORWARD_MODES:
        raise ValueError(f"unknown rollout forward mode {mode!r}")
    cache = getattr(model, "_kaggriculture_rollout_forwards", None)
    if cache is None:
        cache = {}
        # Avoid registering compiled wrappers as child modules, which would
        # pollute checkpoints with a second copy of every parameter.
        object.__setattr__(model, "_kaggriculture_rollout_forwards", cache)
    compiled = cache.get(mode)
    if compiled is None:
        if mode == "cudagraphs":
            compiled = torch.compile(
                model.forward, backend="cudagraphs", fullgraph=True, dynamic=False
            )
        else:
            compiled = torch.compile(
                model.forward, mode="reduce-overhead", fullgraph=True, dynamic=False
            )
        cache[mode] = compiled
    return compiled


def _rollout_model_forward(
    model: FarmActor | StructuredActor,
    *inputs: Any,
    mode: str = "cudagraphs",
) -> ActorOutput | torch.Tensor:
    """Run one static rollout wave through the execution mode `mode` names.

    The mode is the only thing consulted here. An earlier shape took a separate
    `compile_model` boolean as well, which could veto the mode: a caller stating
    `inductor` while that boolean was false silently got an eager forward, and a
    run's provenance would record a configuration it did not execute. One knob
    cannot contradict itself, so `eager` is spelled as a mode rather than as the
    absence of a flag. The frozen league ensemble takes the same mode, so a
    league wave cannot end up compiling one of its two forwards and not the
    other.
    """
    if mode == "eager" or _leading_tensor(inputs).device.type != "cuda":
        return model(*inputs)
    return _cached_compiled_forward(model, mode)(*inputs)


class _StackedFrozenEnsemble:
    """One batched forward over every frozen league seat via stacked weights.

    League opponents share an architecture but not weights. Stacking their
    parameters lane-wise and running a single vmapped functional call replaces
    the per-opponent forward loop, so the whole frozen side of a mixed wave is
    one large kernel sequence instead of several small ones. Instances persist
    for the process and are refilled in place each collection call: a captured
    CUDA graph keeps reading current weights at stable addresses without any
    per-step parameter copies.
    """

    def __init__(self, models: Sequence[FarmActor | StructuredActor]) -> None:
        self.template = type(models[0])(models[0].config).to("meta")
        self.template.eval()
        import torch._dynamo

        # The stacked tensors outlive any inference-mode region the collector
        # runs under; inference tensors would reject the in-place `load`
        # refills on later calls made outside that region.
        with torch.inference_mode(False):
            self.params = self._stacked("named_parameters", models)
            self.buffers = self._stacked("named_buffers", models)
        for tensor in (*self.params.values(), *self.buffers.values()):
            torch._dynamo.mark_static_address(tensor)
        self._compiled: dict[tuple[str, int], Any] = {}

    @staticmethod
    def _stacked(
        source: str, models: Sequence[FarmActor | StructuredActor]
    ) -> dict[str, torch.Tensor]:
        states = [dict(getattr(model, source)()) for model in models]
        return {name: torch.stack([state[name].detach() for state in states]) for name in states[0]}

    def load(self, models: Sequence[FarmActor | StructuredActor]) -> None:
        for source, stacked_group in (
            ("named_parameters", self.params),
            ("named_buffers", self.buffers),
        ):
            for lane, model in enumerate(models):
                state = dict(getattr(model, source)())
                for name, stacked in stacked_group.items():
                    stacked[lane].copy_(state[name])

    def _forward(self, *inputs: Any) -> ActorOutput:
        def run(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            *inner: Any,
        ) -> ActorOutput:
            return torch.func.functional_call(self.template, (params, buffers), inner)

        return torch.vmap(run)(self.params, self.buffers, *inputs)

    def __call__(self, *inputs: Any, mode: str) -> ActorOutput:
        """Run every frozen league seat under the same mode as the learner.

        A league wave runs this forward once per step in addition to the
        learner's, so leaving it on a backend the learner abandoned would cap the
        collection speedup at whatever fraction of steps are pure self-play. It
        takes the same mode for the same measured reason: `cudagraphs` is slower
        here than not compiling, because this forward is limited by per-kernel
        overhead rather than launch cost, and only fusion reduces the kernel
        count.
        """
        leading = _leading_tensor(inputs)
        if mode == "eager" or leading.device.type != "cuda":
            return self._forward(*inputs)
        # One compiled callable per lane width and mode: league assignments may
        # change the padded width between waves, and sharing one callable would
        # burn through Dynamo's per-code recompile budget before falling back to
        # eager silently.
        key = (mode, leading.shape[1])
        compiled = self._compiled.get(key)
        if compiled is None:
            if mode == "cudagraphs":
                compiled = torch.compile(
                    self._forward, backend="cudagraphs", fullgraph=True, dynamic=False
                )
            else:
                compiled = torch.compile(
                    self._forward, mode="reduce-overhead", fullgraph=True, dynamic=False
                )
            self._compiled[key] = compiled
        return compiled(*inputs)


_FROZEN_ENSEMBLE_CACHE: dict[tuple[Any, ...], _StackedFrozenEnsemble] = {}


def _stacked_frozen_ensemble(
    models: Sequence[FarmActor | StructuredActor],
) -> _StackedFrozenEnsemble:
    """Fetch or build the persistent stacked ensemble for these league lanes."""
    key = (type(models[0]), models[0].config, len(models), next(models[0].parameters()).device)
    ensemble = _FROZEN_ENSEMBLE_CACHE.get(key)
    if ensemble is None:
        ensemble = _StackedFrozenEnsemble(models)
        _FROZEN_ENSEMBLE_CACHE[key] = ensemble
    else:
        ensemble.load(models)
    return ensemble


def _mark_cuda_graph_step(device: torch.device, enabled: bool) -> None:
    if not enabled or device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


def _categorical_draws(
    generator: np.random.Generator, rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw in the same autoregressive component order as the Python sampler."""
    units = np.empty((rows, MAX_UNITS), dtype=np.float32)
    kinds = np.empty((rows, MAX_MARKET_ORDERS), dtype=np.float32)
    quantities = np.empty((rows, MAX_MARKET_ORDERS), dtype=np.float32)
    for unit in range(MAX_UNITS):
        units[:, unit] = generator.random(rows)
    for slot in range(MAX_MARKET_ORDERS):
        kinds[:, slot] = generator.random(rows)
        quantities[:, slot] = generator.random(rows)
    # NumPy generates f64 by default. Values in the last half-ULP below one
    # round to exactly 1.0 when assigned into these f32 transport buffers, but
    # a categorical uniform must remain in [0, 1). Preserve the generator's
    # f64 stream and clamp only the transport representation.
    for draws in (units, kinds, quantities):
        np.minimum(draws, _MAX_FLOAT32_CATEGORICAL_DRAW, out=draws)
    return units, kinds, quantities


def _builtin_agent_rows(
    rows: int,
    frozen_rows: np.ndarray,
    learner_rows: np.ndarray,
    assignments: np.ndarray,
    lane_codes: np.ndarray,
    lane_names: Sequence[str],
) -> np.ndarray:
    """Per-row built-in codes for the wave, refusing any on a learner seat.

    The native binding takes a code for every row and cannot tell a learner
    row from an opponent row, so it deliberately checks nothing here. This
    side knows the seats. A built-in code that landed on a learner row would
    train the policy on an action it never chose, under log-probabilities the
    binding writes as zero — a corruption that looks exactly like ordinary
    data, so it has to be an error rather than a surprise in the journal.
    """
    codes = np.zeros(rows, dtype=np.uint8)
    codes[frozen_rows] = lane_codes[assignments]
    violations = np.flatnonzero(codes[learner_rows])
    if violations.size:
        row = int(learner_rows[violations[0]])
        lane = int(assignments[int(np.flatnonzero(frozen_rows == row)[0])])
        raise ValueError(
            f"built-in lane {lane} ({lane_names[lane]}) was assigned to learner row {row}"
        )
    return codes


def _validate_learner_temperature(temperature: float) -> None:
    """Require behavior logits to match the unit-temperature PPO replay policy."""
    if not np.isfinite(temperature) or temperature != 1.0:
        raise ValueError("on-policy rollout collection requires learner temperature 1.0")


_SAMPLED_FIELD_SOURCES = {
    "unit_actions": "unit_actions",
    "market_kinds": "market_kinds",
    "market_quantities": "market_quantities",
    "unit_masks": "unit_masks",
    "market_kind_masks": "market_kind_masks",
    "market_quantity_masks": "market_quantity_masks",
    "unit_active": "unit_active",
    "market_active": "market_active",
    "market_quantity_active": "market_quantity_active",
    "old_unit_logprobs": "unit_logprobs",
    "old_market_kind_logprobs": "market_kind_logprobs",
    "old_market_quantity_logprobs": "market_quantity_logprobs",
}

_CONV_ENCODED_FIELDS = ("board", "global_features", "critic_features", "units", "unit_positions")
_STRUCTURED_ENCODED_FIELDS = (
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


def _store_native_wave(
    architecture: str,
    fields: dict[str, np.ndarray],
    step: int,
    encoded: dict[str, np.ndarray],
    sampled: dict[str, np.ndarray],
    rewards: np.ndarray,
    rows: np.ndarray | slice,
    pair_rows: np.ndarray,
) -> None:
    if architecture == CONV_ENTITY:
        for name in _CONV_ENCODED_FIELDS:
            fields[name][:, step] = np.asarray(encoded[name])[rows]
    else:
        for name in _STRUCTURED_ENCODED_FIELDS:
            fields[name][:, step] = np.asarray(encoded[name])[rows]
        # Centralized-critic extras are the paired seat's own view of the same
        # buffers, so no second encoding pass exists anywhere.
        fields["opponent_unit_categorical"][:, step] = np.asarray(encoded["unit_categorical"])[
            pair_rows
        ]
        fields["opponent_unit_continuous"][:, step] = np.asarray(encoded["unit_continuous"])[
            pair_rows
        ]
        fields["opponent_unit_active"][:, step] = np.asarray(encoded["unit_active"])[pair_rows]
        fields["critic_products"][:, step] = np.asarray(encoded["products"])[pair_rows][
            :, :, _PRODUCT_STOCK_COLUMNS
        ]
        fields["critic_crops"][:, step] = np.asarray(encoded["crops"])[pair_rows][
            :, :, _CROP_SEED_COLUMNS
        ]
    for destination, source in _SAMPLED_FIELD_SOURCES.items():
        fields[destination][:, step] = np.asarray(sampled[source])[rows]
    fields["rewards"][:, step] = rewards


def _native_batch(
    architecture: str,
    fields: dict[str, np.ndarray],
    *,
    episode_seeds: np.ndarray,
    final_money: np.ndarray,
    opponent_money: np.ndarray,
    seats: np.ndarray,
    entropy_sums: np.ndarray,
    started: float,
) -> RolloutBatch:
    state_names = set(_state_field_specs(architecture))
    return RolloutBatch(
        architecture=architecture,
        states={name: array for name, array in fields.items() if name in state_names},
        **{name: array for name, array in fields.items() if name not in state_names},
        episode_seeds=episode_seeds,
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        entropy_sums=entropy_sums,
        elapsed_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def collect_mixed_play_rust(
    actor: FarmActor | StructuredActor,
    opponents: Sequence[FarmActor | StructuredActor] = (),
    *,
    self_play_games: int = 0,
    league_games: int = 0,
    opponent_indices: Sequence[int] | np.ndarray | None = None,
    builtin_lanes: Sequence[str] = (),
    seed_start: int,
    episode_steps: int = 720,
    deterministic: bool = False,
    temperature: float = 1.0,
    # Matches `temperature` above, so a caller that omits it gets the symmetric
    # wave production runs. It defaulted to 0.8 while training sharpened its
    # league seats, and that default silently reached instruments which never
    # passed it -- including the schedule sweep that chose the production
    # learning rate, whose conclusions only transfer if its wave is production's.
    opponent_temperature: float = 1.0,
    opponent_temperatures: Sequence[float] | np.ndarray | None = None,
    deterministic_opponent: bool = False,
    deterministic_opponents: Sequence[bool] | np.ndarray | None = None,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect self-play and frozen-league games in one native wave.

    Every game advances inside the same BatchEnv step, so the learner runs a
    single large forward per step covering both seats of every self-play game
    plus the current seat of every league game, instead of separate smaller
    self-play and league waves. All frozen league seats run as one additional
    stacked-weight forward regardless of how many distinct opponents are
    assigned. Stored trajectories are ordered self-play first, then league,
    matching a caller-provided storage arena.

    ``builtin_lanes`` names engine reference agents that play league lanes
    natively inside the wave, with no network behind them. They extend the
    lane index space that ``opponent_indices`` addresses: lanes below
    ``len(opponents)`` are frozen networks, the rest are these built-ins in
    order. A built-in lane still occupies a slot in the stacked frozen
    forward, borrowing the last frozen opponent's weights and discarding the
    result, so the ensemble's batch shape depends on the total lane count and
    not on how many lanes happen to be built-in this iteration — a CUDA graph
    captured for one mix stays valid for every other.

    Rollouts capture only behavior policy state. Value predictions for GAE
    are replayed from the stored features in one large batched critic pass
    at update time, where the critic weights are still exactly the behavior
    weights, instead of paying a small synchronous forward every step.
    """
    if self_play_games < 0 or league_games < 0:
        raise ValueError("game counts cannot be negative")
    if self_play_games + league_games < 1:
        raise ValueError("at least one game is required")
    if episode_steps != 720:
        raise ValueError("the native simulator currently supports the competition horizon 720")
    opponents = tuple(opponents)
    builtin_lanes = tuple(builtin_lanes)
    unknown = sorted(set(builtin_lanes) - BUILTIN_AGENT_CODES.keys())
    if unknown:
        raise ValueError(f"unknown built-in league agents: {', '.join(unknown)}")
    if len(set(builtin_lanes)) != len(builtin_lanes):
        raise ValueError("built-in league agents must be distinct")
    lane_count = len(opponents) + len(builtin_lanes)
    if league_games and not lane_count:
        raise ValueError("league games require at least one frozen or built-in opponent")
    if lane_count and not league_games:
        raise ValueError("league opponents require league games")
    if len(opponents) > np.iinfo(np.uint16).max:
        raise ValueError("too many frozen opponents for native head identifiers")
    _validate_learner_temperature(temperature)
    if opponent_temperatures is None:
        frozen_temperatures = np.full(len(opponents), opponent_temperature, dtype=np.float32)
    else:
        frozen_temperatures = np.asarray(opponent_temperatures, dtype=np.float32)
        if frozen_temperatures.shape != (len(opponents),):
            raise ValueError(f"opponent temperatures must have shape {(len(opponents),)}")
    if opponents and (
        not np.isfinite(frozen_temperatures).all() or (frozen_temperatures <= 0.0).any()
    ):
        raise ValueError("opponent temperatures must be finite and positive")
    if deterministic_opponents is None:
        frozen_deterministic = np.full(len(opponents), deterministic_opponent, dtype=np.bool_)
    else:
        raw_deterministic = np.asarray(deterministic_opponents)
        if raw_deterministic.shape != (len(opponents),):
            raise ValueError(f"deterministic opponent flags must have shape {(len(opponents),)}")
        if not np.issubdtype(raw_deterministic.dtype, np.bool_):
            raise ValueError("deterministic opponent flags must be booleans")
        frozen_deterministic = raw_deterministic.astype(np.bool_, copy=False)
    if opponent_indices is None:
        assignments = np.zeros(league_games, dtype=np.int64)
    else:
        raw_assignments = np.asarray(opponent_indices)
        if raw_assignments.shape != (league_games,):
            raise ValueError(f"opponent indices must have shape {(league_games,)}")
        if not np.issubdtype(raw_assignments.dtype, np.integer):
            raise ValueError("opponent indices must be integers")
        assignments = raw_assignments.astype(np.int64, copy=False)
    if (assignments < 0).any() or (assignments >= max(1, lane_count)).any():
        raise ValueError("opponent index is outside the league lane list")
    started = time.perf_counter()
    actor.eval()
    for opponent in opponents:
        opponent.eval()
    device = next(actor.parameters()).device
    if any(next(opponent.parameters()).device != device for opponent in opponents):
        raise ValueError("current and all frozen models must use the same device")
    if any(
        type(opponent) is not type(actor) or opponent.config != actor.config
        for opponent in opponents
    ):
        raise ValueError("current and frozen actors must use the same model configuration")
    architecture = architecture_of(actor).name

    games = self_play_games + league_games
    seeds = np.arange(seed_start, seed_start + games, dtype=np.uint64)
    environment = load_native().BatchEnv(seeds)
    rows = games * 2
    horizon = episode_steps - 1
    self_play_rows = self_play_games * 2
    trajectories = self_play_rows + league_games
    fields = _native_rollout_storage(storage, architecture, trajectories, horizon)
    encoded_wave = _native_wave(architecture, environment, device)
    encoded = encoded_wave.arrays
    sampled = environment.sample_buffers()

    league_seats = (seeds[self_play_games:] % 2).astype(np.int64)
    league_game_rows = self_play_rows + 2 * np.arange(league_games, dtype=np.int64)
    league_current_rows = league_game_rows + league_seats
    frozen_rows = league_game_rows + (1 - league_seats)
    stored_rows = np.concatenate([np.arange(self_play_rows, dtype=np.int64), league_current_rows])
    generator = np.random.default_rng(sampling_seed)
    frozen_generator = np.random.default_rng(sampling_seed ^ 0x5EED_1EAF)
    kind_gate, quantity_values, quantity_bias = _quantity_heads((actor, *opponents))
    # Per-lane decode of everything a frozen row needs. A built-in lane has no
    # network, so it borrows the learner's quantity head and neutral sampling
    # settings; the native agent replaces that row's whole action regardless.
    lane_names = (*(f"frozen-{index}" for index in range(len(opponents))), *builtin_lanes)
    lane_heads = np.zeros(lane_count, dtype=np.uint16)
    lane_heads[: len(opponents)] = np.arange(1, len(opponents) + 1, dtype=np.uint16)
    lane_temperatures = np.ones(lane_count, dtype=np.float32)
    lane_temperatures[: len(opponents)] = frozen_temperatures
    lane_deterministic = np.zeros(lane_count, dtype=np.bool_)
    lane_deterministic[: len(opponents)] = frozen_deterministic
    lane_codes = np.zeros(lane_count, dtype=np.uint8)
    lane_codes[len(opponents) :] = [BUILTIN_AGENT_CODES[name] for name in builtin_lanes]
    head_ids = np.zeros(rows, dtype=np.uint16)
    head_ids[frozen_rows] = lane_heads[assignments]
    deterministic_rows = np.full(rows, deterministic, dtype=np.bool_)
    deterministic_rows[frozen_rows] = lane_deterministic[assignments]
    temperatures = np.full(rows, temperature, dtype=np.float32)
    temperatures[frozen_rows] = lane_temperatures[assignments]
    builtin_agents = _builtin_agent_rows(
        rows, frozen_rows, stored_rows, assignments, lane_codes, lane_names
    )
    entropy_sums = np.zeros(trajectories, dtype=np.float64)
    final = None
    packed_transfer: _PackedTransfer | None = None

    # A pure self-play wave keeps the learner forward over the contiguous full
    # batch and stores every row, avoiding gather/scatter work entirely.
    store_rows: np.ndarray | slice = slice(None) if not league_games else stored_rows
    stored_pair_rows = stored_rows ^ 1
    current_tensor = None if not league_games else torch.as_tensor(stored_rows, device=device)
    if league_games:
        frozen_groups = tuple(
            frozen_rows[np.flatnonzero(assignments == lane)] for lane in range(lane_count)
        )
        active_indices = [index for index, group in enumerate(frozen_groups) if group.size]
        active_groups = [frozen_groups[index] for index in active_indices]
        # Pad every lane to the widest group so the stacked forward keeps one
        # static shape. Padding replicates a real row of the same lane; those
        # outputs are exact duplicates and are discarded on scatter.
        lanes = len(active_indices)
        lane_width = max(group.size for group in active_groups)
        lane_rows = np.empty((lanes, lane_width), dtype=np.int64)
        lane_valid = np.zeros((lanes, lane_width), dtype=np.bool_)
        for lane, group in enumerate(active_groups):
            lane_rows[lane, : group.size] = group
            lane_rows[lane, group.size :] = group[0]
            lane_valid[lane, : group.size] = True
        lane_valid_flat = lane_valid.reshape(-1)
        frozen_store_rows = np.concatenate(active_groups)
        frozen_tensor = torch.as_tensor(lane_rows.reshape(-1), device=device)
        # Built-in lanes keep their slot in the stack, borrowing the last
        # frozen opponent's weights, so the ensemble's batch shape follows the
        # lane count instead of this iteration's snapshot/built-in mix and a
        # captured CUDA graph survives the next draw. Their logits are never
        # read. A wave with no frozen network at all runs no ensemble.
        ensemble = (
            _stacked_frozen_ensemble(
                [opponents[min(index, len(opponents) - 1)] for index in active_indices]
            )
            if opponents
            else None
        )
        # Zeroed rather than uninitialized: with no ensemble nothing scatters
        # into the frozen rows, and handing the sampler uninitialized memory --
        # even in rows it is contracted to ignore -- is not worth the page.
        unit_logits = np.zeros((rows, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.float32)
        kind_logits = np.zeros((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.float32)
        quantity_context = np.zeros(
            (rows, MAX_MARKET_ORDERS, actor.config.quantity_rank), dtype=np.float32
        )
    # Collection runs the actor under whatever precision the audited decision
    # chose. It is bf16 in the update path regardless, so an fp32 collection
    # forward is not the conservative option: it is a second precision, and the
    # gap between the two is what `update_replay_parity` measures.
    autocast_forward = forward_autocast and device.type == "cuda"
    compiled_forward = forward_mode != "eager"

    def run_actor(*inputs: Any) -> ActorOutput:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_forward):
            output = _rollout_model_forward(actor, *inputs, mode=forward_mode)
        assert isinstance(output, ActorOutput)
        return output

    for step in range(horizon):
        encoded_wave.refresh(environment)
        encoded_wave.copy_to_device()
        _mark_cuda_graph_step(device, compiled_forward)
        if not league_games:
            output = run_actor(*encoded_wave.inputs())
            host_outputs, packed_transfer = _packed_outputs_to_host((output,), packed_transfer)
            host = host_outputs[0]
            step_unit_logits = host.unit_logits
            step_kind_logits = host.market_kind_logits
            step_quantity_context = host.market_quantity_context
            unit_draws, kind_draws, quantity_draws = _categorical_draws(generator, rows)
        else:
            current_output = run_actor(*_select_inputs(encoded_wave.inputs(), current_tensor))
            if ensemble is None:
                host_outputs, packed_transfer = _packed_outputs_to_host(
                    (current_output,), packed_transfer
                )
                (current_host,) = host_outputs
                frozen_host = None
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_forward):
                    lane_output = ensemble(
                        *_lane_view_inputs(encoded_wave.inputs(), frozen_tensor, lanes, lane_width),
                        mode=forward_mode,
                    )
                frozen_output = ActorOutput(*(tensor.flatten(0, 1) for tensor in lane_output))
                host_outputs, packed_transfer = _packed_outputs_to_host(
                    (current_output, frozen_output), packed_transfer
                )
                current_host, frozen_host = host_outputs
            for destination, current_values, frozen_values in zip(
                (unit_logits, kind_logits, quantity_context),
                (
                    current_host.unit_logits,
                    current_host.market_kind_logits,
                    current_host.market_quantity_context,
                ),
                (None, None, None)
                if frozen_host is None
                else (
                    frozen_host.unit_logits,
                    frozen_host.market_kind_logits,
                    frozen_host.market_quantity_context,
                ),
                strict=True,
            ):
                destination[stored_rows] = current_values
                if frozen_values is not None:
                    destination[frozen_store_rows] = frozen_values[lane_valid_flat]
            step_unit_logits = unit_logits
            step_kind_logits = kind_logits
            step_quantity_context = quantity_context
            unit_draws = np.empty((rows, MAX_UNITS), dtype=np.float32)
            kind_draws = np.empty((rows, MAX_MARKET_ORDERS), dtype=np.float32)
            quantity_draws = np.empty((rows, MAX_MARKET_ORDERS), dtype=np.float32)
            current_draws = _categorical_draws(generator, stored_rows.size)
            frozen_draws = _categorical_draws(frozen_generator, league_games)
            for destination, current_values, frozen_values in zip(
                (unit_draws, kind_draws, quantity_draws),
                current_draws,
                frozen_draws,
                strict=True,
            ):
                destination[stored_rows] = current_values
                destination[frozen_rows] = frozen_values
        environment.sample_and_step_into(
            step_unit_logits,
            step_kind_logits,
            step_quantity_context,
            kind_gate,
            quantity_values,
            quantity_bias,
            head_ids,
            unit_draws,
            kind_draws,
            quantity_draws,
            deterministic_rows,
            temperatures,
            builtin_agents,
            sampled,
        )
        rewards = np.asarray(sampled["shaped_rewards"], dtype=np.float32).reshape(-1)
        _store_native_wave(
            architecture,
            fields,
            step,
            encoded,
            sampled,
            rewards[store_rows],
            store_rows,
            stored_pair_rows,
        )
        counts = (
            np.asarray(sampled["unit_active"])[store_rows].sum(axis=1)
            + np.asarray(sampled["market_active"])[store_rows].sum(axis=1)
            + np.asarray(sampled["market_quantity_active"])[store_rows].sum(axis=1)
        )
        entropy_sums += np.asarray(sampled["entropy"])[store_rows] * counts
        dones = np.asarray(sampled["dones"], dtype=np.bool_)
        if step + 1 < horizon and dones.any():
            raise RuntimeError("native rollout terminated before the competition horizon")
        if step + 1 == horizon and not dones.all():
            raise RuntimeError("native rollout did not terminate at the competition horizon")
        final = np.asarray(sampled["final_money"], dtype=np.float32)

    assert final is not None
    self_final = final[:self_play_games]
    league_final = final[self_play_games:]
    league_index = np.arange(league_games)
    final_money = np.concatenate([self_final.reshape(-1), league_final[league_index, league_seats]])
    opponent_money = np.concatenate(
        [self_final[:, ::-1].reshape(-1), league_final[league_index, 1 - league_seats]]
    )
    return _native_batch(
        architecture,
        fields,
        episode_seeds=np.concatenate(
            [
                np.repeat(seeds[:self_play_games].astype(np.int64), 2),
                seeds[self_play_games:].astype(np.int64),
            ]
        ),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=np.concatenate(
            [
                np.tile(np.asarray([0, 1], dtype=np.int8), self_play_games),
                league_seats.astype(np.int8),
            ]
        ),
        entropy_sums=entropy_sums,
        started=started,
    )


def collect_self_play_rust(
    actor: FarmActor | StructuredActor,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    deterministic: bool = False,
    temperature: float = 1.0,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect both on-policy seats through the exact batched Rust simulator."""
    if games < 1:
        raise ValueError("games must be positive")
    return collect_mixed_play_rust(
        actor,
        self_play_games=games,
        seed_start=seed_start,
        episode_steps=episode_steps,
        deterministic=deterministic,
        temperature=temperature,
        sampling_seed=sampling_seed,
        forward_mode=forward_mode,
        forward_autocast=forward_autocast,
        storage=storage,
    )


def collect_frozen_opponents_play_rust(
    actor: FarmActor | StructuredActor,
    opponents: Sequence[FarmActor | StructuredActor],
    *,
    games: int,
    opponent_indices: Sequence[int] | np.ndarray | None = None,
    seed_start: int,
    episode_steps: int = 720,
    temperature: float = 1.0,
    opponent_temperature: float = 0.8,
    opponent_temperatures: Sequence[float] | np.ndarray | None = None,
    deterministic_opponent: bool = False,
    deterministic_opponents: Sequence[bool] | np.ndarray | None = None,
    deterministic: bool = False,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect one current-policy seat per native game against assigned frozen actors."""
    if games < 1:
        raise ValueError("games must be positive")
    return collect_mixed_play_rust(
        actor,
        opponents,
        league_games=games,
        opponent_indices=opponent_indices,
        seed_start=seed_start,
        episode_steps=episode_steps,
        deterministic=deterministic,
        temperature=temperature,
        opponent_temperature=opponent_temperature,
        opponent_temperatures=opponent_temperatures,
        deterministic_opponent=deterministic_opponent,
        deterministic_opponents=deterministic_opponents,
        sampling_seed=sampling_seed,
        forward_mode=forward_mode,
        forward_autocast=forward_autocast,
        storage=storage,
    )


def collect_frozen_opponent_play_rust(
    actor: FarmActor | StructuredActor,
    opponent: FarmActor | StructuredActor,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    temperature: float = 1.0,
    opponent_temperature: float = 0.8,
    deterministic_opponent: bool = False,
    deterministic: bool = False,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
) -> RolloutBatch:
    """Collect one current-policy seat per native game against one frozen actor."""
    return collect_frozen_opponents_play_rust(
        actor,
        (opponent,),
        games=games,
        seed_start=seed_start,
        episode_steps=episode_steps,
        temperature=temperature,
        opponent_temperature=opponent_temperature,
        deterministic_opponent=deterministic_opponent,
        deterministic=deterministic,
        sampling_seed=sampling_seed,
        forward_mode=forward_mode,
        forward_autocast=forward_autocast,
    )


def collect_self_play(
    actor: FarmActor | StructuredActor,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    deterministic: bool = False,
    temperature: float = 1.0,
    sampling_seed: int = 0,
) -> RolloutBatch:
    """Collect both valid on-policy trajectories from every self-play game."""
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps < 2:
        raise ValueError("episode_steps must be at least two")
    _validate_learner_temperature(temperature)
    started = time.perf_counter()
    actor.eval()
    architecture = architecture_of(actor).name
    generator = np.random.default_rng(sampling_seed)
    environments = [
        make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed_start + index},
            debug=False,
        )
        for index in range(games)
    ]
    states = [environment.reset(2) for environment in environments]
    potentials = [pair_potential(state[0].observation, state[1].observation) for state in states]
    fields = _new_fields(architecture)
    trajectories = games * 2
    final_money = np.zeros(trajectories, dtype=np.float32)
    opponent_money = np.zeros(trajectories, dtype=np.float32)
    entropy_sums = np.zeros(trajectories, dtype=np.float64)
    for _ in range(episode_steps):
        if all(environment.done for environment in environments):
            break
        if any(environment.done for environment in environments):
            raise RuntimeError("synchronous environments terminated at different horizons")
        policy_step = act_batch(
            actor,
            _observations(states),
            _opponent_privates(states),
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
        )
        _record_policy_step(architecture, fields, policy_step)
        entropy_sums += policy_step.factors.entropy_sums

        next_states = []
        step_rewards = np.zeros(trajectories, dtype=np.float32)
        for game, environment in enumerate(environments):
            offset = game * 2
            next_state = environment.step(policy_step.actions[offset : offset + 2])
            next_states.append(next_state)
            if any(agent.status == "ERROR" for agent in next_state):
                raise RuntimeError(f"agent error in self-play seed {seed_start + game}")
            if environment.done:
                farms = next_state[0].observation["farms"]
                money = np.asarray(
                    [float(farms[0]["money"]), float(farms[1]["money"])], dtype=np.float32
                )
                final_money[offset : offset + 2] = money
                opponent_money[offset : offset + 2] = money[::-1]
                next_potential = terminal_pair_potential(
                    next_state[0].observation, next_state[1].observation
                )
            else:
                next_potential = pair_potential(
                    next_state[0].observation, next_state[1].observation
                )
            pair_rewards = shaped_pair_reward(potentials[game], next_potential)
            potentials[game] = next_potential
            step_rewards[offset : offset + 2] = pair_rewards
        fields["rewards"].append(step_rewards)
        fields["valid"].append(np.ones(trajectories, dtype=np.bool_))
        states = next_states
    else:
        raise RuntimeError("self-play rollout exceeded the configured episode horizon")

    if not all(environment.done for environment in environments):
        raise RuntimeError("self-play rollout ended before all environments reached DONE")
    return _finish_rollout(
        architecture,
        fields,
        episode_seeds=np.repeat(
            np.arange(seed_start, seed_start + games, dtype=np.int64), repeats=2
        ),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=np.tile(np.asarray([0, 1], dtype=np.int8), games),
        entropy_sums=entropy_sums,
        started=started,
    )


def collect_frozen_opponent_play(
    actor: FarmActor | StructuredActor,
    opponent: FarmActor | StructuredActor,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    temperature: float = 1.0,
    opponent_temperature: float = 0.8,
    deterministic_opponent: bool = False,
    deterministic: bool = False,
    sampling_seed: int = 0,
) -> RolloutBatch:
    """Collect one current-policy trajectory per game against a frozen snapshot."""
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps < 2:
        raise ValueError("episode_steps must be at least two")
    _validate_learner_temperature(temperature)
    started = time.perf_counter()
    actor.eval()
    opponent.eval()
    architecture = architecture_of(actor).name
    device = next(actor.parameters()).device
    if next(opponent.parameters()).device != device:
        raise ValueError("current and frozen policies must use the same device")
    current_generator = np.random.default_rng(sampling_seed)
    opponent_generator = np.random.default_rng(sampling_seed ^ 0x5EED_1EAF)
    environments = [
        make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed_start + index},
            debug=False,
        )
        for index in range(games)
    ]
    states = [environment.reset(2) for environment in environments]
    seats = np.asarray([(seed_start + index) % 2 for index in range(games)], dtype=np.int8)
    potentials = [pair_potential(state[0].observation, state[1].observation) for state in states]
    fields = _new_fields(architecture)
    final_money = np.zeros(games, dtype=np.float32)
    opponent_money = np.zeros(games, dtype=np.float32)
    entropy_sums = np.zeros(games, dtype=np.float64)
    for _ in range(episode_steps):
        if all(environment.done for environment in environments):
            break
        if any(environment.done for environment in environments):
            raise RuntimeError("synchronous environments terminated at different horizons")
        current_observations = [
            state[int(seat)].observation for state, seat in zip(states, seats, strict=True)
        ]
        frozen_observations = [
            state[1 - int(seat)].observation for state, seat in zip(states, seats, strict=True)
        ]
        current_step = act_batch(
            actor,
            current_observations,
            [observation["private"] for observation in frozen_observations],
            deterministic=deterministic,
            temperature=temperature,
            generator=current_generator,
        )
        frozen_step = act_batch(
            opponent,
            frozen_observations,
            deterministic=deterministic_opponent,
            temperature=opponent_temperature,
            generator=opponent_generator,
        )
        _record_policy_step(architecture, fields, current_step)
        entropy_sums += current_step.factors.entropy_sums

        next_states = []
        step_rewards = np.zeros(games, dtype=np.float32)
        for game, (environment, seat) in enumerate(zip(environments, seats, strict=True)):
            actions = [None, None]
            actions[int(seat)] = current_step.actions[game]
            actions[1 - int(seat)] = frozen_step.actions[game]
            next_state = environment.step(actions)
            next_states.append(next_state)
            if any(agent.status == "ERROR" for agent in next_state):
                raise RuntimeError(f"agent error in league seed {seed_start + game}")
            if environment.done:
                farms = next_state[0].observation["farms"]
                player_money = (float(farms[0]["money"]), float(farms[1]["money"]))
                final_money[game] = player_money[int(seat)]
                opponent_money[game] = player_money[1 - int(seat)]
                next_potential = terminal_pair_potential(
                    next_state[0].observation, next_state[1].observation
                )
            else:
                next_potential = pair_potential(
                    next_state[0].observation, next_state[1].observation
                )
            pair_rewards = shaped_pair_reward(potentials[game], next_potential)
            potentials[game] = next_potential
            step_rewards[game] = pair_rewards[int(seat)]
        fields["rewards"].append(step_rewards)
        fields["valid"].append(np.ones(games, dtype=np.bool_))
        states = next_states
    else:
        raise RuntimeError("league rollout exceeded the configured episode horizon")

    if not all(environment.done for environment in environments):
        raise RuntimeError("league rollout ended before all environments reached DONE")
    return _finish_rollout(
        architecture,
        fields,
        episode_seeds=np.arange(seed_start, seed_start + games, dtype=np.int64),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        entropy_sums=entropy_sums,
        started=started,
    )


_TRAJECTORY_METADATA_FIELDS = (
    "episode_seeds",
    "final_money",
    "opponent_money",
    "seats",
    "entropy_sums",
)


def _combined_rollout_metadata(batches: list[RolloutBatch]) -> dict[str, Any]:
    """Concatenate per-trajectory metadata across compatible batches."""
    combined: dict[str, Any] = {
        field: np.concatenate([getattr(batch, field) for batch in batches], axis=0)
        for field in _TRAJECTORY_METADATA_FIELDS
    }
    combined["elapsed_seconds"] = sum(batch.elapsed_seconds for batch in batches)
    return combined


def slice_trajectories(batch: RolloutBatch, start: int, stop: int) -> RolloutBatch:
    """View a contiguous trajectory range of a batch without copying states.

    The slice shares the underlying arrays, so per-part diagnostics of a
    merged wave cost no memory. The wave's elapsed time is indivisible and
    carried over unchanged.
    """
    if not 0 <= start < stop <= batch.trajectories:
        raise ValueError("trajectory slice is out of range")
    return RolloutBatch(
        architecture=batch.architecture,
        states={name: array[start:stop] for name, array in batch.states.items()},
        **{field: getattr(batch, field)[start:stop] for field in _SHARED_ROLLOUT_FIELDS},
        **{field: getattr(batch, field)[start:stop] for field in _TRAJECTORY_METADATA_FIELDS},
        elapsed_seconds=batch.elapsed_seconds,
    )


def concatenate_rollouts(batches: list[RolloutBatch]) -> RolloutBatch:
    """Concatenate compatible current-policy trajectories from several match sources."""
    if not batches:
        raise ValueError("at least one rollout batch is required")
    if len(batches) == 1:
        return batches[0]
    if len({batch.horizon for batch in batches}) != 1:
        raise ValueError("rollout horizons must match")
    if len({batch.architecture for batch in batches}) != 1:
        raise ValueError("rollout architectures must match")
    return RolloutBatch(
        architecture=batches[0].architecture,
        states={
            name: np.concatenate([batch.states[name] for batch in batches], axis=0)
            for name in batches[0].states
        },
        **{
            field: np.concatenate([getattr(batch, field) for batch in batches], axis=0)
            for field in _SHARED_ROLLOUT_FIELDS
        },
        **_combined_rollout_metadata(batches),
    )


def merge_contiguous_rollouts(
    storage: dict[str, np.ndarray], batches: list[RolloutBatch]
) -> RolloutBatch:
    """Combine batches collected into adjacent views of one storage arena.

    The per-state arrays are taken from the arena without copying; only the
    small per-trajectory metadata arrays are concatenated. Every batch must
    occupy exactly its expected row range of the arena, in order.
    """
    if not batches:
        raise ValueError("at least one rollout batch is required")
    if len({batch.horizon for batch in batches}) != 1:
        raise ValueError("rollout horizons must match")
    if len({batch.architecture for batch in batches}) != 1:
        raise ValueError("rollout architectures must match")
    architecture = batches[0].architecture
    state_names = tuple(_state_field_specs(architecture))
    field_names = (*state_names, *_SHARED_ROLLOUT_FIELDS)
    rows = sum(batch.trajectories for batch in batches)
    offset = 0
    for batch in batches:
        for field in field_names:
            expected = storage[field][offset : offset + batch.trajectories]
            actual = _rollout_array(batch, field)
            if (
                actual.shape != expected.shape
                or actual.__array_interface__["data"][0] != expected.__array_interface__["data"][0]
            ):
                raise ValueError("rollout batches are not adjacent views of the storage arena")
        offset += batch.trajectories
    if any(storage[field].shape[0] != rows for field in field_names):
        raise ValueError("storage arena rows do not match the combined batches")
    return RolloutBatch(
        architecture=architecture,
        states={name: storage[name] for name in state_names},
        **{field: storage[field] for field in _SHARED_ROLLOUT_FIELDS},
        **_combined_rollout_metadata(batches),
    )
