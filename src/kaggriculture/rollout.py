"""Pipelined self-play rollout collection against the official simulator."""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from kaggle_environments import make

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    DEFAULT_REWARD_GAMMA,
    EPISODE_STEPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
)
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
    pair_potential,
    shaped_pair_reward,
    terminal_pair_utility,
)
from kaggriculture.model import ActorOutput, FarmActor
from kaggriculture.opponents import BUILTIN_AGENT_ORDER
from kaggriculture.orientation import (
    orient_boards,
    orient_unit_actions,
    orient_unit_features,
    orient_unit_logits,
    orient_unit_masks,
    seat_orientations,
)
from kaggriculture.policy import PolicyStep, act_batch, categorical_statistics
from kaggriculture.registry import CONV_ENTITY, STRUCTURED, architecture_of
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredActor, StructuredInputs
from kaggriculture.tokens import (
    ANIMAL_PRIVATE_FIELDS,
    ANIMAL_TOKEN_FIELDS,
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
_NATIVE_TEMPERATURE_FLOOR = 1e-4
_POPULATION_PAIRING_SEED_SALT = 0x5041_4952

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
    # Which population member sampled the row. A single-learner wave stores
    # zeros; a population wave stores the pairing schedule's agent index, and
    # the update partitions on it so one member's advantage scale never
    # normalizes another's.
    agents: np.ndarray
    # Per-trajectory board symmetry. Both seats of a game share one code;
    # games cycle identity / mirror-x / mirror-y / rotate-180. Mixed and
    # Python collectors store zeros (identity).
    orientations: np.ndarray
    # True only when every stored learner action was sampled from the behavior
    # distribution rather than selected by argmax. PPO rejects false batches.
    learner_stochastic: bool
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
            "animals": ((len(ANIMALS), len(ANIMAL_TOKEN_FIELDS)), np.float16),
            "crops": ((len(CROPS), len(CROP_TOKEN_FIELDS)), np.float16),
            "farms": ((2, len(FARM_TOKEN_FIELDS)), np.float16),
            "town": ((len(TOWN_TOKEN_FIELDS),), np.float16),
            "opponent_unit_categorical": ((MAX_UNITS, N_UNIT_CATEGORICAL), np.int8),
            "opponent_unit_continuous": ((MAX_UNITS, N_UNIT_CONTINUOUS), np.float16),
            "opponent_unit_active": ((MAX_UNITS,), np.bool_),
            "critic_products": ((len(PRODUCTS), len(PRODUCT_PRIVATE_FIELDS)), np.float16),
            "critic_animals": ((len(ANIMALS), len(ANIMAL_PRIVATE_FIELDS)), np.float16),
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
_ANIMAL_STOCK_COLUMNS = slice(
    ANIMAL_TOKEN_FIELDS.index("shed_stock"), ANIMAL_TOKEN_FIELDS.index("carried_stock") + 1
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
    agents: np.ndarray,
    entropy_sums: np.ndarray,
    learner_stochastic: bool,
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
        agents=agents,
        orientations=np.zeros(agents.shape[0], dtype=np.int8),
        learner_stochastic=learner_stochastic,
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


def _empty_selected_inputs(
    inputs: tuple[Any, ...], leading_shape: tuple[int, ...]
) -> tuple[Any, ...]:
    """Allocate a reusable row-gather destination matching model inputs."""

    def empty(tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (*leading_shape, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    return tuple(
        type(entry)(*(empty(tensor) for tensor in entry))
        if isinstance(entry, tuple)
        else empty(entry)
        for entry in inputs
    )


def _select_inputs(
    inputs: tuple[Any, ...],
    rows: torch.Tensor,
    out: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Gather rows of a model-argument tuple, optionally into persistent storage."""
    if out is None:
        return tuple(
            type(entry)(*(tensor.index_select(0, rows) for tensor in entry))
            if isinstance(entry, tuple)
            else entry.index_select(0, rows)
            for entry in inputs
        )
    for entry, destination in zip(inputs, out, strict=True):
        if isinstance(entry, tuple):
            for tensor, target in zip(entry, destination, strict=True):
                torch.index_select(tensor, 0, rows, out=target)
        else:
            torch.index_select(entry, 0, rows, out=destination)
    return out


def _lane_view_inputs(
    inputs: tuple[Any, ...],
    rows: torch.Tensor,
    lanes: int,
    width: int,
    out: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Gather rows and fold them into [lanes, width, ...] shapes."""
    if out is None:

        def folded(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.index_select(0, rows).view(lanes, width, *tensor.shape[1:])

        return tuple(
            type(entry)(*(folded(tensor) for tensor in entry))
            if isinstance(entry, tuple)
            else folded(entry)
            for entry in inputs
        )
    for entry, destination in zip(inputs, out, strict=True):
        if isinstance(entry, tuple):
            for tensor, target in zip(entry, destination, strict=True):
                torch.index_select(
                    tensor, 0, rows, out=target.view(rows.numel(), *tensor.shape[1:])
                )
        else:
            torch.index_select(entry, 0, rows, out=destination.view(rows.numel(), *entry.shape[1:]))
    return out


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
        """Upload the encoded block, then convert it in a separate device kernel.

        The Rust encoder emits fp16 and the model consumes its inference dtype. A
        single cross-dtype `copy_` uses a same-dtype device temporary before the
        conversion. This spells out both buffers so they are allocated once and
        their addresses remain static for CUDA graph capture.
        """
        non_blocking = self.device_features.device.type == "cuda"
        self.staged_features.copy_(self.host_features, non_blocking=non_blocking)
        self.device_features.copy_(self.staged_features, non_blocking=non_blocking)
        self.unit_positions.copy_(self.host_positions, non_blocking=non_blocking)

    def refresh(self, environment: Any) -> None:
        environment.encoded_into(self.arrays)

    def inputs(self) -> tuple[torch.Tensor, ...]:
        return (self.board, self.global_features, self.units, self.unit_positions)


def _native_encoded_wave(
    environment: Any,
    device: torch.device,
    floating_dtype: torch.dtype = torch.float32,
    device_wave: _NativeEncodedWave | None = None,
) -> _NativeEncodedWave:
    """Build reusable Rust output plus packed, optionally pinned model input buffers."""
    arrays = {name: np.asarray(value) for name, value in environment.encoded_buffers().items()}
    feature_names = ("board", "global_features", "critic_features", "units")
    feature_sizes = [arrays[name].size for name in feature_names]
    pin_memory = device.type == "cuda"
    host_features = torch.empty(sum(feature_sizes), dtype=torch.float16, pin_memory=pin_memory)
    staged_features = (
        torch.empty(sum(feature_sizes), dtype=host_features.dtype, device=device)
        if device_wave is None
        else device_wave.staged_features
    )
    device_features = (
        torch.empty(sum(feature_sizes), dtype=floating_dtype, device=device)
        if device_wave is None
        else device_wave.device_features
    )
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
    unit_positions = (
        torch.empty(host_positions.shape, dtype=torch.long, device=device)
        if device_wave is None
        else device_wave.unit_positions
    )
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


# Structured Rust output uses three storage dtypes, but all of them travel as
# raw bytes in one pinned block. The typed views below retain the encoder and
# model layouts while reducing each CUDA wave to one host-to-device upload.
_STRUCTURED_CONTINUOUS_BUFFERS = (
    "tile_continuous",
    "unit_continuous",
    "products",
    "animals",
    "crops",
    "farms",
    "town",
)
_STRUCTURED_CATEGORICAL_BUFFERS = ("tile_categorical", "unit_categorical", "unit_tile_gather")
_STRUCTURED_FLAG_BUFFERS = ("unit_active", "unit_tile_gather_valid")


@dataclass(frozen=True)
class _NativeStructuredWave:
    arrays: dict[str, np.ndarray]
    host_transport: torch.Tensor
    staged_transport: torch.Tensor
    staged_continuous: torch.Tensor
    staged_categorical: torch.Tensor
    device_continuous: torch.Tensor
    device_categorical: torch.Tensor
    device_inputs: StructuredInputs

    def copy_to_device(self) -> None:
        """Upload once, then convert the two model-input groups in place.

        Keeping the same-dtype transport and final model-input buffers removes
        per-step temporaries. Packing flags into the byte transport also removes
        two H2D launches; the model reads bool views directly from that block.
        """
        non_blocking = self.device_continuous.device.type == "cuda"
        self.staged_transport.copy_(self.host_transport, non_blocking=non_blocking)
        self.device_continuous.copy_(self.staged_continuous)
        self.device_categorical.copy_(self.staged_categorical)

    def refresh(self, environment: Any) -> None:
        environment.structured_into(self.arrays)

    def inputs(self) -> tuple[StructuredInputs]:
        return (self.device_inputs,)


def _native_structured_wave(
    environment: Any,
    device: torch.device,
    floating_dtype: torch.dtype = torch.float32,
    device_wave: _NativeStructuredWave | None = None,
) -> _NativeStructuredWave:
    """Build reusable structured Rust output plus packed device token tensors."""
    arrays = {name: np.asarray(value) for name, value in environment.structured_buffers().items()}
    continuous_elements = sum(arrays[name].size for name in _STRUCTURED_CONTINUOUS_BUFFERS)
    categorical_elements = sum(arrays[name].size for name in _STRUCTURED_CATEGORICAL_BUFFERS)
    flag_elements = sum(arrays[name].size for name in _STRUCTURED_FLAG_BUFFERS)
    continuous_bytes = continuous_elements * 2
    categorical_end = continuous_bytes + categorical_elements
    transport_bytes = categorical_end + flag_elements

    host_transport = torch.empty(
        transport_bytes,
        dtype=torch.uint8,
        pin_memory=device.type == "cuda",
    )
    host_continuous = host_transport[:continuous_bytes].view(torch.float16)
    host_categorical = host_transport[continuous_bytes:categorical_end].view(torch.int8)
    host_flags = host_transport[categorical_end:].view(torch.bool)
    if device_wave is None:
        staged_transport = torch.empty(transport_bytes, dtype=torch.uint8, device=device)
        device_continuous = torch.empty(continuous_elements, dtype=floating_dtype, device=device)
        device_categorical = torch.empty(categorical_elements, dtype=torch.int64, device=device)
    else:
        staged_transport = device_wave.staged_transport
        device_continuous = device_wave.device_continuous
        device_categorical = device_wave.device_categorical
    staged_continuous = staged_transport[:continuous_bytes].view(torch.float16)
    staged_categorical = staged_transport[continuous_bytes:categorical_end].view(torch.int8)
    staged_flags = staged_transport[categorical_end:].view(torch.bool)
    views: dict[str, torch.Tensor] = {}

    def map_group(names: tuple[str, ...], host: torch.Tensor, destination: torch.Tensor) -> None:
        cursor = 0
        for name in names:
            shape = arrays[name].shape
            size = arrays[name].size
            arrays[name] = host[cursor : cursor + size].reshape(shape).numpy()
            views[name] = destination[cursor : cursor + size].reshape(shape)
            cursor += size

    map_group(_STRUCTURED_CONTINUOUS_BUFFERS, host_continuous, device_continuous)
    map_group(_STRUCTURED_CATEGORICAL_BUFFERS, host_categorical, device_categorical)
    map_group(_STRUCTURED_FLAG_BUFFERS, host_flags, staged_flags)
    return _NativeStructuredWave(
        arrays=arrays,
        host_transport=host_transport,
        staged_transport=staged_transport,
        staged_continuous=staged_continuous,
        staged_categorical=staged_categorical,
        device_continuous=device_continuous,
        device_categorical=device_categorical,
        device_inputs=(
            StructuredInputs(**{name: views[name] for name in StructuredInputs._fields})
            if device_wave is None
            else device_wave.device_inputs
        ),
    )


def _native_wave(
    architecture: str,
    environment: Any,
    device: torch.device,
    floating_dtype: torch.dtype = torch.float32,
    device_wave: _NativeEncodedWave | _NativeStructuredWave | None = None,
) -> _NativeEncodedWave | _NativeStructuredWave:
    if architecture == CONV_ENTITY:
        assert device_wave is None or isinstance(device_wave, _NativeEncodedWave)
        return _native_encoded_wave(environment, device, floating_dtype, device_wave)
    assert device_wave is None or isinstance(device_wave, _NativeStructuredWave)
    return _native_structured_wave(environment, device, floating_dtype, device_wave)


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


@dataclass(frozen=True)
class _GpuPreferenceTransfer:
    """Persistent conversion buffer and pinned compact native inputs."""

    device_quantity_context: torch.Tensor | None
    units: torch.Tensor
    kinds: torch.Tensor
    quantity_context: torch.Tensor
    quantity_draws: torch.Tensor


def _quantity_head_tensors(
    actors: Sequence[FarmActor | StructuredActor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([actor.market_quantity_kind_gate.weight for actor in actors]),
        torch.stack([actor.market_quantity_value.weight for actor in actors]),
        torch.stack([actor.market_quantity_bias for actor in actors]),
    )


def _selected_quantity_logits(
    context: torch.Tensor,
    kinds: torch.Tensor,
    head_ids: torch.Tensor,
    heads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    kind_gate, values, bias = heads
    kinds = kinds.long()
    row = torch.arange(context.shape[0], device=context.device)[:, None]
    row_gate = kind_gate[head_ids]
    row_bias = bias[head_ids]
    selected_gate = row_gate[row, kinds]
    selected_bias = row_bias[row, kinds]
    features = context.float() * (1.0 + selected_gate.float())
    return torch.einsum("bsr,bqr->bsq", features, values[head_ids].float()) + selected_bias.float()


def _row_random(
    shape: tuple[int, ...],
    current_rows: torch.Tensor,
    frozen_rows: torch.Tensor,
    current_generator: torch.Generator,
    frozen_generator: torch.Generator,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Draw independent current/frozen streams into their physical row order."""
    values = torch.empty(shape, dtype=torch.float32, device=device)
    for rows, generator in (
        (current_rows, current_generator),
        (frozen_rows, frozen_generator),
    ):
        if rows.numel():
            values[rows] = torch.rand(
                (rows.numel(), *shape[1:]),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
    return values


def _gumbel_utilities(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    deterministic_rows: torch.Tensor,
    current_rows: torch.Tensor,
    frozen_rows: torch.Tensor,
    current_generator: torch.Generator,
    frozen_generator: torch.Generator,
) -> torch.Tensor:
    effective_temperatures = temperatures.clamp_min(_NATIVE_TEMPERATURE_FLOOR)
    scores = logits.float() / effective_temperatures.reshape(
        effective_temperatures.shape[0], *((1,) * (logits.ndim - 1))
    )
    uniforms = _row_random(
        tuple(scores.shape),
        current_rows,
        frozen_rows,
        current_generator,
        frozen_generator,
        device=scores.device,
    )
    uniforms.clamp_(
        min=torch.finfo(torch.float32).tiny,
        max=1.0 - torch.finfo(torch.float32).eps,
    )
    noise = -torch.log(-torch.log(uniforms))
    return torch.where(
        deterministic_rows.reshape(deterministic_rows.shape[0], *((1,) * (logits.ndim - 1))),
        scores,
        scores + noise,
    )


def _gpu_preferences_to_host(
    output: ActorOutput,
    temperatures: torch.Tensor,
    deterministic_rows: torch.Tensor,
    current_rows: torch.Tensor,
    frozen_rows: torch.Tensor,
    current_generator: torch.Generator,
    frozen_generator: torch.Generator,
    transfer: _GpuPreferenceTransfer | None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], _GpuPreferenceTransfer]:
    unit_utilities = _gumbel_utilities(
        output.unit_logits,
        temperatures,
        deterministic_rows,
        current_rows,
        frozen_rows,
        current_generator,
        frozen_generator,
    )
    kind_utilities = _gumbel_utilities(
        output.market_kind_logits,
        temperatures,
        deterministic_rows,
        current_rows,
        frozen_rows,
        current_generator,
        frozen_generator,
    )
    quantity_draws = _row_random(
        (output.market_quantity_context.shape[0], MAX_MARKET_ORDERS),
        current_rows,
        frozen_rows,
        current_generator,
        frozen_generator,
        device=output.market_quantity_context.device,
    )
    device_preferences = (unit_utilities, kind_utilities, quantity_draws)
    if (
        transfer is None
        or transfer.quantity_context.shape != output.market_quantity_context.shape
        or (transfer.device_quantity_context is None)
        != (output.market_quantity_context.dtype == torch.float32)
        or any(
            host.shape != preference.shape or host.dtype != preference.dtype
            for host, preference in zip(
                (transfer.units, transfer.kinds, transfer.quantity_draws),
                device_preferences,
                strict=True,
            )
        )
    ):
        transfer = _GpuPreferenceTransfer(
            device_quantity_context=(
                None
                if output.market_quantity_context.dtype == torch.float32
                else torch.empty(
                    output.market_quantity_context.shape,
                    dtype=torch.float32,
                    device=output.market_quantity_context.device,
                )
            ),
            units=torch.empty(
                unit_utilities.shape,
                dtype=unit_utilities.dtype,
                device="cpu",
                pin_memory=True,
            ),
            kinds=torch.empty(
                kind_utilities.shape,
                dtype=kind_utilities.dtype,
                device="cpu",
                pin_memory=True,
            ),
            quantity_context=torch.empty(
                output.market_quantity_context.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            ),
            quantity_draws=torch.empty(
                quantity_draws.shape,
                dtype=quantity_draws.dtype,
                device="cpu",
                pin_memory=True,
            ),
        )
    if output.market_quantity_context.dtype == torch.float32:
        device_quantity_context = output.market_quantity_context
    else:
        assert transfer.device_quantity_context is not None
        transfer.device_quantity_context.copy_(output.market_quantity_context)
        device_quantity_context = transfer.device_quantity_context
    host_tensors = (
        transfer.units,
        transfer.kinds,
        transfer.quantity_context,
        transfer.quantity_draws,
    )
    preferences = (
        unit_utilities,
        kind_utilities,
        device_quantity_context,
        quantity_draws,
    )
    for host, preference in zip(host_tensors, preferences, strict=True):
        host.copy_(preference, non_blocking=True)
    torch.cuda.current_stream(output.unit_logits.device).synchronize()
    arrays = tuple(host.numpy() for host in host_tensors)
    return (arrays[0], arrays[1], arrays[2], arrays[3]), transfer


@dataclass(frozen=True)
class _GpuStatisticsTransfer:
    """Persistent pinned staging for sampled policy statistics."""

    device: torch.Tensor
    host: torch.Tensor
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ready: torch.cuda.Event
    inputs: tuple[torch.Tensor, ...]


_GPU_STATISTIC_INPUT_NAMES = (
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


def _fill_gpu_policy_statistics(
    sampled: dict[str, np.ndarray],
    output: ActorOutput,
    heads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    head_ids: torch.Tensor,
    builtin_agents: torch.Tensor,
    temperatures: torch.Tensor,
    transfer: _GpuStatisticsTransfer | None = None,
    *,
    synchronize: bool = True,
) -> _GpuStatisticsTransfer | None:
    device = output.unit_logits.device
    if device.type == "cpu":
        input_tensors = tuple(torch.as_tensor(sampled[name]) for name in _GPU_STATISTIC_INPUT_NAMES)
    else:
        host_inputs = tuple(
            torch.from_numpy(np.asarray(sampled[name])) for name in _GPU_STATISTIC_INPUT_NAMES
        )
        statistic_shapes = (
            tuple(host_inputs[0].shape),
            tuple(host_inputs[1].shape),
            tuple(host_inputs[2].shape),
            (host_inputs[0].shape[0],),
        )
        total = sum(int(np.prod(shape)) for shape in statistic_shapes)
        if (
            transfer is None
            or transfer.device.numel() != total
            or tuple(array.shape for array in transfer.arrays) != statistic_shapes
            or tuple(tensor.shape for tensor in transfer.inputs)
            != tuple(tensor.shape for tensor in host_inputs)
        ):
            device_buffer = torch.empty(total, dtype=torch.float32, device=device)
            host_buffer = torch.empty(total, dtype=torch.float32, device="cpu", pin_memory=True)
            flat = host_buffer.numpy()
            cursor = 0
            arrays: list[np.ndarray] = []
            for shape in statistic_shapes:
                size = int(np.prod(shape))
                arrays.append(flat[cursor : cursor + size].reshape(shape))
                cursor += size
            transfer = _GpuStatisticsTransfer(
                device=device_buffer,
                host=host_buffer,
                arrays=tuple(arrays),
                ready=torch.cuda.Event(),
                inputs=tuple(torch.empty_like(tensor, device=device) for tensor in host_inputs),
            )
        for target, source in zip(transfer.inputs, host_inputs, strict=True):
            target.copy_(source, non_blocking=True)
        input_tensors = transfer.inputs
    (
        unit_actions,
        market_kinds,
        market_quantities,
        unit_masks,
        kind_masks,
        quantity_masks,
        unit_active,
        kind_active,
        quantity_active,
    ) = input_tensors
    scale = temperatures.clamp_min(_NATIVE_TEMPERATURE_FLOOR)[:, None, None]
    unit_logits = output.unit_logits / scale
    kind_logits = output.market_kind_logits / scale
    quantity_logits = (
        _selected_quantity_logits(output.market_quantity_context, market_kinds, head_ids, heads)
        / scale
    )
    unit_logprob, unit_entropy = categorical_statistics(unit_logits, unit_masks, unit_actions)
    kind_logprob, kind_entropy = categorical_statistics(kind_logits, kind_masks, market_kinds)
    quantity_logprob, quantity_entropy = categorical_statistics(
        quantity_logits, quantity_masks, market_quantities
    )
    learned = builtin_agents == 0
    for values in (
        unit_logprob,
        kind_logprob,
        quantity_logprob,
        unit_entropy,
        kind_entropy,
        quantity_entropy,
    ):
        values.mul_(learned[:, None])
    counts = unit_active.sum(1) + kind_active.sum(1) + quantity_active.sum(1)
    entropy = (
        (unit_entropy * unit_active).sum(1)
        + (kind_entropy * kind_active).sum(1)
        + (quantity_entropy * quantity_active).sum(1)
    ) / counts.clamp_min(1)
    statistics = (
        ("unit_logprobs", unit_logprob),
        ("market_kind_logprobs", kind_logprob),
        ("market_quantity_logprobs", quantity_logprob),
        ("entropy", entropy),
    )
    if device.type == "cpu":
        for name, values in statistics:
            np.copyto(np.asarray(sampled[name]), values.float().numpy())
        return transfer

    assert transfer is not None
    cursor = 0
    for _, values in statistics:
        size = values.numel()
        transfer.device[cursor : cursor + size].view(values.shape).copy_(values)
        cursor += size
    transfer.host.copy_(transfer.device, non_blocking=True)
    transfer.ready.record()
    if synchronize:
        transfer.ready.synchronize()
        for (name, _), values in zip(statistics, transfer.arrays, strict=True):
            np.copyto(np.asarray(sampled[name]), values)
    return transfer


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
#:
#: `graph` replays the *eager* kernel sequence under one collector-owned CUDA
#: graph: launch overhead goes, but the ~1,900 eager kernels of a structured
#: step remain and inside a graph each still costs its own scheduling slot.
#: `inductor_graph` captures the Inductor-fused forward instead -- the same
#: `mode="default"` compilation as `inductor_default`, so no cudagraph-tree
#: bookkeeping -- and replays the fused kernels through the same explicit
#: `_CapturedStep`. It is fusion and capture together; `_CapturedStep`'s
#: warmup runs compile before the capture begins, which is what the capture
#: rules require.
ROLLOUT_FORWARD_MODES = (
    "eager",
    "graph",
    "cudagraphs",
    "inductor",
    "inductor_default",
    "inductor_graph",
)


def _validate_forward_mode(mode: str) -> None:
    if mode not in ROLLOUT_FORWARD_MODES:
        raise ValueError(f"unknown rollout forward mode {mode!r}")


#: Modes that reach the device through `torch.compile`, and so through
#: `torch._inductor.cudagraph_trees`' generation bookkeeping.
COMPILED_ROLLOUT_FORWARD_MODES = ("cudagraphs", "inductor", "inductor_default", "inductor_graph")

#: Modes the collector captures itself with `_CapturedStep`.
CAPTURED_ROLLOUT_FORWARD_MODES = ("graph", "inductor_graph")

#: The subset of those that lets `cudagraph_trees` capture rather than only
#: fuse. Capture is what makes a forward's outputs live in a reused private
#: pool, so it is what `_cuda_graph_generation` has to serialize.
CAPTURING_ROLLOUT_FORWARD_MODES = ("cudagraphs", "inductor")


def _cached_compiled_forward(model: FarmActor | StructuredActor, mode: str = "cudagraphs") -> Any:
    """Return the cached compiled collection forward for `mode`.

    CUDA convolution and GEMM kernels are not bitwise invariant across eager,
    graph, and fused execution, nor across batch shapes, so any mode here moves
    the behavior policy relative to the update-path replay. That drift is bounded
    semantically rather than bitwise, by `update_replay_parity`.

    `inductor` is `reduce-overhead`, which adds CUDA graphs to the fusion, and
    `inductor_default` is the fusion alone. Compilation is lazy: the first call,
    not this cache lookup, pays the compile cost.

    One compiled callable is cached per mode so a parity audit can measure
    several in one process without recompiling.
    """
    _validate_forward_mode(mode)
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
                model.forward,
                mode="reduce-overhead" if mode == "inductor" else "default",
                fullgraph=True,
                dynamic=False,
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
    absence of a flag. The stacked ensemble takes the same mode, so a wave
    cannot end up compiling one of its two forwards and not the other.
    """
    _validate_forward_mode(mode)
    if mode not in COMPILED_ROLLOUT_FORWARD_MODES or _leading_tensor(inputs).device.type != "cuda":
        return model(*inputs)
    return _cached_compiled_forward(model, mode)(*inputs)


class _StackedActorEnsemble:
    """One batched forward over several same-architecture actors via stacked weights.

    The lanes share an architecture but not weights. Stacking their parameters
    lane-wise and running a single vmapped functional call replaces the
    per-model forward loop, so a whole multi-model side of a wave is one large
    kernel sequence instead of several small ones. Instances persist for the
    process and are refilled in place each collection call: a captured CUDA
    graph keeps reading current weights at stable addresses without any
    per-step parameter copies.

    Nothing here requires the lanes be frozen, which is what lets a population
    wave run every concurrently learning member through it: rollout takes no
    gradient, and the update replays the stored actions through the plain
    module. The in-place refill reads live weights exactly as it reads a
    snapshot's.
    """

    def __init__(self, models: Sequence[FarmActor | StructuredActor]) -> None:
        first = models[0]
        self.template = (
            StructuredActor(first.config)
            if isinstance(first, StructuredActor)
            else FarmActor(first.config)
        ).to("meta")
        self.template.eval()
        import torch._dynamo

        # Dynamo's recompile budget is per Python code object, not per compiled
        # callable. Every lane-count-specific ensemble below shares
        # `_forward.__code__`, so the normal growth from one to N league lanes
        # otherwise trips the default limit of eight even though each shape has
        # its own persistent callable. N is also a strict upper bound on the
        # distinct positive lane counts that can precede this instance.
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, len(models)
        )

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
        """Run every stacked lane under the same mode as the learner forward.

        A league wave runs this forward once per step in addition to the
        learner's, and a population wave runs it as the only forward, so leaving
        it on a backend the learner abandoned would cap the collection speedup
        at whatever fraction of steps are pure self-play -- and in a population
        wave it would set every stored row's behavior policy from a path the
        update does not replay. It takes the same mode for the same measured
        reason: `cudagraphs` is slower here than not compiling, because this
        forward is limited by per-kernel overhead rather than launch cost, and
        only fusion reduces the kernel count.

        `graph` takes the uncompiled path along with `eager`, because there the
        capture is the collector's and this forward is inside it. Entering the
        compiler from within a stream capture is not merely slow, it is
        forbidden: it raises `cudaErrorStreamCaptureUnsupported` and invalidates
        the capture in progress.
        """
        _validate_forward_mode(mode)
        leading = _leading_tensor(inputs)
        if mode not in COMPILED_ROLLOUT_FORWARD_MODES or leading.device.type != "cuda":
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
                    self._forward,
                    mode="reduce-overhead" if mode == "inductor" else "default",
                    fullgraph=True,
                    dynamic=False,
                )
            self._compiled[key] = compiled
        return compiled(*inputs)


_STACKED_ENSEMBLE_CACHE: dict[tuple[Any, ...], _StackedActorEnsemble] = {}


def _stacked_actor_ensemble(
    models: Sequence[FarmActor | StructuredActor],
    namespace: int = 0,
) -> _StackedActorEnsemble:
    """Fetch or build the persistent stacked ensemble for these lanes."""
    model_references = tuple(weakref.ref(model) for model in models)
    cache_key = (
        model_references,
        namespace,
        type(models[0]),
        models[0].config,
        len(models),
        next(models[0].parameters()).device,
        tuple(parameter.dtype for parameter in models[0].parameters()),
        tuple(buffer.dtype for buffer in models[0].buffers()),
    )
    ensemble = _STACKED_ENSEMBLE_CACHE.get(cache_key)
    if ensemble is None:

        def evict(_reference: weakref.ReferenceType[FarmActor | StructuredActor]) -> None:
            _STACKED_ENSEMBLE_CACHE.pop(cache_key, None)

        model_references = tuple(weakref.ref(model, evict) for model in models)
        cache_key = (model_references, *cache_key[1:])
        ensemble = _StackedActorEnsemble(models)
        _STACKED_ENSEMBLE_CACHE[cache_key] = ensemble
    else:
        ensemble.load(models)
    return ensemble


def _mark_cuda_graph_step(device: torch.device, enabled: bool) -> None:
    if not enabled or device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


# `torch._inductor.cudagraph_trees` keeps one tree manager per *thread* but a
# single process-wide generation counter: `MarkStepBox.mark_step_counter`,
# which `cudagraph_mark_step_begin` decrements and which every manager reads
# through `get_curr_generation`. A manager concludes that a new generation has
# begun -- and so that the previous generation's output buffers may be reused --
# by seeing that counter differ from the value it recorded on its own last
# compiled call. Concurrent collectors can otherwise let one thread's mark land
# between another thread's actor and ensemble forwards, retiring an output that
# its `index_copy_` has not consumed yet.
#
# Holding this across a collector's mark and every compiled forward of one step
# restores the intended invariant: the counter changes between steps and never
# inside one. Only cudagraph-tree capture pays for the lock.
_CUDA_GRAPH_GENERATION = threading.Lock()


#: A device-assembled whole-wave output, or the separate current/frozen
#: outputs that the CPU sampling path assembles after host transfer.
_StepOutputs = tuple["ActorOutput | None", "ActorOutput | None", "ActorOutput | None"]

#: Whatever a captured region returns. The mixed-play wave and the population
#: wave capture different shapes and each keeps its own.
_Captured = TypeVar("_Captured")


# Capture switches the caching allocator to a private pool. Serialize this
# once-per-collection operation against any other collector in the process.
_CUDA_GRAPH_CAPTURE = threading.Lock()
_ROLLOUT_STREAMS = threading.local()


def _rollout_cuda_streams(device: torch.device) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """Keep capture/warmup and frozen activation owners stable across waves.

    Like PPO's stream pair, these are thread/device-owned rather than
    shape-owned. Only streams persist: inputs, outputs and graphs remain owned
    by the collecting wave, and every use establishes its own dependencies.
    """
    streams = getattr(_ROLLOUT_STREAMS, "devices", None)
    if streams is None:
        streams = _ROLLOUT_STREAMS.devices = {}
    index = torch.cuda.current_device() if device.index is None else device.index
    if index not in streams:
        streams[index] = (torch.cuda.Stream(device=index), torch.cuda.Stream(device=index))
    return streams[index]


class _CapturedStep(Generic[_Captured]):
    """One CUDA graph over the whole per-step forward region.

    The graph is deliberately bounded before sampling and storage, which
    round-trip through native host code. Capturing only the forward still
    removes its Python launch overhead while preserving that boundary.

    This is deliberately not `torch.compile(mode="reduce-overhead")`, which
    reaches the same idea through `torch._inductor.cudagraph_trees` and its
    process-global generation bookkeeping. Owning the graph directly makes the
    recording lifetime explicit.

    The collector holds every input at a fixed address for the life of the
    wave. `_NativeStructuredWave.copy_to_device` refreshes persistent device
    buffers in place, `_select_inputs` and `_lane_view_inputs` gather into
    persistent `out=` storage, every row-index tensor is built once before the
    step loop, and `_pipeline_replica` refreshes weights through
    `load_state_dict` rather than rebuilding the module. A replay therefore
    reads exactly what the step just uploaded.

    Outputs live in the graph's private pool and are overwritten by the next
    replay, which is the same contract the step already honours: it consumes
    them into persistent storage before it advances.
    """

    def __init__(self, run: Callable[[], _Captured], warmup: int = 3) -> None:
        with _CUDA_GRAPH_CAPTURE:
            # The documented recipe: warm up on a side stream so that allocator
            # growth, cuBLAS handle creation, and any lazy kernel load happen
            # before the capture rather than inside it.
            side, _ = _rollout_cuda_streams(torch.cuda.current_stream().device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            # Thread-local capture remains robust if an unrelated CUDA-using
            # thread exists in the embedding process.
            with torch.cuda.graph(self._graph, stream=side, capture_error_mode="thread_local"):
                self._outputs = run()

    def __call__(self) -> _Captured:
        self._graph.replay()
        return self._outputs

    def close(self) -> None:
        """Release the executable graph and the pool its outputs live in.

        A wave allocates its own device buffers, so a graph cannot outlive the
        wave that captured it and a run collects hundreds of times. Dropping the
        outputs first matters: they are allocated *inside* the private pool, so
        the pool cannot come back while anything still points into it. Leaving
        this to refcounting alone works but leaves the order implicit, and the
        order is the whole point.
        """
        del self._outputs
        self._graph.reset()


@contextmanager
def _cuda_graph_generation(device: torch.device, mode: str) -> Iterator[None]:
    """Run one step's compiled forwards as a single CUDA graph generation."""
    with ExitStack() as stack:
        if device.type == "cuda" and mode in CAPTURING_ROLLOUT_FORWARD_MODES:
            stack.enter_context(_CUDA_GRAPH_GENERATION)
        _mark_cuda_graph_step(device, mode in COMPILED_ROLLOUT_FORWARD_MODES)
        yield


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


def _validate_reward_gamma(gamma: float) -> None:
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("reward gamma must be finite and in (0, 1]")


def _native_pair_rewards(sampled: dict[str, Any], gamma: float) -> np.ndarray:
    """Build discounted shaping rewards from native state potentials."""
    _validate_reward_gamma(gamma)
    previous = np.asarray(sampled["previous_potentials"], dtype=np.float32)
    following = np.asarray(sampled["potentials"], dtype=np.float32)
    dones = np.asarray(sampled["dones"], dtype=np.bool_)
    reward_zero = np.float32(gamma) * following - previous
    if dones.any():
        utilities = np.asarray(sampled["terminal_utilities"], dtype=np.float32)
        reward_zero[dones] = utilities[dones] - previous[dones]
    return np.column_stack((reward_zero, -reward_zero)).astype(np.float32, copy=False)


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
}
_POLICY_STATISTIC_FIELD_SOURCES = {
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
    "animals",
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
    *,
    store_policy_statistics: bool = True,
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
        fields["critic_animals"][:, step] = np.asarray(encoded["animals"])[pair_rows][
            :, :, _ANIMAL_STOCK_COLUMNS
        ]
        fields["critic_crops"][:, step] = np.asarray(encoded["crops"])[pair_rows][
            :, :, _CROP_SEED_COLUMNS
        ]
    for destination, source in _SAMPLED_FIELD_SOURCES.items():
        fields[destination][:, step] = np.asarray(sampled[source])[rows]
    if store_policy_statistics:
        for destination, source in _POLICY_STATISTIC_FIELD_SOURCES.items():
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
    agents: np.ndarray,
    entropy_sums: np.ndarray,
    learner_stochastic: bool,
    started: float,
    orientations: np.ndarray | None = None,
) -> RolloutBatch:

    state_names = set(_state_field_specs(architecture))
    if orientations is None:
        orientations = np.zeros(agents.shape[0], dtype=np.int8)
    return RolloutBatch(
        architecture=architecture,
        states={name: array for name, array in fields.items() if name in state_names},
        **{name: array for name, array in fields.items() if name not in state_names},
        episode_seeds=episode_seeds,
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        agents=agents,
        orientations=orientations,
        learner_stochastic=learner_stochastic,
        entropy_sums=entropy_sums,
        elapsed_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def _collect_mixed_play_rust_wave(
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
    gamma: float = DEFAULT_REWARD_GAMMA,
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
    _validate_forward_mode(forward_mode)
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
    _validate_reward_gamma(gamma)
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
    floating_dtype = (
        next(actor.trunk.parameters()).dtype
        if isinstance(actor, StructuredActor)
        else next(actor.parameters()).dtype
    )
    encoded_wave = _native_wave(architecture, environment, device, floating_dtype)
    next_encoded_wave = _native_wave(
        architecture,
        environment,
        device,
        floating_dtype,
        device_wave=encoded_wave,
    )
    encoded_wave.refresh(environment)
    encoded_wave.copy_to_device()
    wave_inputs = encoded_wave.inputs()
    gpu_sampling = device.type == "cuda"
    sampled = environment.sample_buffers()
    if gpu_sampling:
        for name in _GPU_STATISTIC_INPUT_NAMES:
            sampled[name] = torch.from_numpy(np.asarray(sampled[name])).pin_memory().numpy()

    league_seats = (seeds[self_play_games:] % 2).astype(np.int64)
    league_game_rows = self_play_rows + 2 * np.arange(league_games, dtype=np.int64)
    league_current_rows = league_game_rows + league_seats
    frozen_rows = league_game_rows + (1 - league_seats)
    stored_rows = np.concatenate([np.arange(self_play_rows, dtype=np.int64), league_current_rows])
    generator = np.random.default_rng(sampling_seed)
    frozen_generator = np.random.default_rng(sampling_seed ^ 0x5EED_1EAF)
    kind_gate, quantity_values, quantity_bias = _quantity_heads((actor, *opponents))
    gpu_heads = _quantity_head_tensors((actor, *opponents)) if gpu_sampling else None
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
    preference_transfer: _GpuPreferenceTransfer | None = None
    gpu_current_generator: torch.Generator | None = None
    gpu_frozen_generator: torch.Generator | None = None
    gpu_current_rows: torch.Tensor | None = None
    gpu_frozen_rows: torch.Tensor | None = None
    gpu_head_ids: torch.Tensor | None = None
    gpu_deterministic_rows: torch.Tensor | None = None
    gpu_temperatures: torch.Tensor | None = None
    statistics_transfer: _GpuStatisticsTransfer | None = None
    pending_statistics: tuple[int, np.ndarray] | None = None
    gpu_builtin_agents: torch.Tensor | None = None
    if gpu_sampling:
        gpu_current_generator = torch.Generator(device=device)
        gpu_current_generator.manual_seed(sampling_seed)
        gpu_frozen_generator = torch.Generator(device=device)
        gpu_frozen_generator.manual_seed(sampling_seed ^ 0x5EED_1EAF)
        gpu_current_rows = torch.as_tensor(stored_rows, dtype=torch.long, device=device)
        gpu_frozen_rows = torch.as_tensor(frozen_rows, dtype=torch.long, device=device)
        gpu_head_ids = torch.as_tensor(head_ids, dtype=torch.long, device=device)
        gpu_deterministic_rows = torch.as_tensor(
            deterministic_rows, dtype=torch.bool, device=device
        )
        gpu_temperatures = torch.as_tensor(temperatures, dtype=torch.float32, device=device)
        gpu_builtin_agents = torch.as_tensor(builtin_agents, dtype=torch.uint8, device=device)

    # A pure self-play wave keeps the learner forward over the contiguous full
    # batch and stores every row, avoiding gather/scatter work entirely.
    store_rows: np.ndarray | slice = slice(None) if not league_games else stored_rows
    stored_pair_rows = stored_rows ^ 1
    current_tensor = None if not league_games else torch.as_tensor(stored_rows, device=device)
    current_gather: tuple[Any, ...] | None = None
    lane_gather: tuple[Any, ...] | None = None
    ensemble: _StackedActorEnsemble | None = None
    frozen_tensor: torch.Tensor | None = None
    frozen_store_tensor: torch.Tensor | None = None
    lane_valid_indices: torch.Tensor | None = None
    frozen_store_rows: np.ndarray | None = None
    lane_valid_flat: np.ndarray | None = None
    lanes = 0
    lane_width = 0
    unit_logits: torch.Tensor | np.ndarray | None = None
    kind_logits: torch.Tensor | np.ndarray | None = None
    quantity_context: torch.Tensor | np.ndarray | None = None
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
        frozen_store_tensor = torch.as_tensor(frozen_store_rows, device=device)
        # Resolve padding on the host once. CUDA boolean indexing would compact
        # dynamically (and synchronize) every step, and cannot be captured.
        lane_valid_indices = torch.as_tensor(np.flatnonzero(lane_valid_flat), device=device)
        # Built-in lanes keep their slot in the stack, borrowing the last
        # frozen opponent's weights. The ensemble batch shape therefore follows
        # total lane count, so a captured CUDA graph survives the next mix.
        # Their logits are never read. A wave with no frozen network runs none.
        ensemble = (
            _stacked_actor_ensemble(
                [opponents[min(index, len(opponents) - 1)] for index in active_indices]
            )
            if opponents
            else None
        )
        # Structured mixed play gathers eleven token tensors for the learner
        # and, when present, eleven more for the lane ensemble on every step.
        # Fixed destinations keep those CUDA addresses stable and replace the
        # per-step allocator traffic with index_select writes into owned memory.
        if architecture == STRUCTURED:
            current_gather = _empty_selected_inputs(wave_inputs, (stored_rows.size,))
            if ensemble is not None:
                lane_gather = _empty_selected_inputs(wave_inputs, (lanes, lane_width))
        # Zeroed rather than uninitialized: with no ensemble nothing scatters
        # into the frozen rows, and handing the sampler uninitialized memory --
        # even in rows it is contracted to ignore -- is not worth the page.
        if gpu_sampling:
            output_dtype = torch.bfloat16 if forward_autocast else floating_dtype
            unit_logits = torch.zeros(
                (rows, MAX_UNITS, N_UNIT_ACTIONS), dtype=output_dtype, device=device
            )
            kind_logits = torch.zeros(
                (rows, MAX_MARKET_ORDERS, N_MARKET_KINDS),
                dtype=output_dtype,
                device=device,
            )
            quantity_context = torch.zeros(
                (rows, MAX_MARKET_ORDERS, actor.config.quantity_rank),
                dtype=output_dtype,
                device=device,
            )
        else:
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

    def run_actor(*inputs: Any) -> ActorOutput:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_forward):
            output = _rollout_model_forward(actor, *inputs, mode=forward_mode)
        assert isinstance(output, ActorOutput)
        return output

    # The learner and frozen-opponent forwards read disjoint gathered buffers.
    # Schedule them on separate streams while keeping one physical-game wave;
    # both streams join before sampling, so every game still advances together.
    frozen_forward_stream = (
        _rollout_cuda_streams(device)[1]
        if league_games and ensemble is not None and device.type == "cuda"
        else None
    )

    def run_frozen() -> ActorOutput | None:
        if ensemble is None:
            return None
        assert frozen_tensor is not None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_forward):
            lane_output = ensemble(
                *_lane_view_inputs(wave_inputs, frozen_tensor, lanes, lane_width, lane_gather),
                mode=forward_mode,
            )
        return ActorOutput(*(tensor.flatten(0, 1) for tensor in lane_output))

    def step_forwards() -> _StepOutputs:
        """Gather, forward and assemble device outputs before host sampling.

        Fixed integer gathers and scatters belong inside the captured region;
        sampling stays outside so graph bootstrap never consumes random draws.
        """
        if not league_games:
            return run_actor(*wave_inputs), None, None
        assert current_tensor is not None
        if frozen_forward_stream is None:
            current = run_actor(*_select_inputs(wave_inputs, current_tensor, current_gather))
            frozen = run_frozen()
        else:
            current_stream = torch.cuda.current_stream(device)
            frozen_forward_stream.wait_stream(current_stream)
            with torch.cuda.stream(frozen_forward_stream):
                frozen = run_frozen()
            current = run_actor(*_select_inputs(wave_inputs, current_tensor, current_gather))
            current_stream.wait_stream(frozen_forward_stream)
        if not gpu_sampling:
            return None, current, frozen

        assert isinstance(unit_logits, torch.Tensor)
        assert isinstance(kind_logits, torch.Tensor)
        assert isinstance(quantity_context, torch.Tensor)
        assert gpu_current_rows is not None
        assert frozen_store_tensor is not None
        assert lane_valid_indices is not None
        for destination, current_values, frozen_values in zip(
            (unit_logits, kind_logits, quantity_context),
            current,
            (None, None, None) if frozen is None else frozen,
            strict=True,
        ):
            destination.index_copy_(0, gpu_current_rows, current_values.to(destination.dtype))
            if frozen_values is not None:
                selected_frozen = frozen_values.index_select(0, lane_valid_indices)
                destination.index_copy_(
                    0, frozen_store_tensor, selected_frozen.to(destination.dtype)
                )
        return ActorOutput(unit_logits, kind_logits, quantity_context), None, None

    graphed = forward_mode in CAPTURED_ROLLOUT_FORWARD_MODES and device.type == "cuda"
    step_graph: _CapturedStep[_StepOutputs] | None = None
    pending_outputs: _StepOutputs | None = None

    for step in range(horizon):
        if graphed and step_graph is None:
            # Capture on the first real uploaded wave and replay the static
            # forward region for the remaining steps.
            step_graph = _CapturedStep(step_forwards)
        if pending_outputs is not None:
            full_output, current_output, frozen_output = pending_outputs
            pending_outputs = None
        elif step_graph is not None:
            full_output, current_output, frozen_output = step_graph()
        else:
            with _cuda_graph_generation(device, forward_mode):
                full_output, current_output, frozen_output = step_forwards()
        if league_games and not gpu_sampling:
            assert current_output is not None
            assert isinstance(unit_logits, np.ndarray)
            assert isinstance(kind_logits, np.ndarray)
            assert isinstance(quantity_context, np.ndarray)
            assert frozen_store_rows is not None
            assert lane_valid_flat is not None
            transfer_outputs = (
                (current_output,) if frozen_output is None else (current_output, frozen_output)
            )
            host_outputs, packed_transfer = _packed_outputs_to_host(
                transfer_outputs, packed_transfer
            )
            current_host = host_outputs[0]
            frozen_host = None if frozen_output is None else host_outputs[1]
            for destination, current_values, frozen_values in zip(
                (unit_logits, kind_logits, quantity_context),
                (
                    current_host.unit_logits,
                    current_host.market_kind_logits,
                    current_host.market_quantity_context,
                ),
                (
                    (None, None, None)
                    if frozen_host is None
                    else (
                        frozen_host.unit_logits,
                        frozen_host.market_kind_logits,
                        frozen_host.market_quantity_context,
                    )
                ),
                strict=True,
            ):
                destination[stored_rows] = current_values
                if frozen_values is not None:
                    destination[frozen_store_rows] = frozen_values[lane_valid_flat]
            full_output = current_output

        assert full_output is not None
        if gpu_sampling:
            assert gpu_heads is not None
            assert gpu_head_ids is not None
            assert gpu_temperatures is not None
            assert gpu_deterministic_rows is not None
            assert gpu_current_rows is not None
            assert gpu_frozen_rows is not None
            assert gpu_current_generator is not None
            assert gpu_frozen_generator is not None
            assert gpu_builtin_agents is not None
            preferences, preference_transfer = _gpu_preferences_to_host(
                full_output,
                gpu_temperatures,
                gpu_deterministic_rows,
                gpu_current_rows,
                gpu_frozen_rows,
                gpu_current_generator,
                gpu_frozen_generator,
                preference_transfer,
            )
            (
                unit_utilities,
                kind_utilities,
                step_quantity_context,
                quantity_draws,
            ) = preferences
            if pending_statistics is not None:
                pending_step, pending_counts = pending_statistics
                assert statistics_transfer is not None
                statistics_transfer.ready.synchronize()
                for destination, values in zip(
                    _POLICY_STATISTIC_FIELD_SOURCES, statistics_transfer.arrays[:3], strict=True
                ):
                    fields[destination][:, pending_step] = values[store_rows]
                entropy_sums += statistics_transfer.arrays[3][store_rows] * pending_counts
                pending_statistics = None
            environment.select_and_step_into(
                unit_utilities,
                kind_utilities,
                step_quantity_context,
                kind_gate,
                quantity_values,
                quantity_bias,
                head_ids,
                quantity_draws,
                deterministic_rows,
                temperatures,
                builtin_agents,
                sampled,
            )
            statistics_transfer = _fill_gpu_policy_statistics(
                sampled,
                full_output,
                gpu_heads,
                gpu_head_ids,
                gpu_builtin_agents,
                gpu_temperatures,
                statistics_transfer,
                synchronize=False,
            )
        else:
            if not league_games:
                host_outputs, packed_transfer = _packed_outputs_to_host(
                    (full_output,), packed_transfer
                )
                host = host_outputs[0]
                step_unit_logits = host.unit_logits
                step_kind_logits = host.market_kind_logits
                step_quantity_context = host.market_quantity_context
                unit_draws, kind_draws, quantity_draws = _categorical_draws(generator, rows)
            else:
                assert isinstance(unit_logits, np.ndarray)
                assert isinstance(kind_logits, np.ndarray)
                assert isinstance(quantity_context, np.ndarray)
                step_unit_logits = np.asarray(unit_logits)
                step_kind_logits = np.asarray(kind_logits)
                step_quantity_context = np.asarray(quantity_context)
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
        if step + 1 < horizon:
            # Both host waves share fixed device destinations. Upload and graph
            # replay use the sampling/statistics stream, so all reads of this
            # step's outputs finish before the next replay overwrites them.
            # The next preference transfer synchronizes that stream before the
            # native step or either pinned host buffer can be refilled.
            next_encoded_wave.refresh(environment)
            next_encoded_wave.copy_to_device()
            if step_graph is not None:
                # Launch ahead of CPU arena storage and consume once at the
                # next loop head. Bootstrap runs above; the final step never
                # queues a speculative extra forward.
                pending_outputs = step_graph()
        rewards = _native_pair_rewards(sampled, gamma).reshape(-1)
        _store_native_wave(
            architecture,
            fields,
            step,
            encoded_wave.arrays,
            sampled,
            rewards[store_rows],
            store_rows,
            stored_pair_rows,
            store_policy_statistics=not gpu_sampling,
        )
        counts = (
            np.asarray(sampled["unit_active"])[store_rows].sum(axis=1)
            + np.asarray(sampled["market_active"])[store_rows].sum(axis=1)
            + np.asarray(sampled["market_quantity_active"])[store_rows].sum(axis=1)
        )
        if gpu_sampling:
            pending_statistics = (step, counts)
        else:
            entropy_sums += np.asarray(sampled["entropy"])[store_rows] * counts
        dones = np.asarray(sampled["dones"], dtype=np.bool_)
        if step + 1 < horizon and dones.any():
            raise RuntimeError("native rollout terminated before the competition horizon")
        if step + 1 == horizon and not dones.all():
            raise RuntimeError("native rollout did not terminate at the competition horizon")
        final = np.asarray(sampled["final_money"], dtype=np.float32)
        if step + 1 < horizon:
            encoded_wave, next_encoded_wave = next_encoded_wave, encoded_wave

    if pending_statistics is not None:
        pending_step, pending_counts = pending_statistics
        assert statistics_transfer is not None
        statistics_transfer.ready.synchronize()
        for destination, values in zip(
            _POLICY_STATISTIC_FIELD_SOURCES, statistics_transfer.arrays[:3], strict=True
        ):
            fields[destination][:, pending_step] = values[store_rows]
        entropy_sums += statistics_transfer.arrays[3][store_rows] * pending_counts

    if step_graph is not None:
        step_graph.close()

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
        agents=np.zeros(trajectories, dtype=np.int64),
        entropy_sums=entropy_sums,
        learner_stochastic=not deterministic,
        started=started,
    )


def _pipeline_replica(
    model: FarmActor | StructuredActor,
    *,
    dtype: torch.dtype | None = None,
    namespace: int = 0,
) -> FarmActor | StructuredActor:
    device = next(model.parameters()).device
    replica_dtype = dtype or next(model.parameters()).dtype
    replicas = getattr(model, "_kaggriculture_pipeline_replicas", None)
    if replicas is None:
        replicas = {}
        object.__setattr__(model, "_kaggriculture_pipeline_replicas", replicas)
    key = (device, replica_dtype, namespace)
    replica = replicas.get(key)
    if replica is None or type(replica) is not type(model) or replica.config != model.config:
        replica = (
            StructuredActor(model.config)
            if isinstance(model, StructuredActor)
            else FarmActor(model.config)
        ).to(device=device, dtype=replica_dtype)
        if replica_dtype == torch.bfloat16:
            # These heads are outside actor.forward: GPU factor sampling uses
            # them in fp32, so keeping them unquantized preserves the current
            # behavior policy while the expensive forward runs natively BF16.
            replica.market_quantity_kind_gate.float()
            replica.market_quantity_value.float()
            replica.market_quantity_bias.data = replica.market_quantity_bias.data.float()
        replicas[key] = replica
    replica.load_state_dict(model.state_dict())
    replica.eval()
    return replica


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
    gamma: float = DEFAULT_REWARD_GAMMA,
    opponent_temperature: float = 1.0,
    opponent_temperatures: Sequence[float] | np.ndarray | None = None,
    deterministic_opponent: bool = False,
    deterministic_opponents: Sequence[bool] | np.ndarray | None = None,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect every game in one native wave and preserve trajectory order."""
    _validate_forward_mode(forward_mode)
    native_bfloat16 = (
        forward_autocast
        and next(actor.parameters()).device.type == "cuda"
        and isinstance(actor, StructuredActor)
    )
    if native_bfloat16:
        rollout_actor = _pipeline_replica(actor, dtype=torch.bfloat16)
        rollout_opponents = tuple(
            _pipeline_replica(opponent, dtype=torch.bfloat16) for opponent in opponents
        )
    else:
        rollout_actor = actor
        rollout_opponents = tuple(opponents)
    return _collect_mixed_play_rust_wave(
        rollout_actor,
        rollout_opponents,
        self_play_games=self_play_games,
        league_games=league_games,
        opponent_indices=opponent_indices,
        builtin_lanes=builtin_lanes,
        seed_start=seed_start,
        episode_steps=episode_steps,
        deterministic=deterministic,
        temperature=temperature,
        gamma=gamma,
        opponent_temperature=opponent_temperature,
        opponent_temperatures=opponent_temperatures,
        deterministic_opponent=deterministic_opponent,
        deterministic_opponents=deterministic_opponents,
        sampling_seed=sampling_seed,
        forward_mode=forward_mode,
        forward_autocast=False if native_bfloat16 else forward_autocast,
        storage=storage,
    )


def population_pairings(population: int, games: int, *, sampling_seed: int = 0) -> np.ndarray:
    """Enumerate a seeded, balanced round-robin population schedule.

    Row ``g`` is ``(seat 0 agent, seat 1 agent)`` for game ``g``. Every ordered
    pair of distinct members appears the same number of times, so each member
    meets every opponent equally often on both seats and the seat *counts*
    cancel exactly rather than in expectation -- which is why `games` must be
    a positive multiple of `population * (population - 1)`. No row ever seats
    a member against itself.

    ``sampling_seed`` permutes the base ordered-pair list, then stratifies each
    complete repetition over game-index orientations. Every pair cycles through
    all four frames in four repetitions, including when the pair count is only
    two modulo four. This keeps exact pair and seat counts while breaking the
    permanent pair-to-map coupling. The one wave, per-agent update partition,
    and head-to-head telemetry all consume these same rows.
    """
    if population < 2:
        raise ValueError("a population wave needs at least two agents")
    orderings = population * (population - 1)
    if games < 1 or games % orderings:
        raise ValueError(
            f"a population of {population} needs games to be a positive multiple of {orderings}"
        )
    ordered = np.asarray(
        [
            (first, second)
            for first in range(population)
            for second in range(population)
            if first != second
        ],
        dtype=np.int64,
    )
    shuffled = ordered[np.random.default_rng(sampling_seed).permutation(orderings)]
    repetitions = games // orderings
    source_orientations = np.arange(orderings) % 4
    orientation_strata = np.asarray(((0, 1, 2, 3), (2, 3, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0)))
    pairings = np.empty((games, 2), dtype=np.int64)
    positions = np.arange(orderings)
    for repetition in range(repetitions):
        desired = orientation_strata[repetition % 4, source_orientations]
        available = (repetition * orderings + positions) % 4
        block = pairings[repetition * orderings : (repetition + 1) * orderings]
        for orientation in range(4):
            block[available == orientation] = shuffled[desired == orientation]
    return pairings


@torch.inference_mode()
def collect_population_play_rust(
    actors: Sequence[FarmActor | StructuredActor],
    *,
    games: int,
    seed_start: int,
    episode_steps: int = EPISODE_STEPS,
    temperature: float = 1.0,
    gamma: float = DEFAULT_REWARD_GAMMA,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect one wave in which every seat is a concurrently learning member.

    Both seats belong to learners and both are stored, so a wave of `games`
    returns `2 * games` trajectories in the native batch's own game-major,
    seat-minor order: `agents[2 * g]` is the seat 0 member of game `g` and
    `agents[2 * g + 1]` its seat 1 member, matching
    `population_pairings(len(actors), games)` row for row. There is no frozen
    lane, no built-in lane and no mirror pairing anywhere in the wave.

    One stacked-ensemble forward covers every row, with lane index equal to
    agent index. That is strictly less work than the mixed league wave it
    replaces, which runs the learner's rows and the frozen ensemble's rows as
    two forwards per step: this is one launch sequence at the same total width.
    The balanced schedule gives every member exactly the same number of rows,
    so the lanes need none of the padding an uneven league mix needs.

    Convolutional games cycle identity / mirror-x / mirror-y / rotate-180,
    with both seats sharing one frame. Every step the encoder output is flipped
    into the game's frame before the forward, and oriented movement logits are
    restriped back to real-action columns for the native sampler. Storage keeps
    the unit factors in the oriented label space, so a row's stored features,
    masks and actions replay consistently through whichever member collected
    them. The structured encoding has no orientation mapping yet, so structured
    population games use the identity frame; their seeded pairing permutation
    still prevents a fixed ordered pair from remaining tied to one map stratum.


    Rollouts capture only behavior policy state. Value predictions for GAE are
    replayed from the stored features at update time, where the critic weights
    are still exactly the behavior weights.
    """
    _validate_forward_mode(forward_mode)
    actors = tuple(actors)
    population = len(actors)
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps != EPISODE_STEPS:
        raise ValueError("the native simulator currently supports the competition horizon 720")
    _validate_learner_temperature(temperature)
    _validate_reward_gamma(gamma)
    pairings = population_pairings(
        population,
        games,
        sampling_seed=sampling_seed ^ _POPULATION_PAIRING_SEED_SALT,
    )
    started = time.perf_counter()
    for member in actors:
        member.eval()
    device = next(actors[0].parameters()).device
    if any(next(member.parameters()).device != device for member in actors):
        raise ValueError("every population member must use the same device")
    if any(
        type(member) is not type(actors[0]) or member.config != actors[0].config
        for member in actors
    ):
        raise ValueError("every population member must use the same model configuration")
    architecture = architecture_of(actors[0]).name

    # The native wave is already game-major and seat-minor, so the schedule
    # flattens straight into the per-row agent index the sampler wants as its
    # lane. The convolutional encoding can cycle symmetries; structured waves
    # remain in the identity frame until that encoding has an equivalent map.
    agents = pairings.reshape(-1)
    codes = (
        seat_orientations(games)
        if architecture == CONV_ENTITY
        else np.zeros(games * 2, dtype=np.int8)
    )

    seeds = np.arange(seed_start, seed_start + games, dtype=np.uint64)
    environment = load_native().BatchEnv(seeds)
    rows = games * 2
    horizon = episode_steps - 1
    fields = _native_rollout_storage(storage, architecture, rows, horizon)
    encoded_wave = _native_wave(architecture, environment, device)
    encoded = encoded_wave.arrays
    sampled = environment.sample_buffers()
    kind_gate, quantity_values, quantity_bias = _quantity_heads(actors)
    head_ids = agents.astype(np.uint16)
    deterministic_rows = np.zeros(rows, dtype=np.bool_)
    temperatures = np.full(rows, temperature, dtype=np.float32)
    builtin_agents = np.zeros(rows, dtype=np.uint8)
    generator = np.random.default_rng(sampling_seed)
    entropy_sums = np.zeros(rows, dtype=np.float64)
    pair_rows = np.arange(rows, dtype=np.int64) ^ 1
    final = None
    packed_transfer: _PackedTransfer | None = None

    ensemble = _stacked_actor_ensemble(actors)
    # Equal row counts per member mean a stable sort by agent folds the wave
    # into full lanes, so the ensemble's shape is fixed by the population size
    # alone and a captured CUDA graph survives every later wave. The scatter
    # back is the inverse of that gather; every row is written each step, which
    # is why the staging buffers below need no initialization.
    lane_width = rows // population
    lane_rows = np.argsort(agents, kind="stable")
    # `_lane_view_inputs` folds with a `view`, so an unbalanced schedule would
    # mis-group the lanes silently rather than fail: a member's rows would
    # replay under another's weights, and stored behavior log-probabilities
    # that belong to the wrong policy look exactly like ordinary data.
    if not (np.bincount(agents, minlength=population) == lane_width).all():
        raise ValueError("a population wave needs the same row count for every member")
    lane_tensor = torch.as_tensor(lane_rows, device=device)
    unit_logits = np.empty((rows, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.float32)
    kind_logits = np.empty((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.float32)
    quantity_context = np.empty(
        (rows, MAX_MARKET_ORDERS, actors[0].config.quantity_rank), dtype=np.float32
    )
    autocast_forward = forward_autocast and device.type == "cuda"

    def step_forward() -> ActorOutput:
        """The population wave's whole device-side forward, as one capturable unit."""
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_forward):
            lane_output = ensemble(
                *_lane_view_inputs(encoded_wave.inputs(), lane_tensor, population, lane_width),
                mode=forward_mode,
            )
        return ActorOutput(*(tensor.flatten(0, 1) for tensor in lane_output))

    graphed = forward_mode in CAPTURED_ROLLOUT_FORWARD_MODES and device.type == "cuda"
    step_graph: _CapturedStep[ActorOutput] | None = None
    for step in range(horizon):
        encoded_wave.refresh(environment)
        # Convolutional rows are re-encoded in the native frame every step, so
        # apply their selected symmetry before upload. Structured rows use the
        # identity frame and expose different field names.
        if architecture == CONV_ENTITY:
            orient_boards(encoded["board"], codes)
            orient_unit_features(encoded["units"], encoded["unit_positions"], codes)
        encoded_wave.copy_to_device()
        if graphed and step_graph is None:
            step_graph = _CapturedStep(step_forward)
        if step_graph is not None:
            output = step_graph()
        else:
            with _cuda_graph_generation(device, forward_mode):
                output = step_forward()
        host_outputs, packed_transfer = _packed_outputs_to_host((output,), packed_transfer)
        (host,) = host_outputs
        for destination, values in zip(
            (unit_logits, kind_logits, quantity_context),
            (host.unit_logits, host.market_kind_logits, host.market_quantity_context),
            strict=True,
        ):
            destination[lane_rows] = values
        unit_draws, kind_draws, quantity_draws = _categorical_draws(generator, rows)
        # Only convolutional forwards can emit symmetry-oriented movement
        # logits. Structured logits already use native action columns.
        if architecture == CONV_ENTITY:
            orient_unit_logits(unit_logits, codes)
        environment.sample_and_step_into(
            unit_logits,
            kind_logits,
            quantity_context,
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
        rewards = _native_pair_rewards(sampled, gamma).reshape(-1)
        # Storage keeps the unit factors in oriented space so the replay path
        # reads features, masks and actions out of one label space. The stored
        # log-probabilities need no remap: P(oriented index i) and P(the real
        # action it executes) are the same categorical event, and neither the
        # market fields nor entropy touch movement columns.
        oriented_sampled = dict(sampled)
        oriented_sampled["unit_actions"] = orient_unit_actions(sampled["unit_actions"], codes)
        oriented_sampled["unit_masks"] = orient_unit_masks(sampled["unit_masks"], codes)
        _store_native_wave(
            architecture,
            fields,
            step,
            encoded,
            oriented_sampled,
            rewards,
            slice(None),
            pair_rows,
        )
        counts = (
            np.asarray(sampled["unit_active"]).sum(axis=1)
            + np.asarray(sampled["market_active"]).sum(axis=1)
            + np.asarray(sampled["market_quantity_active"]).sum(axis=1)
        )
        entropy_sums += np.asarray(sampled["entropy"]) * counts
        dones = np.asarray(sampled["dones"], dtype=np.bool_)
        if step + 1 < horizon and dones.any():
            raise RuntimeError("native rollout terminated before the competition horizon")
        if step + 1 == horizon and not dones.all():
            raise RuntimeError("native rollout did not terminate at the competition horizon")
        final = np.asarray(sampled["final_money"], dtype=np.float32)

    if step_graph is not None:
        step_graph.close()

    assert final is not None
    # Copied off the native buffer rather than viewed: the batch outlives this
    # environment, and a view would keep the whole simulator batch alive.
    return _native_batch(
        architecture,
        fields,
        episode_seeds=np.repeat(seeds.astype(np.int64), 2),
        final_money=np.array(final.reshape(-1)),
        opponent_money=np.array(final[:, ::-1].reshape(-1)),
        seats=np.tile(np.asarray([0, 1], dtype=np.int8), games),
        agents=agents,
        orientations=codes,
        learner_stochastic=True,
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
    gamma: float = DEFAULT_REWARD_GAMMA,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect both on-policy seats through the exact batched Rust simulator."""
    _validate_forward_mode(forward_mode)
    if games < 1:
        raise ValueError("games must be positive")
    return collect_mixed_play_rust(
        actor,
        self_play_games=games,
        seed_start=seed_start,
        episode_steps=episode_steps,
        deterministic=deterministic,
        temperature=temperature,
        gamma=gamma,
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
    gamma: float = DEFAULT_REWARD_GAMMA,
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
    _validate_forward_mode(forward_mode)
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
        gamma=gamma,
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
    gamma: float = DEFAULT_REWARD_GAMMA,
    opponent_temperature: float = 0.8,
    deterministic_opponent: bool = False,
    deterministic: bool = False,
    sampling_seed: int = 0,
    forward_mode: str = "cudagraphs",
    forward_autocast: bool = False,
) -> RolloutBatch:
    """Collect one current-policy seat per native game against one frozen actor."""
    _validate_forward_mode(forward_mode)
    return collect_frozen_opponents_play_rust(
        actor,
        (opponent,),
        games=games,
        seed_start=seed_start,
        episode_steps=episode_steps,
        temperature=temperature,
        gamma=gamma,
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
    gamma: float = DEFAULT_REWARD_GAMMA,
    sampling_seed: int = 0,
) -> RolloutBatch:
    """Collect both valid on-policy trajectories from every self-play game."""
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps < 2:
        raise ValueError("episode_steps must be at least two")
    _validate_learner_temperature(temperature)
    _validate_reward_gamma(gamma)
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
    potentials = np.asarray(
        [pair_potential(state[0].observation, state[1].observation) for state in states],
        dtype=np.float32,
    )
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
                utility = terminal_pair_utility(
                    next_state[0].observation, next_state[1].observation
                )
                next_potential = np.float32(0.0)
                pair_rewards = shaped_pair_reward(
                    potentials[game], None, terminal_utility=utility, gamma=gamma
                )
            else:
                next_potential = np.float32(
                    pair_potential(next_state[0].observation, next_state[1].observation)
                )
                pair_rewards = shaped_pair_reward(potentials[game], next_potential, gamma=gamma)
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
        learner_stochastic=not deterministic,
        agents=np.zeros(trajectories, dtype=np.int64),
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
    gamma: float = DEFAULT_REWARD_GAMMA,
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
    _validate_reward_gamma(gamma)
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
    potentials = np.asarray(
        [pair_potential(state[0].observation, state[1].observation) for state in states],
        dtype=np.float32,
    )
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
            actions: list[dict[str, Any] | None] = [None, None]
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
                utility = terminal_pair_utility(
                    next_state[0].observation, next_state[1].observation
                )
                next_potential = np.float32(0.0)
                pair_rewards = shaped_pair_reward(
                    potentials[game], None, terminal_utility=utility, gamma=gamma
                )
            else:
                next_potential = np.float32(
                    pair_potential(next_state[0].observation, next_state[1].observation)
                )
                pair_rewards = shaped_pair_reward(potentials[game], next_potential, gamma=gamma)
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
        agents=np.zeros(games, dtype=np.int64),
        entropy_sums=entropy_sums,
        learner_stochastic=not deterministic,
        started=started,
    )


_TRAJECTORY_METADATA_FIELDS = (
    "episode_seeds",
    "final_money",
    "opponent_money",
    "seats",
    "agents",
    "orientations",
    "entropy_sums",
)


def _combined_rollout_metadata(batches: list[RolloutBatch]) -> dict[str, Any]:
    """Concatenate per-trajectory metadata across compatible batches."""
    provenance = {batch.learner_stochastic for batch in batches}
    if len(provenance) != 1:
        raise ValueError("rollout learner sampling provenance must match")
    combined: dict[str, Any] = {
        field: np.concatenate([getattr(batch, field) for batch in batches], axis=0)
        for field in _TRAJECTORY_METADATA_FIELDS
    }
    combined["elapsed_seconds"] = sum(batch.elapsed_seconds for batch in batches)
    combined["learner_stochastic"] = batches[0].learner_stochastic
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
        learner_stochastic=batch.learner_stochastic,
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
