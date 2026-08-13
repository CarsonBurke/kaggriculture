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
from kaggriculture.constants import BOARD_SIZE, MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
    pair_potential,
    shaped_pair_reward,
)
from kaggriculture.model import ActorOutput, DistributionalCritic, FarmActor
from kaggriculture.policy import PolicyStep, act_batch
from kaggriculture.rust_env import load_native

_MAX_FLOAT32_CATEGORICAL_DRAW = np.nextafter(np.float32(1.0), np.float32(0.0))


@dataclass(frozen=True)
class RolloutBatch:
    board: np.ndarray
    global_features: np.ndarray
    critic_features: np.ndarray
    units: np.ndarray
    unit_positions: np.ndarray
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
    old_values: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    episode_seeds: np.ndarray
    final_money: np.ndarray
    opponent_money: np.ndarray
    seats: np.ndarray
    mean_entropy: float
    elapsed_seconds: float

    @property
    def trajectories(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def states(self) -> int:
        return int(self.valid.sum())


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


def _new_fields() -> dict[str, list[np.ndarray]]:
    return {
        name: []
        for name in (
            "board",
            "global_features",
            "critic_features",
            "units",
            "unit_positions",
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "unit_masks",
            "market_kind_masks",
            "market_quantity_masks",
            "unit_active",
            "market_active",
            "market_quantity_active",
            "old_unit_logprobs",
            "old_market_kind_logprobs",
            "old_market_quantity_logprobs",
            "old_values",
            "rewards",
            "valid",
        )
    }


def _record_policy_step(fields: dict[str, list[np.ndarray]], policy_step: PolicyStep) -> None:
    factors = policy_step.factors
    fields["board"].append(np.stack([row.board for row in policy_step.encoded]).astype(np.float16))
    fields["global_features"].append(
        np.stack([row.global_features for row in policy_step.encoded]).astype(np.float16)
    )
    fields["critic_features"].append(
        np.stack([row.critic_features for row in policy_step.encoded]).astype(np.float16)
    )
    fields["units"].append(np.stack([row.units for row in policy_step.encoded]).astype(np.float16))
    fields["unit_positions"].append(
        np.stack([row.unit_positions for row in policy_step.encoded]).astype(np.int8)
    )
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
    fields["old_values"].append(factors.values)


def _finish_rollout(
    fields: dict[str, list[np.ndarray]],
    *,
    episode_seeds: np.ndarray,
    final_money: np.ndarray,
    opponent_money: np.ndarray,
    seats: np.ndarray,
    entropies: list[float],
    started: float,
) -> RolloutBatch:
    return RolloutBatch(
        board=_trajectory_first(fields["board"]),
        global_features=_trajectory_first(fields["global_features"]),
        critic_features=_trajectory_first(fields["critic_features"]),
        units=_trajectory_first(fields["units"]),
        unit_positions=_trajectory_first(fields["unit_positions"]),
        unit_actions=_trajectory_first(fields["unit_actions"]),
        market_kinds=_trajectory_first(fields["market_kinds"]),
        market_quantities=_trajectory_first(fields["market_quantities"]),
        unit_masks=_trajectory_first(fields["unit_masks"]),
        market_kind_masks=_trajectory_first(fields["market_kind_masks"]),
        market_quantity_masks=_trajectory_first(fields["market_quantity_masks"]),
        unit_active=_trajectory_first(fields["unit_active"]),
        market_active=_trajectory_first(fields["market_active"]),
        market_quantity_active=_trajectory_first(fields["market_quantity_active"]),
        old_unit_logprobs=_trajectory_first(fields["old_unit_logprobs"]),
        old_market_kind_logprobs=_trajectory_first(fields["old_market_kind_logprobs"]),
        old_market_quantity_logprobs=_trajectory_first(fields["old_market_quantity_logprobs"]),
        old_values=_trajectory_first(fields["old_values"]),
        rewards=_trajectory_first(fields["rewards"]),
        valid=_trajectory_first(fields["valid"]),
        episode_seeds=episode_seeds,
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        mean_entropy=float(np.mean(entropies)),
        elapsed_seconds=time.perf_counter() - started,
    )


def _native_field_specs(trajectories: int, horizon: int) -> dict[str, tuple[tuple[int, ...], type]]:
    """Return the trajectory-major shape and dtype of every native rollout field."""
    prefix = (trajectories, horizon)
    return {
        "board": ((*prefix, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), np.float16),
        "global_features": ((*prefix, GLOBAL_FEATURES), np.float16),
        "critic_features": ((*prefix, CRITIC_FEATURES), np.float16),
        "units": ((*prefix, MAX_UNITS, UNIT_FEATURES), np.float16),
        "unit_positions": ((*prefix, MAX_UNITS, 2), np.int8),
        "unit_actions": ((*prefix, MAX_UNITS), np.int8),
        "market_kinds": ((*prefix, MAX_MARKET_ORDERS), np.int8),
        "market_quantities": ((*prefix, MAX_MARKET_ORDERS), np.int8),
        "unit_masks": ((*prefix, MAX_UNITS, N_UNIT_ACTIONS), np.bool_),
        "market_kind_masks": ((*prefix, MAX_MARKET_ORDERS, N_MARKET_KINDS), np.bool_),
        "market_quantity_masks": ((*prefix, MAX_MARKET_ORDERS, N_QUANTITIES), np.bool_),
        "unit_active": ((*prefix, MAX_UNITS), np.bool_),
        "market_active": ((*prefix, MAX_MARKET_ORDERS), np.bool_),
        "market_quantity_active": ((*prefix, MAX_MARKET_ORDERS), np.bool_),
        "old_unit_logprobs": ((*prefix, MAX_UNITS), np.float32),
        "old_market_kind_logprobs": ((*prefix, MAX_MARKET_ORDERS), np.float32),
        "old_market_quantity_logprobs": ((*prefix, MAX_MARKET_ORDERS), np.float32),
        "old_values": (prefix, np.float32),
        "rewards": (prefix, np.float32),
        "valid": (prefix, np.bool_),
    }


_TORCH_STORAGE_DTYPES = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.bool_): torch.bool,
}


def allocate_rollout_storage(
    trajectories: int, horizon: int, *, pin_memory: bool = False
) -> dict[str, np.ndarray]:
    """Allocate reusable trajectory-major rollout storage.

    Pinned storage is allocated through page-locked torch tensors and exposed
    as NumPy views, so replay staging can upload the complete rollout to the
    accelerator asynchronously instead of through pageable-memory copies.
    """
    if trajectories < 1 or horizon < 1:
        raise ValueError("rollout storage requires positive trajectories and horizon")
    storage: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in _native_field_specs(trajectories, horizon).items():
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
    storage: dict[str, np.ndarray] | None, trajectories: int, horizon: int
) -> dict[str, np.ndarray]:
    """Validate caller-provided storage or allocate a fresh full-horizon block."""
    if storage is None:
        return allocate_rollout_storage(trajectories, horizon)
    specs = _native_field_specs(trajectories, horizon)
    if set(storage) != set(specs):
        raise ValueError("rollout storage fields do not match the native layout")
    for name, (shape, dtype) in specs.items():
        array = storage[name]
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"rollout storage field {name} has the wrong shape or dtype")
    storage["valid"][:] = True
    return storage


def _quantity_heads(actors: tuple[FarmActor, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the small selected-kind quantity heads once per rollout."""

    def parameter(actor: FarmActor, name: str) -> np.ndarray:
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


@dataclass(frozen=True)
class _NativeEncodedWave:
    arrays: dict[str, np.ndarray]
    host_features: torch.Tensor
    host_positions: torch.Tensor
    device_features: torch.Tensor
    board: torch.Tensor
    global_features: torch.Tensor
    critic_features: torch.Tensor
    units: torch.Tensor
    unit_positions: torch.Tensor

    def copy_to_device(self) -> None:
        non_blocking = self.device_features.device.type == "cuda"
        self.device_features.copy_(self.host_features, non_blocking=non_blocking)
        self.unit_positions.copy_(self.host_positions, non_blocking=non_blocking)


def _native_encoded_wave(environment: Any, device: torch.device) -> _NativeEncodedWave:
    """Build reusable Rust output plus packed, optionally pinned model input buffers."""
    arrays = {name: np.asarray(value) for name, value in environment.encoded_buffers().items()}
    feature_names = ("board", "global_features", "critic_features", "units")
    feature_sizes = [arrays[name].size for name in feature_names]
    pin_memory = device.type == "cuda"
    host_features = torch.empty(sum(feature_sizes), dtype=torch.float16, pin_memory=pin_memory)
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
        device_features=device_features,
        board=device_views["board"],
        global_features=device_views["global_features"],
        critic_features=device_views["critic_features"],
        units=device_views["units"],
        unit_positions=unit_positions,
    )


@dataclass(frozen=True)
class _HostActorOutput:
    unit_logits: np.ndarray
    market_kind_logits: np.ndarray
    market_quantity_context: np.ndarray


def _packed_outputs_to_host(
    outputs: tuple[ActorOutput, ...],
    values: torch.Tensor | None,
    pinned_buffer: torch.Tensor | None,
) -> tuple[list[_HostActorOutput], np.ndarray | None, torch.Tensor | None]:
    """Transfer all policy heads and optional values with one device sync."""
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
            None if values is None else values.float().numpy(),
            pinned_buffer,
        )

    tensors: list[torch.Tensor] = []
    shapes: list[tuple[int, ...]] = []
    for output in outputs:
        for tensor in (
            output.unit_logits,
            output.market_kind_logits,
            output.market_quantity_context,
        ):
            tensor = tensor.float()
            tensors.append(tensor.reshape(-1))
            shapes.append(tuple(tensor.shape))
    if values is not None:
        values = values.float()
        tensors.append(values.reshape(-1))
        shapes.append(tuple(values.shape))
    packed = torch.cat(tensors)
    if pinned_buffer is None or pinned_buffer.numel() != packed.numel():
        pinned_buffer = torch.empty(
            packed.numel(), dtype=torch.float32, device="cpu", pin_memory=True
        )
    pinned_buffer.copy_(packed, non_blocking=True)
    # Native sampling consumes the host buffer immediately. This is the
    # single required D2H synchronization for the complete model wave.
    torch.cuda.current_stream(packed.device).synchronize()
    flat = pinned_buffer.numpy()

    cursor = 0
    arrays = []
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(flat[cursor : cursor + size].reshape(shape))
        cursor += size
    host_outputs = [
        _HostActorOutput(*arrays[index : index + 3]) for index in range(0, 3 * len(outputs), 3)
    ]
    host_values = arrays[-1] if values is not None else None
    return host_outputs, host_values, pinned_buffer


def _cached_compiled_forward(model: FarmActor | DistributionalCritic) -> Any:
    """Capture the native ATen rollout forward in a CUDA graph.

    PPO replays stored behavior likelihoods through the eager FP32 actor. The
    CUDA-graphs-only backend retains native ATen operations while removing their
    repeated launch overhead. CUDA convolution and GEMM kernels are not bitwise
    invariant across eager/graph execution or batch shapes; the rollout benchmark
    therefore enforces a tight semantic importance-ratio bound. Keep Inductor out
    of this path: its additional fusion creates materially larger policy drift.
    """
    compiled = getattr(model, "_kaggriculture_rollout_forward", None)
    if compiled is None:
        compiled = torch.compile(
            model.forward,
            backend="cudagraphs",
            fullgraph=True,
            dynamic=False,
        )
        # Avoid registering the compiled wrapper as a child module, which
        # would pollute checkpoints with a second copy of every parameter.
        object.__setattr__(model, "_kaggriculture_rollout_forward", compiled)
    return compiled


def _rollout_model_forward(
    model: FarmActor | DistributionalCritic,
    *inputs: torch.Tensor,
    compile_model: bool,
) -> ActorOutput | torch.Tensor:
    """Run one static rollout wave, optionally through a cached compiled graph."""
    if not compile_model or inputs[0].device.type != "cuda":
        return model(*inputs)
    return _cached_compiled_forward(model)(*inputs)


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


def _validate_learner_temperature(temperature: float) -> None:
    """Require behavior logits to match the unit-temperature PPO replay policy."""
    if not np.isfinite(temperature) or temperature != 1.0:
        raise ValueError("on-policy rollout collection requires learner temperature 1.0")


def _store_native_wave(
    fields: dict[str, np.ndarray],
    step: int,
    encoded: dict[str, np.ndarray],
    sampled: dict[str, np.ndarray],
    values: np.ndarray,
    rewards: np.ndarray,
    rows: np.ndarray | slice = slice(None),
) -> None:
    for name in ("board", "global_features", "critic_features", "units", "unit_positions"):
        fields[name][:, step] = np.asarray(encoded[name])[rows]
    mapping = {
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
    for destination, source in mapping.items():
        fields[destination][:, step] = np.asarray(sampled[source])[rows]
    fields["old_values"][:, step] = values
    fields["rewards"][:, step] = rewards


def _native_batch(
    fields: dict[str, np.ndarray],
    *,
    episode_seeds: np.ndarray,
    final_money: np.ndarray,
    opponent_money: np.ndarray,
    seats: np.ndarray,
    entropy_sum: float,
    component_count: int,
    started: float,
) -> RolloutBatch:
    return RolloutBatch(
        **fields,
        episode_seeds=episode_seeds,
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        mean_entropy=entropy_sum / max(1, component_count),
        elapsed_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def collect_self_play_rust(
    actor: FarmActor,
    critic: DistributionalCritic,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    deterministic: bool = False,
    temperature: float = 1.0,
    sampling_seed: int = 0,
    compile_models: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect both on-policy seats through the exact batched Rust simulator."""
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps != 720:
        raise ValueError("the native simulator currently supports the competition horizon 720")
    _validate_learner_temperature(temperature)
    started = time.perf_counter()
    actor.eval()
    critic.eval()
    device = next(actor.parameters()).device
    if next(critic.parameters()).device != device:
        raise ValueError("actor and critic must use the same device")
    seeds = np.arange(seed_start, seed_start + games, dtype=np.uint64)
    environment = load_native().BatchEnv(seeds)
    trajectories = games * 2
    horizon = episode_steps - 1
    fields = _native_rollout_storage(storage, trajectories, horizon)
    encoded_wave = _native_encoded_wave(environment, device)
    encoded = encoded_wave.arrays
    sampled = environment.sample_buffers()
    generator = np.random.default_rng(sampling_seed)
    kind_gate, quantity_values, quantity_bias = _quantity_heads((actor,))
    head_ids = np.zeros(trajectories, dtype=np.uint16)
    deterministic_rows = np.full(trajectories, deterministic, dtype=np.bool_)
    temperatures = np.full(trajectories, temperature, dtype=np.float32)
    entropy_sum = 0.0
    component_count = 0
    final = None
    packed_host = None
    for step in range(horizon):
        environment.encoded_into(encoded)
        encoded_wave.copy_to_device()
        _mark_cuda_graph_step(device, compile_models)
        output = _rollout_model_forward(
            actor,
            encoded_wave.board,
            encoded_wave.global_features,
            encoded_wave.units,
            encoded_wave.unit_positions,
            compile_model=compile_models,
        )
        assert isinstance(output, ActorOutput)
        value_tensor = critic.value(
            _rollout_model_forward(
                critic,
                encoded_wave.board,
                encoded_wave.critic_features,
                compile_model=compile_models,
            )
        )
        host_outputs, values, packed_host = _packed_outputs_to_host(
            (output,), value_tensor, packed_host
        )
        host_output = host_outputs[0]
        assert values is not None
        unit_draws, kind_draws, quantity_draws = _categorical_draws(generator, trajectories)
        environment.sample_and_step_into(
            host_output.unit_logits,
            host_output.market_kind_logits,
            host_output.market_quantity_context,
            kind_gate,
            quantity_values,
            quantity_bias,
            head_ids,
            unit_draws,
            kind_draws,
            quantity_draws,
            deterministic_rows,
            temperatures,
            sampled,
        )
        rewards = np.asarray(sampled["shaped_rewards"], dtype=np.float32).reshape(-1)
        _store_native_wave(fields, step, encoded, sampled, values, rewards)
        counts = (
            np.asarray(sampled["unit_active"]).sum(axis=1)
            + np.asarray(sampled["market_active"]).sum(axis=1)
            + np.asarray(sampled["market_quantity_active"]).sum(axis=1)
        )
        entropy_sum += float(np.dot(np.asarray(sampled["entropy"]), counts))
        component_count += int(counts.sum())
        dones = np.asarray(sampled["dones"], dtype=np.bool_)
        if step + 1 < horizon and dones.any():
            raise RuntimeError("native self-play terminated before the competition horizon")
        if step + 1 == horizon and not dones.all():
            raise RuntimeError("native self-play did not terminate at the competition horizon")
        final = np.asarray(sampled["final_money"], dtype=np.float32)

    assert final is not None
    final_money = final.reshape(-1)
    opponent_money = final[:, ::-1].reshape(-1)
    return _native_batch(
        fields,
        episode_seeds=np.repeat(seeds.astype(np.int64), 2),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=np.tile(np.asarray([0, 1], dtype=np.int8), games),
        entropy_sum=entropy_sum,
        component_count=component_count,
        started=started,
    )


@torch.inference_mode()
def collect_frozen_opponents_play_rust(
    actor: FarmActor,
    critic: DistributionalCritic,
    opponents: Sequence[FarmActor],
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
    compile_models: bool = False,
    storage: dict[str, np.ndarray] | None = None,
) -> RolloutBatch:
    """Collect one current-policy seat per game against assigned frozen actors."""
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps != 720:
        raise ValueError("the native simulator currently supports the competition horizon 720")
    opponents = tuple(opponents)
    if not opponents:
        raise ValueError("at least one frozen opponent is required")
    if len(opponents) > np.iinfo(np.uint16).max:
        raise ValueError("too many frozen opponents for native head identifiers")
    _validate_learner_temperature(temperature)
    if opponent_temperatures is None:
        frozen_temperatures = np.full(len(opponents), opponent_temperature, dtype=np.float32)
    else:
        frozen_temperatures = np.asarray(opponent_temperatures, dtype=np.float32)
        if frozen_temperatures.shape != (len(opponents),):
            raise ValueError(f"opponent temperatures must have shape {(len(opponents),)}")
    if not np.isfinite(frozen_temperatures).all() or (frozen_temperatures <= 0.0).any():
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
        assignments = np.zeros(games, dtype=np.int64)
    else:
        raw_assignments = np.asarray(opponent_indices)
        if raw_assignments.shape != (games,):
            raise ValueError(f"opponent indices must have shape {(games,)}")
        if not np.issubdtype(raw_assignments.dtype, np.integer):
            raise ValueError("opponent indices must be integers")
        assignments = raw_assignments.astype(np.int64, copy=False)
    if (assignments < 0).any() or (assignments >= len(opponents)).any():
        raise ValueError("opponent index is outside the frozen opponent list")
    started = time.perf_counter()
    actor.eval()
    critic.eval()
    for opponent in opponents:
        opponent.eval()
    device = next(actor.parameters()).device
    if next(critic.parameters()).device != device or any(
        next(opponent.parameters()).device != device for opponent in opponents
    ):
        raise ValueError("current, critic, and all frozen models must use the same device")
    if any(opponent.config != actor.config for opponent in opponents):
        raise ValueError("current and frozen actors must use the same model configuration")
    seeds = np.arange(seed_start, seed_start + games, dtype=np.uint64)
    environment = load_native().BatchEnv(seeds)
    rows = games * 2
    horizon = episode_steps - 1
    fields = _native_rollout_storage(storage, games, horizon)
    encoded_wave = _native_encoded_wave(environment, device)
    encoded = encoded_wave.arrays
    sampled = environment.sample_buffers()
    seats = (seeds % 2).astype(np.int8)
    current_rows = np.arange(games, dtype=np.int64) * 2 + seats
    frozen_rows = np.arange(games, dtype=np.int64) * 2 + (1 - seats)
    generator = np.random.default_rng(sampling_seed)
    frozen_generator = np.random.default_rng(sampling_seed ^ 0x5EED_1EAF)
    kind_gate, quantity_values, quantity_bias = _quantity_heads((actor, *opponents))
    head_ids = np.zeros(rows, dtype=np.uint16)
    head_ids[frozen_rows] = (assignments + 1).astype(np.uint16)
    head_ids[current_rows] = 0
    deterministic_rows = np.zeros(rows, dtype=np.bool_)
    deterministic_rows[frozen_rows] = frozen_deterministic[assignments]
    deterministic_rows[current_rows] = deterministic
    temperatures = np.ones(rows, dtype=np.float32)
    temperatures[frozen_rows] = frozen_temperatures[assignments]
    temperatures[current_rows] = temperature
    entropy_sum = 0.0
    component_count = 0
    final = None
    current_tensor = torch.as_tensor(current_rows, device=device)
    frozen_groups = tuple(
        frozen_rows[np.flatnonzero(assignments == opponent_index)]
        for opponent_index in range(len(opponents))
    )
    active_frozen = tuple(
        (opponent, selected_rows, torch.as_tensor(selected_rows, device=device))
        for opponent, selected_rows in zip(opponents, frozen_groups, strict=True)
        if selected_rows.size
    )
    unit_logits = np.empty((rows, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.float32)
    kind_logits = np.empty((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.float32)
    quantity_context = np.empty(
        (rows, MAX_MARKET_ORDERS, actor.config.quantity_rank), dtype=np.float32
    )
    packed_host = None
    for step in range(horizon):
        environment.encoded_into(encoded)
        encoded_wave.copy_to_device()
        _mark_cuda_graph_step(device, compile_models)
        current_output = _rollout_model_forward(
            actor,
            encoded_wave.board.index_select(0, current_tensor),
            encoded_wave.global_features.index_select(0, current_tensor),
            encoded_wave.units.index_select(0, current_tensor),
            encoded_wave.unit_positions.index_select(0, current_tensor),
            compile_model=compile_models,
        )
        assert isinstance(current_output, ActorOutput)
        frozen_outputs: list[ActorOutput] = []
        selected_groups: list[np.ndarray] = []
        for opponent, selected_rows, selected_tensor in active_frozen:
            frozen_output = _rollout_model_forward(
                opponent,
                encoded_wave.board.index_select(0, selected_tensor),
                encoded_wave.global_features.index_select(0, selected_tensor),
                encoded_wave.units.index_select(0, selected_tensor),
                encoded_wave.unit_positions.index_select(0, selected_tensor),
                compile_model=compile_models,
            )
            assert isinstance(frozen_output, ActorOutput)
            frozen_outputs.append(frozen_output)
            selected_groups.append(selected_rows)
        value_tensor = critic.value(
            _rollout_model_forward(
                critic,
                encoded_wave.board.index_select(0, current_tensor),
                encoded_wave.critic_features.index_select(0, current_tensor),
                compile_model=compile_models,
            )
        )
        host_outputs, values, packed_host = _packed_outputs_to_host(
            (current_output, *frozen_outputs), value_tensor, packed_host
        )
        assert values is not None
        for output, selected in zip(
            host_outputs,
            (current_rows, *selected_groups),
            strict=True,
        ):
            unit_logits[selected] = output.unit_logits
            kind_logits[selected] = output.market_kind_logits
            quantity_context[selected] = output.market_quantity_context
        unit_draws = np.empty((rows, MAX_UNITS), dtype=np.float32)
        kind_draws = np.empty((rows, MAX_MARKET_ORDERS), dtype=np.float32)
        quantity_draws = np.empty((rows, MAX_MARKET_ORDERS), dtype=np.float32)
        current_draws = _categorical_draws(generator, games)
        frozen_draws = _categorical_draws(frozen_generator, games)
        for destination, current_values, frozen_values in zip(
            (unit_draws, kind_draws, quantity_draws), current_draws, frozen_draws, strict=True
        ):
            destination[current_rows] = current_values
            destination[frozen_rows] = frozen_values
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
            sampled,
        )
        all_rewards = np.asarray(sampled["shaped_rewards"], dtype=np.float32).reshape(-1)
        _store_native_wave(
            fields,
            step,
            encoded,
            sampled,
            values,
            all_rewards[current_rows],
            current_rows,
        )
        counts = (
            np.asarray(sampled["unit_active"])[current_rows].sum(axis=1)
            + np.asarray(sampled["market_active"])[current_rows].sum(axis=1)
            + np.asarray(sampled["market_quantity_active"])[current_rows].sum(axis=1)
        )
        entropy_sum += float(np.dot(np.asarray(sampled["entropy"])[current_rows], counts))
        component_count += int(counts.sum())
        dones = np.asarray(sampled["dones"], dtype=np.bool_)
        if step + 1 < horizon and dones.any():
            raise RuntimeError("native league play terminated before the competition horizon")
        if step + 1 == horizon and not dones.all():
            raise RuntimeError("native league play did not terminate at the competition horizon")
        final = np.asarray(sampled["final_money"], dtype=np.float32)

    assert final is not None
    game_indices = np.arange(games)
    final_money = final[game_indices, seats]
    opponent_money = final[game_indices, 1 - seats]
    return _native_batch(
        fields,
        episode_seeds=seeds.astype(np.int64),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        entropy_sum=entropy_sum,
        component_count=component_count,
        started=started,
    )


def collect_frozen_opponent_play_rust(
    actor: FarmActor,
    critic: DistributionalCritic,
    opponent: FarmActor,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    temperature: float = 1.0,
    opponent_temperature: float = 0.8,
    deterministic_opponent: bool = False,
    deterministic: bool = False,
    sampling_seed: int = 0,
    compile_models: bool = False,
) -> RolloutBatch:
    """Collect one current-policy seat per native game against one frozen actor."""
    return collect_frozen_opponents_play_rust(
        actor,
        critic,
        (opponent,),
        games=games,
        seed_start=seed_start,
        episode_steps=episode_steps,
        temperature=temperature,
        opponent_temperature=opponent_temperature,
        deterministic_opponent=deterministic_opponent,
        deterministic=deterministic,
        sampling_seed=sampling_seed,
        compile_models=compile_models,
    )


def collect_self_play(
    actor: FarmActor,
    critic: DistributionalCritic,
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
    critic.eval()
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
    fields = _new_fields()
    trajectories = games * 2
    final_money = np.zeros(trajectories, dtype=np.float32)
    opponent_money = np.zeros(trajectories, dtype=np.float32)
    entropies = []
    for _ in range(episode_steps):
        if all(environment.done for environment in environments):
            break
        if any(environment.done for environment in environments):
            raise RuntimeError("synchronous environments terminated at different horizons")
        policy_step = act_batch(
            actor,
            critic,
            _observations(states),
            _opponent_privates(states),
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
        )
        _record_policy_step(fields, policy_step)
        entropies.append(policy_step.factors.entropy)

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
            next_potential = pair_potential(next_state[0].observation, next_state[1].observation)
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
        fields,
        episode_seeds=np.repeat(
            np.arange(seed_start, seed_start + games, dtype=np.int64), repeats=2
        ),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=np.tile(np.asarray([0, 1], dtype=np.int8), games),
        entropies=entropies,
        started=started,
    )


def collect_frozen_opponent_play(
    actor: FarmActor,
    critic: DistributionalCritic,
    opponent: FarmActor,
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
    critic.eval()
    opponent.eval()
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
    fields = _new_fields()
    final_money = np.zeros(games, dtype=np.float32)
    opponent_money = np.zeros(games, dtype=np.float32)
    entropies = []
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
            critic,
            current_observations,
            [observation["private"] for observation in frozen_observations],
            deterministic=deterministic,
            temperature=temperature,
            generator=current_generator,
        )
        frozen_step = act_batch(
            opponent,
            None,
            frozen_observations,
            deterministic=deterministic_opponent,
            temperature=opponent_temperature,
            generator=opponent_generator,
        )
        _record_policy_step(fields, current_step)
        entropies.append(current_step.factors.entropy)

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
            next_potential = pair_potential(next_state[0].observation, next_state[1].observation)
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
        fields,
        episode_seeds=np.arange(seed_start, seed_start + games, dtype=np.int64),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        entropies=entropies,
        started=started,
    )


_TRAJECTORY_METADATA_FIELDS = ("episode_seeds", "final_money", "opponent_money", "seats")


def _combined_rollout_metadata(batches: list[RolloutBatch]) -> dict[str, Any]:
    """Combine per-trajectory metadata and component-weighted entropy."""
    component_counts = [
        int(
            batch.unit_active.sum() + batch.market_active.sum() + batch.market_quantity_active.sum()
        )
        for batch in batches
    ]
    mean_entropy = sum(
        batch.mean_entropy * count for batch, count in zip(batches, component_counts, strict=True)
    ) / max(1, sum(component_counts))
    combined: dict[str, Any] = {
        field: np.concatenate([getattr(batch, field) for batch in batches], axis=0)
        for field in _TRAJECTORY_METADATA_FIELDS
    }
    combined["mean_entropy"] = mean_entropy
    combined["elapsed_seconds"] = sum(batch.elapsed_seconds for batch in batches)
    return combined


def concatenate_rollouts(batches: list[RolloutBatch]) -> RolloutBatch:
    """Concatenate compatible current-policy trajectories from several match sources."""
    if not batches:
        raise ValueError("at least one rollout batch is required")
    if len(batches) == 1:
        return batches[0]
    if len({batch.horizon for batch in batches}) != 1:
        raise ValueError("rollout horizons must match")
    state_fields = tuple(_native_field_specs(1, 1))
    combined = {
        field: np.concatenate([getattr(batch, field) for batch in batches], axis=0)
        for field in state_fields
    }
    return RolloutBatch(**combined, **_combined_rollout_metadata(batches))


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
    rows = sum(batch.trajectories for batch in batches)
    state_fields = tuple(_native_field_specs(1, 1))
    offset = 0
    for batch in batches:
        for field in state_fields:
            expected = storage[field][offset : offset + batch.trajectories]
            actual = getattr(batch, field)
            if (
                actual.shape != expected.shape
                or actual.__array_interface__["data"][0] != expected.__array_interface__["data"][0]
            ):
                raise ValueError("rollout batches are not adjacent views of the storage arena")
        offset += batch.trajectories
    if any(storage[field].shape[0] != rows for field in state_fields):
        raise ValueError("storage arena rows do not match the combined batches")
    return RolloutBatch(
        **{field: storage[field] for field in state_fields},
        **_combined_rollout_metadata(batches),
    )
