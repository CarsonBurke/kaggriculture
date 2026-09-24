"""Batched masked action sampling shared by rollout collection and submission inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    QUANTIFIED_MARKET_KINDS,
    MarketKind,
    MarketLedger,
    UnitAction,
    _apply_ledger_order,
    _ledger_kind_mask,
    _ledger_quantity_mask,
    apply_unit_shed_effect,
    apply_unit_tile_effect,
    compile_action,
    copy_tile_grid,
    unit_action_mask,
)
from kaggriculture.causal_actor import CausalActor, CausalChoice, CausalOutput
from kaggriculture.constants import (
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    QUANTITY_BINS,
)
from kaggriculture.device_ledger import get_device_ledger, pack_observations, validate_packed
from kaggriculture.encoding import EncodedObservation, encode_observation
from kaggriculture.entity import EntityActor
from kaggriculture.market_set import (
    MARKET_SET_MAX_VALUE,
    N_MARKET_SET_KINDS,
    MarketSetOrder,
    compile_market_set,
    marginalize_market_set_all,
    market_set_value_mask,
)
from kaggriculture.model import ActorOutput, FarmActor
from kaggriculture.orientation import (
    Orientation,
    flip_board,
    movement_permutation,
    orient_unit_features,
    orient_unit_masks,
)
from kaggriculture.registry import architecture_of
from kaggriculture.strategic_actor import PlanChoice, StrategicActor, StrategicOutput
from kaggriculture.structured import StructuredActor, stack_structured
from kaggriculture.tokens import StructuredObservation, encode_structured_observation


@dataclass(frozen=True)
class TensorObservationBatch:
    board: Tensor
    global_features: Tensor
    critic_features: Tensor
    units: Tensor
    unit_positions: Tensor


@dataclass(frozen=True)
class ActionFactors:
    # Unit movement indices and mask columns live in the orientation space the
    # actor stepped in -- identity unless `act_batch` was given another one --
    # so a replay that re-forwards the stored features gathers likelihoods at
    # these indices directly. Market factors are always real-space.
    unit_actions: np.ndarray
    market_kinds: np.ndarray
    market_quantities: np.ndarray
    unit_masks: np.ndarray
    market_kind_masks: np.ndarray
    market_quantity_masks: np.ndarray
    unit_active: np.ndarray
    market_active: np.ndarray
    market_quantity_active: np.ndarray
    unit_logprobs: np.ndarray
    market_kind_logprobs: np.ndarray
    market_quantity_logprobs: np.ndarray
    # Behavior entropy summed over each row's active components. Keeping the
    # per-row sums lets any trajectory subset recover its exact mean entropy.
    entropy_sums: np.ndarray
    plan: np.ndarray | None = None
    policy_ledger: np.ndarray | None = None
    market_set_values: np.ndarray | None = None
    market_set_masks: np.ndarray | None = None
    market_set_active: np.ndarray | None = None
    market_set_logprobs: np.ndarray | None = None


@dataclass(frozen=True)
class PolicyStep:
    actions: list[dict[str, Any]]
    # Per-architecture encodings: EncodedObservation for the convolutional
    # family, StructuredObservation for the structured transformer.
    encoded: list[EncodedObservation] | list[StructuredObservation]
    factors: ActionFactors


@dataclass(frozen=True)
class PreparedQuantityHeads:
    """Immutable CPU quantity parameters for a frozen inference actor."""

    kind_gate: np.ndarray
    values: np.ndarray
    bias: np.ndarray


def prepare_quantity_heads(
    actor: FarmActor | StructuredActor | EntityActor,
) -> PreparedQuantityHeads:
    """Materialize quantity parameters once for repeated frozen-policy actions."""

    def frozen(parameter: Tensor) -> np.ndarray:
        array = parameter.detach().float().cpu().numpy().copy()
        array.setflags(write=False)
        return array

    return PreparedQuantityHeads(
        kind_gate=frozen(actor.market_quantity_kind_gate.weight),
        values=frozen(actor.market_quantity_value.weight),
        bias=frozen(actor.market_quantity_bias),
    )


def percentage_quantity_logits_numpy(parameters: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Reference FP32 integer logits for the percentage quantity interface."""
    if parameters.shape[-1] != 7 or masks.shape != (*parameters.shape[:-1], N_QUANTITIES):
        raise ValueError("percentage parameters and quantity masks must align")
    maximum = np.maximum(masks.sum(axis=-1), 1).astype(np.int64)
    quantity = np.arange(1, N_QUANTITIES + 1, dtype=np.float32)
    scale = np.logaddexp(np.float32(0), parameters[..., 6:7]) + np.float32(0.02)
    location = np.float32(1) / (np.float32(1) + np.exp(-parameters[..., 5:6]))
    upper = (quantity / maximum[..., None] - location) / scale
    lower = ((quantity - 1) / maximum[..., None] - location) / scale
    delta = np.float32(1) / (maximum[..., None] * scale)

    def log_sigmoid(value: np.ndarray) -> np.ndarray:
        return -np.logaddexp(np.float32(0), -value)

    log_mass = log_sigmoid(upper) + log_sigmoid(-lower) + np.log(-np.expm1(-delta))
    high = (np.float32(1) - location) / scale
    low = -location / scale
    log_total = (
        log_sigmoid(high)
        + log_sigmoid(-low)
        + np.log(-np.expm1(-np.float32(1) / scale))
    )
    result = parameters[..., 4:5] + log_mass - log_total
    for atom, amount in enumerate((1, 2, 3, 0)):
        destinations = maximum if atom == 3 else np.full_like(maximum, amount)
        active = masks.any(-1) if atom == 3 else maximum >= amount
        indices = (destinations - 1).clip(0, N_QUANTITIES - 1)
        row = np.arange(indices.size)
        flat = result.reshape(-1, N_QUANTITIES)
        old = flat[row, indices.reshape(-1)]
        flat[row, indices.reshape(-1)] = np.where(
            active.reshape(-1),
            np.logaddexp(old, parameters[..., atom].reshape(-1)),
            old,
        )
    return result


def stack_encoded(
    encoded: list[EncodedObservation], device: torch.device
) -> TensorObservationBatch:
    return TensorObservationBatch(
        board=torch.as_tensor(
            np.stack([row.board for row in encoded]), device=device, dtype=torch.float32
        ),
        global_features=torch.as_tensor(
            np.stack([row.global_features for row in encoded]),
            device=device,
            dtype=torch.float32,
        ),
        critic_features=torch.as_tensor(
            np.stack([row.critic_features for row in encoded]),
            device=device,
            dtype=torch.float32,
        ),
        units=torch.as_tensor(
            np.stack([row.units for row in encoded]), device=device, dtype=torch.float32
        ),
        unit_positions=torch.as_tensor(
            np.stack([row.unit_positions for row in encoded]),
            device=device,
            dtype=torch.long,
        ),
    )


def mask_logits(logits: Tensor, mask: Tensor, *, validate: bool = True) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} != {mask.shape}")
    if validate and not bool(mask.any(dim=-1).all()):
        raise ValueError("every categorical decision needs at least one valid action")
    return logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)


def greedy_disagreement(first: Tensor, second: Tensor, mask: Tensor, active: Tensor) -> float:
    """Share of active decisions where two policies' greedy actions differ.

    The operational question behind a farming program: money is earned by taking
    one particular action at each step, so the fraction of decisions that differ
    reads "are these two the same program" more directly than any distance
    between distributions, and unlike a KL it cannot be moved by the tails.

    Masking before the argmax matters: an illegal action can hold the largest raw
    logit, and two policies that would never take it must not be recorded as
    disagreeing about it.
    """
    if not bool(active.any()):
        return float("nan")
    chosen = mask_logits(first, mask, validate=False).argmax(dim=-1)
    other = mask_logits(second, mask, validate=False).argmax(dim=-1)
    return float((chosen != other)[active.bool()].float().mean())


def population_disagreement(unit_logits: Sequence[Tensor], mask: Tensor, active: Tensor) -> Tensor:
    """The full N x N greedy-disagreement matrix over one fixed batch of states.

    Symmetric with a zero diagonal, so the population's diversity is the mean of
    the off-diagonal entries. A population that has converged into mirror play
    under another name shows it here and nowhere else: every reward stays 0.5 and
    every other metric reads healthy.

    All members are scored on the SAME states, which is what makes the entries
    comparable -- scoring each on its own visited states would confound flattened
    weights with a moved state distribution.
    """
    count = len(unit_logits)
    if count < 2:
        raise ValueError("a disagreement matrix needs at least two policies")
    matrix = torch.zeros((count, count), dtype=torch.float64)
    for i in range(count):
        for j in range(i + 1, count):
            value = greedy_disagreement(unit_logits[i], unit_logits[j], mask, active)
            matrix[i, j] = matrix[j, i] = value
    return matrix


def mean_off_diagonal(matrix: Tensor) -> float:
    """Mean of a symmetric matrix's off-diagonal entries, its diagonal being zero."""
    count = matrix.shape[0]
    if count < 2:
        raise ValueError("an off-diagonal mean needs at least two rows")
    return float(matrix.sum() / (count * (count - 1)))


def categorical_statistics(
    logits: Tensor,
    mask: Tensor,
    actions: Tensor,
    *,
    validate_mask: bool = True,
) -> tuple[Tensor, Tensor]:
    masked = mask_logits(logits, mask, validate=validate_mask)
    log_probabilities = masked.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    selected = log_probabilities.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    return selected, entropy


def categorical_logprob(
    logits: Tensor,
    mask: Tensor,
    actions: Tensor,
    *,
    validate_mask: bool = True,
) -> Tensor:
    masked = mask_logits(logits, mask, validate=validate_mask)
    log_probabilities = masked.log_softmax(dim=-1)
    return log_probabilities.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


def _sample_numpy_categorical(
    logits: np.ndarray,
    mask: np.ndarray,
    deterministic: bool,
    temperature: float,
    generator: np.random.Generator,
    *,
    draws: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if logits.shape != mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} != {mask.shape}")
    if not mask.any(axis=-1).all():
        raise ValueError("every categorical decision needs at least one valid action")
    masked = np.where(mask, logits / max(temperature, 1e-4), -np.inf)
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    positive_mass = weights > 0
    if deterministic:
        total = weights.sum(axis=-1, keepdims=True, dtype=np.float64)
        actions = masked.argmax(axis=-1)
    else:
        cumulative = np.cumsum(weights, axis=-1, dtype=np.float64)
        total = cumulative[:, -1:]
        if draws is None:
            draws = generator.random((weights.shape[0], 1))
        elif draws.shape != (weights.shape[0], 1):
            raise ValueError(
                f"categorical draw shape mismatch: {draws.shape} != {(weights.shape[0], 1)}"
            )
        intervals = positive_mass & (draws * total < cumulative)
        last_positive = weights.shape[-1] - 1 - positive_mass[:, ::-1].argmax(axis=-1)
        actions = np.where(intervals.any(axis=-1), intervals.argmax(axis=-1), last_positive)
    log_total = np.log(total[:, 0])
    logprobs = shifted[np.arange(weights.shape[0]), actions] - log_total
    # Zero-mass entries contribute zero even when their masked logit is -inf.
    np.multiply(weights, shifted, out=weights, where=positive_mass)
    entropy = log_total - weights.sum(axis=-1, dtype=np.float64) / total[:, 0]
    return actions, logprobs.astype(np.float32), entropy.astype(np.float32)


def component_logprobs(
    output: ActorOutput,
    market_quantity_logits: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    market_kind_masks: Tensor,
    market_quantity_masks: Tensor,
    *,
    validate_masks: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    unit_logprob, unit_entropy = categorical_statistics(
        output.unit_logits,
        unit_masks,
        unit_actions,
        validate_mask=validate_masks,
    )
    kind_logprob, kind_entropy = categorical_statistics(
        output.market_kind_logits,
        market_kind_masks,
        market_kinds,
        validate_mask=validate_masks,
    )
    quantity_logprob, quantity_entropy = categorical_statistics(
        market_quantity_logits,
        market_quantity_masks,
        market_quantities,
        validate_mask=validate_masks,
    )
    return (
        unit_logprob,
        kind_logprob,
        quantity_logprob,
        unit_entropy,
        kind_entropy,
        quantity_entropy,
    )


def component_selected_logprobs(
    output: ActorOutput,
    market_quantity_logits: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    market_kind_masks: Tensor,
    market_quantity_masks: Tensor,
    *,
    validate_masks: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Selected-action log-likelihoods only, skipping the entropy reductions.

    The behavior replay and the parity audit gather one likelihood per
    component for every valid state and discard entropy, so the entropy
    branch of `component_logprobs` would spend a softmax-sized reduction per
    head on the full batch for nothing.
    """
    return (
        categorical_logprob(
            output.unit_logits, unit_masks, unit_actions, validate_mask=validate_masks
        ),
        categorical_logprob(
            output.market_kind_logits, market_kind_masks, market_kinds, validate_mask=validate_masks
        ),
        categorical_logprob(
            market_quantity_logits,
            market_quantity_masks,
            market_quantities,
            validate_mask=validate_masks,
        ),
    )


def _causal_policy_step(
    output: CausalOutput,
    observations: list[dict[str, Any]],
    encoded: list[StructuredObservation],
    packed_ledger: np.ndarray,
) -> PolicyStep:
    """Compile the already selected prefix; never resample its conditional logits."""
    names = output._fields[3:]
    tensors = output[3:]
    flat = torch.cat([tensor.flatten().float() for tensor in tensors]).cpu().numpy()
    values = {}
    cursor = 0
    for name, tensor in zip(names, tensors, strict=True):
        end = cursor + tensor.numel()
        dtype = (
            np.bool_
            if tensor.dtype == torch.bool
            else np.int64
            if not tensor.is_floating_point()
            else np.float32
        )
        values[name] = flat[cursor:end].reshape(tensor.shape).astype(dtype, copy=False)
        cursor = end
    logprobs = values.pop("factor_logprobs")
    entropies = values.pop("factor_entropies")
    factors = ActionFactors(
        **values,
        unit_logprobs=logprobs[:, :16],
        market_kind_logprobs=logprobs[:, 16::2],
        market_quantity_logprobs=logprobs[:, 17::2],
        entropy_sums=entropies.sum(axis=1, dtype=np.float64),
        policy_ledger=packed_ledger,
    )
    actions = [
        compile_action(observation, units, kinds, quantities)
        for observation, units, kinds, quantities in zip(
            observations,
            factors.unit_actions,
            factors.market_kinds,
            factors.market_quantities,
            strict=True,
        )
    ]
    return PolicyStep(actions=actions, encoded=encoded, factors=factors)


@torch.inference_mode()
def _market_set_policy_step(
    actor: EntityActor,
    observations: list[dict[str, Any]],
    encoded: list[StructuredObservation],
    unit_actions: np.ndarray,
    real_unit_actions: np.ndarray,
    unit_masks: np.ndarray,
    unit_active: np.ndarray,
    unit_logprobs: np.ndarray,
    unit_entropies: np.ndarray,
    post_unit_sheds: list[dict[str, int]],
    context: np.ndarray,
    heads: PreparedQuantityHeads,
    deterministic: bool,
    temperature: float,
    generator: np.random.Generator,
) -> PolicyStep:
    """Sample the opt-in per-kind market set after the unit-phase ledger."""
    batch_size = len(observations)
    order = MarketSetOrder(actor.config.market_set_sell_order, actor.config.market_set_hire_last)
    kinds = order.decision_kinds
    if context.shape[:2] != (batch_size, N_MARKET_SET_KINDS):
        raise ValueError("market-set context does not have one row per kind")
    choices = np.zeros((batch_size, N_MARKET_SET_KINDS), dtype=np.uint8)
    masks = np.zeros((batch_size, N_MARKET_SET_KINDS, MARKET_SET_MAX_VALUE + 1), dtype=np.bool_)
    active = np.zeros((batch_size, N_MARKET_SET_KINDS), dtype=np.bool_)
    logprobs = np.zeros((batch_size, N_MARKET_SET_KINDS), dtype=np.float32)
    entropies = np.zeros((batch_size, N_MARKET_SET_KINDS), dtype=np.float32)
    ledgers = [
        MarketLedger.from_observation(observation, shed=dict(post_unit_sheds[row]))
        for row, observation in enumerate(observations)
    ]
    used_slots = np.zeros(batch_size, dtype=np.uint8)
    for index, kind in enumerate(kinds):
        for row, observation in enumerate(observations):
            mask = market_set_value_mask(observation, kind, ledgers[row], int(used_slots[row]))
            masks[row, index] = mask
            active[row, index] = bool(mask[1:].any())
        features = context[:, index] * (1.0 + heads.kind_gate[int(kind)])
        raw_logits = features @ heads.values.T + heads.bias[int(kind)]
        effective_logits = np.stack(
            [
                marginalize_market_set_all(raw_logits[row], masks[row, index])
                for row in range(batch_size)
            ]
        )
        sampled, selected_logprobs, selected_entropies = _sample_numpy_categorical(
            effective_logits, masks[:, index], deterministic, temperature, generator
        )
        choices[:, index] = sampled
        logprobs[:, index] = selected_logprobs
        entropies[:, index] = selected_entropies
        for row, value in enumerate(sampled):
            if not value:
                continue
            if kind == MarketKind.HIRE:
                for _ in range(int(value)):
                    _apply_ledger_order(observations[row], kind, 1, ledgers[row])
                used_slots[row] += int(value)
            else:
                _apply_ledger_order(observations[row], kind, int(value), ledgers[row])
                used_slots[row] += 1

    market_kinds = np.full((batch_size, MAX_MARKET_ORDERS), int(MarketKind.STOP), dtype=np.int64)
    market_quantities = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.int64)
    kind_masks = np.zeros((batch_size, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.bool_)
    quantity_masks = np.zeros((batch_size, MAX_MARKET_ORDERS, N_QUANTITIES), dtype=np.bool_)
    market_active = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.bool_)
    quantity_active = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.bool_)
    actions = []
    for row, observation in enumerate(observations):
        compiled, factors = compile_market_set(
            observation,
            choices[row],
            order=order,
            post_unit_shed=post_unit_sheds[row],
        )
        if not np.array_equal(factors.masks, masks[row]):
            raise ValueError("market-set sampler and compiler legality masks differ")
        for slot, raw in enumerate(compiled):
            opcode = str(raw[0])
            kind = MarketKind[opcode if len(raw) == 1 else f"{opcode}_{raw[1]}"]
            market_kinds[row, slot] = int(kind)
            market_active[row, slot] = True
            kind_masks[row, slot, int(kind)] = True
            quantity = int(raw[2]) - 1 if len(raw) > 2 else 0
            market_quantities[row, slot] = quantity
            quantity_masks[row, slot, quantity] = True
            quantity_active[row, slot] = kind in QUANTIFIED_MARKET_KINDS
        kind_masks[row, len(compiled) :, MarketKind.STOP] = True
        quantity_masks[row, len(compiled) :, 0] = True
        actions.append(
            compile_action(
                observation, real_unit_actions[row], market_kinds[row], market_quantities[row]
            )
        )
    entropy_sums = (
        (unit_entropies * unit_active).sum(axis=1) + (entropies * active).sum(axis=1)
    ).astype(np.float64)
    factors = ActionFactors(
        unit_actions=unit_actions,
        market_kinds=market_kinds,
        market_quantities=market_quantities,
        unit_masks=unit_masks,
        market_kind_masks=kind_masks,
        market_quantity_masks=quantity_masks,
        unit_active=unit_active,
        market_active=market_active,
        market_quantity_active=quantity_active,
        unit_logprobs=unit_logprobs,
        market_kind_logprobs=np.zeros_like(market_kinds, dtype=np.float32),
        market_quantity_logprobs=np.zeros_like(market_quantities, dtype=np.float32),
        entropy_sums=entropy_sums,
        market_set_values=choices,
        market_set_masks=masks,
        market_set_active=active,
        market_set_logprobs=logprobs,
    )
    return PolicyStep(actions=actions, encoded=encoded, factors=factors)


@torch.inference_mode()
def act_batch(
    actor: FarmActor | StructuredActor | EntityActor,
    observations: list[dict[str, Any]],
    opponent_privates: list[dict[str, Any] | None] | None = None,
    *,
    deterministic: bool = False,
    temperature: float = 1.0,
    generator: np.random.Generator | None = None,
    orientation: Orientation = Orientation.IDENTITY,
    quantity_heads: PreparedQuantityHeads | None = None,
) -> PolicyStep:
    """Encode, sample, mask, and compile a batch of decentralized actions.

    Under a non-identity ``orientation`` every row's encoded board and unit
    geometry is flipped before the forward pass, so the actor sees the world
    rendered through that grid symmetry and its movement head scores oriented
    actions. Movement masks are restriped into those oriented columns and the
    sampled indices stay in oriented space -- the factor arrays then match the
    stored features a PPO replay re-forwards -- while each chosen movement is
    mapped back to the real action it executes before engine-facing actions,
    ledger evolution, and per-unit effects are compiled. Market heads carry no
    spatial semantics and are never permuted.
    """
    if not observations:
        raise ValueError("act_batch requires at least one observation")
    if opponent_privates is None:
        opponent_privates = [None] * len(observations)
    if len(opponent_privates) != len(observations):
        raise ValueError("opponent private-state count must match observations")
    device = next(actor.parameters()).device
    generator = generator or np.random.default_rng()
    if architecture_of(actor).structured_inputs:
        if orientation is not Orientation.IDENTITY:
            raise NotImplementedError(
                "orientations flip board surfaces, so only the convolutional "
                "actor plays under a non-identity orientation"
            )
        encoded = [
            encode_structured_observation(observation, opponent_private)
            for observation, opponent_private in zip(observations, opponent_privates, strict=True)
        ]
        inputs, _ = stack_structured(encoded, device=device)
        if isinstance(actor, CausalActor):
            if actor.device_ledger is None:
                actor.set_device_ledger(get_device_ledger(device))
            packed_ledger = pack_observations(observations)
            validate_packed(packed_ledger, actor.device_ledger.minimum, actor.device_ledger.maximum)
            batch = len(observations)
            choice = CausalChoice(
                torch.as_tensor(packed_ledger, device=device),
                torch.full((batch, MAX_UNITS), -1, dtype=torch.long, device=device),
                torch.full((batch, MAX_MARKET_ORDERS), -1, dtype=torch.long, device=device),
                torch.full((batch, MAX_MARKET_ORDERS), -1, dtype=torch.long, device=device),
                torch.as_tensor(generator.random((batch, 36), dtype=np.float32), device=device),
                torch.full((batch,), max(temperature, 1e-4), device=device),
                torch.full((batch,), deterministic, dtype=torch.bool, device=device),
            )
            output = actor(inputs, choice)
            return _causal_policy_step(output, observations, encoded, packed_ledger)
        if isinstance(actor, StrategicActor):
            batch = len(observations)
            choice = PlanChoice(
                torch.full((batch,), -1, dtype=torch.long, device=device),
                torch.as_tensor(generator.random(batch), dtype=torch.float32, device=device),
                torch.full((batch,), max(temperature, 1e-4), device=device),
                torch.full((batch,), deterministic, dtype=torch.bool, device=device),
                torch.zeros(batch, device=device),
                torch.ones(batch, dtype=torch.bool, device=device),
            )
            output = actor(inputs, choice)
        else:
            output = actor(inputs)
    else:
        encoded = [
            encode_observation(observation, opponent_private)
            for observation, opponent_private in zip(observations, opponent_privates, strict=True)
        ]
        if orientation is not Orientation.IDENTITY:
            # Orienting the rows themselves keeps PolicyStep.encoded holding
            # exactly the features this forward consumed, which is what makes
            # a replay of the stored factors on-policy.
            row_codes = np.full(1, int(orientation), dtype=np.int8)
            for row in encoded:
                flip_board(row.board, orientation)
                orient_unit_features(
                    row.units[np.newaxis], row.unit_positions[np.newaxis], row_codes
                )
        tensors = stack_encoded(encoded, device)
        output = actor(
            tensors.board, tensors.global_features, tensors.units, tensors.unit_positions
        )
    unit_logits = output.unit_logits.float().cpu().numpy()
    market_kind_logits = output.market_kind_logits.float().cpu().numpy()
    market_quantity_context = output.market_quantity_context.float().cpu().numpy()
    if quantity_heads is None:
        quantity_heads = prepare_quantity_heads(actor)
    quantity_kind_gate = quantity_heads.kind_gate
    quantity_values = quantity_heads.values
    quantity_bias = quantity_heads.bias
    batch_size = len(observations)
    movement_map = movement_permutation(orientation)

    unit_actions = np.zeros((batch_size, MAX_UNITS), dtype=np.int64)
    unit_masks = np.zeros((batch_size, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.bool_)
    real_unit_actions = np.zeros((batch_size, MAX_UNITS), dtype=np.int64)
    unit_active = np.stack([row.unit_active for row in encoded])
    remaining_seeds = [
        dict((observation.get("private") or {}).get("seeds") or {}) for observation in observations
    ]
    remaining_unit_sheds = [
        dict((observation.get("private") or {}).get("shed") or {}) for observation in observations
    ]
    unit_tiles = [
        copy_tile_grid(
            (observation.get("farms") or [])[int(observation.get("player", 0) or 0)].get("tiles")
            or []
        )
        for observation in observations
    ]
    unit_logprobs = np.zeros((batch_size, MAX_UNITS), dtype=np.float32)
    unit_entropies = np.zeros((batch_size, MAX_UNITS), dtype=np.float32)
    for unit_index in range(MAX_UNITS):
        for row, observation in enumerate(observations):
            if unit_active[row, unit_index]:
                unit_masks[row, unit_index] = unit_action_mask(
                    observation,
                    unit_index,
                    remaining_seeds[row],
                    remaining_unit_sheds[row],
                    unit_tiles[row],
                )
            else:
                unit_masks[row, unit_index, UnitAction.PASS] = True
        if orientation is Orientation.IDENTITY:
            oriented_masks = unit_masks[:, unit_index]
        else:
            # The restripe orient_unit_masks performs on a batched array, done
            # on one unit slice: oriented column j holds real column perm[j].
            oriented_masks = unit_masks[:, unit_index][:, movement_map]
        sampled_cpu, logprob, entropy = _sample_numpy_categorical(
            unit_logits[:, unit_index],
            oriented_masks,
            deterministic,
            temperature,
            generator,
        )
        # `sampled_cpu` is the oriented index the flipped forward scored; the
        # engine and every mask-evolving side effect consume its real image.
        unit_actions[:, unit_index] = sampled_cpu
        real_unit_actions[:, unit_index] = movement_map[sampled_cpu]
        unit_logprobs[:, unit_index] = logprob
        unit_entropies[:, unit_index] = entropy
        for row, raw_action in enumerate(real_unit_actions[:, unit_index]):
            if UnitAction.PLANT_WHEAT <= raw_action <= UnitAction.PLANT_MELON:
                crop = CROPS[int(raw_action) - int(UnitAction.PLANT_WHEAT)]
                remaining_seeds[row][crop] = remaining_seeds[row].get(crop, 0) - 1
            apply_unit_shed_effect(
                observations[row],
                unit_index,
                int(raw_action),
                remaining_unit_sheds[row],
                unit_tiles[row],
            )
            apply_unit_tile_effect(observations[row], unit_index, int(raw_action), unit_tiles[row])

    if isinstance(actor, EntityActor) and actor.config.action_interface == 3:
        return _market_set_policy_step(
            actor,
            observations,
            encoded,
            unit_actions,
            real_unit_actions,
            unit_masks,
            unit_active,
            unit_logprobs,
            unit_entropies,
            remaining_unit_sheds,
            market_quantity_context,
            quantity_heads,
            deterministic,
            temperature,
            generator,
        )

    market_kinds = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.int64)
    market_quantities = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.int64)
    kind_masks = np.zeros((batch_size, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.bool_)
    quantity_masks = np.zeros((batch_size, MAX_MARKET_ORDERS, N_QUANTITIES), dtype=np.bool_)
    market_active = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.bool_)
    quantity_active = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.bool_)
    kind_logprobs = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    quantity_logprobs = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    kind_entropies = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    quantity_entropies = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    still_active = np.ones(batch_size, dtype=np.bool_)
    ledgers = [
        MarketLedger.from_observation(observation, shed=dict(remaining_unit_sheds[row]))
        for row, observation in enumerate(observations)
    ]

    for slot in range(MAX_MARKET_ORDERS):
        for row, observation in enumerate(observations):
            if still_active[row]:
                kind_masks[row, slot] = _ledger_kind_mask(observation, ledgers[row])
                market_active[row, slot] = True
            else:
                kind_masks[row, slot, MarketKind.STOP] = True
        sampled_cpu, logprob, entropy = _sample_numpy_categorical(
            market_kind_logits[:, slot],
            kind_masks[:, slot],
            deterministic,
            temperature,
            generator,
        )
        market_kinds[:, slot] = sampled_cpu
        kind_logprobs[:, slot] = logprob
        kind_entropies[:, slot] = entropy

        for row, (observation, raw_kind) in enumerate(zip(observations, sampled_cpu, strict=True)):
            kind = MarketKind(int(raw_kind))
            if not still_active[row] or kind == MarketKind.STOP:
                quantity_masks[row, slot, 0] = True
                still_active[row] = False
                continue
            quantity_masks[row, slot] = _ledger_quantity_mask(observation, kind, ledgers[row])
            quantity_active[row, slot] = kind in QUANTIFIED_MARKET_KINDS
        sampled_quantity_cpu = np.zeros(batch_size, dtype=np.int64)
        logprob = np.zeros(batch_size, dtype=np.float32)
        entropy = np.zeros(batch_size, dtype=np.float32)
        active_rows = np.flatnonzero(quantity_active[:, slot])
        # Preserve the RNG stream and each row's draw while avoiding the exact
        # quantity GEMM for STOP, HIRE, BUY_LAND, and already-stopped rows. At
        # the sparse initialization policy this skips nearly all B*10*R*100
        # CPU work without changing sampled behavior.
        quantity_draws = None if deterministic else generator.random((batch_size, 1))[active_rows]
        if active_rows.size:
            active_kinds = sampled_cpu[active_rows]
            quantity_features = market_quantity_context[active_rows, slot] * (
                1.0 + quantity_kind_gate[active_kinds]
            )
            slot_quantity_logits = (
                quantity_features @ quantity_values.T + quantity_bias[active_kinds]
            )
            if quantity_values.shape[0] == 7:
                slot_quantity_logits = percentage_quantity_logits_numpy(
                    slot_quantity_logits, quantity_masks[active_rows, slot]
                )
            elif quantity_values.shape[0] == N_QUANTITIES + 1:
                active_mask = quantity_masks[active_rows, slot]
                maximum = active_mask.sum(axis=-1).astype(np.int64) - 1
                if np.any(maximum < 0):
                    raise ValueError("quantified market rows need a legal quantity")
                rows = np.arange(active_rows.size)
                slot_quantity_logits[rows, maximum] = np.logaddexp(
                    slot_quantity_logits[rows, maximum],
                    slot_quantity_logits[:, N_QUANTITIES],
                )
                slot_quantity_logits = slot_quantity_logits[:, :N_QUANTITIES]
            active_quantities, active_logprobs, active_entropies = _sample_numpy_categorical(
                slot_quantity_logits,
                quantity_masks[active_rows, slot],
                deterministic,
                temperature,
                generator,
                draws=quantity_draws,
            )
            sampled_quantity_cpu[active_rows] = active_quantities
            logprob[active_rows] = active_logprobs
            entropy[active_rows] = active_entropies
        market_quantities[:, slot] = sampled_quantity_cpu
        quantity_logprobs[:, slot] = logprob
        quantity_entropies[:, slot] = entropy
        for row, (raw_kind, raw_quantity) in enumerate(
            zip(sampled_cpu, sampled_quantity_cpu, strict=True)
        ):
            if not market_active[row, slot] or raw_kind == MarketKind.STOP:
                continue
            _apply_ledger_order(
                observations[row],
                MarketKind(int(raw_kind)),
                QUANTITY_BINS[int(raw_quantity)],
                ledgers[row],
            )

    actions = [
        compile_action(observation, units, kinds, quantities)
        for observation, units, kinds, quantities in zip(
            observations,
            real_unit_actions,
            market_kinds,
            market_quantities,
            strict=True,
        )
    ]
    entropy_sums = (
        (unit_entropies * unit_active).sum(axis=1)
        + (kind_entropies * market_active).sum(axis=1)
        + (quantity_entropies * quantity_active).sum(axis=1)
    ).astype(np.float64)
    factors = ActionFactors(
        unit_actions=unit_actions,
        market_kinds=market_kinds,
        market_quantities=market_quantities,
        # The stored mask columns must describe the stored action indices:
        # under a non-identity orientation both live in the oriented space the
        # actor stepped in, so a replay re-forward of the stored features
        # scores the same categorical event the sampler drew.
        unit_masks=unit_masks
        if orientation is Orientation.IDENTITY
        else orient_unit_masks(unit_masks, np.full(batch_size, int(orientation), dtype=np.int8)),
        market_kind_masks=kind_masks,
        market_quantity_masks=quantity_masks,
        unit_active=unit_active,
        market_active=market_active,
        market_quantity_active=quantity_active,
        unit_logprobs=unit_logprobs,
        market_kind_logprobs=kind_logprobs,
        market_quantity_logprobs=quantity_logprobs,
        entropy_sums=entropy_sums
        + (output.plan[:, 2].float().cpu().numpy() if isinstance(output, StrategicOutput) else 0),
        plan=output.plan.float().cpu().numpy() if isinstance(output, StrategicOutput) else None,
    )
    return PolicyStep(actions=actions, encoded=encoded, factors=factors)
