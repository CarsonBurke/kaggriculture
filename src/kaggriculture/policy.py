"""Batched masked action sampling shared by rollout collection and submission inference."""

from __future__ import annotations

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
    UnitAction,
    apply_unit_shed_effect,
    apply_unit_tile_effect,
    compile_action,
    copy_tile_grid,
    market_kind_mask,
    quantity_mask,
    unit_action_mask,
)
from kaggriculture.constants import (
    ANIMAL_COST,
    CROPS,
    LAND_PRICES,
    MARKET_I0,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRICE_FLOOR,
    PRODUCTS,
    QUANTITY_BINS,
    SEED_COST,
    SHED_CAPACITY,
    fibonacci_hire_cost,
    market_price,
)
from kaggriculture.encoding import EncodedObservation, encode_observation
from kaggriculture.model import ActorOutput, DistributionalCritic, FarmActor

_MARKET_SEED_ITEMS = dict(
    zip(
        range(MarketKind.BUY_SEED_WHEAT, MarketKind.BUY_SEED_MELON + 1),
        CROPS,
        strict=True,
    )
)
_MARKET_PRODUCT_ITEMS = {
    MarketKind.BUY_PRODUCT_WHEAT: "WHEAT",
    MarketKind.BUY_PRODUCT_FERTILIZER: "FERTILIZER",
}
_MARKET_ANIMAL_ITEMS = {
    MarketKind.BUY_ANIMAL_GOOSE: "GOOSE",
    MarketKind.BUY_ANIMAL_COW: "COW",
    MarketKind.BUY_ANIMAL_SHEEP: "SHEEP",
}
_MARKET_SELL_ITEMS = dict(
    zip(range(MarketKind.SELL_WHEAT, MarketKind.SELL_FERTILIZER + 1), PRODUCTS, strict=True)
)


@dataclass(frozen=True)
class TensorObservationBatch:
    board: Tensor
    global_features: Tensor
    critic_features: Tensor
    units: Tensor
    unit_positions: Tensor


@dataclass(frozen=True)
class ActionFactors:
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
    values: np.ndarray
    entropy: float


@dataclass(frozen=True)
class PolicyStep:
    actions: list[dict[str, Any]]
    encoded: list[EncodedObservation]
    factors: ActionFactors


@dataclass
class MarketLedger:
    money: float
    shed: dict[str, int]
    hires: int
    extra_land: int
    inventory: dict[str, int]


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


def mask_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} != {mask.shape}")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every categorical decision needs at least one valid action")
    return logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)


def categorical_statistics(logits: Tensor, mask: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
    masked = mask_logits(logits, mask)
    log_probabilities = masked.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    selected = log_probabilities.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    return selected, entropy


def _sample_numpy_categorical(
    logits: np.ndarray,
    mask: np.ndarray,
    deterministic: bool,
    temperature: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if logits.shape != mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} != {mask.shape}")
    if not mask.any(axis=-1).all():
        raise ValueError("every categorical decision needs at least one valid action")
    masked = np.where(mask, logits / max(temperature, 1e-4), -np.inf)
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    if deterministic:
        actions = masked.argmax(axis=-1)
    else:
        cumulative = np.cumsum(probabilities, axis=-1)
        draws = generator.random((probabilities.shape[0], 1))
        actions = np.minimum((draws > cumulative).sum(axis=-1), probabilities.shape[1] - 1).astype(
            np.int64
        )
    selected = probabilities[np.arange(probabilities.shape[0]), actions]
    logprobs = np.log(np.maximum(selected, np.finfo(np.float32).tiny))
    entropy = -(probabilities * np.log(np.maximum(probabilities, np.finfo(np.float32).tiny))).sum(
        axis=-1
    )
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    unit_logprob, unit_entropy = categorical_statistics(
        output.unit_logits, unit_masks, unit_actions
    )
    kind_logprob, kind_entropy = categorical_statistics(
        output.market_kind_logits, market_kind_masks, market_kinds
    )
    quantity_logprob, quantity_entropy = categorical_statistics(
        market_quantity_logits, market_quantity_masks, market_quantities
    )
    return (
        unit_logprob,
        kind_logprob,
        quantity_logprob,
        unit_entropy,
        kind_entropy,
        quantity_entropy,
    )


def _ledger_kind_mask(
    observation: dict[str, Any],
    ledger: MarketLedger,
) -> np.ndarray:
    mask = market_kind_mask(observation)
    player = int(observation.get("player", 0) or 0)
    farm = (observation.get("farms") or [])[player]
    mask[MarketKind.HIRE] = len(farm.get("hands") or []) + max(
        0, ledger.hires - int(farm.get("hires_today", 0) or 0)
    ) < MAX_UNITS - 1 and ledger.money >= fibonacci_hire_cost(ledger.hires)
    mask[MarketKind.BUY_LAND] = (
        ledger.extra_land < len(LAND_PRICES) and ledger.money >= LAND_PRICES[ledger.extra_land]
    )
    room = SHED_CAPACITY - sum(ledger.shed.values())
    for raw_kind, crop in _MARKET_SEED_ITEMS.items():
        mask[raw_kind] = ledger.money >= SEED_COST[crop]
    market_params = (observation.get("market") or {}).get("params")
    for kind, item in _MARKET_PRODUCT_ITEMS.items():
        quote = market_price(item, ledger.inventory[item] - 1, market_params)
        mask[kind] = room > 0 and ledger.money >= quote
    for kind, animal in _MARKET_ANIMAL_ITEMS.items():
        mask[kind] = room > 0 and ledger.money >= ANIMAL_COST[animal]
    for kind, product in _MARKET_SELL_ITEMS.items():
        mask[kind] = ledger.shed.get(product, 0) > 0
    return mask


def _ledger_quantity_mask(
    observation: dict[str, Any],
    kind: MarketKind,
    ledger: MarketLedger,
) -> np.ndarray:
    if kind not in QUANTIFIED_MARKET_KINDS:
        return quantity_mask(observation, kind)
    room = SHED_CAPACITY - sum(ledger.shed.values())
    if kind in _MARKET_SELL_ITEMS:
        maximum = ledger.shed.get(_MARKET_SELL_ITEMS[kind], 0)
    elif kind in _MARKET_SEED_ITEMS:
        maximum = int(ledger.money // SEED_COST[_MARKET_SEED_ITEMS[kind]])
    elif kind in _MARKET_ANIMAL_ITEMS:
        maximum = min(room, int(ledger.money // ANIMAL_COST[_MARKET_ANIMAL_ITEMS[kind]]))
    else:
        item = _MARKET_PRODUCT_ITEMS[kind]
        market_params = (observation.get("market") or {}).get("params")
        balance = ledger.money
        inventory = ledger.inventory[item]
        maximum = 0
        while maximum < room:
            quote = market_price(item, inventory - 1, market_params)
            if balance < quote:
                break
            balance -= quote
            inventory -= 1
            maximum += 1
    return np.asarray([quantity <= maximum for quantity in QUANTITY_BINS], dtype=np.bool_)


def _apply_ledger_order(
    observation: dict[str, Any],
    kind: MarketKind,
    quantity: int,
    ledger: MarketLedger,
) -> None:
    market_params = (observation.get("market") or {}).get("params")
    if kind == MarketKind.HIRE:
        ledger.money -= fibonacci_hire_cost(ledger.hires)
        ledger.hires += 1
    elif kind == MarketKind.BUY_LAND:
        ledger.money -= LAND_PRICES[ledger.extra_land]
        ledger.extra_land += 1
    elif kind in _MARKET_SEED_ITEMS:
        ledger.money -= SEED_COST[_MARKET_SEED_ITEMS[kind]] * quantity
    elif kind in _MARKET_PRODUCT_ITEMS:
        item = _MARKET_PRODUCT_ITEMS[kind]
        for _ in range(quantity):
            quote = market_price(item, ledger.inventory[item] - 1, market_params)
            if ledger.money < quote or sum(ledger.shed.values()) >= SHED_CAPACITY:
                break
            ledger.money -= quote
            ledger.shed[item] = ledger.shed.get(item, 0) + 1
            ledger.inventory[item] -= 1
    elif kind in _MARKET_ANIMAL_ITEMS:
        animal = _MARKET_ANIMAL_ITEMS[kind]
        ledger.money -= ANIMAL_COST[animal] * quantity
        ledger.shed[animal] = ledger.shed.get(animal, 0) + quantity
    elif kind in _MARKET_SELL_ITEMS:
        item = _MARKET_SELL_ITEMS[kind]
        for _ in range(quantity):
            if ledger.shed.get(item, 0) <= 0:
                break
            quote = market_price(item, ledger.inventory[item], market_params)
            ledger.shed[item] -= 1
            ledger.money += quote
            if quote > PRICE_FLOOR:
                ledger.inventory[item] += 1


@torch.inference_mode()
def act_batch(
    actor: FarmActor,
    critic: DistributionalCritic | None,
    observations: list[dict[str, Any]],
    opponent_privates: list[dict[str, Any] | None] | None = None,
    *,
    deterministic: bool = False,
    temperature: float = 1.0,
    generator: np.random.Generator | None = None,
) -> PolicyStep:
    """Encode, sample, mask, and compile a batch of decentralized actions."""
    if not observations:
        raise ValueError("act_batch requires at least one observation")
    if opponent_privates is None:
        opponent_privates = [None] * len(observations)
    if len(opponent_privates) != len(observations):
        raise ValueError("opponent private-state count must match observations")
    device = next(actor.parameters()).device
    encoded = [
        encode_observation(observation, opponent_private)
        for observation, opponent_private in zip(observations, opponent_privates, strict=True)
    ]
    tensors = stack_encoded(encoded, device)
    output = actor(tensors.board, tensors.global_features, tensors.units, tensors.unit_positions)
    unit_logits = output.unit_logits.float().cpu().numpy()
    market_kind_logits = output.market_kind_logits.float().cpu().numpy()
    market_quantity_context = output.market_quantity_context.float().cpu().numpy()
    quantity_kind_gate = actor.market_quantity_kind_gate.weight.float().cpu().numpy()
    quantity_values = actor.market_quantity_value.weight.float().cpu().numpy()
    quantity_bias = actor.market_quantity_bias.float().cpu().numpy()
    batch_size = len(observations)
    generator = generator or np.random.default_rng()

    unit_actions = np.zeros((batch_size, MAX_UNITS), dtype=np.int64)
    unit_masks = np.zeros((batch_size, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.bool_)
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
        sampled_cpu, logprob, entropy = _sample_numpy_categorical(
            unit_logits[:, unit_index],
            unit_masks[:, unit_index],
            deterministic,
            temperature,
            generator,
        )
        unit_actions[:, unit_index] = sampled_cpu
        unit_logprobs[:, unit_index] = logprob
        unit_entropies[:, unit_index] = entropy
        for row, raw_action in enumerate(sampled_cpu):
            if UnitAction.PLANT_WHEAT <= raw_action <= UnitAction.PLANT_MELON:
                crop = CROPS[int(raw_action) - int(UnitAction.PLANT_WHEAT)]
                remaining_seeds[row][crop] = remaining_seeds[row].get(crop, 0) - 1
            apply_unit_shed_effect(
                observations[row], unit_index, int(raw_action), remaining_unit_sheds[row]
            )
            apply_unit_tile_effect(observations[row], unit_index, int(raw_action), unit_tiles[row])

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
    ledgers: list[MarketLedger] = []
    for row, observation in enumerate(observations):
        player = int(observation.get("player", 0) or 0)
        farm = (observation.get("farms") or [])[player]
        shed = dict(remaining_unit_sheds[row])
        market_inventory = (observation.get("market") or {}).get("inventory") or {}
        ledgers.append(
            MarketLedger(
                money=float(farm.get("money", 0) or 0),
                shed=shed,
                hires=int(farm.get("hires_today", 0) or 0),
                extra_land=max(0, len(farm.get("unlocked_quadrants") or []) - 1),
                inventory={item: int(market_inventory.get(item, MARKET_I0)) for item in PRODUCTS},
            )
        )

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
        quantity_features = market_quantity_context[:, slot] * (
            1.0 + quantity_kind_gate[sampled_cpu]
        )
        slot_quantity_logits = quantity_features @ quantity_values.T + quantity_bias[sampled_cpu]
        sampled_quantity_cpu, logprob, entropy = _sample_numpy_categorical(
            slot_quantity_logits,
            quantity_masks[:, slot],
            deterministic,
            temperature,
            generator,
        )
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

    values = np.zeros(batch_size, dtype=np.float32)
    if critic is not None:
        critic_logits = critic(tensors.board, tensors.critic_features)
        values = critic.value(critic_logits).cpu().numpy().astype(np.float32, copy=False)
    actions = [
        compile_action(observation, units, kinds, quantities)
        for observation, units, kinds, quantities in zip(
            observations,
            unit_actions,
            market_kinds,
            market_quantities,
            strict=True,
        )
    ]
    component_count = unit_active.sum() + market_active.sum() + quantity_active.sum()
    entropy_sum = (
        (unit_entropies * unit_active).sum()
        + (kind_entropies * market_active).sum()
        + (quantity_entropies * quantity_active).sum()
    )
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
        market_kind_logprobs=kind_logprobs,
        market_quantity_logprobs=quantity_logprobs,
        values=values,
        entropy=float(entropy_sum / max(1, component_count)),
    )
    return PolicyStep(actions=actions, encoded=encoded, factors=factors)
