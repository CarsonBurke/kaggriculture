"""Training-only typed transition predictor for the structured actor."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import torch
from torch import Tensor, nn

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import BOARD_SIZE, CROPS, MAX_MARKET_ORDERS, MAX_UNITS, PRODUCTS
from kaggriculture.latent_dynamics import (
    DecodeContext,
    DecodeMasks,
    latent_decode_kl_terms,
)
from kaggriculture.model import RMSNorm
from kaggriculture.structured import (
    Block,
    StructuredBelief,
    StructuredConfig,
    StructuredInputs,
)
from kaggriculture.tokens import TILE_COUNT


class StructuredActionEncoder(nn.Module):
    """Keep action slots and their entity identities separate."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.unit_action = nn.Embedding(N_UNIT_ACTIONS, width)
        self.unit_slot = nn.Embedding(MAX_UNITS, width)
        self.unit_row = nn.Embedding(BOARD_SIZE, width)
        self.unit_column = nn.Embedding(BOARD_SIZE, width)
        self.unit_active = nn.Embedding(2, width)
        self.market_kind = nn.Embedding(N_MARKET_KINDS, width)
        self.market_quantity = nn.Embedding(N_QUANTITIES, width)
        self.market_slot = nn.Embedding(MAX_MARKET_ORDERS, width)
        self.action_type = nn.Embedding(2, width)

    def forward(
        self,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        unit_categorical: Tensor,
        unit_active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        unit_slots = torch.arange(MAX_UNITS, device=unit_actions.device)
        market_slots = torch.arange(MAX_MARKET_ORDERS, device=unit_actions.device)
        unit_tokens = (
            self.unit_action(unit_actions)
            + self.unit_slot(unit_slots)
            + self.unit_row(unit_categorical[..., 2])
            + self.unit_column(unit_categorical[..., 3])
            + self.unit_active(unit_active.long())
            + self.action_type.weight[0]
        )
        market_tokens = (
            self.market_kind(market_kinds)
            + self.market_quantity(market_quantities)
            + self.market_slot(market_slots)
            + self.action_type.weight[1]
        )
        return unit_tokens, market_tokens


class StructuredDynamics(nn.Module):
    """One shared cross-attention transition over every typed belief family."""

    _TYPE_COUNT = 7

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        width = config.model_dim
        predictor_config = replace(config, zero_init_branches=False, global_modulation=False)
        self.width = width
        self.action = StructuredActionEncoder(width)
        self.type_identity = nn.Embedding(self._TYPE_COUNT, width)
        economy_tokens = len(PRODUCTS) + len(CROPS) + 2 + (2 if config.split_clock_token else 1)
        counts = (
            TILE_COUNT,
            TILE_COUNT,
            config.opponent_latents,
            economy_tokens,
            config.latents,
            MAX_UNITS,
            MAX_MARKET_ORDERS,
        )
        self.position_identity = nn.ModuleList(nn.Embedding(count, width) for count in counts)
        self.context_norm = RMSNorm(width)
        self.transition = Block(predictor_config)

    def _query(self, value: Tensor, kind: int) -> Tensor:
        tokens = value.shape[1]
        identity = self.position_identity[kind]
        assert isinstance(identity, nn.Embedding)
        if tokens > identity.num_embeddings:
            raise ValueError(
                f"belief type {kind} has more tokens than its configured identity table"
            )
        positions = identity.weight[:tokens]
        return value + positions + self.type_identity.weight[kind]

    def forward(
        self,
        belief: StructuredBelief,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        unit_categorical: Tensor,
        unit_active: Tensor,
    ) -> StructuredBelief:
        values = tuple(belief)
        queries = [self._query(value, kind) for kind, value in enumerate(values)]
        unit_action, market_action = self.action(
            unit_actions,
            market_kinds,
            market_quantities,
            unit_categorical,
            unit_active,
        )
        context = torch.cat((belief.central_latents, unit_action, market_action), dim=1)
        joined = torch.cat(queries, dim=1)
        transitioned = self.transition(joined, context, context_norm=self.context_norm)
        residual = transitioned - joined
        sizes = [value.shape[1] for value in values]
        deltas = residual.split(sizes, dim=1)
        return StructuredBelief(
            *(value + delta for value, delta in zip(values, deltas, strict=True))
        )


class StructuredDynamicsTerms(NamedTuple):
    decision: Tensor
    decision_one: Tensor
    decision_final: Tensor
    decision_unit: Tensor
    decision_market_kind: Tensor
    decision_market_quantity: Tensor
    patch: Tensor
    patch_one: Tensor
    patch_final: Tensor
    patch_all: Tensor
    patch_changed: Tensor
    patch_unchanged: Tensor
    economy: Tensor
    opponent_summary: Tensor
    opponent_patches: Tensor
    eligible: Tensor
    residual_ratio: Tensor
    residual_own_patches: Tensor
    residual_opponent_patches: Tensor
    residual_opponent_summary: Tensor
    residual_economy_entities: Tensor
    residual_central_latents: Tensor
    residual_unit_decisions: Tensor
    residual_market_decisions: Tensor


def _target_index(
    episode_index: Tensor,
    step: Tensor,
    offset: int,
) -> tuple[Tensor, Tensor]:
    rows = torch.arange(episode_index.shape[0], device=episode_index.device)
    unclamped = rows + offset
    index = unclamped.clamp_max(episode_index.shape[0] - 1)
    eligible = (
        (unclamped < episode_index.shape[0])
        & (episode_index[index] == episode_index)
        & (step[index] == step + offset)
    )
    return index, eligible


def _rms_normalize(value: Tensor) -> Tensor:
    return value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + 1e-6)


def _feature_l1(predicted: Tensor, target: Tensor, eligible: Tensor) -> Tensor:
    error = (_rms_normalize(predicted) - _rms_normalize(target.detach())).abs().mean(dim=-1)
    weight = eligible.float().unsqueeze(-1)
    return (error * weight).sum() / (weight.sum() * error.shape[1]).clamp_min(1.0)


def _eligible_rms_ratio(predicted: Tensor, previous: Tensor, eligible: Tensor) -> Tensor:
    weight = eligible.float().reshape(-1, *([1] * (predicted.ndim - 1)))
    elements = weight.sum() * predicted[0].numel()
    residual_rms = (
        ((predicted.float() - previous.float()).square() * weight).sum() / elements.clamp_min(1)
    ).sqrt()
    baseline_rms = ((previous.float().square() * weight).sum() / elements.clamp_min(1)).sqrt()
    return residual_rms / baseline_rms.clamp_min(1e-6)


def _patch_losses(
    predicted: Tensor,
    target: Tensor,
    eligible: Tensor,
    changed: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    error = (_rms_normalize(predicted) - _rms_normalize(target.detach())).abs().mean(dim=-1)
    row_weight = eligible.bool().unsqueeze(-1)
    changed_weight = row_weight & changed
    unchanged_weight = row_weight & ~changed

    def mean(mask: Tensor) -> Tensor:
        return (error * mask).sum() / mask.sum().clamp_min(1)

    all_loss = mean(row_weight.expand_as(error))
    changed_loss = mean(changed_weight)
    unchanged_loss = mean(unchanged_weight)
    return 0.5 * (all_loss + changed_loss), all_loss, changed_loss, unchanged_loss


def structured_horizon_loss(
    dynamics: StructuredDynamics,
    belief: StructuredBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    decode: DecodeContext | None,
    decision_horizon: int,
    patch_horizon: int,
    economy_active: bool = False,
    opponent_summary_active: bool = False,
    opponent_patches_active: bool = False,
    target_belief: StructuredBelief | None = None,
) -> StructuredDynamicsTerms:
    """Unroll typed dynamics against exact contiguous demonstrated successors."""
    if decision_horizon < 0 or patch_horizon < 0:
        raise ValueError("structured horizons cannot be negative")
    auxiliary_horizon = (
        patch_horizon
        if (patch_horizon or economy_active or opponent_summary_active or opponent_patches_active)
        else 0
    )
    max_horizon = max(decision_horizon, auxiliary_horizon)
    if max_horizon < 1:
        raise ValueError("at least one structured auxiliary horizon must be active")
    targets = belief if target_belief is None else target_belief
    predicted = belief
    zero = belief.central_latents.new_zeros((), dtype=torch.float32)
    sums = [zero for _ in range(12)]
    eligible_sum = zero
    decision_steps = 0
    patch_steps = 0
    state_steps = 0
    decision_one = decision_final = zero
    patch_one = patch_final = zero
    residual_sums = [zero for _ in StructuredBelief._fields]

    rows = torch.arange(
        factors["episode_index"].shape[0],
        device=factors["episode_index"].device,
    )
    for offset in range(1, max_horizon + 1):
        action_index = (rows + offset - 1).clamp_max(rows.shape[0] - 1)
        previous = predicted
        predicted = dynamics(
            predicted,
            factors["unit_actions"][action_index],
            factors["market_kinds"][action_index],
            factors["market_quantities"][action_index],
            inputs.unit_categorical[action_index],
            inputs.unit_active[action_index],
        )
        target_index, eligible = _target_index(factors["episode_index"], factors["step"], offset)
        joined_predicted = torch.cat(tuple(predicted), dim=1)
        joined_previous = torch.cat(tuple(previous), dim=1)
        sums[11] = sums[11] + _eligible_rms_ratio(joined_predicted, joined_previous, eligible)
        for kind, (predicted_value, previous_value) in enumerate(
            zip(predicted, previous, strict=True)
        ):
            residual_sums[kind] = residual_sums[kind] + _eligible_rms_ratio(
                predicted_value, previous_value, eligible
            )

        if offset <= decision_horizon:
            if decode is None:
                raise ValueError("decision horizon requires a decode context")
            predicted_decisions = torch.cat(
                (predicted.unit_decisions, predicted.market_decisions), dim=1
            )
            target_decisions = torch.cat(
                (
                    targets.unit_decisions[target_index],
                    targets.market_decisions[target_index],
                ),
                dim=1,
            )
            teacher = decode.heads.decode(target_decisions.detach())
            masks = DecodeMasks(*(field[target_index] for field in decode.masks))
            terms = latent_decode_kl_terms(
                predicted_decisions,
                teacher.unit_logits,
                teacher.market_kind_logits,
                teacher.market_quantity_context,
                decode.heads,
                masks,
                eligible,
            )
            sums[0] = sums[0] + terms.pooled
            sums[1] = sums[1] + terms.unit
            sums[2] = sums[2] + terms.market_kind
            sums[3] = sums[3] + terms.market_quantity
            if offset == 1:
                decision_one = terms.pooled
            if offset == decision_horizon:
                decision_final = terms.pooled
            decision_steps += 1

        if offset <= patch_horizon:
            source_categorical = inputs.tile_categorical[:, :TILE_COUNT]
            target_categorical = inputs.tile_categorical[target_index, :TILE_COUNT]
            source_continuous = inputs.tile_continuous[:, :TILE_COUNT]
            target_continuous = inputs.tile_continuous[target_index, :TILE_COUNT]
            changed = (source_categorical != target_categorical).any(dim=-1) | (
                source_continuous != target_continuous
            ).any(dim=-1)
            patch_terms = _patch_losses(
                predicted.own_patches,
                targets.own_patches[target_index],
                eligible,
                changed,
            )
            for position, value in enumerate(patch_terms, start=4):
                sums[position] = sums[position] + value
            if offset == 1:
                patch_one = patch_terms[0]
            if offset == patch_horizon:
                patch_final = patch_terms[0]
            patch_steps += 1

            if economy_active:
                sums[8] = sums[8] + _feature_l1(
                    predicted.economy_entities,
                    targets.economy_entities[target_index],
                    eligible,
                )
            if opponent_summary_active:
                sums[9] = sums[9] + _feature_l1(
                    predicted.opponent_summary,
                    targets.opponent_summary[target_index],
                    eligible,
                )
            if opponent_patches_active:
                sums[10] = sums[10] + _feature_l1(
                    predicted.opponent_patches,
                    targets.opponent_patches[target_index],
                    eligible,
                )
            state_steps += 1
        eligible_sum = eligible_sum + eligible.float().sum()

    decision_divisor = max(decision_steps, 1)
    patch_divisor = max(patch_steps, 1)
    state_divisor = max(state_steps, 1)
    return StructuredDynamicsTerms(
        decision=sums[0] / decision_divisor,
        decision_one=decision_one,
        decision_final=decision_final,
        decision_unit=sums[1] / decision_divisor,
        decision_market_kind=sums[2] / decision_divisor,
        decision_market_quantity=sums[3] / decision_divisor,
        patch=sums[4] / patch_divisor,
        patch_one=patch_one,
        patch_final=patch_final,
        patch_all=sums[5] / patch_divisor,
        patch_changed=sums[6] / patch_divisor,
        patch_unchanged=sums[7] / patch_divisor,
        economy=sums[8] / state_divisor,
        opponent_summary=sums[9] / state_divisor,
        opponent_patches=sums[10] / state_divisor,
        eligible=eligible_sum / max_horizon,
        residual_ratio=sums[11] / max_horizon,
        residual_own_patches=residual_sums[0] / max_horizon,
        residual_opponent_patches=residual_sums[1] / max_horizon,
        residual_opponent_summary=residual_sums[2] / max_horizon,
        residual_economy_entities=residual_sums[3] / max_horizon,
        residual_central_latents=residual_sums[4] / max_horizon,
        residual_unit_decisions=residual_sums[5] / max_horizon,
        residual_market_decisions=residual_sums[6] / max_horizon,
    )
