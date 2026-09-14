"""PPO NextLat on normalized actor output-head inputs, with live sources."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import torch
from torch import Tensor, nn

from kaggriculture.actions import MarketKind
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.latent_dynamics import (
    DecodeHeads,
    DecodeKLTerms,
    DecodeMasks,
    latent_decode_kl_terms,
)
from kaggriculture.model import RMSNorm
from kaggriculture.structured import (
    Block,
    StructuredActor,
    StructuredConfig,
    StructuredDecisionBelief,
    StructuredInputs,
)
from kaggriculture.structured_dynamics import (
    ShuffledActionDynamics,
    StructuredActionEncoder,
    StructuredHorizonPlan,
    _belief_rms_ratio,
    _target_index,
)


class ActorDynamics(nn.Module):
    """Shared residual transition of head-input slots under valid joint actions."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        predictor_config = replace(config, zero_init_branches=False, global_modulation=False)
        self.action = StructuredActionEncoder(config.model_dim)
        self.type_identity = nn.Embedding(2, config.model_dim)
        self.unit_identity = nn.Embedding(MAX_UNITS, config.model_dim)
        self.market_identity = nn.Embedding(MAX_MARKET_ORDERS, config.model_dim)
        self.context_norm = RMSNorm(config.model_dim)
        self.transition = Block(predictor_config)

    def forward(
        self,
        belief: StructuredDecisionBelief,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        unit_categorical: Tensor,
        unit_active: Tensor,
        unit_state_active: Tensor | None = None,
    ) -> StructuredDecisionBelief:
        state_active = unit_active.bool()
        if unit_state_active is not None:
            state_active = state_active & unit_state_active.bool()
        units = torch.where(state_active.unsqueeze(-1), belief.unit_decisions, 0.0)
        market = belief.market_decisions
        unit_action, market_action = self.action(
            unit_actions, market_kinds, market_quantities, unit_categorical, unit_active
        )
        stopped = (market_kinds == MarketKind.STOP.value).long()
        market_action_valid = stopped.cumsum(dim=1) - stopped == 0
        # Every market queue role has a head-input state, even after this
        # action's STOP. Actions include STOP itself, never subsequent slots.
        # Newborn units have observed actions but no recursively predicted state.
        context_valid = torch.cat(
            (
                state_active,
                torch.ones_like(market_kinds, dtype=torch.bool),
                unit_active.bool(),
                market_action_valid,
            ),
            dim=1,
        )
        context = torch.cat((units, market, unit_action, market_action), dim=1)
        query = torch.cat(
            (
                units + self.unit_identity.weight + self.type_identity.weight[0],
                market + self.market_identity.weight + self.type_identity.weight[1],
            ),
            dim=1,
        )
        transitioned = self.transition(
            query, context, context_norm=self.context_norm, context_valid=context_valid
        )
        unit_delta, market_delta = (transitioned - query).split(
            (units.shape[1], market.shape[1]), dim=1
        )
        return StructuredDecisionBelief(
            torch.where(state_active.unsqueeze(-1), units + unit_delta, 0.0),
            market + market_delta,
        )


class ActorDynamicsTerms(NamedTuple):
    latent: Tensor
    decision: Tensor
    decision_one: Tensor
    decision_final: Tensor
    decision_unit: Tensor
    decision_market_kind: Tensor
    decision_market_quantity: Tensor
    eligible: Tensor
    residual_ratio: Tensor


def _actor_decode_kl(
    heads: DecodeHeads,
    predicted: StructuredDecisionBelief,
    target: StructuredDecisionBelief,
    masks: DecodeMasks,
    eligible: Tensor,
) -> DecodeKLTerms:
    """Teacher-to-student KL through detached final policy projections only."""
    teacher = heads.decode(torch.cat(tuple(value.detach() for value in target), dim=1))
    return latent_decode_kl_terms(
        torch.cat(predicted, dim=1),
        teacher.unit_logits,
        teacher.market_kind_logits,
        teacher.market_quantity_context,
        heads,
        masks,
        eligible,
    )


def _actor_latent_loss(
    predicted: StructuredDecisionBelief,
    target: StructuredDecisionBelief,
    eligible: Tensor,
    unit_valid: Tensor,
    market_valid: Tensor,
) -> Tensor:
    """Mean SmoothL1 over eligible latent coordinates, not family means."""
    total = predicted.unit_decisions.new_zeros((), dtype=torch.float32)
    elements = total
    for current, teacher, valid in zip(predicted, target, (unit_valid, market_valid), strict=True):
        error = nn.functional.smooth_l1_loss(
            current.float(), teacher.detach().float(), reduction="none"
        )
        weight = (eligible.bool().unsqueeze(-1) & valid.bool()).float().unsqueeze(-1)
        total = total + (error * weight).sum()
        elements = elements + weight.sum() * current.shape[-1]
    return total / elements.clamp_min(1)


def _actor_loss(
    dynamics: nn.Module,
    actor: StructuredActor,
    belief: StructuredDecisionBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    decision_horizon: int,
    latent_horizon: int,
    plan: StructuredHorizonPlan | None,
    windowed: bool,
) -> ActorDynamicsTerms:
    if decision_horizon < 0 or latent_horizon < 0:
        raise ValueError("actor horizons cannot be negative")
    horizon = max(decision_horizon, latent_horizon)
    if horizon < 1:
        raise ValueError("at least one actor auxiliary horizon must be active")
    rows = belief.unit_decisions.shape[0]
    if rows == 0:
        raise ValueError("actor auxiliary belief cannot be empty")
    width = horizon + 1
    if windowed and rows % width:
        raise ValueError("actor window rows do not contain complete windows")
    # The matched shuffled control permutes the original dense action population.
    if isinstance(dynamics, ShuffledActionDynamics):
        plan = None
    if plan is not None and plan.eligible.shape[0] < horizon:
        raise ValueError("actor plan does not cover the requested horizon")
    all_rows = torch.arange(rows, device=belief.unit_decisions.device)
    source = all_rows if plan is None else plan.indices[0]
    predicted = (
        belief if plan is None else StructuredDecisionBelief(*(value[source] for value in belief))
    )
    surviving_units = inputs.unit_active[source].bool()
    heads = DecodeHeads.from_actor(actor, normalized_units=True) if decision_horizon else None
    zero = belief.unit_decisions.new_zeros((), dtype=torch.float32)
    latent = decision = unit = kind = quantity = eligible_sum = residual = zero
    decision_one = decision_final = zero
    for offset in range(1, horizon + 1):
        if windowed:
            positions = width - offset
            source = all_rows.reshape(-1, width)[:, :positions].flatten()
            previous_width = width if offset == 1 else positions + 1
            predicted = StructuredDecisionBelief(
                *(
                    value.reshape(-1, previous_width, *value.shape[1:])[:, :positions].flatten(0, 1)
                    for value in predicted
                )
            )
            surviving_units = surviving_units.reshape(-1, previous_width, MAX_UNITS)[
                :, :positions
            ].flatten(0, 1)
            action_index = source + offset - 1
            target_index = source + offset
            if "episode_index" in factors and "step" in factors:
                _, eligible = _target_index(factors["episode_index"], factors["step"], offset)
                eligible = eligible[source]
            else:
                eligible = torch.ones_like(source, dtype=torch.bool)
        elif plan is None:
            action_index = (all_rows + offset - 1).clamp_max(rows - 1)
            target_index, eligible = _target_index(
                factors["episode_index"], factors["step"], offset
            )
        else:
            action_index = plan.indices[offset]
            target_index = plan.indices[plan.eligible.shape[0] + offset]
            eligible = plan.eligible[offset - 1]
        previous = predicted
        action_active = inputs.unit_active.detach()[action_index]
        surviving_units = surviving_units & action_active.bool()
        predicted = dynamics(
            previous,
            factors["unit_actions"][action_index],
            factors["market_kinds"][action_index],
            factors["market_quantities"][action_index],
            inputs.unit_categorical.detach()[action_index],
            action_active,
            surviving_units,
        )
        # Availability is ancestry, not merely target occupancy: a birth or a
        # death/reappearance cannot restore a state absent from this source.
        surviving_units = surviving_units & inputs.unit_active[target_index].bool()
        residual = residual + _belief_rms_ratio(predicted, previous, eligible)
        eligible_sum = eligible_sum + eligible.float().sum()
        target = StructuredDecisionBelief(*(value.detach()[target_index] for value in belief))
        if offset <= latent_horizon:
            latent = latent + _actor_latent_loss(
                predicted,
                target,
                eligible,
                surviving_units,
                factors["market_active"][target_index],
            )
        if offset <= decision_horizon:
            masks = DecodeMasks(
                *(factors[name].detach()[target_index] for name in DecodeMasks._fields)
            )
            masks = masks._replace(unit_active=masks.unit_active & surviving_units)
            terms = _actor_decode_kl(heads, predicted, target, masks, eligible)
            decision = decision + terms.pooled
            unit = unit + terms.unit
            kind = kind + terms.market_kind
            quantity = quantity + terms.market_quantity
            if offset == 1:
                decision_one = terms.pooled
            if offset == decision_horizon:
                decision_final = terms.pooled
    return ActorDynamicsTerms(
        latent / max(latent_horizon, 1),
        decision / max(decision_horizon, 1),
        decision_one,
        decision_final,
        unit / max(decision_horizon, 1),
        kind / max(decision_horizon, 1),
        quantity / max(decision_horizon, 1),
        eligible_sum / horizon,
        residual / horizon,
    )


def actor_horizon_loss(
    dynamics: nn.Module,
    actor: StructuredActor,
    belief: StructuredDecisionBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    decision_horizon: int,
    latent_horizon: int,
    plan: StructuredHorizonPlan | None = None,
) -> ActorDynamicsTerms:
    """Unroll head inputs with cumulative dense or compact ancestry masks."""
    return _actor_loss(
        dynamics,
        actor,
        belief,
        inputs,
        factors,
        decision_horizon=decision_horizon,
        latent_horizon=latent_horizon,
        plan=plan,
        windowed=False,
    )


def actor_window_loss(
    dynamics: nn.Module,
    actor: StructuredActor,
    belief: StructuredDecisionBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    decision_horizon: int,
    latent_horizon: int,
) -> ActorDynamicsTerms:
    """Shrink recursive head predictions inside fixed complete transition windows."""
    return _actor_loss(
        dynamics,
        actor,
        belief,
        inputs,
        factors,
        decision_horizon=decision_horizon,
        latent_horizon=latent_horizon,
        plan=None,
        windowed=True,
    )
