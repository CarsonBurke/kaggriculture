"""PPO NextLat on normalized actor output-head inputs, with live sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn

from kaggriculture.actions import N_MARKET_KINDS, QUANTIFIED_MARKET_KINDS, MarketKind
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.latent_dynamics import (
    DecodeHeads,
    DecodeKLTerms,
    DecodeMasks,
    _decision_kl,
    _frozen_linear,
)
from kaggriculture.model import RMSNorm
from kaggriculture.structured import (
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

if TYPE_CHECKING:
    from kaggriculture.entity import EntityActor, EntityConfig


class ActorHeadBelief(NamedTuple):
    """Independent recurrence states at the three normalized actor head inputs."""

    unit_decisions: Tensor
    market_kind_decisions: Tensor
    market_quantity_decisions: Tensor


class _ActorHeadPredictor(nn.Module):
    """NextLat residual MLP for one head, conditioned on the joint action."""

    def __init__(self, width: int) -> None:
        super().__init__()
        input_dim = 2 * width
        hidden_dim = 128 * max(1, round(input_dim / 128))
        # NextLat's LayerNorm(bias=False) implements affine RMS normalization.
        self.norm_x = RMSNorm(input_dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, width, bias=False),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        transition = torch.cat((action.unsqueeze(1).expand_as(state), state), dim=-1)
        return state + self.mlp(self.norm_x(transition))


class ActorDynamics(nn.Module):
    """Three independent residual predictors sharing one valid joint-action code."""

    def __init__(self, config: StructuredConfig | EntityConfig) -> None:
        super().__init__()
        width = config.model_dim
        self.action = StructuredActionEncoder(width)
        self.action_projection = nn.Linear(
            (MAX_UNITS + MAX_MARKET_ORDERS) * width, width, bias=False
        )
        self.unit_predictor = _ActorHeadPredictor(width)
        self.market_kind_predictor = _ActorHeadPredictor(width)
        self.market_quantity_predictor = _ActorHeadPredictor(width)
        self.register_buffer(
            "quantified_market_kinds",
            torch.tensor([kind in QUANTIFIED_MARKET_KINDS for kind in range(N_MARKET_KINDS)]),
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        belief: ActorHeadBelief,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        unit_categorical: Tensor,
        unit_active: Tensor,
        unit_state_active: Tensor | None = None,
    ) -> ActorHeadBelief:
        action_active = unit_active.bool()
        state_active = action_active
        if unit_state_active is not None:
            state_active = state_active & unit_state_active.bool()
        units = torch.where(state_active.unsqueeze(-1), belief.unit_decisions, 0.0)
        stopped = (market_kinds == MarketKind.STOP.value).long()
        market_action_valid = stopped.cumsum(dim=1) - stopped == 0
        valid_kinds = torch.where(market_action_valid, market_kinds, MarketKind.STOP.value)
        valid_quantities = torch.where(
            market_action_valid & self.quantified_market_kinds[valid_kinds],
            market_quantities,
            0,
        )
        unit_action, market_action = self.action(
            torch.where(action_active, unit_actions, 0),
            valid_kinds,
            valid_quantities,
            torch.where(action_active.unsqueeze(-1), unit_categorical, 0),
            action_active,
        )
        # Include STOP itself, but no later queue slots or inactive unit tokens.
        # A newborn unit's action is valid even without a recurrent source state.
        action = self.action_projection(
            torch.cat(
                (
                    torch.where(action_active.unsqueeze(-1), unit_action, 0.0),
                    torch.where(market_action_valid.unsqueeze(-1), market_action, 0.0),
                ),
                dim=1,
            ).flatten(1)
        )
        return ActorHeadBelief(
            torch.where(state_active.unsqueeze(-1), self.unit_predictor(units, action), 0.0),
            self.market_kind_predictor(belief.market_kind_decisions, action),
            self.market_quantity_predictor(belief.market_quantity_decisions, action),
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
    predicted: ActorHeadBelief,
    target: ActorHeadBelief,
    masks: DecodeMasks,
    eligible: Tensor,
) -> DecodeKLTerms:
    """Sum head-mean teacher-to-student KL through the matching frozen readouts."""
    row = eligible.bool().unsqueeze(-1)
    # Keep decode/KL numerics consistent with DecodeHeads.decode; predictor
    # execution remains under the caller's autocast context.
    with torch.autocast(device_type=predicted.unit_decisions.device.type, enabled=False):
        unit_kl, unit_weight = _decision_kl(
            _frozen_linear(heads.unit_projection, predicted.unit_decisions.float()),
            _frozen_linear(heads.unit_projection, target.unit_decisions.detach().float()),
            masks.unit_masks,
            (row & masks.unit_active).float(),
        )
        kind_kl, kind_weight = _decision_kl(
            _frozen_linear(heads.market_kind, predicted.market_kind_decisions.float()),
            _frozen_linear(heads.market_kind, target.market_kind_decisions.detach().float()),
            masks.market_kind_masks,
            (row & masks.market_active).float(),
        )
        student_quantity = _frozen_linear(
            heads.market_quantity_context, predicted.market_quantity_decisions.float()
        )
        teacher_quantity = _frozen_linear(
            heads.market_quantity_context, target.market_quantity_decisions.detach().float()
        )
        quantity_kl, quantity_weight = _decision_kl(
            heads.quantity_logits(
                student_quantity, masks.market_kinds, masks.market_quantity_masks
            ),
            heads.quantity_logits(
                teacher_quantity, masks.market_kinds, masks.market_quantity_masks
            ),
            masks.market_quantity_masks,
            (row & masks.market_quantity_active).float(),
        )
        unit = unit_kl / unit_weight.clamp_min(1.0)
        kind = kind_kl / kind_weight.clamp_min(1.0)
        quantity = quantity_kl / quantity_weight.clamp_min(1.0)
    return DecodeKLTerms(unit + kind + quantity, unit, kind, quantity)


def _actor_latent_loss(
    predicted: ActorHeadBelief,
    target: ActorHeadBelief,
    eligible: Tensor,
    unit_valid: Tensor,
    market_valid: Tensor,
    quantity_valid: Tensor,
) -> Tensor:
    """Sum three independently normalized eligible-coordinate SmoothL1 means."""
    total = predicted.unit_decisions.new_zeros((), dtype=torch.float32)
    for current, teacher, valid in zip(
        predicted, target, (unit_valid, market_valid, quantity_valid), strict=True
    ):
        error = nn.functional.smooth_l1_loss(
            current.float(), teacher.detach().float(), reduction="none"
        )
        weight = (eligible.bool().unsqueeze(-1) & valid.bool()).float().unsqueeze(-1)
        elements = weight.sum() * current.shape[-1]
        total = total + (error * weight).sum() / elements.clamp_min(1)
    return total


def _actor_loss(
    dynamics: nn.Module,
    actor: StructuredActor | EntityActor,
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
    source_belief = (
        belief if plan is None else StructuredDecisionBelief(*(value[source] for value in belief))
    )
    predicted = ActorHeadBelief(
        source_belief.unit_decisions,
        source_belief.market_decisions,
        source_belief.market_decisions,
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
            predicted = ActorHeadBelief(
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
                _, eligible = _target_index(
                    factors["episode_index"],
                    factors["step"],
                    offset,
                    factors.get("transition_valid"),
                )
                eligible = eligible[source]
            else:
                eligible = torch.ones_like(source, dtype=torch.bool)
        elif plan is None:
            action_index = (all_rows + offset - 1).clamp_max(rows - 1)
            target_index, eligible = _target_index(
                factors["episode_index"],
                factors["step"],
                offset,
                factors.get("transition_valid"),
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
        target_units = belief.unit_decisions.detach()[target_index]
        target_market = belief.market_decisions.detach()[target_index]
        target = ActorHeadBelief(target_units, target_market, target_market)
        if offset <= latent_horizon:
            latent = latent + _actor_latent_loss(
                predicted,
                target,
                eligible,
                surviving_units,
                factors["market_active"][target_index],
                factors["market_quantity_active"][target_index],
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
    actor: StructuredActor | EntityActor,
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
    actor: StructuredActor | EntityActor,
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
