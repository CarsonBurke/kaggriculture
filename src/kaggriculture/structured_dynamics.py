"""Training-only typed transition predictor for the structured actor."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS
from kaggriculture.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
)
from kaggriculture.latent_dynamics import (
    DecodeContext,
    DecodeMasks,
    latent_decode_kl_terms,
)
from kaggriculture.model import RMSNorm, softcap_value_logits
from kaggriculture.structured import (
    Block,
    StructuredBelief,
    StructuredConfig,
    StructuredCriticBelief,
    StructuredInputs,
)
from kaggriculture.tokens import TILE_COUNT

if TYPE_CHECKING:
    from kaggriculture.entity import EntityConfig


class PersistenceDynamics(nn.Module):
    """Metric-only no-change transition, with identical recursive horizons."""

    def forward(
        self,
        belief: Tensor | StructuredBelief | StructuredCriticBelief,
        *actions: Tensor,
        active_fields: tuple[bool, ...] | None = None,
    ) -> Tensor | StructuredBelief | StructuredCriticBelief:
        return belief


class ShuffledActionDynamics(nn.Module):
    """Metric-only cyclic action permutation; never touches a training RNG.

    Move complete action rows together so market kinds and quantities stay
    paired. Entity identities, masks, targets and decoder context stay fixed.
    """

    def __init__(self, dynamics: nn.Module) -> None:
        super().__init__()
        self.dynamics = dynamics

    def forward(
        self,
        belief: Tensor | StructuredBelief | StructuredCriticBelief,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        *context: Tensor,
        active_fields: tuple[bool, ...] | None = None,
    ) -> Tensor | StructuredBelief | StructuredCriticBelief:
        arguments = {} if active_fields is None else {"active_fields": active_fields}
        return self.dynamics(
            belief,
            unit_actions.roll(1, dims=0),
            market_kinds.roll(1, dims=0),
            market_quantities.roll(1, dims=0),
            *context,
            **arguments,
        )


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
        economy_tokens = (
            len(PRODUCTS) + len(ANIMALS) + len(CROPS) + 2 + (2 if config.split_clock_token else 1)
        )
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
        *,
        active_fields: tuple[bool, ...] | None = None,
    ) -> StructuredBelief:
        values = tuple(belief)
        if active_fields is None:
            active_fields = (True,) * len(values)
        if len(active_fields) != len(values) or not any(active_fields):
            raise ValueError("structured dynamics active fields must select belief families")
        selected = [
            (kind, value)
            for kind, (value, active) in enumerate(zip(values, active_fields, strict=True))
            if active
        ]
        queries = [
            self._query(value, StructuredBelief._fields.index(belief._fields[kind]))
            for kind, value in selected
        ]
        unit_action, market_action = self.action(
            unit_actions,
            market_kinds,
            market_quantities,
            unit_categorical,
            unit_active,
        )
        state_context = (
            belief.central_latents
            if any(active_fields[:5])
            else torch.cat((belief.unit_decisions, belief.market_decisions), dim=1)
        )
        context = torch.cat((state_context, unit_action, market_action), dim=1)
        joined = torch.cat(queries, dim=1)
        transitioned = self.transition(joined, context, context_norm=self.context_norm)
        residual = transitioned - joined
        deltas = residual.split([value.shape[1] for _, value in selected], dim=1)
        outputs = list(values)
        for (kind, value), delta in zip(selected, deltas, strict=True):
            outputs[kind] = value + delta
        return type(belief)(*outputs)


class StructuredCriticDynamics(nn.Module):
    """Predict only the normalized value-head input from itself and own actions."""

    def __init__(self, config: StructuredConfig | EntityConfig) -> None:
        super().__init__()
        predictor_config = replace(config, zero_init_branches=False, global_modulation=False)
        self.action = StructuredActionEncoder(config.model_dim)
        self.context_norm = RMSNorm(config.model_dim)
        self.transition = Block(predictor_config)

    def forward(
        self,
        belief: StructuredCriticBelief,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        unit_categorical: Tensor,
        unit_active: Tensor,
    ) -> StructuredCriticBelief:
        unit_action, market_action = self.action(
            unit_actions,
            market_kinds,
            market_quantities,
            unit_categorical,
            unit_active,
        )
        context = torch.cat((belief.value_decision, unit_action, market_action), dim=1)
        return StructuredCriticBelief(
            self.transition(belief.value_decision, context, context_norm=self.context_norm)
        )


class StructuredCriticDynamicsTerms(NamedTuple):
    latent: Tensor
    value: Tensor
    eligible: Tensor
    residual_ratio: Tensor


class StructuredDynamicsTerms(NamedTuple):
    latent: Tensor
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
    opponent_patch_all: Tensor
    opponent_patch_changed: Tensor
    opponent_patch_unchanged: Tensor
    eligible: Tensor
    residual_ratio: Tensor
    residual_own_patches: Tensor
    residual_opponent_patches: Tensor
    residual_opponent_summary: Tensor
    residual_economy_entities: Tensor
    residual_central_latents: Tensor
    residual_unit_decisions: Tensor
    residual_market_decisions: Tensor


class StructuredHorizonPlan(NamedTuple):
    """Bucketed source/action/target indices and explicit loss eligibility."""

    indices: Tensor
    eligible: Tensor


def structured_horizon_plan(
    episode_index: np.ndarray,
    step: np.ndarray,
    horizon: int,
) -> StructuredHorizonPlan:
    """Plan immutable CPU metadata; never discover variable shapes on CUDA.

    Eligibility is cumulative across every intervening step: a matching later
    endpoint cannot repair a broken recursive ancestry. Retain only sources
    with a valid first edge. The plan always has one column per row: shuffled
    runs that happen to continue each other make the eligible count vary from
    minibatch to minibatch, and any count-dependent width eventually reaches a
    shape a settled compiled update has never seen. All-false padding keeps
    zero-loss backward alive.
    """
    if horizon < 1:
        raise ValueError("structured plan horizon must be positive")
    if episode_index.ndim != 1 or step.shape != episode_index.shape or not step.size:
        raise ValueError("structured plan metadata must be matching nonempty vectors")
    rows = np.arange(step.size, dtype=np.int64)
    offsets = np.arange(1, horizon + 1, dtype=np.int64)[:, None]
    unclamped = rows[None, :] + offsets
    targets = np.minimum(unclamped, step.size - 1)
    eligible = (
        (unclamped < step.size)
        & (episode_index[targets] == episode_index[None, :])
        & (step[targets] == step[None, :] + offsets)
    )
    eligible = np.logical_and.accumulate(eligible, axis=0)
    sources = np.flatnonzero(eligible[0])
    count = sources.size
    bucket = step.size
    padded_sources = np.zeros(bucket, dtype=np.int64)
    padded_sources[:count] = sources
    indices = np.empty((1 + 2 * horizon, bucket), dtype=np.int64)
    indices[0] = padded_sources
    indices[1 : horizon + 1] = np.minimum(padded_sources[None, :] + offsets - 1, step.size - 1)
    indices[horizon + 1 :] = targets[:, padded_sources]
    mask = np.zeros((horizon, bucket), dtype=np.bool_)
    mask[:, :count] = eligible[:, sources]
    return StructuredHorizonPlan(torch.from_numpy(indices), torch.from_numpy(mask))


def _target_index(
    episode_index: Tensor,
    step: Tensor,
    offset: int,
    transition_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The row ``offset`` steps on, and whether every edge up to it is a real transition.

    ``transition_valid[j]`` says row j's recorded action is the one that produced
    row j + 1. A recovery demonstration labels a perturbed step with the teacher's
    action while the engine ran a deviation, so that edge is not the dynamics of
    its own label, and no chain through it is either.
    """
    rows = torch.arange(episode_index.shape[0], device=episode_index.device)
    last = episode_index.shape[0] - 1
    unclamped = rows + offset
    index = unclamped.clamp_max(last)
    eligible = unclamped < episode_index.shape[0]
    for distance in range(1, offset + 1):
        intermediate = (rows + distance).clamp_max(last)
        eligible = (
            eligible
            & (episode_index[intermediate] == episode_index)
            & (step[intermediate] == step + distance)
        )
        if transition_valid is not None:
            eligible = eligible & transition_valid[(rows + distance - 1).clamp_max(last)]
    return index, eligible


def _rms_normalize(value: Tensor) -> Tensor:
    return value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + 1e-6)


def _feature_l1(predicted: Tensor, target: Tensor, eligible: Tensor) -> Tensor:
    error = (_rms_normalize(predicted) - _rms_normalize(target.detach())).abs().mean(dim=-1)
    weight = eligible.float().unsqueeze(-1)
    return (error * weight).sum() / (weight.sum() * error.shape[1]).clamp_min(1.0)


def _latent_smooth_l1(
    predicted: Tensor,
    target: Tensor,
    eligible: Tensor,
) -> Tensor:
    """Mean SmoothL1 over eligible predicted-latent elements, as in NextLat."""
    error = nn.functional.smooth_l1_loss(
        predicted.float(),
        target.detach().float(),
        reduction="none",
    )
    weight = eligible.float().reshape(-1, *([1] * (error.ndim - 1)))
    return (error * weight).sum() / weight.expand_as(error).sum().clamp_min(1.0)


def _eligible_rms_ratio(predicted: Tensor, previous: Tensor, eligible: Tensor) -> Tensor:
    """Measure transition size without adding a diagnostic branch to backward."""
    predicted_value = predicted.detach().float()
    previous_value = previous.detach().float()
    weight = eligible.float().reshape(-1, *([1] * (predicted.ndim - 1)))
    elements = weight.sum() * predicted[0].numel()
    residual_rms = (
        ((predicted_value - previous_value).square() * weight).sum() / elements.clamp_min(1)
    ).sqrt()
    baseline_rms = ((previous_value.square() * weight).sum() / elements.clamp_min(1)).sqrt()
    return residual_rms / baseline_rms.clamp_min(1e-6)


def _belief_latent_smooth_l1(
    predicted: tuple[Tensor, ...],
    target: tuple[Tensor, ...],
    eligible: Tensor,
) -> Tensor:
    """Element-weighted SmoothL1 over every latent coordinate of eligible rows."""
    total = predicted[0].new_zeros((), dtype=torch.float32)
    elements = total
    for current, teacher in zip(predicted, target, strict=True):
        error = nn.functional.smooth_l1_loss(
            current.float(), teacher.detach().float(), reduction="none"
        )
        weight = eligible.float().reshape(-1, *([1] * (current.ndim - 1)))
        elements = elements + eligible.float().sum() * current[0].numel()
        total = total + (error * weight).sum()
    return total / elements.clamp_min(1)


def _belief_rms_ratio(
    predicted: tuple[Tensor, ...],
    previous: tuple[Tensor, ...],
    eligible: Tensor,
) -> Tensor:
    """Global residual/baseline RMS, not a mean of differently sized families."""
    residual = predicted[0].new_zeros((), dtype=torch.float32)
    baseline = residual
    elements_per_row = 0
    for current, source in zip(predicted, previous, strict=True):
        current_value = current.detach().float()
        source_value = source.detach().float()
        weight = eligible.float().reshape(-1, *([1] * (current.ndim - 1)))
        residual = residual + ((current_value - source_value).square() * weight).sum()
        baseline = baseline + (source_value.square() * weight).sum()
        elements_per_row += current[0].numel()
    elements = (eligible.float().sum() * elements_per_row).clamp_min(1)
    return (residual / elements).sqrt() / (baseline / elements).sqrt().clamp_min(1e-6)


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


def _tile_changes(
    source_categorical: Tensor,
    target_categorical: Tensor,
    source_continuous: Tensor,
    target_continuous: Tensor,
) -> Tensor:
    """Exact per-tile semantic or continuous transition mask."""
    return (source_categorical != target_categorical).any(dim=-1) | (
        source_continuous != target_continuous
    ).any(dim=-1)


def _active_belief_fields(
    *,
    decision_horizon: int,
    own_patches_active: bool,
    recurrent_workspace: bool,
    economy_active: bool,
    opponent_summary_active: bool,
    opponent_patches_active: bool,
) -> tuple[bool, ...]:
    return (
        own_patches_active,
        opponent_patches_active,
        opponent_summary_active,
        economy_active,
        recurrent_workspace
        and (
            own_patches_active
            or economy_active
            or opponent_summary_active
            or opponent_patches_active
        ),
        bool(decision_horizon),
        bool(decision_horizon),
    )


def _selected_belief_fields(belief, active: tuple[bool, ...]) -> tuple[bool, ...]:
    selected = tuple(active[StructuredBelief._fields.index(name)] for name in belief._fields)
    # An actor built with `actor_opponent_farm=False` produces zero-token
    # opponent patches and summary, so an objective over them would average a
    # loss over nothing and report a clean zero rather than refusing.
    for name, chosen in zip(belief._fields, selected, strict=True):
        if chosen and name.startswith("opponent_") and getattr(belief, name).shape[1] == 0:
            raise ValueError(f"{name} objective requires an actor that reads the opponent farm")
    return selected


def structured_horizon_loss(
    dynamics: StructuredDynamics,
    belief: StructuredBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    decode: DecodeContext | None,
    decision_horizon: int,
    patch_horizon: int,
    own_patches_active: bool = True,
    latent_horizon: int = 0,
    economy_active: bool = False,
    opponent_summary_active: bool = False,
    opponent_patches_active: bool = False,
    target_belief: StructuredBelief | None = None,
    plan: StructuredHorizonPlan | None = None,
) -> StructuredDynamicsTerms:
    """Unroll typed dynamics against exact contiguous demonstrated successors."""
    if decision_horizon < 0 or patch_horizon < 0 or latent_horizon < 0:
        raise ValueError("structured horizons cannot be negative")
    auxiliary_horizon = (
        patch_horizon
        if (patch_horizon or economy_active or opponent_summary_active or opponent_patches_active)
        else 0
    )
    max_horizon = max(decision_horizon, auxiliary_horizon, latent_horizon)
    if max_horizon < 1:
        raise ValueError("at least one structured auxiliary horizon must be active")
    targets = belief if target_belief is None else target_belief
    # The shuffled control permutes the original batch, including invalid rows.
    if isinstance(dynamics, ShuffledActionDynamics):
        plan = None
    if plan is not None and plan.eligible.shape[0] < max_horizon:
        raise ValueError("structured plan does not cover the requested horizon")
    source_index = (
        torch.arange(belief.unit_decisions.shape[0], device=belief.unit_decisions.device)
        if plan is None
        else plan.indices[0]
    )
    predicted = belief if plan is None else type(belief)(*(value[source_index] for value in belief))
    zero = belief.unit_decisions.new_zeros((), dtype=torch.float32)
    sums = [zero for _ in range(16)]
    eligible_sum = zero
    decision_steps = 0
    patch_steps = 0
    state_steps = 0
    decision_one = decision_final = zero
    patch_one = patch_final = zero
    residual_sums = [zero for _ in StructuredBelief._fields]
    active_fields = _active_belief_fields(
        decision_horizon=max(decision_horizon, latent_horizon),
        recurrent_workspace=max_horizon > 1,
        own_patches_active=own_patches_active and patch_horizon > 0,
        economy_active=economy_active,
        opponent_summary_active=opponent_summary_active,
        opponent_patches_active=opponent_patches_active,
    )
    active_fields = _selected_belief_fields(belief, active_fields)

    rows = torch.arange(
        factors["episode_index"].shape[0],
        device=factors["episode_index"].device,
    )
    for offset in range(1, max_horizon + 1):
        action_index = (
            (rows + offset - 1).clamp_max(rows.shape[0] - 1)
            if plan is None
            else plan.indices[offset]
        )
        previous = predicted
        predicted = dynamics(
            predicted,
            factors["unit_actions"][action_index],
            factors["market_kinds"][action_index],
            factors["market_quantities"][action_index],
            inputs.unit_categorical[action_index],
            inputs.unit_active[action_index],
            active_fields=active_fields,
        )
        if plan is None:
            target_index, eligible = _target_index(
                factors["episode_index"],
                factors["step"],
                offset,
                factors.get("transition_valid"),
            )
        else:
            target_index = plan.indices[plan.eligible.shape[0] + offset]
            eligible = plan.eligible[offset - 1]
        sums[11] = sums[11] + _belief_rms_ratio(
            tuple(value for value, active in zip(predicted, active_fields, strict=True) if active),
            tuple(value for value, active in zip(previous, active_fields, strict=True) if active),
            eligible,
        )
        for kind, (predicted_value, previous_value) in enumerate(
            zip(predicted, previous, strict=True)
        ):
            if active_fields[kind]:
                field_index = StructuredBelief._fields.index(belief._fields[kind])
                residual_sums[field_index] = residual_sums[field_index] + _eligible_rms_ratio(
                    predicted_value, previous_value, eligible
                )

        if offset <= latent_horizon:
            sums[12] = sums[12] + _belief_latent_smooth_l1(
                (predicted.unit_decisions, predicted.market_decisions),
                (targets.unit_decisions[target_index], targets.market_decisions[target_index]),
                eligible,
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
            if own_patches_active:
                source_categorical = inputs.tile_categorical[source_index, :TILE_COUNT]
                target_categorical = inputs.tile_categorical[target_index, :TILE_COUNT]
                source_continuous = inputs.tile_continuous[source_index, :TILE_COUNT]
                target_continuous = inputs.tile_continuous[target_index, :TILE_COUNT]
                changed = _tile_changes(
                    source_categorical,
                    target_categorical,
                    source_continuous,
                    target_continuous,
                )
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
                opponent_slice = slice(TILE_COUNT, 2 * TILE_COUNT)
                changed = _tile_changes(
                    inputs.tile_categorical[source_index, opponent_slice],
                    inputs.tile_categorical[target_index, opponent_slice],
                    inputs.tile_continuous[source_index, opponent_slice],
                    inputs.tile_continuous[target_index, opponent_slice],
                )
                opponent_patch_terms = _patch_losses(
                    predicted.opponent_patches,
                    targets.opponent_patches[target_index],
                    eligible,
                    changed,
                )
                sums[10] = sums[10] + opponent_patch_terms[0]
                for position, value in enumerate(opponent_patch_terms[1:], start=13):
                    sums[position] = sums[position] + value
            state_steps += 1
        eligible_sum = eligible_sum + eligible.float().sum()

    decision_divisor = max(decision_steps, 1)
    patch_divisor = max(patch_steps, 1)
    state_divisor = max(state_steps, 1)
    return StructuredDynamicsTerms(
        latent=sums[12] / max(latent_horizon, 1),
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
        opponent_patch_all=sums[13] / state_divisor,
        opponent_patch_changed=sums[14] / state_divisor,
        opponent_patch_unchanged=sums[15] / state_divisor,
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


def structured_window_loss(
    dynamics: StructuredDynamics,
    belief: StructuredBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    decode: DecodeContext | None,
    decision_horizon: int,
    patch_horizon: int,
    latent_horizon: int = 0,
    own_patches_active: bool,
    economy_active: bool = False,
    opponent_summary_active: bool = False,
    opponent_patches_active: bool = False,
) -> StructuredDynamicsTerms:
    """Score complete fixed-width windows without running invalid trailing rows.

    ``structured_horizon_loss`` accepts arbitrary flat sequences, optionally
    compacted with an immutable CPU plan. This window path instead consumes
    validated groups of exactly ``max_horizon + 1`` rows and shrinks recursive
    predictions as each trailing position loses its successor.
    """
    if decision_horizon < 0 or patch_horizon < 0 or latent_horizon < 0:
        raise ValueError("structured horizons cannot be negative")
    max_horizon = max(decision_horizon, patch_horizon, latent_horizon)
    if max_horizon < 1:
        raise ValueError("structured window objective needs a positive horizon")
    width = max_horizon + 1
    rows = belief.unit_decisions.shape[0]
    if rows % width:
        raise ValueError("structured window rows do not contain complete windows")
    windows = rows // width
    if "transition_valid" in factors:
        # Every edge inside a window is taken as a transition, with no mask to
        # drop one; a recovery demonstration's perturbed steps need the flat path.
        raise ValueError("the window objective cannot mask invalid transitions")

    def window(value: Tensor) -> Tensor:
        return value.reshape(windows, width, *value.shape[1:])

    windowed_belief = type(belief)(*(window(value) for value in belief))
    windowed_inputs = StructuredInputs(*(window(value) for value in inputs))
    windowed_factors = {
        name: window(value)
        for name, value in factors.items()
        if name not in {"episode_index", "step"}
    }
    zero = belief.unit_decisions.new_zeros((), dtype=torch.float32)
    sums = [zero for _ in range(15)]
    latent_sum = zero
    eligible_sum = zero
    decision_steps = 0
    patch_steps = 0
    state_steps = 0
    decision_one = decision_final = zero
    patch_one = patch_final = zero
    residual_sums = [zero for _ in StructuredBelief._fields]
    active_fields = _active_belief_fields(
        decision_horizon=max(decision_horizon, latent_horizon),
        recurrent_workspace=max_horizon > 1,
        own_patches_active=own_patches_active and patch_horizon > 0,
        economy_active=economy_active,
        opponent_summary_active=opponent_summary_active,
        opponent_patches_active=opponent_patches_active,
    )
    active_fields = _selected_belief_fields(belief, active_fields)
    predicted: StructuredBelief | None = None

    for offset in range(1, max_horizon + 1):
        source_positions = width - offset
        if predicted is None:
            previous = type(belief)(
                *(value[:, :source_positions].flatten(0, 1) for value in windowed_belief)
            )
        else:
            previous = type(belief)(
                *(
                    value.reshape(windows, source_positions + 1, *value.shape[1:])[
                        :, :source_positions
                    ].flatten(0, 1)
                    for value in predicted
                )
            )
        action_slice = slice(offset - 1, offset - 1 + source_positions)
        current = dynamics(
            previous,
            windowed_factors["unit_actions"][:, action_slice].flatten(0, 1),
            windowed_factors["market_kinds"][:, action_slice].flatten(0, 1),
            windowed_factors["market_quantities"][:, action_slice].flatten(0, 1),
            windowed_inputs.unit_categorical[:, action_slice].flatten(0, 1),
            windowed_inputs.unit_active[:, action_slice].flatten(0, 1),
            active_fields=active_fields,
        )
        predicted = current
        target_slice = slice(offset, offset + source_positions)
        targets = type(belief)(*(value[:, target_slice].flatten(0, 1) for value in windowed_belief))
        eligible = torch.ones(
            windows * source_positions,
            dtype=torch.bool,
            device=belief.unit_decisions.device,
        )
        sums[11] = sums[11] + _belief_rms_ratio(
            tuple(value for value, active in zip(current, active_fields, strict=True) if active),
            tuple(value for value, active in zip(previous, active_fields, strict=True) if active),
            eligible,
        )
        for kind, (predicted_value, previous_value) in enumerate(
            zip(current, previous, strict=True)
        ):
            if active_fields[kind]:
                field_index = StructuredBelief._fields.index(belief._fields[kind])
                residual_sums[field_index] = residual_sums[field_index] + _eligible_rms_ratio(
                    predicted_value, previous_value, eligible
                )

        if offset <= latent_horizon:
            latent_sum = latent_sum + _belief_latent_smooth_l1(
                (current.unit_decisions, current.market_decisions),
                (targets.unit_decisions, targets.market_decisions),
                eligible,
            )

        if offset <= decision_horizon:
            if decode is None:
                raise ValueError("decision horizon requires a decode context")
            predicted_decisions = torch.cat(
                (current.unit_decisions, current.market_decisions), dim=1
            )
            target_decisions = torch.cat((targets.unit_decisions, targets.market_decisions), dim=1)
            teacher = decode.heads.decode(target_decisions.detach())
            masks = DecodeMasks(
                *(
                    windowed_factors[field][:, target_slice].flatten(0, 1)
                    for field in DecodeMasks._fields
                )
            )
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
            if own_patches_active:
                source_categorical = windowed_inputs.tile_categorical[
                    :, :source_positions, :TILE_COUNT
                ].flatten(0, 1)
                target_categorical = windowed_inputs.tile_categorical[
                    :, target_slice, :TILE_COUNT
                ].flatten(0, 1)
                source_continuous = windowed_inputs.tile_continuous[
                    :, :source_positions, :TILE_COUNT
                ].flatten(0, 1)
                target_continuous = windowed_inputs.tile_continuous[
                    :, target_slice, :TILE_COUNT
                ].flatten(0, 1)
                changed = _tile_changes(
                    source_categorical,
                    target_categorical,
                    source_continuous,
                    target_continuous,
                )
                patch_terms = _patch_losses(
                    current.own_patches,
                    targets.own_patches,
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
                    current.economy_entities,
                    targets.economy_entities,
                    eligible,
                )
            if opponent_summary_active:
                sums[9] = sums[9] + _feature_l1(
                    current.opponent_summary,
                    targets.opponent_summary,
                    eligible,
                )
            if opponent_patches_active:
                opponent_slice = slice(TILE_COUNT, 2 * TILE_COUNT)
                source_categorical = windowed_inputs.tile_categorical[
                    :, :source_positions, opponent_slice
                ].flatten(0, 1)
                target_categorical = windowed_inputs.tile_categorical[
                    :, target_slice, opponent_slice
                ].flatten(0, 1)
                source_continuous = windowed_inputs.tile_continuous[
                    :, :source_positions, opponent_slice
                ].flatten(0, 1)
                target_continuous = windowed_inputs.tile_continuous[
                    :, target_slice, opponent_slice
                ].flatten(0, 1)
                changed = _tile_changes(
                    source_categorical,
                    target_categorical,
                    source_continuous,
                    target_continuous,
                )
                opponent_patch_terms = _patch_losses(
                    current.opponent_patches,
                    targets.opponent_patches,
                    eligible,
                    changed,
                )
                sums[10] = sums[10] + opponent_patch_terms[0]
                for position, value in enumerate(opponent_patch_terms[1:], start=12):
                    sums[position] = sums[position] + value
            state_steps += 1
        eligible_sum = eligible_sum + eligible.float().sum()

    decision_divisor = max(decision_steps, 1)
    patch_divisor = max(patch_steps, 1)
    state_divisor = max(state_steps, 1)
    return StructuredDynamicsTerms(
        latent=latent_sum / max(latent_horizon, 1),
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
        opponent_patch_all=sums[12] / state_divisor,
        opponent_patch_changed=sums[13] / state_divisor,
        opponent_patch_unchanged=sums[14] / state_divisor,
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


def _critic_value_kl(
    predicted: Tensor,
    target: Tensor,
    value_head: nn.Linear,
    eligible: Tensor | None = None,
) -> Tensor:
    """Teacher-to-student value KL without auxiliary head gradients.

    Scalar MSE readouts represent unit-variance Gaussians, whose KL is half
    squared mean error. A one-category softmax would erase this supervision.
    """
    weight = value_head.weight.detach()
    bias = None if value_head.bias is None else value_head.bias.detach()
    # Match Linear's explicit bias epilogue, without gradients into the head.
    student_logits = nn.functional.linear(predicted, weight, None)
    teacher_logits = nn.functional.linear(target.detach(), weight, None)
    if bias is not None:
        student_logits = student_logits + bias.to(student_logits.dtype)
        teacher_logits = teacher_logits + bias.to(teacher_logits.dtype)
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float().detach()
    if weight.shape[0] == 1:
        per_row = 0.5 * (student_logits - teacher_logits).square().sum(dim=-1)
    else:
        student_logits = softcap_value_logits(student_logits)
        teacher_logits = softcap_value_logits(teacher_logits)
        teacher_log_probabilities = teacher_logits.log_softmax(dim=-1)
        teacher_probabilities = teacher_log_probabilities.exp()
        per_row = (
            teacher_probabilities * (teacher_log_probabilities - student_logits.log_softmax(dim=-1))
        ).sum(dim=-1)
    # Beliefs keep a singleton token dimension. Reduce tokens before applying
    # the row mask: [B, 1] * [B] would broadcast to [B, B], admitting invalid
    # successors and scaling the value objective by the entire minibatch.
    per_row = per_row.reshape(per_row.shape[0], -1).mean(dim=-1)
    if eligible is None:
        return per_row.mean()
    weight_rows = eligible.float()
    return (per_row * weight_rows).sum() / weight_rows.sum().clamp_min(1.0)


def structured_critic_window_loss(
    dynamics: StructuredCriticDynamics,
    belief: StructuredCriticBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    value_head: nn.Linear,
    horizon: int,
) -> StructuredCriticDynamicsTerms:
    """Average recursive critic-belief losses over complete transition windows."""
    if horizon < 1:
        raise ValueError("structured critic horizon must be positive")
    width = horizon + 1
    rows = belief.value_decision.shape[0]
    if rows == 0 or rows % width:
        raise ValueError("structured critic rows do not contain complete windows")
    for value in inputs:
        if value.shape[0] != rows:
            raise ValueError("structured critic inputs must match belief rows")
    for name in ("unit_actions", "market_kinds", "market_quantities"):
        if factors[name].shape[0] != rows:
            raise ValueError(f"structured critic factor {name} must match belief rows")

    windows = rows // width
    if "transition_valid" in factors:
        # Every edge inside a window is taken as a transition, with no mask to
        # drop one; a recovery demonstration's perturbed steps need the flat path.
        raise ValueError("the window objective cannot mask invalid transitions")

    def window(value: Tensor) -> Tensor:
        return value.reshape(windows, width, *value.shape[1:])

    windowed_belief = StructuredCriticBelief(*(window(value) for value in belief))
    windowed_inputs = StructuredInputs(*(window(value) for value in inputs))
    windowed_factors = {
        name: window(factors[name])
        for name in ("unit_actions", "market_kinds", "market_quantities")
    }
    zero = belief.value_decision.new_zeros((), dtype=torch.float32)
    latent_sum = zero
    value_sum = zero
    eligible_sum = zero
    residual_sum = zero
    predicted: StructuredCriticBelief | None = None

    for offset in range(1, horizon + 1):
        source_positions = width - offset
        if predicted is None:
            previous = StructuredCriticBelief(
                *(value[:, :source_positions].flatten(0, 1) for value in windowed_belief)
            )
        else:
            previous = StructuredCriticBelief(
                *(
                    value.reshape(windows, source_positions + 1, *value.shape[1:])[
                        :, :source_positions
                    ].flatten(0, 1)
                    for value in predicted
                )
            )
        action_slice = slice(offset - 1, offset - 1 + source_positions)
        current = dynamics(
            previous,
            windowed_factors["unit_actions"][:, action_slice].flatten(0, 1),
            windowed_factors["market_kinds"][:, action_slice].flatten(0, 1),
            windowed_factors["market_quantities"][:, action_slice].flatten(0, 1),
            windowed_inputs.unit_categorical[:, action_slice].flatten(0, 1),
            windowed_inputs.unit_active[:, action_slice].flatten(0, 1),
        )
        predicted = current
        target_slice = slice(offset, offset + source_positions)
        target = StructuredCriticBelief(
            *(value[:, target_slice].flatten(0, 1) for value in windowed_belief)
        )
        eligible = torch.ones(
            windows * source_positions,
            dtype=torch.bool,
            device=belief.value_decision.device,
        )
        latent_sum = latent_sum + _belief_latent_smooth_l1(current, target, eligible)
        value_sum = value_sum + _critic_value_kl(
            current.value_decision,
            target.value_decision,
            value_head,
        )
        residual_sum = residual_sum + _belief_rms_ratio(current, previous, eligible)
        eligible_sum = eligible_sum + eligible.float().sum()

    return StructuredCriticDynamicsTerms(
        latent=latent_sum / horizon,
        value=value_sum / horizon,
        eligible=eligible_sum / horizon,
        residual_ratio=residual_sum / horizon,
    )


def structured_critic_horizon_loss(
    dynamics: StructuredCriticDynamics,
    belief: StructuredCriticBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    value_head: nn.Linear,
    horizon: int,
    plan: StructuredHorizonPlan | None = None,
) -> StructuredCriticDynamicsTerms:
    """Unroll critic dynamics against in-batch successors, as NextLat shifts h_t.

    Unlike the window path, this accepts a flat run-length minibatch and masks
    pairs that are not adjacent steps of one trajectory.
    """
    if horizon < 1:
        raise ValueError("structured critic horizon must be positive")
    if isinstance(dynamics, ShuffledActionDynamics):
        plan = None
    if plan is not None and plan.eligible.shape[0] < horizon:
        raise ValueError("structured plan does not cover the requested horizon")
    predicted = (
        belief
        if plan is None
        else StructuredCriticBelief(*(value[plan.indices[0]] for value in belief))
    )
    zero = belief.value_decision.new_zeros((), dtype=torch.float32)
    latent_sum = zero
    value_sum = zero
    eligible_sum = zero
    residual_sum = zero
    rows = torch.arange(
        factors["episode_index"].shape[0],
        device=factors["episode_index"].device,
    )
    for offset in range(1, horizon + 1):
        action_index = (
            (rows + offset - 1).clamp_max(rows.shape[0] - 1)
            if plan is None
            else plan.indices[offset]
        )
        previous = predicted
        predicted = dynamics(
            predicted,
            factors["unit_actions"][action_index],
            factors["market_kinds"][action_index],
            factors["market_quantities"][action_index],
            inputs.unit_categorical[action_index],
            inputs.unit_active[action_index],
        )
        if plan is None:
            target_index, eligible = _target_index(
                factors["episode_index"],
                factors["step"],
                offset,
                factors.get("transition_valid"),
            )
        else:
            target_index = plan.indices[plan.eligible.shape[0] + offset]
            eligible = plan.eligible[offset - 1]
        target = StructuredCriticBelief(*(value[target_index] for value in belief))
        latent_sum = latent_sum + _belief_latent_smooth_l1(predicted, target, eligible)
        value_sum = value_sum + _critic_value_kl(
            predicted.value_decision,
            target.value_decision,
            value_head,
            eligible,
        )
        residual_sum = residual_sum + _belief_rms_ratio(predicted, previous, eligible)
        eligible_sum = eligible_sum + eligible.float().sum()
    return StructuredCriticDynamicsTerms(
        latent=latent_sum / horizon,
        value=value_sum / horizon,
        eligible=eligible_sum / horizon,
        residual_ratio=residual_sum / horizon,
    )
