"""Self-predictive latent dynamics auxiliary: NextLat transplanted to episode steps.

Transplants the next-latent objective of NextLat (Teoh et al., arXiv 2511.05963;
reference implementation ``models/model_nextlat.py``) from "position t in a token
sequence" to "step t in an episode", as NEXTLAT_AUX_PLAN.md argues it must be. A
small dynamics model p_psi predicts the actor's own next-step belief latent from
the current one plus the executed joint action, and two terms hold it honest:

1. ``latent_dynamics_loss`` -- SmoothL1 onto a stop-gradient target, reduced over
   masked *elements* exactly as the reference does. The reference calls this term
   "MSE" throughout; the name is a misnomer for `F.smooth_l1_loss` and is not
   carried over here.
2. ``latent_decode_kl`` -- decode the *predicted* latent through the policy's own
   action heads with the head weights detached, and match the decode of the true
   next latent under KL(teacher || student). This is what makes the latent
   decision-relevant rather than merely self-predictable: term 1 alone is
   satisfied by any quantity the dynamics model can extrapolate, including a
   collapsed one.

The reference's third term (``lambda_ce``, cross-entropy on the next-next token)
is zero in every shipped configuration and is deliberately absent here.

Structure follows the reference -- normalized concatenation, two hidden layers,
residual delta -- while the primitives are the house ones: `RMSNorm` and
`ReluSquared` from `kaggriculture.model` rather than LayerNorm and GELU, fp32
parameters, and no dropout (this codebase has none anywhere).

The module is training-only, and deliberately *not* an actor submodule: league
snapshots, inference bundles and frozen-ensemble stacks all consume the actor's
state dict whole, and p_psi has no business in any of them.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    QUANTIFIED_MARKET_KINDS,
)
from kaggriculture.model import ReluSquared, RMSNorm, factored_quantity_logits
from kaggriculture.policy import mask_logits


class LatentDynamics(nn.Module):
    """p_psi: predict the next-step belief latent from (h_t, executed action).

    ``belief + mlp(norm(cat[action, belief]))``, following the reference's
    `NextLatDynamicsModel`: the concatenation is normalized before the MLP mixes
    the two sources, and the MLP produces a *delta* rather than the next latent
    itself. Both details matter. Without the norm the action embedding's scale
    (which tracks how busy the turn was) sets the mixing weight; without the
    residual the predictor must reconstruct the whole latent from scratch at
    every step, which is a far harder regression than the small step-to-step
    change the environment actually makes.

    Hidden width is 4x model_dim, the house feed-forward multiplier, over the
    2x model_dim concatenation.
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("latent dynamics model width must be positive")
        self.model_dim = model_dim
        self.unit_action = nn.Embedding(N_UNIT_ACTIONS, model_dim)
        self.market_kind = nn.Embedding(N_MARKET_KINDS, model_dim)
        self.market_quantity = nn.Embedding(N_QUANTITIES, model_dim)
        # The joint action sums a variable number of units and orders, so its raw
        # magnitude tracks how busy the turn was rather than what was done;
        # normalize before the projection mixes the factors.
        self.action_norm = RMSNorm(model_dim)
        self.action_projection = nn.Linear(model_dim, model_dim)
        self.transition_norm = RMSNorm(2 * model_dim)
        self.predictor = nn.Sequential(
            nn.Linear(2 * model_dim, 4 * model_dim),
            ReluSquared(),
            nn.Linear(4 * model_dim, 4 * model_dim),
            ReluSquared(),
            nn.Linear(4 * model_dim, model_dim),
        )
        # Only quantified order kinds carry a quantity decision at all, so the
        # quantity factor gates exactly those. The table is derived here rather
        # than passed in because it is a property of the action space, not of a
        # batch: `market_quantity_active` in the staged factors is this same
        # predicate applied to `market_kinds`.
        quantified = torch.zeros(N_MARKET_KINDS, dtype=torch.bool)
        quantified[list(QUANTIFIED_MARKET_KINDS)] = True
        self.register_buffer("quantified_kinds", quantified, persistent=False)

    def embed_action(
        self,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
    ) -> Tensor:
        """Embed one executed joint action through the policy's own factors.

        NEXTLAT_AUX_PLAN.md's factorization: the sum of per-unit action
        embeddings plus the sum of per-order kind embeddings gated by the
        embedding of the quantity bin they were filled at, projected to
        model_dim. The sums are over slots, so the joint action is a multiset --
        slot identity is already in the belief this conditions.

        No active masks are needed and none are accepted: an inactive unit slot
        carries `UnitAction.PASS` and an inactive order slot carries
        `MarketKind.STOP`, which is what the demonstration projection writes
        (`demonstrations.py`) and the only action either slot could legally take.
        The canonical no-op is the honest embedding of "nothing happened here".
        """
        units = self.unit_action(unit_actions).sum(dim=-2)
        kinds = self.market_kind(market_kinds)
        quantity = torch.where(
            self.quantified_kinds[market_kinds].unsqueeze(-1),
            self.market_quantity(market_quantities),
            torch.ones((), dtype=kinds.dtype, device=kinds.device),
        )
        markets = (kinds * quantity).sum(dim=-2)
        return self.action_projection(self.action_norm(units + markets))

    def forward(
        self,
        belief: Tensor,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
    ) -> Tensor:
        """Predict ``h_hat_{t+1}`` for every row, shaped ``(rows, model_dim)``.

        Runs in fp32 regardless of the caller's autocast state. The regression
        target is fp32 by construction and the loss reduces in fp32 anyway, so
        pinning the precision here costs one cast on a tiny module and removes a
        silent dependency on the caller's autocast state.
        """
        if belief.ndim != 2:
            raise ValueError("belief must be one vector per row")
        if belief.shape[-1] != self.model_dim:
            raise ValueError("belief width does not match the dynamics model width")
        with torch.autocast(device_type=belief.device.type, enabled=False):
            belief = belief.float()
            action = self.embed_action(unit_actions, market_kinds, market_quantities)
            transition = self.transition_norm(torch.cat((action.float(), belief), dim=-1))
            return belief + self.predictor(transition)


def latent_dynamics_loss(predicted: Tensor, target: Tensor, eligible: Tensor) -> Tensor:
    """SmoothL1 regression onto the stop-gradient successor belief.

    The denominator is the masked *element* count, matching the reference's
    division by masked ``B * T * n_embd``: the loss is the mean per-coordinate
    error over eligible rows, so masking rows out does not shrink it. ``eligible``
    zeroes rows whose successor is not the next row -- the last step of an
    episode, and any row whose successor fell outside the minibatch.
    """
    if predicted.shape != target.shape:
        raise ValueError("predicted and target beliefs must have the same shape")
    if eligible.shape != predicted.shape[:1]:
        raise ValueError("eligibility mask must have one entry per row")
    weights = eligible.to(torch.float32).unsqueeze(-1)
    errors = F.smooth_l1_loss(predicted.float(), target.detach().float(), reduction="none")
    elements = weights.sum() * predicted.shape[-1]
    return (errors * weights).sum() / elements.clamp_min(1.0)


class BeliefDecode(NamedTuple):
    """One belief's action distribution parameters, mirroring `ActorOutput`."""

    unit_logits: Tensor
    market_kind_logits: Tensor
    market_quantity_context: Tensor


class _FrozenEmbedding(NamedTuple):
    """An `nn.Embedding`'s interface over a detached weight.

    `factored_quantity_logits` reads both ``.weight`` and ``__call__``, so this is
    what lets the quantity decode reuse the policy's own scoring function without
    the gradient reaching the embeddings it scores with.
    """

    weight: Tensor

    def __call__(self, index: Tensor) -> Tensor:
        return F.embedding(index, self.weight)


def _frozen_linear(module: nn.Linear, inputs: Tensor) -> Tensor:
    bias = None if module.bias is None else module.bias.detach()
    return F.linear(inputs, module.weight.detach(), bias)


def _frozen_norm(module: RMSNorm, inputs: Tensor) -> Tensor:
    return F.rms_norm(inputs, module.normalized_shape, module.weight.detach(), module.eps)


class DecodeHeads(NamedTuple):
    """References to the actor's own action heads, used with detached weights.

    Every actor family builds these through `initialize_policy_heads` and so
    shares the attribute names `from_actor` reads.
    """

    unit_norm: RMSNorm
    unit_projection: nn.Linear
    market_norm: RMSNorm
    market_kind: nn.Linear
    market_quantity_context: nn.Linear
    market_quantity_kind_gate: nn.Embedding
    market_quantity_value: nn.Embedding
    market_quantity_bias: Tensor
    quantity_rank: int

    @classmethod
    def from_actor(cls, actor: nn.Module) -> DecodeHeads:
        return cls(
            unit_norm=actor.unit_head[0],
            unit_projection=actor.unit_head[-1],
            market_norm=actor.market_norm,
            market_kind=actor.market_kind,
            market_quantity_context=actor.market_quantity_context,
            market_quantity_kind_gate=actor.market_quantity_kind_gate,
            market_quantity_value=actor.market_quantity_value,
            market_quantity_bias=actor.market_quantity_bias,
            quantity_rank=actor.config.quantity_rank,
        )

    def decode(self, belief: Tensor) -> BeliefDecode:
        """Score one belief per row through the heads, with the weights detached.

        The reference's `F.linear(pred, lm_head.weight.detach())`, generalized to
        three factored heads. Detached in both roles it is used for: as the
        student the gradient must reach the predicted latent but never the heads,
        and as the teacher nothing should be reached at all.
        """
        with torch.autocast(device_type=belief.device.type, enabled=False):
            belief = belief.float()
            market = _frozen_norm(self.market_norm, belief)
            return BeliefDecode(
                unit_logits=_frozen_linear(
                    self.unit_projection, _frozen_norm(self.unit_norm, belief)
                ),
                market_kind_logits=_frozen_linear(self.market_kind, market),
                market_quantity_context=_frozen_linear(self.market_quantity_context, market),
            )

    def quantity_logits(self, quantity_context: Tensor, market_kinds: Tensor) -> Tensor:
        """`FarmActor.quantity_logits` with every head weight detached."""
        return factored_quantity_logits(
            quantity_context,
            market_kinds,
            _FrozenEmbedding(self.market_quantity_kind_gate.weight.detach()),
            _FrozenEmbedding(self.market_quantity_value.weight.detach()),
            self.market_quantity_bias.detach(),
            self.quantity_rank,
        )


class DecodeMasks(NamedTuple):
    """The legality masks and slot-active flags the policy itself uses.

    Field names match the staged minibatch keys so a caller can hand over its own
    factors verbatim. ``market_kinds`` is not a mask: it is the selected order
    kind the quantity head conditions on, exactly as `quantity_logits` receives
    it in the policy path.
    """

    unit_masks: Tensor
    market_kind_masks: Tensor
    market_quantity_masks: Tensor
    unit_active: Tensor
    market_active: Tensor
    market_quantity_active: Tensor
    market_kinds: Tensor

    def rows(self, start: int, stop: int) -> DecodeMasks:
        return DecodeMasks(*(field[start:stop] for field in self))


def _decision_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    mask: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    """Masked KL(teacher || student) summed over decisions, and its weight sum.

    Masks are not validated: an inactive order slot carries an all-false mask,
    which the policy path also tolerates (`train_bc._clone_loss` passes
    ``validate_masks=False``). Such a slot decodes to the same uniform
    distribution on both sides, contributes exactly zero, and is excluded by
    ``weight`` regardless.
    """
    student = mask_logits(student_logits, mask, validate=False).log_softmax(dim=-1)
    teacher = mask_logits(teacher_logits, mask, validate=False).log_softmax(dim=-1)
    pointwise = F.kl_div(student, teacher, log_target=True, reduction="none")
    return (pointwise.sum(dim=-1) * weight).sum(), weight.sum()


def latent_decode_kl(
    predicted: Tensor,
    teacher_unit_logits: Tensor,
    teacher_kind_logits: Tensor,
    teacher_quantity_context: Tensor,
    heads: DecodeHeads,
    masks: DecodeMasks,
    eligible: Tensor,
) -> Tensor:
    """KL(teacher || student) between the decodes of the true and predicted latent.

    The reference's ``lambda_kl`` term. The teacher arguments are the decode of
    the *true* next belief -- `DecodeHeads.decode` on the stop-gradient target --
    and the student is the decode of the prediction; they are detached and share
    the head weights, so the term is exactly zero when the prediction is exact
    and can only be reduced by improving the prediction, never by moving a head.

    All three action factors participate, each under the same legality mask the
    policy applies, and each masked distribution is summed over its categorical
    and averaged over eligible decisions. Decisions are pooled across factors
    the same way the clone loss pools its log-likelihoods: one mean over the
    concatenated active components, so a factor's weight is its decision count.
    """
    if predicted.ndim != 2:
        raise ValueError("predicted belief must be one vector per row")
    if eligible.shape != predicted.shape[:1]:
        raise ValueError("eligibility mask must have one entry per row")
    student = heads.decode(predicted)
    row = eligible.bool().unsqueeze(-1)
    # The heads score one distribution per row; the policy asks the same question
    # once per slot under a per-slot mask, so the row's decode is broadcast and it
    # is the masks that make the slots differ.
    unit_slots = masks.unit_masks.shape[-2]
    order_slots = masks.market_kind_masks.shape[-2]
    unit_kl, unit_weight = _decision_kl(
        student.unit_logits.unsqueeze(-2).expand(-1, unit_slots, -1),
        teacher_unit_logits.unsqueeze(-2).expand(-1, unit_slots, -1),
        masks.unit_masks,
        (row & masks.unit_active).float(),
    )
    kind_kl, kind_weight = _decision_kl(
        student.market_kind_logits.unsqueeze(-2).expand(-1, order_slots, -1),
        teacher_kind_logits.unsqueeze(-2).expand(-1, order_slots, -1),
        masks.market_kind_masks,
        (row & masks.market_active).float(),
    )
    quantity_kl, quantity_weight = _decision_kl(
        heads.quantity_logits(
            student.market_quantity_context.unsqueeze(-2).expand(-1, order_slots, -1),
            masks.market_kinds,
        ),
        heads.quantity_logits(
            teacher_quantity_context.unsqueeze(-2).expand(-1, order_slots, -1),
            masks.market_kinds,
        ),
        masks.market_quantity_masks,
        (row & masks.market_quantity_active).float(),
    )
    decisions = unit_weight + kind_weight + quantity_weight
    return (unit_kl + kind_kl + quantity_kl) / decisions.clamp_min(1.0)


def consecutive_rows(episode_index: Tensor, step: Tensor) -> Tensor:
    """Rows whose successor step is the very next row of the minibatch.

    The pairing contract in one place: staged rows carry an episode index and a
    step, and the sampler lays consecutive steps of one episode-seat down in
    order, so row j's successor is row j+1 exactly when they agree on the episode
    and their steps are adjacent. The last row of a minibatch is never eligible,
    having no successor to compare against.
    """
    if episode_index.ndim != 1 or step.shape != episode_index.shape:
        raise ValueError("episode index and step must be one flat entry per row")
    if episode_index.numel() == 0:
        return torch.zeros_like(episode_index, dtype=torch.bool)
    paired = (episode_index[:-1] == episode_index[1:]) & (step[1:] == step[:-1] + 1)
    return torch.cat((paired, paired.new_zeros(1)))


class DecodeContext(NamedTuple):
    """Everything `latent_decode_kl` needs beyond the beliefs themselves."""

    heads: DecodeHeads
    masks: DecodeMasks


class LatentHorizonLoss(NamedTuple):
    """The two NextLat terms, each already averaged over the horizon."""

    dynamics: Tensor
    decode: Tensor
    # Eligible rows per unrolled step, for journaling how much of the minibatch
    # each step of the horizon actually supervises. A tensor rather than a tuple
    # of ints so reading it never synchronizes the device.
    eligible: Tensor


def latent_horizon_loss(
    dynamics: LatentDynamics,
    belief: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    episode_index: Tensor,
    step: Tensor,
    *,
    horizon: int = 1,
    decode: DecodeContext | None = None,
) -> LatentHorizonLoss:
    """Unroll p_psi for ``horizon`` steps, averaging both terms over the horizon.

    The reference's ``mtp_horizon`` loop: the prediction is fed back in as the
    next step's input, conditioned on the action executed at that step, and
    regressed onto the true belief that many rows later. Each step's terms are
    accumulated and divided by ``horizon``, so ``horizon=1`` is bit-for-bit the
    single-step call (a division by one is exact).

    Eligibility tightens by one row per extra step: predicting k steps ahead
    needs k consecutive pairs, so an episode-seat run shorter than k+1 rows
    contributes nothing to step k. Steps with no eligible row contribute zero to
    the sum and still count in the denominator, again as the reference does.

    ``decode=None`` computes the dynamics term alone and reports zero for the
    decode term, which is the reference's ``lambda_kl = 0`` configuration; the
    decode is by far the more expensive of the two and there is no reason to run
    it for a coefficient of zero.
    """
    if horizon < 1:
        raise ValueError("latent horizon must be at least one step")
    rows = belief.shape[0]
    paired = consecutive_rows(episode_index, step)
    if paired.shape[0] != rows:
        raise ValueError("row metadata and beliefs must describe the same rows")
    dynamics_total = torch.zeros((), dtype=torch.float32, device=belief.device)
    decode_total = torch.zeros((), dtype=torch.float32, device=belief.device)
    eligible_counts = []
    predicted = belief
    chain = torch.ones(rows, dtype=torch.bool, device=belief.device)
    for offset in range(horizon):
        keep = max(rows - offset - 1, 0)
        stop = offset + keep
        predicted = dynamics(
            predicted[:keep],
            unit_actions[offset:stop],
            market_kinds[offset:stop],
            market_quantities[offset:stop],
        )
        chain = chain[:keep] & paired[offset:stop]
        target = belief[offset + 1 : stop + 1]
        dynamics_total = dynamics_total + latent_dynamics_loss(predicted, target, chain)
        if decode is not None:
            teacher = decode.heads.decode(target.detach())
            decode_total = decode_total + latent_decode_kl(
                predicted,
                teacher.unit_logits,
                teacher.market_kind_logits,
                teacher.market_quantity_context,
                decode.heads,
                decode.masks.rows(offset + 1, stop + 1),
                chain,
            )
        eligible_counts.append(chain.sum())
    return LatentHorizonLoss(
        dynamics=dynamics_total / horizon,
        decode=decode_total / horizon,
        eligible=torch.stack(eligible_counts),
    )


class BeliefSpread(NamedTuple):
    """How much the batch's beliefs differ from one another."""

    cosine_similarity: Tensor  # mean pairwise cosine; exactly 1 under collapse
    dispersion: Tensor  # mean squared deviation from the mean direction


def belief_spread(belief: Tensor) -> BeliefSpread:
    """Collapse diagnostics for a batch of belief latents.

    Representational collapse is the failure mode the stop-gradient exists to
    prevent, and it drives the dynamics loss to zero by making every state's
    latent identical. Both statistics detect exactly that, and both are
    reported because they have resolution in different places: the belief site
    is not mean-centered, so a healthy batch already sits near cosine 0.99 and
    the remaining approach to 1 is hard to read, while the complementary
    dispersion falls through orders of magnitude as the beliefs converge.
    """
    rows = belief.shape[0]
    if rows < 2:
        one = torch.ones((), device=belief.device, dtype=torch.float32)
        return BeliefSpread(one, torch.zeros_like(one))
    directions = F.normalize(belief.detach().float().flatten(1), dim=-1)
    # mean_i ||u_i - mean(u)||^2, which is 0 exactly when every direction
    # coincides and approaches 1 when they are mutually orthogonal. Measured
    # from the centered directions rather than from ||sum u||^2, whose closed
    # form is the difference of two quantities of order n^2 and reaches exactly
    # zero in fp32 three decades above the collapse this is watching for.
    centered = directions - directions.mean(dim=0)
    dispersion = centered.square().sum(dim=-1, dtype=torch.float64).mean()
    return BeliefSpread(
        # mean_{i != j} <u_i, u_j>, recovered from the dispersion by the exact
        # identity dispersion = 1 - (1 + (n - 1) * cosine) / n.
        cosine_similarity=((1.0 - dispersion) * rows - 1.0).div(rows - 1).float(),
        dispersion=dispersion.float(),
    )
