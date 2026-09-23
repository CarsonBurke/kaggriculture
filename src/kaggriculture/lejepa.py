"""LeJEPA world-model objective: attached next-step targets, SIGReg, and reward.

The auxiliary here is LeWorldModel (Maes, Le Lidec, Scieur, LeCun and Balestriero,
arXiv:2603.19312; reference checkout at ``../le-wm``), which is LeJEPA
(arXiv:2511.08544, ``../lejepa``) applied to transitions rather than to augmented
views. LeWM's whole claim is that a stable end-to-end world model needs exactly
two terms -- a next-embedding prediction loss and a regularizer forcing the
embeddings to be isotropic Gaussian -- with no EMA, no teacher, no stop-gradient,
no frozen pretrained encoder, and no masking or corruption of the input.

This transplant keeps that shape exactly:

* **Attached targets.** The regression target is the encoder's own embedding of
  ``o_{t+1}``, with its gradient live. Every other self-predictive objective in
  this repository (`latent_dynamics`, `structured_dynamics`, `actor_dynamics`)
  detaches that target, because an attached target is minimized by a constant
  encoder. LeWM's answer is that the anti-collapse job belongs to a distributional
  constraint, not to an asymmetry in the graph.
* **SIGReg.** Random unit directions are drawn through the embedding and each
  one-dimensional marginal is pushed onto ``N(0, 1)`` by an Epps-Pulley
  characteristic-function statistic. A collapsed embedding has a degenerate
  marginal in every direction and is maximally penalized.
* **Nothing is corrupted.** Observations enter the encoder exactly as the policy
  sees them. The only supervision signal is the passage of one step.
* **No temporal conditioning.** LeWM's predictor is autoregressive over a short
  frame history with per-frame position embeddings. This one is Markov: it reads
  the current step's embedding and the executed joint action, nothing else. That
  matches what the policy and the critic on this encoder are allowed to know --
  neither carries recurrent state across turns -- so a latent that needed a
  history to be predictable would be a latent no consumer here could use.

Two additions the setting forces:

* **Reward.** LeWM predicts observations only; this predicts the reward collected
  on the transition as well, which is the one part of the next step that the
  encoded observation cannot contain.
* **Structured, multi-token latents.** LeWM pools a frame to one CLS embedding.
  This environment's decisions are per unit and per market slot, so the embedding
  is per token, and SIGReg runs per latent group -- a farm tile and a market-order
  slot share neither support nor scale, and one normality test over their union is
  satisfiable by a mixture that is degenerate inside every component.

**One backbone, owned by this objective.** The encoder is shared by the actor and
the critic, and both put their own capacity *after* the world model rather than
inside it. The critic reads `JepaBelief.detach()`; the actor's heads read it
attached (`LejepaConfig.policy_shapes_backbone`), so the policy's loss shapes the
encoder beside this one. Pure LeJEPA -- a representation fixed by the
self-supervised objective alone -- is kept as the detached ablation, and it does
not survive sampling: it clones the teacher at argmax and goes bankrupt once a
sampled decision leaves the teacher's trajectory, because nothing in a
teacher-prediction objective says which of the off-trajectory facts a decision
needs. One encoder is still the right count; a second would be the same
observation fitted twice.

What this does *not* defend against, stated plainly because the telemetry exists
to measure it: a one-step attached target is also minimized by an encoder that is
constant *along a trajectory* while varying across the batch, and SIGReg cannot
see that -- across the batch such an embedding is still perfectly normal, so both
the statistic and `dispersion` read clean while the backbone both towers share is
destroyed. The defenses are the reward term and the two controls journaled every
wave (`JepaPersistenceControl`, `JepaShuffledControl`). A prediction loss that
matches persistence, or that survives shuffling the actions across rows, is
measuring nothing -- though persistence alone cannot separate the two, because a
trajectory-constant encoder sends the baseline to zero along with the loss. The
column that does is `motion`: the RMS displacement of the *latents* -- the trunk
output the policy and the value function read, not the projected embedding, since
the projector is discarded after training -- between a row and its successor,
relative to their own scale and over supervised tokens only. That number going to
zero *is* the failure, and it is the one to cull a run on -- by hand, since
nothing gates a run on it automatically. The policy
gradient that also reaches the encoder is a third defense, but a weak one: it
constrains only what the decisions read, so `motion` stays the number to watch.

The module is training-only. Nothing here is an actor submodule: league snapshots,
inference bundles and frozen-ensemble stacks consume the actor's state dict whole
and have no business carrying a predictor or a projector the deployed model never
evaluates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from kaggriculture.model import Linear, ReluSquared, RMSNorm
from kaggriculture.structured import (
    Attention,
    FeedForward,
    GatedResidual,
    JepaBelief,
    StructuredInputs,
)
from kaggriculture.structured_dynamics import (
    StructuredActionEncoder,
    StructuredHorizonPlan,
    _eligible_rms_ratio,
    _target_index,
    _tile_changes,
)
from kaggriculture.tokens import TILE_COUNT

#: Upper quadrature limit of the Epps-Pulley statistic and the knots across it.
#: Both are the reference's (`../le-wm/module.py`, `../lejepa/MINIMAL.md`): the
#: Gaussian window ``exp(-t^2/2)`` has spent 99.7% of its mass below ``t = 3``, and
#: seventeen knots resolve the oscillations the window admits there.
SIGREG_T_MAX = 3.0
SIGREG_KNOTS = 17

#: Element budget for one chunk of the Epps-Pulley intermediate. The intermediate
#: is `[chunk, tokens, slices, knots]` of fp32, so sixteen million elements is sixty-four
#: megabytes -- small enough to stay resident beside a production minibatch's
#: trunk activations, large enough that the chunk loop does not become a launch
#: queue.
SIGREG_CHUNK_ELEMENTS = 1 << 24

#: Latent groups the objective regularizes and predicts, in belief order: two
#: decision groups the policy heads read, and two observation groups that make
#: this a world model rather than a policy-readout regularizer. One tuple and
#: not two, because one backbone serves both the actor and the critic.
JEPA_GROUPS = ("unit", "market", "economy", "tile")


class SIGReg(nn.Module):
    """Sliced Epps-Pulley normality statistic, weighted over the sample axis.

    ``forward`` consumes a ``[samples, features]`` or ``[samples, tokens, features]``
    embedding and a ``[features, slices]`` matrix of unit-norm directions.
    Projecting an isotropic standard Gaussian onto a unit direction gives exactly
    ``N(0, 1)`` whatever the feature count, which is why one fixed reference
    characteristic function ``exp(-t^2/2)`` serves every slice.

    The reduction runs over the sample axis alone and the statistic is averaged
    over token positions -- ``../le-wm/module.py``'s ``.mean(-3)``, which reduces
    the batch axis and leaves every position its own test. Flattening the token
    axis into the sample axis instead would test only the union, and a union of
    degenerate components is not degenerate: an encoder that gave every farm tile
    a distinct constant -- which `TileEmbedder.slot_embedding` alone can express,
    and which makes the prediction term exactly zero -- passes a pooled test at a
    tenth the penalty of real collapse. Per position it does not, because the
    population behind each position is then that slot across the batch, and a
    slot that is constant across the batch is an atom. Positions no row supervises
    are dropped rather than scored against an empty sample.

    The one deliberate departure from the reference is its ``* n`` scaling, which
    is dropped. That factor is what turns the quantity into a calibrated test
    statistic -- ``O(1)`` under the null whatever the sample count -- and it is
    right for a hypothesis test. It is wrong here, and not harmlessly so. It
    cancels exactly out of the per-sample gradient, so what it actually scales is
    the ratio between this term and the prediction term, which is a mean and whose
    per-sample gradient therefore falls like ``1/n``. Keeping it would make the
    trade-off coefficient a function of the minibatch size, the token count per
    group, and -- through the mask -- how many units happen to be alive. The
    reference can afford that because its ``n`` is a fixed 128 with one token per
    sample; this trainer's is tens of thousands and differs group by group. So the
    weighted quadrature error is returned as a mean over samples, the coefficient
    means the same thing in every configuration, and the reference's statistic is
    this value times the sample count whenever anyone wants it back.

    Positions are then combined in proportion to their own support rather than
    equally, and that is what makes the two paragraphs above compatible. Dropping
    ``* n`` is only safe while ``n`` is common to everything being compared; the
    mask makes it differ *within* one call, and a position's statistic falls like
    ``1 / n_position``, so a unit slot alive in five rows scores like a collapsed
    embedding on sampling noise alone. Measured at 1024 rows over sixteen
    positions: hold one position at five supporting rows and an equal-weight mean
    reads 0.014 against a null of 0.0010 and total collapse at 0.92 -- seven
    percent of the way to collapse with nothing having changed. Worse than the
    magnitude is the direction, because the single-sample Epps-Pulley error is
    minimized at the origin: the anti-collapse term would pull hardest on exactly
    the slots it has least evidence about, and every unit the agent learns to hire
    would trip it afresh. Weighting by ``count`` is the reference's ``* n``
    restored per position and divided out once at the end, so each *row* carries
    the same gradient wherever it sits, the aggregate is flat in occupancy, and
    under a uniform mask it is identical to the plain mean it replaces.

    In eager the samples are chunked and each chunk's partial sums are
    rematerialized in backward: the reference's single-shot form holds a
    ``[samples, tokens, slices, knots]`` intermediate, which at this trainer's
    sample counts is measured in gigabytes. Under `torch.compile` the loop is
    skipped entirely -- Inductor's min-cut partitioner already decides what to
    keep and what to recompute across that intermediate, and a Python-level chunk
    loop only multiplies the traced subgraphs (one checkpointed higher-order op
    per chunk per group) and the compile time it costs to reach the same answer.

    Weighted because inactive unit slots carry an exactly zero latent, and an atom
    at the origin is a departure from normality the encoder cannot fix and should
    not be asked to.
    """

    def __init__(self, knots: int = SIGREG_KNOTS, t_max: float = SIGREG_T_MAX) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError("SIGReg needs at least two quadrature knots")
        if not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("SIGReg quadrature limit must be finite and positive")
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        step = t_max / (knots - 1)
        # Trapezoid over [0, t_max], doubled for the mirrored half of the axis:
        # the empirical characteristic function of a real sample is conjugate
        # symmetric, so integrating one half and doubling is exact, not an
        # approximation, and it buys twice the quadrature resolution for free.
        weights = torch.full((knots,), 2 * step, dtype=torch.float32)
        weights[0] = weights[-1] = step
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("reference", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def _chunk_sums(self, embeddings: Tensor, directions: Tensor, weight: Tensor) -> Tensor:
        angles = (embeddings @ directions).unsqueeze(-1) * self.t
        scaled = weight.unsqueeze(-1).unsqueeze(-1)
        return torch.stack(((angles.cos() * scaled).sum(0), (angles.sin() * scaled).sum(0)))

    def forward(
        self, embeddings: Tensor, directions: Tensor, weight: Tensor | None = None
    ) -> Tensor:
        if embeddings.ndim not in (2, 3):
            raise ValueError("SIGReg consumes [samples, features] or [samples, tokens, features]")
        if directions.ndim != 2 or directions.shape[0] != embeddings.shape[-1]:
            raise ValueError("slice directions must be [features, slices]")
        embeddings = embeddings.float()
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(1)
        directions = directions.float()
        samples, tokens = embeddings.shape[0], embeddings.shape[1]
        slices, knots = directions.shape[1], self.t.numel()
        weight = (
            embeddings.new_ones((samples, tokens))
            if weight is None
            else weight.float().reshape(samples, tokens)
        )
        fused = torch.compiler.is_compiling()
        chunk = max(1, samples if fused else SIGREG_CHUNK_ELEMENTS // (tokens * slices * knots))
        total = torch.zeros(2, tokens, slices, knots, device=embeddings.device)
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            piece = (embeddings[start:stop], directions, weight[start:stop])
            total = total + (
                checkpoint(self._chunk_sums, *piece, use_reentrant=False)
                if torch.is_grad_enabled() and not fused
                else self._chunk_sums(*piece)
            )
        count = weight.sum(0)
        cosine, sine = (total / count.clamp_min(1.0).reshape(-1, 1, 1)).unbind(0)
        error = (cosine - self.reference).square() + sine.square()
        # Support-weighted, which drops the unsupported positions as the k == 0
        # case of the same rule rather than as a second one.
        return ((error @ self.weights).mean(dim=-1) * count).sum() / count.sum().clamp_min(1.0)


def sigreg_directions(features: int, slices: int, generator: np.random.Generator) -> np.ndarray:
    """Unit-norm slice directions, drawn on the host from the auxiliary stream.

    A Gaussian column normalized to unit length is uniform on the sphere, which
    is what the sketching argument needs. Both references draw this inside the
    training step; here it is drawn outside and copied into a buffer, for two
    reasons. The update runs under `torch.compile`, and device RNG inside a
    compiled region is either a graph break or a captured state this trainer
    would then have to replay. And the trainer already owns an independent
    auxiliary generator whose stream is part of a wave's reproducible record, so
    drawing from it keeps the objective replayable from the run's seed alone.
    """
    if features < 1 or slices < 1:
        raise ValueError("slice directions need positive feature and slice counts")
    columns = generator.standard_normal((features, slices), dtype=np.float32)
    norms = np.linalg.norm(columns, axis=0, keepdims=True)
    return columns / np.maximum(norms, 1e-12)


def belief_groups(belief: JepaBelief, tile_index: Tensor) -> dict[str, Tensor]:
    """Name a belief's latent groups, with the tiles reduced to the shared sample."""
    return {
        "unit": belief.unit_decisions,
        "market": belief.market_decisions,
        "economy": belief.economy,
        "tile": belief.tiles.index_select(1, tile_index),
    }


class JepaMlp(nn.Module):
    """The reference's projector shape: one hidden layer, normalized, then out.

    ``../le-wm/module.py::MLP``, with two substitutions. The reference normalizes
    the hidden layer with ``BatchNorm1d``; this uses the house `RMSNorm` because a
    batch statistic computed over token slots that are sometimes masked is a
    statistic over padding, and because nothing else in this network carries
    running buffers. The reference's GELU becomes the house ReLU-squared for the
    same reason the rest of the trunk uses it.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input = Linear(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        self.activation = ReluSquared()
        self.output = Linear(hidden_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.activation(self.norm(self.input(inputs))))


class JepaProjectors(nn.Module):
    """One projector per latent group, trunk latent width in and out.

    The embedding is what SIGReg regularizes and what the predictor predicts; the
    policy and value heads never see it. That split is the reference's and it is
    the reason the deployed artifact carries none of this: the projector is
    discarded after training exactly as in LeJEPA's linear-probe protocol.

    Width is unchanged through the projector, as in `../le-wm`, where the
    projector is an MLP from the encoder's width back to it. The projector's job
    is to give the constraint and the prediction a space of their own to be
    satisfied in, not to compress: a bottleneck here would let the encoder park
    whatever the objective dislikes in the discarded complement and report a
    well-shaped embedding over a badly shaped latent. Per group rather than
    shared, because a market-order slot and a farm tile share no support and a
    projector that had to serve both would spend its capacity reconciling them.
    """

    def __init__(self, model_dim: int, hidden_dim: int, groups: tuple[str, ...]) -> None:
        super().__init__()
        self.groups = groups
        # The belief hands over the trunk's residual stream, whose scale is
        # whatever the encoder's depth left behind and differs group by group.
        # Normalizing on the way in gives the statistic and the predictor a
        # common footing, and takes the place of the `BatchNorm1d` both
        # references put inside this MLP.
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(RMSNorm(model_dim), JepaMlp(model_dim, hidden_dim, model_dim))
                for name in groups
            }
        )

    def forward(self, latents: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: self.heads[name](value) for name, value in latents.items()}


class JepaPredictor(nn.Module):
    """p_psi: one Markov attention round over the embeddings, reading the action.

    Queries are the embedded tokens being predicted; the context is those tokens
    *plus* the executed joint action's per-slot tokens. Two properties follow, and
    both are why this is an attention round rather than the per-token residual MLP
    the detached NextLat transplant uses:

    * a tile can attend to the action of the unit standing on it, which is how
      nearly every observable transition in this environment is actually caused;
    * one token's prediction can depend on the others, so the model can express
      "this unit harvested, so that tile empties and that product token gains
      stock" instead of extrapolating every slot in isolation.

    The reference conditions a stack of AdaLN-zero blocks on a single pooled action
    vector per frame. A pooled vector cannot say *which* of sixteen units did what,
    and this action is a joint action over sixteen unit slots and ten order slots,
    so the conditioning is per-slot attention context instead. No position
    embedding and no frame history: the prediction is Markov by construction.

    ``prediction`` composes the reference's predictor and its ``pred_proj`` into
    one residual: ``emb + pred_proj(round(emb, action))`` with the output layer
    zero-initialized. At initialization the predictor is the identity, which is the
    better of the two constant predictions available -- a farm is overwhelmingly
    unchanged from one step to the next -- so the objective's early gradient
    carries transition signal rather than the noise of an untrained readout.

    The reward head deliberately sits outside the attention round. Under this
    trainer's production terminal-outcome rewards the entire signal lives on the
    final step of an episode, and that step is never a transition source: it has no
    successor inside the rollout. A reward head placed behind the round would be
    trained on an all-zero target. This one reads pooled embeddings and the pooled
    action directly, so the caller evaluates it on every staged row while the
    expensive round stays on the rows that have a successor.
    """

    def __init__(self, config) -> None:
        super().__init__()
        width = config.model_dim
        hidden = config.jepa_hidden_dim
        self.action = StructuredActionEncoder(width)
        self.action_norm = RMSNorm(width)
        self.token_norm = RMSNorm(width)
        self.attention = Attention(config)
        self.attention_gate = GatedResidual(width)
        self.ffn_norm = RMSNorm(width)
        self.ffn = FeedForward(config)
        self.ffn_gate = GatedResidual(width)
        self.output_norm = RMSNorm(width)
        self.output = JepaMlp(width, hidden, width)
        nn.init.zeros_(self.output.output.weight)
        nn.init.zeros_(self.output.output.bias)
        self.reward_body = JepaMlp(2 * width, hidden, width)
        # Named for `optim.route_parameters`: a one-row readout is a head, not a
        # feature-to-feature map, and NorMuon's orthogonalization would simply
        # normalize its only row. Same treatment as `value_head`.
        self.reward_head = Linear(width, 1)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

    def embed_action(
        self,
        unit_actions: Tensor,
        market_kinds: Tensor,
        market_quantities: Tensor,
        unit_categorical: Tensor,
        unit_active: Tensor,
    ) -> Tensor:
        unit_tokens, market_tokens = self.action(
            unit_actions, market_kinds, market_quantities, unit_categorical, unit_active
        )
        return self.action_norm(torch.cat((unit_tokens, market_tokens), dim=1))

    @staticmethod
    def _pool(tokens: Tensor, token_valid: Tensor) -> Tensor:
        """Mean over valid tokens: a row with few live units is not a dim row."""
        weight = token_valid.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def prediction(self, tokens: Tensor, actions: Tensor, token_valid: Tensor) -> Tensor:
        normalized = self.token_norm(tokens)
        context = torch.cat((normalized, actions.to(tokens.dtype)), dim=1)
        context_valid = torch.cat(
            (token_valid, torch.ones_like(actions[..., 0], dtype=torch.bool)), dim=1
        )
        hidden = self.attention_gate(
            tokens, self.attention(normalized, context, context_valid=context_valid)
        )
        hidden = self.ffn_gate(hidden, self.ffn(self.ffn_norm(hidden)))
        update = self.output(self.output_norm(hidden))
        return tokens + torch.where(token_valid.unsqueeze(-1), update, 0.0)

    def reward_prediction(
        self,
        tokens: Tensor | Sequence[Tensor],
        actions: Tensor,
        token_valid: Tensor | Sequence[Tensor],
    ) -> Tensor:
        """The pooled state and the pooled action to the transition's reward.

        `tokens` and `token_valid` may be handed over per latent group. The head
        reads nothing but a masked mean over the token axis, and that mean is the
        same whether the groups are concatenated first or pooled and recombined --
        so taking them per group spares the objective a ``[rows, tokens, width]``
        concatenation it would otherwise materialize, on every staged row, and
        retain for backward. `token_norm` is per token, so it too commutes.
        """
        pieces = [tokens] if isinstance(tokens, Tensor) else list(tokens)
        masks = [token_valid] if isinstance(token_valid, Tensor) else list(token_valid)
        totals, counts = [], []
        for piece, mask in zip(pieces, masks, strict=True):
            weight = mask.unsqueeze(-1).to(piece.dtype)
            totals.append((self.token_norm(piece) * weight).sum(dim=1))
            counts.append(weight.sum(dim=1))
        state = torch.stack(totals).sum(0) / torch.stack(counts).sum(0).clamp_min(1.0)
        action_valid = torch.ones_like(actions[..., 0], dtype=torch.bool)
        acted = self._pool(actions.to(state.dtype), action_valid)
        return self.reward_head(self.reward_body(torch.cat((state, acted), dim=-1))).squeeze(-1)


class JepaObjective(nn.Module):
    """The training-only half of the world model: projectors, predictor, statistic.

    One module because the three step together under one optimizer, and because
    all three are equally disposable: the deployed artifact reads ``model_dim``
    trunk latents and evaluates none of this.

    That optimizer additionally owns the shared backbone, which is the one thing
    here that is *not* disposable. The encoder's gradient is this objective's plus
    the policy's (the critic's tower reads a detached belief), and stepping it with
    the projector and predictor is stepping one model: splitting them across two
    optimizers would only give one objective two learning rates.

    The minibatch's slice directions and its tile sample live in non-persistent
    buffers that `refresh_slices` overwrites in place between minibatches. Buffer
    identity, shape and dtype never change, so the compiled update graph is traced
    once and no recompilation follows a refresh; only the values move. Drawing
    them here rather than inside the step is what keeps the compiled region free
    of RNG -- see `sigreg_directions`.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.groups = JEPA_GROUPS
        self.model_dim = config.model_dim
        self.slices = config.jepa_slices
        self.tile_samples = config.jepa_tile_samples
        self.sigreg_rows = config.jepa_sigreg_rows
        self.tile_tokens = 2 * TILE_COUNT
        if not 0 < self.tile_samples <= self.tile_tokens:
            raise ValueError("the tile sample must name a subset of the encoded tiles")
        if self.slices < 1 or self.sigreg_rows < 1:
            raise ValueError("slice count and SIGReg row budget must be positive")
        self.projectors = JepaProjectors(config.model_dim, config.jepa_hidden_dim, self.groups)
        self.predictor = JepaPredictor(config)
        self.statistic = SIGReg()
        # A never-refreshed objective is still a valid one: seed the buffers from
        # a fixed stream so construction, tests and a first traced minibatch all
        # see well-formed directions rather than zeros the statistic cannot use.
        seed = np.random.default_rng(0)
        self.register_buffer(
            "tile_index",
            torch.from_numpy(np.sort(seed.choice(self.tile_tokens, self.tile_samples, False))),
            persistent=False,
        )
        for name in self.groups:
            self.register_buffer(
                f"directions_{name}",
                torch.from_numpy(sigreg_directions(config.model_dim, self.slices, seed)),
                persistent=False,
            )

    @torch.no_grad()
    def refresh_slices(self, generator: np.random.Generator) -> None:
        """Redraw this minibatch's directions and tile sample into the buffers.

        The tile sample is shared across every row of the minibatch and across a
        source row and its successor: scoring tile slot 7 at ``t`` against tile
        slot 31 at ``t + 1`` would be a different objective in every minibatch,
        and a per-row gather over two hundred encoded tiles is the one part of
        this objective that would not otherwise be nearly free.

        Sampling tiles budgets what is *supervised*, never what is *encoded*. The
        trunk still reads every tile, and the sample rotates every minibatch, so
        one epoch covers the board many times over. That is the whole difference
        between this and the corruption the design deliberately excludes.
        """
        # Staged through pinned memory, as the trainer stages its horizon plans.
        # Out of pageable memory a "non-blocking" copy is staged by the host
        # through a driver bounce buffer instead of queued as a DMA, so the host
        # does the copy work that it should be spending enqueueing the next
        # forward. The caching host allocator keeps each pinned block alive
        # until its copy lands.
        pinned = self.tile_index.is_cuda

        def stage(array: np.ndarray) -> Tensor:
            host = torch.from_numpy(array)
            return host.pin_memory() if pinned else host

        index = np.sort(generator.choice(self.tile_tokens, self.tile_samples, replace=False))
        self.tile_index.copy_(stage(index), non_blocking=True)
        for name in self.groups:
            draw = sigreg_directions(self.model_dim, self.slices, generator)
            getattr(self, f"directions_{name}").copy_(stage(draw), non_blocking=True)

    def directions(self, group: str) -> Tensor:
        return getattr(self, f"directions_{group}")

    def project(self, latents: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.projectors(latents)

    def embed_action(self, *arguments: Tensor) -> Tensor:
        return self.predictor.embed_action(*arguments)

    def prediction(self, tokens: Tensor, actions: Tensor, token_valid: Tensor) -> Tensor:
        return self.predictor.prediction(tokens, actions, token_valid)

    def reward_prediction(self, tokens: Tensor, actions: Tensor, token_valid: Tensor) -> Tensor:
        return self.predictor.reward_prediction(tokens, actions, token_valid)


class JepaControl:
    """A metric-only transition sharing one objective's projectors and heads.

    Not an ``nn.Module``: it must never reach a parameter set, a state dict or a
    gradient-norm reduction, and it lives exactly as long as the one no-grad
    diagnostic call that builds it. It shares the objective's buffers so the
    control and the live objective are scored on the same slices and the same
    tiles -- a control drawn against different directions would not be a control.
    """

    def __init__(self, objective: JepaObjective) -> None:
        self.objective = objective
        self.groups = objective.groups
        self.sigreg_rows = objective.sigreg_rows
        self.statistic = objective.statistic
        self.tile_index = objective.tile_index

    def directions(self, group: str) -> Tensor:
        return self.objective.directions(group)

    def project(self, latents: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.objective.project(latents)

    def embed_action(self, *arguments: Tensor) -> Tensor:
        return self.objective.embed_action(*arguments)

    def reward_prediction(
        self,
        tokens: Tensor | Sequence[Tensor],
        actions: Tensor,
        token_valid: Tensor | Sequence[Tensor],
    ) -> Tensor:
        return self.objective.reward_prediction(tokens, actions, token_valid)


class JepaPersistenceControl(JepaControl):
    """The "nothing changed" baseline. A predictor that cannot beat it is idle."""

    def prediction(self, tokens: Tensor, actions: Tensor, token_valid: Tensor) -> Tensor:
        return tokens


class JepaShuffledControl(JepaControl):
    """The live predictor reading another trajectory's action.

    Action tokens roll whole, so a market kind keeps its quantity and a unit
    action keeps the unit identity it is scored against. Scoring as well as the
    live predictor means the transition is not action-conditioned and the
    prediction term is measuring persistence under another name.

    `stride` is why this is a control at all. The roll runs over the transition
    sources, and each contiguous same-trajectory run contributes
    `run_length - horizon` consecutive ones, so rolling by one would hand most
    sources the action their *own* trajectory took one step away --
    autocorrelated enough that a genuinely action-conditioned predictor would
    still score well and the control would read as passed. Rolling by the run
    length always leaves the run, and the runs are shuffled independently.
    """

    def __init__(self, objective: JepaObjective, stride: int = 1) -> None:
        super().__init__(objective)
        if stride < 1:
            raise ValueError("the shuffled control stride must be positive")
        self.stride = stride

    def prediction(self, tokens: Tensor, actions: Tensor, token_valid: Tensor) -> Tensor:
        return self.objective.prediction(tokens, actions.roll(self.stride, dims=0), token_valid)


class JepaTerms(NamedTuple):
    """World-model journal. `prediction`, `sigreg` and `reward` are the objective.

    Two columns are baselines rather than losses, and both need their comparison
    stated. `reward` is a mean square and `reward_scale` the target's RMS, so the
    constant-zero predictor's loss is exactly ``reward_scale ** 2`` -- which the
    zero-initialized head starts at, and which under terminal-outcome rewards is
    about ``1 / EPISODE_STEPS``. A small `reward` on its own says nothing, and at
    a zero reward coefficient both read zero together, which is how "not scored"
    is told apart from "scored perfectly". And `motion` is the encoder's own
    temporal displacement: see the module docstring, it is the column a
    collapsing run is culled on.
    """

    prediction: Tensor
    unit: Tensor
    market: Tensor
    economy: Tensor
    tile: Tensor
    tile_all: Tensor
    tile_changed: Tensor
    tile_unchanged: Tensor
    sigreg: Tensor
    sigreg_unit: Tensor
    sigreg_market: Tensor
    sigreg_economy: Tensor
    sigreg_tile: Tensor
    reward: Tensor
    reward_scale: Tensor
    dispersion: Tensor
    motion: Tensor
    residual_ratio: Tensor
    eligible: Tensor


JEPA_METRICS = JepaTerms._fields

#: Where a behavior-cloning artifact carries the `JepaObjective` its backbone was
#: trained beside. The projector and the predictor are one model with the
#: backbone, so a warm start resumes them rather than fitting fresh ones against
#: an encoder that was shaped through different ones.
JEPA_OBJECTIVE_ARTIFACT_KEY = "jepa_objective"


def load_artifact_objective(payload: Mapping[str, Any], objective: nn.Module | None) -> None:
    """Resume the objective a warm-start artifact's backbone was cloned beside.

    Every entry point that warm-starts from a clone goes through here, so a
    benchmark or a gate measures the run it stands for rather than a cloned
    backbone under a fresh projector. An artifact without one cannot warm-start
    a `JepaObjective`, and an objective in an artifact nothing here trains is a
    mismatched artifact rather than an extra to ignore.
    """
    stored = payload.get(JEPA_OBJECTIVE_ARTIFACT_KEY)
    if isinstance(objective, JepaObjective):
        if stored is None:
            raise ValueError(
                "a lejepa warm start resumes the LeJEPA objective its backbone was cloned "
                "beside, and this artifact carries none; re-clone it with train_bc.py"
            )
        objective.load_state_dict(stored)
    elif stored is not None:
        raise ValueError(
            "initial actor artifact carries a LeJEPA objective this run has no use for"
        )


def _group_valid(
    groups: dict[str, Tensor], inputs: StructuredInputs, unit_active: Tensor
) -> dict[str, Tensor]:
    """Per-token supervision masks.

    Only unit slots have one. An inactive unit carries an exactly zero latent, so
    including it would ask the predictor to reproduce padding and would hand
    SIGReg an atom at the origin that no encoder can move. Market order slots,
    economy tokens, tiles and the pooled valuation state are always real state --
    an empty market slot is a fact about the market, not an absent row.
    """
    valid = {
        name: torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
        for name, value in groups.items()
    }
    if "unit" in valid:
        valid["unit"] = unit_active
    return valid


def _weighted_mse(predicted: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Mean squared error over supervised tokens, in fp32.

    Squared error and not the SmoothL1 the detached auxiliaries use. SmoothL1 is
    there to stop a stop-gradient teacher's outliers from dominating; here the
    target carries gradient and is itself being shaped, so a robust loss would
    only be a licence for the pair to disagree on the tokens that matter most.
    This is `../le-wm`'s ``(pred_emb - tgt_emb).pow(2).mean()`` with the mean
    taken over supervised tokens.
    """
    error = (predicted.float() - target.float()).square().mean(dim=-1)
    weight = weight.float()
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def _token_rms_ratio(current: Tensor, reference: Tensor, weight: Tensor) -> Tensor:
    """Displacement from `reference` to `current`, relative to `reference`'s scale.

    Token-weighted, unlike `_eligible_rms_ratio`'s row weighting, and that is the
    point: a dead unit slot carries a constant latent, so it contributes nothing
    to the numerator and its own norm to the denominator. With fourteen of
    sixteen slots typically empty, a row-weighted version of this ratio reads
    mostly as unit occupancy, and occupancy moves between waves for reasons that
    have nothing to do with the encoder.
    """
    mask = weight.float().unsqueeze(-1)
    elements = (mask.sum() * current.shape[-1]).clamp_min(1.0)
    displacement = (((current - reference).square() * mask).sum() / elements).sqrt()
    scale = ((reference.square() * mask).sum() / elements).sqrt()
    return displacement / scale.clamp_min(1e-6)


def _embedding_dispersion(embeddings: Tensor, weight: Tensor) -> Tensor:
    """Per-coordinate standard deviation across the supervised population.

    SIGReg drives this to one. Journaled separately because it is the direct
    readout of the failure this objective is built to prevent: a number decaying
    toward zero is collapse, whatever the prediction loss says.
    """
    values = embeddings.detach().float().reshape(-1, embeddings.shape[-1])
    mask = weight.detach().float().reshape(-1, 1)
    count = mask.sum().clamp_min(1.0)
    mean = (values * mask).sum(dim=0) / count
    variance = ((values - mean).square() * mask).sum(dim=0) / count
    return variance.clamp_min(0.0).sqrt().mean()


def _tile_change_mask(
    inputs: StructuredInputs, source: Tensor, target: Tensor, tile_index: Tensor
) -> Tensor:
    """Exact per-tile transition mask over the minibatch's sampled slots.

    Columns before rows. Only `jepa_tile_samples` of the farm's two hundred tiles
    are supervised, so selecting them first makes both gathers narrow; the other
    order builds the full-width row gather and throws most of it away, once per
    horizon step.
    """
    categorical = inputs.tile_categorical.index_select(1, tile_index)
    continuous = inputs.tile_continuous.index_select(1, tile_index)
    return _tile_changes(
        categorical[source], categorical[target], continuous[source], continuous[target]
    )


def jepa_horizon_loss(
    objective: JepaObjective | JepaControl,
    belief: JepaBelief,
    inputs: StructuredInputs,
    factors: dict[str, Tensor],
    *,
    horizon: int,
    plan: StructuredHorizonPlan | None = None,
    sample_weight: Tensor | None = None,
    score_reward: bool = True,
) -> JepaTerms:
    """Score one minibatch under LeWM's two terms, plus the reward this env needs.

    Three deliberately different populations, because three different questions
    are being asked:

    * **SIGReg** runs over every staged row's own embedding (subject to the row
      budget). It is a constraint on the representation, not on transitions, so
      restricting it to rows that happen to have a successor would only shrink
      the sample for no reason.
    * **The transition** runs only on rows the plan admits as sources -- rows with
      a demonstrated successor inside the same episode. Everything else has no
      target.
    * **The reward** runs over every staged row. This is not symmetry-breaking
      laziness: under this trainer's production terminal-outcome rewards, the
      entire nonzero signal sits on each episode's final step, which by
      construction is never a transition source. Scoring reward only on source
      rows would train it exclusively on zeros.

    The projector runs once over the full staged batch and both the source and the
    target embeddings are row gathers of that single pass. With this trainer's
    contiguous-run minibatches a row and its successor are both already in the
    batch, so the entire world-model objective costs no additional encoder
    forward -- which is what makes it affordable at all.
    """
    if horizon < 1:
        raise ValueError("the LeJEPA horizon must be at least one step")
    if plan is not None and plan.eligible.shape[0] < horizon:
        raise ValueError("the staged plan does not cover the requested horizon")
    order = objective.groups
    latents = belief_groups(belief, objective.tile_index)
    valid = _group_valid(latents, inputs, inputs.unit_active)
    if sample_weight is not None:
        # `_fixed_minibatch_positions` wraps the epoch's final minibatch back to
        # the head of the ordering to keep compiled shapes fixed, and requires
        # its callers to zero-weight what it duplicated. Folding the row weight
        # into the token masks is how every population here inherits that: the
        # statistic, the transition and the reward all read these masks.
        row_valid = sample_weight.reshape(-1, 1) > 0.0
        valid = {name: value & row_valid for name, value in valid.items()}
    embeddings = objective.project(latents)

    rows = embeddings[order[0]].shape[0]
    budget = min(objective.sigreg_rows, rows)
    sigreg_terms: dict[str, Tensor] = {}
    dispersion = embeddings[order[0]].new_zeros((), dtype=torch.float32)
    for name in order:
        sampled = embeddings[name][:budget]
        weight = valid[name][:budget]
        sigreg_terms[name] = objective.statistic(sampled, objective.directions(name), weight)
        dispersion = dispersion + _embedding_dispersion(sampled, weight)
    sigreg = torch.stack([sigreg_terms[name] for name in order]).mean()
    dispersion = dispersion / len(order)

    steps = torch.arange(rows, device=inputs.unit_active.device)
    source = steps if plan is None else plan.indices[0]
    sizes = tuple(embeddings[name].shape[1] for name in order)

    def gather(index: Tensor) -> Tensor:
        return torch.cat([embeddings[name][index] for name in order], dim=1)

    def gather_latent(index: Tensor) -> Tensor:
        # Detached before the gather, so the `motion` diagnostic adds no node to
        # the backward graph -- only a narrow forward read of what already exists.
        return torch.cat([latents[name].detach()[index] for name in order], dim=1).float()

    def actions_at(index: Tensor) -> Tensor:
        return objective.embed_action(
            factors["unit_actions"][index],
            factors["market_kinds"][index],
            factors["market_quantities"][index],
            inputs.unit_categorical[index],
            inputs.unit_active[index],
        )

    # The reward head reads the whole staged batch, not the plan's source rows.
    if score_reward:
        reward_target = factors["rewards"].float()
        reward_weight = (
            torch.ones_like(reward_target) if sample_weight is None else sample_weight.float()
        )
        reward = _weighted_mse(
            objective.reward_prediction(
                [embeddings[name][steps] for name in order],
                actions_at(steps),
                [valid[name][steps] for name in order],
            ).unsqueeze(-1),
            reward_target.unsqueeze(-1),
            reward_weight,
        )
        # Weighted like `reward` is, so the baseline the `JepaTerms` docstring
        # says to compare it against is taken over the same rows the loss was.
        reward_scale = (
            (reward_target.detach().square() * reward_weight).sum()
            / reward_weight.sum().clamp_min(1.0)
        ).sqrt()
    else:
        # At a zero coefficient the term is a graph and a backward for a gradient
        # that is multiplied by zero; the head reads every staged row, so that is
        # the most expensive nothing in the objective. Behavior cloning has no
        # reward to read at all. The baseline is zeroed with the loss, because a
        # zero loss beside a live baseline reads as a perfect predictor and the
        # pair is the only thing that says which.
        reward = reward_scale = embeddings[order[0]].new_zeros((), dtype=torch.float32)

    predicted = gather(source)
    encoded = gather_latent(source)
    # Supervision follows the source row's occupancy: a slot that is empty at the
    # source has a zero latent to predict from, and a slot that fills later is a
    # spawn no Markov transition from that zero could have produced.
    token_valid = torch.cat([valid[name][source] for name in order], dim=1)
    zero = predicted.new_zeros((), dtype=torch.float32)
    group_sums = dict.fromkeys(order, zero)
    tile_sums = [zero, zero, zero]
    residual = motion = eligible_total = zero
    for offset in range(1, horizon + 1):
        action_index = (
            (steps + offset - 1).clamp_max(rows - 1) if plan is None else plan.indices[offset]
        )
        previous = predicted
        predicted = objective.prediction(predicted, actions_at(action_index), token_valid)
        if plan is None:
            target_index, eligible = _target_index(
                factors["episode_index"], factors["step"], offset
            )
        else:
            target_index = plan.indices[plan.eligible.shape[0] + offset]
            eligible = plan.eligible[offset - 1]
        residual = residual + _eligible_rms_ratio(predicted, previous, eligible)
        eligible_total = eligible_total + eligible.float().sum()
        target = gather(target_index)
        weight = token_valid & eligible.unsqueeze(-1)
        # How far the *encoder* moves between a source row and the successor it is
        # scored against, relative to its own scale. Measured on the latents the
        # policy and the value function read rather than on the projected
        # embeddings, because the projector is discarded after training and a
        # collapsed trunk behind an expansive projector is still a collapsed
        # trunk. Cumulative from the source, where `residual` beside it is
        # per-step, so the two coincide only at `horizon == 1`; averaged over the
        # horizon either way. This is the one number that sees the failure the
        # module docstring names: an encoder that is constant along a trajectory
        # drives the prediction term to zero, drags the persistence baseline down
        # with it so the two agree, and leaves SIGReg and the dispersion
        # untouched -- because across the batch the embedding is still perfectly
        # normal. Only this ratio collapses with it.
        motion = motion + _token_rms_ratio(gather_latent(target_index), encoded, weight)
        for name, piece, piece_target, piece_weight in zip(
            order,
            predicted.split(sizes, dim=1),
            target.split(sizes, dim=1),
            weight.split(sizes, dim=1),
            strict=True,
        ):
            if name != "tile":
                group_sums[name] = group_sums[name] + _weighted_mse(
                    piece, piece_target, piece_weight
                )
                continue
            changed = _tile_change_mask(inputs, source, target_index, objective.tile_index)
            tile_all = _weighted_mse(piece, piece_target, piece_weight)
            tile_changed = _weighted_mse(piece, piece_target, piece_weight & changed)
            tile_unchanged = _weighted_mse(piece, piece_target, piece_weight & ~changed)
            # Upweight the tiles that actually moved, exactly as the detached
            # patch objective does. Nearly every tile is unchanged from one step
            # to the next, so an unweighted mean is dominated by slots whose
            # correct prediction is "no change" -- a target the zero-initialized
            # identity predictor already meets.
            group_sums[name] = group_sums[name] + 0.5 * (tile_all + tile_changed)
            for position, value in enumerate((tile_all, tile_changed, tile_unchanged)):
                tile_sums[position] = tile_sums[position] + value

    group_means = {name: value / horizon for name, value in group_sums.items()}
    tile_all, tile_changed, tile_unchanged = (value / horizon for value in tile_sums)
    prediction = torch.stack([group_means[name] for name in order]).mean()
    return JepaTerms(
        prediction,
        group_means["unit"],
        group_means["market"],
        group_means["economy"],
        group_means["tile"],
        tile_all,
        tile_changed,
        tile_unchanged,
        sigreg,
        sigreg_terms["unit"],
        sigreg_terms["market"],
        sigreg_terms["economy"],
        sigreg_terms["tile"],
        reward,
        reward_scale,
        dispersion,
        motion / horizon,
        residual / horizon,
        eligible_total / horizon,
    )
