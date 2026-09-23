"""The `lejepa` family: one shared world-model backbone under two private towers.

The entity family splits cleanly into a wide trunk and narrow per-decision
readouts. This family keeps that split and moves the line: the trunk becomes a
world model owned by `lejepa.JepaObjective` alone, and everything that decides
something sits *after* it.

* **One backbone, owned by the world model's optimizer.** `LejepaBackbone` wraps
  the entity trunk and hands back a `JepaBelief` -- the per-slot decision states
  beside the encoded observation. `lejepa.jepa_horizon_loss` regularizes and
  predicts exactly that. The actor's heads read the attached belief
  (`LejepaConfig.policy_shapes_backbone`), so the policy's loss -- the clone
  loss, then PPO's -- shapes the encoder beside the objective; the critic's tower
  reads `JepaBelief.detach()`, so no value gradient reaches it.

  The policy has to reach it. A backbone fitted only to predict the teacher's
  trajectory cloned to teacher accuracy at argmax and went bankrupt in every
  sampled game: once a sampled departure left the trajectory, nothing in the
  features said which plant needed water. The value gradient is the one kept out,
  because the value target is the noisiest signal in the trainer and the policy
  reads the same features. One backbone is still the right count: the critic
  folds its privileged inputs in downstream, so a second encoder would only be
  the same observation fitted twice.

* **The towers are what comes after.** Both towers read the belief through
  private rounds of cross-attention, the critic's detached. The actor's decision slots read the
  whole encoded observation before their readouts, because the world model has no
  reason to copy what a decision needs into that decision's own slot: the economy
  already lives in the economy tokens, so a market slot read on its own cannot see
  it. The critic's tower also exists because a critic that only pooled the
  actor's latents would be a centralized critic with nothing centralized in it.
  Its privileged inputs -- the opponent's unit slots, and the private economic
  columns the seat cannot see -- are embedded by its own parameters and folded
  into the detached latents by cross-attention. Privileged information therefore
  enters strictly downstream of the world model, which is also what keeps it out
  of the policy: the backbone encodes the seat's own observation and nothing
  else, so there is no path by which an opponent's unit could reach a logit.

Three consequences of the ownership split, all deliberate.

`ppo._validate_optimizer_ownership` requires exact, pairwise-disjoint ownership.
The backbone is a submodule of the actor -- that is what keeps league snapshots,
inference bundles and the fused-to-portable rewrite reading one flat actor state
dict, under the `trunk.` prefix they already walk -- but it is *owned* by the
world model's optimizer, not the actor's. (The encoder's own keys sit one level
deeper than the entity family's, at `trunk.trunk.*`, because the backbone wraps
the entity trunk rather than being one. Nothing generic depends on the depth;
a `lejepa` archive written before this arrangement does not load, and there is
no migration, because a checkpoint whose critic carried a second encoder has no
answer to which of the two the shared one should be.)

Ownership is therefore stated as a partition of the actor's parameters
(`LejepaActor.head_parameters` and `backbone_parameters`) rather than as whole
modules, and the critic holds the backbone by reference
without registering it, so it appears in neither the critic's state dict nor its
parameter set.

The critic never carries the backbone in its own checkpoint, so a resumed or
snapshot-loaded critic is inert until `attach_backbone` hands it one -- through
`build_lejepa_pair` here, or `registry.pair_towers` where the trainer builds
both towers from a family it does not know the name of. A critic built any other
way raises on its first forward rather than encoding against nothing.

Because the encoder sits outside the critic's module tree, everything the critic
does to itself -- `.to`, `.cuda`, `.eval`, `.requires_grad_`, a `load_state_dict`
-- stops at its own tower. The actor's equivalents are what move the encoder,
and the trainer's `actor.eval()` / `critic.train()` pair is therefore the actor's
call to make for both. Two deep copies are likewise two object graphs: a copied
pair comes back unshared and must be re-paired.

The clone loss alone does not define this family: without its objective the
backbone would be an entity trunk under another name, and with the detached
ablation (`policy_shapes_backbone=False`) the heads would fit a readout over an
encoder at initialization. `scripts/train_bc.py` therefore clones this family only
beside its `JepaObjective`, which trains the backbone on the demonstration
transitions while the heads clone, and refuses it without one. PPO likewise steps the backbone only
on the minibatches where the policy steps: the heads read it, so a backbone that
kept moving after a KL stop would move the policy outside its trust region.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.entity import EntityActor, EntityConfig, EntityTrunk
from kaggriculture.model import (
    ActorOutput,
    Linear,
    RMSNorm,
    categorical_value,
    categorical_value_support,
    softcap_value_logits,
)
from kaggriculture.structured import (
    Attention,
    EconomyEmbedder,
    FeedForward,
    FusedFeedForward,
    GatedResidual,
    JepaBelief,
    StructuredCriticBelief,
    StructuredInputs,
    UnitEmbedder,
)
from kaggriculture.tokens import (
    ANIMAL_TOKEN_FIELDS,
    CROP_TOKEN_FIELDS,
    PRODUCT_TOKEN_FIELDS,
    TILE_COUNT,
)


@dataclass(frozen=True)
class LejepaConfig(EntityConfig):
    """Entity geometry plus the objective's four shapes and the two towers' depth.

    The objective's knobs live on the model configuration rather than on
    `PpoConfig` because they fix parameter shapes -- the projector's hidden width
    and the direction buffers' second dimension -- and every shape a checkpoint
    has to reproduce belongs to the artifact that records it. The coefficients,
    which fix nothing, stay on `PpoConfig` where the rest of the trade-offs are.
    """

    #: Hidden width of every projector, of the predictor's output head, and of
    #: the reward body. `../le-wm` runs a 192-wide encoder through a 2048-wide
    #: projector hidden layer; four times the model width is the same shape of
    #: choice at this model's scale and matches the trunk's own feed-forwards.
    jepa_hidden_dim: int = 384
    #: Random directions per SIGReg evaluation. The references use 1024 over a
    #: batch of 128 single-token samples. The statistic here is taken per token
    #: position, so the population behind each one is `jepa_sigreg_rows` rather
    #: than the whole minibatch -- comparable to the reference's, and eight times
    #: it. What the direction count buys is resolution of the *sphere*, and that
    #: saturates; 128 keeps the Epps-Pulley intermediate inside a few hundred
    #: megabytes at the production minibatch and the estimator unbiased.
    jepa_slices: int = 128
    #: Farm tiles supervised per minibatch, resampled every minibatch. Predicting
    #: all two hundred at the production minibatch size would cost more than the
    #: policy update it is meant to support.
    jepa_tile_samples: int = 32
    #: Rows entering SIGReg, taken as the leading slice of an already shuffled
    #: minibatch. The statistic is a constraint on the representation, not a
    #: per-row loss, and the reference enforces it on 128 samples a step.
    jepa_sigreg_rows: int = 1024
    #: Cross-attention rounds between the belief's decision slots and their
    #: readouts, each reading the whole encoded observation. Zero makes the
    #: actor's heads the entity family's linear ones over each slot alone, which
    #: against a frozen representation is a linear probe -- the reference
    #: protocol, and worth being able to run as the control it is. It is not a
    #: usable policy: even a gated feed-forward per slot cloned to only 0.76
    #: market-kind accuracy where the end-to-end entity actor reaches 0.999,
    #: because a market slot is not where the world model keeps the economy.
    policy_readout_layers: int = 1
    #: Let the policy's loss -- the clone loss, then PPO's -- reach the backbone
    #: beside the LeJEPA objective instead of stopping at a detach. The backbone
    #: stays owned, clipped and stepped by the objective's optimizer; it now
    #: answers to both losses. Off is the pure world-model ablation, and it does
    #: not survive sampling: its 12-epoch clone matched the teacher at argmax
    #: (0.997 market-kind accuracy) but went bankrupt in every sampled self-play
    #: game, its farm unwatered and its animals unfed within two days of the
    #: first departure, because features fitted only to predict the teacher's
    #: trajectory carry nothing a decision needs to recover off it. On, the same
    #: clone plays teacher-level banks (70-126k) under temperature-1 sampling.
    policy_shapes_backbone: bool = True
    #: Cross-attention rounds in the critic's private tower. One round is enough
    #: to let every privileged token read the whole encoded observation; the
    #: tower's job is to fold private information into a representation that is
    #: already built, not to build one.
    critic_private_layers: int = 1
    #: The entity family's critic readout feed-forward, on by default here.
    critic_readout_ffn: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("jepa_hidden_dim", "jepa_slices", "jepa_tile_samples", "jepa_sigreg_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("policy_readout_layers", "critic_private_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.critic_private_layers < 1:
            raise ValueError(
                "the LeJEPA critic needs at least one private round; with none, its "
                "privileged inputs would reach the value head through pooling alone"
            )
        if self.jepa_tile_samples > 2 * TILE_COUNT:
            raise ValueError("jepa_tile_samples cannot exceed the encoded tile count")
        if self.critic_architecture != "entity":
            raise ValueError(
                "the LeJEPA critic reads the shared backbone; keep critic_architecture"
            )
        if self.bixt_latents:
            raise ValueError("the LeJEPA belief reads entity trunk memory, which BiXT rewrites")


class LejepaBackbone(nn.Module):
    """The shared world model: the entity trunk, read as a `JepaBelief`.

    The trunk is built with `private_columns=False` -- the actor's configuration --
    because it encodes the seat's own observation and only that. The critic's
    private columns are not missing from the model; they are embedded by the
    critic, downstream of here, which is the only arrangement in which one encoder
    can serve both seats without leaking.
    """

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = EntityTrunk(config, private_columns=False, rematerialize_farms=True)

    def _encode(self, inputs: StructuredInputs) -> tuple[Tensor, Tensor]:
        """States and source memory from one trunk pass, with no optional output.

        `EntityTrunk.forward_with_memory` also returns a validity mask that is
        `None` without opponent-unit memory, which this encoder never has.
        """
        states, memory, _ = self.trunk.forward_with_memory(inputs)
        return states, memory

    def _belief(self, states: Tensor, memory: Tensor) -> JepaBelief:
        units, markets = states.split((MAX_UNITS, MAX_MARKET_ORDERS), dim=1)
        tiles, economy = memory.split((2 * TILE_COUNT, memory.shape[1] - 2 * TILE_COUNT), dim=1)
        if economy.shape[1] < 1:
            raise ValueError("backbone memory carries no economy tokens")
        return JepaBelief(units, markets, economy, tiles)

    def forward(self, inputs: StructuredInputs) -> JepaBelief:
        return self._belief(*self._encode(inputs))

    def rematerialized(self, inputs: StructuredInputs) -> JepaBelief:
        """`forward`, whose trunk replays its farm blocks in backward when training.

        The belief carries the trunk's whole tile memory and the objective scores
        a sample of its columns, so the update retains the trunk's activations
        across every term that reads it. The farm blocks, which run over every
        tile token, are most of those; the trunk is built to recompute only them.
        At the production minibatch that peaks ~3.7 GiB under recomputing the
        whole trunk, as this once did, and skips the rest of its replay.
        """
        return self.forward(inputs)


class _ReadoutRound(nn.Module):
    """One round of a tower: its tokens read the world, then each other.

    Cross-attention first, and deliberately. The critic's private tokens arrive as
    bare embeddings of privileged fields, and the actor's decision slots hold what
    the world model chose to keep there; the detached latents are the
    representation it has already built. Letting a tower's tokens talk among
    themselves before they have read the observation would be a round spent on the
    thinner half.
    """

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.cross_norm = RMSNorm(config.model_dim)
        self.cross_attention = Attention(config)
        self.cross_gate = GatedResidual(config.model_dim)
        self.self_norm = RMSNorm(config.model_dim)
        self.self_attention = Attention(config)
        self.self_gate = GatedResidual(config.model_dim)
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = FusedFeedForward(config) if config.fused_mlp else FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim)

    def forward(
        self, tokens: Tensor, context: Tensor, tokens_valid: Tensor, context_valid: Tensor
    ) -> Tensor:
        valid = tokens_valid.unsqueeze(-1)
        query = self.cross_norm(tokens)
        read = self.cross_attention(query, context, context_valid=context_valid)
        tokens = torch.where(valid, self.cross_gate(tokens, read), 0.0)
        query = self.self_norm(tokens)
        tokens = torch.where(
            valid,
            self.self_gate(tokens, self.self_attention(query, query, context_valid=tokens_valid)),
            0.0,
        )
        return torch.where(valid, self.ffn_gate(tokens, self.ffn(self.ffn_norm(tokens))), 0.0)


class LejepaActor(EntityActor):
    """Entity actor whose trunk is the shared world model its policy also shapes.

    `EntityActor.__init__` builds `self.trunk`; this family replaces it with the
    backbone module, so the actor's flat state dict still carries the encoder
    under `trunk.` keys and every snapshot, bundle and portable rewrite keeps
    working unchanged.
    """

    def __init__(self, config: LejepaConfig | None = None) -> None:
        super().__init__(config or LejepaConfig())

    def _build_trunk(self, config: EntityConfig) -> nn.Module:
        return LejepaBackbone(config)

    def _initialize_heads(self, config: EntityConfig) -> None:
        super()._initialize_heads(config)
        assert isinstance(config, LejepaConfig)
        layers = config.policy_readout_layers
        # A state read with no live shortcut to protect, as in the critic's tower.
        readout_config = (
            replace(config, zero_init_branches=False) if config.zero_init_branches else config
        )
        self.readout_context_norm = RMSNorm(config.model_dim) if layers else None
        self.readout = nn.ModuleList(_ReadoutRound(readout_config) for _ in range(layers))
        # The rounds end on a gated residual sum, which is not unit-RMS;
        # `initialize_policy_heads` scales every readout below for one that is.
        self.unit_readout_norm = RMSNorm(config.model_dim) if layers else None
        self.market_readout_norm = RMSNorm(config.model_dim) if layers else None

    def backbone_parameters(self):
        """The world model's parameters, which the actor's optimizer must not own."""
        return self.trunk.parameters()

    def head_parameters(self):
        """Everything the policy objective may train: the actor minus its backbone."""
        backbone = {id(parameter) for parameter in self.trunk.parameters()}
        return (parameter for parameter in self.parameters() if id(parameter) not in backbone)

    def encode_belief(self, inputs: StructuredInputs) -> JepaBelief:
        return self.trunk(inputs)

    def auxiliary_belief(self, inputs: StructuredInputs) -> JepaBelief:
        return self.trunk.rematerialized(inputs)

    def forward_with_belief(self, inputs: StructuredInputs) -> tuple[ActorOutput, JepaBelief]:
        belief = self.encode_belief(inputs)
        return self.decode_belief(belief, inputs.unit_active), belief

    def forward_with_auxiliary_belief(
        self, inputs: StructuredInputs
    ) -> tuple[ActorOutput, JepaBelief]:
        belief = self.auxiliary_belief(inputs)
        return self.decode_belief(belief, inputs.unit_active), belief

    def _read(
        self, units: Tensor, markets: Tensor, belief: JepaBelief, unit_active: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The decision slots after the readout rounds, renormalized for the heads.

        Inactive unit slots are neither read nor attended to, the trunk's own
        convention; the belief already holds them at zero.
        """
        assert self.readout_context_norm is not None
        assert self.unit_readout_norm is not None and self.market_readout_norm is not None
        batch = units.shape[0]
        always = torch.ones(
            batch,
            MAX_MARKET_ORDERS + belief.economy.shape[1] + belief.tiles.shape[1],
            dtype=torch.bool,
            device=units.device,
        )
        context = self.readout_context_norm(torch.cat(tuple(belief), dim=1))
        context_valid = torch.cat((unit_active, always), dim=1)
        decisions = torch.cat((units, markets), dim=1)
        decisions_valid = context_valid[:, : MAX_UNITS + MAX_MARKET_ORDERS]
        for round_ in self.readout:
            decisions = round_(decisions, context, decisions_valid, context_valid)
        units, markets = decisions.split((MAX_UNITS, MAX_MARKET_ORDERS), dim=1)
        return self.unit_readout_norm(units), self.market_readout_norm(markets)

    def decode_belief(self, belief: JepaBelief, unit_active: Tensor) -> ActorOutput:
        """Read the belief through the heads, attached unless the config detaches.

        This is the single point that decides whether the policy's gradient
        reaches the world model. It sits here rather than in the trainer so that
        every consumer -- PPO, behavior cloning, a rollout, an inference bundle --
        gets the same model, and no caller can change the family by forgetting to
        detach or by detaching. Under `no_grad` it costs nothing.

        The head normalizations run here rather than at encode time, which is
        where `EntityActor` puts them. They cannot run there in this family:
        the belief is what the world model predicts, and it has to be the
        encoder's own output, not a rescaling of it that the policy owns. They
        still have to run, because `initialize_policy_heads` scales every
        readout below for a unit-RMS input.
        """
        if not self.config.policy_shapes_backbone:
            belief = belief.detach()
        units = self.unit_head[0](belief.unit_decisions)
        markets = self.market_norm(belief.market_decisions)
        if self.readout:
            units, markets = self._read(units, markets, belief, unit_active)
        return ActorOutput(
            unit_logits=self.unit_head[-1](units).contiguous(),
            market_kind_logits=self.market_kind(markets).contiguous(),
            market_quantity_context=self.market_quantity_context(markets).contiguous(),
        )


class _OpponentUnitEmbedder(nn.Module):
    """The critic's own embedder for the opponent's unit slots.

    Its own, and not the backbone's `trunk.units`, for the reason the whole family
    is arranged this way: those parameters belong to the world model's optimizer,
    and a value gradient arriving in them would be the leak the detach exists to
    prevent. Constructed without local tile context, because the opponent's farm
    tiles are already in the encoded observation the tower cross-attends to.
    """

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.embedder = UnitEmbedder(config, local_init=False, local_context=False)

    def forward(self, categorical: Tensor, continuous: Tensor, active: Tensor) -> Tensor:
        return self.embedder(categorical, continuous, active, None, opponent=True)


class LejepaCritic(nn.Module):
    """A centralized critic built entirely on top of the shared world model.

    It owns no encoder. Its inputs are the detached `JepaBelief` and two groups of
    privileged tokens it embeds itself: the opponent's unit slots, and the economy
    with its private columns. Those run through `critic_private_layers` rounds of
    cross-attention into the observation, and a single learned query then pools
    everything into the one valuation state the value head reads.

    `critic_source_read` keeps the meaning it has in the entity family: with it
    off, the pooled query reads the decision slots and the private tokens; with it
    on, the encoded tiles and economy join the context too.
    """

    def __init__(self, config: LejepaConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or LejepaConfig()
        self._backbone: list[LejepaBackbone] = []
        # A state read, not a residual branch with a live shortcut -- the entity
        # critic's own reasoning for its pooling attention, and the private tower
        # is the same kind of readout.
        readout_config = (
            replace(config, zero_init_branches=False) if config.zero_init_branches else config
        )
        self.opponent_units = _OpponentUnitEmbedder(config)
        self.private_economy = EconomyEmbedder(config, private_columns=True)
        self.private_norm = RMSNorm(config.model_dim)
        self.rounds = nn.ModuleList(
            _ReadoutRound(readout_config) for _ in range(config.critic_private_layers)
        )
        self.context_norm = RMSNorm(config.model_dim)
        self.pool_norm = RMSNorm(config.model_dim)
        self.value_query = nn.Parameter(
            torch.nn.functional.rms_norm(torch.randn(1, config.model_dim), (config.model_dim,))
        )
        self.pool_attention = Attention(readout_config)
        self.value_ffn_norm = RMSNorm(config.model_dim) if config.critic_readout_ffn else None
        self.value_ffn = (
            (FusedFeedForward(config) if config.fused_mlp else FeedForward(config))
            if config.critic_readout_ffn
            else None
        )
        self.value_ffn_gate = GatedResidual(config.model_dim) if config.critic_readout_ffn else None
        self.value_norm = RMSNorm(config.model_dim, eps=1e-5)
        self.value_head = Linear(config.model_dim, 1 if config.scalar_value else config.value_atoms)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.register_buffer(
            "support",
            categorical_value_support(config.value_min, config.value_max, config.value_atoms),
            persistent=True,
        )

    def attach_backbone(self, backbone: LejepaBackbone) -> None:
        """Bind the actor's world model, by reference and outside the module tree.

        Held in a plain list so PyTorch does not register it: the backbone must
        appear in exactly one state dict and one parameter set, the actor's, or
        `ppo._validate_optimizer_ownership` sees the same tensors owned twice and
        a critic checkpoint silently carries a second copy of the encoder.
        Everything that moves the actor -- `.to`, `.float`, a load -- therefore
        moves this too, which is the intent: there is one backbone. The reverse
        does not hold, and that is the trap: `critic.eval()`, `critic.to(...)`
        and `critic.requires_grad_(False)` all stop at the critic's own tower.
        The encoder follows the actor, in mode as in device.
        """
        if not isinstance(backbone, LejepaBackbone):
            raise TypeError("the LeJEPA critic requires a LejepaBackbone")
        if backbone.config.to_dict() != self.config.to_dict():
            raise ValueError("the shared backbone must carry the critic's model configuration")
        self._backbone = [backbone]

    @property
    def backbone(self) -> LejepaBackbone:
        if not self._backbone:
            raise RuntimeError(
                "the LeJEPA critic has no backbone; build the pair with build_lejepa_pair "
                "or call attach_backbone with the actor's"
            )
        return self._backbone[0]

    def _private_tokens(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        opponents = self.opponent_units(
            opponent_unit_categorical, opponent_unit_continuous, opponent_unit_active
        )
        economy = self.private_economy(
            inputs.products, inputs.animals, inputs.crops, inputs.farms, inputs.town
        )
        tokens = torch.cat((opponents, economy.to(opponents.dtype)), dim=1)
        valid = torch.cat(
            (
                opponent_unit_active,
                torch.ones(
                    economy.shape[0],
                    economy.shape[1],
                    dtype=torch.bool,
                    device=economy.device,
                ),
            ),
            dim=1,
        )
        return self.private_norm(tokens), valid

    def _context(self, belief: JepaBelief, inputs: StructuredInputs) -> tuple[Tensor, Tensor]:
        """The detached observation the private tokens read and the query pools."""
        batch = belief.unit_decisions.shape[0]
        device = belief.unit_decisions.device
        pieces = [belief.unit_decisions, belief.market_decisions]
        valid = [
            inputs.unit_active,
            torch.ones(batch, MAX_MARKET_ORDERS, dtype=torch.bool, device=device),
        ]
        if self.config.critic_source_read:
            pieces.extend((belief.economy, belief.tiles))
            valid.extend(
                torch.ones(batch, piece.shape[1], dtype=torch.bool, device=device)
                for piece in (belief.economy, belief.tiles)
            )
        return self.context_norm(torch.cat(pieces, dim=1)), torch.cat(valid, dim=1)

    def _valuation_state(
        self,
        belief: JepaBelief,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
    ) -> Tensor:
        context, context_valid = self._context(belief, inputs)
        private, private_valid = self._private_tokens(
            inputs, opponent_unit_categorical, opponent_unit_continuous, opponent_unit_active
        )
        for round_ in self.rounds:
            private = round_(private, context, private_valid, context_valid)
        pooled_context = torch.cat((context, self.pool_norm(private)), dim=1)
        pooled_valid = torch.cat((context_valid, private_valid), dim=1)
        query = self.value_query.unsqueeze(0).expand(pooled_context.shape[0], -1, -1)
        pooled = self.pool_attention(query, pooled_context, context_valid=pooled_valid)
        if self.value_ffn is not None:
            assert self.value_ffn_norm is not None and self.value_ffn_gate is not None
            pooled = self.value_ffn_gate(pooled, self.value_ffn(self.value_ffn_norm(pooled)))
        return self.value_norm(pooled)

    @staticmethod
    def _public_inputs(inputs: StructuredInputs) -> StructuredInputs:
        """The seat's own observation, with the critic's private columns trimmed off.

        `ppo._critic_batch_args` hands the critic the actor's inputs with the
        opponent's private economic columns concatenated onto the product, animal
        and crop tokens. The backbone is the *actor's* encoder and its economy
        embedder is built for the narrow widths, so the private columns are
        sliced away here and re-enter through `private_economy` below. Slicing
        rather than restaging is what keeps the critic's minibatch free of a
        second gather.
        """
        return inputs._replace(
            products=inputs.products[..., : len(PRODUCT_TOKEN_FIELDS)],
            animals=inputs.animals[..., : len(ANIMAL_TOKEN_FIELDS)],
            crops=inputs.crops[..., : len(CROP_TOKEN_FIELDS)],
        )

    def encode_belief(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
        backbone_belief: JepaBelief | None = None,
    ) -> StructuredCriticBelief:
        """One backbone pass, detached, then the private tower over it.

        The pass runs under `no_grad` rather than merely being detached
        afterwards: no value gradient can reach the backbone either way, so
        building the graph would only retain the trunk's activations for a
        backward that will never use them.

        `backbone_belief` skips the pass for a caller that already holds this
        minibatch's encoding at the current weights. The PPO update is that
        caller: the actor encodes the same public inputs through the same
        module before any optimizer step, so a second pass here would be the
        identical trunk forward run twice per minibatch. It is detached here
        whatever the caller hands over, so the value gradient stops exactly
        where it did.
        """
        if backbone_belief is None:
            with torch.no_grad():
                backbone_belief = self.backbone(self._public_inputs(inputs))
        belief = backbone_belief.detach()
        return StructuredCriticBelief(
            self._valuation_state(
                belief,
                inputs,
                opponent_unit_categorical,
                opponent_unit_continuous,
                opponent_unit_active,
            )
        )

    def decode_belief(self, belief: StructuredCriticBelief) -> Tensor:
        readout = self.value_head(belief.value_decision[:, 0])
        return (readout if self.config.scalar_value else softcap_value_logits(readout)).contiguous()

    def forward_with_belief(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
        backbone_belief: JepaBelief | None = None,
    ) -> tuple[Tensor, StructuredCriticBelief]:
        belief = self.encode_belief(
            inputs,
            opponent_unit_categorical,
            opponent_unit_continuous,
            opponent_unit_active,
            backbone_belief,
        )
        return self.decode_belief(belief), belief

    def forward(
        self,
        inputs: StructuredInputs,
        opponent_unit_categorical: Tensor,
        opponent_unit_continuous: Tensor,
        opponent_unit_active: Tensor,
        backbone_belief: JepaBelief | None = None,
    ) -> Tensor:
        return self.forward_with_belief(
            inputs,
            opponent_unit_categorical,
            opponent_unit_continuous,
            opponent_unit_active,
            backbone_belief,
        )[0]

    def value(self, logits: Tensor) -> Tensor:
        if self.config.scalar_value:
            return logits.float().squeeze(-1)
        return categorical_value(logits, self.support)


def build_lejepa_pair(
    config: LejepaConfig | None = None,
) -> tuple[LejepaActor, LejepaCritic]:
    """The only way to build a matched actor and critic: one backbone between them."""
    config = config or LejepaConfig()
    actor = LejepaActor(config)
    critic = LejepaCritic(config)
    critic.attach_backbone(actor.trunk)
    return actor, critic
