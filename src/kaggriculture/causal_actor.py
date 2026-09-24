"""Execution-order policy: one public encoding, exact prefix ledger, causal actions.

Teacher forcing evaluates the neural decoder in parallel. Collection uses fixed
36-token cached decoding entirely on device; state never persists across turns.
The immutable native quote table belongs to the collector, not the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend

from kaggriculture.device_ledger import DeviceLedger, PolicyLedger, ReplayMasks, initial_ledger
from kaggriculture.entity import EntityActor, EntityConfig
from kaggriculture.model import Linear, RMSNorm
from kaggriculture.strategic_actor import PublicEncoder
from kaggriculture.structured import (
    Attention,
    FeedForward,
    FusedFeedForward,
    GatedResidual,
    StructuredInputs,
    _fused_attention,
)

DECISIONS = 36
MARKET_DECISIONS = 20
MARKET_CACHE = 24  # Efficient SDPA requires a multiple-of-eight key stride.
BOS = 190
LEDGER_FEATURES = 57


@torch.library.custom_op("kaggriculture::causal_distribution", mutates_args=(), device_types="cuda")
def causal_distribution(
    logits: Tensor, mask: Tensor, selected: Tensor, active: Tensor, temperatures: Tensor
) -> tuple[Tensor, Tensor]:
    """Keep the likelihood reduction separate from the next ledger decision.

    Inductor otherwise fuses this reduction with the 100-bin market ledger and
    emits a Triton temporary outside its defining loop. The operator preserves
    the ordinary FP32 categorical calculation and supports actor gradients.
    """
    logprob = (logits.float() / temperatures[:, None]).masked_fill(~mask, -1e9).log_softmax(-1)
    chosen = logprob.gather(-1, selected[:, None]).squeeze(-1)
    entropy = -(logprob.exp() * logprob).sum(-1)
    return torch.where(active, chosen, 0), torch.where(active, entropy, 0)


@causal_distribution.register_fake
def _fake_causal_distribution(
    logits: Tensor, mask: Tensor, selected: Tensor, active: Tensor, temperatures: Tensor
) -> tuple[Tensor, Tensor]:
    del mask, selected, active, temperatures
    result = logits.new_empty(logits.shape[0], dtype=torch.float32)
    return result, torch.empty_like(result)


def _causal_distribution_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _causal_distribution_backward(ctx, grad_chosen: Tensor, grad_entropy: Tensor):
    logits, mask, selected, active, temperatures = ctx.saved_tensors
    if grad_chosen is None:
        grad_chosen = torch.zeros_like(temperatures)
    if grad_entropy is None:
        grad_entropy = torch.zeros_like(temperatures)
    logprob = (logits.float() / temperatures[:, None]).masked_fill(~mask, -1e9).log_softmax(-1)
    probabilities = logprob.exp()
    entropy = -(probabilities * logprob).sum(-1)
    selected_gradient = torch.nn.functional.one_hot(selected.long(), logits.shape[-1]).float()
    gradient = grad_chosen[:, None] * (selected_gradient - probabilities)
    gradient -= grad_entropy[:, None] * probabilities * (logprob + entropy[:, None])
    gradient = gradient * active[:, None] * mask / temperatures[:, None]
    return gradient.to(logits.dtype), None, None, None, None


causal_distribution.register_autograd(
    _causal_distribution_backward, setup_context=_causal_distribution_context
)


def _flatten_vmap_rows(
    batch_size: int, in_dims: tuple[int | None, ...], *values: Tensor
) -> tuple[Tensor, ...]:
    """Present every mapped lane and its rows as one custom-op batch."""
    return tuple(
        (
            value.unsqueeze(0).expand(batch_size, *value.shape)
            if dim is None
            else value.movedim(dim, 0)
        ).flatten(0, 1)
        for value, dim in zip(values, in_dims, strict=True)
    )


@torch.library.register_vmap(causal_distribution)
def _vmap_causal_distribution(
    info: object,
    in_dims: tuple[int | None, ...],
    logits: Tensor,
    mask: Tensor,
    selected: Tensor,
    active: Tensor,
    temperatures: Tensor,
) -> tuple[tuple[Tensor, Tensor], tuple[int, int]]:
    flattened = _flatten_vmap_rows(
        info.batch_size, in_dims, logits, mask, selected, active, temperatures
    )
    chosen, entropy = causal_distribution(*flattened)
    return (
        chosen.unflatten(0, (info.batch_size, -1)),
        entropy.unflatten(0, (info.batch_size, -1)),
    ), (0, 0)


@torch.library.custom_op("kaggriculture::causal_choose", mutates_args=(), device_types="cuda")
def causal_choose(
    logits: Tensor,
    mask: Tensor,
    recorded: Tensor,
    uniform: Tensor,
    temperatures: Tensor,
    deterministic: Tensor,
) -> Tensor:
    """Materialize a selected factor before updating the exact prefix ledger."""
    scaled = (logits.float() / temperatures[:, None]).masked_fill(~mask, -1e9)
    cdf = scaled.softmax(-1).cumsum(-1)
    last = torch.where(mask, torch.arange(mask.shape[-1], device=mask.device), 0).amax(-1)
    sampled = torch.minimum((cdf <= uniform[:, None]).sum(-1), last)
    sampled = torch.where(deterministic, scaled.argmax(-1), sampled)
    proposed = torch.where(recorded >= 0, recorded, sampled).long()
    return torch.where(mask.gather(1, proposed[:, None]).squeeze(1), proposed, 0)


@causal_choose.register_fake
def _fake_causal_choose(
    logits: Tensor,
    mask: Tensor,
    recorded: Tensor,
    uniform: Tensor,
    temperatures: Tensor,
    deterministic: Tensor,
) -> Tensor:
    del logits, mask, uniform, temperatures, deterministic
    return torch.empty_like(recorded)


@torch.library.register_vmap(causal_choose)
def _vmap_causal_choose(
    info: object,
    in_dims: tuple[int | None, ...],
    logits: Tensor,
    mask: Tensor,
    recorded: Tensor,
    uniform: Tensor,
    temperatures: Tensor,
    deterministic: Tensor,
) -> tuple[Tensor, int]:
    flattened = _flatten_vmap_rows(
        info.batch_size,
        in_dims,
        logits,
        mask,
        recorded,
        uniform,
        temperatures,
        deterministic,
    )
    selected = causal_choose(*flattened)
    return selected.unflatten(0, (info.batch_size, -1)), 0


@dataclass(frozen=True)
class CausalConfig(EntityConfig):
    decoder_layers: int = 2
    parallel_unit_decode: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.decoder_layers, bool)
            or not isinstance(self.decoder_layers, int)
            or self.decoder_layers < 1
        ):
            raise ValueError("decoder_layers must be a positive integer")
        if not isinstance(self.parallel_unit_decode, bool):
            raise ValueError("parallel_unit_decode must be a boolean")
        if any(
            (
                self.bixt_latents,
                self.memory_writeback,
                self.unit_tile_bias,
                self.tile_cross_rope,
                self.unit_local_readout,
                self.shared_memory_kv,
                self.inter_attention_ffn,
            )
        ):
            raise ValueError("causal decoding does not use entity-only experimental paths")


class CausalChoice(NamedTuple):
    packed_ledger: Tensor
    unit_actions: Tensor
    market_kinds: Tensor
    market_quantities: Tensor
    uniforms: Tensor  # [B,36], generated outside compiled forward
    temperatures: Tensor  # [B], strictly positive
    deterministic: Tensor  # [B] bool


class CausalReplay(NamedTuple):
    packed_ledger: Tensor
    unit_actions: Tensor
    market_kinds: Tensor
    market_quantities: Tensor


class CausalOutput(NamedTuple):
    unit_logits: Tensor
    market_kind_logits: Tensor
    market_quantity_context: Tensor
    unit_actions: Tensor
    market_kinds: Tensor
    market_quantities: Tensor
    unit_masks: Tensor
    market_kind_masks: Tensor
    market_quantity_masks: Tensor
    unit_active: Tensor
    market_active: Tensor
    market_quantity_active: Tensor
    factor_logprobs: Tensor  # [B,36], execution order, inactive positions zero
    factor_entropies: Tensor


def ledger_features(state: PolicyLedger, unit: int | None) -> Tensor:
    """Own information only; exact resources use signed log magnitude encoding."""
    scalars = torch.stack(
        (
            state.day,
            state.money,
            state.hires,
            state.units,
            state.land,
            state.capacity,
            state.hire_multiplier,
        ),
        dim=-1,
    )
    if unit is None:
        local = torch.zeros_like(state.tiles[:, 0])
        held = torch.zeros_like(state.inventories[:, 0])
        position = torch.zeros_like(state.positions[:, 0])
    else:
        local, held, position = (
            state.tiles[:, unit],
            state.inventories[:, unit],
            state.positions[:, unit],
        )
    values = torch.cat(
        (
            scalars,
            state.seeds,
            state.shed,
            state.market,
            local,
            held,
            position,
            state.market_active[:, None],
        ),
        dim=-1,
    ).float()
    return values.sign() * values.abs().log1p()


class CausalLayer(nn.Module):
    def __init__(self, config: CausalConfig) -> None:
        super().__init__()
        self.self_norm = RMSNorm(config.model_dim)
        self.self_attention = Attention(config)
        self.self_gate = GatedResidual(config.model_dim, 0.1)
        self.cross_norm = RMSNorm(config.model_dim)
        self.cross_attention = Attention(config)
        self.cross_gate = GatedResidual(config.model_dim, 0.1)
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = FusedFeedForward(config) if config.fused_mlp else FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim, 0.1)
        self.modulation = (
            Linear(config.model_dim, 4 * config.model_dim) if config.global_modulation else None
        )
        if self.modulation is not None:
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    @staticmethod
    def keys(attention: Attention, context: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = context.shape
        key, value = (
            attention.key_value(context)
            .view(batch, length, 2, attention.kv_heads, attention.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        return attention.key_norm(key), value

    @staticmethod
    def attend(
        attention: Attention, inputs: Tensor, keys: tuple[Tensor, Tensor], mask: Tensor
    ) -> Tensor:
        batch, length, width = inputs.shape
        query = attention.query_norm(
            attention.query(inputs)
            .view(batch, length, attention.heads, attention.head_dim)
            .transpose(1, 2)
        )
        result = _fused_attention(
            query,
            *keys,
            mask,
            enable_gqa=attention.heads != attention.kv_heads,
            scale=attention.head_dim**-0.5,
            # Use the same fused arithmetic family for cached and parallel paths.
            backend=SDPBackend.EFFICIENT_ATTENTION if query.is_cuda else None,
        )
        return attention.output(
            result.transpose(1, 2).reshape(batch, length, width).to(inputs.dtype)
        )

    def condition(self, summary: Tensor) -> tuple[Tensor, ...] | None:
        return None if self.modulation is None else self.modulation(summary).chunk(4, dim=-1)

    @staticmethod
    def modulate(
        inputs: Tensor, conditioning: tuple[Tensor, ...] | None, *, ffn: bool = False
    ) -> Tensor:
        if conditioning is None:
            return inputs
        scale, shift = conditioning[2:] if ffn else conditioning[:2]
        return inputs * (1 + scale[:, None].to(inputs.dtype)) + shift[:, None].to(inputs.dtype)

    def finish(
        self,
        hidden: Tensor,
        memory: tuple[Tensor, Tensor],
        valid: Tensor,
        conditioning: tuple[Tensor, ...] | None,
    ) -> Tensor:
        query = self.modulate(self.cross_norm(hidden), conditioning)
        hidden = self.cross_gate(
            hidden, self.attend(self.cross_attention, query, memory, valid[:, None, None])
        )
        inputs = self.modulate(self.ffn_norm(hidden), conditioning, ffn=True)
        return self.ffn_gate(hidden, self.ffn(inputs))

    def parallel(
        self,
        hidden: Tensor,
        memory: tuple[Tensor, Tensor],
        valid: Tensor,
        conditioning: tuple[Tensor, ...] | None,
        causal: Tensor,
    ) -> Tensor:
        query = self.modulate(self.self_norm(hidden), conditioning)
        hidden = self.self_gate(
            hidden,
            self.attend(self.self_attention, query, self.keys(self.self_attention, query), causal),
        )
        return self.finish(hidden, memory, valid, conditioning)

    def step(
        self,
        hidden: Tensor,
        memory: tuple[Tensor, Tensor],
        valid: Tensor,
        conditioning: tuple[Tensor, ...] | None,
        cache: tuple[Tensor, Tensor],
        position: int,
        positions: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        query = self.modulate(self.self_norm(hidden), conditioning)
        current = self.keys(self.self_attention, query)
        write = (positions == position)[None, None, :, None]
        updated = tuple(
            torch.where(write, value, old) for value, old in zip(current, cache, strict=True)
        )
        mask = (positions <= position)[None, None, None]
        hidden = self.self_gate(hidden, self.attend(self.self_attention, query, updated, mask))
        return self.finish(hidden, memory, valid, conditioning), updated


class CausalTrunk(nn.Module):
    def __init__(self, config: CausalConfig) -> None:
        super().__init__()
        self.encoder = PublicEncoder(config)
        self.position = nn.Embedding(DECISIONS, config.model_dim)
        self.previous_action = nn.Embedding(BOS + 1, config.model_dim)
        self.ledger = nn.Sequential(
            Linear(LEDGER_FEATURES, config.model_dim), RMSNorm(config.model_dim)
        )
        self.memory_norm = RMSNorm(config.model_dim)
        self.layers = nn.ModuleList(CausalLayer(config) for _ in range(config.decoder_layers))
        self.register_buffer("positions", torch.arange(DECISIONS), persistent=False)

    def encode(self, inputs: StructuredInputs):
        units, economy, memory, valid = self.encoder(inputs)
        summary = economy.mean(1)
        memory = self.memory_norm(memory)
        queries = torch.cat((units, summary[:, None].expand(-1, 20, -1)), dim=1)
        keys = [layer.keys(layer.cross_attention, memory) for layer in self.layers]
        conditioning = [layer.condition(summary) for layer in self.layers]
        return queries, keys, valid, conditioning

    def tokens(self, queries: Tensor, ledger: Tensor, previous: Tensor) -> Tensor:
        return queries + self.position.weight + self.previous_action(previous) + self.ledger(ledger)


class CausalActor(EntityActor):
    def __init__(
        self, config: CausalConfig | None = None, *, ledger: DeviceLedger | None = None
    ) -> None:
        nn.Module.__init__(self)
        self.config = config = config or CausalConfig()
        self.trunk = CausalTrunk(config)
        self._initialize_heads(config)
        self.device_ledger = ledger

    def set_device_ledger(self, ledger: DeviceLedger) -> None:
        """Attach one collector-owned exact quote table outside compile/capture."""
        self.device_ledger = ledger

    def _rules(self) -> DeviceLedger:
        if self.device_ledger is None:
            raise RuntimeError("attach a shared DeviceLedger before causal forward")
        return self.device_ledger

    @staticmethod
    def _distribution(
        logits: Tensor, mask: Tensor, selected: Tensor, active: Tensor, temperatures: Tensor
    ) -> tuple[Tensor, Tensor]:
        if logits.is_cuda:
            return causal_distribution(logits, mask, selected, active, temperatures)
        logprob = (logits.float() / temperatures[:, None]).masked_fill(~mask, -1e9).log_softmax(-1)
        chosen = logprob.gather(-1, selected[:, None]).squeeze(-1)
        entropy = -(logprob.exp() * logprob).sum(-1)
        return torch.where(active, chosen, 0), torch.where(active, entropy, 0)

    @staticmethod
    def _choose(
        logits: Tensor,
        mask: Tensor,
        recorded: Tensor,
        uniform: Tensor,
        temperatures: Tensor,
        deterministic: Tensor,
    ) -> Tensor:
        if logits.is_cuda:
            return causal_choose(logits, mask, recorded, uniform, temperatures, deterministic)
        scaled = (logits.float() / temperatures[:, None]).masked_fill(~mask, -1e9)
        cdf = scaled.softmax(-1).cumsum(-1)
        # The final legal category absorbs FP32 cumulative-rounding residue.
        last = torch.where(mask, torch.arange(mask.shape[-1], device=mask.device), 0).amax(-1)
        sampled = torch.minimum((cdf <= uniform[:, None]).sum(-1), last)
        sampled = torch.where(deterministic, scaled.argmax(-1), sampled)
        proposed = torch.where(recorded >= 0, recorded, sampled).long()
        return torch.where(mask.gather(1, proposed[:, None]).squeeze(1), proposed, 0)

    def _output(
        self,
        states: Tensor,
        units: Tensor,
        kinds: Tensor,
        quantities: Tensor,
        masks: ReplayMasks,
        temperatures: Tensor,
    ) -> CausalOutput:
        unit_logits = self.unit_head(states[:, :16])
        kind_states = self.market_norm(states[:, 16::2])
        quantity_states = self.market_norm(states[:, 17::2])
        kind_logits = self.market_kind(kind_states)
        context = self.market_quantity_context(quantity_states)
        quantity_logits = self.quantity_logits(context, kinds, masks.market_quantity_masks)
        logprobs, entropies = [], []
        for position in range(DECISIONS):
            if position < 16:
                logits, mask, selected, active = (
                    unit_logits[:, position],
                    masks.unit_masks[:, position],
                    units[:, position],
                    masks.unit_active[:, position],
                )
            else:
                slot = (position - 16) // 2
                if position % 2 == 0:
                    logits, mask, selected, active = (
                        kind_logits[:, slot],
                        masks.market_kind_masks[:, slot],
                        kinds[:, slot],
                        masks.market_active[:, slot],
                    )
                else:
                    logits, mask, selected, active = (
                        quantity_logits[:, slot],
                        masks.market_quantity_masks[:, slot],
                        quantities[:, slot],
                        masks.market_quantity_active[:, slot],
                    )
            lp, entropy = self._distribution(logits, mask, selected, active, temperatures)
            logprobs.append(lp)
            entropies.append(entropy)
        return CausalOutput(
            unit_logits,
            kind_logits,
            context,
            units,
            kinds,
            quantities,
            *masks,
            torch.stack(logprobs, 1),
            torch.stack(entropies, 1),
        )

    def teacher_force(
        self,
        inputs: StructuredInputs,
        packed: Tensor,
        units: Tensor,
        kinds: Tensor,
        quantities: Tensor,
        temperatures: Tensor | None = None,
    ) -> CausalOutput:
        """Parallel neural prefix replay; supplied actions must be legal recorded factors."""
        rules = self._rules()
        state = initial_ledger(packed)
        units = torch.where(
            torch.arange(16, device=packed.device)[None] < state.units[:, None], units, 0
        )
        features, tokens = [], []
        canonical_kinds, canonical_quantities = [], []
        unit_masks, kind_masks, quantity_masks, active, qactive = [], [], [], [], []
        for unit in range(16):
            if not self.config.parallel_unit_decode:
                features.append(ledger_features(state, unit))
            unit_masks.append(rules.unit_mask(state, unit))
            tokens.append(units[:, unit])
            state = rules.apply_unit(state, unit, units[:, unit])
        for slot in range(10):
            features.append(ledger_features(state, None))
            kind_masks.append(rules.market_kind_mask(state))
            active.append(state.market_active)
            kind = torch.where(state.market_active, kinds[:, slot], 0)
            quantity = torch.where(state.market_active & (kind >= 3), quantities[:, slot], 0)
            canonical_kinds.append(kind)
            canonical_quantities.append(quantity)
            quote = rules.market_quote(state, kind)
            tokens.append(kind + 68)
            features.append(ledger_features(state, None))
            quantity_masks.append(quote.mask)
            qactive.append(state.market_active & (kind >= 3))
            tokens.append(quantity + 90)
            state = rules.apply_market(state, quote, quantity)
        kinds = torch.stack(canonical_kinds, 1)
        quantities = torch.stack(canonical_quantities, 1)
        previous = torch.cat(
            (torch.full_like(units[:, :1], BOS), torch.stack(tokens[:-1], 1)), dim=1
        )
        queries, memory, valid, conditioning = self.trunk.encode(inputs)
        if self.config.parallel_unit_decode:
            # Unit neural decisions share the encoded observation. Their legality
            # and resource effects still replay in exact execution order above.
            market_hidden = (
                queries[:, 16:]
                + self.trunk.position.weight[16:]
                + self.trunk.previous_action(previous[:, 16:])
                + self.trunk.ledger(torch.stack(features, 1))
            )
            market_hidden = torch.nn.functional.pad(
                market_hidden, (0, 0, 0, MARKET_CACHE - MARKET_DECISIONS)
            )
            positions = self.trunk.positions[:MARKET_CACHE]
            unit_hidden = queries[:, :16]
            hidden = market_hidden
        else:
            hidden = self.trunk.tokens(queries, torch.stack(features, 1), previous)
            positions = self.trunk.positions
        causal = (positions[:, None] >= positions[None])[None, None]
        for layer, keys, condition in zip(self.trunk.layers, memory, conditioning, strict=True):
            hidden = layer.parallel(hidden, keys, valid, condition, causal)
        if self.config.parallel_unit_decode:
            hidden = torch.cat((unit_hidden, hidden[:, :MARKET_DECISIONS]), 1)
        masks = ReplayMasks(
            torch.stack(unit_masks, 1),
            torch.stack(kind_masks, 1),
            torch.stack(quantity_masks, 1),
            torch.arange(16, device=packed.device)[None] < state.units[:, None],
            torch.stack(active, 1),
            torch.stack(qactive, 1),
        )
        if temperatures is None:
            temperatures = torch.ones_like(packed[:, 0], dtype=torch.float32)
        return self._output(hidden, units, kinds, quantities, masks, temperatures)

    def forward(
        self, inputs: StructuredInputs, choice: CausalChoice | CausalReplay
    ) -> CausalOutput:
        if isinstance(choice, CausalReplay):
            return self.teacher_force(inputs, *choice)
        if self.config.parallel_unit_decode:
            return self._forward_market_causal(inputs, choice)
        rules = self._rules()
        state = initial_ledger(choice.packed_ledger)
        queries, memory, valid, conditioning = self.trunk.encode(inputs)
        batch = queries.shape[0]
        cache = [
            (
                keys[0].new_zeros(
                    (
                        batch,
                        self.config.attention_kv_heads,
                        DECISIONS,
                        self.config.model_dim // self.config.attention_heads,
                    )
                ),
                keys[1].new_zeros(
                    (
                        batch,
                        self.config.attention_kv_heads,
                        DECISIONS,
                        self.config.model_dim // self.config.attention_heads,
                    )
                ),
            )
            for keys in memory
        ]
        previous = torch.full_like(choice.unit_actions[:, 0], BOS)
        units, kinds, quantities = [], [], []
        unit_logits, kind_logits, quantity_contexts = [], [], []
        logprobs, entropies = [], []
        unit_masks, kind_masks, quantity_masks, active, qactive = [], [], [], [], []
        for position in range(DECISIONS):
            unit = position if position < 16 else None
            hidden = (
                queries[:, position : position + 1]
                + self.trunk.position.weight[position]
                + self.trunk.previous_action(previous)[:, None]
                + self.trunk.ledger(ledger_features(state, unit))[:, None]
            )
            for index, layer in enumerate(self.trunk.layers):
                hidden, cache[index] = layer.step(
                    hidden,
                    memory[index],
                    valid,
                    conditioning[index],
                    cache[index],
                    position,
                    self.trunk.positions,
                )
            if position < 16:
                logits = self.unit_head(hidden[:, 0])
                unit_logits.append(logits)
                decision_active = position < state.units
                mask = rules.unit_mask(state, position)
                selected = self._choose(
                    logits,
                    mask,
                    choice.unit_actions[:, position],
                    choice.uniforms[:, position],
                    choice.temperatures,
                    choice.deterministic,
                )
                unit_masks.append(mask)
                units.append(selected)
                state = rules.apply_unit(state, position, selected)
                previous = selected
            elif position % 2 == 0:
                slot = (position - 16) // 2
                logits = self.market_kind(self.market_norm(hidden[:, 0]))
                kind_logits.append(logits)
                decision_active = state.market_active
                mask = rules.market_kind_mask(state)
                selected = self._choose(
                    logits,
                    mask,
                    choice.market_kinds[:, slot],
                    choice.uniforms[:, position],
                    choice.temperatures,
                    choice.deterministic,
                )
                quote = rules.market_quote(state, selected)
                kind_masks.append(mask)
                active.append(state.market_active)
                kinds.append(selected)
                previous = selected + 68
            else:
                slot = (position - 16) // 2
                context = self.market_quantity_context(self.market_norm(hidden[:, 0]))
                quantity_contexts.append(context)
                decision_active = state.market_active & (kinds[-1] >= 3)
                logits = self.quantity_logits(
                    context[:, None], kinds[-1][:, None], quote.mask[:, None]
                )[:, 0]
                mask = quote.mask
                selected = self._choose(
                    logits,
                    mask,
                    choice.market_quantities[:, slot],
                    choice.uniforms[:, position],
                    choice.temperatures,
                    choice.deterministic,
                )
                quantity_masks.append(mask)
                qactive.append(state.market_active & (kinds[-1] >= 3))
                quantities.append(selected)
                state = rules.apply_market(state, quote, selected)
                previous = selected + 90
            logprob, entropy = self._distribution(
                logits, mask, selected, decision_active, choice.temperatures
            )
            logprobs.append(logprob)
            entropies.append(entropy)
        masks = ReplayMasks(
            torch.stack(unit_masks, 1),
            torch.stack(kind_masks, 1),
            torch.stack(quantity_masks, 1),
            torch.arange(16, device=queries.device)[None] < state.units[:, None],
            torch.stack(active, 1),
            torch.stack(qactive, 1),
        )
        return CausalOutput(
            torch.stack(unit_logits, 1),
            torch.stack(kind_logits, 1),
            torch.stack(quantity_contexts, 1),
            torch.stack(units, 1),
            torch.stack(kinds, 1),
            torch.stack(quantities, 1),
            *masks,
            torch.stack(logprobs, 1),
            torch.stack(entropies, 1),
        )

    def _forward_market_causal(
        self, inputs: StructuredInputs, choice: CausalChoice
    ) -> CausalOutput:
        """Parallel unit readout, exact unit replay, then cached market decoding."""
        rules = self._rules()
        state = initial_ledger(choice.packed_ledger)
        queries, memory, valid, conditioning = self.trunk.encode(inputs)
        unit_logits = self.unit_head(queries[:, :16])
        units, unit_masks, logprobs, entropies = [], [], [], []
        for unit in range(16):
            mask = rules.unit_mask(state, unit)
            selected = self._choose(
                unit_logits[:, unit],
                mask,
                choice.unit_actions[:, unit],
                choice.uniforms[:, unit],
                choice.temperatures,
                choice.deterministic,
            )
            lp, entropy = self._distribution(
                unit_logits[:, unit],
                mask,
                selected,
                unit < state.units,
                choice.temperatures,
            )
            unit_masks.append(mask)
            units.append(selected)
            logprobs.append(lp)
            entropies.append(entropy)
            state = rules.apply_unit(state, unit, selected)

        batch = queries.shape[0]
        cache = [
            (
                keys[0].new_zeros(
                    (
                        batch,
                        self.config.attention_kv_heads,
                        MARKET_CACHE,
                        self.config.model_dim // self.config.attention_heads,
                    )
                ),
                keys[1].new_zeros(
                    (
                        batch,
                        self.config.attention_kv_heads,
                        MARKET_CACHE,
                        self.config.model_dim // self.config.attention_heads,
                    )
                ),
            )
            for keys in memory
        ]
        previous = units[-1]
        kinds, quantities = [], []
        kind_logits, quantity_contexts = [], []
        kind_masks, quantity_masks, active, qactive = [], [], [], []
        positions = self.trunk.positions[:MARKET_CACHE]
        for step in range(MARKET_DECISIONS):
            position = 16 + step
            hidden = (
                queries[:, position : position + 1]
                + self.trunk.position.weight[position]
                + self.trunk.previous_action(previous)[:, None]
                + self.trunk.ledger(ledger_features(state, None))[:, None]
            )
            for index, layer in enumerate(self.trunk.layers):
                hidden, cache[index] = layer.step(
                    hidden,
                    memory[index],
                    valid,
                    conditioning[index],
                    cache[index],
                    step,
                    positions,
                )
            if step % 2 == 0:
                slot = step // 2
                logits = self.market_kind(self.market_norm(hidden[:, 0]))
                mask = rules.market_kind_mask(state)
                decision_active = state.market_active
                selected = self._choose(
                    logits,
                    mask,
                    choice.market_kinds[:, slot],
                    choice.uniforms[:, position],
                    choice.temperatures,
                    choice.deterministic,
                )
                quote = rules.market_quote(state, selected)
                kind_logits.append(logits)
                kind_masks.append(mask)
                active.append(decision_active)
                kinds.append(selected)
                previous = selected + 68
            else:
                slot = step // 2
                context = self.market_quantity_context(self.market_norm(hidden[:, 0]))
                logits = self.quantity_logits(
                    context[:, None], kinds[-1][:, None], quote.mask[:, None]
                )[:, 0]
                mask = quote.mask
                decision_active = state.market_active & (kinds[-1] >= 3)
                selected = self._choose(
                    logits,
                    mask,
                    choice.market_quantities[:, slot],
                    choice.uniforms[:, position],
                    choice.temperatures,
                    choice.deterministic,
                )
                quantity_contexts.append(context)
                quantity_masks.append(mask)
                qactive.append(decision_active)
                quantities.append(selected)
                state = rules.apply_market(state, quote, selected)
                previous = selected + 90
            lp, entropy = self._distribution(
                logits, mask, selected, decision_active, choice.temperatures
            )
            logprobs.append(lp)
            entropies.append(entropy)
        return CausalOutput(
            unit_logits,
            torch.stack(kind_logits, 1),
            torch.stack(quantity_contexts, 1),
            torch.stack(units, 1),
            torch.stack(kinds, 1),
            torch.stack(quantities, 1),
            torch.stack(unit_masks, 1),
            torch.stack(kind_masks, 1),
            torch.stack(quantity_masks, 1),
            torch.arange(16, device=queries.device)[None] < state.units[:, None],
            torch.stack(active, 1),
            torch.stack(qactive, 1),
            torch.stack(logprobs, 1),
            torch.stack(entropies, 1),
        )
