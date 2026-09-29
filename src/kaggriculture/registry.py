"""Architecture registry: artifact tags -> model families (VIT Stage 0B).

Every actor artifact carries an ``architecture`` tag naming its family.
Loading and evaluation dispatch through this registry so checkpoints from
different architectures build the right model class and evaluate side by
side. Artifacts written before the tag existed are all convolutional entity
transformers, so a missing tag resolves to that family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from kaggriculture.causal_actor import CausalActor, CausalConfig
from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.lejepa_model import LejepaActor, LejepaConfig, LejepaCritic
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.strategic_actor import StrategicActor, StrategicConfig
from kaggriculture.structured import (
    JepaBelief,
    StructuredActor,
    StructuredConfig,
    StructuredCritic,
    StructuredCriticBelief,
    StructuredDecisionBelief,
)
from kaggriculture.tokens import SUPPORTED_OBSERVATION_SCHEMA_VERSIONS

CONV_ENTITY = "entity-cnn"
STRUCTURED = "structured"
ENTITY_ATTENTION = "entity-attention"
STRATEGIC = "strategic-plan"
CAUSAL = "causal-execution"
LEJEPA = "lejepa"
DEFAULT_ARCHITECTURE = CONV_ENTITY


def pair_towers(actor: nn.Module, critic: nn.Module) -> None:
    """Hand the critic whatever encoder the actor owns, for families that share one.

    A no-op everywhere but `lejepa`, where the critic holds no encoder of its own
    and raises rather than guess if it is asked to encode without one. Separate
    from construction because actors and critics are also built apart -- a league
    snapshot restores an actor alone -- and only a training pair needs wiring.
    """
    attach = getattr(critic, "attach_backbone", None)
    if attach is not None:
        attach(actor.trunk)


@dataclass(frozen=True)
class Architecture:
    """One registered actor family."""

    name: str
    config_class: type
    actor_class: type
    critic_class: type
    structured_inputs: bool = False
    full_belief: bool = False
    #: Belief types the PPO update reassembles from a minibatch forward's packed
    #: tail. Every family but `lejepa` reports the two decision-state fields and
    #: one pooled valuation state; the LeJEPA families additionally report the
    #: encoded observation their world-model objective predicts.
    actor_belief_class: type = StructuredDecisionBelief
    critic_belief_class: type = StructuredCriticBelief

    def build_actor(self, model_config: dict[str, Any]) -> nn.Module:
        return self.actor_class(self.build_config(model_config))

    def build_critic(self, model_config: dict[str, Any]) -> nn.Module:
        return self.critic_class(self.build_config(model_config))

    def build_config(self, model_config: dict[str, Any]) -> Any:
        """Decode a saved model configuration, rejecting stale observation schemas."""
        if self.structured_inputs and (
            model_config.get("observation_schema_version")
            not in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS
        ):
            raise ValueError(
                "stale structured observation schema; fresh encoding and training required"
            )
        if self.name == LEJEPA:
            # Saved policies predate these parameterized heads, the WDL
            # critic, the all-quantities interface and the affordance scorer.
            # New-model defaults must not silently change a checkpoint's
            # architecture.
            model_config = dict(model_config)
            model_config.setdefault("action_interface", 1)
            model_config.setdefault("unit_affordance_scorer", False)
            model_config.setdefault("unit_target_navigation", False)
            model_config.setdefault("market_resource_conditioning", False)
            model_config.setdefault("wdl_value", False)
        return self.config_class(**model_config)


ARCHITECTURES: dict[str, Architecture] = {
    CAUSAL: Architecture(
        name=CAUSAL,
        config_class=CausalConfig,
        actor_class=CausalActor,
        critic_class=EntityCritic,
        structured_inputs=True,
    ),
    STRATEGIC: Architecture(
        name=STRATEGIC,
        config_class=StrategicConfig,
        actor_class=StrategicActor,
        critic_class=EntityCritic,
        structured_inputs=True,
    ),
    CONV_ENTITY: Architecture(
        name=CONV_ENTITY,
        config_class=ModelConfig,
        actor_class=FarmActor,
        critic_class=DistributionalCritic,
    ),
    STRUCTURED: Architecture(
        name=STRUCTURED,
        config_class=StructuredConfig,
        actor_class=StructuredActor,
        critic_class=StructuredCritic,
        structured_inputs=True,
        full_belief=True,
    ),
    ENTITY_ATTENTION: Architecture(
        name=ENTITY_ATTENTION,
        config_class=EntityConfig,
        actor_class=EntityActor,
        critic_class=EntityCritic,
        structured_inputs=True,
    ),
    LEJEPA: Architecture(
        name=LEJEPA,
        config_class=LejepaConfig,
        actor_class=LejepaActor,
        critic_class=LejepaCritic,
        structured_inputs=True,
        actor_belief_class=JepaBelief,
        critic_belief_class=StructuredCriticBelief,
    ),
}


def resolve_architecture(payload_or_name: dict[str, Any] | str | None) -> Architecture:
    """Resolve an artifact payload (or explicit name) to its architecture."""
    if payload_or_name is None:
        name = DEFAULT_ARCHITECTURE
    elif isinstance(payload_or_name, str):
        name = payload_or_name
    else:
        name = str(payload_or_name.get("architecture") or DEFAULT_ARCHITECTURE)
    try:
        return ARCHITECTURES[name]
    except KeyError:
        known = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unknown actor architecture {name!r}; known: {known}") from None


def architecture_of(actor: nn.Module) -> Architecture:
    """Look up the registered family of a constructed actor."""
    for architecture in ARCHITECTURES.values():
        if type(actor) is architecture.actor_class:
            return architecture
    raise ValueError(f"actor type {type(actor).__name__} is not a registered architecture")


def architecture_of_config(config: Any) -> Architecture:
    """Look up the registered family of a constructed model configuration."""
    for architecture in ARCHITECTURES.values():
        if type(config) is architecture.config_class:
            return architecture
    raise ValueError(f"config type {type(config).__name__} is not a registered architecture")
