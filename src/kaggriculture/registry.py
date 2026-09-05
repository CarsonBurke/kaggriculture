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

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic
from kaggriculture.tokens import OBSERVATION_SCHEMA_VERSION

CONV_ENTITY = "entity-cnn"
STRUCTURED = "structured"
DEFAULT_ARCHITECTURE = CONV_ENTITY


@dataclass(frozen=True)
class Architecture:
    """One registered actor family."""

    name: str
    config_class: type
    actor_class: type
    critic_class: type

    def build_actor(self, model_config: dict[str, Any]) -> nn.Module:
        return self.actor_class(self.build_config(model_config))

    def build_critic(self, model_config: dict[str, Any]) -> nn.Module:
        return self.critic_class(self.build_config(model_config))

    def build_config(self, model_config: dict[str, Any]) -> Any:
        """Decode a saved model configuration, rejecting stale observation schemas."""
        if self.name == STRUCTURED and (
            model_config.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "stale structured observation schema; fresh encoding and training required"
            )
        return self.config_class(**model_config)


ARCHITECTURES: dict[str, Architecture] = {
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
