"""Architecture-aware model-configuration command line surface.

Every entry point that constructs a model — VAPO training, behavior cloning,
the iteration benchmark — exposes the same structural hyperparameters, and
they must agree exactly: warm-starting a run from a cloned actor compares
``model_config`` dictionaries for equality, so a flag whose default silently
differs between two scripts makes the artifact unusable.

The flags therefore default to ``None`` and the dataclass defaults are the
single source of truth: an entry point that receives no model flag builds
exactly ``ModelConfig()`` / ``StructuredConfig()``, whichever family was
selected. Passing a flag that belongs to another family is an error rather
than a silent no-op, because an ignored ``--transformer-layers`` on a
structured run would otherwise report a model that was never trained.

The value-support fields are deliberately absent: they are calibrated with
the reward scale, not tuned per run.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any

from kaggriculture.registry import ARCHITECTURES, Architecture

CALIBRATED_MODEL_FIELDS = frozenset(("value_atoms", "value_min", "value_max", "value_sigma_ratio"))


def _flag(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def model_config_fields(architecture: Architecture) -> tuple[str, ...]:
    """Structural configuration fields this family exposes as flags."""
    return tuple(
        field.name
        for field in fields(architecture.config_class)
        if field.name not in CALIBRATED_MODEL_FIELDS
    )


def _families_by_field() -> dict[str, list[str]]:
    """Map every exposed field to the families that accept it, in registry order."""
    families: dict[str, list[str]] = {}
    for architecture in ARCHITECTURES.values():
        for name in model_config_fields(architecture):
            families.setdefault(name, []).append(architecture.name)
    return families


def add_model_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every family's structural flags, each defaulting to its dataclass value."""
    for name, families in _families_by_field().items():
        applies = "every architecture" if len(families) == len(ARCHITECTURES) else families[0]
        parser.add_argument(
            _flag(name),
            type=int,
            default=None,
            help=f"{name.replace('_', ' ')} ({applies}); defaults to the architecture default",
        )


def model_config_from_args(architecture: Architecture, args: argparse.Namespace) -> Any:
    """Build this family's model configuration from the explicitly-passed flags."""
    accepted = set(model_config_fields(architecture))
    overrides: dict[str, int] = {}
    foreign: list[str] = []
    for name in _families_by_field():
        value = getattr(args, name, None)
        if value is None:
            continue
        if name in accepted:
            overrides[name] = value
        else:
            foreign.append(_flag(name))
    if foreign:
        raise ValueError(
            f"{', '.join(sorted(foreign))} do not apply to the {architecture.name} architecture"
        )
    return architecture.config_class(**overrides)
