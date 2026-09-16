"""Architecture-aware model-configuration command line surface.

Every entry point that constructs a model — PPO training, behavior cloning,
the iteration benchmark — exposes the same structural hyperparameters.
Warm-starting from a cloned actor compares actor-affecting configuration;
recovery checkpoints compare the complete model configuration, including
critic-only fields.

The flags therefore default to ``None`` and the dataclass defaults are the
single source of truth: an entry point that receives no model flag builds
the registered family's default configuration. Passing a flag that belongs
to another family is an error rather
than a silent no-op, because an ignored ``--transformer-layers`` on a
structured run would otherwise report a model that was never trained.

The value bounds and atom count remain calibrated with the reward scale.
Gaussian smoothing is tunable independently of that support: changing the bin
count while retaining the same sigma/bin ratio also changes the return-space
bandwidth (HL-Gauss, arXiv:2403.03950, section 5.1.2).
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, get_args, get_origin, get_type_hints

from kaggriculture.registry import ARCHITECTURES, Architecture

CALIBRATED_MODEL_FIELDS = frozenset(("value_atoms", "value_min", "value_max"))
CRITIC_ONLY_MODEL_FIELDS = frozenset(
    (
        "scalar_value",
        "value_sigma_ratio",
        *CALIBRATED_MODEL_FIELDS,
        "critic_core_layers",
        "critic_latents",
        "critic_state_read",
        "per_entity_critic",
        "critic_readout_ffn",
    )
)


def actor_model_config(config: Any) -> dict[str, Any]:
    """Identity for actor-only loading, never training checkpoint recovery."""
    values = config if isinstance(config, dict) else config.to_dict()
    return {name: value for name, value in values.items() if name not in CRITIC_ONLY_MODEL_FIELDS}


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


def _default_summary(name: str, families: list[str]) -> str:
    """Render the per-family default, which is the whole point of the flag."""
    defaults = {family: getattr(ARCHITECTURES[family].config_class(), name) for family in families}
    if len(set(defaults.values())) == 1:
        return f"default {next(iter(defaults.values()))}"
    return "default " + ", ".join(f"{value} for {family}" for family, value in defaults.items())


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _field_parser(annotation: Any):
    origin = get_origin(annotation)
    if annotation is bool:
        return _parse_bool
    if origin is tuple and get_args(annotation) == (int, Ellipsis):

        def parse_tuple(value: str) -> tuple[int, ...]:
            try:
                return tuple(int(part) for part in value.split(",") if part)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"expected comma-separated integers, got {value!r}"
                ) from error

        return parse_tuple
    if annotation in (int, float, str):
        return annotation
    raise TypeError(f"unsupported model-config field type: {annotation!r}")


def add_model_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every family's structural flags, each defaulting to its dataclass value."""
    annotations = {
        architecture.name: get_type_hints(architecture.config_class)
        for architecture in ARCHITECTURES.values()
    }
    for name, families in _families_by_field().items():
        applies = (
            "every architecture" if len(families) == len(ARCHITECTURES) else ", ".join(families)
        )
        field_types = {annotations[family][name] for family in families}
        if len(field_types) != 1:
            raise TypeError(f"model-config field {name!r} has inconsistent family types")
        parser.add_argument(
            _flag(name),
            type=_field_parser(field_types.pop()),
            default=None,
            help=f"{name.replace('_', ' ')} ({applies}); {_default_summary(name, families)}",
        )


def model_config_from_args(architecture: Architecture, args: argparse.Namespace) -> Any:
    """Build this family's model configuration from the explicitly-passed flags."""
    accepted = set(model_config_fields(architecture))
    overrides: dict[str, Any] = {}
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


def model_config_arguments(architecture: Architecture, config: dict[str, Any]) -> list[str]:
    """Serialize the family's actual structural fields for an explicit launch."""
    arguments: list[str] = []
    for name in model_config_fields(architecture):
        value = config[name]
        if isinstance(value, (tuple, list)):
            encoded = ",".join(str(item) for item in value)
        elif isinstance(value, bool):
            encoded = str(value).lower()
        else:
            encoded = str(value)
        arguments.extend((_flag(name), encoded))
    return arguments
