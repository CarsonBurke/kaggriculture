#!/usr/bin/env python3
"""Export a fused structured checkpoint as a CPU-portable actor artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from kaggriculture.inference import actor_artifact_from_checkpoint
from kaggriculture.registry import resolve_architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert fused structured MLP weights into the equivalent portable layout."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--agent",
        type=int,
        default=None,
        help="population member to export; required for a multi-member checkpoint",
    )
    return parser.parse_args()


def _portable_structured_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Replace every bias-free fused MLP pair with ordinary Linear weights."""
    converted: dict[str, Tensor] = {}
    up_prefixes: set[str] = set()
    down_prefixes: set[str] = set()

    def store(name: str, value: Tensor) -> None:
        if name in converted:
            raise ValueError(f"fused MLP conversion would overwrite state key {name!r}")
        converted[name] = value.detach().to(device="cpu", copy=True)

    for name, value in state.items():
        if name.endswith(".ffn.up_weight"):
            prefix = name.removesuffix("up_weight")
            store(f"{prefix}input.weight", value)
            store(f"{prefix}input.bias", value.new_zeros(value.shape[0]))
            up_prefixes.add(prefix)
        elif name.endswith(".ffn.down_weight"):
            prefix = name.removesuffix("down_weight")
            store(f"{prefix}output.weight", value.detach().T.contiguous())
            store(f"{prefix}output.bias", value.new_zeros(value.shape[1]))
            down_prefixes.add(prefix)
        else:
            store(name, value)
    if not up_prefixes:
        raise ValueError("artifact has fused_mlp enabled but contains no fused MLP weights")
    if up_prefixes != down_prefixes:
        missing_up = sorted(down_prefixes - up_prefixes)
        missing_down = sorted(up_prefixes - down_prefixes)
        raise ValueError(
            f"incomplete fused MLP state: missing up={missing_up}, missing down={missing_down}"
        )
    return converted


def export_cpu_actor(checkpoint: Mapping[str, Any], *, agent: int | None = None) -> dict[str, Any]:
    """Return a strict-loadable actor artifact whose MLP path runs on CPU."""
    artifact = actor_artifact_from_checkpoint(dict(checkpoint), agent=agent)
    if artifact["architecture"] != "structured":
        raise ValueError("CPU conversion applies only to structured actor artifacts")
    model_config = artifact.get("model_config")
    if not isinstance(model_config, Mapping) or not model_config.get("fused_mlp", False):
        raise ValueError("actor artifact does not use fused structured MLPs")
    # Reject malformed or hybrid source states before key conversion can mask
    # them behind a valid portable layout.
    fused_actor = resolve_architecture(artifact).build_actor(model_config)
    fused_actor.load_state_dict(artifact["actor"], strict=True)

    portable = copy.deepcopy(artifact)
    portable["model_config"] = dict(model_config)
    portable["model_config"]["fused_mlp"] = False
    portable["actor"] = _portable_structured_state(artifact["actor"])

    # Strict construction catches incomplete renames and future layout changes at
    # export time instead of producing an archive that fails inside Kaggle.
    actor = resolve_architecture(portable).build_actor(portable["model_config"])
    actor.load_state_dict(portable["actor"], strict=True)
    actor.eval()
    return portable


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if checkpoint_path == output or (output.exists() and output.samefile(checkpoint_path)):
        raise ValueError("output must not overwrite the training checkpoint")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    artifact = export_cpu_actor(checkpoint, agent=args.agent)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_output = Path(handle.name)
    try:
        torch.save(artifact, temporary_output)
        output_digest = hashlib.sha256(temporary_output.read_bytes()).hexdigest()
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "event": "cpu_actor_exported",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "output": str(output),
                "output_sha256": output_digest,
                "iteration": artifact["iteration"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
