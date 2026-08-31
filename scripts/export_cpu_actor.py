#!/usr/bin/env python3
"""Export a fused structured checkpoint as a CPU-portable actor artifact."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import torch

from kaggriculture.inference import cpu_portable_actor_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a fused structured actor into its Kaggle CPU inference layout."
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


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if checkpoint_path == output or (output.exists() and output.samefile(checkpoint_path)):
        raise ValueError("output must not overwrite the training checkpoint")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    artifact = cpu_portable_actor_artifact(checkpoint, agent=args.agent)
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
