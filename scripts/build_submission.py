#!/usr/bin/env python3
"""Build a minimal actor-only Kaggriculture submission tarball."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import torch

from kaggriculture.inference import actor_artifact_from_checkpoint

PACKAGE_FILES = (
    "__init__.py",
    "actions.py",
    "constants.py",
    "encoding.py",
    "inference.py",
    "model.py",
    "policy.py",
)
MAIN = '''"""Kaggriculture VAPO submission entrypoint."""
from pathlib import Path

from kaggriculture.inference import CheckpointAgent

_AGENT = CheckpointAgent(Path(__file__).with_name("model.pt"))


def agent(obs):
    return _AGENT(obs)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build(checkpoint_path: Path, output: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    artifact = actor_artifact_from_checkpoint(checkpoint)
    source_package = Path(__file__).parents[1] / "src" / "kaggriculture"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggriculture-submission-") as temporary_name:
        root = Path(temporary_name)
        package = root / "kaggriculture"
        package.mkdir()
        (root / "main.py").write_text(MAIN, encoding="utf-8")
        torch.save(artifact, root / "model.pt")
        for name in PACKAGE_FILES:
            shutil.copy2(source_package / name, package / name)
        with tarfile.open(output, "w:gz") as archive:
            archive.add(root / "main.py", arcname="main.py")
            archive.add(root / "model.pt", arcname="model.pt")
            archive.add(package, arcname="kaggriculture")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    build(args.checkpoint, args.output)
    print(f"built {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
