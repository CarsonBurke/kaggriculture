#!/usr/bin/env python3
"""Run one reproducible four-seed structured-VIT B1 arm and fixed play panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from kaggriculture.provenance import file_sha256, source_identity

DATASETS = (
    "data/bc-v16-current-mirror-64",
    "data/bc-v16-current-starter-64",
    "data/bc-v16-current-pass-64",
    "data/bc-v16-current-random-64",
)
OPPONENTS = ("public-v27", "public-v16")
ARM_ARGS: dict[str, tuple[str, ...]] = {
    "v0": (),
    "g10": ("--global-refresh-layers", "4", "--global-refresh-context", "economy"),
    "g11": ("--global-refresh-layers", "4", "--global-refresh-context", "all"),
    "g12": ("--global-refresh-layers", "2,5", "--global-refresh-context", "economy"),
    "g13": ("--split-clock-token", "true"),
    "g14": ("--global-modulation", "true"),
    "r10": ("--input-reinject-layers", "3,6"),
    "r11": ("--core-skip-source", "2", "--core-skip-target", "6"),
    "r13": ("--zero-init-branches", "true"),
    "r14": ("--mudd-lite", "true"),
    "q11": ("--fuse-market-decoder", "true"),
    "q12": ("--fuse-unit-decoder", "true"),
    "e10": ("--latents", "24"),
    "e11": ("--core-layers", "6"),
}
COMMON_TRAIN_ARGS = (
    "--architecture",
    "structured",
    "--model-dim",
    "80",
    "--seeds-per-dataset",
    "64",
    "--holdout-seeds",
    "8",
    "--epochs",
    "12",
    "--patience",
    "12",
    "--batch-size",
    "2048",
    "--run-length",
    "4",
    "--matrix-learning-rate",
    "4.2e-3",
    "--matrix-weight-decay",
    "1.2",
    "--adam-weight-decay",
    "0.005",
    "--compile-mode",
    "default",
    "--device",
    "cuda",
    "--encode-workers",
    "2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=tuple(ARM_ARGS))
    parser.add_argument("--artifact-root", type=Path, default=Path("runs/vit-b1-current"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_complete(path: Path, artifact: Path, opponent: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    summary = payload.get("summary")
    provenance = payload.get("artifact_provenance")
    return (
        payload.get("valid_for_selection") is True
        and payload.get("artifact") == str(artifact.resolve())
        and isinstance(provenance, dict)
        and provenance.get("sha256") == file_sha256(artifact)
        and payload.get("opponent_label") == opponent
        and payload.get("seed_start") == 10000
        and payload.get("seed_count") == 16
        and payload.get("workers") == 1
        and payload.get("batch_size") == 32
        and payload.get("device") == "cuda"
        and isinstance(summary, dict)
        and summary.get("games_requested") == 32
        and summary.get("games_completed") == 32
    )


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[1]
    python = source_root / ".venv/bin/python"
    datasets = tuple(source_root / path for path in DATASETS)
    for path in (python, *(dataset / "manifest.json" for dataset in datasets)):
        if not path.is_file():
            raise FileNotFoundError(path)

    arm_root = (source_root / args.artifact_root / args.arm).resolve()
    manifest: dict[str, Any] = {
        "format_version": 1,
        "arm": args.arm,
        "arm_args": list(ARM_ARGS[args.arm]),
        "common_train_args": list(COMMON_TRAIN_ARGS),
        "datasets": [str(path) for path in datasets],
        "dataset_manifest_sha256": {
            str(path): _hash(path / "manifest.json") for path in datasets
        },
        "training_seeds": [1, 2, 3, 4],
        "evaluation": {
            "opponents": list(OPPONENTS),
            "seed_start": 10000,
            "seed_count": 16,
            "both_seats": True,
            "batch_size": 32,
        },
        "source": source_identity(source_root),
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    print(encoded, end="", flush=True)
    if not args.dry_run:
        arm_root.mkdir(parents=True, exist_ok=True)
        manifest_path = arm_root / "manifest.json"
        if manifest_path.exists() and manifest_path.read_text() != encoded:
            raise RuntimeError(f"experiment manifest drift: {manifest_path}")
        manifest_path.write_text(encoded)

    for seed in manifest["training_seeds"]:
        seed_root = arm_root / f"seed-{seed}"
        artifact = seed_root / "bc-actor.pt"
        if artifact.is_file():
            print(f"reuse completed artifact {artifact}", flush=True)
        else:
            _run(
                [
                    str(python),
                    "-u",
                    str(source_root / "scripts/train_bc.py"),
                    "--dataset",
                    *(str(path) for path in datasets),
                    "--output",
                    str(seed_root),
                    *COMMON_TRAIN_ARGS,
                    "--seed",
                    str(seed),
                    *ARM_ARGS[args.arm],
                ],
                cwd=source_root,
                dry_run=args.dry_run,
            )

        for opponent in OPPONENTS:
            output = seed_root / f"eval-{opponent}.json"
            if _evaluation_complete(output, artifact, opponent):
                print(f"reuse completed evaluation {output}", flush=True)
                continue
            _run(
                [
                    str(python),
                    "-u",
                    str(source_root / "scripts/evaluate_checkpoint.py"),
                    "--artifact",
                    str(artifact),
                    "--opponent",
                    opponent,
                    "--seeds",
                    "16",
                    "--seed-start",
                    "10000",
                    "--workers",
                    "1",
                    "--batch-size",
                    "32",
                    "--torch-threads",
                    "1",
                    "--device",
                    "cuda",
                    "--output",
                    str(output),
                ],
                cwd=source_root,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
