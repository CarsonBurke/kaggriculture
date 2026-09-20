#!/usr/bin/env python3
"""Run one fully matched BC ablation arm and its fixed evaluation panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.provenance import file_sha256
from kaggriculture.provenance import source_identity as pipeline_source_identity

SEEDS = tuple(range(16))
DATASETS = (
    "data/bc-v16-current-mirror-64",
    "data/bc-v16-current-starter-64",
    "data/bc-v16-current-pass-64",
    "data/bc-v16-current-random-64",
)
OPPONENTS = ("starter", "public-v16", "public-v27")
COMMON_TRAIN_ARGS = (
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
    "--device",
    "cuda",
    "--encode-workers",
    "2",
    "--compile-mode",
    "default",
    "--latent-dynamics-coefficient",
    "0",
    "--latent-horizon",
    "2",
)
ARM_ARGS = {
    "control": ("--architecture", "entity-cnn", "--latent-decode-coefficient", "0"),
    "nextlat": ("--architecture", "entity-cnn", "--latent-decode-coefficient", "0.5"),
    # model_dim=80 gives 1.320M actor parameters versus the control's 1.421M;
    # the structured default at 128 would be 3.331M and confound representation
    # with more than twice the capacity. The 7.1% shortfall is the closest valid
    # width because four heads with axial RoPE require a multiple of sixteen.
    "structured": (
        "--architecture",
        "structured",
        "--model-dim",
        "80",
        "--latent-decode-coefficient",
        "0",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=tuple(ARM_ARGS))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("runs/matched-bc-20260825"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_identity(source_root: Path) -> dict[str, Any]:
    return pipeline_source_identity(source_root)


def _evaluation_parallelism(opponent: str) -> tuple[int, int]:
    _, runnable = normalize_opponent(opponent)
    # Lockstep is opt-in at the evaluator layer until CUDA evidence is bound
    # to this exact source, batch size, and actor architecture.
    return (1, 1) if runnable in BUILTIN_OPPONENTS else (8, 1)


def _opponent_provenance(opponent: str) -> dict[str, Any]:
    label, runnable = normalize_opponent(opponent)
    if runnable in BUILTIN_OPPONENTS:
        return {"kind": "builtin", "name": label}
    path = Path(runnable).resolve()
    return {
        "kind": "python_file",
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _evaluation_complete(path: Path, artifact: Path, opponent: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    workers, batch_size = _evaluation_parallelism(opponent)
    summary = payload.get("summary")
    provenance = payload.get("artifact_provenance")
    opponent_provenance = _opponent_provenance(opponent)
    return (
        payload.get("valid_for_selection") is True
        and payload.get("artifact") == str(artifact.resolve())
        and isinstance(provenance, dict)
        and provenance.get("sha256") == _sha256(artifact.read_bytes())
        and payload.get("opponent_label") == opponent
        and payload.get("opponent_provenance") == opponent_provenance
        and payload.get("seed_start") == 5000
        and payload.get("seed_count") == 8
        and payload.get("workers") == workers
        and payload.get("batch_size") == batch_size
        and payload.get("device") == "cuda"
        and isinstance(summary, dict)
        and summary.get("games_requested") == 16
        and summary.get("games_completed") == 16
    )


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[1]
    while not (repository_root / "data").is_dir():
        if repository_root.parent == repository_root:
            raise RuntimeError("could not locate the repository data directory")
        repository_root = repository_root.parent
    # Worktrees live below <repo>/worktrees/<name>; the first ancestor with data/
    # is the shared artifact owner. Main resolves to itself.
    artifact_root = args.artifact_root
    if not artifact_root.is_absolute():
        artifact_root = repository_root / artifact_root
    arm_root = artifact_root / args.arm
    python = source_root / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"missing worktree interpreter: {python}")

    datasets = tuple(repository_root / path for path in DATASETS)
    for dataset in datasets:
        if not (dataset / "manifest.json").is_file():
            raise FileNotFoundError(f"missing dataset manifest: {dataset}")

    manifest = {
        "arm": args.arm,
        "source": _source_identity(source_root),
        "seeds": list(SEEDS),
        "datasets": [str(path) for path in datasets],
        "dataset_manifest_sha256": {
            str(path): _sha256((path / "manifest.json").read_bytes()) for path in datasets
        },
        "opponents": list(OPPONENTS),
        "opponent_provenance": {opponent: _opponent_provenance(opponent) for opponent in OPPONENTS},
        "evaluation_seed_start": 5000,
        "evaluation_seed_clusters": 8,
        "common_train_args": list(COMMON_TRAIN_ARGS),
        "arm_args": list(ARM_ARGS[args.arm]),
        "evaluation_parallelism": {
            opponent: dict(
                zip(("workers", "batch_size"), _evaluation_parallelism(opponent), strict=True)
            )
            for opponent in OPPONENTS
        },
    }
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if not args.dry_run:
        arm_root.mkdir(parents=True, exist_ok=True)
        manifest_path = arm_root / "manifest.json"
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if manifest_path.exists() and manifest_path.read_text() != encoded:
            raise RuntimeError(
                f"refusing to change an existing experiment manifest: {manifest_path}"
            )
        manifest_path.write_text(encoded)

    for seed in SEEDS:
        seed_root = arm_root / f"seed-{seed:02d}"
        artifact = seed_root / "bc-actor.pt"
        if not artifact.is_file():
            train = [
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
            ]
            _run(train, cwd=source_root, dry_run=args.dry_run)
        else:
            print(f"reuse complete training artifact {artifact}", flush=True)

        for opponent in OPPONENTS:
            workers, batch_size = _evaluation_parallelism(opponent)
            output = seed_root / f"eval-{opponent}.json"
            if _evaluation_complete(output, artifact, opponent):
                print(f"reuse complete evaluation {output}", flush=True)
                continue
            evaluate = [
                str(python),
                "-u",
                str(source_root / "scripts/evaluate_checkpoint.py"),
                "--artifact",
                str(artifact),
                "--opponent",
                opponent,
                "--seeds",
                "8",
                "--seed-start",
                "5000",
                "--workers",
                str(workers),
                "--batch-size",
                str(batch_size),
                "--torch-threads",
                "1",
                "--device",
                "cuda",
                "--output",
                str(output),
            ]
            _run(evaluate, cwd=source_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
