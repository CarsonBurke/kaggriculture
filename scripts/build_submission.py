#!/usr/bin/env python3
"""Build a minimal actor-only Kaggriculture submission tarball."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch

from kaggriculture.inference import actor_artifact_from_checkpoint
from kaggriculture.provenance import file_sha256, require_source_identity, validate_run_provenance

PACKAGE_FILES = (
    "__init__.py",
    "actions.py",
    "constants.py",
    "encoding.py",
    "inference.py",
    "model.py",
    "policy.py",
    "provenance.py",
)
MAIN = '''"""Kaggriculture VAPO submission entrypoint."""
from pathlib import Path

import kaggriculture
from kaggriculture.inference import CheckpointAgent

_BUNDLE_ROOT = Path(kaggriculture.__file__).resolve().parent.parent
_AGENT = CheckpointAgent(_BUNDLE_ROOT / "model.pt")


def agent(obs):
    return _AGENT(obs)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
        help="successful finalist evaluation that cryptographically binds the checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_evaluation(
    contents: bytes,
    checkpoint_digest: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(contents.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("valid_for_selection") is not True:
        raise ValueError("submission requires a successful finalist evaluation")
    provenance = payload.get("artifact_provenance")
    if not isinstance(provenance, dict) or provenance.get("sha256") != checkpoint_digest:
        raise ValueError("finalist evaluation does not bind the selected checkpoint bytes")
    if provenance.get("source_identity") != source:
        raise ValueError("finalist evaluation source identity does not match the checkpoint")
    run_provenance = validate_run_provenance(provenance.get("run_provenance"))
    if run_provenance != payload.get("artifact_provenance", {}).get("run_provenance"):
        raise ValueError("finalist evaluation run provenance is inconsistent")
    if payload.get("opponent_label") != "public-v27":
        raise ValueError("submission requires finalist evaluation against the fixed public v27")
    if payload.get("paired_seats") is not True or payload.get("seed_count", 0) < 128:
        raise ValueError("submission requires at least 128 paired-seat finalist seed clusters")
    selection = payload.get("selection_provenance")
    if not isinstance(selection, dict) or selection.get("best_output_sha256") != checkpoint_digest:
        raise ValueError("finalist evaluation is not bound to checkpoint-selection evidence")
    selection_digest = selection.get("sha256")
    if (
        not isinstance(selection_digest, str)
        or len(selection_digest) != 64
        or any(character not in "0123456789abcdef" for character in selection_digest)
    ):
        raise ValueError("finalist evaluation has an invalid selection report digest")
    if selection.get("run_provenance") != run_provenance:
        raise ValueError("checkpoint-selection run provenance differs from finalist checkpoint")
    screening_start = selection.get("screening_seed_start")
    screening_count = selection.get("screening_seed_count")
    finalist_start = payload.get("seed_start")
    finalist_count = payload.get("seed_count")
    if (
        not all(type(value) is int and value >= 0 for value in (screening_start, finalist_start))
        or type(screening_count) is not int
        or screening_count < 1
        or max(screening_start, finalist_start)
        < min(screening_start + screening_count, finalist_start + finalist_count)
    ):
        raise ValueError("finalist evaluation reuses checkpoint-selection screening seeds")
    selected_v27 = selection.get("opponent_provenance", {}).get("public-v27")
    finalist_v27 = payload.get("opponent_provenance", {})
    if not isinstance(selected_v27, dict) or (
        selected_v27.get("kind") != "python_file"
        or selected_v27.get("sha256") != finalist_v27.get("sha256")
        or selected_v27.get("size_bytes") != finalist_v27.get("size_bytes")
    ):
        raise ValueError("finalist public v27 bytes differ from checkpoint selection")
    return payload


def build(checkpoint_path: Path, evaluation_report: Path, output: Path) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    evaluation_report = evaluation_report.expanduser().resolve()
    output = output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not evaluation_report.is_file():
        raise FileNotFoundError(evaluation_report)
    checkpoint_contents = checkpoint_path.read_bytes()
    checkpoint_digest = hashlib.sha256(checkpoint_contents).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_contents), map_location="cpu", weights_only=False)
    artifact = actor_artifact_from_checkpoint(checkpoint)
    source = require_source_identity(artifact["source_identity"])
    run_provenance = validate_run_provenance(artifact.get("run_provenance"))
    evaluation_contents = evaluation_report.read_bytes()
    evaluation = _load_evaluation(
        evaluation_contents,
        checkpoint_digest,
        source,
    )
    if evaluation["artifact_provenance"].get("run_provenance") != run_provenance:
        raise ValueError("finalist evaluation run provenance does not match the checkpoint")
    if evaluation["selection_provenance"].get("run_provenance") != run_provenance:
        raise ValueError("checkpoint-selection run provenance does not match the checkpoint")
    evaluation_digest = hashlib.sha256(evaluation_contents).hexdigest()
    artifact["training_checkpoint_sha256"] = checkpoint_digest
    run_provenance_sha256 = None if run_provenance is None else run_provenance["sha256"]
    artifact["run_provenance_sha256"] = run_provenance_sha256
    artifact["selection_report_sha256"] = evaluation["selection_provenance"]["sha256"]
    artifact["evaluation_report_sha256"] = evaluation_digest
    source_package = Path(__file__).parents[1] / "src" / "kaggriculture"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggriculture-submission-") as temporary_name:
        root = Path(temporary_name)
        package = root / "kaggriculture"
        package.mkdir()
        (root / "main.py").write_text(MAIN, encoding="utf-8")
        torch.save(artifact, root / "model.pt")
        (root / "evaluation.json").write_bytes(evaluation_contents)
        for name in PACKAGE_FILES:
            shutil.copy2(source_package / name, package / name)
        packaged_files = [root / "main.py", root / "model.pt", root / "evaluation.json"] + [
            package / name for name in PACKAGE_FILES
        ]
        files = {path.relative_to(root).as_posix(): file_sha256(path) for path in packaged_files}
        for name in PACKAGE_FILES:
            packaged = files[f"kaggriculture/{name}"]
            expected = source["files"].get(f"src/kaggriculture/{name}")
            if packaged != expected:
                raise ValueError(
                    f"submission package source does not match source identity: {name}"
                )
        manifest = {
            "format_version": 1,
            "source_identity": source,
            "run_provenance": run_provenance,
            "checkpoint": {
                "sha256": checkpoint_digest,
                "iteration": int(artifact["iteration"]),
                "run_provenance_sha256": run_provenance_sha256,
            },
            "evaluation": {
                "sha256": evaluation_digest,
                "selection_report_sha256": evaluation["selection_provenance"]["sha256"],
                "opponent": evaluation.get("opponent_label"),
                "seed_count": evaluation.get("seed_count"),
                "opponent_sha256": evaluation.get("opponent_provenance", {}).get("sha256"),
            },
            "files": dict(sorted(files.items())),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        descriptor, temporary_output_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary_output = Path(temporary_output_name)
        try:
            with tarfile.open(temporary_output, "w:gz") as archive:
                archive.add(root / "main.py", arcname="main.py")
                archive.add(root / "model.pt", arcname="model.pt")
                archive.add(root / "evaluation.json", arcname="evaluation.json")
                archive.add(root / "manifest.json", arcname="manifest.json")
                for name in PACKAGE_FILES:
                    archive.add(
                        package / name,
                        arcname=f"kaggriculture/{name}",
                        recursive=False,
                    )
            with temporary_output.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_output, output)
        finally:
            temporary_output.unlink(missing_ok=True)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build(args.checkpoint, args.evaluation_report, args.output)
    print(
        json.dumps(
            {
                "event": "submission_built",
                "output": str(args.output.expanduser().resolve()),
                "size_bytes": args.output.stat().st_size,
                "archive_sha256": file_sha256(args.output),
                "manifest": manifest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
