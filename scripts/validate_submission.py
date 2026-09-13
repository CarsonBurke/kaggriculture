#!/usr/bin/env python3
"""Validate the exact submission archive in an isolated full-horizon Kaggle run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch

from kaggriculture.evaluation import artifact_seed_usage, validate_finalist_protocol
from kaggriculture.provenance import (
    file_sha256,
    validate_inference_equivalence,
    validate_run_provenance,
    validate_source_identity,
)

PUBLIC_V27_OPPONENT = Path("/var/tmp/kaggriculture-kaito-v27-main.py")
REQUIRED_MEMBERS = frozenset(
    {
        "main.py",
        "model.pt",
        "evaluation.json",
        "manifest.json",
        "kaggriculture/__init__.py",
        "kaggriculture/actions.py",
        "kaggriculture/constants.py",
        "kaggriculture/encoding.py",
        "kaggriculture/inference.py",
        "kaggriculture/evaluation.py",
        "kaggriculture/model.py",
        "kaggriculture/orientation.py",
        "kaggriculture/policy.py",
        "kaggriculture/provenance.py",
        "kaggriculture/registry.py",
        "kaggriculture/structured.py",
        "kaggriculture/triton_mlp.py",
        "kaggriculture/triton_norm.py",
        "kaggriculture/tokens.py",
    }
)


_PROBE = r"""
import json
import math
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
opponent = sys.argv[2]
seeds = int(sys.argv[3])
seed_start = int(sys.argv[4])
action_timeout = float(sys.argv[5])
sys.path.insert(0, str(root))
# Kaggle's CPU image includes an older Triton without this optional CUDA helper.
# Block the local workstation's newer copy so archive import exercises the same
# dependency surface instead of passing only because development has more packages.
sys.modules["triton.tools.tensor_descriptor"] = None


from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

main_path = root / "main.py"
raw_agent = get_last_callable(main_path.read_text(encoding="utf-8"), path=str(main_path))

games = []
all_action_seconds = []
for seed in range(seed_start, seed_start + seeds):
    for candidate_seat in (0, 1):
        action_seconds = []

        def timed_agent(observation):
            started = time.perf_counter()
            try:
                return raw_agent(observation)
            finally:
                action_seconds.append(time.perf_counter() - started)

        players = [opponent, opponent]
        players[candidate_seat] = timed_agent
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": seed},
            debug=False,
        )
        environment.run(players)
        final = environment.steps[-1]
        candidate = final[candidate_seat]
        rival = final[1 - candidate_seat]
        if not environment.done or len(environment.steps) != 720:
            raise RuntimeError(
                f"incomplete game seed={seed} seat={candidate_seat}: "
                f"done={environment.done}, steps={len(environment.steps)}"
            )
        if candidate.status != "DONE" or rival.status != "DONE":
            raise RuntimeError(
                f"bad status seed={seed} seat={candidate_seat}: "
                f"candidate={candidate.status}, opponent={rival.status}"
            )
        rewards = (float(candidate.reward), float(rival.reward))
        if not all(math.isfinite(value) for value in rewards):
            raise RuntimeError(f"non-finite reward seed={seed} seat={candidate_seat}")
        if not action_seconds or max(action_seconds) >= action_timeout:
            raise RuntimeError(
                f"action timeout seed={seed} seat={candidate_seat}: "
                f"max={max(action_seconds, default=float('inf')):.6f}s"
            )
        all_action_seconds.extend(action_seconds)
        games.append(
            {
                "seed": seed,
                "candidate_seat": candidate_seat,
                "candidate_reward": rewards[0],
                "opponent_reward": rewards[1],
                "margin": rewards[0] - rewards[1],
                "actions": len(action_seconds),
                "action_seconds_max": max(action_seconds),
            }
        )

print(
    json.dumps(
        {
            "valid": True,
            "games": games,
            "games_completed": len(games),
            "action_calls": len(all_action_seconds),
            "action_seconds_mean": sum(all_action_seconds) / len(all_action_seconds),
            "action_seconds_max": max(all_action_seconds),
        },
        sort_keys=True,
        allow_nan=False,
    )
)
"""


def _opponent(value: str) -> str:
    if value in {"pass", "random", "starter"}:
        return value
    if value in {"v27", "public-v27"}:
        path = PUBLIC_V27_OPPONENT.resolve()
    else:
        path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _extract(archive_path: Path, destination: Path) -> tuple[list[str], dict[str, Any]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("submission archive contains duplicate members")
        missing = sorted(REQUIRED_MEMBERS - set(names))
        if missing:
            raise ValueError(f"submission archive is incomplete: missing={missing}")
        extra = sorted(set(names) - REQUIRED_MEMBERS)
        if extra:
            raise ValueError(f"submission archive contains unbound members: extra={extra}")
        non_files = sorted(member.name for member in members if not member.isfile())
        if non_files:
            raise ValueError(f"submission archive contains non-regular members: {non_files}")
        archive.extractall(destination, filter="data")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    required_keys = {
        "format_version",
        "source_identity",
        "run_provenance",
        "checkpoint",
        "evaluation",
        "files",
    }
    allowed_keys = required_keys | {
        "bundle_smoke",
        "inference_equivalence",
        "strength_gate",
    }
    if not isinstance(manifest, dict) or not required_keys <= set(manifest) <= allowed_keys:
        raise ValueError("submission manifest has an invalid schema")
    if manifest["format_version"] != 2:
        raise ValueError(f"unsupported submission manifest: {manifest['format_version']}")
    source = validate_source_identity(manifest["source_identity"])
    # The archive already hashed its payload against this identity. The live
    # checkout may have moved a validator or probe; that must not refuse a
    # bundle whose packaged bytes still match the recorded tree.

    run_provenance = validate_run_provenance(manifest["run_provenance"])
    files = manifest["files"]
    expected_files = REQUIRED_MEMBERS - {"manifest.json"}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("submission manifest does not bind exactly the required payload files")
    for relative, expected_digest in files.items():
        actual_digest = file_sha256(destination / relative)
        if actual_digest != expected_digest:
            raise ValueError(f"submission payload digest mismatch: {relative}")
        if relative.startswith("kaggriculture/"):
            source_relative = f"src/{relative}"
            if source["files"].get(source_relative) != actual_digest:
                raise ValueError(
                    f"submission package does not match source identity: {source_relative}"
                )

    checkpoint = manifest["checkpoint"]
    evaluation_binding = manifest["evaluation"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "sha256",
        "iteration",
        "run_provenance_sha256",
        "agent",
    }:
        raise ValueError("submission checkpoint binding has an invalid schema")
    if not isinstance(evaluation_binding, dict) or not {
        "sha256",
        "selection_report_sha256",
        "opponent",
        "seed_count",
        "opponent_sha256",
    } <= set(evaluation_binding):
        raise ValueError("submission evaluation binding has an invalid schema")

    evaluation = json.loads((destination / "evaluation.json").read_text(encoding="utf-8"))
    if file_sha256(destination / "evaluation.json") != evaluation_binding["sha256"]:
        raise ValueError("submission finalist evaluation digest does not match its binding")
    provenance = evaluation.get("artifact_provenance", {})
    if evaluation.get("valid_for_selection") is not True:
        raise ValueError("submission finalist evaluation is not valid for selection")
    if evaluation.get("device") != "cpu":
        raise ValueError("submission finalist evaluation did not run on Kaggle's CPU backend")
    if evaluation.get("opponent_label") != "public-v27":
        raise ValueError("submission finalist evaluation did not use the fixed public v27")
    if evaluation.get("paired_seats") is not True or evaluation.get("seed_count", 0) < 32:
        raise ValueError("submission finalist evaluation is too small or not paired by seat")
    selection = evaluation.get("selection_provenance")
    validate_finalist_protocol(evaluation)
    if (
        evaluation_binding.get("seed_protocol") != evaluation["seed_protocol"]
        or evaluation_binding.get("statistical_selection") != selection["statistical_selection"]
        or evaluation_binding.get("score_confidence") != evaluation["summary"]["score_confidence"]
    ):
        raise ValueError("submission statistical protocol binding is inconsistent")
    if selection is not None:
        if (
            not isinstance(selection, dict)
            or selection.get("best_output_sha256") != checkpoint["sha256"]
        ):
            raise ValueError("submission finalist evaluation lacks matching selection evidence")
        screening_start = selection.get("screening_seed_start")
        screening_count = selection.get("screening_seed_count")
        finalist_start = evaluation.get("seed_start")
        finalist_count = evaluation.get("seed_count")
        if (
            not all(
                type(value) is int and value >= 0 for value in (screening_start, finalist_start)
            )
            or type(screening_count) is not int
            or screening_count < 1
            or max(screening_start, finalist_start)
            < min(screening_start + screening_count, finalist_start + finalist_count)
        ):
            raise ValueError("submission finalist seeds overlap screening seeds")
    finalist_opponent = evaluation.get("opponent_provenance", {})
    if isinstance(selection, dict):
        selected_opponent = selection.get("opponent_provenance", {}).get("public-v27")
        if (
            not isinstance(selected_opponent, dict)
            or selected_opponent.get("sha256") != finalist_opponent.get("sha256")
            or selected_opponent.get("size_bytes") != finalist_opponent.get("size_bytes")
            or finalist_opponent.get("sha256") != evaluation_binding["opponent_sha256"]
        ):
            raise ValueError("submission public v27 provenance chain is inconsistent")
    elif finalist_opponent.get("sha256") != evaluation_binding["opponent_sha256"]:
        raise ValueError("submission public v27 provenance chain is inconsistent")
    artifact = torch.load(destination / "model.pt", map_location="cpu", weights_only=False)
    if provenance.get("seed_usage") != artifact_seed_usage(artifact):
        raise ValueError("submission model seed exposure differs from finalist evidence")
    artifact_source = validate_source_identity(artifact.get("source_identity"))
    witness = manifest.get("inference_equivalence")
    validated_witness = None if witness is None else validate_inference_equivalence(witness)
    if provenance.get("sha256") != checkpoint["sha256"]:
        raise ValueError("submission finalist evaluation targets different checkpoint bytes")
    if provenance.get("source_identity") != artifact_source:
        raise ValueError("submission finalist evaluation has a different source identity")
    if provenance.get("agent") != checkpoint["agent"]:
        raise ValueError("submission finalist evaluation measured a different population member")
    if artifact_source != source and (
        validated_witness is None
        or validated_witness["artifact_sha256"] != checkpoint["sha256"]
        or validated_witness["candidate_identity"] != source["sha256"]
        or validated_witness["expected_identity"] != artifact_source["sha256"]
    ):
        raise ValueError("submission model has a different source identity")

    if artifact.get("run_provenance") != run_provenance:
        raise ValueError("submission model has different run provenance")
    run_provenance_sha256 = None if run_provenance is None else run_provenance["sha256"]
    if run_provenance_sha256 != checkpoint["run_provenance_sha256"]:
        raise ValueError("submission manifest run provenance digest is inconsistent")
    if artifact.get("run_provenance_sha256") != checkpoint["run_provenance_sha256"]:
        raise ValueError("submission model has a different run provenance digest")
    if provenance.get("run_provenance") != run_provenance:
        raise ValueError("submission finalist evaluation has different run provenance")
    if isinstance(selection, dict):
        if selection.get("run_provenance") != run_provenance:
            raise ValueError("submission checkpoint selection has different run provenance")
        if selection.get("sha256") != evaluation_binding["selection_report_sha256"]:
            raise ValueError(
                "submission finalist evaluation has a different selection report binding"
            )
    elif evaluation_binding["selection_report_sha256"] is not None:
        raise ValueError("submission without selection evidence has a selection report binding")
    if artifact.get("training_checkpoint_sha256") != checkpoint["sha256"]:
        raise ValueError("submission model has a different training checkpoint binding")
    if artifact.get("checkpoint_agent") != checkpoint["agent"]:
        raise ValueError("submission model has a different population member binding")
    if artifact.get("selection_report_sha256") != evaluation_binding["selection_report_sha256"]:
        raise ValueError("submission model has a different selection report binding")
    if artifact.get("evaluation_report_sha256") != evaluation_binding["sha256"]:
        raise ValueError("submission model has a different finalist evaluation binding")
    if artifact.get("iteration") != checkpoint["iteration"]:
        raise ValueError("submission model iteration does not match the manifest")
    return names, manifest


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--opponent", default="v27")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=30_000_000)
    parser.add_argument("--action-timeout", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.archive.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    if not math.isfinite(args.action_timeout) or args.action_timeout <= 0.0:
        raise ValueError("action timeout must be finite and positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise ValueError("process timeout must be finite and positive")
    opponent = _opponent(args.opponent)
    archive_contents = archive_path.read_bytes()
    digest = hashlib.sha256(archive_contents).hexdigest()
    with tempfile.TemporaryDirectory(prefix="kaggriculture-bundle-validation-") as name:
        temporary_root = Path(name)
        archive_snapshot = temporary_root / "submission.tar.gz"
        archive_snapshot.write_bytes(archive_contents)
        root = temporary_root / "payload"
        root.mkdir()
        members, manifest = _extract(archive_snapshot, root)
        opponent_path = Path(opponent)
        if not opponent_path.is_file():
            raise ValueError("release validation requires the selected public v27 Python file")
        opponent_contents = opponent_path.read_bytes()
        opponent_digest = hashlib.sha256(opponent_contents).hexdigest()
        expected_opponent_digest = manifest["evaluation"]["opponent_sha256"]
        if opponent_digest != expected_opponent_digest:
            raise ValueError(
                "validation opponent bytes do not match finalist evidence: "
                f"{opponent_digest} != {expected_opponent_digest}"
            )
        opponent_snapshot = temporary_root / "opponent.py"
        opponent_snapshot.write_bytes(opponent_contents)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                _PROBE,
                str(root),
                str(opponent_snapshot),
                str(args.seeds),
                str(args.seed_start),
                str(args.action_timeout),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    lines = [line for line in completed.stdout.splitlines() if line]
    if not lines:
        raise RuntimeError("isolated bundle validation produced no result")
    result = json.loads(lines[-1])
    if result.get("valid") is not True or result.get("games_completed") != args.seeds * 2:
        raise RuntimeError(f"isolated bundle validation failed: {result}")
    payload = {
        "valid": True,
        "archive": str(archive_path),
        "archive_sha256": digest,
        "archive_size_bytes": len(archive_contents),
        "archive_members": members,
        "manifest": manifest,
        "opponent": opponent,
        "opponent_sha256": opponent_digest,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "isolated_result": result,
        "captured_stderr": completed.stderr[-4_000:] or None,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output is not None:
        _write_atomic(args.output, payload)


if __name__ == "__main__":
    main()
