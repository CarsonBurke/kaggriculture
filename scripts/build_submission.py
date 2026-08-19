#!/usr/bin/env python3
"""Build a minimal actor-only Kaggriculture submission tarball.

Two independent gates must pass. The provenance gate binds the shipped bytes to
the checkpoint, the run, and the seeds that selected it. The strength gate reads
what the agent actually scored, which the provenance gate cannot see: a report's
`valid_for_selection` means the evaluation itself ran correctly, not that the
agent won anything. Both finalists in `evaluations/` carry `score_rate` 0.0 with
0 wins over 256 seats against `public-v27` and are stamped valid, so provenance
alone has already packaged agents that lose every game they play.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from kaggriculture.inference import actor_artifact_from_checkpoint
from kaggriculture.opponents import BUILTIN_OPPONENTS
from kaggriculture.provenance import file_sha256, require_source_identity, validate_run_provenance

# `starter` actually farms; `pass` and `random` do not, so it is the only
# built-in whose defeat is evidence of competence rather than of merely acting.
_REQUIRED_BUILTIN = "starter"

PACKAGE_FILES = (
    "__init__.py",
    "actions.py",
    "constants.py",
    "encoding.py",
    "inference.py",
    "model.py",
    "policy.py",
    "provenance.py",
    "registry.py",
    "structured.py",
    "tokens.py",
)
MAIN = '''"""Kaggriculture PPO submission entrypoint."""
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
    parser.add_argument(
        "--builtin-evaluation-report",
        type=Path,
        action="append",
        default=[],
        metavar="REPORT",
        help=(
            "evaluation against a built-in reference agent, repeatable; a report for "
            f"{_REQUIRED_BUILTIN!r} is mandatory because it is the strongest built-in and "
            "the one this pipeline has measurably lost to"
        ),
    )
    parser.add_argument(
        "--minimum-score-rate",
        type=float,
        default=0.5,
        help=(
            "floor on the finalist score rate against the public opponent. The default "
            "refuses to ship an agent that loses more than it wins; lower it only as a "
            "deliberate deadline decision, which then appears in the shipped manifest"
        ),
    )
    parser.add_argument(
        "--minimum-builtin-score-rate",
        type=float,
        default=0.9,
        help=(
            "floor on the score rate against each built-in. These are heuristics a "
            "competent farmer beats nearly always, so the default leaves only enough "
            "room for genuine seed variance"
        ),
    )
    parser.add_argument(
        "--minimum-builtin-seed-count",
        type=int,
        default=32,
        help="paired-seat seed clusters required per built-in report, so a pass is not a fluke",
    )
    parser.add_argument(
        "--inference-equivalence",
        type=Path,
        help=(
            "measured witness from scripts/audit_inference_equivalence.py admitting a "
            "checkpoint whose bound tree has moved without changing its inference surface"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _score_rate(payload: dict[str, Any], context: str) -> float:
    """Read a report's score rate, refusing anything that is not a real number.

    A missing or null rate is what an aborted evaluation writes, and `float(None)`
    would raise a TypeError far from the cause, so it is named here instead.
    """
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{context} evaluation has no summary")
    rate = summary.get("score_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate):
        raise ValueError(f"{context} evaluation has no finite score rate")
    return float(rate)


def _load_builtin_evaluation(
    contents: bytes,
    checkpoint_digest: str,
    source: dict[str, Any],
    minimum_score_rate: float,
    minimum_seed_count: int,
) -> tuple[str, float]:
    """Check one built-in report and return its label and score rate.

    Deliberately lighter than the finalist gate: the selection-provenance and
    seed-disjointness machinery there exists to stop a checkpoint being chosen and
    validated on the same seeds, and these reports choose nothing. What they must
    still do is bind these exact checkpoint bytes, or they would license shipping a
    different agent than the one that was measured.
    """
    payload = json.loads(contents.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("valid_for_selection") is not True:
        raise ValueError("built-in evaluation did not complete successfully")
    label = payload.get("opponent_label")
    if label not in BUILTIN_OPPONENTS:
        raise ValueError(f"built-in evaluation names a non-built-in opponent: {label!r}")
    provenance = payload.get("artifact_provenance")
    if not isinstance(provenance, dict) or provenance.get("sha256") != checkpoint_digest:
        raise ValueError(f"{label} evaluation does not bind the selected checkpoint bytes")
    if provenance.get("source_identity") != source:
        raise ValueError(f"{label} evaluation source identity does not match the checkpoint")
    if payload.get("paired_seats") is not True:
        raise ValueError(f"{label} evaluation must use paired seats to cancel the seat advantage")
    seed_count = payload.get("seed_count", 0)
    if not isinstance(seed_count, int) or seed_count < minimum_seed_count:
        raise ValueError(
            f"{label} evaluation has {seed_count} seed clusters, below the required "
            f"{minimum_seed_count}"
        )
    rate = _score_rate(payload, label)
    if rate < minimum_score_rate:
        raise ValueError(
            f"submission scores {rate:.4f} against the built-in {label}, below the required "
            f"{minimum_score_rate:.4f}"
        )
    return label, rate


def _load_evaluation(
    contents: bytes,
    checkpoint_digest: str,
    source: dict[str, Any],
    minimum_score_rate: float,
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
    # Last, so a report that fails provenance is reported as such rather than as a
    # weak score: the bytes must be trustworthy before the number means anything.
    rate = _score_rate(payload, "finalist")
    if rate < minimum_score_rate:
        raise ValueError(
            f"submission scores {rate:.4f} against public-v27, below the required "
            f"{minimum_score_rate:.4f}"
        )
    return payload


#: Long enough for the agent to be asked for several distinct decisions and short
#: enough to stay a wiring check. Strength is the evaluation reports' job; this
#: only answers whether the packaged bundle runs at all.
_SMOKE_STEPS = 40

#: Run inside the bundle, against the real engine, with the repository nowhere on
#: the path. Every failure mode it catches is silent otherwise: an import error, a
#: missing packaged module, a checkpoint the packaged loader refuses, or an agent
#: that returns something the interpreter reads as "do nothing" -- all of which
#: bank the untouched starting money and look like a merely weak submission.
_SMOKE = """import json, sys
from kaggle_environments import make

import main

environment = make(
    "kaggriculture",
    configuration={"episodeSteps": %(steps)d, "seed": 90_017},
    debug=True,
)
environment.run([main.agent, "pass"])
seat = environment.steps[-1][0]
submitted = [
    step[0].action
    for step in environment.steps[1:]
    if isinstance(step[0].action, dict)
]


def _acts(action):
    commands = [action.get("farmer")] + list(action.get("hands") or [])
    return bool(action.get("market")) or any(c != ["PASS"] for c in commands if c)


acting = [action for action in submitted if _acts(action)]
print(json.dumps({
    "status": seat.status,
    "reward": seat.reward,
    "submitted": len(submitted),
    "acting": len(acting),
    "error": seat.info.get("error") if isinstance(seat.info, dict) else None,
}))
"""


def _smoke_test(root: Path) -> dict[str, Any]:
    """Play the packaged bundle against the engine before it can be shipped.

    The bundle is the artifact that scores, not the checkpoint it was cut from,
    and every way it can be broken is quiet: `kaggle_environments` swallows an
    agent exception, seats a PASS for the rest of the episode, and still reports
    DONE with the starting bank intact. A submission that fails this way is
    indistinguishable from a weak one until a day of leaderboard budget is gone.
    """
    script = root / "_smoke.py"
    script.write_text(_SMOKE % {"steps": _SMOKE_STEPS}, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, script.name],
        cwd=root,
        capture_output=True,
        text=True,
        # The bundle has to satisfy its own imports from its own directory; an
        # inherited path would let the repository stand in for a module the
        # package forgot to ship.
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        },
    )
    script.unlink()
    if completed.returncode != 0:
        raise ValueError(
            f"submission bundle failed to run:\n{completed.stdout}\n{completed.stderr}"
        )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result["error"]:
        raise ValueError(f"submission bundle raised inside the engine: {result['error']}")
    if result["status"] != "DONE":
        raise ValueError(f"submission bundle did not finish its episode: {result['status']}")
    if result["submitted"] != _SMOKE_STEPS - 1:
        raise ValueError(
            f"submission bundle answered {result['submitted']} of {_SMOKE_STEPS - 1} steps"
        )
    if not result["acting"]:
        raise ValueError("submission bundle passed on every step: it would bank the start money")
    return result


def build(
    checkpoint_path: Path,
    evaluation_report: Path,
    output: Path,
    *,
    builtin_evaluation_reports: Sequence[Path] = (),
    minimum_score_rate: float = 0.0,
    minimum_builtin_score_rate: float = 0.0,
    minimum_builtin_seed_count: int = 1,
    inference_equivalence: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    evaluation_report = evaluation_report.expanduser().resolve()
    output = output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not evaluation_report.is_file():
        raise FileNotFoundError(evaluation_report)
    builtin_paths = [path.expanduser().resolve() for path in builtin_evaluation_reports]
    for path in builtin_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_contents = checkpoint_path.read_bytes()
    checkpoint_digest = hashlib.sha256(checkpoint_contents).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_contents), map_location="cpu", weights_only=False)
    artifact = actor_artifact_from_checkpoint(checkpoint)
    witness = (
        json.loads(inference_equivalence.expanduser().resolve().read_text(encoding="utf-8"))
        if inference_equivalence is not None
        else None
    )
    source = require_source_identity(
        artifact["source_identity"],
        equivalence=witness,
        artifact_sha256=file_sha256(checkpoint_path),
    )
    run_provenance = validate_run_provenance(artifact.get("run_provenance"))
    builtin_score_rates: dict[str, float] = {}
    for path in builtin_paths:
        label, rate = _load_builtin_evaluation(
            path.read_bytes(),
            checkpoint_digest,
            source,
            minimum_builtin_score_rate,
            minimum_builtin_seed_count,
        )
        if label in builtin_score_rates:
            raise ValueError(f"two evaluations supplied for the built-in {label}")
        builtin_score_rates[label] = rate
    if _REQUIRED_BUILTIN not in builtin_score_rates:
        raise ValueError(
            f"submission requires an evaluation against the built-in {_REQUIRED_BUILTIN}"
        )
    evaluation_contents = evaluation_report.read_bytes()
    evaluation = _load_evaluation(
        evaluation_contents,
        checkpoint_digest,
        source,
        minimum_score_rate,
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
        # After the hashes agree and before anything is archived: the bundle that
        # scores is this directory, so it is the thing that has to be seen playing.
        smoke = _smoke_test(root)
        manifest = {
            "format_version": 1,
            "source_identity": source,
            # Recorded beside the identity, never inside it: a reader who sees a
            # shipped tree that differs from the checkpoint's own needs the
            # measurement that admitted it, not a matching hash and no reason.
            "inference_equivalence": witness,
            "bundle_smoke": smoke,
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
                "score_rate": _score_rate(evaluation, "finalist"),
            },
            # The thresholds ride along with the rates they admitted, so a bundle
            # built under a relaxed deadline floor says so on its face instead of
            # looking identical to one that cleared the default.
            "strength_gate": {
                "builtin_score_rates": dict(sorted(builtin_score_rates.items())),
                "minimum_score_rate": minimum_score_rate,
                "minimum_builtin_score_rate": minimum_builtin_score_rate,
                "minimum_builtin_seed_count": minimum_builtin_seed_count,
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
    manifest = build(
        args.checkpoint,
        args.evaluation_report,
        args.output,
        builtin_evaluation_reports=args.builtin_evaluation_report,
        minimum_score_rate=args.minimum_score_rate,
        minimum_builtin_score_rate=args.minimum_builtin_score_rate,
        minimum_builtin_seed_count=args.minimum_builtin_seed_count,
        inference_equivalence=args.inference_equivalence,
    )
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
