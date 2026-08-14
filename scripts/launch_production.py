#!/usr/bin/env python3
"""Launch production VAPO directly, without the pre-flight benchmark ceremony.

Compilation is enabled on standing evidence (measured ~2.4x across every
calibration to date); correctness is guarded by the gates train_vapo.py runs
inside the production process itself — the once-per-process replay-parity
audit and the per-iteration first-minibatch KL gate abort a numerically broken
run at iteration one.  Use launch_calibrated_training.py instead when
performance-critical paths change and the compile decision needs fresh
matched evidence.

Direct launches bind no calibration decision, so their checkpoints carry
run_provenance null.  Such a run can be resumed here or trained further, but
launch_calibrated_training.py refuses to resume it: the decision-to-checkpoint
provenance binding cannot be established after the fact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from kaggriculture.production import (
    build_training_command,
    require_repository_launcher,
    resolve_resume_checkpoint,
)
from kaggriculture.provenance import source_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def _write_atomic(path: Path, rendered: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    require_repository_launcher(Path(__file__))
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isfinite(args.max_hours) or args.max_hours < 0.0:
        raise ValueError("max hours must be finite and non-negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    run_directory = args.run_dir.expanduser().resolve()
    resume_checkpoint = resolve_resume_checkpoint(run_directory)
    command = build_training_command(
        run_directory,
        iterations=args.iterations,
        max_hours=args.max_hours,
        seed=args.seed,
        compile_models=True,
        resume_checkpoint=resume_checkpoint,
    )
    launch = {
        "event": "direct_launch",
        "compile_models": True,
        "iterations": args.iterations,
        "max_hours": args.max_hours,
        "seed": args.seed,
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "source_identity": source_identity(),
        "training_command": command,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        run_directory / "launch.json",
        json.dumps(launch, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(launch, sort_keys=True), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
