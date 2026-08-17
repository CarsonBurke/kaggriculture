#!/usr/bin/env python3
"""Launch production PPO directly, without the pre-flight benchmark ceremony.

Compilation follows standing evidence, which is per phase rather than per run:
the update is compiled and the rollout is not, because on the conv model
compiling the collector is a loss -- its per-step graph replay costs more than
the kernel launches it removes -- while compiling the update is a large win.
The magnitudes are deliberately not repeated here. They belong to a particular
calibration, two copies of a measured number drift apart the moment one is
re-run, and this script binds no calibration decision of its own; read the
README's summary for the shape and any run's own decision file for its
numbers. The older "~2.4x across every calibration" was a blended total
measured on the pre-conv model, whose update was far cheaper, so it neither
describes this architecture nor separates the two phases. Correctness is guarded by the
gates train_ppo.py runs
inside the production process itself — the per-iteration first-minibatch KL
gate, and the replay-parity audit, which runs at iteration one and every
REPLAY_PARITY_AUDIT_INTERVAL iterations thereafter.  A fresh run's first audit
aborts on any breach, so a numerically broken launch still dies at iteration
one; later audits abort on a step change away from the previous one or on
passing the absolute ceiling, and warn on drift in between rather than killing
a healthy multi-day run over expected numerics.  Use
launch_calibrated_training.py instead when
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
    PRODUCTION_ROLLOUT_BFLOAT16,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_UPDATE_COMPILE_MODE,
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
        rollout_forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
        update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE,
        resume_checkpoint=resume_checkpoint,
    )
    launch = {
        "event": "direct_launch",
        "rollout_forward_mode": PRODUCTION_ROLLOUT_FORWARD_MODE,
        "rollout_bfloat16": PRODUCTION_ROLLOUT_BFLOAT16,
        "update_compile_mode": PRODUCTION_UPDATE_COMPILE_MODE,
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
