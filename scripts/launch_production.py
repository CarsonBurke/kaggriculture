#!/usr/bin/env python3
"""Launch production PPO directly, without the pre-flight benchmark ceremony.

Execution follows standing per-phase evidence rather than applying one compile
switch to the whole iteration. The collector advances every physical game in
one native batch and owns one explicit whole-wave CUDA graph. This bypasses
`torch.compile`'s `cudagraph_trees` bookkeeping and keeps the forward's input
addresses fixed while the host-side engine runs all games through Rayon. The
update remains compiled, where the established gain is large. Exact
measurements live beside the production constants and in each run's evidence
rather than being copied here.

Correctness is guarded by the gates train_ppo.py runs
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


def source_identity() -> dict[str, object]:
    """Load provenance only when launch work needs it."""
    from kaggriculture.provenance import source_identity as calculate

    return calculate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--init-actor-from",
        type=Path,
        help=(
            "behavior-cloned actor artifact to initialize a fresh production run; "
            "the critic and both optimizers still start fresh"
        ),
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        help=(
            "checkpoint to resume into --run-dir; when omitted, an existing "
            "--run-dir/latest.pt is resumed automatically"
        ),
    )
    parser.add_argument(
        "--critic-warmup-iterations",
        type=int,
        help="minimum critic-only iterations before the adaptive readiness gate "
        "(default: 5; maximum: 40)",
    )
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


def _recorded_warm_start(directory: Path) -> tuple[Path, int | None] | None:
    """Recover the original clone record before a resume rewrites launch metadata."""
    for name in ("launch.json", "calibration-decision.json"):
        path = directory / name
        if not path.is_file():
            continue
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(recorded, dict) or recorded.get("initial_actor") is None:
            continue
        warmup = recorded.get("critic_warmup_iterations")
        if warmup is not None and (type(warmup) is not int or warmup < 1):
            raise ValueError(f"invalid critic warmup provenance in {path}")
        return Path(str(recorded["initial_actor"])).expanduser().resolve(), warmup
    return None


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isfinite(args.max_hours) or args.max_hours < 0.0:
        raise ValueError("max hours must be finite and non-negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")

    # Keep `--help` on the standard-library-only path. Production configuration
    # reaches PyTorch through the model/PPO modules and otherwise makes a parser
    # query pay the entire training import cost.
    from kaggriculture.production import (
        PRODUCTION_ARCHITECTURE,
        PRODUCTION_CRITIC_WARMUP_ITERATIONS,
        PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS,
        PRODUCTION_ROLLOUT_BFLOAT16,
        PRODUCTION_ROLLOUT_FORWARD_MODE,
        PRODUCTION_UPDATE_COMPILE_MODE,
        build_training_command,
        production_model_config,
        require_repository_launcher,
        resolve_resume_checkpoint,
    )

    require_repository_launcher(Path(__file__))
    run_directory = args.run_dir.expanduser().resolve()
    resume_checkpoint = resolve_resume_checkpoint(run_directory, args.resume)
    requested_actor = (
        None if args.init_actor_from is None else args.init_actor_from.expanduser().resolve()
    )
    if requested_actor is not None and not requested_actor.is_file():
        raise FileNotFoundError(requested_actor)
    if args.critic_warmup_iterations is not None and args.critic_warmup_iterations < 1:
        raise ValueError("critic warmup iterations must be positive")
    if (
        args.critic_warmup_iterations is not None
        and args.critic_warmup_iterations > PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS
    ):
        raise ValueError(
            "critic warmup cannot exceed the "
            f"{PRODUCTION_CRITIC_WARMUP_MAX_ITERATIONS}-iteration readiness deadline"
        )
    if resume_checkpoint is None and requested_actor is None:
        raise ValueError("a fresh production run requires --init-actor-from or --resume")
    critic_warmup_iterations = (
        PRODUCTION_CRITIC_WARMUP_ITERATIONS
        if args.critic_warmup_iterations is None
        else args.critic_warmup_iterations
    )
    recorded_warm_start = None
    if resume_checkpoint is not None:
        recorded_warm_start = _recorded_warm_start(resume_checkpoint.parent)
        if requested_actor is not None:
            if recorded_warm_start is None:
                raise ValueError(
                    "automatic resume cannot verify --init-actor-from against prior launch "
                    "provenance"
                )
            if requested_actor != recorded_warm_start[0]:
                raise ValueError(
                    "--init-actor-from conflicts with the actor recorded by the resumed run"
                )
    effective_actor = (
        recorded_warm_start[0]
        if recorded_warm_start is not None
        else (requested_actor if resume_checkpoint is None else None)
    )
    effective_warmup = (
        recorded_warm_start[1]
        if recorded_warm_start is not None
        else (critic_warmup_iterations if resume_checkpoint is None else None)
    )
    if resume_checkpoint is None and critic_warmup_iterations >= args.iterations:
        raise ValueError("critic warmup must leave iterations for the actor to train in")
    command = build_training_command(
        run_directory,
        iterations=args.iterations,
        max_hours=args.max_hours,
        seed=args.seed,
        rollout_forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
        update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE,
        resume_checkpoint=resume_checkpoint,
        initial_actors=(
            () if resume_checkpoint is not None or requested_actor is None else (requested_actor,)
        ),
        critic_warmup_iterations=(
            None if resume_checkpoint is not None else critic_warmup_iterations
        ),
    )
    launch = {
        "event": "direct_launch",
        "architecture": PRODUCTION_ARCHITECTURE,
        "model": production_model_config(),
        "rollout_forward_mode": PRODUCTION_ROLLOUT_FORWARD_MODE,
        "rollout_bfloat16": PRODUCTION_ROLLOUT_BFLOAT16,
        "update_compile_mode": PRODUCTION_UPDATE_COMPILE_MODE,
        "iterations": args.iterations,
        "max_hours": args.max_hours,
        "seed": args.seed,
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "initial_actor": None if effective_actor is None else str(effective_actor),
        "critic_warmup_iterations": effective_warmup,
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
