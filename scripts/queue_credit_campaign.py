#!/usr/bin/env python3
"""Queue full-budget credit and valuation comparisons against the promoted recipe."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from kaggriculture.production import (
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_UPDATE_COMPILE_MODE,
    build_training_command,
)
from kaggriculture.provenance import file_sha256, freeze_source, source_identity
from kaggriculture.registry import ENTITY_ATTENTION

ARMS = {
    "monte-carlo": {"--actor-gae-lambda": "1.0"},
    "no-nextlat": {
        "--structured-critic-latent-coefficient": "0.0",
        "--structured-critic-value-coefficient": "0.0",
    },
    "economic": {"--critic-architecture": "economic"},
}


def replace_argument(command: list[str], flag: str, value: str) -> None:
    if command.count(flag) != 1:
        raise ValueError(f"expected one explicit {flag}")
    command[command.index(flag) + 1] = value


#: The baseline's actor trace, VAPO's `1 - 1 / (0.05 * 719)`, which every arm
#: but the Monte Carlo one keeps after production adopted lambda 1.
HISTORICAL_ACTOR_GAE_LAMBDA = 1 - 1 / (0.05 * 719)
MAX_JOB_MINUTES = 30
TRAINER_HOURS = 27 / 60


def build_commands(root: Path, source: Path, output: Path, actor: Path, arms: list[str]) -> dict:
    """Build explicit production commands without constructing or running a model."""
    commands = {}
    for arm in arms:
        run = root / "runs" / output.name / arm
        train = build_training_command(
            run,
            iterations=500,
            max_hours=TRAINER_HOURS,
            seed=20260812,
            rollout_forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
            update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE,
            initial_actors=(actor,),
            # The baseline these arms are measured against is entity-attention.
            architecture=ENTITY_ATTENTION,
        )
        train[1] = str(source / "scripts/train_ppo.py")
        train.remove("--external-eval")
        # Online-proxy autocull is this campaign's stop rule; the architecture
        # panel that production later adopted cannot share a run with it.
        train.append("--autocull")
        replace_argument(train, "--architecture-panel", "0")
        # Pin the mechanisms this experiment names as production defaults evolve;
        # the rest of the schedule (rates, minibatch, lanes) follows production.
        for term in ("latent", "value"):
            replace_argument(train, f"--structured-critic-{term}-coefficient", "1.0")
        replace_argument(train, "--actor-gae-lambda", str(HISTORICAL_ACTOR_GAE_LAMBDA))
        for flag, value in ARMS[arm].items():
            replace_argument(train, flag, value)
        benchmark = [
            sys.executable,
            str(source / "scripts/benchmark_ppo_iteration.py"),
            "--games",
            "128",
            "--league-games",
            "64",
            "--league-opponents",
            "8",
            "--repeats",
            "6",
            "--seed",
            "20260914",
            "--auxiliary-mode",
            "off" if arm == "no-nextlat" else "enabled",
            "--update-compile-mode",
            PRODUCTION_UPDATE_COMPILE_MODE,
            "--rollout-forward-mode",
            PRODUCTION_ROLLOUT_FORWARD_MODE,
            "--rollout-bfloat16",
            "--init-actor-from",
            str(actor),
            "--output",
            str(output / f"{arm}-benchmark.jsonl"),
        ]
        benchmark.extend(("--actor-gae-lambda", train[train.index("--actor-gae-lambda") + 1]))
        for term in ("latent", "value"):
            flag = f"--structured-critic-{term}-coefficient"
            benchmark.extend((flag, train[train.index(flag) + 1]))
        if arm == "economic":
            benchmark.extend(("--critic-architecture", "economic"))
        evaluations = {
            mode: [
                sys.executable,
                str(source / "scripts/evaluate_architecture_campaign.py"),
                "--artifact",
                f"bc={actor}",
                "--artifact",
                f"ppo={run / 'latest.pt'}",
                "--decoding",
                mode,
                "--output",
                str(output / f"{arm}-{mode}.json"),
            ]
            for mode in ("argmax", "sampled")
        }
        commands[arm] = {"train": train, "benchmark": benchmark, "evaluations": evaluations}
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--name", default="credit-valuation-20260918")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if len(args.arms) != len(set(args.arms)):
        parser.error("arms must be distinct")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a single directory name")
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(args.baseline.read_text())
    baseline_evaluations = [
        baseline["jobs"][f"evaluate-{mode}"]["submission"]["id"] for mode in ("argmax", "sampled")
    ]
    if any(type(job) is not int or job <= 0 for job in baseline_evaluations):
        raise ValueError("baseline evaluation dependencies must be submitted positive job IDs")
    baseline["jobs"]["train"]["command"]
    baseline["source_identity"]["sha256"]
    original = Path(baseline["initial_actor"]["path"])
    actor_digest = file_sha256(original)
    if actor_digest != baseline["initial_actor"]["sha256"]:
        raise ValueError("baseline BC artifact changed")
    identity = source_identity()
    source = root / "artifacts/source-snapshots" / identity["sha256"]
    output = root / "artifacts/probes" / args.name
    manifest_path = output / "campaign.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    # Validate all flags/commands before freezing or submitting anything.
    commands = build_commands(root, source, output, output / "bc-actor.pt", args.arms)
    freeze_source(source)
    output.mkdir(parents=True, exist_ok=True)
    actor = output / "bc-actor.pt"
    shutil.copyfile(original, actor)
    actor.chmod(0o444)
    if file_sha256(actor) != actor_digest:
        raise ValueError("copied BC artifact changed")
    tests = output / "tests"
    tests.mkdir()
    test_files = ["conftest.py", "test_economic_critic.py", "test_entity.py"]
    for name in test_files:
        shutil.copyfile(root / "tests" / name, tests / name)
        (tests / name).chmod(0o444)
    tests.chmod(0o555)
    environment = {
        **baseline["environment"],
        "PYTHONPATH": str(source / "src"),
        "CARGO_TARGET_DIR": str(root / "artifacts/cargo-target" / identity["sha256"]),
        "KRAGG_SOURCE_DIGEST": identity["sha256"],
    }
    manifest = {
        "source": str(source),
        "source_identity": identity,
        "environment": environment,
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": file_sha256(args.baseline),
        "initial_actor": {"path": str(actor), "sha256": actor_digest},
        "tests": {name: file_sha256(tests / name) for name in test_files},
        "arms": args.arms,
        "jobs": {},
        "plan": {
            "budget": (
                "Up to 500 production waves, one seed, 27-minute trainer / "
                "30-minute per-job hard cap; longer studies resume in bounded jobs"
            ),
            "autocull": baseline["plan"]["autocull"],
            "controls": (
                "Same actor, production hardness/source-read, BF16, full horizon and minibatch"
            ),
            "no_nextlat_caveat": (
                "Disabling NextLat also restores plain-PPO individual-state shuffling"
            ),
            "economic_scope": "Independent valuation representation; no new forecasting target",
            "comparison": (
                "Fixed-panel argmax and sampled scores; actor updates and "
                "GPU time; no automatic promotion"
            ),
        },
    }

    def submit(label: str, command: list[str], minutes: int, *, success=(), terminal=()) -> int:
        if not 0 < minutes <= MAX_JOB_MINUTES:
            raise ValueError(f"{label} exceeds the {MAX_JOB_MINUTES}-minute per-job limit")
        queued = [
            "mlq",
            "submit",
            "--json",
            "--name",
            f"{args.name}-{label}",
            "--idempotency-key",
            f"{args.name}-{identity['sha256']}-{label}",
            "--cwd",
            str(source),
            "--max-parallel-runs",
            "1",
            "--max-attempts",
            "1",
            "--time-limit",
            f"{minutes}m",
        ]
        for key, value in environment.items():
            queued.extend(("--env", f"{key}={value}"))
        for parent in success:
            queued.extend(("--after-success", str(parent)))
        for parent in terminal:
            queued.extend(("--after-terminal", str(parent)))
        result = (
            json.loads(subprocess.check_output([*queued, "--", *command], text=True))
            if args.submit
            else {"id": -(len(manifest["jobs"]) + 1), "state": "dry-run"}
        )
        manifest["jobs"][label] = {
            "command": command,
            "time_limit_minutes": minutes,
            "after_success": list(success),
            "after_terminal": list(terminal),
            "submission": result,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(label, result["id"], flush=True)
        return result["id"]

    validation = submit(
        "correctness",
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(source / "pyproject.toml"),
            "-p",
            "no:cacheprovider",
            str(tests / "test_economic_critic.py"),
            str(tests / "test_entity.py"),
            "-q",
        ],
        20,
    )
    evaluations = []
    for arm, command in commands.items():
        gate = submit(f"gate-{arm}", command["benchmark"], 15, success=[validation])
        train = submit(f"train-{arm}", command["train"], 30, success=[gate])
        for mode, evaluation in command["evaluations"].items():
            evaluations.append(submit(f"evaluate-{arm}-{mode}", evaluation, 10, success=[train]))
    submit(
        "summarize",
        [sys.executable, str(source / "scripts/summarize_credit_campaign.py"), str(manifest_path)],
        5,
        terminal=[*evaluations, *baseline_evaluations],
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
