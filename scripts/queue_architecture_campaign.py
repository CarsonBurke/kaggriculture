#!/usr/bin/env python3
"""Freeze and queue the matched, <=25-minute entity architecture experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kaggriculture.modelargs import model_config_arguments
from kaggriculture.production import (
    build_training_command,
    production_model_config,
    production_ppo_config,
)
from kaggriculture.provenance import file_sha256, freeze_source, source_identity
from kaggriculture.registry import ENTITY_ATTENTION, resolve_architecture

ARMS = {
    "control": {},
    "critic-source": {"critic_source_read": True},
    "writeback": {"memory_writeback": True},
    "tile-bias": {"unit_tile_bias": True},
    "hardness-league": {},
}
DEFAULT_ARMS = tuple(ARMS)
ARMS["bixt"] = {"bixt_latents": 32, "global_modulation": False}


def replace_argument(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="entity-review-20260916")
    parser.add_argument("--validation-job", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=DEFAULT_ARMS)
    parser.add_argument(
        "--reuse-bc-root",
        type=Path,
        help="reuse existing matched BC artifacts after a numerically equivalent systems change",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not contain duplicates")
    root = Path(__file__).resolve().parents[1]
    destination = root / "artifacts" / "probes" / args.name
    manifest_path = destination / "campaign.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    identity = source_identity()
    source = root / "artifacts" / "source-snapshots" / identity["sha256"]
    freeze_source(source)
    environment = {
        "PYTHONPATH": str(source / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CARGO_TARGET_DIR": str(root / "artifacts" / "mlq-cargo-target"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
        "KRAGG_PROJECT_ROOT": str(root),
        "KRAGG_SOURCE_DIGEST": identity["sha256"],
    }
    manifest = {
        "source": str(source),
        "source_identity": identity,
        "environment": environment,
        "arms": list(args.arms),
        "analysis_plan": {
            "primary": "Argmax native development score on 256 common seeds per opponent",
            "comparison": "Equal wall-clock budget; report iterations and paired seed uncertainty",
            "warm_start": "Matched two-epoch BC; critic and league reuse control BC exactly",
            "training": "Production 128 self-play + 64 league games, production minibatch",
            "limits": "24-minute trainer budget; 25-minute hard cap including startup",
            "culling": "No plateau cull at this horizon; numerical/readiness gates remain",
            "promotion": "No automatic default changes; exploratory single-training-seed evidence",
        },
        "jobs": {},
    }
    destination.mkdir(parents=True, exist_ok=True)

    def submit(label: str, command: list[str], minutes: int, parents: list[int]) -> int:
        if not 0 < minutes <= 30:
            raise ValueError(f"{label} exceeds the 30-minute per-job limit")
        queue_command = [
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
            queue_command.extend(("--env", f"{key}={value}"))
        for parent in parents:
            queue_command.extend(("--after-success", str(parent)))
        queue_command.extend(("--", *command))
        result = (
            json.loads(subprocess.check_output(queue_command, text=True))
            if args.submit
            else {"id": -(len(manifest["jobs"]) + 1), "state": "dry-run"}
        )
        manifest["jobs"][label] = {
            "command": command,
            "minutes": minutes,
            "parents": parents,
            "submission": result,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{label}: {result['id']}", flush=True)
        return result["id"]

    def script(name: str) -> list[str]:
        return [sys.executable, str(source / "scripts" / name)]

    architecture = resolve_architecture(ENTITY_ATTENTION)
    # Preserve the historical control after production promotes successful arms.
    control_config = production_model_config() | {"critic_source_read": False}
    configs = {arm: control_config | changes for arm, changes in ARMS.items()}
    if any(config["shared_memory_kv"] for config in configs.values()):
        raise ValueError("This experiment requires the untied-K/V control")
    model_args = {
        arm: model_config_arguments(architecture, config) for arm, config in configs.items()
    }
    runs = root / "runs" / args.name
    bc_runs = runs if args.reuse_bc_root is None else args.reuse_bc_root.resolve()
    owners = {
        arm: "control" if arm in ("critic-source", "hardness-league") else arm for arm in ARMS
    }
    gate_arms = tuple(
        dict.fromkeys("control" if arm == "hardness-league" else arm for arm in args.arms)
    )
    bc_arms = tuple(dict.fromkeys(owners[arm] for arm in gate_arms))
    bc_paths = {arm: bc_runs / owner / "bc" / "bc-actor.pt" for arm, owner in owners.items()}
    bc_jobs = {}
    for arm in bc_arms:
        if args.reuse_bc_root is not None:
            manifest.setdefault("reused_bc", {})[arm] = {
                "path": str(bc_paths[arm]),
                "sha256": file_sha256(bc_paths[arm]),
            }
            bc_jobs[arm] = args.validation_job
            continue
        command = [
            *script("train_bc.py"),
            "--dataset",
            *[
                str(root / "data" / f"bc-v16-current-{opponent}-64")
                for opponent in ("mirror", "starter", "pass", "random")
            ],
            "--output",
            str(bc_paths[arm].parent),
            "--encoded-cache",
            str(root / "data" / ".bc-encoded-cache"),
            "--device",
            "cuda",
            "--epochs",
            "2",
            "--batch-size",
            "1024",
            "--compile-mode",
            "default",
            "--seed",
            "20260812",
            "--architecture",
            ENTITY_ATTENTION,
            *model_args[arm],
        ]
        bc_jobs[arm] = submit(f"bc-{arm}", command, 25, [args.validation_job])
    bc_jobs.update({arm: bc_jobs[owners[arm]] for arm in gate_arms})
    gates = {}
    for arm in gate_arms:
        command = [
            *script("benchmark_ppo_iteration.py"),
            "--architecture",
            ENTITY_ATTENTION,
            *model_args[arm],
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
            "--reward-mode",
            "terminal-outcome",
            "--auxiliary-mode",
            "enabled",
            "--structured-critic-latent-coefficient",
            "1.0",
            "--structured-critic-value-coefficient",
            "1.0",
            "--update-compile-mode",
            "reduce-overhead",
            "--rollout-forward-mode",
            "inductor_graph",
            "--rollout-bfloat16",
            "--minibatch-size",
            str(production_ppo_config(update_compile_mode="reduce-overhead")["minibatch_size"]),
            "--init-actor-from",
            str(bc_paths[arm]),
            "--output",
            str(destination / f"{arm}-benchmark.jsonl"),
        ]
        gates[arm] = submit(f"gate-{arm}", command, 10, [bc_jobs[arm]])
    if "hardness-league" in args.arms:
        gates["hardness-league"] = gates["control"]
    for arm in args.arms:
        command = build_training_command(
            runs / arm / "ppo",
            iterations=500,
            max_hours=0.4,
            seed=20260812,
            rollout_forward_mode="inductor_graph",
            update_compile_mode="reduce-overhead",
            initial_actors=(bc_paths[arm],),
        )
        command[1] = str(source / "scripts" / "train_ppo.py")
        command.remove("--external-eval")
        for term in ("latent", "value"):
            replace_argument(command, f"--structured-critic-{term}-coefficient", "1.0")
        for flag, value in zip(model_args[arm][::2], model_args[arm][1::2], strict=True):
            replace_argument(command, flag, value)
        replace_argument(
            command, "--league-selection", "hardness" if arm == "hardness-league" else "stratified"
        )
        learning_job = submit(f"learn-{arm}", command, 25, [gates[arm]])
        submit(
            f"evaluate-{arm}",
            [
                *script("evaluate_architecture_campaign.py"),
                "--artifact",
                f"bc={bc_paths[arm]}",
                "--artifact",
                f"ppo={runs / arm / 'ppo' / 'latest.pt'}",
                "--output",
                str(destination / f"{arm}-evaluation.json"),
            ],
            5,
            [learning_job],
        )
    if "critic-source" in args.arms:
        submit(
            "common-policy-fit",
            [
                *script("fit_common_policy_critics.py"),
                "--actor",
                str(bc_paths["control"]),
                "--output",
                str(destination / "common-policy-fit.json"),
                "--epochs",
                "20",
                "--max-seconds",
                "840",
            ],
            15,
            [bc_jobs["control"], gates["critic-source"]],
        )
    print(manifest_path)


if __name__ == "__main__":
    main()
