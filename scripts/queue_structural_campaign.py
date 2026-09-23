#!/usr/bin/env python3
"""Freeze and queue full-budget shared-plan and feed-forward forecast comparisons."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from kaggriculture.modelargs import model_config_arguments
from kaggriculture.production import (
    build_training_command,
    production_model_config,
    production_ppo_config,
)
from kaggriculture.provenance import freeze_source, source_identity
from kaggriculture.registry import (
    CAUSAL,
    ENTITY_ATTENTION,
    LEJEPA,
    STRATEGIC,
    resolve_architecture,
)

# BC owner, actor family, model changes, PPO ratio, forecasting coefficient.
ARMS = {
    "component-control": ("entity", ENTITY_ATTENTION, {}, "components", 0.0),
    "joint-control": ("entity", ENTITY_ATTENTION, {}, "joint", 0.0),
    "workspace": ("workspace", STRATEGIC, {"plan_count": 1}, "joint", 0.0),
    "shared-plan": ("shared-plan", STRATEGIC, {"plan_count": 8}, "joint", 0.0),
    "causal": ("causal", CAUSAL, {}, "joint", 0.0),
    "economic-control": (
        "entity",
        ENTITY_ATTENTION,
        {"critic_architecture": "economic"},
        "components",
        0.0,
    ),
    "forecast": (
        "entity",
        ENTITY_ATTENTION,
        {"critic_architecture": "forecast"},
        "components",
        1.0,
    ),
    # The production contract is the entity family's, whose critic has no readout
    # feed-forward; this family's critic is designed with one (LejepaConfig).
    "lejepa": ("lejepa", LEJEPA, {"critic_readout_ffn": True}, "components", 0.0),
}
FAMILIES = {
    "actor": ("component-control", "joint-control", "workspace", "shared-plan"),
    "critic": ("component-control", "economic-control", "forecast"),
    "causal": ("joint-control", "causal"),
    "lejepa": ("component-control", "lejepa"),
}
#: The LeJEPA objective's shipped weights (README, "LeJEPA world model"). The
#: family trains its backbone with nothing else, in BC as in PPO, so every job
#: of a `lejepa` arm carries them; demonstrations hold no reward, so the clone
#: takes the two transition terms and PPO adds the reward head.
JEPA_COEFFICIENTS = {"prediction": 1.0, "sigreg": 0.09, "reward": 0.1}
JEPA_HORIZON = 1
#: Clone epochs, (every other family, `lejepa`). The entity actor's clone is
#: converged in two (holdout NLL 0.0012). A `lejepa` clone fits heads to an
#: encoder its own objective is still building, and reaches parity only later:
#: market-kind accuracy 0.90 after two epochs, 0.997 (NLL 0.0033) after twelve.
#: Cloning keeps the best held-out epoch, so the longer budget cannot overfit.
BC_EPOCHS = (2, 12)


def jepa_arguments(arm: str, *, reward: bool) -> list[str]:
    """The objective's flags for a `lejepa` arm, and none for any other."""
    if ARMS[arm][1] != LEJEPA:
        return []
    terms = (
        JEPA_COEFFICIENTS
        if reward
        else {term: value for term, value in JEPA_COEFFICIENTS.items() if term != "reward"}
    )
    arguments = [f"--jepa-{term}-coefficient={value}" for term, value in terms.items()]
    return [*arguments, f"--jepa-horizon={JEPA_HORIZON}"]


def set_argument(command: list[str], flag: str, value: Any) -> None:
    if command.count(flag) > 1:
        raise ValueError(f"duplicate argument: {flag}")
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def arm_config(arm: str):
    _, family, changes, _, _ = ARMS[arm]
    architecture = resolve_architecture(family)
    config = architecture.config_class(**(production_model_config() | changes))
    return architecture, config


def training_command(arm: str, run: Path, bc: Path, source: Path) -> list[str]:
    architecture, config = arm_config(arm)
    command = build_training_command(
        run,
        iterations=500,
        max_hours=4.0,
        seed=20260812,
        rollout_forward_mode="inductor_graph",
        update_compile_mode="reduce-overhead",
        initial_actors=(bc,),
    )
    command[1] = str(source / "scripts" / "train_ppo.py")
    # This campaign evaluates synchronously through the same compiled native
    # GPU collector; the ordinary exported CPU external worker is unsuitable.
    command.remove("--external-eval")
    set_argument(command, "--architecture", architecture.name)
    arguments = model_config_arguments(architecture, config.to_dict())
    for flag, value in zip(arguments[::2], arguments[1::2], strict=True):
        set_argument(command, flag, value)
    set_argument(command, "--policy-ratio-scope", ARMS[arm][3])
    set_argument(command, "--economic-forecast-coefficient", ARMS[arm][4])
    set_argument(command, "--architecture-panel", 25)
    command.extend(jepa_arguments(arm, reward=True))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", choices=tuple(FAMILIES), default="actor")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS))
    parser.add_argument("--validation-job", type=int, required=True)
    parser.add_argument("--causal-validation-job", type=int)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.validation_job < 1:
        parser.error("validation-job must be a submitted positive job ID")
    arms = tuple(args.arms or FAMILIES[args.family])
    if "causal" in arms and (args.causal_validation_job is None or args.causal_validation_job < 1):
        parser.error("causal arm requires a positive --causal-validation-job")
    if len(set(arms)) != len(arms):
        parser.error("duplicate arms")
    root = Path(os.environ.get("KRAGG_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
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
        "format_version": 1,
        "source": str(source),
        "source_identity": identity,
        "environment": environment,
        "arms": list(arms),
        "family": args.family,
        "analysis_plan": {
            "primary": "Sampled and argmax native fixed-panel win/draw scores; paired maps",
            "training": (
                "500 waves, 128 self + 64 league, production minibatch, 720 steps, "
                "one training seed"
            ),
            "time_limit": "4 trainer hours / 250 minute hard limit, not equal-update evidence",
            "initialization": (
                "Same four corpora, BC to convergence per family; critic arms share actor bytes"
            ),
            "defaults": (
                "hardness league, source-read critic, actor/critic NextLat off (the lejepa "
                "arm instead trains its backbone with the LeJEPA objective); GAE unchanged"
            ),
            "actor_controls": "Component versus joint PPO; workspace plan1 versus sampled plan8",
            "critic_controls": "Ordinary critic versus economic trunk versus forecast supervision",
            "culling": {
                "panel_interval_actor_waves": 25,
                "actor_warmup_waves": 150,
                "patience_waves": 100,
                "material_score_gain": 0.01,
                "material_critic_mse_reduction": 0.01,
                "minimum_decline_from_initialization": 0.05,
                "rule": (
                    "Neither smoothed panel score nor heldout critic MSE improves; "
                    "score deteriorated"
                ),
            },
            "selection": "Final panels256 maps/mode, then leaders head-to-head on held-out maps",
            "comparison": (
                "Report actor updates, elapsed GPU time and gain from own BC initialization"
            ),
            "promotion": "No automatic default promotion; no recurrence or prefix critic",
        },
        "jobs": {},
    }
    destination.mkdir(parents=True)

    def submit(label: str, command: list[str], minutes: int, parents: list[int], *, terminal=False):
        queue = [
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
            queue.extend(("--env", f"{key}={value}"))
        for parent in parents:
            queue.extend(("--after-terminal" if terminal else "--after-success", str(parent)))
        queue.extend(("--", *command))
        result = (
            json.loads(subprocess.check_output(queue, text=True))
            if args.submit
            else {"id": -(len(manifest["jobs"]) + 1), "state": "dry-run"}
        )
        manifest["jobs"][label] = {
            "command": command,
            "minutes": minutes,
            "parents": parents,
            "dependency": "terminal" if terminal else "success",
            "submission": result,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{label}: {result['id']}", flush=True)
        return result["id"]

    def script(name):
        return [sys.executable, str(source / "scripts" / name)]

    runs = root / "runs" / args.name
    owners = tuple(dict.fromkeys(ARMS[arm][0] for arm in arms))
    bc_paths = {owner: runs / owner / "bc" / "bc-actor.pt" for owner in owners}
    bc_jobs = {}
    for owner in owners:
        arm = next(arm for arm in arms if ARMS[arm][0] == owner)
        architecture, config = arm_config(arm)
        # Critic configuration has no place in actor-only BC initialization.
        config = architecture.config_class(**(config.to_dict() | {"critic_architecture": "entity"}))
        bc_jobs[owner] = submit(
            f"bc-{owner}",
            [
                *script("train_bc.py"),
                "--dataset",
                *[
                    str(root / "data" / f"bc-v16-current-{opponent}-64")
                    for opponent in ("mirror", "starter", "pass", "random")
                ],
                "--output",
                str(bc_paths[owner].parent),
                "--encoded-cache",
                str(root / "data" / ".bc-encoded-cache"),
                "--device",
                "cuda",
                "--epochs",
                str(BC_EPOCHS[architecture.name == LEJEPA]),
                "--batch-size",
                "1024",
                "--compile-mode",
                "default",
                "--seed",
                "20260812",
                "--architecture",
                architecture.name,
                *model_config_arguments(architecture, config.to_dict()),
                # The objective's successor pairs need two-row runs, exactly
                # the runs its PPO update draws.
                *(["--run-length", str(JEPA_HORIZON + 1)] if architecture.name == LEJEPA else []),
                *jepa_arguments(arm, reward=False),
            ],
            60,
            [args.causal_validation_job if owner == "causal" else args.validation_job],
        )
    for arm in arms:
        owner, _, _, ratio, forecast = ARMS[arm]
        architecture, config = arm_config(arm)
        ppo_defaults = production_ppo_config(update_compile_mode="reduce-overhead")
        gate = submit(
            f"gate-{arm}",
            [
                *script("benchmark_ppo_iteration.py"),
                "--architecture",
                architecture.name,
                "--device",
                "cuda",
                *model_config_arguments(architecture, config.to_dict()),
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
                "--policy-ratio-scope",
                ratio,
                "--actor-gae-lambda",
                str(ppo_defaults["actor_gae_lambda"]),
                "--epochs",
                str(ppo_defaults["epochs"]),
                "--critic-epochs",
                str(ppo_defaults["critic_epochs"]),
                "--structured-latent-coefficient",
                "0.0",
                "--structured-decision-coefficient",
                "0.0",
                "--structured-critic-latent-coefficient",
                "0.0",
                "--structured-critic-value-coefficient",
                "0.0",
                "--economic-forecast-coefficient",
                str(forecast),
                "--update-compile-mode",
                "reduce-overhead",
                "--rollout-forward-mode",
                "inductor_graph",
                "--rollout-bfloat16",
                "--minibatch-size",
                str(ppo_defaults["minibatch_size"]),
                "--init-actor-from",
                str(bc_paths[owner]),
                "--output",
                str(destination / f"{arm}-benchmark.jsonl"),
                *jepa_arguments(arm, reward=True),
            ],
            45,
            [bc_jobs[owner]],
        )
        learning = submit(
            f"learn-{arm}",
            training_command(arm, runs / arm / "ppo", bc_paths[owner], source),
            250,
            [gate],
        )
        for decoding in ("argmax", "sampled"):
            submit(
                f"evaluate-{arm}-{decoding}",
                [
                    *script("evaluate_architecture_campaign.py"),
                    "--artifact",
                    f"bc={bc_paths[owner]}",
                    "--artifact",
                    f"ppo={runs / arm / 'ppo' / 'latest.pt'}",
                    "--decoding",
                    decoding,
                    "--games",
                    "256",
                    "--output",
                    str(destination / f"{arm}-{decoding}-evaluation.json"),
                ],
                20,
                [learning],
                terminal=True,
            )
    print(manifest_path)


if __name__ == "__main__":
    main()
