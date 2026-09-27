#!/usr/bin/env python3
"""Queue matched neural core ablations using an explicitly frozen training tree.

The four arms add ALL quantities, unit affordances, and margin observations
one at a time to schema-v3 LeJEPA with legacy absolute quantities. All runs
receive the same BC corpus, initialization seed, production workload and budget.
Submission queues BC only. PPO must pass the full-game admission wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

VARIANTS = ("legacy", "schema3-no-affordance", "schema3", "control")


def options(values: dict[str, object]) -> list[str]:
    result = []
    for name, value in values.items():
        result.extend((f"--{name}", str(value).lower() if isinstance(value, bool) else str(value)))
    return result


def commands(root: Path, source: Path, variant: str) -> dict[str, list[str]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown core variant: {variant}")
    schema = 4 if variant == "control" else 3
    model = {
        "architecture": "lejepa",
        "observation-schema-version": schema,
        "action-interface": 1 if variant == "legacy" else 2,
        "model-dim": 96,
        "attention-heads": 4,
        "attention-kv-heads": 2,
        "ffn-multiplier": 2,
        "farm-blocks": 2,
        "core-layers": 4,
        "quantity-rank": 32,
        "global-modulation": True,
        "zero-init-branches": False,
        "fused-mlp": False,
        "split-clock-token": False,
        "shared-memory-kv": False,
        "inter-attention-ffn": False,
        "unit-local-readout": False,
        "critic-readout-ffn": True,
        "unit-local-init": True,
        "tile-cross-rope": False,
        "critic-source-read": True,
        "critic-architecture": "entity",
        "memory-writeback": False,
        "unit-tile-bias": False,
        "bixt-latents": 0,
        "value-sigma-ratio": 3.0,
        "scalar-value": False,
        "policy-readout-layers": 1,
        "policy-shapes-backbone": True,
        "critic-private-layers": 1,
        "unit-affordance-scorer": variant in ("schema3", "control"),
        "market-affordance-scorer": False,
        "jepa-hidden-dim": 384,
        "jepa-slices": 128,
        "jepa-tile-samples": 32,
        "jepa-sigreg-rows": 1024,
    }
    objective = {
        "jepa-prediction-coefficient": 1.0,
        "jepa-sigreg-coefficient": 0.09,
        "jepa-horizon": 1,
    }
    directory = root / "runs/core-step-model-20260926" / variant
    python = str(root / ".venv/bin/python")
    bc = [python, str(source / "scripts/train_bc.py"), "--dataset"]
    bc.extend(
        str(root / "data" / f"bc-v16-current-{name}-64")
        for name in ("mirror", "starter", "pass", "random")
    )
    bc += options(
        model
        | objective
        | {
            "encoded-cache": root / "data/.bc-encoded-cache",
            "device": "cuda",
            "compile-mode": "default",
            "batch-size": 1024,
            "epochs": 2,
            "run-length": 2,
            "seed": 20260812,
            "matrix-learning-rate": 0.003,
            "adam-learning-rate-ratio": 0.35,
            "output": directory / "bc",
        }
    )
    ppo = [python, str(source / "scripts/train_ppo.py")]
    ppo += options(
        model
        | objective
        | {
            "run-dir": directory / "ppo",
            "init-actor-from": directory / "bc/bc-actor.pt",
            "critic-warmup-iterations": 10,
            "iterations": 540,
            "actor-waves": 500,
            "max-hours": 0.0,
            "seed": 20800000,
            "reward-mode": "terminal-soft-outcome",
            "device": "cuda",
            "population": 1,
            "games": 128,
            "league-games": 64,
            "league-selection": "hardness",
            "league-active-opponents": 2,
            "league-historical-opponents": 6,
            "league-active-pool-size": 16,
            "league-builtin-opponents": "pass,random,starter,scripted-v27,scripted-v16",
            "league-builtin-lanes": 4,
            "episode-steps": 720,
            "temperature": 1.0,
            "checkpoint-seconds": 420,
            "actor-lr": 0.00015,
            "critic-lr": 0.00015,
            "critic-head-lr": 0.0004375,
            "lr-warmup-steps": 32,
            "epochs": 1,
            "critic-epochs": 1,
            "minibatch-size": 4096,
            "policy-loss-reduction": "states",
            "policy-ratio-scope": "components",
            "clip-low": 0.8,
            "clip-high": 1.28,
            "gamma": 1.0,
            "actor-gae-lambda": 0.972183588317107,
            "critic-gae-lambda": 1.0,
            "target-kl": 0.03,
            "optimizer": "normuon",
            "nextlat-max-gradient-norm": 1.0,
            "structured-latent-coefficient": 0.0,
            "structured-decision-coefficient": 0.0,
            "structured-critic-latent-coefficient": 0.0,
            "structured-critic-value-coefficient": 0.0,
            "economic-forecast-coefficient": 0.0,
            "rollout-forward-mode": "inductor_graph",
            "update-compile-mode": "default",
            "jepa-reward-coefficient": 0.1,
            "structured-learning-rate": 0.000015,
            "architecture-panel": 25,
            "actor-average-half-life": 8.0,
            "actor-lr-cooldown-frac": 0.0,
        }
    )
    ppo += ["--rollout-bfloat16", "--no-structured-critic-gradient-balance"]
    return {"bc": bc, "ppo": ppo}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    native_source = (args.native_source or source).resolve()
    for name in ("scripts/train_bc.py", "scripts/train_ppo.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    native = native_source / "_kagg_env/_kagg_env.abi3.so"
    if not native.is_file():
        raise FileNotFoundError(native)
    native_files = ["Cargo.toml", "Cargo.lock", "pyproject.toml"]
    native_files += [
        str(p.relative_to(source / "rust/kagg_env"))
        for p in (source / "rust/kagg_env/src").rglob("*.rs")
    ]
    for name in native_files:
        if (source / "rust/kagg_env" / name).read_bytes() != (
            native_source / "rust/kagg_env" / name
        ).read_bytes():
            raise ValueError(f"native extension's source differs: {name}")
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in VARIANTS:
        directory = root / "runs/core-step-model-20260926" / name
        if directory.exists():
            raise FileExistsError(f"refusing to reuse an existing training arm: {directory}")
    variants = {name: commands(root, source, name) for name in VARIANTS}
    manifest = {
        "source": str(source),
        "native_source": str(native_source),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "variants": variants,
        "source_files": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "scripts/train_bc.py",
                "scripts/train_ppo.py",
            )
        },
        "selection": (
            "Fresh matched gameplay; deployment argmax primary, sampled robustness secondary"
        ),
        "culling": "ArchitecturePanelGuard: after 150 actor waves, deterioration >=0.05 from init, "
        "100 waves without material score-EMA or critic-MSE improvement",
        "precision": "CUDA BF16, compiled updates and Inductor graph collection",
        "jobs": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.submit:
        print(args.output)
        return
    common = [
        "mlq",
        "submit",
        "--json",
        "--max-parallel-runs",
        "1",
        "--max-attempts",
        "1",
        "--cwd",
        str(root),
        "--env",
        f"PYTHONPATH={source / 'src'}:{native_source}",
    ]
    # Clone all arms first, allowing initialization quality to be compared before PPO.
    for name, stages in variants.items():
        command = [*common, "--name", f"core-{name}-bc", "--time-limit", "20m"]
        completed = subprocess.run(
            [*command, "--", *stages["bc"]], text=True, capture_output=True, check=True
        )
        result = json.loads(completed.stdout)
        job = result["id"]
        print(f"submitted job {job} [{result['name']}]", flush=True)
        manifest["jobs"][name] = {"bc": job}
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
