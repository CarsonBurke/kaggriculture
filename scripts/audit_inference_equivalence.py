#!/usr/bin/env python3
"""Prove that two source trees present an actor with the same inference surface.

`require_source_identity` binds an artifact to the whole tree that produced it,
which is the right default and too coarse to be true: a reward change, a new
league opponent, or a telemetry layout cannot alter how a frozen actor maps an
observation to an action, yet each moves the tree hash and locks the artifact out
of its own evaluation. Weakening the gate to a hand-picked file list would trade
a false refusal for a silent acceptance, because the list is a claim about
reachability that nothing checks.

This measures the claim instead. Each tree dumps digests of everything a frozen
policy actually reads -- the encoded observation tensors, the legality masks, and
the actor's own logits over a shared driven trajectory -- and the two dumps are
compared exactly. Identical dumps mean the trees are interchangeable for
inference on this artifact, whatever else moved between them.

Both dumps must come from the same driving sequence, so the factor stream is
derived from a seed rather than sampled from the policy: a policy-driven rollout
would diverge on the first differing logit and report the divergence as many
differing observations instead of one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.inference import load_actor_artifact
from kaggriculture.rust_env import load_native

#: Digested rather than stored: a dump is a claim about equality, and equality of
#: 719 steps of float tensors is checked the same way with 32 bytes per field.
_DIGEST_FIELDS = ("board", "global_features", "units", "unit_positions")


def _digest(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode("utf-8"))
        hasher.update(str(contiguous.shape).encode("utf-8"))
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def _driven_factors(
    generator: np.random.Generator, games: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One step of the shared driving stream, identical in both trees."""
    return (
        generator.integers(0, N_UNIT_ACTIONS, (games, 2, MAX_UNITS), dtype=np.uint8),
        generator.integers(0, N_MARKET_KINDS, (games, 2, MAX_MARKET_ORDERS), dtype=np.uint8),
        generator.integers(0, N_QUANTITIES, (games, 2, MAX_MARKET_ORDERS), dtype=np.uint8),
    )


def _dump(args: argparse.Namespace) -> dict[str, Any]:
    module = load_native()
    seeds = np.arange(args.seed_start, args.seed_start + args.games, dtype=np.uint64)
    environment = module.BatchEnv(seeds)
    actor, _ = load_actor_artifact(
        Path(args.artifact).resolve(),
        device=torch.device("cpu"),
        agent=getattr(args, "agent", None),
    )

    actor = actor.eval().requires_grad_(False)
    generator = np.random.default_rng(args.driver_seed)

    observation_digests: list[str] = []
    mask_digests: list[str] = []
    logit_digests: list[str] = []
    for _ in range(args.steps):
        units, kinds, quantities = _driven_factors(generator, args.games)
        masks = environment.factor_masks(units, kinds, quantities)
        mask_digests.append(_digest(*(np.asarray(masks[name]) for name in sorted(masks.keys()))))
        encoded = environment.encoded()
        observation_digests.append(_digest(*(np.asarray(encoded[name]) for name in _DIGEST_FIELDS)))
        with torch.inference_mode():
            output = actor(
                *(
                    torch.as_tensor(np.asarray(encoded[name]), dtype=torch.float32)
                    if name != "unit_positions"
                    else torch.as_tensor(np.asarray(encoded[name]))
                    for name in _DIGEST_FIELDS
                )
            )
        logit_digests.append(
            _digest(
                output.unit_logits.numpy(),
                output.market_kind_logits.numpy(),
                output.market_quantity_context.numpy(),
            )
        )
        environment.step_factors(units, kinds, quantities)

    return {
        "artifact": str(Path(args.artifact).resolve()),
        "games": args.games,
        "steps": args.steps,
        "seed_start": args.seed_start,
        "driver_seed": args.driver_seed,
        "observations": _digest(np.frombuffer("".join(observation_digests).encode(), np.uint8)),
        "masks": _digest(np.frombuffer("".join(mask_digests).encode(), np.uint8)),
        "logits": _digest(np.frombuffer("".join(logit_digests).encode(), np.uint8)),
        "per_step": {
            "observations": observation_digests,
            "masks": mask_digests,
            "logits": logit_digests,
        },
    }


def _compare(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    surfaces: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    for key in ("games", "steps", "seed_start", "driver_seed"):
        if reference[key] != candidate[key]:
            failures.append(f"{key}: {reference[key]} vs {candidate[key]} -- dumps are not paired")
    if failures:
        return failures
    for surface in surfaces:
        if reference[surface] == candidate[surface]:
            continue
        steps = [
            index
            for index, (left, right) in enumerate(
                zip(
                    reference["per_step"][surface],
                    candidate["per_step"][surface],
                    strict=True,
                )
            )
            if left != right
        ]
        first = steps[0] if steps else "none"
        failures.append(
            f"{surface} differ at {len(steps)} of {reference['steps']} steps, first {first}"
        )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--agent",
        type=int,
        default=None,
        help="population member to drive; required for a multi-member checkpoint",
    )

    parser.add_argument(
        "--reference-tree",
        type=Path,
        help="checkout to compare against; this script is copied in and run there",
    )
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--seed-start", type=int, default=880_001)
    parser.add_argument("--driver-seed", type=int, default=41_009)
    parser.add_argument("--dump", type=Path, help="write this tree's dump and exit")
    parser.add_argument("--output", type=Path, help="write the comparison report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps(_dump(args), indent=1), encoding="utf-8")
        print(f"dumped {args.dump}")
        return 0
    if args.reference_tree is None:
        raise SystemExit("--reference-tree is required unless --dump is given")

    # Imported here, not at module scope: the staged copy runs `--dump` inside the
    # reference tree, whose package predates these names. Only the comparing side
    # -- always the current tree -- needs them.
    from kaggriculture.provenance import (
        INFERENCE_SURFACES,
        file_sha256,
        source_identity,
    )

    tree = args.reference_tree.resolve()
    # The harness has to be this file so both sides drive the identical sequence.
    # Frozen snapshots are read-only, so the copy lives in a temp dir and the
    # reference package is injected via PYTHONPATH rather than by nesting the
    # script under the tree (which is what `source_identity` hashes).
    with tempfile.TemporaryDirectory(prefix="kaggriculture-inference-eq-") as name:
        staged = Path(name) / Path(__file__).name
        staged.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        reference_dump = Path(name) / "inference-equivalence-reference.json"
        shared = [
            "--artifact",
            str(Path(args.artifact).resolve()),
            "--games",
            str(args.games),
            "--steps",
            str(args.steps),
            "--seed-start",
            str(args.seed_start),
            "--driver-seed",
            str(args.driver_seed),
        ]
        if args.agent is not None:
            shared.extend(["--agent", str(args.agent)])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tree / "src")
        completed = subprocess.run(
            [sys.executable, str(staged), *shared, "--dump", str(reference_dump)],
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"reference tree dump failed:\n{completed.stdout}\n{completed.stderr}"
            )
        reference = json.loads(reference_dump.read_text(encoding="utf-8"))
    candidate = _dump(args)
    failures = _compare(reference, candidate, INFERENCE_SURFACES)

    artifact = Path(args.artifact).resolve()
    # The witness is only usable if it names which pair of trees it measured and
    # which artifact it drove, so a reader -- and `require_source_identity` -- can
    # refuse a stale or borrowed one instead of trusting the word "equivalent".
    report = {
        "artifact": str(artifact),
        "artifact_sha256": file_sha256(artifact),
        "reference_tree": str(tree),
        "expected_identity": json.loads(
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys, json; sys.path.insert(0, 'src')\n"
                    "from kaggriculture.provenance import source_identity\n"
                    "print(json.dumps(source_identity()['sha256']))",
                ],
                cwd=tree,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        ),
        "candidate_identity": source_identity()["sha256"],
        "games": args.games,
        "steps": args.steps,
        "surfaces": {
            surface: {
                "reference": reference[surface],
                "candidate": candidate[surface],
                "equal": reference[surface] == candidate[surface],
            }
            for surface in INFERENCE_SURFACES
        },
        "failures": failures,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    for surface, record in report["surfaces"].items():
        state = "identical" if record["equal"] else "DIFFER"
        print(f"{surface}: {state} ({record['candidate'][:16]})")
    if failures:
        raise SystemExit("inference surfaces are not equivalent:\n  " + "\n  ".join(failures))
    print(f"equivalent over {args.games} games x {args.steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
