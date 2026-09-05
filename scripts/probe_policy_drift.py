#!/usr/bin/env python3
"""Attribute a policy collapse to flattened weights or to unseen states.

Entropy in the training journal is measured over the states the collector
happened to visit, so a rise has two indistinguishable explanations. Either the
update flattened the policy where it already knew what to do, or the policy
wandered somewhere it had never been and the clone -- which never saw those
states -- is near-uniform there through no change of its own. The journal cannot
tell these apart and they call for opposite fixes: a smaller trust region for the
first, broader demonstrations or a recovery signal for the second.

Holding the states fixed separates them. One reference policy rolls out once,
its visited states are recorded, and every checkpoint is scored on that single
batch. Any entropy difference is then weight movement and nothing else, because
the states did not move. Rolling each checkpoint out on its own states measures
the other half: the gap between the two readings is what the collector's
distribution shift contributes.

Single-threaded CPU only, no autocast, deterministic given `--seed`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.inference import load_actor_artifact
from kaggriculture.model import FarmActor
from kaggriculture.policy import (
    categorical_statistics,
    mean_off_diagonal,
    population_disagreement,
)
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import allocate_rollout_storage, collect_self_play_rust

#: The actor's forward arguments with the dtype the update path feeds them at
#: (`ppo.py:667-670`): the rollout arena stores features as fp16 to halve its
#: upload, and the model's own parameters are fp32, so reading the arena straight
#: into the forward fails on the first convolution.
_FORWARD_FIELDS: tuple[tuple[str, torch.dtype], ...] = (
    ("board", torch.float32),
    ("global_features", torch.float32),
    ("units", torch.float32),
    ("unit_positions", torch.long),
)

#: The heads this scores, as (name, logit field, mask field, active field). The
#: mask says which actions are legal at all and the active flag which decisions
#: the game actually asked for. Both are needed: entropy over an illegal action is
#: meaningless, and a slot the game never queried holds no decision to report.
_HEADS = (
    ("unit", "unit_logits", "unit_masks", "unit_active"),
    ("kind", "market_kind_logits", "market_kind_masks", "market_active"),
)


def _load(path: Path, device: torch.device) -> tuple[FarmActor, str]:
    """Load either a training checkpoint or an exported actor artifact."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "actor" in payload and "architecture" in payload:
        entry = resolve_architecture(payload["architecture"])
        actor = entry.build_actor(payload["model_config"])
        actor.load_state_dict(payload["actor"])
        architecture = payload["architecture"]
    else:
        actor, provenance = load_actor_artifact(path)
        architecture = provenance["architecture"]
    assert isinstance(actor, FarmActor)
    return actor.to(device).eval(), architecture


def _forward(actor: FarmActor, states: dict[str, torch.Tensor]) -> Any:
    with torch.inference_mode():
        return actor(*(states[name] for name, _ in _FORWARD_FIELDS))


def _head_entropies(actor: FarmActor, states: dict[str, torch.Tensor]) -> dict[str, float]:
    """Mean entropy per policy head over a fixed batch of states.

    Reported through the update path's own masked math (`categorical_statistics`)
    so the number is comparable to the journal's `entropy`, and averaged over
    active decisions only. A slot the game never queried carries no choice, and
    counting it as zero would dilute the reading by however many such slots the
    batch happens to hold.
    """
    output = _forward(actor, states)
    entropies: dict[str, float] = {}
    for name, logit_field, mask_field, active_field in _HEADS:
        logits = getattr(output, logit_field)
        masks = states[mask_field].bool()
        actions = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
        _, entropy = categorical_statistics(logits, masks, actions, validate_mask=False)
        active = states[active_field].bool()
        entropies[name] = float(entropy[active].mean()) if active.any() else float("nan")
    return entropies


def _unit_logits(actor: FarmActor, states: dict[str, torch.Tensor]) -> torch.Tensor:
    """The unit head's raw logits on a fixed batch, for the disagreement matrix."""
    return _forward(actor, states).unit_logits


def _rollout_states(
    actor: FarmActor,
    *,
    architecture: str,
    games: int,
    steps: int,
    seed: int,
    device: torch.device,
    rows: int,
) -> tuple[dict[str, torch.Tensor], float]:
    """Roll one wave out and return a fixed sample of its states plus the money."""
    storage = allocate_rollout_storage(architecture, trajectories=games * 2, horizon=steps - 1)
    rollout = collect_self_play_rust(
        actor,
        games=games,
        seed_start=seed,
        episode_steps=steps,
        temperature=1.0,
        sampling_seed=seed ^ 0x5EED,
        forward_mode="eager",
        forward_autocast=False,
        storage=storage,
    )
    # Every array is (trajectories, horizon, ...), so the leading pair collapses to
    # one row axis. Only rows the collector marked valid carry a decision -- an
    # episode that ended early leaves the rest of its lane untouched -- so sampling
    # the raw block would mix real states with stale padding and pull every
    # statistic toward whatever that padding happens to hold.
    valid = np.flatnonzero(np.asarray(rollout.valid).reshape(-1))
    index = np.random.default_rng(seed).permutation(valid)[:rows]
    fields: dict[str, tuple[np.ndarray, torch.dtype | None]] = {
        name: (np.asarray(rollout.states[name]), dtype) for name, dtype in _FORWARD_FIELDS
    }
    for _, _, mask_field, active_field in _HEADS:
        fields[mask_field] = (np.asarray(getattr(rollout, mask_field)), torch.bool)
        fields[active_field] = (np.asarray(getattr(rollout, active_field)), torch.bool)
    states = {
        name: torch.as_tensor(value.reshape(-1, *value.shape[2:])[index]).to(
            device=device, dtype=dtype
        )
        for name, (value, dtype) in fields.items()
    }
    return states, float(np.asarray(rollout.final_money, dtype=np.float64).mean())


def _label(path: Path) -> str:
    """A member's name in the report, unique across the paths actually given.

    A bare file name collides the moment the members are per-run artifacts:
    `runs/ab-kl-s1/bc-actor.pt` and `runs/ab-baseline/bc-actor.pt` are both
    `bc-actor.pt`, which silently labels every row of the matrix identically.
    League snapshots are the opposite case -- one directory, distinct file names
    -- so neither half of the path is sufficient alone.
    """
    return path.name if path.parent == Path() else f"{path.parent.name}/{path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    reference, architecture = _load(args.reference, device)
    shared, reference_money = _rollout_states(
        reference,
        architecture=architecture,
        games=args.games,
        steps=args.steps,
        seed=args.seed,
        device=device,
        rows=args.rows,
    )
    # Every member is scored on the reference's states, which is what makes the
    # matrix entries comparable: scoring each on its own visited states would
    # confound flattened weights with a moved state distribution, and separating
    # those two is the whole point of the per-checkpoint entropy pair below.
    logits = [_unit_logits(reference, shared)]
    names = [_label(args.reference)]
    records: list[dict[str, Any]] = []
    for path in args.checkpoints:
        actor, actor_architecture = _load(path, device)
        if actor_architecture != architecture:
            raise ValueError(
                f"{path.name} is {actor_architecture}, not the reference's {architecture}"
            )
        own, money = _rollout_states(
            actor,
            architecture=architecture,
            games=args.games,
            steps=args.steps,
            seed=args.seed,
            device=device,
            rows=args.rows,
        )
        logits.append(_unit_logits(actor, shared))
        names.append(_label(path))
        record = {
            "checkpoint": _label(path),
            "money": money,
            "entropy_on_reference_states": _head_entropies(actor, shared),
            "entropy_on_own_states": _head_entropies(actor, own),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    matrix = population_disagreement(logits, shared["unit_masks"], shared["unit_active"])
    for record, row in zip(records, matrix[1:].tolist(), strict=True):
        record["disagreement_on_reference_states"] = row[0]

    report = {
        "reference": _label(args.reference),
        "reference_money": reference_money,
        "reference_entropy": _head_entropies(reference, shared),
        "games": args.games,
        "rows": args.rows,
        "seed": args.seed,
        "members": names,
        "disagreement_matrix": matrix.tolist(),
        "disagreement_mean": mean_off_diagonal(matrix),
        "checkpoints": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
