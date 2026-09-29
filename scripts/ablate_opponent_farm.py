#!/usr/bin/env python3
"""Measure how much the trained policy and value actually use the opponent farm.

The structured trunk pushes both farms -- 200 tile tokens -- through every
`farm_local` block, and the opponent half of that result reaches the rest of the
network only as `opponent_summary`, which is `opponent_latents` tokens wide
(`structured.py` `StructuredTrunk.forward`). A FakeTensor traffic model puts the
farm blocks at 39% of the actor's forward bytes, so the opponent half is about
19% of a forward that is memory-bound by roughly 17x. That is a large bill for a
compression to eight tokens, and the question this script answers is whether the
network cashes it: if the opponent farm's *content* barely moves the policy or
the value, the block depth spent on it is not buying prediction.

The intervention is a permutation, not an erasure. Replacing the opponent farm
with zeros or with a constant asks "what happens off the data manifold", whose
answer is uninformative -- any network reacts to an input it has never seen.
Permuting the opponent half across the batch keeps every marginal statistic of
the input exactly intact and destroys only the pairing between a state and its
own opponent's board. A network that reads the opponent farm for content must
move; a network that only reads it for the occupancy statistics it shares with
every other state in the batch will not.

`own` is the positive control, and it is the reason the numbers below can be
believed. The same permutation applied to the learner's own farm must produce a
large shift; if it does not, the measurement is broken rather than the finding
being interesting. `both` bounds the pair.

States come from a real native wave played by the audited actor against itself,
so the distribution is the one the update actually trains on rather than a
scripted proxy. Everything runs on CPU: this is a diagnostic, and it must not
contend with a training run for the device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kaggriculture.ppo import actor_forward_args
from kaggriculture.registry import pair_towers, resolve_architecture
from kaggriculture.rollout import collect_mixed_play_rust
from kaggriculture.tokens import TILE_COUNT

#: Tile tokens are the own farm followed by the opponent farm, so a farm is a
#: contiguous slice of the token axis and an intervention is a slice assignment.
_FARM_SLICES = {
    "own": slice(0, TILE_COUNT),
    "opponent": slice(TILE_COUNT, 2 * TILE_COUNT),
}
_TILE_FIELDS = ("tile_categorical", "tile_continuous")


def _load(path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = resolve_architecture(checkpoint["architecture"])
    config = architecture.build_config(checkpoint["model_config"])
    actor = architecture.actor_class(config).to(device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    critic = architecture.critic_class(config).to(device)
    pair_towers(actor, critic)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    actor.eval().requires_grad_(False)
    critic.eval().requires_grad_(False)
    return actor, critic, checkpoint


def _permuted(states: dict[str, np.ndarray], farms: tuple[str, ...], rng) -> dict[str, np.ndarray]:
    """Copy `states` with each named farm's tokens permuted across the batch."""
    if not farms:
        return states
    permuted = dict(states)
    order = rng.permutation(len(states["tile_categorical"]))
    for field in _TILE_FIELDS:
        values = states[field].copy()
        for farm in farms:
            tokens = _FARM_SLICES[farm]
            values[:, tokens] = states[field][order][:, tokens]
        permuted[field] = values
    return permuted


def _log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


def _mean_kl(reference: torch.Tensor, candidate: torch.Tensor, active: torch.Tensor) -> float:
    """Mean KL(reference || candidate) over active rows of a categorical head."""
    reference_log = _log_probabilities(reference)
    candidate_log = _log_probabilities(candidate)
    divergence = (reference_log.exp() * (reference_log - candidate_log)).sum(dim=-1)
    weights = active.float()
    total = weights.sum()
    if total.item() == 0.0:
        return 0.0
    return float((divergence * weights).sum() / total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4, help="self-play games in the sample wave")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--states", type=int, default=1024, help="states scored per arm")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    device = torch.device("cpu")
    actor, critic, checkpoint = _load(arguments.checkpoint, device)
    if getattr(actor.config, "action_interface", 1) == 2:
        raise ValueError("this counterfactual probe cannot reconstruct ALL quantity masks")
    architecture = checkpoint["architecture"]
    rollout = collect_mixed_play_rust(
        actor,
        self_play_games=arguments.games,
        seed_start=arguments.seed,
        episode_steps=arguments.episode_steps,
        forward_mode="eager",
    )

    rng = np.random.default_rng(arguments.seed)
    flat = {
        name: values.reshape((-1, *values.shape[2:])) for name, values in rollout.states.items()
    }
    flat["unit_active"] = rollout.unit_active.reshape((-1, *rollout.unit_active.shape[2:]))
    sample = rng.choice(len(flat["tile_categorical"]), size=arguments.states, replace=False)
    states = {name: values[sample] for name, values in flat.items()}

    arms = {
        "identity": (),
        "opponent": ("opponent",),
        "own": ("own",),
        "both": ("own", "opponent"),
    }
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for name, farms in arms.items():
        intervened = _permuted(states, farms, np.random.default_rng(arguments.seed))
        unit_logits, kind_logits, quantity_context, values = [], [], [], []
        for start in range(0, arguments.states, arguments.chunk):
            window = slice(start, start + arguments.chunk)
            chunk = {field: values_[window] for field, values_ in intervened.items()}
            actor_args = actor_forward_args(architecture, chunk, device)
            output = actor(*actor_args)
            unit_logits.append(output.unit_logits)
            kind_logits.append(output.market_kind_logits)
            quantity_context.append(output.market_quantity_context)
            critic_inputs = actor_args[0]._replace(
                products=torch.cat(
                    (
                        actor_args[0].products,
                        torch.from_numpy(chunk["critic_products"]).to(torch.float32),
                    ),
                    dim=-1,
                ),
                animals=torch.cat(
                    (
                        actor_args[0].animals,
                        torch.from_numpy(chunk["critic_animals"]).to(torch.float32),
                    ),
                    dim=-1,
                ),
                crops=torch.cat(
                    (
                        actor_args[0].crops,
                        torch.from_numpy(chunk["critic_crops"]).to(torch.float32),
                    ),
                    dim=-1,
                ),
            )
            logits = critic(
                critic_inputs,
                torch.from_numpy(chunk["opponent_unit_categorical"]).to(torch.long),
                torch.from_numpy(chunk["opponent_unit_continuous"]).to(torch.float32),
                torch.from_numpy(chunk["opponent_unit_active"]).to(torch.bool),
            )
            values.append(critic.value(logits))
        outputs[name] = {
            "unit": torch.cat(unit_logits),
            "kind": torch.cat(kind_logits),
            "quantity_context": torch.cat(quantity_context),
            "value": torch.cat(values).flatten(),
        }

    # The quantity head scores exact amounts only for an already-selected kind,
    # so every arm is conditioned on the *unperturbed* greedy kind. Letting each
    # arm pick its own kind would fold the kind head's shift into the quantity
    # measurement and report one head's movement twice.
    reference_kinds = outputs["identity"]["kind"].argmax(dim=-1)
    for values_ in outputs.values():
        values_["quantity"] = actor.quantity_logits(values_["quantity_context"], reference_kinds)

    unit_active = torch.from_numpy(states["unit_active"]).to(torch.bool)
    reference = outputs["identity"]
    value_spread = float(reference["value"].std())
    # A permuted input cannot move the value further than an unrelated state's
    # value already sits, so the mean absolute gap between the identity arm's
    # own values under a random pairing is the ceiling this metric can reach.
    # Without it a shift of "1.17 spreads" reads as a strong dependence when it
    # actually means the prediction has simply decorrelated completely, and any
    # arm at the ceiling is indistinguishable from any other.
    decorrelated = float(
        (reference["value"][torch.randperm(arguments.states)] - reference["value"]).abs().mean()
    )
    report: dict[str, object] = {
        "checkpoint": str(arguments.checkpoint),
        "iteration": checkpoint.get("iteration"),
        "states": arguments.states,
        "value_std_across_states": value_spread,
        "value_decorrelation_ceiling": decorrelated,
    }
    for name in arms:
        if name == "identity":
            continue
        candidate = outputs[name]
        value_shift = (candidate["value"] - reference["value"]).abs()
        report[name] = {
            "unit_kl": _mean_kl(reference["unit"], candidate["unit"], unit_active),
            "kind_kl": _mean_kl(
                reference["kind"],
                candidate["kind"],
                torch.ones(reference["kind"].shape[:-1], dtype=torch.bool),
            ),
            "quantity_kl": _mean_kl(
                reference["quantity"],
                candidate["quantity"],
                torch.ones(reference["quantity"].shape[:-1], dtype=torch.bool),
            ),
            "value_mean_abs_shift": float(value_shift.mean()),
            # A freshly initialized critic predicts one constant, so its ceiling
            # is zero and the ratio is undefined rather than infinite.
            "value_shift_over_ceiling": (
                float(value_shift.mean()) / decorrelated if decorrelated > 0.0 else None
            ),
            "greedy_unit_action_change_rate": float(
                ((reference["unit"].argmax(-1) != candidate["unit"].argmax(-1)) & unit_active).sum()
                / unit_active.sum()
            ),
        }
    text = json.dumps(report, indent=1)
    print(text)
    if arguments.report is not None:
        arguments.report.write_text(text + "\n")


if __name__ == "__main__":
    main()
