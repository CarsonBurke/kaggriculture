#!/usr/bin/env python3
"""Separate "the weights were destroyed" from "sampling breaks the program".

`runs/ppo-bc5mix` warm-starts PPO from a behavior clone that is verified strong
under DETERMINISTIC play (151,799 own money against `starter`, score rate
1.0000), and the run's `money_mean` falls from ~48,000 to ~3,900 within four
iterations of unfreezing the actor. Two incompatible stories fit that journal:

  1. the update destroyed the weights, so the policy no longer knows how to
     farm at all; or
  2. the weights are intact and the *wave* is the casualty -- the rollout is
     collected at temperature 1.0 (pinned by the replay-parity contract in
     `rollout.py`), farming is a long precise multi-step program, and a small
     per-step deviation rate breaks the chain multiplicatively.

The two stories are separated by one measurement: play the same checkpoints on
the same seeds against the same opponent under both decode modes. Argmax money
that stays high while sampled money collapses is story 2; argmax money that
collapses with it is story 1.

Both cells of a checkpoint therefore run identical seeds, opponent and horizon
and differ only in the decode rule, so their difference is attributable to
decode alone. Because the native league seat is `seed % 2`, a contiguous seed
block covers both seats in equal numbers and every cell inherits the same seat
assignment, which keeps the comparison paired across cells rather than merely
matched in aggregate.

Money is reported with the policy's own sharpness on the states each mode
actually visits -- mean entropy per active component, mean probability of the
top action, the realized argmax-agreement rate, and the implied number of
off-argmax components per episode. Entropy alone cannot distinguish a peaked
policy in familiar states from a peaked policy that sampling has walked into
unfamiliar ones; entropy measured separately along each mode's own trajectory
can, and that is the quantity the journal's rising `rollout_entropy` is made of.

Two independent simulators back the argmax numbers. The wave runs in the native
batched Rust engine (`collect_mixed_play_rust`), which is what training uses;
the cross-check replays a handful of games through the official
`kaggle_environments` simulator with the same `CheckpointAgent` the run's own
external evaluator uses, on the same seeds and seats, so its output is directly
comparable to the `external_eval` records in `metrics-external.jsonl`.

CPU-only and single-threaded by default: the model is 1.4M parameters, the
comparison is a handful of waves, and the GPU is not this probe's to take.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROBE_FORMAT_VERSION = 1

# `scripts/` is importable when this file is run directly, but a worker that
# reaches for the official-simulator game loop must not depend on that.
_SCRIPTS = Path(__file__).resolve().parent


# The competition's own relative bank score, mirrored from `relative_score` in
# rust/kagg_env/src/core.rs so a cell reports the objective and not only money.
def relative_score(own: float, opponent: float) -> float:
    total = own + opponent
    return 0.0 if total == 0.0 else (own - opponent) / total


@dataclass(frozen=True)
class Entry:
    """One set of actor weights, named by every path that carries them."""

    iteration: int
    path: Path
    aliases: tuple[str, ...]
    actor_digest: str

    @property
    def label(self) -> str:
        return f"iteration-{self.iteration:06d}"


@dataclass(frozen=True)
class Cell:
    entry_index: int
    path: str
    deterministic: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("runs/ppo-bc5mix"),
        help="run directory to discover checkpoints in",
    )
    parser.add_argument(
        "--from-iteration",
        type=int,
        default=40,
        help="lowest iteration to measure (the frozen-actor baseline)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="extra actor artifact or resume checkpoint to measure; repeatable",
    )
    parser.add_argument(
        "--reference-actor",
        type=Path,
        default=Path("runs/bc5-mixed-conv/bc-actor.pt"),
        help="behavior clone the run warm-started from; skipped when absent",
    )
    parser.add_argument(
        "--no-league-snapshots",
        action="store_true",
        help="measure only checkpoint-*/latest.pt, not the per-iteration league snapshots",
    )
    parser.add_argument("--opponent", default="starter", help="built-in opponent for every cell")
    parser.add_argument("--games", type=int, default=16, help="seeds per cell; both seats")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=20260818)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument(
        "--sharpness-states",
        type=int,
        default=4096,
        help="states per cell replayed through torch for entropy and top-action probability",
    )
    parser.add_argument(
        "--kaggle-seeds",
        type=int,
        default=2,
        help="seeds cross-checked against the official simulator, both seats; 0 disables",
    )
    parser.add_argument("--device", default="cpu", help="torch device; this probe is CPU-only")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--rayon-threads",
        type=int,
        default=2,
        help="native engine worker threads; kept small to respect the thermal ceiling",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="cell processes; at most 2 by policy on this workstation",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


# --- Checkpoint discovery ----------------------------------------------------


def _actor_digest(payload: dict[str, Any]) -> str:
    """Stable digest of the actor weights alone, ignoring optimizer/RNG state.

    Identity is what makes a redundant cell skippable and an equality claim
    ("iteration 40 still holds the clone's weights") checkable, so it is
    computed from the tensors rather than from the file.
    """
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(payload["actor"].items()):
        digest.update(name.encode())
        assert isinstance(tensor, torch.Tensor)
        digest.update(np.ascontiguousarray(tensor.detach().float().numpy()).tobytes())
    return digest.hexdigest()


def _candidate_paths(args: argparse.Namespace) -> list[Path]:
    """Discovery order is also priority order: later paths only add aliases."""
    run = args.run
    candidates = [
        path
        for path in sorted(run.glob("checkpoint-*.pt"))
        if int(path.stem.split("-")[-1]) >= args.from_iteration
    ]
    latest = run / "latest.pt"
    if latest.is_file():
        candidates.append(latest)
    if not args.no_league_snapshots:
        candidates.extend(
            path
            for path in sorted((run / "league").glob("league-actor-*.pt"))
            if int(path.stem.split("-")[-1]) >= args.from_iteration
        )
    if args.reference_actor is not None and args.reference_actor.is_file():
        candidates.append(args.reference_actor)
    candidates.extend(args.checkpoint)
    return [path for path in candidates if path.is_file()]


def discover_entries(args: argparse.Namespace) -> list[Entry]:
    """Unique actor weights to measure, keyed by digest and labelled by iteration.

    The reference clone, `checkpoint-000000.pt`, `checkpoint-000040.pt` and
    `league-actor-00000040.pt` are all the same tensors during a frozen-actor
    warmup, and `latest.pt` duplicates the last league snapshot. Measuring one
    of each set and recording the aliases spends the wave budget on distinct
    policies instead of on re-proving that a frozen actor did not move.
    """
    import torch

    by_digest: dict[str, dict[str, Any]] = {}
    for path in _candidate_paths(args):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        digest = _actor_digest(payload)
        iteration = int(payload.get("iteration", 0))
        record = by_digest.get(digest)
        if record is None:
            by_digest[digest] = {
                "iteration": iteration,
                "path": path,
                "aliases": [path.as_posix()],
            }
            continue
        record["aliases"].append(path.as_posix())
        # The reference clone reports iteration 0 while carrying the same
        # weights the warmup ends on; the measured policy is named by the
        # iteration it belongs to in the run under study.
        record["iteration"] = max(record["iteration"], iteration)
    entries = [
        Entry(
            iteration=record["iteration"],
            path=record["path"],
            aliases=tuple(record["aliases"]),
            actor_digest=digest,
        )
        for digest, record in by_digest.items()
    ]
    entries.sort(key=lambda entry: entry.iteration)
    if not entries:
        raise SystemExit("no checkpoints matched; nothing to measure")
    return entries


# --- Per-cell measurement ---------------------------------------------------

_WORKER: dict[str, Any] = {}


def _initialize_worker(device: str, torch_threads: int) -> None:
    import torch

    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    _WORKER["device"] = device


def _load_actor(path: str) -> tuple[Any, Any]:
    """One actor and its orientation, per worker process per artifact.

    Resume checkpoints and exported actor artifacts share `load_actor_artifact`;
    per-iteration league snapshots are a separate, narrower format (version 2)
    with its own strict reader, and including them is what turns a two-point
    before/after into the per-iteration trace of the collapse. The recorded
    orientation travels with the weights: a population member trained under a
    mirror renders every board flipped, and playing it upright would measure
    the rendering change rather than the policy. Payloads that predate the
    field read as identity, which is what those runs played under.
    """
    import torch

    from kaggriculture.inference import (
        checkpoint_orientation,
        load_actor_artifact,
    )
    from kaggriculture.league import LEAGUE_SNAPSHOT_FORMAT_VERSION, load_actor_snapshot
    from kaggriculture.orientation import Orientation

    cached = _WORKER.get("actor")
    if cached is not None and _WORKER.get("actor_path") == path:
        return cached
    version = torch.load(path, map_location="meta", weights_only=False).get("format_version")
    if version == LEAGUE_SNAPSHOT_FORMAT_VERSION:
        actor = load_actor_snapshot(Path(path)).to(_WORKER["device"])
        actor.eval()
        orientation = Orientation.IDENTITY
    else:
        actor, payload = load_actor_artifact(Path(path), device=_WORKER["device"])
        orientation = checkpoint_orientation(payload)
    _WORKER["actor"] = (actor, orientation)
    _WORKER["actor_path"] = path
    return actor, orientation


def _flat(array: np.ndarray) -> np.ndarray:
    """Collapse (trajectory, step, ...) to (state, ...) without copying."""
    return array.reshape((-1, *array.shape[2:]))


def policy_sharpness(actor, batch, *, limit: int, seed: int, chunk: int = 256) -> dict[str, Any]:
    """Entropy, top-action probability and argmax agreement on visited states.

    Measured on the states of the wave that produced `batch`, because that is
    the distinction the probe exists to draw: the same weights are razor sharp
    on the states argmax visits and can be far less sharp on the states their
    own sampling walks into. Weighting is per active policy component over all
    three heads -- the denominator `RolloutBatch.mean_entropy` and the update's
    `entropy` metric both use -- so the entropy here is directly comparable to
    the journal's, and `entropy_native_all_states` reports the engine's own
    figure over every state as a check on the subsample.
    """
    import torch

    from kaggriculture.policy import categorical_statistics, mask_logits

    valid = np.flatnonzero(_flat(batch.valid))
    generator = np.random.default_rng(seed)
    size = min(limit, valid.size)
    picked = np.sort(generator.choice(valid, size=size, replace=False))
    states = {name: _flat(array) for name, array in batch.states.items()}
    fields = {
        name: _flat(getattr(batch, name))
        for name in (
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "unit_masks",
            "market_kind_masks",
            "market_quantity_masks",
            "unit_active",
            "market_active",
            "market_quantity_active",
        )
    }
    totals = dict.fromkeys(("entropy", "top", "selected", "agreement", "components"), 0.0)
    per_head: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for start in range(0, size, chunk):
            index = picked[start : start + chunk]
            output = actor(
                torch.from_numpy(states["board"][index]).to(torch.float32),
                torch.from_numpy(states["global_features"][index]).to(torch.float32),
                torch.from_numpy(states["units"][index]).to(torch.float32),
                torch.from_numpy(states["unit_positions"][index]).to(torch.long),
            )
            kinds = torch.from_numpy(fields["market_kinds"][index]).to(torch.long)
            heads = (
                (
                    "unit",
                    output.unit_logits,
                    fields["unit_masks"],
                    fields["unit_actions"],
                    fields["unit_active"],
                ),
                (
                    "market_kind",
                    output.market_kind_logits,
                    fields["market_kind_masks"],
                    fields["market_kinds"],
                    fields["market_active"],
                ),
                (
                    "market_quantity",
                    actor.quantity_logits(output.market_quantity_context, kinds),
                    fields["market_quantity_masks"],
                    fields["market_quantities"],
                    fields["market_quantity_active"],
                ),
            )
            for name, logits, masks, actions, active in heads:
                mask = torch.from_numpy(masks[index]).to(torch.bool)
                taken = torch.from_numpy(actions[index]).to(torch.long)
                weight = torch.from_numpy(active[index]).to(torch.float32)
                selected_logprob, entropy = categorical_statistics(
                    logits, mask, taken, validate_mask=False
                )
                probabilities = mask_logits(logits, mask, validate=False).softmax(dim=-1)
                top_probability, top_action = probabilities.max(dim=-1)
                contributions = {
                    "entropy": (entropy * weight).sum(),
                    "top": (top_probability * weight).sum(),
                    "selected": (selected_logprob.exp() * weight).sum(),
                    "agreement": (top_action.eq(taken).to(torch.float32) * weight).sum(),
                    "components": weight.sum(),
                }
                head = per_head.setdefault(name, dict.fromkeys(contributions, 0.0))
                for key, value in contributions.items():
                    totals[key] += float(value)
                    head[key] += float(value)

    def summarize(sums: dict[str, float]) -> dict[str, float | int]:
        components = sums["components"]
        divisor = max(components, 1.0)
        return {
            "components": round(components),
            "entropy": sums["entropy"] / divisor,
            "top_action_probability": sums["top"] / divisor,
            "selected_action_probability": sums["selected"] / divisor,
            "argmax_agreement": sums["agreement"] / divisor,
        }

    report: dict[str, Any] = summarize(totals)
    report["states"] = int(size)
    report["heads"] = {name: summarize(sums) for name, sums in sorted(per_head.items())}
    return report


def measure_cell(cell: Cell, args: dict[str, Any]) -> dict[str, Any]:
    """One wave under one decode rule, plus the sharpness of the states it saw."""
    from kaggriculture.rollout import collect_mixed_play_rust

    actor, orientation = _load_actor(cell.path)
    # The native mixed collector encodes every row upright; only the population
    # collector carries per-row orientations. A mirrored member probed here
    # would read a rendering it never trained on, so refuse rather than
    # mis-measure -- identity artifacts and league snapshots are unaffected.
    if orientation is not Orientation.IDENTITY:
        raise ValueError(
            f"{cell.path}: orientation {orientation.name} requires a surface that "
            "replays it; the native mixed collector renders upright"
        )
    games = int(args["games"])
    started = time.perf_counter()
    batch = collect_mixed_play_rust(
        actor,
        opponents=(),
        self_play_games=0,
        league_games=games,
        opponent_indices=np.zeros(games, dtype=np.int64),
        builtin_lanes=(args["opponent"],),
        seed_start=int(args["seed_start"]),
        episode_steps=int(args["episode_steps"]),
        deterministic=cell.deterministic,
        # Pinned by the replay-parity contract and by the question: this is the
        # temperature the training wave is actually collected at.
        temperature=1.0,
        sampling_seed=int(args["sampling_seed"]),
        forward_mode="eager",
    )
    own = np.asarray(batch.final_money, dtype=np.float64)
    opponent = np.asarray(batch.opponent_money, dtype=np.float64)
    seats = np.asarray(batch.seats, dtype=np.int64)
    seeds = np.asarray(batch.episode_seeds, dtype=np.int64)
    scores = np.asarray([relative_score(a, b) for a, b in zip(own, opponent, strict=True)])
    components = int(
        batch.unit_active.sum() + batch.market_active.sum() + batch.market_quantity_active.sum()
    )
    states = int(batch.valid.sum())
    sharpness = policy_sharpness(
        actor,
        batch,
        limit=int(args["sharpness_states"]),
        seed=int(args["sampling_seed"]) ^ (0x5A17 if cell.deterministic else 0x2E11),
    )
    components_per_state = components / max(states, 1)
    components_per_episode = components_per_state * (int(args["episode_steps"]) - 1)
    deviation_rate = 1.0 - sharpness["top_action_probability"]
    return {
        "entry_index": cell.entry_index,
        "decode": "deterministic" if cell.deterministic else "sampled",
        "games": int(own.size),
        "money_mean": float(own.mean()),
        "money_median": float(np.median(own)),
        "money_min": float(own.min()),
        "money_max": float(own.max()),
        "opponent_money_mean": float(opponent.mean()),
        "relative_score_mean": float(scores.mean()),
        "win_rate": float((own > opponent).mean()),
        "money_by_seat": {
            str(seat): float(own[seats == seat].mean()) for seat in sorted(set(seats.tolist()))
        },
        "per_seed": [
            {"seed": int(seed), "seat": int(seat), "money": float(value)}
            for seed, seat, value in zip(seeds, seats, own, strict=True)
        ],
        "entropy_native_all_states": float(batch.mean_entropy),
        "entropy": sharpness["entropy"],
        "top_action_probability": sharpness["top_action_probability"],
        "selected_action_probability": sharpness["selected_action_probability"],
        "argmax_agreement": sharpness["argmax_agreement"],
        "components_per_state": components_per_state,
        "components_per_episode": components_per_episode,
        "expected_off_argmax_components_per_episode": deviation_rate * components_per_episode,
        "argmax_clean_episode_probability": float(
            np.exp(components_per_episode * np.log(max(sharpness["top_action_probability"], 1e-12)))
        ),
        "sharpness": sharpness,
        "states": states,
        "wave_seconds": time.perf_counter() - started,
    }


def measure_official(cell: Cell, args: dict[str, Any]) -> dict[str, Any]:
    """Argmax money in the official simulator, on the run's own eval protocol.

    `scripts/evaluate_checkpoint.py` cannot be used here: it calls
    `require_source_identity`, which demands the checkout match the artifact's
    recorded identity exactly, and this tree is deliberately dirty while the
    collapse is being investigated. The parts that decide the number -- a
    deterministic `act_batch` agent and the seed/seat sweep of
    `scripts/external_eval_worker.py` -- are reused as they are, so the result
    is directly comparable to the `external_eval` records the run already wrote.

    The agent MUST be a plain single-argument function, not
    `inference.CheckpointAgent`. `kaggle_environments` sizes an agent's call by
    `getfullargspec`, which counts `self` for a callable object, so it invokes
    such an object with (observation, configuration); the resulting TypeError is
    swallowed under `debug=False`, the seat submits nothing for all 719 steps,
    and the episode still reports status DONE with the bank untouched at 3,000.
    That failure is indistinguishable from a policy that stopped farming, which
    is exactly the quantity this probe exists to measure, so it is written out
    here rather than left as a call-shape detail.
    """
    from kaggriculture.policy import act_batch

    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from external_eval_worker import _play_game

    actor, orientation = _load_actor(cell.path)

    def agent(observation: dict[str, Any]) -> dict[str, Any]:
        return act_batch(
            actor, [observation], deterministic=True, orientation=orientation
        ).actions[0]

    started = time.perf_counter()
    outcomes = [
        _play_game(agent, args["opponent"], seed, seat, int(args["episode_steps"]))
        for seed in range(int(args["seed_start"]), int(args["seed_start"]) + int(args["seeds"]))
        for seat in (0, 1)
    ]
    complete = [outcome for outcome in outcomes if outcome.error is None]
    errors = sorted({outcome.error for outcome in outcomes if outcome.error})
    return {
        "entry_index": cell.entry_index,
        "games": len(outcomes),
        "completed_games": len(complete),
        "money_mean": (
            statistics.fmean(outcome.snapshot_money for outcome in complete) if complete else None
        ),
        "opponent_money_mean": (
            statistics.fmean(outcome.opponent_money for outcome in complete) if complete else None
        ),
        "per_game": [
            {
                "seed": outcome.seed,
                "seat": outcome.snapshot_seat,
                "money": outcome.snapshot_money,
                "opponent_money": outcome.opponent_money,
            }
            for outcome in outcomes
        ],
        "errors": errors,
        "seconds": time.perf_counter() - started,
    }


def _run_cell(payload: tuple[str, Cell, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    kind, cell, args = payload
    if kind == "native":
        return kind, measure_cell(cell, args)
    return kind, measure_official(cell, args)


# --- Reporting ---------------------------------------------------------------


def format_table(report: dict[str, Any]) -> str:
    header = (
        f"{'checkpoint':>12} {'argmax money':>13} {'sampled money':>14} "
        f"{'H(argmax)':>10} {'H(sampled)':>11} {'p_top(arg)':>11} {'p_top(sam)':>11} "
        f"{'agree(sam)':>11} {'official':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in report["checkpoints"]:
        deterministic = row["deterministic"]
        sampled = row["sampled"]
        official = row.get("official") or {}
        official_money = official.get("money_mean")
        lines.append(
            f"{row['iteration']:>12} "
            f"{deterministic['money_mean']:>13,.0f} "
            f"{sampled['money_mean']:>14,.0f} "
            f"{deterministic['entropy']:>10.5f} "
            f"{sampled['entropy']:>11.5f} "
            f"{deterministic['top_action_probability']:>11.5f} "
            f"{sampled['top_action_probability']:>11.5f} "
            f"{sampled['argmax_agreement']:>11.5f} "
            + (f"{official_money:>10,.0f}" if official_money is not None else f"{'-':>10}")
        )
    parities = [
        row["native_official_parity"]
        for row in report["checkpoints"]
        if (row.get("native_official_parity") or {}).get("max_absolute_difference") is not None
    ]
    if parities:
        worst = max(parity["max_absolute_difference"] for parity in parities)
        compared = sum(parity["compared_games"] for parity in parities)
        lines.append(
            "native vs official argmax money on identical seed and seat: "
            f"max absolute difference {worst:,.1f} over {compared} games"
        )
    return "\n".join(lines)


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    """State, in numbers, which of the two stories the table supports."""
    rows = report["checkpoints"]
    baseline = rows[0]
    final = rows[-1]
    argmax_retention = final["deterministic"]["money_mean"] / max(
        baseline["deterministic"]["money_mean"], 1.0
    )
    sampled_retention = final["sampled"]["money_mean"] / max(baseline["sampled"]["money_mean"], 1.0)
    return {
        "baseline_iteration": baseline["iteration"],
        "final_iteration": final["iteration"],
        "argmax_money_baseline": baseline["deterministic"]["money_mean"],
        "argmax_money_final": final["deterministic"]["money_mean"],
        "argmax_money_retained_fraction": argmax_retention,
        "sampled_money_baseline": baseline["sampled"]["money_mean"],
        "sampled_money_final": final["sampled"]["money_mean"],
        "sampled_money_retained_fraction": sampled_retention,
        "argmax_survived": bool(argmax_retention >= 0.5),
        "sampling_gap_baseline": (
            baseline["deterministic"]["money_mean"] - baseline["sampled"]["money_mean"]
        ),
        "sampling_gap_final": final["deterministic"]["money_mean"] - final["sampled"]["money_mean"],
        "native_official_max_absolute_difference": (
            max(
                row["native_official_parity"]["max_absolute_difference"]
                for row in rows
                if (row.get("native_official_parity") or {}).get("max_absolute_difference")
                is not None
            )
            if any(
                (row.get("native_official_parity") or {}).get("max_absolute_difference") is not None
                for row in rows
            )
            else None
        ),
    }


def _parity(
    deterministic: dict[str, Any], official: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Compare the two simulators where they play the identical game.

    The native league seat is `seed % 2`, so only half of the official sweep's
    seed/seat pairs have a native counterpart. Those halves must agree to the
    last unit of money: same weights, same argmax rule, same seed, same seat. A
    nonzero difference here means one of the two numbers in the table below is
    measuring something other than the game, and the verdict is void.
    """
    if official is None:
        return None
    native_money = {
        (game["seed"], game["seat"]): game["money"] for game in deterministic["per_seed"]
    }
    pairs = [
        {
            "seed": game["seed"],
            "seat": game["seat"],
            "native_money": native_money[(game["seed"], game["seat"])],
            "official_money": game["money"],
        }
        for game in official["per_game"]
        if (game["seed"], game["seat"]) in native_money and game["money"] is not None
    ]
    return {
        "compared_games": len(pairs),
        "max_absolute_difference": (
            max(abs(pair["native_money"] - pair["official_money"]) for pair in pairs)
            if pairs
            else None
        ),
        "pairs": pairs,
    }


def main() -> None:
    args = parse_args()
    if args.games < 8:
        raise SystemExit("--games must be at least 8 so both seats carry 4+ seeds")
    if args.workers < 1 or args.workers > 2:
        raise SystemExit("--workers must be 1 or 2 on this workstation")
    if args.device != "cpu":
        raise SystemExit("this probe is CPU-only by contract")
    # Set before the native module or torch can build a thread pool.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(args.torch_threads, 1)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(args.torch_threads, 1)))
    os.environ["RAYON_NUM_THREADS"] = str(max(args.rayon_threads, 1))

    import torch

    torch.set_num_threads(max(args.torch_threads, 1))
    started = time.perf_counter()
    entries = discover_entries(args)
    shared = {
        "games": args.games,
        "seed_start": args.seed_start,
        "sampling_seed": args.sampling_seed,
        "episode_steps": args.episode_steps,
        "opponent": args.opponent,
        "sharpness_states": args.sharpness_states,
        "torch_threads": args.torch_threads,
        "seeds": args.kaggle_seeds,
    }
    work: list[tuple[str, Cell, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        for deterministic in (True, False):
            work.append(("native", Cell(index, entry.path.as_posix(), deterministic), shared))
        if args.kaggle_seeds > 0:
            work.append(("official", Cell(index, entry.path.as_posix(), True), shared))

    native: dict[tuple[int, str], dict[str, Any]] = {}
    official: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(args.device, args.torch_threads),
    ) as pool:
        for kind, result in pool.map(_run_cell, work):
            if kind == "native":
                native[(result["entry_index"], result["decode"])] = result
                print(
                    json.dumps(
                        {
                            "iteration": entries[result["entry_index"]].iteration,
                            "decode": result["decode"],
                            "money_mean": round(result["money_mean"], 1),
                            "entropy": round(result["entropy"], 6),
                            "top_action_probability": round(result["top_action_probability"], 6),
                            "wave_seconds": round(result["wave_seconds"], 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                official[result["entry_index"]] = result

    checkpoints = []
    for index, entry in enumerate(entries):
        deterministic = native[(index, "deterministic")]
        row = {
            "iteration": entry.iteration,
            "path": entry.path.as_posix(),
            "aliases": list(entry.aliases),
            "actor_sha256": entry.actor_digest,
            "deterministic": deterministic,
            "sampled": native[(index, "sampled")],
            "official": official.get(index),
        }
        row["native_official_parity"] = _parity(deterministic, row["official"])
        checkpoints.append(row)
    report: dict[str, Any] = {
        "probe": "sampling_collapse",
        "probe_format_version": PROBE_FORMAT_VERSION,
        "command": shlex.join([sys.executable, *sys.argv]),
        "arguments": {
            key: (value.as_posix() if isinstance(value, Path) else value)
            for key, value in sorted(vars(args).items())
            if key != "checkpoint"
        },
        "extra_checkpoints": [path.as_posix() for path in args.checkpoint],
        "checkpoints": checkpoints,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["verdict"] = verdict(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(format_table(report), flush=True)
    print(json.dumps(report["verdict"], indent=1, sort_keys=True), flush=True)
    print(f"wrote {args.output} in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
