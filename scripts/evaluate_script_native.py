#!/usr/bin/env python3
"""Evaluate an actor artifact against a Python agent file inside the native engine.

The official evaluator steps `kaggle_environments` and the opponent file one
game at a time, which costs about six minutes per 64 games against
demand-advance4. This runs the same paired-seat protocol -- every seed from
both seats, argmax candidate -- as one native batched wave: the candidate is
the collector's deterministic learner and the opponent file plays its seats in
worker processes (`kaggriculture.script_opponents`), a fresh agent namespace
per game seat as Kaggle gives it.

The summary has the official evaluator's shape and statistics, but this is a
screening instrument, not admission evidence: the opponent's turns execute
natively exactly as submitted, yet only a replay proves this report's games
match. ``--parity-seeds`` measures that directly: it replays the candidate's
recorded native actions in `kaggle_environments` against the real opponent
file and compares both banks.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from evaluate_checkpoint import (
    EPISODE_STEPS,
    GameResult,
    _make_environment,
    _opponent_provenance,
    _reward,
    _write_atomic,
    render_json,
    summarize,
)

from kaggriculture.actions import compile_action
from kaggriculture.evaluation import SEED_DOMAINS, artifact_seed_usage, seed_protocol
from kaggriculture.inference import load_actor_artifact
from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.provenance import source_identity
from kaggriculture.rollout import ROLLOUT_FORWARD_MODES, collect_mixed_play_rust
from kaggriculture.script_opponents import ScriptAgentPool, ScriptOpponent

DEFAULT_SEED_CLUSTERS = 64
#: Relative bank gap the native replay may show against the official engine.
DEFAULT_PARITY_TOLERANCE = 1e-3


def _script_opponent(spec: str) -> ScriptOpponent:
    """``NAME=PATH``, or anything `normalize_opponent` resolves to an agent file."""
    name, separator, reference = spec.partition("=")
    if not separator:
        name, reference = "", spec
    label, runnable = normalize_opponent(reference)
    if runnable in BUILTIN_OPPONENTS:
        raise ValueError(f"{spec!r} is a built-in agent, not a Python agent file")
    if not name:
        path = Path(runnable)
        # Agent packages are directories holding a `main.py`.
        name = path.parent.name if path.stem == "main" else Path(label).stem
    return ScriptOpponent.from_path(name, runnable)


def _artifact_record(path: Path, metadata: dict[str, Any], agent: int | None) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    bound = metadata.get("source_identity") or {}
    current = source_identity()
    return {
        "path": str(path),
        "sha256": digest,
        "format_version": int(metadata["format_version"]),
        "architecture": str(metadata.get("architecture", "entity-cnn")),
        "iteration": int(metadata.get("iteration", 0)),
        "agent": agent,
        # Recorded, not enforced: an ablation screen runs artifacts from many
        # trees. The official evaluator is the one that refuses a moved tree.
        "source_identity": bound.get("sha256"),
        "current_source_identity": current["sha256"],
        "source_identity_matches": bound.get("sha256") == current["sha256"],
    }


def _replay_official(
    task: tuple[int, int, str, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Play one recorded native candidate open-loop in `kaggle_environments`."""
    seed, seat, opponent, units, kinds, quantities = task
    turn = 0

    def replay(observation: Any) -> dict[str, Any]:
        nonlocal turn
        action = compile_action(observation, units[turn], kinds[turn], quantities[turn])
        turn += 1
        return action

    players: list[Any] = [opponent, opponent]
    players[seat] = replay
    environment = _make_environment(seed, EPISODE_STEPS)
    environment.run(players)
    final = environment.steps[-1]
    return {
        "seed": seed,
        "seat": seat,
        "turns": turn,
        "candidate_status": str(final[seat].status),
        "opponent_status": str(final[1 - seat].status),
        "official_candidate_bank": _reward(final[seat].reward),
        "official_opponent_bank": _reward(final[1 - seat].reward),
    }


def _relative_gap(native: float, official: float | None) -> float:
    if official is None:
        return math.inf
    return abs(native - official) / max(abs(official), 1.0)


def parity_check(
    batch: Any, games: range, opponent: ScriptOpponent, tolerance: float, workers: int
) -> dict[str, Any]:
    """Compare native banks with an official replay of the same candidate actions.

    The replay is open-loop, so it measures exactly what this report depends
    on: whether the opponent file, run natively, meets the candidate's actions
    the way it would in the official engine. Any divergence it finds (an
    engine mismatch, an agent that behaves differently natively) compounds
    over the game, so a small final gap bounds every step's.
    """
    tasks = [
        (
            int(batch.episode_seeds[game]),
            int(batch.seats[game]),
            str(opponent.path),
            np.asarray(batch.unit_actions[game]),
            np.asarray(batch.market_kinds[game]),
            np.asarray(batch.market_quantities[game]),
        )
        for game in games
    ]
    started = time.perf_counter()
    # Spawned: the parent holds CUDA state, and forking it is undefined.
    with mp.get_context("spawn").Pool(max(1, min(workers, len(tasks)))) as pool:
        replays = pool.map(_replay_official, tasks)
    rows = []
    for game, replay in zip(games, replays, strict=True):
        native_candidate = float(batch.final_money[game])
        native_opponent = float(batch.opponent_money[game])
        rows.append(
            {
                **replay,
                "native_candidate_bank": native_candidate,
                "native_opponent_bank": native_opponent,
                "candidate_relative_gap": _relative_gap(
                    native_candidate, replay["official_candidate_bank"]
                ),
                "opponent_relative_gap": _relative_gap(
                    native_opponent, replay["official_opponent_bank"]
                ),
            }
        )
    worst = max(max(row["candidate_relative_gap"], row["opponent_relative_gap"]) for row in rows)
    return {
        "games": len(rows),
        "tolerance": tolerance,
        "max_relative_gap": worst,
        "within_tolerance": bool(worst <= tolerance)
        and all(
            row["candidate_status"] == "DONE" and row["opponent_status"] == "DONE" for row in rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if not 0 <= args.parity_seeds <= args.seeds:
        raise ValueError("--parity-seeds must lie in [0, --seeds]")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.seed_start is None:
        args.seed_start = SEED_DOMAINS[args.seed_domain][0]
    device = torch.device(args.device)
    forward_mode = args.forward_mode or ("graph" if device.type == "cuda" else "eager")
    opponent = _script_opponent(args.opponent)
    artifact = args.artifact.expanduser().resolve()
    started = time.perf_counter()
    actor, metadata = load_actor_artifact(artifact, device=device, agent=args.agent)
    protocol = seed_protocol(
        args.seed_domain, args.seed_start, args.seeds, usage=artifact_seed_usage(metadata)
    )
    games = 2 * args.seeds
    with ScriptAgentPool([opponent], min(args.workers, games)) as pool:
        pool_seconds = time.perf_counter() - started
        collected = time.perf_counter()
        batch = collect_mixed_play_rust(
            actor,
            league_games=games,
            opponent_indices=np.zeros(games, dtype=np.int64),
            script_lanes=(opponent,),
            script_pool=pool,
            seed_start=args.seed_start,
            paired_league_seats=True,
            deterministic=True,
            forward_mode=forward_mode,
            forward_autocast=args.bf16 and device.type == "cuda",
        )
        wave_seconds = time.perf_counter() - collected
        script_statistics = pool.take_statistics()
    results = [
        GameResult(
            seed=int(batch.episode_seeds[game]),
            candidate_seat=int(batch.seats[game]),
            candidate_reward=float(batch.final_money[game]),
            opponent_reward=float(batch.opponent_money[game]),
            # The collector raises unless every game reaches the horizon.
            candidate_status="DONE",
            opponent_status="DONE",
            environment_done=True,
            steps=EPISODE_STEPS,
            expected_steps=EPISODE_STEPS,
            elapsed_seconds=wave_seconds / games,
        )
        for game in range(games)
    ]
    summary = summarize(results, args.seeds)
    # An errored turn plays PASS, where the official runner would forfeit the
    # seat or fail the episode.
    summary["valid_for_selection"] = bool(
        summary["valid_for_selection"]
        and not script_statistics["agent_errors"]
        and not script_statistics["action_errors"]
    )
    parity = (
        parity_check(
            batch,
            range(2 * args.parity_seeds),
            opponent,
            args.parity_tolerance,
            args.workers,
        )
        if args.parity_seeds
        else None
    )
    return {
        "valid_for_selection": summary["valid_for_selection"],
        "artifact": str(artifact),
        "agent": args.agent,
        "artifact_provenance": _artifact_record(artifact, metadata, args.agent),
        "opponent": str(opponent.path),
        "opponent_label": opponent.name,
        "opponent_provenance": _opponent_provenance(opponent.name, str(opponent.path)),
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seed_protocol": protocol,
        "paired_seats": True,
        "device": str(device),
        "inference": {
            "mode": forward_mode,
            "autocast_dtype": "bfloat16" if args.bf16 and device.type == "cuda" else None,
            "deterministic": True,
            "environment_backend": "native-rust",
            "opponent_backend": "script-agent-pool",
            "script_workers": min(args.workers, games),
        },
        "script_opponent": script_statistics,
        "timing": {
            "setup_seconds": pool_seconds,
            "wave_seconds": wave_seconds,
        },
        "parity": parity,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
        "games": [
            {
                "seed": result.seed,
                "candidate_seat": result.candidate_seat,
                "candidate_reward": result.candidate_reward,
                "opponent_reward": result.opponent_reward,
                "margin": result.margin,
                "outcome": result.outcome,
            }
            for result in results
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--agent",
        type=int,
        default=None,
        help="population member to evaluate; required for a multi-member checkpoint",
    )
    parser.add_argument(
        "--opponent",
        required=True,
        help="Python agent file as PATH or NAME=PATH (or the v27/v16 aliases)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEED_CLUSTERS,
        help="seed clusters; both seats of each run, so the wave holds twice this many games",
    )
    parser.add_argument(
        "--seed-domain", choices=("development", "screening", "finalist"), default="screening"
    )
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        default=None,
        help="collection forward backend; defaults to the captured CUDA graph on CUDA "
        "(no Inductor compile) and eager on CPU",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bf16 CUDA forward, as training collection and --cuda-bf16-compiled use",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="opponent agent processes, and official replay processes for --parity-seeds",
    )
    parser.add_argument(
        "--parity-seeds",
        type=int,
        default=0,
        help="replay this many leading seeds (both seats) in the official engine and compare banks",
    )
    parser.add_argument("--parity-tolerance", type=float, default=DEFAULT_PARITY_TOLERANCE)
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help="exit nonzero when the parity replay exceeds its tolerance",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    rendered = render_json(payload)
    if args.output:
        _write_atomic(args.output, rendered)
    summary = payload["summary"]
    headline = {
        key: summary.get(key)
        for key in ("games", "wins", "ties", "losses", "score_rate", "mean_margin", "margin_95ci")
    }
    headline["mean_candidate_bank"] = float(
        np.mean([game["candidate_reward"] for game in payload["games"]])
    )
    headline["mean_opponent_bank"] = float(
        np.mean([game["opponent_reward"] for game in payload["games"]])
    )
    headline["wave_seconds"] = payload["timing"]["wave_seconds"]
    headline["elapsed_seconds"] = payload["elapsed_seconds"]
    headline["script_errors"] = (
        payload["script_opponent"]["agent_errors"] + payload["script_opponent"]["action_errors"]
    )
    parity = payload["parity"]
    if parity is not None:
        headline["parity_max_relative_gap"] = parity["max_relative_gap"]
        headline["parity_within_tolerance"] = parity["within_tolerance"]
    print(render_json(headline))
    if args.require_parity and parity is not None and not parity["within_tolerance"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
