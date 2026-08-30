#!/usr/bin/env python3
"""Project recorded ladder replays into the same BC archives extract_bc_dataset writes.

Official daily dumps and the public episode parquet store the engine's own
steps. The current meta deposits harvest stacks with PLACE item N — a command
the factored space now represents as deposit-all of that product — so those
seats can be cloned instead of skipped. A seat the ledger still cannot
represent is omitted and counted; aborting the corpus on one adversarial
replay would throw away every other game in the dump.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_bc_dataset import (
    DATASET_FORMAT_VERSION,
    _archived_record,
    commit_dataset_generation,
    extract_episode,
)

from kaggriculture.demonstrations import DemonstrationError
from kaggriculture.provenance import file_sha256, source_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument(
        "--min-bank",
        type=float,
        default=0.0,
        help="keep a seat only when its terminal reward is at least this",
    )
    parser.add_argument(
        "--winners-only",
        action="store_true",
        help="keep only the higher-bank seat; ties keep both",
    )
    return parser.parse_args()


def wrap_steps(raw_steps: list[Any]) -> list[list[SimpleNamespace]]:
    wrapped: list[list[SimpleNamespace]] = []
    for turn in raw_steps:
        wrapped.append(
            [
                SimpleNamespace(
                    observation=seat.get("observation") or {},
                    action=seat.get("action"),
                    reward=seat.get("reward"),
                    status=seat.get("status"),
                )
                for seat in turn
            ]
        )
    return wrapped


def selected_seats(rewards: list[Any], *, min_bank: float, winners_only: bool) -> tuple[int, ...]:
    banks = [float(reward or 0.0) for reward in rewards[:2]]
    if winners_only:
        best = max(banks)
        seats = tuple(seat for seat, bank in enumerate(banks) if bank == best and bank >= min_bank)
    else:
        seats = tuple(seat for seat, bank in enumerate(banks) if bank >= min_bank)
    return seats


def replay_episode_id(path: Path) -> int:
    replay = json.loads(path.read_text(encoding="utf-8"))
    return int((replay.get("info") or {}).get("EpisodeId") or path.stem)


def reject_duplicate_episode_ids(paths: list[Path]) -> None:
    owners: dict[int, Path] = {}
    for path in paths:
        episode_id = replay_episode_id(path)
        previous = owners.setdefault(episode_id, path)
        if previous != path:
            raise ValueError(
                f"duplicate replay EpisodeId {episode_id}: {previous.name} and {path.name}"
            )


def extract_replay(
    path: Path,
    output_dir: Path,
    *,
    episode_steps: int,
    min_bank: float,
    winners_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    episode_id = int((replay.get("info") or {}).get("EpisodeId") or path.stem)
    rewards = list(replay.get("rewards") or [])
    steps = wrap_steps(replay["steps"])
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for seat in selected_seats(rewards, min_bank=min_bank, winners_only=winners_only):
        archive = output_dir / f"episode-{episode_id:08d}-seat{seat}.npz"
        try:
            arrays = extract_episode(steps, seat, episode_steps=episode_steps)
        except DemonstrationError as error:
            skipped.append(
                {
                    "episode_id": episode_id,
                    "seat": seat,
                    "file": path.name,
                    "error": str(error),
                    "reward": float(rewards[seat]) if seat < len(rewards) else None,
                }
            )
            continue
        np.savez_compressed(archive, **arrays)
        record = _archived_record(archive, episode_id, seat, episode_steps)
        record["episode_id"] = episode_id
        record["engine_version"] = replay.get("module_version")
        record["reward"] = float(rewards[seat]) if seat < len(rewards) else None
        record["replay_file"] = path.name
        record["replay_sha256"] = file_sha256(path)
        kept.append(record)
    return kept, skipped


def main() -> None:
    args = parse_args()
    if args.episode_steps != 720:
        raise ValueError("extraction needs the competition horizon")
    replays = sorted(args.replay_dir.expanduser().resolve().glob("*.json"))
    if not replays:
        raise FileNotFoundError(f"{args.replay_dir}: no replay json")
    reject_duplicate_episode_ids(replays)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    ) as temporary:
        staging_dir = Path(temporary)
        episodes: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for path in replays:
            kept, missed = extract_replay(
                path,
                staging_dir,
                episode_steps=args.episode_steps,
                min_bank=args.min_bank,
                winners_only=args.winners_only,
            )
            episodes.extend(kept)
            skipped.extend(missed)
            print(
                f"{path.name}: kept {len(kept)} skipped {len(missed)} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
        episodes.sort(key=lambda record: (record["seed"], record["seat"]))
        manifest = {
            "format_version": DATASET_FORMAT_VERSION,
            "teacher": {"label": "ladder-replay", "sha256": None},
            "opponent": {"label": "ladder-replay", "sha256": None},
            "episode_steps": args.episode_steps,
            "seed_start": min((record["seed"] for record in episodes), default=0),
            "episodes": episodes,
            "skipped": skipped,
            "extractor_source_identity": source_identity(),
            "command": sys.argv,
        }
        manifest_path = commit_dataset_generation(staging_dir, output_dir, manifest)
    print(
        f"wrote {len(episodes)} episode-seats, skipped {len(skipped)}, {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
