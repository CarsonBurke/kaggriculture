#!/usr/bin/env python3
"""Extract projected demonstration episodes from official public-v27 games.

Runs complete kaggle_environments episodes with the public v27 agent in the
recorded seat(s), projects every recorded engine action through the exact
sequential legality ledger (`kaggriculture.demonstrations`), verifies the
factored round trip against the recorded dict, and stores one compressed
archive per episode-seat: raw observations (retokenizable for any future
architecture), factored targets, teacher-forced masks, and active flags.

Any representability gap or mask divergence aborts extraction with the
offending step — silent clamping would corrupt the dataset. CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.demonstrations import (
    DemonstrationError,
    project_demonstration,
    verify_round_trip,
)
from kaggriculture.opponents import BUILTIN_OPPONENTS, normalize_opponent
from kaggriculture.provenance import file_sha256, source_identity

DATASET_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128, help="environment seeds to play")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--teacher", default="public-v27", help="demonstrating agent (spec for opponents registry)"
    )
    parser.add_argument(
        "--opponent",
        default="public-v27",
        help="other seat; when it equals the teacher, both seats are recorded",
    )
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "parallel episode extractors; seeds are independent games, so this "
            "divides wall time almost exactly and changes nothing in the output"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse archives already in --output-dir instead of replaying their "
            "seeds. Each episode costs about 3 CPU-seconds and the manifest is "
            "only written at the end, so an interrupted run otherwise strands "
            "every episode it completed"
        ),
    )
    return parser.parse_args()


def _seat_observation(steps: list[Any], step_index: int, seat: int) -> dict[str, Any]:
    observation = dict(steps[step_index][seat].observation)
    # The engine strips `step` from the non-first seat's stored schema; it is
    # positional, so restore it from the walk index.
    observation.setdefault("step", step_index)
    return observation


def extract_episode(
    environment_steps: list[Any],
    seat: int,
    *,
    episode_steps: int,
) -> dict[str, np.ndarray | bytes]:
    """Project one recorded seat of a complete episode into training arrays.

    `steps[t+1][seat].action` is the action applied to `steps[t][seat]`'s
    observation, so a T-step episode yields T-1 demonstration pairs.
    """
    if len(environment_steps) != episode_steps:
        raise DemonstrationError(
            f"episode has {len(environment_steps)} steps; expected {episode_steps}"
        )
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    projections = []
    for step_index in range(episode_steps - 1):
        observation = _seat_observation(environment_steps, step_index, seat)
        action = environment_steps[step_index + 1][seat].action
        if not isinstance(action, dict):
            raise DemonstrationError(f"step {step_index} seat {seat}: no recorded action")
        try:
            projected = project_demonstration(observation, action)
            verify_round_trip(observation, action, projected)
        except DemonstrationError as error:
            raise DemonstrationError(f"step {step_index} seat {seat}: {error}") from error
        opponent_private = environment_steps[step_index][1 - seat].observation.get("private")
        observations.append({"observation": observation, "opponent_private": opponent_private})
        actions.append(action)
        projections.append(projected)

    def stacked(name: str) -> np.ndarray:
        return np.stack([getattr(projection, name) for projection in projections])

    raw = {
        "observations": observations,
        "actions": actions,
    }
    return {
        "unit_actions": stacked("unit_actions"),
        "market_kinds": stacked("market_kinds"),
        "market_quantities": stacked("market_quantities"),
        "unit_masks": stacked("unit_masks"),
        "market_kind_masks": stacked("market_kind_masks"),
        "market_quantity_masks": stacked("market_quantity_masks"),
        "unit_active": stacked("unit_active"),
        "market_active": stacked("market_active"),
        "market_quantity_active": stacked("market_quantity_active"),
        "raw_json_zlib": zlib.compress(
            json.dumps(raw, separators=(",", ":"), allow_nan=False).encode("utf-8"), level=6
        ),
    }


def _play_episode(teacher: str, opponent: str, seed: int, episode_steps: int) -> list[Any]:
    from kaggle_environments import make

    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=False,
    )
    environment.run([teacher, opponent])
    if not environment.done:
        raise RuntimeError(f"seed {seed}: environment did not finish")
    for seat in (0, 1):
        status = str(environment.steps[-1][seat].status)
        if status != "DONE":
            raise RuntimeError(f"seed {seed}: seat {seat} ended with status {status}")
    return environment.steps


def _agent_digest(runnable: str) -> str | None:
    return None if runnable in BUILTIN_OPPONENTS else file_sha256(Path(runnable))


def _extract_seed(
    teacher: str,
    opponent: str,
    seed: int,
    episode_steps: int,
    seats: tuple[int, ...],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Play one seed and archive every recorded seat of it.

    Seeds are wholly independent games, so this is the unit of parallelism.
    Each seat writes a uniquely named archive, so workers never contend.

    Each record is read back out of the archive that was just written, so a
    resumed seed and a freshly played one produce byte-identical provenance and
    every written archive is proven to round-trip before the run can succeed.
    """
    steps = _play_episode(teacher, opponent, seed, episode_steps)
    records = []
    for seat in seats:
        arrays = extract_episode(steps, seat, episode_steps=episode_steps)
        path = output_dir / f"episode-{seed:08d}-seat{seat}.npz"
        np.savez_compressed(path, **arrays)
        records.append(_archived_record(path, seed, seat, episode_steps))
    return records


def _archived_record(path: Path, seed: int, seat: int, episode_steps: int) -> dict[str, Any]:
    """Derive one manifest record from an archive on disk.

    Extraction is a long CPU job on a thermally shared machine, so it gets
    interrupted, and an interrupted run used to strand its completed archives:
    the manifest is written once at the end, and without it `train_bc` cannot read
    the directory at all. One cancelled 512-episode run left 439 valid archives
    with no way to be used, which is what `--resume` recovers.

    Money is the bank in the last archived observation, which is a step short of
    the episode's terminal reward: the terminal step carries no action, so it is
    not archived, and its final day of income is not recoverable from the file. On
    a measured seed the gap was 138,754 against a 138,973 reward. Deriving both
    halves of a resumed directory the same way is worth more than 0.16% of a field
    no consumer reads -- it is dataset diagnostics, not a training signal.
    """
    with np.load(path, allow_pickle=False) as archive:
        raw = json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()).decode("utf-8"))
    farms = raw["observations"][-1]["observation"]["farms"]
    return {
        "file": path.name,
        "seed": seed,
        "seat": seat,
        "steps": episode_steps - 1,
        "teacher_money": float(farms[seat]["money"]),
        "opponent_money": float(farms[1 - seat]["money"]),
    }


def collect_extractions(pending: dict[Future, int], started: float) -> list[dict[str, Any]]:
    """Gather every seed's records, aborting the run on the first failure.

    A dataset that silently omits the seeds the ledger could not represent is
    a biased dataset, so a projection error has to propagate. Cancelling the
    queued futures here, in this thread, is what makes that abort prompt:
    `Executor.shutdown(cancel_futures=True)` only asks the pool's manager
    thread to cancel them later, and the shutdown that runs while the
    exception unwinds resets the request before the manager ever acts, so
    every remaining seed would still play out in full before the offending
    step became visible.
    """
    episodes: list[dict[str, Any]] = []
    try:
        for future in as_completed(pending):
            episodes.extend(future.result())
            elapsed = time.perf_counter() - started
            print(
                f"seed {pending[future]}: extracted "
                f"({elapsed:.1f}s elapsed, {len(episodes)} episode-seats)",
                flush=True,
            )
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    return episodes


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.episode_steps != 720:
        raise ValueError("extraction needs at least one episode at the competition horizon")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    teacher_label, teacher = normalize_opponent(args.teacher)
    opponent_label, opponent = normalize_opponent(args.opponent)
    mirrored = teacher == opponent
    seats = (0, 1) if mirrored else (0,)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    recovered: list[dict[str, Any]] = []
    if args.resume:
        # A seed counts as done only when every seat it owes exists, so a seed
        # interrupted between its two mirrored seats is replayed rather than
        # half-recorded.
        outstanding = []
        for seed in seeds:
            paths = [args.output_dir / f"episode-{seed:08d}-seat{seat}.npz" for seat in seats]
            if all(path.is_file() for path in paths):
                recovered.extend(
                    _archived_record(path, seed, seat, args.episode_steps)
                    for path, seat in zip(paths, seats, strict=True)
                )
            else:
                outstanding.append(seed)
        print(
            f"resuming: {len(recovered)} episode-seats already archived, "
            f"{len(outstanding)} seeds to play",
            flush=True,
        )
        seeds = outstanding
    started = time.perf_counter()
    episodes = []
    if seeds:
        pool = ProcessPoolExecutor(max_workers=min(args.workers, len(seeds)))
        try:
            pending = {
                pool.submit(
                    _extract_seed,
                    teacher,
                    opponent,
                    seed,
                    args.episode_steps,
                    seats,
                    args.output_dir,
                ): seed
                for seed in seeds
            }
            episodes = collect_extractions(pending, started)
        finally:
            # After a cancellation this waits only for the seeds already in
            # flight, which is the shortest correct abort: their worker processes
            # own open archive handles.
            pool.shutdown(wait=True)

    episodes.extend(recovered)
    # Completion order is nondeterministic under parallelism; the manifest is
    # provenance and is hashed, so it is written in seed order regardless.
    episodes.sort(key=lambda record: (record["seed"], record["seat"]))

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "teacher": {"label": teacher_label, "sha256": _agent_digest(teacher)},
        "opponent": {"label": opponent_label, "sha256": _agent_digest(opponent)},
        "episode_steps": args.episode_steps,
        "seed_start": args.seed_start,
        "episodes": episodes,
        "extractor_source_identity": source_identity(),
        "command": sys.argv,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(episodes)} episode-seats and {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
