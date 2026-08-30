#!/usr/bin/env python3
"""Extract projected demonstration episodes from official public-v16 games.

Runs complete kaggle_environments episodes with the teacher in both seats.
Against a distinct opponent that means two games per seed (teacher first,
then seats swapped). Against a copy of itself one game already has the
teacher on both sides. Every recorded engine action is projected through
the exact sequential legality ledger (`kaggriculture.demonstrations`),
verified against the recorded dict, and stored as one compressed archive
per episode-seat: raw observations (retokenizable for any future
architecture), factored targets, teacher-forced masks, and active flags.

Any representability gap or mask divergence aborts extraction with the
offending step — silent clamping would corrupt the dataset. CPU-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zlib
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
        "--teacher", default="public-v16", help="demonstrating agent (spec for opponents registry)"
    )
    parser.add_argument(
        "--opponent",
        default="public-v16",
        help="other seat; when it equals the teacher, both seats are recorded",
    )
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "parallel episode extractors; seeds are independent games, so this "
            "divides wall time almost exactly and changes nothing in the output. "
            "Each worker is one host thread; 8 used to saturate a shared box"
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


def _pin_extract_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def teacher_jobs(teacher: str, opponent: str) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Games that put the teacher in a recorded seat.

    A self-play seed is one game with both seats. Against anyone else the
    teacher has to sit twice: once on the left, once on the right, same
    map seed. Recording only seat 0 is how a clone never saw the other
    farm's private state as its own.
    """
    if teacher == opponent:
        return ((teacher, opponent, (0, 1)),)
    return ((teacher, opponent, (0,)), (opponent, teacher, (1,)))


def _extract_seed(
    teacher: str,
    opponent: str,
    seed: int,
    episode_steps: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Play one seed and archive every teacher seat of it.

    Seeds are wholly independent games, so this is the unit of parallelism.
    Each seat writes a uniquely named archive, so workers never contend.

    Each record is read back out of the archive that was just written, so a
    resumed seed and a freshly played one produce byte-identical provenance and
    every written archive is proven to round-trip before the run can succeed.
    """
    records = []
    for left, right, seats in teacher_jobs(teacher, opponent):
        steps = _play_episode(left, right, seed, episode_steps)
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
        "sha256": file_sha256(path),
    }


def _manifest_configuration(
    *,
    teacher_label: str,
    teacher: str,
    opponent_label: str,
    opponent: str,
    episode_steps: int,
    seed_start: int,
    episode_count: int,
) -> dict[str, Any]:
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "teacher": {"label": teacher_label, "sha256": _agent_digest(teacher)},
        "opponent": {"label": opponent_label, "sha256": _agent_digest(opponent)},
        "episode_steps": episode_steps,
        "seed_start": seed_start,
        "episode_count": episode_count,
        "extractor_source_identity": source_identity(),
    }


def _load_resumable_records(
    output_dir: Path,
    staging_dir: Path,
    expected_configuration: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("--resume requires a previously committed manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_configuration = {key: manifest.get(key) for key in expected_configuration}
    if actual_configuration != expected_configuration:
        raise ValueError("--resume extraction provenance does not match the committed dataset")

    records = list(manifest.get("episodes") or [])
    expected_pairs = {
        (seed, seat)
        for seed in range(
            int(expected_configuration["seed_start"]),
            int(expected_configuration["seed_start"])
            + int(expected_configuration["episode_count"]),
        )
        for seat in (0, 1)
    }
    actual_pairs = {(int(record["seed"]), int(record["seat"])) for record in records}
    if actual_pairs != expected_pairs or len(records) != len(expected_pairs):
        raise ValueError("--resume manifest does not contain the configured seed range")

    recovered: list[dict[str, Any]] = []
    for record in records:
        source = output_dir / str(record["file"])
        expected_digest = record.get("sha256")
        if not source.is_file() or not isinstance(expected_digest, str):
            raise ValueError(f"--resume archive provenance is incomplete: {source}")
        if file_sha256(source) != expected_digest:
            raise ValueError(f"--resume archive digest mismatch: {source}")
        destination = staging_dir / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        recovered.append({**record, "file": destination.name})
    return recovered


def _write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def commit_dataset_generation(
    staging_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    """Publish complete archives before atomically switching the manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generations = output_dir / ".generations"
    generations.mkdir(exist_ok=True)
    generation_dir = generations / staging_dir.name
    committed = {
        **manifest,
        "episodes": [
            {
                **record,
                "file": str(Path(".generations") / generation_dir.name / Path(record["file"]).name),
            }
            for record in manifest["episodes"]
        ],
    }
    for record in manifest["episodes"]:
        archive = staging_dir / Path(record["file"]).name
        if not archive.is_file() or file_sha256(archive) != record.get("sha256"):
            raise ValueError(f"staged archive failed verification: {archive}")

    os.replace(staging_dir, generation_dir)
    manifest_path = output_dir / "manifest.json"
    try:
        _write_manifest_atomic(manifest_path, committed)
    except BaseException:
        shutil.rmtree(generation_dir)
        raise
    return manifest_path


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
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    configuration = _manifest_configuration(
        teacher_label=teacher_label,
        teacher=teacher,
        opponent_label=opponent_label,
        opponent=opponent,
        episode_steps=args.episode_steps,
        seed_start=args.seed_start,
        episode_count=args.episodes,
    )

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    ) as temporary:
        staging_dir = Path(temporary)
        seeds = list(range(args.seed_start, args.seed_start + args.episodes))
        recovered: list[dict[str, Any]] = []
        if args.resume:
            recovered = _load_resumable_records(output_dir, staging_dir, configuration)
            seeds = []
            print(
                f"resuming: {len(recovered)} episode-seats already archived, 0 seeds to play",
                flush=True,
            )

        started = time.perf_counter()
        episodes = []
        if seeds:
            pool = ProcessPoolExecutor(
                max_workers=min(args.workers, len(seeds)),
                initializer=_pin_extract_worker,
            )
            try:
                pending = {
                    pool.submit(
                        _extract_seed,
                        teacher,
                        opponent,
                        seed,
                        args.episode_steps,
                        staging_dir,
                    ): seed
                    for seed in seeds
                }
                episodes = collect_extractions(pending, started)
            finally:
                pool.shutdown(wait=True)

        episodes.extend(recovered)
        episodes.sort(key=lambda record: (record["seed"], record["seat"]))
        manifest = {
            **configuration,
            "episodes": episodes,
            "command": sys.argv,
        }
        manifest_path = commit_dataset_generation(staging_dir, output_dir, manifest)
    print(f"wrote {len(episodes)} episode-seats and {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
