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

`--perturbation-rate` makes recovery demonstrations (DART): the teacher's
seats sometimes execute a random legal deviation, and the archived label is
still the teacher's own action at every state, so the clone also sees the
off-route states a deviation leads to and how the teacher acts from them.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from kaggriculture.demonstrations import (
    DemonstrationError,
    perturb_action,
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
        "--deposit-all-products",
        action="store_true",
        help=(
            "label a teacher's partial product deposit as depositing everything held, "
            "the nearest factored action, instead of aborting. demand-advance4 needs "
            "it on about 1%% of steps; recorded in the manifest and counted per episode"
        ),
    )
    parser.add_argument(
        "--perturbation-rate",
        type=float,
        default=0.0,
        help=(
            "DART recovery demonstrations: per-step probability that a teacher seat "
            "executes a random legal deviation (`perturb_action`) instead of its own "
            "action. The archived label is always the teacher's own action at the "
            "state, so the clone learns how the teacher recovers from off-route "
            "states. Default 0: clean demonstrations"
        ),
    )
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        default=0,
        help="seeds the deviation draws together with each game's map seed and seat",
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
    deposit_all_products: bool = False,
    redundant_fertilize_as_pass: bool = False,
    recovery: RecoveryRecord | None = None,
) -> dict[str, np.ndarray | bytes]:
    """Project one recorded seat of a complete episode into training arrays.

    `steps[t+1][seat].action` is the action applied to `steps[t][seat]`'s
    observation, so a T-step episode yields T-1 demonstration pairs. With a
    `recovery` record the label is the teacher's own action instead, which
    differs from the executed one exactly at the perturbed steps. The factor
    arrays then hold the labels while `raw["actions"]` stays the executed game,
    so replaying it still reproduces the recorded states; the labels travel as
    `raw["teacher_actions"]` and a `perturbed` flag marks each step whose
    recorded transition is not the dynamics of its label. Recovery labels come
    from off-route states where a scripted teacher may fertilize an already
    fertilized tile, so they always project with `redundant_fertilize_as_pass`;
    hosted replays opt into it. Wherever it may apply, a per-step count
    travels as `relabeled_redundant_fertilize`, since the recorded transition
    spent a fertilizer its PASS label does not.
    """
    if len(environment_steps) != episode_steps:
        raise DemonstrationError(
            f"episode has {len(environment_steps)} steps; expected {episode_steps}"
        )
    if recovery is not None and not (
        len(recovery.labels) == len(recovery.perturbed) == episode_steps - 1
    ):
        raise DemonstrationError(
            f"seat {seat}: recovery record holds {len(recovery.labels)} labels and "
            f"{len(recovery.perturbed)} flags for {episode_steps - 1} steps"
        )
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    projections = []
    redundant_fertilize_as_pass = redundant_fertilize_as_pass or recovery is not None
    for step_index in range(episode_steps - 1):
        observation = _seat_observation(environment_steps, step_index, seat)
        action = environment_steps[step_index + 1][seat].action
        if not isinstance(action, dict):
            raise DemonstrationError(f"step {step_index} seat {seat}: no recorded action")
        label = action if recovery is None else recovery.label(step_index, action, seat)
        try:
            projected = project_demonstration(
                observation,
                label,
                deposit_all_products=deposit_all_products,
                redundant_fertilize_as_pass=redundant_fertilize_as_pass,
            )
            verify_round_trip(observation, label, projected)
        except DemonstrationError as error:
            raise DemonstrationError(f"step {step_index} seat {seat}: {error}") from error
        opponent_private = environment_steps[step_index][1 - seat].observation.get("private")
        observations.append({"observation": observation, "opponent_private": opponent_private})
        actions.append(action)
        labels.append(label)
        projections.append(projected)

    def stacked(name: str) -> np.ndarray:
        return np.stack([getattr(projection, name) for projection in projections])

    raw: dict[str, Any] = {
        "observations": observations,
        "actions": actions,
    }
    relabel_arrays = {}
    if recovery is not None:
        raw["teacher_actions"] = labels
        relabel_arrays["perturbed"] = np.asarray(recovery.perturbed, dtype=np.bool_)
    if redundant_fertilize_as_pass:
        relabel_arrays["relabeled_redundant_fertilize"] = stacked(
            "relabeled_redundant_fertilize"
        ).astype(np.int8)
    return {
        **relabel_arrays,
        "unit_actions": stacked("unit_actions"),
        "market_kinds": stacked("market_kinds"),
        "market_quantities": stacked("market_quantities"),
        "unit_masks": stacked("unit_masks"),
        "market_kind_masks": stacked("market_kind_masks"),
        "market_quantity_masks": stacked("market_quantity_masks"),
        "unit_active": stacked("unit_active"),
        "market_active": stacked("market_active"),
        "market_quantity_active": stacked("market_quantity_active"),
        "relabeled_partial_deposits": stacked("relabeled_partial_deposits").astype(np.int8),
        "raw_json_zlib": zlib.compress(
            json.dumps(raw, separators=(",", ":"), allow_nan=False).encode("utf-8"), level=6
        ),
    }


@dataclass
class RecoveryRecord:
    """A teacher seat's own actions and where a deviation replaced them."""

    labels: list[dict[str, Any]] = field(default_factory=list)
    perturbed: list[bool] = field(default_factory=list)

    def label(self, step: int, executed: dict[str, Any], seat: int) -> dict[str, Any]:
        """The teacher's action at `step`, checked against the executed one.

        Off the perturbed steps the two must be identical; a mismatch means the
        record is misaligned with the engine's steps, which would silently
        pair every later state with a neighbour's label.
        """
        if step >= len(self.labels):
            raise DemonstrationError(f"step {step} seat {seat}: no teacher action recorded")
        if (_json_form(executed) == self.labels[step]) == self.perturbed[step]:
            where = (
                "equals the teacher's at a perturbed"
                if self.perturbed[step]
                else "differs from the teacher's at an unperturbed"
            )
            raise DemonstrationError(
                f"step {step} seat {seat}: the executed action {where} step; "
                "recovery record is misaligned"
            )
        return self.labels[step]


def _json_form(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


class RecoveryTeacher:
    """A teacher seat that executes a random legal deviation with probability `rate`.

    The teacher is asked for its action at every step, so its own state keeps
    evolving exactly as when it plays, and that action is recorded as the
    label whether or not a deviation replaces it (DART, Laskey et al. 2017).
    The teacher is built and called exactly as `kaggle_environments` would
    (`build_agent`, then only as many arguments as its code takes), and its
    label is normalized through the action schema the engine applies, so off
    the perturbed steps label and executed action are the same object.

    Any exception here, the teacher's own included, would reach the engine as
    an agent error, and the seat would then play PASS to the end and still
    finish DONE. So the first one is kept in `error` for `_play_episode` to
    raise after the game, and this seat passes until then.
    """

    def __init__(
        self,
        runnable: str,
        environment: Any,
        rate: float,
        rng: np.random.Generator,
        *,
        deposit_all_products: bool = False,
    ) -> None:
        from kaggle_environments.agent import build_agent

        self.agent = build_agent(runnable, environment.agents, environment.name)[0]
        self.action_schema = environment.specification.action
        self.rate = rate
        self.rng = rng
        self.deposit_all_products = deposit_all_products
        self.record = RecoveryRecord()
        self.error: Exception | None = None

    def __call__(self, observation: Any, configuration: Any) -> dict[str, Any]:
        if self.error is not None:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        step = len(self.record.labels)
        try:
            return self._act(observation, configuration, step)
        except Exception as error:
            self.error = DemonstrationError(f"step {step}: {type(error).__name__}: {error}")
            self.error.__cause__ = error
            return {"farmer": ["PASS"], "hands": [], "market": []}

    def _act(self, observation: Any, configuration: Any, step: int) -> dict[str, Any]:
        from kaggle_environments.utils import process_schema

        if int(observation.get("step", step)) != step:
            raise DemonstrationError(f"observation step {observation.get('step')} is not {step}")
        arguments = (observation, configuration)
        code = getattr(self.agent, "__code__", None)
        if code is not None:
            arguments = arguments[: code.co_argcount]
        error, label = process_schema(self.action_schema, self.agent(*arguments))
        if error:
            raise DemonstrationError(f"the teacher's action is invalid: {error}")
        label = _json_form(label)
        executed = None
        if self.rng.random() < self.rate:
            state = _json_form(dict(observation))
            state.setdefault("step", step)
            executed = perturb_action(
                state,
                label,
                self.rng,
                deposit_all_products=self.deposit_all_products,
                redundant_fertilize_as_pass=True,
            )
        self.record.labels.append(label)
        self.record.perturbed.append(executed is not None)
        return label if executed is None else executed


def _play_episode(
    teacher: str,
    opponent: str,
    seed: int,
    episode_steps: int,
    *,
    recovery_seats: tuple[int, ...] = (),
    perturbation_rate: float = 0.0,
    perturbation_seed: int = 0,
    deposit_all_products: bool = False,
) -> tuple[list[Any], dict[int, RecoveryRecord]]:
    """Play one official game; the `recovery_seats` play as `RecoveryTeacher`s."""
    from kaggle_environments import make

    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=False,
    )
    agents: list[Any] = [teacher, opponent]
    recoveries: dict[int, RecoveryTeacher] = {}
    for seat in recovery_seats:
        recoveries[seat] = RecoveryTeacher(
            agents[seat],
            environment,
            perturbation_rate,
            np.random.default_rng((perturbation_seed, seed, seat)),
            deposit_all_products=deposit_all_products,
        )
        agents[seat] = recoveries[seat]
    environment.run(agents)
    for seat, agent in recoveries.items():
        if agent.error is not None:
            raise DemonstrationError(f"seed {seed} seat {seat}: {agent.error}") from agent.error
        if len(agent.record.labels) != episode_steps - 1:
            raise DemonstrationError(
                f"seed {seed} seat {seat}: the teacher acted {len(agent.record.labels)} "
                f"times in {episode_steps - 1} steps"
            )
    if not environment.done:
        raise RuntimeError(f"seed {seed}: environment did not finish")
    for seat in (0, 1):
        status = str(environment.steps[-1][seat].status)
        if status != "DONE":
            raise RuntimeError(f"seed {seed}: seat {seat} ended with status {status}")
    return environment.steps, {seat: agent.record for seat, agent in recoveries.items()}


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
    deposit_all_products: bool = False,
    perturbation_rate: float = 0.0,
    perturbation_seed: int = 0,
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
        steps, recoveries = _play_episode(
            left,
            right,
            seed,
            episode_steps,
            recovery_seats=seats if perturbation_rate > 0 else (),
            perturbation_rate=perturbation_rate,
            perturbation_seed=perturbation_seed,
            deposit_all_products=deposit_all_products,
        )
        for seat in seats:
            arrays = extract_episode(
                steps,
                seat,
                episode_steps=episode_steps,
                deposit_all_products=deposit_all_products,
                recovery=recoveries.get(seat),
            )
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
        relabeled = (
            int(archive["relabeled_partial_deposits"].sum())
            if "relabeled_partial_deposits" in archive.files
            else 0
        )
        # Recovery archives also count their deviations and relabels.
        recovery = {
            name: int(archive[key].sum())
            for name, key in (
                ("perturbed_steps", "perturbed"),
                ("relabeled_redundant_fertilize", "relabeled_redundant_fertilize"),
            )
            if key in archive.files
        }
    farms = raw["observations"][-1]["observation"]["farms"]
    return {
        "file": path.name,
        "seed": seed,
        "seat": seat,
        "steps": episode_steps - 1,
        "teacher_money": float(farms[seat]["money"]),
        "opponent_money": float(farms[1 - seat]["money"]),
        "relabeled_partial_deposits": relabeled,
        "sha256": file_sha256(path),
        **recovery,
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
    deposit_all_products: bool = False,
    perturbation_rate: float = 0.0,
    perturbation_seed: int = 0,
) -> dict[str, Any]:
    configuration = {
        "format_version": DATASET_FORMAT_VERSION,
        "teacher": {"label": teacher_label, "sha256": _agent_digest(teacher)},
        "opponent": {"label": opponent_label, "sha256": _agent_digest(opponent)},
        "episode_steps": episode_steps,
        "seed_start": seed_start,
        "episode_count": episode_count,
        "extractor_source_identity": source_identity(),
    }
    # Stated only when used, so strictly projected datasets keep the manifest
    # they were written with and remain resumable.
    if deposit_all_products:
        configuration["relabels"] = ["deposit_all_products"]
    if perturbation_rate > 0:
        configuration["perturbation"] = {"rate": perturbation_rate, "seed": perturbation_seed}
        configuration["relabels"] = [
            *configuration.get("relabels", []),
            "redundant_fertilize_as_pass",
        ]
    return configuration


def _load_resumable_records(
    output_dir: Path,
    staging_dir: Path,
    expected_configuration: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("--resume requires a previously committed manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Every committed key, not only the ones this run configures: a key this run
    # would not write (a perturbation, a relabel) still describes the archives.
    actual_configuration = {
        key: value for key, value in manifest.items() if key not in {"episodes", "command"}
    }
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
    if not 0.0 <= args.perturbation_rate < 1.0:
        raise ValueError("--perturbation-rate must lie in [0, 1)")
    if args.perturbation_seed < 0:
        raise ValueError("--perturbation-seed must be non-negative")
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
        deposit_all_products=args.deposit_all_products,
        perturbation_rate=args.perturbation_rate,
        perturbation_seed=args.perturbation_seed,
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
                        args.deposit_all_products,
                        args.perturbation_rate,
                        args.perturbation_seed,
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
