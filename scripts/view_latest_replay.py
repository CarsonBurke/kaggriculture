#!/usr/bin/env python3
"""Play one of our agents and open the official engine replay.

Runs one PvP match on the real kaggle_environments engine, renders the bundled
Kaggriculture visualizer to a self-contained HTML file under
<run-dir>/replays/, and opens it in the browser. Inference is CPU-only so it
never contends with training on the GPU.

The candidate is the newest run's `latest.pt` by default, or any actor artifact
passed to `--artifact` (a BC clone, an exported submission actor, an older
checkpoint). A population checkpoint plays the member last richest against
v27 (`--agent` overrides). The opponent is a mirror of the candidate by
default; `--opponent` takes any spec the evaluation entry points accept, so
`--opponent v27` renders the same head-to-head the acceptance evaluations
score.

Sampled mode (the default) draws actions at the training temperature and shows
the behavior the training metrics measure; deterministic mode plays the argmax
policy — exactly what a Kaggle submission does.

"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.inference import CheckpointAgent, checkpoint_agent_count
from kaggriculture.opponents import normalize_opponent
from kaggriculture.policy import act_batch
from kaggriculture.production import PRODUCTION_EPISODE_STEPS, PRODUCTION_TEMPERATURE
from kaggriculture.provenance import repository_root
from kaggriculture.telemetry import read_jsonl_snapshot


def _snapshot_file(source: Path, destination: Path) -> None:
    """Snapshot an atomic artifact without copying on the common filesystem."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


SELF_OPPONENT = "self"


def resolve_latest_run(runs_root: Path) -> Path:
    """Pick the run whose latest.pt checkpoint was written most recently."""
    candidates = [
        directory
        for directory in runs_root.iterdir()
        if directory.is_dir() and (directory / "latest.pt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no run with a latest.pt checkpoint under {runs_root}")
    return max(candidates, key=lambda directory: (directory / "latest.pt").stat().st_mtime)


def replay_output_path(
    run_directory: Path,
    iteration: int,
    mode: str,
    seed: int,
    opponent_label: str,
    member: int | None = None,
) -> Path:
    stem = f"iteration-{iteration:06d}-{mode}-seed{seed}-vs-{opponent_label}"
    if member is not None:
        stem = f"{stem}-member{member}"
    return run_directory / "replays" / f"{stem}.html"


_V27_LABELS = frozenset({"public-v27", "v27"})


def select_replay_member(run_directory: Path, population: int) -> int:
    """The member last seen richest against v27, else richest in self-play.

    External eval is the only absolute reading; self-play money is the fallback
    when that journal has not landed yet. Ties keep the lowest index.
    """
    if population < 1:
        raise ValueError("population must be positive")
    external = run_directory / "metrics-external.jsonl"
    if external.is_file():
        latest_iteration: int | None = None
        scores: dict[int, float] = {}
        for record in read_jsonl_snapshot(external).records:
            if record.get("opponent") not in _V27_LABELS:
                continue
            iteration = record.get("iteration")
            if type(iteration) is not int:
                continue
            if latest_iteration is None or iteration > latest_iteration:
                latest_iteration = iteration
                scores = {}
            if iteration != latest_iteration:
                continue
            agent = record.get("agent")
            if agent is None:
                continue
            member = int(agent)
            if not 0 <= member < population:
                continue
            games = record.get("games")
            completed_games = record.get("completed_games")
            money_mean = record.get("money_mean")
            if (
                type(games) is not int
                or games < 1
                or completed_games != games
                or not isinstance(money_mean, (int, float))
                or isinstance(money_mean, bool)
                or not math.isfinite(money_mean)
            ):
                continue
            scores[member] = float(money_mean)
        if scores:
            return min(scores, key=lambda member: (-scores[member], member))
    metrics = run_directory / "metrics.jsonl"
    if metrics.is_file():
        lines = [line for line in metrics.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            record = json.loads(lines[-1])
            scored = []
            for member in range(population):
                key = f"agent{member}_money_mean"
                if key in record:
                    scored.append((float(record[key]), member))
            if scored:
                return min(scored, key=lambda item: (-item[0], item[1]))[1]
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="run directory; defaults to the run with the newest latest.pt under runs/",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="actor artifact or checkpoint to play; defaults to <run-dir>/latest.pt",
    )
    parser.add_argument(
        "--agent",
        type=int,
        default=None,
        help="population member to play; default is the latest v27 money leader",
    )
    parser.add_argument(
        "--opponent",
        default=SELF_OPPONENT,
        help=f"'{SELF_OPPONENT}' for a mirror match, or a built-in name, Python agent "
        "path, or 'v27' for the fixed public opponent",
    )
    parser.add_argument(
        "--seat",
        type=int,
        choices=(0, 1),
        default=0,
        help="seat our agent occupies when an external opponent plays",
    )
    parser.add_argument("--mode", choices=("sampled", "deterministic"), default="sampled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=PRODUCTION_EPISODE_STEPS)
    parser.add_argument("--temperature", type=float, default=PRODUCTION_TEMPERATURE)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--no-open", action="store_true", help="write the replay without opening a browser"
    )
    return parser.parse_args()


def play_match(
    agent: CheckpointAgent,
    *,
    mode: str,
    seed: int,
    episode_steps: int,
    temperature: float,
    opponent: str | None = None,
    candidate_seat: int = 0,
) -> Any:
    """Play one engine match; `opponent` None mirrors the candidate on both seats."""
    generators = [
        np.random.default_rng(sequence) for sequence in np.random.SeedSequence(seed).spawn(2)
    ]

    def seat(player: int):
        def act(observation: dict[str, Any]) -> dict[str, Any]:
            if mode == "deterministic":
                return agent(observation)
            return act_batch(
                agent.actor,
                [observation],
                deterministic=False,
                temperature=temperature,
                generator=generators[player],
            ).actions[0]

        return act

    from kaggle_environments import make

    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=False,
    )
    if opponent is None:
        environment.run([seat(0), seat(1)])
        return environment
    players: list[Any] = [opponent, opponent]
    players[candidate_seat] = seat(candidate_seat)
    environment.run(players)
    return environment


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    if args.episode_steps < 1:
        raise ValueError("episode steps must be positive")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if args.torch_threads < 1:
        raise ValueError("torch threads must be positive")
    if args.artifact is None:
        run_directory = (
            resolve_latest_run(repository_root() / "runs")
            if args.run_dir is None
            else args.run_dir.expanduser().resolve()
        )
        artifact = run_directory / "latest.pt"
        if not artifact.is_file():
            raise FileNotFoundError(f"run has no latest.pt checkpoint: {run_directory}")
    else:
        artifact = args.artifact.expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"artifact does not exist: {artifact}")
        run_directory = (
            artifact.parent if args.run_dir is None else args.run_dir.expanduser().resolve()
        )
    opponent_label, opponent = (
        (SELF_OPPONENT, None)
        if args.opponent == SELF_OPPONENT
        else normalize_opponent(args.opponent)
    )
    with tempfile.TemporaryDirectory(prefix="kaggriculture-replay-") as name:
        # Snapshot the artifact so a concurrent training save cannot race the load.
        snapshot = Path(name) / artifact.name
        _snapshot_file(artifact, snapshot)

        payload = torch.load(snapshot, map_location="cpu", weights_only=False)
        population = checkpoint_agent_count(payload)
        member = args.agent
        if member is None and population > 1:
            member = select_replay_member(run_directory, population)
        if member is not None and not 0 <= member < population:
            raise ValueError(f"agent {member} is outside 0..{population - 1}")
        agent = CheckpointAgent(
            snapshot, device="cpu", torch_threads=args.torch_threads, agent=member
        )
    iteration = int(agent.metadata.get("iteration", 0))
    environment = play_match(
        agent,
        mode=args.mode,
        seed=args.seed,
        episode_steps=args.episode_steps,
        temperature=args.temperature,
        opponent=opponent,
        candidate_seat=args.seat,
    )
    output = replay_output_path(
        run_directory, iteration, args.mode, args.seed, opponent_label, member
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(environment.render(mode="html"), encoding="utf-8")
    final = environment.steps[-1]
    print(
        json.dumps(
            {
                "event": "replay",
                "run_dir": str(run_directory),
                "artifact": str(artifact),
                "architecture": agent.metadata.get("architecture"),
                "agent": member,
                "opponent": opponent_label,
                "candidate_seat": args.seat,
                "iteration": iteration,
                "mode": args.mode,
                "seed": args.seed,
                "steps": len(environment.steps),
                "final": [{"reward": state.reward, "status": str(state.status)} for state in final],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
