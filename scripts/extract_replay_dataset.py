#!/usr/bin/env python3
"""Extract demonstration episodes from hosted leaderboard replays.

Kaggle publishes each day's leaderboard episodes (`kaggle/kaggriculture-episodes-
YYYY-MM-DD`: one replay JSON per episode, between agents rated near the top of
the board). A replay's stored observations are not what the agent saw: the
engine strips shared fields from the second seat's record. So this replays both
seats' recorded actions through the local official engine on the episode's own
map seed, admits the episode only when every seat's terminal reward reproduces
the hosted one exactly, and then archives each seat through the same projection
and verification as live extraction (`extract_bc_dataset.extract_episode`).

A seat whose play the factored action space cannot represent is skipped, not
aborted: unlike a teacher we choose, a leaderboard agent is not ours to fix, and
the manifest counts every skip and its reason. Two engine-legal habits the
factored space has no action for are relabelled to their nearest action, as
for demand-advance4 and the recovery corpora: a partial product deposit as
depositing everything held, and a FERTILIZE of a tile already fertilized (a
fertilizer spent for nothing) as PASS. Both mark their steps so the world-model
terms skip the transition out of them. Unknown unit opcodes such as NOOP are
PASS already, since the engine ignores them.

Each daily dump ships a `manifest.csv` whose `min_score` is the lower of the
two agents' ratings when the episode was played; `--min-score` keeps episodes
between two agents rated at least that.

Hosted map seeds are arbitrary 32-bit integers, while `train_bc` keys its
holdout split on a seed in the reserved BC domain. Each admitted episode is
therefore keyed `--key-start` plus its rank by episode id, and its map seed and
episode id travel beside the key. Episodes on a map seed inside a reserved
evaluation domain are dropped, so no evaluation map is ever trained on.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_bc_dataset import (
    DATASET_FORMAT_VERSION,
    _archived_record,
    _pin_extract_worker,
    commit_dataset_generation,
    extract_episode,
)

from kaggriculture.demonstrations import DemonstrationError
from kaggriculture.evaluation import SEED_DOMAINS
from kaggriculture.provenance import file_sha256, source_identity

EPISODE_STEPS = 720
EVALUATION_DOMAINS = ("development", "screening", "finalist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives", type=Path, nargs="+", required=True, help="daily episode dataset zips"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--key-start",
        type=int,
        required=True,
        help="first BC-domain key; disjoint corpora need disjoint key ranges",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=2600.0,
        help="keep episodes whose lower-rated agent was rated at least this",
    )
    parser.add_argument("--limit", type=int, default=None, help="episodes to replay, for probes")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def replay_episode(replay: dict[str, Any]) -> Any:
    """Step the official engine through both seats' recorded actions.

    `Environment.run` would ask an agent for each seat's action, deep-copying
    the seat's observation every step, which is most of a replay's cost.
    Stepping with the recorded actions is the same game: `run` passes each
    ACTIVE seat's action and None for any other seat, and an agent that
    answers at once spends no overage time, so its logs change no state.
    """
    from kaggle_environments import make

    configuration = {**replay["configuration"], "seed": replay["info"]["seed"]}
    environment = make("kaggriculture", configuration=configuration, debug=False)
    environment.reset(2)
    for recorded in replay["steps"][1:]:
        if environment.done:
            break
        environment.step(
            [
                seat["action"] if state.status == "ACTIVE" else None
                for seat, state in zip(recorded, environment.state, strict=True)
            ]
        )
    return environment


def _reserved_for_evaluation(seed: int) -> bool:
    return any(
        SEED_DOMAINS[domain][0] <= seed < SEED_DOMAINS[domain][1] for domain in EVALUATION_DOMAINS
    )


def _screen(replay: dict[str, Any]) -> str | None:
    """Why an episode cannot be replayed, or None when it can."""
    import kaggle_environments

    if replay.get("module_version") != kaggle_environments.__version__:
        return f"engine {replay.get('module_version')} is not {kaggle_environments.__version__}"
    if replay.get("statuses") != ["DONE", "DONE"]:
        return "an agent did not finish"
    if len(replay["steps"]) != EPISODE_STEPS:
        return f"{len(replay['steps'])} steps"
    seed = replay["info"].get("seed")
    if type(seed) is not int:
        return "no map seed"
    if _reserved_for_evaluation(seed):
        return "map seed reserved for evaluation"
    return None


def _extract_replay(
    archive: str, member: str, key: int, output_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replay one episode; return its archived seats and each skipped seat's reason."""
    with zipfile.ZipFile(archive) as bundle:
        replay = json.loads(bundle.read(member))
    reason = _screen(replay)
    if reason is not None:
        return [], [reason]
    seed = replay["info"]["seed"]
    environment = replay_episode(replay)
    if len(environment.steps) != EPISODE_STEPS:
        return [], [f"replay ended after {len(environment.steps)} steps"]
    rewards = [state.reward for state in environment.steps[-1]]
    if rewards != replay["rewards"]:
        return [], [f"replayed rewards {rewards} differ from hosted {replay['rewards']}"]
    records, skipped = [], []
    teams = replay["info"]["TeamNames"]
    for seat in (0, 1):
        try:
            arrays = extract_episode(
                environment.steps,
                seat,
                episode_steps=EPISODE_STEPS,
                deposit_all_products=True,
                redundant_fertilize_as_pass=True,
            )
        except DemonstrationError as error:
            skipped.append(f"unrepresentable: {str(error).split(':', 2)[-1].strip()[:80]}")
            continue
        path = output_dir / f"episode-{key:08d}-seat{seat}.npz"
        np.savez_compressed(path, **arrays)
        records.append(
            {
                **_archived_record(path, key, seat, EPISODE_STEPS),
                "hosted_episode": replay["info"]["EpisodeId"],
                "map_seed": seed,
                "team": teams[seat],
                "opponent_team": teams[1 - seat],
                "hosted_reward": replay["rewards"][seat],
            }
        )
    return records, skipped


def _rated_members(archive: Path, min_score: float) -> tuple[list[str], int]:
    """The archive's replays between agents rated at least `min_score`, and how many were not."""
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open("manifest.csv") as handle:
            rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
        replays = {name for name in bundle.namelist() if name.endswith(".json")}
    rated = {f"{row['episode_id']}.json" for row in rows if float(row["min_score"]) >= min_score}
    if missing := rated - replays:
        raise ValueError(f"{archive}: manifest lists absent replays {sorted(missing)[:3]}")
    return sorted(rated), len(replays) - len(rated)


def main() -> None:
    args = parse_args()
    members = []
    below = 0
    for archive in args.archives:
        rated, unrated = _rated_members(archive, args.min_score)
        members.extend((str(archive.resolve()), name) for name in rated)
        below += unrated
    # Episode ids are chronological, so keys (and train_bc's highest-key holdout)
    # follow time: the holdout is the latest play.
    members.sort(key=lambda item: int(Path(item[1]).stem))
    members = members[: args.limit]
    if not members:
        raise ValueError("no replays in the given archives")
    keys = range(args.key_start, args.key_start + len(members))
    if not SEED_DOMAINS["bc"][0] <= keys.start < keys.stop <= SEED_DOMAINS["bc"][1]:
        raise ValueError("replay keys must lie in the BC seed domain")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    episodes: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ) as temporary:
        staging_dir = Path(temporary)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_pin_extract_worker) as pool:
            pending = {
                pool.submit(_extract_replay, archive, member, key, staging_dir): member
                for (archive, member), key in zip(members, keys, strict=True)
            }
            for done, future in enumerate(as_completed(pending), start=1):
                records, reasons = future.result()
                episodes.extend(records)
                skipped.update(reasons)
                if done % 25 == 0 or done == len(pending):
                    print(
                        f"{done}/{len(pending)} replays, {len(episodes)} seats archived, "
                        f"{sum(skipped.values())} skipped "
                        f"({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )
        episodes.sort(key=lambda record: (record["seed"], record["seat"]))
        label = {"label": "leaderboard-replay", "sha256": None}
        manifest = {
            "format_version": DATASET_FORMAT_VERSION,
            "teacher": label,
            "opponent": label,
            "episode_steps": EPISODE_STEPS,
            "seed_start": args.key_start,
            "episode_count": len(members),
            "relabels": ["deposit_all_products", "redundant_fertilize_as_pass"],
            "min_score": args.min_score,
            "below_min_score": below,
            "replay_archives": [
                {"name": Path(archive).name, "sha256": file_sha256(Path(archive))}
                for archive in sorted({archive for archive, _ in members})
            ],
            "skipped": dict(skipped.most_common()),
            "extractor_source_identity": source_identity(),
            "episodes": episodes,
            "command": sys.argv,
        }
        manifest_path = commit_dataset_generation(staging_dir, output_dir, manifest)
    print(f"wrote {len(episodes)} episode-seats and {manifest_path}", flush=True)
    for reason, count in skipped.most_common():
        print(f"  skipped {count}: {reason}", flush=True)


if __name__ == "__main__":
    main()
