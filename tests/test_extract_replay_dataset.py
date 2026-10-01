from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from kaggle_environments import make

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hosted_game():
    """A locally played game in the shape of a hosted replay, and its engine steps."""
    environment = make("kaggriculture", configuration={"seed": 3}, debug=False)
    environment.run(["starter", "starter"])
    replay = environment.toJSON()
    # Hosted replays carry the map seed and team names in `info`, and the
    # configuration's own seed is unset.
    replay["info"] = {"seed": 3, "EpisodeId": 900, "TeamNames": ["alpha", "beta"]}
    replay["configuration"]["seed"] = None
    return replay, environment.steps


def _archive(path: Path, replays: dict[int, dict], scores: dict[int, float]) -> Path:
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    writer.writerow(["episode_id", "create_time", "avg_score", "min_score"])
    for episode, score in scores.items():
        writer.writerow([episode, "2026-09-28T00:00:00", score + 10, score])
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("manifest.csv", manifest.getvalue())
        for episode, replay in replays.items():
            bundle.writestr(f"{episode}.json", json.dumps(replay))
    return path


def test_replayed_seats_match_live_extraction_and_filter_by_rating(
    tmp_path, monkeypatch, hosted_game
) -> None:
    replay, steps = hosted_game
    tampered = json.loads(json.dumps(replay))
    tampered["info"]["EpisodeId"] = 902
    tampered["rewards"] = [reward + 1 for reward in tampered["rewards"]]
    archive = _archive(
        tmp_path / "day.zip",
        {900: replay, 901: replay, 902: tampered},
        {900: 2700.0, 901: 2500.0, 902: 2800.0},
    )
    output = tmp_path / "corpus"
    replayer = _load("extract_replay_dataset")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_replay_dataset.py",
            "--archives",
            str(archive),
            "--output-dir",
            str(output),
            "--key-start",
            "3000000",
            "--workers",
            "1",
        ],
    )

    replayer.main()

    manifest = json.loads((output / "manifest.json").read_text())
    # 901 is below the rating floor; 902's hosted rewards do not reproduce.
    assert manifest["below_min_score"] == 1
    assert manifest["episode_count"] == 2
    assert [skip.startswith("replayed rewards") for skip in manifest["skipped"]] == [True]
    assert [(record["seed"], record["seat"]) for record in manifest["episodes"]] == [
        (3000000, 0),
        (3000000, 1),
    ]
    extractor = _load("extract_bc_dataset")
    for record in manifest["episodes"]:
        seat = record["seat"]
        assert record["map_seed"] == 3
        assert record["hosted_episode"] == 900
        assert record["team"] == ["alpha", "beta"][seat]
        assert record["hosted_reward"] == replay["rewards"][seat]
        live = extractor.extract_episode(
            steps,
            seat,
            episode_steps=720,
            deposit_all_products=True,
            redundant_fertilize_as_pass=True,
        )
        with np.load(output / record["file"]) as archived:
            assert sorted(archived.files) == sorted(live)
            for name, value in live.items():
                np.testing.assert_array_equal(archived[name], value)


def test_evaluation_map_seeds_are_never_trained_on(hosted_game) -> None:
    from kaggriculture.evaluation import SEED_DOMAINS

    replayer = _load("extract_replay_dataset")
    replay, _ = hosted_game
    for domain in replayer.EVALUATION_DOMAINS:
        reserved = {**replay, "info": {**replay["info"], "seed": SEED_DOMAINS[domain][0]}}
        assert replayer._screen(reserved) == "map seed reserved for evaluation"
    assert replayer._screen(replay) is None
