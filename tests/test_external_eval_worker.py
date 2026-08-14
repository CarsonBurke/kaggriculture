from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kaggriculture.opponents import normalize_opponent


def _script():
    path = Path(__file__).parents[1] / "scripts" / "external_eval_worker.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_external_eval_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Python 3.13 dataclass creation requires the defining module in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builtin_opponents_pass_through_and_missing_files_fail(tmp_path: Path) -> None:
    assert normalize_opponent("starter") == ("starter", "starter")

    agent = tmp_path / "agent.py"
    agent.write_text("def act(observation):\n    return {}\n")
    label, runnable = normalize_opponent(str(agent))
    assert label == "agent.py"
    assert runnable == str(agent)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        normalize_opponent(str(tmp_path / "missing.py"))


def test_game_outcome_scores_wins_ties_and_losses() -> None:
    module = _script()
    win = module.GameOutcome(0, 0, 100.0, 50.0, None)
    tie = module.GameOutcome(0, 1, 70.0, 70.0, None)
    loss = module.GameOutcome(1, 0, 10.0, 20.0, None)
    failed = module.GameOutcome(1, 1, None, None, "seat 0 status is ERROR")

    assert (win.score, tie.score, loss.score) == (1.0, 0.5, 0.0)
    assert win.complete and not failed.complete


def test_evaluate_opponent_summarizes_only_completed_games(monkeypatch, tmp_path: Path) -> None:
    module = _script()
    outcomes = iter(
        [
            module.GameOutcome(0, 0, 100.0, 50.0, None),
            module.GameOutcome(0, 1, 60.0, 60.0, None),
            module.GameOutcome(1, 0, None, None, "environment did not reach DONE"),
            module.GameOutcome(1, 1, 30.0, 40.0, None),
        ]
    )
    monkeypatch.setattr(module, "_play_game", lambda *arguments: next(outcomes))

    record = module.evaluate_opponent(
        object(),
        "starter",
        "starter",
        iteration=40,
        snapshot_name="league-actor-00000040.pt",
        snapshot_digest="ab" * 32,
        seeds=range(7, 9),
        episode_steps=720,
    )

    assert record["event"] == "external_eval"
    assert record["iteration"] == 40
    assert record["opponent"] == "starter"
    assert record["opponent_sha256"] is None
    assert record["seed_start"] == 7
    assert (record["games"], record["completed_games"]) == (4, 3)
    assert record["money_mean"] == pytest.approx((100.0 + 60.0 + 30.0) / 3)
    assert record["opponent_money_mean"] == pytest.approx(50.0)
    assert record["score_rate"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert record["errors"] == ["environment did not reach DONE"]


def test_opponent_digest_follows_the_runnable_not_the_label(monkeypatch, tmp_path: Path) -> None:
    # An agent file literally named after a built-in must still record its
    # file digest; only true built-ins (runnable == name) omit one.
    module = _script()
    imposter = tmp_path / "starter"
    imposter.write_text("def act(observation):\n    return {}\n")
    label, runnable = normalize_opponent(str(imposter))
    assert label == "starter"
    monkeypatch.setattr(
        module,
        "_play_game",
        lambda *arguments: module.GameOutcome(0, 0, 1.0, 0.0, None),
    )

    record = module.evaluate_opponent(
        object(),
        label,
        runnable,
        iteration=1,
        snapshot_name="league-actor-00000001.pt",
        snapshot_digest="ab" * 32,
        seeds=range(1),
        episode_steps=720,
    )

    assert isinstance(record["opponent_sha256"], str) and len(record["opponent_sha256"]) == 64


def test_append_record_produces_one_json_line_per_call(tmp_path: Path) -> None:
    module = _script()
    output = tmp_path / "journal" / "metrics-external.jsonl"

    module.append_record(output, {"event": "external_eval", "iteration": 1})
    module.append_record(output, {"event": "external_eval", "iteration": 2})

    lines = output.read_text().splitlines()
    assert [json.loads(line)["iteration"] for line in lines] == [1, 2]
