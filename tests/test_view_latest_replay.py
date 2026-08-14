from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _script():
    path = Path(__file__).parents[1] / "scripts" / "view_latest_replay.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_view_latest_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_run_is_the_one_with_the_newest_checkpoint(tmp_path: Path) -> None:
    module = _script()
    for name, age_seconds in (("older", 200), ("newest", 100), ("empty", 0)):
        run = tmp_path / name
        run.mkdir()
        if name == "empty":
            continue
        checkpoint = run / "latest.pt"
        checkpoint.write_bytes(b"checkpoint")
        stamp = checkpoint.stat().st_mtime - age_seconds
        os.utime(checkpoint, (stamp, stamp))

    assert module.resolve_latest_run(tmp_path) == tmp_path / "newest"

    for name in ("older", "newest"):
        (tmp_path / name / "latest.pt").unlink()
    with pytest.raises(FileNotFoundError, match=r"latest\.pt"):
        module.resolve_latest_run(tmp_path)


def test_replay_output_is_named_by_iteration_mode_seed_and_opponent(tmp_path: Path) -> None:
    module = _script()
    output = module.replay_output_path(tmp_path, 61, "sampled", 424242, "self")
    assert output == tmp_path / "replays" / "iteration-000061-sampled-seed424242-vs-self.html"
    versus = module.replay_output_path(tmp_path, 61, "deterministic", 0, "public-v27")
    assert (
        versus == tmp_path / "replays" / "iteration-000061-deterministic-seed0-vs-public-v27.html"
    )


def test_external_opponent_occupies_the_seat_our_agent_does_not(monkeypatch) -> None:
    """The engine must receive our callable in --seat and the opponent elsewhere."""
    module = _script()
    played: list[list[object]] = []

    class FakeEnvironment:
        def run(self, players):
            played.append(players)

    monkeypatch.setattr(module, "make", lambda *args, **kwargs: FakeEnvironment(), raising=False)
    import sys
    import types

    fake = types.ModuleType("kaggle_environments")
    fake.make = lambda *args, **kwargs: FakeEnvironment()
    monkeypatch.setitem(sys.modules, "kaggle_environments", fake)

    for candidate_seat in (0, 1):
        module.play_match(
            object(),
            mode="deterministic",
            seed=0,
            episode_steps=720,
            temperature=1.0,
            opponent="/tmp/v27.py",
            candidate_seat=candidate_seat,
        )
        players = played[-1]
        assert players[1 - candidate_seat] == "/tmp/v27.py"
        assert callable(players[candidate_seat])
