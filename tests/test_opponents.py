from __future__ import annotations

from pathlib import Path

import pytest

from kaggriculture import opponents
from kaggriculture.script_opponents import parse_script_opponent

AGENT = "def agent(observation, configuration):\n    return {}\n"


@pytest.fixture
def reference_dir(tmp_path: Path, monkeypatch) -> Path:
    for name in opponents.REFERENCE_AGENTS:
        (tmp_path / f"{name}.py").write_text(AGENT)
    monkeypatch.setattr(opponents, "REFERENCE_AGENT_DIR", tmp_path)
    return tmp_path


def test_the_league_never_trains_on_a_held_out_reference() -> None:
    league = set(opponents.LEAGUE_REFERENCE_AGENTS)
    held_out = set(opponents.HELDOUT_REFERENCE_AGENTS)
    assert not league & held_out
    assert league | held_out == set(opponents.REFERENCE_AGENTS)


def test_a_reference_name_resolves_to_its_pinned_copy(reference_dir: Path) -> None:
    assert opponents.normalize_opponent("demand-timing") == (
        "demand-timing",
        str(reference_dir / "demand-timing.py"),
    )
    opponent = parse_script_opponent("hybrid-2965")
    assert opponent.name == "hybrid-2965"
    assert Path(opponent.path) == reference_dir / "hybrid-2965.py"


def test_a_missing_reference_copy_is_reported_not_guessed(reference_dir: Path) -> None:
    (reference_dir / "kaito-v48.py").unlink()
    with pytest.raises(FileNotFoundError, match="kaito-v48"):
        opponents.normalize_opponent("kaito-v48")
    with pytest.raises(ValueError, match="NAME=PATH or a reference agent"):
        parse_script_opponent("not-a-reference")


def test_agent_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KAGGRICULTURE_AGENT_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    default = tmp_path / "home" / ".local" / "share" / "kaggriculture" / "agents"
    assert opponents._agent_dir() == default

    monkeypatch.setenv("XDG_DATA_HOME", "relative/data")
    assert opponents._agent_dir() == default

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert opponents._agent_dir() == tmp_path / "data" / "kaggriculture" / "agents"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAGGRICULTURE_AGENT_DIR", "agents")
    assert opponents._agent_dir() == tmp_path.resolve() / "agents"
