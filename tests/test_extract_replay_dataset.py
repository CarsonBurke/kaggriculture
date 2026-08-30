from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_extractor():
    path = Path(__file__).parents[1] / "scripts" / "extract_replay_dataset.py"
    spec = importlib.util.spec_from_file_location("extract_replay_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_duplicate_replay_episode_ids_are_rejected(tmp_path: Path) -> None:
    extractor = _load_extractor()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"info": {"EpisodeId": 42}}), encoding="utf-8")
    second.write_text(json.dumps({"info": {"EpisodeId": 42}}), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"duplicate replay EpisodeId 42: first.json and second.json",
    ):
        extractor.reject_duplicate_episode_ids([first, second])
