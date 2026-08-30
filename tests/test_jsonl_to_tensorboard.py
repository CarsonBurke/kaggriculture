from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_converter():
    path = Path(__file__).parents[1] / "scripts" / "jsonl_to_tensorboard.py"
    spec = importlib.util.spec_from_file_location("jsonl_to_tensorboard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_output_root_destinations_include_canonical_source_identity(tmp_path: Path) -> None:
    converter = _load_converter()
    first = tmp_path / "runs" / "a" / "metrics.jsonl"
    second = tmp_path / "other" / "a" / "metrics.jsonl"
    for journal in (first, second):
        journal.parent.mkdir(parents=True)
        journal.write_text('{"iteration": 1}\n', encoding="utf-8")

    output_root = tmp_path / "tensorboard"
    first_destination = converter._destination(first, output_root)
    second_destination = converter._destination(second, output_root)

    assert first_destination != second_destination
    assert first_destination.parent == output_root.resolve()
    assert converter._destination(first, output_root) == first_destination
