from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _script():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_vapo_iteration.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_benchmark_vapo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completion_record_covers_the_exact_cartesian_product() -> None:
    module = _script()

    completion = module._completion_record([64, 112], 3)

    assert completion == {
        "event": "benchmark_complete",
        "completed": True,
        "self_play_game_counts": [64, 112],
        "repeats": 3,
        "completed_batches": [
            {"self_play_games": 64, "completed_repeats": [0, 1, 2]},
            {"self_play_games": 112, "completed_repeats": [0, 1, 2]},
        ],
        "iteration_records": 6,
        "batch_summaries": 2,
    }


def test_hardware_identity_records_common_cpu_metadata() -> None:
    module = _script()

    identity = module._hardware_identity(torch.device("cpu"))

    assert identity["device_type"] == "cpu"
    assert identity["machine"]
    assert identity["cpu_count"] is None or identity["cpu_count"] > 0
    assert "torch_cuda_version" in identity
    assert "cudnn_version" in identity


def test_report_persistence_is_atomic_complete_jsonl(tmp_path: Path) -> None:
    module = _script()
    destination = tmp_path / "nested" / "benchmark.jsonl"
    configuration = {"event": "configuration", "seed": 7}
    completion = module._completion_record([112], 2)
    try:
        module._configure_report(destination)
        module.emit(configuration)
        module.emit(completion)
    finally:
        module._configure_report(None)

    assert [json.loads(line) for line in destination.read_text().splitlines()] == [
        configuration,
        completion,
    ]
    assert not list(destination.parent.glob(".*.tmp"))


def test_emit_rejects_nonfinite_values_without_extending_report(tmp_path: Path) -> None:
    module = _script()
    destination = tmp_path / "benchmark.jsonl"
    try:
        module._configure_report(destination)
        module.emit({"event": "configuration"})
        with pytest.raises(FloatingPointError, match="non-finite"):
            module.emit({"event": "iteration", "seconds": float("inf")})
    finally:
        module._configure_report(None)

    assert [json.loads(line) for line in destination.read_text().splitlines()] == [
        {"event": "configuration"}
    ]
