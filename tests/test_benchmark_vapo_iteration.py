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


def test_unflagged_conv_benchmark_builds_the_production_model(monkeypatch) -> None:
    """A calibration report is only launch evidence when its model matches
    production exactly, and launch_calibrated_training rejects it otherwise."""
    import sys

    from kaggriculture.modelargs import model_config_from_args
    from kaggriculture.production import production_model_config
    from kaggriculture.registry import CONV_ENTITY, resolve_architecture

    module = _script()
    monkeypatch.setattr(sys, "argv", ["benchmark_vapo_iteration.py"])

    config = model_config_from_args(resolve_architecture(CONV_ENTITY), module.parse_args())

    assert config.to_dict() == production_model_config()


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


def test_first_step_critic_state_verification_covers_both_families() -> None:
    """The pre-update integrity gate must pass on a genuine mixed wave and
    fail once a stored critic-only value is corrupted, for each family."""
    import numpy as np

    from kaggriculture.model import FarmActor, ModelConfig
    from kaggriculture.registry import CONV_ENTITY, STRUCTURED
    from kaggriculture.rollout import collect_mixed_play_rust
    from kaggriculture.structured import StructuredActor, StructuredConfig

    module = _script()
    torch.manual_seed(0)
    cases = (
        (
            FarmActor(ModelConfig(cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3)),
            CONV_ENTITY,
            "critic_features",
        ),
        (
            StructuredActor(
                StructuredConfig(
                    model_dim=16,
                    attention_heads=2,
                    ffn_multiplier=1,
                    farm_blocks=1,
                    opponent_latents=2,
                    latents=4,
                    core_layers=1,
                    quantity_rank=4,
                )
            ),
            STRUCTURED,
            "critic_products",
        ),
    )
    for actor, architecture, critic_field in cases:
        rollout = collect_mixed_play_rust(
            actor,
            [actor],
            self_play_games=1,
            league_games=2,
            opponent_indices=np.asarray([0, 0]),
            seed_start=17,
            sampling_seed=3,
        )
        assert rollout.architecture == architecture

        module._verify_first_step_critic_state(rollout, 1, 17)

        rollout.states[critic_field][0, 0] += 1
        with pytest.raises(RuntimeError, match="fresh native encode"):
            module._verify_first_step_critic_state(rollout, 1, 17)
