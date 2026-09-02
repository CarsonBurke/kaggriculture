from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _script():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_ppo_iteration.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_benchmark_ppo", path)
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
    monkeypatch.setattr(sys, "argv", ["benchmark_ppo_iteration.py"])

    args = module.parse_args()
    config = model_config_from_args(resolve_architecture(CONV_ENTITY), args)

    assert not args.deterministic_training
    assert config.to_dict() == production_model_config()


def test_benchmark_can_measure_the_deterministic_training_contract(monkeypatch) -> None:
    module = _script()
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_ppo_iteration.py", "--deterministic-training"],
    )

    assert module.parse_args().deterministic_training


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


def test_a_gate_override_may_tighten_but_never_loosen_the_shipped_bound() -> None:
    """A ceiling written beside a constant goes stale when the constant moves.

    The ceiling on `--max-first-minibatch-kl` was 1e-3 while the constant was
    1e-4. Recalibrating the constant against a behavior-cloned actor -- which
    diverges four orders of magnitude further than the random init the original
    number came from -- left every benchmark failing in argument parsing, with
    an error naming a limit nothing in the tree enforced any more.
    """
    module = _script()
    shipped = {attribute: ceiling for _flag, attribute, ceiling in module._NUMERICS_GATES}

    def validated(**overrides: float) -> None:
        module._validate_numerics_gates(SimpleNamespace(**{**shipped, **overrides}))

    # Every shipped value is admissible: it is exactly what training enforces.
    validated()
    for flag, attribute, ceiling in module._NUMERICS_GATES:
        validated(**{attribute: ceiling * 0.5})
        for rejected in (ceiling * 1.5, 0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match=f"{re.escape(flag)} must be finite"):
                validated(**{attribute: rejected})


#: A tiny conv model and a two-opponent wave: these tests are about which
#: collection configuration the script records and passes on, which is decided
#: before any of it runs. The knobs themselves are inert here by construction --
#: `rollout._rollout_model_forward` returns an eager forward off CUDA whatever
#: the mode says -- so nothing is gained by making the model production-sized.
_SMALL_WAVE = (
    "--device",
    "cpu",
    "--games",
    "1",
    "--league-games",
    "2",
    "--league-opponents",
    "2",
    "--repeats",
    "2",
    "--cnn-width",
    "8",
    "--cnn-blocks",
    "1",
    "--model-dim",
    "16",
    "--transformer-layers",
    "3",
)


class _ReachedCollector(Exception):
    """Raised in place of a collection, to end a run at the call under test."""


def _capturing_collector(captured: dict[str, object]):
    """A collector that records its keywords and ends the run there."""

    def collect(*_arguments: object, **keywords: object) -> None:
        captured.update(keywords)
        raise _ReachedCollector

    return collect


def test_the_report_and_the_collector_agree_on_the_collection_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    """The collection mode and precision decide what the rollout phase median
    means -- eager/fp32 measures 8.91 s against inductor/bf16's 5.36 s on the
    production wave -- and that median is what selects the shipped
    configuration. So a report has to name the configuration it timed, and it
    has to be the one the collector was actually given.

    The knob is the mode alone. No boolean projection of it is recorded, because
    the launcher requires every configuration key it does not strip to be
    identical across the three chain nodes, and a boolean derived from the mode
    differs exactly where the chain varies it -- so recording both would reject
    every chain.
    """
    module = _script()
    from kaggriculture.rollout import ROLLOUT_FORWARD_MODES

    cases = (
        ((), "inductor", True),
        (("--rollout-forward-mode", "eager", "--no-rollout-bfloat16"), "eager", False),
        (("--rollout-forward-mode", "cudagraphs"), "cudagraphs", True),
        (("--rollout-forward-mode", "inductor", "--no-rollout-bfloat16"), "inductor", False),
        (("--rollout-forward-mode", "inductor_default"), "inductor_default", True),
        (("--rollout-forward-mode", "graph"), "graph", True),
    )
    # Every mode is evidence here, so adding one to the tuple without measuring
    # it fails rather than passing untested.
    assert {mode for _flags, mode, _bfloat16 in cases} == set(ROLLOUT_FORWARD_MODES)

    for index, (flags, mode, bfloat16) in enumerate(cases):
        captured: dict[str, object] = {}
        monkeypatch.setattr(module, "collect_mixed_play_rust", _capturing_collector(captured))
        report = tmp_path / str(index) / "benchmark.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark_ppo_iteration.py", *_SMALL_WAVE, "--output", str(report), *flags],
        )
        try:
            with pytest.raises(_ReachedCollector):
                module.main()
        finally:
            module._configure_report(None)

        configuration = json.loads(report.read_text(encoding="utf-8").splitlines()[0])
        assert configuration["event"] == "configuration"
        assert configuration["rollout_forward_mode"] == mode
        assert configuration["rollout_bfloat16"] is bfloat16
        assert "compile_rollout" not in configuration
        assert captured["forward_mode"] == mode
        assert captured["forward_autocast"] is bfloat16
        # The mode is now the collector's only execution knob: the retired
        # `compile_models` boolean is gone from its signature, so a league wave
        # cannot compile one of its two forwards and not the other.
        assert "compile_models" not in captured


def test_the_report_declares_the_update_mode_the_launcher_reads(
    monkeypatch, tmp_path: Path
) -> None:
    """The update knob rides in the report's nested `ppo` block, whole.

    `launch_calibrated_training._declared_knobs` reads it from exactly there,
    and `_comparable_configuration` strips exactly that key from the cross-node
    comparison. A boolean projection beside it would differ precisely where the
    chain varies the knob, so recording both would reject every chain -- the
    same failure the retired collection boolean caused one phase over.
    """
    module = _script()

    for index, mode in enumerate(module.UPDATE_COMPILE_MODES):
        captured: dict[str, object] = {}
        monkeypatch.setattr(module, "collect_mixed_play_rust", _capturing_collector(captured))
        report = tmp_path / str(index) / "benchmark.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "benchmark_ppo_iteration.py",
                *_SMALL_WAVE,
                "--output",
                str(report),
                "--update-compile-mode",
                mode,
            ],
        )
        try:
            with pytest.raises(_ReachedCollector):
                module.main()
        finally:
            module._configure_report(None)

        configuration = json.loads(report.read_text(encoding="utf-8").splitlines()[0])
        assert configuration["ppo"]["update_compile_mode"] == mode
        assert "compile_update" not in configuration["ppo"]
        assert "update_compile_mode" not in configuration


@pytest.mark.parametrize(
    "flags",
    [
        ("--compile-rollout",),
        ("--no-compile-rollout",),
        ("--compile-update",),
        ("--no-compile-update",),
    ],
)
def test_the_retired_compile_booleans_are_rejected(monkeypatch, flags: tuple[str, ...]) -> None:
    """A recipe still passing either old boolean must fail loudly. Neither can be
    honoured: the mode is what its phase consults, so a boolean could only agree
    redundantly or contradict, and neither can be ignored either -- a run that
    accepted `--compile-rollout` while timing an eager collector, or
    `--compile-update` while timing an eager update, would put a configuration
    nothing measured into the launcher's evidence.
    """
    module = _script()
    monkeypatch.setattr(sys, "argv", ["benchmark_ppo_iteration.py", *_SMALL_WAVE, *flags])

    with pytest.raises(SystemExit) as failure:
        module.main()

    assert failure.value.code != 0
