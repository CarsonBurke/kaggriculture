from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_unflagged_benchmark_builds_the_exact_structured_production_model(monkeypatch) -> None:
    """A calibration report is launch evidence only for the shipped architecture
    and exact production model configuration."""
    from kaggriculture.modelargs import model_config_from_args
    from kaggriculture.production import (
        PRODUCTION_ARCHITECTURE,
        PRODUCTION_SELF_PLAY_GAMES,
        production_model_config,
    )
    from kaggriculture.registry import STRUCTURED, resolve_architecture
    from kaggriculture.structured import StructuredConfig

    module = _script()
    monkeypatch.setattr(sys, "argv", ["benchmark_ppo_iteration.py"])

    args = module.parse_args()
    config = model_config_from_args(resolve_architecture(args.architecture), args)

    assert not args.deterministic_training
    assert args.games == str(PRODUCTION_SELF_PLAY_GAMES)
    assert args.architecture == STRUCTURED == PRODUCTION_ARCHITECTURE
    assert config == StructuredConfig(**production_model_config())
    assert config.model_dim == 80
    assert config.ffn_multiplier == 2
    assert config.global_refresh_layers == ()
    assert config.fuse_market_decoder is True
    assert config.fuse_unit_decoder is False
    assert config.fused_mlp is False
    assert config.global_modulation is True


def test_benchmark_can_measure_the_deterministic_training_contract(monkeypatch) -> None:
    module = _script()
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_ppo_iteration.py", "--deterministic-training"],
    )

    assert module.parse_args().deterministic_training


def test_structured_benchmark_keeps_nextlat_state_and_rngs_per_batch_case(
    monkeypatch,
) -> None:
    from dataclasses import asdict

    from kaggriculture.production import production_ppo_config
    from kaggriculture.structured_dynamics import (
        StructuredCriticDynamics,
        StructuredDynamics,
    )

    module = _script()
    calls: list[dict[str, object]] = []

    def collect(*_arguments: object, **_keywords: object) -> SimpleNamespace:
        return SimpleNamespace(trajectories=2, state_count=4)

    parity = {
        "update_replay_unit_active_count": 1,
        "update_replay_kind_active_count": 1,
        "update_replay_quantity_active_count": 1,
        "update_replay_max_kl": 0.0,
        "update_replay_max_tail_fraction": 0.0,
    }

    def update(*arguments: object, **keywords: object) -> dict[str, float | int]:
        actor_dynamics = keywords["structured_dynamics"]
        critic_dynamics = keywords["structured_critic_dynamics"]
        generator = keywords["generator"]
        auxiliary_generator = keywords["auxiliary_generator"]
        assert isinstance(actor_dynamics, StructuredDynamics)
        assert isinstance(critic_dynamics, StructuredCriticDynamics)
        assert isinstance(generator, np.random.Generator)
        assert isinstance(auxiliary_generator, np.random.Generator)
        calls.append(
            {
                "actor": arguments[0],
                "critic": arguments[1],
                "config": arguments[5],
                "actor_dynamics": actor_dynamics,
                "critic_dynamics": critic_dynamics,
                "actor_optimizer": keywords["structured_dynamics_optimizer"],
                "critic_optimizer": keywords["structured_critic_dynamics_optimizer"],
                "actor_gate": keywords["structured_actor_auxiliary"],
                "critic_gate": keywords["structured_critic_auxiliary"],
                "generator": generator,
                "auxiliary_generator": auxiliary_generator,
                "rollout_draw": int(generator.integers(0, 1 << 62)),
                "auxiliary_draw": int(auxiliary_generator.integers(0, 1 << 62)),
            }
        )
        return {
            "first_minibatch_approx_kl": 0.0,
            "value_target_saturated_fraction": 0.0,
            "actor_updates": 1,
        }

    monkeypatch.setattr(
        module,
        "allocate_rollout_storage",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(module, "collect_mixed_play_rust", collect)
    monkeypatch.setattr(
        module,
        "update_replay_parity",
        lambda *_args, **_kwargs: parity,
    )
    monkeypatch.setattr(module, "_verify_first_step_critic_state", lambda *_args: None)
    monkeypatch.setattr(module, "update_ppo", update)
    monkeypatch.setattr(
        module,
        "rollout_diagnostics",
        lambda _rollout: {"money_mean": 0.0, "tie_fraction": 0.0, "score_rate": 0.5},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_ppo_iteration.py",
            "--device",
            "cpu",
            "--games",
            "1,2",
            "--league-games",
            "0",
            "--repeats",
            "2",
            "--model-dim",
            "16",
            "--attention-heads",
            "2",
            "--ffn-multiplier",
            "1",
            "--farm-blocks",
            "1",
            "--opponent-latents",
            "2",
            "--latents",
            "4",
            "--core-layers",
            "1",
            "--quantity-rank",
            "4",
            "--input-reinject-layers",
            "",
            "--core-skip-source",
            "0",
            "--core-skip-target",
            "0",
            "--mudd-lite",
            "false",
        ],
    )

    module.main()

    assert len(calls) == 4
    expected_ppo = production_ppo_config(update_compile_mode="default")
    assert all(asdict(call["config"]) == expected_ppo for call in calls)
    for call in calls:
        assert call["actor_gate"] is True
        assert call["critic_gate"] is True
        assert call["actor_optimizer"] is not call["critic_optimizer"]
        assert call["generator"] is not call["auxiliary_generator"]
    for first, second in ((calls[0], calls[1]), (calls[2], calls[3])):
        assert first["actor"] is second["actor"]
        assert first["critic"] is second["critic"]
        assert first["actor_dynamics"] is second["actor_dynamics"]
        assert first["critic_dynamics"] is second["critic_dynamics"]
        assert first["actor_optimizer"] is second["actor_optimizer"]
        assert first["critic_optimizer"] is second["critic_optimizer"]
        assert first["generator"] is second["generator"]
        assert first["auxiliary_generator"] is second["auxiliary_generator"]
    assert calls[0]["actor"] is not calls[2]["actor"]
    assert calls[0]["critic"] is not calls[2]["critic"]
    assert calls[0]["actor_dynamics"] is not calls[2]["actor_dynamics"]
    assert calls[0]["critic_dynamics"] is not calls[2]["critic_dynamics"]
    assert calls[0]["generator"] is not calls[2]["generator"]
    assert calls[0]["auxiliary_generator"] is not calls[2]["auxiliary_generator"]
    assert calls[0]["rollout_draw"] == calls[2]["rollout_draw"]
    assert calls[1]["rollout_draw"] == calls[3]["rollout_draw"]
    assert calls[0]["auxiliary_draw"] == calls[2]["auxiliary_draw"]
    assert calls[1]["auxiliary_draw"] == calls[3]["auxiliary_draw"]


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
    "--architecture",
    "entity-cnn",
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
    means. The benchmark defaults to the production explicit CUDA graph path;
    reports must name the exact configuration passed to the collector.

    The knob is the mode alone. No boolean projection of it is recorded, because
    the launcher requires every configuration key it does not strip to be
    identical across the three chain nodes, and a boolean derived from the mode
    differs exactly where the chain varies it -- so recording both would reject
    every chain.
    """
    module = _script()
    from kaggriculture.rollout import ROLLOUT_FORWARD_MODES

    cases = (
        ((), "inductor_graph", True),
        (("--rollout-forward-mode", "eager", "--no-rollout-bfloat16"), "eager", False),
        (("--rollout-forward-mode", "cudagraphs"), "cudagraphs", True),
        (("--rollout-forward-mode", "inductor", "--no-rollout-bfloat16"), "inductor", False),
        (("--rollout-forward-mode", "inductor_default"), "inductor_default", True),
        (("--rollout-forward-mode", "graph"), "graph", True),
        (("--rollout-forward-mode", "inductor_graph"), "inductor_graph", True),
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
