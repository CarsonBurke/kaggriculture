from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kaggriculture.league import (
    BuiltinRef,
    BuiltinSelection,
    SnapshotRef,
    SnapshotSelection,
    save_actor_snapshot,
    snapshot_sha256,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.modelargs import model_config_from_args
from kaggriculture.orientation import row_orientations
from kaggriculture.ppo import PpoConfig
from kaggriculture.registry import CONV_ENTITY, STRUCTURED, resolve_architecture
from kaggriculture.rollout import population_pairings
from kaggriculture.structured import StructuredConfig


def _training_script():
    path = Path(__file__).parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_train_ppo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_defaults_prioritize_fresh_games_and_diverse_league(monkeypatch, tmp_path) -> None:
    module = _training_script()
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path)])

    args = module.parse_args()

    assert (args.games, args.league_games) == (112, 96)
    assert (args.league_active_opponents, args.league_historical_opponents) == (2, 2)
    assert args.league_active_pool_size == 16
    assert (args.league_builtin_opponents, args.league_builtin_lanes) == ("", 0)
    assert args.epochs == 1
    assert args.critic_lr == pytest.approx(2.5e-4)
    assert args.minibatch_size == 2048
    # An unflagged run is exactly the family's dataclass configuration, which
    # is what a warm-start artifact and the calibration benchmark both carry.
    assert model_config_from_args(resolve_architecture(args.architecture), args) == ModelConfig()
    # Entropy is telemetry only; the training CLI has no bonus coefficient.
    assert not hasattr(args, "entropy_coefficient")
    assert args.gamma == pytest.approx(0.99)
    assert args.actor_gae_lambda == pytest.approx(0.95)
    assert not hasattr(args, "gae_lambda")
    assert args.target_kl == PpoConfig.target_kl
    assert args.checkpoint_every == 5
    module._validate_args(args)

    args.gamma = 1.5
    with pytest.raises(ValueError, match="gamma must be finite"):
        module._validate_args(args)
    args.gamma = 0.0
    with pytest.raises(ValueError, match="gamma must be finite"):
        module._validate_args(args)
    args.gamma = PpoConfig.gamma

    # The k3 estimator is non-negative and the trust region stops on
    # `batch_kl > target_kl`, so a non-positive value admits at most the
    # exactly-parity first minibatch and otherwise no optimizer step at all.
    # That collapses the update silently, which is worse than failing to start.
    for rejected in (0.0, -0.01, float("nan"), float("inf")):
        args.target_kl = rejected
        with pytest.raises(ValueError, match="target KL"):
            module._validate_args(args)


def test_built_in_league_flags_reach_selection_and_the_data_provenance(
    monkeypatch, tmp_path
) -> None:
    """A resume that changed which reference agents play is a different data
    generator, so the setting has to be inside `_training_data_config`."""
    module = _training_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--run-dir",
            str(tmp_path),
            "--league-builtin-opponents",
            " starter , pass ",
            "--league-builtin-lanes",
            "2",
        ],
    )

    args = module.parse_args()
    module._validate_args(args)

    assert module._league_builtin_opponents(args) == ["starter", "pass"]
    recorded = module._training_data_config(args, torch.device("cpu"))
    assert recorded["league_builtin_opponents"] == "starter,pass"
    assert recorded["league_builtin_lanes"] == 2


def test_built_in_league_configuration_must_be_admitted_and_reserved_together(
    monkeypatch, tmp_path
) -> None:
    module = _training_script()
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path)])
    args = module.parse_args()

    args.league_builtin_opponents = "starter"
    with pytest.raises(ValueError, match="must be set together"):
        module._validate_args(args)

    args.league_builtin_opponents = ""
    args.league_builtin_lanes = 2
    with pytest.raises(ValueError, match="must be set together"):
        module._validate_args(args)

    args.league_builtin_opponents = "public-v27"
    with pytest.raises(ValueError, match="unknown built-in"):
        module._validate_args(args)

    args.league_builtin_opponents = "starter,starter"
    with pytest.raises(ValueError, match="distinct"):
        module._validate_args(args)


def test_model_flags_are_family_scoped_and_default_to_the_family_configuration(
    monkeypatch, tmp_path
) -> None:
    """Warm starting compares model configurations for equality, so an
    unflagged run must build the family default and a foreign flag must fail
    loudly instead of being silently dropped."""
    module = _training_script()

    monkeypatch.setattr(
        sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path), "--architecture", STRUCTURED]
    )
    args = module.parse_args()
    structured = resolve_architecture(STRUCTURED)
    assert model_config_from_args(structured, args) == StructuredConfig()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--run-dir",
            str(tmp_path),
            "--architecture",
            STRUCTURED,
            "--core-layers",
            "4",
            "--model-dim",
            "64",
        ],
    )
    args = module.parse_args()
    assert model_config_from_args(structured, args) == StructuredConfig(model_dim=64, core_layers=4)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--run-dir",
            str(tmp_path),
            "--architecture",
            STRUCTURED,
            "--transformer-layers",
            "7",
        ],
    )
    args = module.parse_args()
    with pytest.raises(ValueError, match=r"--transformer-layers do not apply"):
        model_config_from_args(structured, args)

    monkeypatch.setattr(
        sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path), "--latents", "16"]
    )
    args = module.parse_args()
    with pytest.raises(ValueError, match=r"--latents do not apply"):
        model_config_from_args(resolve_architecture(CONV_ENTITY), args)


def test_training_rejects_a_league_budget_that_drops_opponent_categories(
    monkeypatch, tmp_path
) -> None:
    module = _training_script()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ppo.py", "--run-dir", str(tmp_path), "--league-games", "2"],
    )

    with pytest.raises(ValueError, match="initial anchor"):
        module._validate_args(module.parse_args())


def test_balanced_opponent_assignments_are_reproducible_and_nearly_equal() -> None:
    module = _training_script()
    first = module._balanced_assignments(11, 4, np.random.default_rng(7))
    second = module._balanced_assignments(11, 4, np.random.default_rng(7))

    np.testing.assert_array_equal(first, second)
    counts = np.bincount(first, minlength=4)
    assert counts.max() - counts.min() == 1
    with pytest.raises(ValueError):
        module._balanced_assignments(0, 4, np.random.default_rng(7))


def test_league_diagnostics_remain_separate_per_frozen_policy(tmp_path) -> None:
    module = _training_script()
    league = SimpleNamespace(
        final_money=np.asarray([100.0, 50.0, 80.0, 90.0, 200.0, 10.0]),
        opponent_money=np.asarray([90.0, 60.0, 80.0, 20.0, 30.0, 40.0]),
    )
    assignments = np.asarray([0, 0, 1, 1, 2, 2])
    selections = [
        SnapshotSelection(SnapshotRef(2, tmp_path / "historical.pt"), "historical"),
        SnapshotSelection(SnapshotRef(9, tmp_path / "active.pt"), "active"),
        BuiltinSelection(BuiltinRef("starter")),
    ]

    diagnostics, measured_rates = module._league_opponent_diagnostics(
        league, assignments, selections
    )

    assert diagnostics["league_opponent_00000002_category"] == "historical"
    assert diagnostics["league_opponent_00000002_games"] == 2
    assert diagnostics["league_opponent_00000002_score_rate"] == 0.5
    assert diagnostics["league_opponent_00000009_category"] == "active"
    assert diagnostics["league_opponent_00000009_games"] == 2
    assert diagnostics["league_opponent_00000009_score_rate"] == 0.75
    # A built-in earns its own journal series and its own PFSP estimate, which
    # is what lets it retire from the league on its own measurements.
    assert diagnostics["league_opponent_builtin_starter_category"] == "builtin"
    assert diagnostics["league_opponent_builtin_starter_games"] == 2
    assert diagnostics["league_opponent_builtin_starter_score_rate"] == 0.5
    assert measured_rates == {"00000002": 0.5, "00000009": 0.75, "builtin_starter": 0.5}


def test_disabled_league_selection_does_not_advance_training_rng(tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        league_games=0,
        league_active_opponents=2,
        league_historical_opponents=2,
        league_active_pool_size=16,
        league_builtin_opponents="pass,random,starter",
        league_builtin_lanes=3,
    )
    generator = np.random.default_rng(41)
    reference = np.random.default_rng(41)
    refs = [SnapshotRef(0, tmp_path / "league-actor-00000000.pt")]

    assert (
        module._select_league_opponents(args, refs, 1, generator, {}, pretrained_start=False) == []
    )
    assert generator.random() == reference.random()


def test_league_score_rate_validation_accepts_only_finite_unit_interval_state() -> None:
    module = _training_script()

    assert module._validate_league_score_rates({}) == {}
    assert module._validate_league_score_rates({"00000003": 0.25, "builtin_starter": 1.0}) == {
        "00000003": 0.25,
        "builtin_starter": 1.0,
    }
    for invalid in (
        None,
        [("00000003", 0.25)],
        {3: 0.5},
        {"3": 0.5},
        {"builtin_v27": 0.5},
        {"00000003": 1},
        {"00000003": float("nan")},
        {"00000003": 1.5},
        {"00000003": -0.1},
    ):
        with pytest.raises(ValueError):
            module._validate_league_score_rates(invalid)


def test_league_score_rate_blend_seeds_from_prior_and_decays_unmeasured() -> None:
    module = _training_script()
    rates = {"00000001": 1.0}

    module._blend_league_score_rates(rates, {"builtin_starter": 1.0})

    # A first measurement blends against the unmeasured prior of 0.5, so one
    # perfect wave can never pin an estimate at exactly 1.0 and hard-retire a
    # freshly met opponent.
    assert rates["builtin_starter"] == pytest.approx(0.75)
    # Opponents that were not sampled decay toward the prior, keeping
    # retirement provisional instead of permanent.
    assert rates["00000001"] == pytest.approx(0.975)

    module._blend_league_score_rates(rates, {"builtin_starter": 0.25})
    assert rates["builtin_starter"] == pytest.approx(0.5)


def test_external_eval_launcher_respects_cadence_and_running_worker(monkeypatch, tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval_every=10,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        episode_steps=720,
        run_dir=tmp_path,
        population=1,
    )
    league_directory = tmp_path / "league"
    launched: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            launched.append(command)
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)

    # Iteration zero and off-cadence iterations never launch a worker.
    assert module._maybe_launch_external_eval(args, 0, league_directory, None) is None
    assert module._maybe_launch_external_eval(args, 7, league_directory, None) is None

    process = module._maybe_launch_external_eval(args, 10, league_directory, None)
    assert isinstance(process, FakeProcess)
    command = launched[0]
    assert command[command.index("--artifact") + 1].endswith("league-actor-00000010.pt")
    # A single learner names no member: the artifact holds exactly one actor.
    assert command[command.index("--agents") + 1] == ""
    assert command[command.index("--iteration") + 1] == "10"
    assert command[command.index("--opponents") + 1] == "starter"
    assert command[command.index("--output") + 1] == str(tmp_path / "metrics-external.jsonl")
    assert command[command.index("--seeds") + 1] == "2"
    assert command[command.index("--episode-steps") + 1] == "720"

    # A still-running worker skips the tick instead of stacking processes; a
    # finished one is replaced on the next due iteration.
    assert module._maybe_launch_external_eval(args, 20, league_directory, process) is process
    assert len(launched) == 1
    process.returncode = 0
    replacement = module._maybe_launch_external_eval(args, 20, league_directory, process)
    assert isinstance(replacement, FakeProcess) and replacement is not process
    assert len(launched) == 2

    disabled = SimpleNamespace(**{**vars(args), "external_eval_every": 0})
    assert module._maybe_launch_external_eval(disabled, 30, league_directory, None) is None
    assert len(launched) == 2


def test_a_population_is_probed_from_its_durable_checkpoint(monkeypatch, tmp_path) -> None:
    # `latest.pt` is rewritten every iteration, so a worker reading it races the
    # trainer. The numbered checkpoint is immutable once written, which is what
    # ties the probe cadence to `--checkpoint-every`.
    module = _training_script()
    args = SimpleNamespace(
        external_eval_every=10,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        episode_steps=720,
        run_dir=tmp_path,
        population=4,
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **_kwargs: launched.append(command) or SimpleNamespace(poll=lambda: 0),
    )

    module._maybe_launch_external_eval(args, 10, tmp_path / "league", None)

    command = launched[0]
    assert command[command.index("--artifact") + 1] == str(tmp_path / "checkpoint-000010.pt")
    # Every member is named, in agent order: one worker covers the population,
    # and the journal rows are told apart by their `agent` field.
    assert command[command.index("--agents") + 1] == "0,1,2,3"


def test_external_eval_launch_failure_never_kills_training(monkeypatch, tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval_every=10,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        episode_steps=720,
        run_dir=tmp_path,
        population=1,
    )

    def refuse(*_args, **_kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(module.subprocess, "Popen", refuse)

    assert module._maybe_launch_external_eval(args, 10, tmp_path / "league", None) is None


def test_external_eval_opponent_resolution_degrades_instead_of_blocking(capsys, tmp_path) -> None:
    module = _training_script()
    missing = tmp_path / "gone.py"
    args = SimpleNamespace(
        external_eval_every=10,
        external_eval_opponents=f"starter,{missing},",
    )

    module._resolve_external_eval_opponents(args)

    assert args.external_eval_every == 10
    assert args.external_eval_opponents == "starter"
    assert "dropped" in capsys.readouterr().err

    args = SimpleNamespace(external_eval_every=10, external_eval_opponents=str(missing))
    module._resolve_external_eval_opponents(args)
    assert args.external_eval_every == 0
    assert "disabled" in capsys.readouterr().err


def test_training_data_config_captures_rollout_semantics(monkeypatch, tmp_path) -> None:
    module = _training_script()
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path)])
    args = module.parse_args()

    config = module._training_data_config(args, module._device("cpu"))

    assert config["games"] == 112
    assert config["league_games"] == 96
    assert config["update_compile_mode"] == "default"
    assert config["device_type"] == "cpu"
    # Measurement selected inductor + bf16 for collection, so an unflagged run
    # is the configuration the parity gate was measured on, and the record says
    # which one it was.
    assert config["rollout_forward_mode"] == "inductor"
    assert config["rollout_bfloat16"] is True
    # All three move what a resume would produce -- the collection pair moves the
    # sampled behavior policy and the update mode moves the graphs that consume
    # it -- so a resume that changes any of them is a different data generator
    # and must not match the checkpoint's record.
    for knob, value in (
        ("rollout_forward_mode", "eager"),
        ("rollout_bfloat16", False),
        ("update_compile_mode", "eager"),
    ):
        changed = SimpleNamespace(**{**vars(args), knob: value})
        assert module._training_data_config(changed, module._device("cpu")) != config
    args.games += 1
    assert module._training_data_config(args, module._device("cpu")) != config


def test_collection_forward_defaults_to_the_measured_configuration(monkeypatch, tmp_path) -> None:
    """Collection is ~64% of a wave and inductor + bf16 is what measurement
    selected on both axes: the rollout sweep moves 8.91 s -> 5.36 s (1.66x) and
    the 4-wave parity gate moves worst max_kl 1.9089e-03 -> 2.2786e-04 (8.4x
    lower drift), because it matches the update path's backend and precision.
    The mode is the whole collection compile decision -- there is no separate
    boolean that could disagree with it -- and the precision is decided
    independently of it."""
    module = _training_script()

    def parsed(*flags: str):
        monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path), *flags])
        return module.parse_args()

    default = parsed()
    assert default.rollout_forward_mode == "inductor"
    assert default.rollout_bfloat16 is True
    module._validate_args(default)
    assert not hasattr(default, "compile_rollout")

    for flags in (
        ("--rollout-forward-mode", "eager", "--no-rollout-bfloat16"),
        ("--rollout-forward-mode", "cudagraphs"),
        ("--no-rollout-bfloat16",),
    ):
        module._validate_args(parsed(*flags))

    with pytest.raises(SystemExit):
        parsed("--rollout-forward-mode", "manual_graph")


def test_the_update_compile_knob_is_a_mode_with_no_boolean_beside_it(monkeypatch, tmp_path) -> None:
    """The update knob is `torch.compile`'s `mode=`, so a boolean cannot name it.

    `--compile-update` is deleted rather than kept as a projection: with both a
    boolean and a mode at the CLI the two could disagree, and the calibration
    chain identifies this phase's knob by the mode. The domain is enforced here,
    at the boundary, because `provenance` re-derives the decision inside the
    submission bundle and cannot import `ppo.py` to learn what the modes are.
    """
    module = _training_script()

    def parsed(*flags: str):
        monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path), *flags])
        return module.parse_args()

    default = parsed()
    assert default.update_compile_mode == "default"
    module._validate_args(default)
    assert not hasattr(default, "compile_update")

    for mode in module.UPDATE_COMPILE_MODES:
        assert parsed("--update-compile-mode", mode).update_compile_mode == mode

    for rejected in ("cudagraphs", "true", "1", "reduce_overhead"):
        with pytest.raises(SystemExit):
            parsed("--update-compile-mode", rejected)


def test_calibration_provenance_binds_initial_command_but_remains_portable(
    monkeypatch, tmp_path
) -> None:
    module = _training_script()
    identity = module.source_identity()
    initial_argv = ["train_ppo.py", "--run-dir", str(tmp_path / "initial")]
    decision = {
        "source_identity": identity,
        "rollout_forward_mode": "eager",
        "update_compile_mode": "eager",
        "eager_report_sha256": "a" * 64,
        "eager_report_size_bytes": 100,
        "mixed_report_sha256": "c" * 64,
        "mixed_report_size_bytes": 110,
        "compiled_report_sha256": "b" * 64,
        "compiled_report_size_bytes": 120,
        "minimum_compile_speedup": 1.05,
        "attributed_knob_speedups": {"rollout_forward_mode": 1.0, "update_compile_mode": 1.0},
        "training_command": [
            sys.executable,
            str((Path(__file__).parents[1] / "scripts" / "train_ppo.py").resolve()),
            *initial_argv[1:],
        ],
    }
    path = tmp_path / "calibration-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", initial_argv)

    provenance = module._load_run_provenance(path, identity, bind_command=True)

    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path / "portable")])
    assert module._load_run_provenance(path, identity, bind_command=False) == provenance
    with pytest.raises(ValueError, match="exact training command"):
        module._load_run_provenance(path, identity, bind_command=True)


def test_league_manifest_restore_is_portable_crash_tolerant_and_rejects_rewinds(
    tmp_path,
) -> None:
    module = _training_script()
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    source_run = tmp_path / "source"
    source_run.mkdir()
    checkpoint = source_run / "checkpoint-000001.pt"
    checkpoint.touch()
    manifest = {}
    for iteration in range(2):
        ref = save_actor_snapshot(source_run / "league", actor, iteration)
        manifest[iteration] = snapshot_sha256(ref.path)
        with torch.no_grad():
            next(actor.parameters()).add_(0.01)

    validated = module._validate_league_manifest(manifest, current_iteration=1)
    destination = tmp_path / "restored" / "league"
    module._restore_league_archive(
        checkpoint=checkpoint,
        destination=destination,
        manifest=validated,
        model_config=model_config,
    )

    assert {
        ref.iteration: snapshot_sha256(ref.path) for ref in module.list_actor_snapshots(destination)
    } == manifest

    save_actor_snapshot(destination, actor, 2)
    module._restore_league_archive(
        checkpoint=checkpoint,
        destination=destination,
        manifest=validated,
        model_config=model_config,
    )
    save_actor_snapshot(destination, actor, 3)
    with pytest.raises(ValueError, match="fresh --run-dir"):
        module._restore_league_archive(
            checkpoint=checkpoint,
            destination=destination,
            manifest=validated,
            model_config=model_config,
        )


def test_league_manifest_requires_every_iteration() -> None:
    module = _training_script()

    with pytest.raises(ValueError, match="one snapshot per iteration"):
        module._validate_league_manifest({0: "a" * 64, 2: "b" * 64}, current_iteration=2)
    with pytest.raises(ValueError, match="digest"):
        module._validate_league_manifest({0: "not-a-digest"}, current_iteration=0)


def test_main_writes_complete_manifests_and_portably_resumes(
    monkeypatch,
    tmp_path,
) -> None:
    module = _training_script()

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    rollout = SimpleNamespace(state_count=1)
    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(module, "collect_mixed_play_rust", lambda *args, **kwargs: rollout)
    monkeypatch.setattr(module, "slice_trajectories", lambda batch, start, stop: batch)
    monkeypatch.setattr(module, "rollout_diagnostics", lambda batch: {})
    monkeypatch.setattr(
        module,
        "update_replay_parity",
        lambda *args, **kwargs: _parity_metrics(module),
    )
    monkeypatch.setattr(
        module,
        "update_ppo",
        lambda *args, **kwargs: {
            "actor_updates": 1,
            "actor_minibatches_intended": 1,
            "critic_updates": 1,
            "first_minibatch_approx_kl": 0.0,
            "value_target_saturated_fraction": 0.0,
            "entropy": 0.2,
        },
    )

    def arguments(run_dir: Path, iterations: int, resume: Path | None = None) -> list[str]:
        values = [
            "train_ppo.py",
            "--run-dir",
            str(run_dir),
            "--iterations",
            str(iterations),
            "--games",
            "1",
            "--league-games",
            "0",
            "--device",
            "cpu",
            "--cnn-width",
            "8",
            "--cnn-blocks",
            "1",
            "--model-dim",
            "16",
            "--transformer-layers",
            "3",
            "--attention-heads",
            "2",
            "--checkpoint-every",
            "1",
            "--no-bfloat16",
        ]
        if resume is not None:
            values.extend(("--resume", str(resume)))
        return values

    source_run = tmp_path / "source"
    monkeypatch.setattr(sys, "argv", arguments(source_run, 1))
    module.main()

    initial = torch.load(source_run / "checkpoint-000000.pt", weights_only=False)
    latest = torch.load(source_run / "latest.pt", weights_only=False)
    numbered = torch.load(source_run / "checkpoint-000001.pt", weights_only=False)
    assert set(initial["league_snapshot_manifest"]) == {0}
    assert set(latest["league_snapshot_manifest"]) == {0, 1}
    assert numbered["league_snapshot_manifest"] == latest["league_snapshot_manifest"]

    # A kill after latest.pt but before its cadence checkpoint is repaired on
    # resume before another rollout starts.
    (source_run / "checkpoint-000001.pt").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(source_run, 1, source_run / "latest.pt"),
    )
    module.main()
    repaired = torch.load(source_run / "checkpoint-000001.pt", weights_only=False)
    assert repaired["league_snapshot_manifest"] == latest["league_snapshot_manifest"]

    monkeypatch.setattr(
        sys,
        "argv",
        arguments(source_run, 1, source_run / "checkpoint-000000.pt"),
    )
    with pytest.raises(ValueError, match="newer than the checkpoint"):
        module.main()

    monkeypatch.setattr(sys, "argv", arguments(source_run, 1))
    with pytest.raises(FileExistsError, match="pre-existing initial checkpoint"):
        module.main()

    # Simulate a kill after snapshot K+1 became visible but before latest.pt
    # was replaced. The mocked update leaves actor weights unchanged, so replay
    # must accept the orphan only if it regenerates the same snapshot state.
    orphan_actor = module.load_actor_snapshot(
        source_run / "league" / "league-actor-00000001.pt",
        device="cpu",
    )
    save_actor_snapshot(source_run / "league", orphan_actor, 2)
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(source_run, 2, source_run / "latest.pt"),
    )
    module.main()
    recovered = torch.load(source_run / "latest.pt", weights_only=False)
    assert set(recovered["league_snapshot_manifest"]) == {0, 1, 2}

    resumed_run = tmp_path / "resumed"
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(resumed_run, 2, source_run / "checkpoint-000001.pt"),
    )
    module.main()

    resumed = torch.load(resumed_run / "checkpoint-000002.pt", weights_only=False)
    assert set(resumed["league_snapshot_manifest"]) == {0, 1, 2}
    assert [ref.iteration for ref in module.list_actor_snapshots(resumed_run / "league")] == [
        0,
        1,
        2,
    ]

    portable_only = tmp_path / "portable-only"
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(portable_only, 1, source_run / "checkpoint-000001.pt"),
    )
    module.main()
    assert (portable_only / "latest.pt").is_file()


def test_parity_audit_is_due_per_staging_configuration_and_on_a_cadence() -> None:
    module = _training_script()
    interval = module.REPLAY_PARITY_AUDIT_INTERVAL
    league = module._parity_staging_key(96)
    self_play = module._parity_staging_key(0)
    population = module._parity_staging_key(0, 4)
    assert {league, self_play, population} == set(module.PARITY_STAGING_KEYS)
    # A population wave stages its behaviour policy as one vmapped ensemble
    # forward, a third staging width, so it audits on its own cadence even
    # though it plays no league rows.
    assert module._parity_staging_key(96, 4) == population

    # Nothing audited yet: due whichever configuration this iteration uses.
    assert module._parity_audit_due({}, 0, league)
    assert module._parity_audit_due({}, 0, self_play)
    # A resumed process has an empty record, so its first iteration audits
    # however far into the run it happens to be.
    assert module._parity_audit_due({}, 137, league)

    audited = {league: 10}
    assert not module._parity_audit_due(audited, 10 + interval - 1, league)
    assert module._parity_audit_due(audited, 10 + interval, league)
    # The league wave being audited says nothing about the self-play-only
    # arena view, which has never been examined and is therefore still due.
    assert module._parity_audit_due(audited, 11, self_play)


def _parity_metrics(module, *, kl: float = 0.0, tail: float = 0.0, head: str = "unit") -> dict:
    """Healthy measurements everywhere, with one head raised to (kl, tail)."""
    metrics: dict[str, float] = {}
    for component in module.PARITY_COMPONENTS:
        raised = component == head
        metrics[f"update_replay_{component}_kl"] = kl if raised else 0.0
        metrics[f"update_replay_{component}_tail_fraction"] = tail if raised else 0.0
        metrics[f"update_replay_{component}_active_count"] = 1
    metrics["update_replay_max_kl"] = kl
    metrics["update_replay_max_tail_fraction"] = tail
    return metrics


def _ceilings(module):
    return module._parity_ceilings()


def test_a_parity_breach_is_a_defect_only_when_it_steps_away_from_the_last_audit() -> None:
    # The gate has to separate two populations that both sit past the bound: a
    # staging defect, which is a step change of orders of magnitude, and
    # numerics drifting upward as RL sharpens the heads, which crosses by a
    # hair. Only the first is worth aborting an otherwise healthy run for.
    module = _training_script()
    bound = module.MAX_UPDATE_REPLAY_KL
    factor = module.REPLAY_PARITY_STEP_CHANGE_FACTOR
    ceilings = _ceilings(module)
    settled = module._parity_measurements(_parity_metrics(module, kl=bound * 0.8))

    # Inside the bound is not reported at all, however much it jumped: the
    # step-change test only ever escalates a value already outside the budget.
    assert module._parity_breaches(_parity_metrics(module, kl=bound), settled, ceilings) == []

    # Past the bound with no history is the launch check, and fatal.
    [(message, is_defect)] = module._parity_breaches(
        _parity_metrics(module, kl=bound * 2.0), None, ceilings
    )
    assert is_defect
    assert "policy divergence exceeded" in message

    drifted = _parity_metrics(module, kl=bound * 0.8 * factor)
    [(message, is_defect)] = module._parity_breaches(drifted, settled, ceilings)
    assert not is_defect
    # The comparison itself is the diagnosis, so it belongs in the message,
    # along with the series to plot rather than a hardcoded one.
    assert f"previous audit {bound * 0.8}" in message
    assert "trend in update_replay_unit_kl" in message
    # Identical warnings say something is off but not whether it is settling or
    # converging on the ceiling, and the distance only means anything in units
    # of the growth producing it. This drift is a clean factor of five, so the
    # ceiling is log(0.025 / 0.02) / log(5) away.
    assert "0.1 audits of headroom" in message

    stepped = _parity_metrics(module, kl=bound * 0.8 * factor * 1.01)
    [(_message, is_defect)] = module._parity_breaches(stepped, settled, ceilings)
    assert is_defect


def test_warned_drift_reports_how_many_audits_of_headroom_are_left() -> None:
    # A warned breach repeats every interval, and identical lines cannot
    # distinguish drift that is settling from drift converging on the ceiling.
    # The remaining distance is only meaningful in units of the growth
    # producing it, so it is reported as audits at the last observed rate --
    # enough to stop at a checkpoint while the run is still healthy, rather
    # than discovering the trajectory once the ceiling has already ended it.
    module = _training_script()
    ceilings = _ceilings(module)
    ceiling = ceilings["kl"]
    bound = module.MAX_UPDATE_REPLAY_KL

    # A realistic 20%-per-audit climb just past the bound: many audits away,
    # and the count is what says so. 1.2^n from 5.5e-3 reaches 0.025 at n = 8.3.
    previous, measured = bound * 1.1 * (1.0 / 1.2), bound * 1.1
    assert measured < module.REPLAY_PARITY_STEP_CHANGE_FACTOR * previous
    baseline = {
        **module._parity_measurements(_parity_metrics(module)),
        "update_replay_unit_kl": previous,
    }
    [(message, is_defect)] = module._parity_breaches(
        _parity_metrics(module, kl=measured), baseline, ceilings
    )
    assert not is_defect
    expected = math.log(ceiling / measured) / math.log(measured / previous)
    assert expected == pytest.approx(8.3, abs=0.05)
    assert f"about {expected:.1f} audits of headroom" in message

    # Drift that has stopped climbing has no countdown to report, and inventing
    # one from a flat or falling pair would read as a prediction of collapse.
    baseline["update_replay_unit_kl"] = measured
    [(message, is_defect)] = module._parity_breaches(
        _parity_metrics(module, kl=measured), baseline, ceilings
    )
    assert not is_defect
    assert "not climbing toward the ceiling" in message

    # A defect is not a countdown: the run is over, so the abort names the
    # recovery instead.
    [(message, is_defect)] = module._parity_breaches(
        _parity_metrics(module, kl=ceiling * 1.01), baseline, ceilings
    )
    assert is_defect
    assert "headroom" not in message


def test_a_single_head_defect_is_judged_against_that_head_not_the_largest() -> None:
    # The gated statistic used to be the max over heads, which exists so a
    # single-head defect is not diluted by the unit head's 1.45M components.
    # Baselining that aggregate would hand the dilution straight back: measured
    # healthy KL is 1.9e-3 on the unit head against 8.0e-4 on the kind head, so
    # a kind defect judged against the aggregate is excused to 11.9x its own
    # healthy level rather than the 5x intended.
    module = _training_script()
    ceilings = _ceilings(module)
    baseline = {
        **module._parity_measurements(_parity_metrics(module, kl=1.9e-3, head="unit")),
        "update_replay_kind_kl": 8.0e-4,
    }
    measured = _parity_metrics(module, kl=8.0e-3, head="kind")

    [(message, is_defect)] = module._parity_breaches(measured, baseline, ceilings)

    # 8e-3 is under 5 x the unit head's 1.9e-3, so aggregate baselining would
    # have called this drift; against kind's own 8e-4 it is a 10x step.
    assert module.REPLAY_PARITY_STEP_CHANGE_FACTOR * 1.9e-3 > 8.0e-3
    assert is_defect
    assert message.startswith("kind ")


def test_gradual_growth_is_still_a_defect_once_it_passes_the_ceiling() -> None:
    # The step-change test is a derivative, so on its own it says nothing about
    # level: a value growing by less than the factor per audit is never fatal at
    # any magnitude, and the baseline advances after every warned audit, so the
    # accepted level would ratchet without limit.
    module = _training_script()
    ceilings = _ceilings(module)
    assert ceilings["kl"] == pytest.approx(0.025)

    # Just under the ceiling, and a modest step from the last audit: drift.
    baseline = module._parity_measurements(_parity_metrics(module, kl=0.02))
    [(_message, is_defect)] = module._parity_breaches(
        _parity_metrics(module, kl=0.024), baseline, ceilings
    )
    assert not is_defect

    # Past it, by a step far too small to trip the factor: still a defect.
    [(_message, is_defect)] = module._parity_breaches(
        _parity_metrics(module, kl=0.026), baseline, ceilings
    )
    assert is_defect

    # The ceiling is one full step change past the calibrated bound, stated in
    # the gate's own terms rather than imported from the update's trust region:
    # target_kl is a safety valve the update never reaches, so a ceiling there
    # would sit an order of magnitude above the movement it claims to match.
    # Deriving it from the bound also keeps ceiling >= bound true by
    # construction, which _validate_parity_baseline's invariant depends on.
    ceilings = module._parity_ceilings()
    for statistic, bound, _description in module.PARITY_STATISTICS:
        assert ceilings[statistic] == pytest.approx(module.REPLAY_PARITY_STEP_CHANGE_FACTOR * bound)
        assert ceilings[statistic] > bound


def test_a_non_finite_measurement_is_a_defect_rather_than_drift() -> None:
    # NaN fails every ordered comparison, so an implementation that only asks
    # "did it exceed the bound" and "did it step" reads NaN as drift, warns,
    # and writes NaN into the baseline, poisoning every later comparison.
    module = _training_script()
    ceilings = _ceilings(module)
    baseline = module._parity_measurements(_parity_metrics(module, kl=1.9e-3))

    for value in (float("nan"), float("inf")):
        [(_message, is_defect)] = module._parity_breaches(
            _parity_metrics(module, kl=value), baseline, ceilings
        )
        assert is_defect


def test_the_reported_abort_line_is_the_step_change_not_the_bound() -> None:
    # "MAX_" reads as a ceiling, and in a running process it is not one: the
    # abort line is the step change above the last audit, capped by the
    # absolute ceiling. A head whose bound sits less than the factor above its
    # own baseline therefore has a band in which a breach only warns, and that
    # band has to be legible in telemetry rather than re-derived by whoever is
    # reading the trend.
    module = _training_script()
    bound = module.MAX_UPDATE_REPLAY_KL
    factor = module.REPLAY_PARITY_STEP_CHANGE_FACTOR
    ceilings = _ceilings(module)

    # No history: the bound is the abort line, which is the launch check.
    fresh = module._parity_fatal_thresholds(None, ceilings)
    assert fresh["update_replay_unit_kl_fatal_at"] == bound
    assert fresh["update_replay_quantity_tail_fraction_fatal_at"] == (
        module.MAX_UPDATE_REPLAY_TAIL_FRACTION
    )

    # A baseline far enough below the bound leaves the bound binding, so the
    # gate aborts on any breach at all -- the tail gate's normal condition.
    tight = module._parity_measurements(_parity_metrics(module, kl=bound / (factor * 2.0)))
    assert module._parity_fatal_thresholds(tight, ceilings)["update_replay_unit_kl_fatal_at"] == (
        bound
    )

    # The KL gate's normal condition is the other one: the clone measures
    # 1.9e-3 against a 5e-3 bound, a ratio of 2.63, so the abort line floats
    # above the bound and widens as the baseline drifts up -- until the
    # ceiling, which it never floats past.
    for measured, expected in ((1.9e-3, factor * 1.9e-3), (4.0e-3, factor * 4.0e-3)):
        thresholds = module._parity_fatal_thresholds(
            module._parity_measurements(_parity_metrics(module, kl=measured)), ceilings
        )
        assert thresholds["update_replay_unit_kl_fatal_at"] == expected > bound
    capped = module._parity_fatal_thresholds(
        module._parity_measurements(_parity_metrics(module, kl=0.02)), ceilings
    )
    assert capped["update_replay_unit_kl_fatal_at"] == ceilings["kl"] < factor * 0.02


def test_a_resumed_parity_baseline_must_be_complete_and_within_the_ceilings() -> None:
    # A malformed baseline read as "no history" would make every breach after
    # a resume fatal again, which is the wedge this state exists to remove, so
    # anything short of a complete record is refused instead. And a value above
    # a ceiling can never have been persisted -- the audit producing it aborts
    # before any checkpoint is written -- so one in a checkpoint is corruption,
    # and accepting it would excuse every breach below five times it.
    module = _training_script()
    ceilings = _ceilings(module)
    complete = module._parity_measurements(_parity_metrics(module, kl=1.9e-3, tail=3.2e-5))
    assert module._validate_parity_baseline(complete, ceilings) == complete
    # An empty record is a run that has not audited yet, not a malformed one.
    assert module._validate_parity_baseline({}, ceilings) == {}

    for invalid, expected in (
        (None, "no valid replay-parity baseline"),
        ({key: 1.0 for key in list(complete)[:-1]}, "baseline is incomplete"),
        ({**complete, "update_replay_unit_kl": -1.0}, "invalid replay-parity"),
        ({**complete, "update_replay_unit_kl": float("nan")}, "invalid replay-parity"),
        ({**complete, "update_replay_unit_kl": 1}, "invalid replay-parity"),
        # Above the ceiling the live gate would have aborted rather than saved.
        ({**complete, "update_replay_unit_kl": ceilings["kl"] * 1.01}, "invalid replay-parity"),
        (
            {**complete, "update_replay_kind_tail_fraction": ceilings["tail_fraction"] * 1.01},
            "invalid replay-parity",
        ),
    ):
        with pytest.raises(ValueError, match=expected):
            module._validate_parity_baseline(invalid, ceilings)


def test_replay_parity_is_re_audited_on_a_cadence_and_on_every_resume(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    # The audited divergence grows as the heads sharpen, so an audit that ran
    # only at iteration zero would measure it where it is smallest and never
    # look again. A resume must also audit immediately rather than wait out
    # the cadence, because it restages every buffer and rebuilds the compiled
    # callables the audit exists to check.
    module = _training_script()

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    settled = module.MAX_UPDATE_REPLAY_KL * 0.8
    audited: list[int] = []

    def parity(*args, **kwargs) -> dict[str, float]:
        audited.append(len(audited))
        return _parity_metrics(module, kl=settled)

    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(
        module,
        "collect_mixed_play_rust",
        lambda *args, **kwargs: SimpleNamespace(state_count=1),
    )
    monkeypatch.setattr(module, "slice_trajectories", lambda batch, start, stop: batch)
    monkeypatch.setattr(module, "rollout_diagnostics", lambda batch: {})
    monkeypatch.setattr(module, "update_replay_parity", parity)
    monkeypatch.setattr(
        module,
        "update_ppo",
        lambda *args, **kwargs: {
            "actor_updates": 1,
            "actor_minibatches_intended": 1,
            "critic_updates": 1,
            "first_minibatch_approx_kl": 0.0,
            "value_target_saturated_fraction": 0.0,
            "entropy": 0.2,
        },
    )
    monkeypatch.setattr(module, "REPLAY_PARITY_AUDIT_INTERVAL", 3)

    def arguments(run_dir: Path, iterations: int, resume: Path | None = None) -> list[str]:
        values = [
            "train_ppo.py",
            "--run-dir",
            str(run_dir),
            "--iterations",
            str(iterations),
            "--games",
            "1",
            "--league-games",
            "0",
            "--device",
            "cpu",
            "--cnn-width",
            "8",
            "--cnn-blocks",
            "1",
            "--checkpoint-every",
            "8",
            "--no-bfloat16",
        ]
        if resume is not None:
            values.extend(("--resume", str(resume)))
        return values

    run_dir = tmp_path / "cadence"
    monkeypatch.setattr(sys, "argv", arguments(run_dir, 7))
    module.main()
    # Iterations 0, 3 and 6 of seven.
    assert len(audited) == 3

    monkeypatch.setattr(sys, "argv", arguments(run_dir, 8, run_dir / "latest.pt"))
    module.main()
    # One further iteration, and it audits even though the cadence would not
    # have come round again until iteration nine.
    assert len(audited) == 4

    # Every per-head measurement is checkpointed, because that is what the
    # next audit is judged against -- per head, so a defect on one is not
    # excused by the largest head's level.
    expected_baseline = module._parity_measurements(_parity_metrics(module, kl=settled))
    assert set(expected_baseline) == {
        f"update_replay_{component}_{statistic}"
        for component in module.PARITY_COMPONENTS
        for statistic, _bound, _description in module.PARITY_STATISTICS
    }
    for name in ("latest.pt", "checkpoint-000000.pt"):
        checkpoint = torch.load(run_dir / name, weights_only=False)
        # checkpoint-000000.pt predates the first audit, so it carries an empty
        # record rather than a missing key -- a missing one is unresumable.
        assert checkpoint["replay_parity_baseline"] == (
            expected_baseline if name == "latest.pt" else {}
        )

    # Drift: past the bound but within a step change of what the same
    # configuration last measured. The divergence grows as RL sharpens the
    # heads and the bound was calibrated at the run's starting sharpness, so
    # killing here would destroy a healthy run over expected numerics -- and
    # unrecoverably, since the failure precedes the update.
    drifted = settled * module.REPLAY_PARITY_STEP_CHANGE_FACTOR
    assert drifted > module.MAX_UPDATE_REPLAY_KL
    monkeypatch.setattr(
        module, "update_replay_parity", lambda *args, **kwargs: _parity_metrics(module, kl=drifted)
    )
    monkeypatch.setattr(sys, "argv", arguments(run_dir, 12, run_dir / "latest.pt"))
    module.main()
    assert (run_dir / "latest.pt").is_file()
    warning = capsys.readouterr().err
    assert "unit sampling-vs-update policy divergence exceeded" in warning
    assert "reads as drift rather than a defect" in warning
    # The series named for the trend is the breached one, not a hardcoded one.
    assert "trend in update_replay_unit_kl" in warning
    # And it reaches the journal, where the trend is the useful artifact.
    journal = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    breached = [record for record in journal if record.get("replay_parity_breached")]
    assert len(breached) == 2
    assert [record["update_replay_max_kl"] for record in breached] == [drifted, drifted]
    # Next to each value, the line it was actually judged against -- and the
    # two differ, because a warned breach advances the baseline and the abort
    # line is a multiple of it. That widening is the price of drift tolerance
    # and the reason the line is telemetry rather than a constant to look up.
    # The first warned breach lifts the abort line to five times the previous
    # audit; the second would lift it to five times the drifted value, but the
    # ceiling caps it there. That cap is what stops warned drift ratcheting the
    # accepted level without limit.
    ceiling = module._parity_ceilings()["kl"]
    assert module.REPLAY_PARITY_STEP_CHANGE_FACTOR * drifted > ceiling
    assert [record["update_replay_unit_kl_fatal_at"] for record in breached] == [
        pytest.approx(module.REPLAY_PARITY_STEP_CHANGE_FACTOR * settled),
        pytest.approx(ceiling),
    ]
    # A head that measured nothing keeps the bound as its abort line, which is
    # the whole point of baselining per head rather than on the aggregate.
    assert breached[-1]["update_replay_kind_kl_fatal_at"] == module.MAX_UPDATE_REPLAY_KL
    # The warning names the journal row it appears in, not the loop counter.
    assert "iteration " + str(breached[0]["iteration"]) in warning

    # Resuming that drifted run must not turn the same measurement fatal.
    # --max-hours makes chunked restarts the designed operating mode, so a
    # process-scoped baseline would abort every restart after the first drift.
    monkeypatch.setattr(sys, "argv", arguments(run_dir, 13, run_dir / "latest.pt"))
    module.main()
    assert torch.load(run_dir / "latest.pt", weights_only=False)["iteration"] == 13

    # A step change away from that same drifted baseline is a defect, and
    # still aborts however deep into the run it appears.
    monkeypatch.setattr(
        module,
        "update_replay_parity",
        lambda *args, **kwargs: _parity_metrics(module, kl=drifted * 100.0),
    )
    monkeypatch.setattr(sys, "argv", arguments(run_dir, 14, run_dir / "latest.pt"))
    with pytest.raises(RuntimeError, match="policy divergence exceeded"):
        module.main()

    # A fresh run has nothing to compare against, so its first audit is the
    # launch check and a breach there is fatal.
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(sys, "argv", arguments(fresh, 7))
    with pytest.raises(RuntimeError, match="policy divergence exceeded"):
        module.main()

    # The tail statistic is gated through the same path, on its own head and
    # against its own baseline, so a run whose mean KL is healthy still dies on
    # a materially divergent share appearing where there was none.
    tail = tmp_path / "tail"
    monkeypatch.setattr(
        module,
        "update_replay_parity",
        lambda *args, **kwargs: _parity_metrics(
            module, tail=module.MAX_UPDATE_REPLAY_TAIL_FRACTION * 2.0, head="quantity"
        ),
    )
    monkeypatch.setattr(sys, "argv", arguments(tail, 7))
    with pytest.raises(RuntimeError, match="quantity sampling-vs-update materially divergent"):
        module.main()


def test_warm_start_flags_validate_freshness_and_sign(monkeypatch, tmp_path) -> None:
    module = _training_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--run-dir",
            str(tmp_path),
            "--init-actor-from",
            str(tmp_path / "bc-actor.pt"),
            "--critic-warmup-iterations",
            "15",
        ],
    )
    args = module.parse_args()
    module._validate_args(args)
    assert args.critic_warmup_iterations == 15

    args.critic_warmup_iterations = -1
    with pytest.raises(ValueError, match="warmup iterations"):
        module._validate_args(args)

    # A warmup that spans the run freezes the actor for its whole life, and
    # the stalled-actor guard is suppressed for exactly those iterations, so
    # nothing downstream would notice the policy never moved.
    args.critic_warmup_iterations = args.iterations
    with pytest.raises(ValueError, match="leave iterations for the actor"):
        module._validate_args(args)

    args.critic_warmup_iterations = 0
    args.resume = tmp_path / "latest.pt"
    with pytest.raises(ValueError, match="fresh run"):
        module._validate_args(args)


def test_critic_warmup_cannot_be_restated_on_a_resume(monkeypatch, tmp_path) -> None:
    """The count is persisted with the warm start, so a relaunch must not be
    able to supply a different one -- and must not be able to supply none.
    A crash inside the warmup window otherwise resumes with no warmup, and the
    actor starts stepping against a critic that never finished fitting."""
    module = _training_script()

    def parsed(*flags: str):
        monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path), *flags])
        return module.parse_args()

    unflagged = parsed()
    module._validate_args(unflagged)
    assert unflagged.critic_warmup_iterations is None

    restated = parsed("--resume", str(tmp_path / "latest.pt"), "--critic-warmup-iterations", "15")
    with pytest.raises(ValueError, match="restored from its checkpoint"):
        module._validate_args(restated)

    orphaned = parsed("--critic-warmup-iterations", "15")
    with pytest.raises(ValueError, match="only to a warm-started run"):
        module._validate_args(orphaned)


def test_warm_start_record_carries_the_warmup_count_and_clone_source(tmp_path) -> None:
    """The warm-start record is the channel that survives a resume, so it has
    to carry both the count the run must keep honoring and the tree that
    tokenized the demonstrations."""
    from kaggriculture.inference import ACTOR_ARTIFACT_FORMAT_VERSION
    from kaggriculture.ppo import PpoConfig
    from kaggriculture.provenance import source_identity
    from kaggriculture.training import CHECKPOINT_FORMAT_VERSION, checkpoint_payload

    module = _training_script()
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    identity = source_identity()
    artifact = tmp_path / "bc-actor.pt"
    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "architecture": CONV_ENTITY,
            "model_config": config.to_dict(),
            "actor": FarmActor(config).state_dict(),
            "iteration": 0,
            "metrics": {},
            "source_identity": identity,
            "run_provenance": None,
            "bc_provenance": {"teacher": {"label": "public-v27"}},
        },
        artifact,
    )

    record = module._load_initial_actor(
        artifact, FarmActor(config), CONV_ENTITY, config, torch.device("cpu")
    )
    record["critic_warmup_iterations"] = 15

    payload = checkpoint_payload(
        agents=[
            {
                "actor": {},
                "critic": {},
                "actor_optimizer": {},
                "critic_optimizer": {},
            }
        ],
        model_config=config,
        ppo_config=PpoConfig(epochs=1, minibatch_size=4, use_bfloat16=False),
        iteration=3,
        next_seed=11,
        metrics={},
        source_identity=identity,
        rng_states={"torch_rng": None, "cuda_rng": None, "numpy_rng": None, "python_rng": None},
        initial_actor=record,
    )

    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["initial_actor"]["critic_warmup_iterations"] == 15
    assert payload["initial_actor"]["source_identity"] == identity
    # This is what the resume branch reads; iteration 3 of a 15-iteration
    # warmup must still be inside it.
    restored = int(payload["initial_actor"].get("critic_warmup_iterations", 0))
    assert payload["iteration"] < restored


def test_initial_actor_loads_pretrained_weights_and_binds_provenance(tmp_path) -> None:
    from kaggriculture.inference import ACTOR_ARTIFACT_FORMAT_VERSION
    from kaggriculture.provenance import source_identity

    module = _training_script()
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    pretrained = FarmActor(config)
    artifact = tmp_path / "bc-actor.pt"
    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": pretrained.state_dict(),
            "iteration": 0,
            "metrics": {},
            "source_identity": source_identity(),
            "run_provenance": None,
            "bc_provenance": {"teacher": {"label": "public-v27"}},
        },
        artifact,
    )
    actor = FarmActor(config)

    provenance = module._load_initial_actor(
        artifact, actor, CONV_ENTITY, config, torch.device("cpu")
    )

    assert all(
        torch.equal(value, pretrained.state_dict()[name])
        for name, value in actor.state_dict().items()
    )
    assert provenance["bc_provenance"]["teacher"]["label"] == "public-v27"
    assert len(provenance["sha256"]) == 64

    other = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    with pytest.raises(ValueError, match="model configuration"):
        module._load_initial_actor(
            artifact, FarmActor(other), CONV_ENTITY, other, torch.device("cpu")
        )


def test_update_gates_stop_the_run_before_the_next_iteration_is_wasted() -> None:
    """Each gate is a different reason the run cannot recover on its own.

    The saturation bound in particular cannot be read off its own description: a
    categorical critic's mean is bounded by its support, so a critic collapsed
    onto the outermost atom saturates about 0.37 of the batch and never more.
    A bound set anywhere at or above that would never fire.
    """
    from kaggriculture.ppo import (
        MAX_FIRST_MINIBATCH_KL,
        MAX_VALUE_TARGET_SATURATED_FRACTION,
        MINIMUM_ACTOR_EPOCH_FRACTION,
        MINIMUM_POLICY_ENTROPY,
    )

    module = _training_script()
    healthy = {
        "first_minibatch_approx_kl": MAX_FIRST_MINIBATCH_KL,
        "value_target_saturated_fraction": MAX_VALUE_TARGET_SATURATED_FRACTION,
        "actor_updates": 113,
        "actor_minibatches_intended": 113,
        "max_approx_kl": 0.02,
        "kl_early_stop": 0,
        # Inside the band every rate that learned measured, 0.14 to 0.37 nats.
        "entropy": 0.2,
    }

    module._gate_update_metrics(healthy, warmup_active=False)
    module._gate_update_metrics({**healthy, "actor_updates": 0}, warmup_active=True)

    with pytest.raises(RuntimeError, match="first-minibatch KL"):
        module._gate_update_metrics(
            {**healthy, "first_minibatch_approx_kl": MAX_FIRST_MINIBATCH_KL * 1.01},
            warmup_active=False,
        )
    with pytest.raises(RuntimeError, match="saturated the critic support"):
        module._gate_update_metrics(
            {**healthy, "value_target_saturated_fraction": 0.374},
            warmup_active=False,
        )
    # A saturated target starves the actor of an advantage, so the saturation
    # gate must report first or the run blames the missing update.
    with pytest.raises(RuntimeError, match="saturated the critic support"):
        module._gate_update_metrics(
            {**healthy, "value_target_saturated_fraction": 0.374, "actor_updates": 0},
            warmup_active=False,
        )
    with pytest.raises(RuntimeError, match="without an actor update"):
        module._gate_update_metrics({**healthy, "actor_updates": 0}, warmup_active=False)
    # The hole this closes: a trust region that latches after the first
    # minibatch reports one update, not zero, so every gate above passes while
    # the iteration trains on 0.9% of the wave. That is the exact shape of the
    # run this gate was added for -- 1 of 113 with `kl_early_stop` set.
    with pytest.raises(RuntimeError, match="of the epoch"):
        module._gate_update_metrics(
            {**healthy, "actor_updates": 1, "kl_early_stop": 1, "max_approx_kl": 0.0857},
            warmup_active=False,
        )
    # Warmup runs no actor at all, so the fraction cannot speak there.
    module._gate_update_metrics(
        {**healthy, "actor_updates": 0, "kl_early_stop": 1}, warmup_active=True
    )
    # An early stop that still applied most of the epoch is the safety valve
    # working, and must not stop a run that is making progress. The shipped
    # schedule's own worst wave applies 31%, so this is not a hypothetical band.
    module._gate_update_metrics(
        {**healthy, "actor_updates": 96, "kl_early_stop": 1}, warmup_active=False
    )
    module._gate_update_metrics(
        {**healthy, "actor_updates": 35, "kl_early_stop": 1}, warmup_active=False
    )
    # The comparator's own boundary, and the only place a `<` relaxed to `<=`
    # would show up. Read from the constant so raising the floor cannot silently
    # turn this into an assertion about somewhere else on the scale.
    boundary = round(MINIMUM_ACTOR_EPOCH_FRACTION * 100)
    module._gate_update_metrics(
        {
            **healthy,
            "actor_minibatches_intended": 100,
            "actor_updates": boundary,
            "kl_early_stop": 1,
        },
        warmup_active=False,
    )
    with pytest.raises(RuntimeError, match="of the epoch"):
        module._gate_update_metrics(
            {
                **healthy,
                "actor_minibatches_intended": 100,
                "actor_updates": boundary - 1,
                "kl_early_stop": 1,
            },
            warmup_active=False,
        )
    # The second failure mode a learning-rate sweep found: at 1e-4 the actor
    # converges onto passing every turn. Every number above reads healthy --
    # the epoch completes 113 of 113 precisely because a deterministic policy
    # has no KL movement to bound, and money rises to the untouched starting
    # bank -- so entropy is the only place it can be caught.
    with pytest.raises(RuntimeError, match=r"is below 0\.01"):
        module._gate_update_metrics({**healthy, "entropy": 0.0}, warmup_active=False)
    with pytest.raises(RuntimeError, match="no sampled alternative"):
        module._gate_update_metrics(
            {**healthy, "entropy": MINIMUM_POLICY_ENTROPY * 0.99}, warmup_active=False
        )
    # The floor itself is admissible, and a warmup iteration cannot speak for a
    # policy that has not been updated yet.
    module._gate_update_metrics({**healthy, "entropy": MINIMUM_POLICY_ENTROPY}, warmup_active=False)
    module._gate_update_metrics({**healthy, "entropy": 0.0}, warmup_active=True)


def test_the_entropy_floor_admits_a_clone_that_starts_sharper_than_the_absolute_level() -> None:
    """A faithful clone begins below the from-scratch floor and still plays well.

    Measured, not hypothetical: the v16-KL clones read 0.00896 nats at their
    first actor-active iteration while holding 0.001 holdout NLL and beating
    `public-v27` in the official engine. An absolute floor calibrated on
    from-scratch entropy (0.14-0.37 healthy, 0.000-0.001 collapsed) refused them
    on their first update, so the floor is the lower of that level and a share of
    the run's own start.
    """
    from kaggriculture.ppo import (
        MINIMUM_POLICY_ENTROPY,
        MINIMUM_POLICY_ENTROPY_REFERENCE,
        POLICY_ENTROPY_FLOOR_FRACTION,
    )

    module = _training_script()
    clone = 0.00896
    # The reference only ever relaxes the floor: a from-scratch run whose quarter
    # is looser than the absolute level is judged against the level, exactly as
    # before this existed. `min` and not `max`, because a policy sharpening as it
    # converges is the expected trajectory rather than a failure.
    assert module._policy_entropy_floor(None) == MINIMUM_POLICY_ENTROPY
    assert module._policy_entropy_floor(0.29) == MINIMUM_POLICY_ENTROPY
    assert module._policy_entropy_floor(clone) == POLICY_ENTROPY_FLOOR_FRACTION * clone

    healthy = {
        "first_minibatch_approx_kl": 0.0,
        "value_target_saturated_fraction": 0.0,
        "actor_updates": 113,
        "actor_minibatches_intended": 113,
        "max_approx_kl": 0.0,
        "entropy": clone,
    }
    # The clone's own first update is admissible; four times sharper is not.
    module._gate_update_metrics(healthy, warmup_active=False, entropy_reference=clone)
    with pytest.raises(RuntimeError, match="no sampled alternative"):
        module._gate_update_metrics(
            {**healthy, "entropy": POLICY_ENTROPY_FLOOR_FRACTION * clone * 0.99},
            warmup_active=False,
            entropy_reference=clone,
        )
    # A run cannot start collapsed. Admitting such a reference would set a floor
    # below the collapse and switch the gate off for every later iteration.
    with pytest.raises(ValueError, match="collapsed range"):
        module._validate_policy_entropy_reference(MINIMUM_POLICY_ENTROPY_REFERENCE * 0.99)
    assert (
        module._validate_policy_entropy_reference(MINIMUM_POLICY_ENTROPY_REFERENCE)
        == MINIMUM_POLICY_ENTROPY_REFERENCE
    )


def test_the_entropy_reference_is_persisted_per_population_member() -> None:
    """A chunked run must not recalibrate its floor against its own sharpening.

    `--max-hours` makes restarts the designed operating mode, so a reference
    re-measured on resume would ratchet the floor down every few hours until it
    admitted a collapsed policy. Per member because members warm-started from
    differently trained artifacts arrive at different sharpnesses.
    """
    module = _training_script()
    # A single learner's record stays the bare scalar every checkpoint carried.
    assert module._entropy_reference_record([0.29], 1) == 0.29
    assert module._validate_entropy_references(0.29, population=1) == [0.29]
    # Written before the warmup ends, no member has stepped and there is nothing
    # to preserve: a resume measures its own.
    assert module._entropy_reference_record([None], 1) is None
    assert module._validate_entropy_references(None, population=1) == [None]

    references = [0.00896, None, 0.29]
    record = module._entropy_reference_record(references, 3)
    assert record == references
    assert module._validate_entropy_references(record, population=3) == references
    with pytest.raises(ValueError, match="every population member"):
        module._validate_entropy_references(record, population=4)
    with pytest.raises(ValueError, match="collapsed range"):
        module._validate_entropy_references([0.0, None, 0.29], population=3)


def _population_wave(module, *, games: int, population: int, steps: int = 2, seed: int = 0):
    """A synthetic population wave with the row layout the collector's contract fixes.

    Row order is game-major and seat-minor, so `agents` is the pairing table read
    flat; the loop's partition, its head-to-head table and its sibling-row opponent
    lookup all depend on exactly that.
    """
    from kaggriculture.rollout import _SHARED_ROLLOUT_FIELDS, allocate_rollout_storage

    rows = games * 2
    storage = allocate_rollout_storage(CONV_ENTITY, rows, steps)
    generator = np.random.default_rng(seed)
    for name in ("board", "global_features", "critic_features", "units"):
        storage[name][:] = generator.standard_normal(storage[name].shape)
    for name in ("unit_masks", "unit_active", "market_active", "market_quantity_active", "valid"):
        storage[name][:] = True
    storage["market_kind_masks"][:] = True
    storage["market_quantity_masks"][:] = True
    money = generator.uniform(0.0, 100.0, rows)
    return SimpleNamespace(
        architecture=CONV_ENTITY,
        states={
            name: storage[name]
            for name in ("board", "global_features", "critic_features", "units", "unit_positions")
        },
        **{name: storage[name] for name in _SHARED_ROLLOUT_FIELDS},
        agents=population_pairings(population, games).reshape(-1),
        seats=np.arange(rows, dtype=np.int64) % 2,
        episode_seeds=np.arange(rows, dtype=np.int64) // 2,
        final_money=money,
        opponent_money=money[np.arange(rows) ^ 1],
        entropy_sums=np.full(rows, 1.0, dtype=np.float64),
        elapsed_seconds=1.0,
        state_count=rows * steps,
        trajectories=rows,
        horizon=steps,
        mean_entropy=1.0,
    )


def _distinct_actors(config: ModelConfig, count: int, *, seed: int = 5) -> list[FarmActor]:
    """`count` actors that genuinely run different programs.

    Fresh initializations do not: this architecture's unit logits carry a large
    fixed action prior that dominates an untrained head, so cold-started actors
    pick the same greedy action everywhere and are one policy for the purpose the
    gate measures. Displacing each head's bias is what four differently trained
    members differ by, expressed in one line.
    """
    torch.manual_seed(seed)
    actors = [FarmActor(config) for _ in range(count)]
    with torch.no_grad():
        for actor in actors:
            actor.unit_head[1].bias.add_(torch.randn(actor.unit_head[1].bias.shape) * 3.0)
    return actors


def _actor_artifact(path: Path, actor: FarmActor, config: ModelConfig) -> Path:
    """One exported actor artifact of the shape a BC clone writes."""
    from kaggriculture.inference import ACTOR_ARTIFACT_FORMAT_VERSION
    from kaggriculture.provenance import source_identity

    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "architecture": CONV_ENTITY,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 0,
            "source_identity": source_identity(),
        },
        path,
    )
    return path


_TINY_CONFIG = ModelConfig(
    cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
)


def _population_arguments(run_dir: Path, *, population: int, games: int, iterations: int = 1):
    return [
        "train_ppo.py",
        "--run-dir",
        str(run_dir),
        "--iterations",
        str(iterations),
        "--population",
        str(population),
        "--games",
        str(games),
        "--league-games",
        "0",
        "--league-builtin-opponents",
        "",
        "--league-builtin-lanes",
        "0",
        "--external-eval-every",
        "0",
        "--device",
        "cpu",
        "--cnn-width",
        "8",
        "--cnn-blocks",
        "1",
        "--model-dim",
        "16",
        "--transformer-layers",
        "3",
        "--attention-heads",
        "2",
        "--checkpoint-every",
        "1",
        "--no-bfloat16",
    ]


def _run_population_main(
    module,
    monkeypatch,
    run_dir: Path,
    *,
    population: int,
    games: int,
    iterations: int = 1,
    initial_actors: tuple[Path, ...] = (),
    resume: Path | None = None,
):
    """Run one iteration of the loop with only the wave and the update mocked out.

    The partition, the disagreement measurement, its gate, the per-member update
    gates and the checkpoint payload all run for real; substituting the collector
    is what makes that possible on a CPU in a test.
    """

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    seen: list[np.ndarray | None] = []

    def update(*args, rows=None, **kwargs):
        seen.append(rows)
        return {
            "actor_updates": 1,
            "actor_minibatches_intended": 1,
            "critic_updates": 1,
            "first_minibatch_approx_kl": 0.0,
            "value_target_saturated_fraction": 0.0,
            "entropy": 0.2,
        }

    wave = _population_wave(module, games=games, population=max(population, 2))
    if population == 1:
        wave.agents = np.zeros(wave.agents.size, dtype=np.int64)
    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(module, "collect_population_play_rust", lambda *a, **k: wave)
    monkeypatch.setattr(module, "collect_mixed_play_rust", lambda *a, **k: wave)
    monkeypatch.setattr(module, "slice_trajectories", lambda batch, start, stop: batch)
    monkeypatch.setattr(module, "rollout_diagnostics", lambda batch: {})
    monkeypatch.setattr(module, "update_replay_parity", lambda *a, **k: _parity_metrics(module))
    monkeypatch.setattr(module, "update_ppo", update)
    arguments = _population_arguments(
        run_dir, population=population, games=games, iterations=iterations
    )
    for artifact in initial_actors:
        arguments.extend(("--init-actor-from", str(artifact)))
    if resume is not None:
        arguments.extend(("--resume", str(resume)))
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    record = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    return record, seen


def test_a_single_learner_run_keeps_todays_metric_layout(monkeypatch, tmp_path) -> None:
    """`--population 1` is the path every shipped run and every downstream reader
    already uses, so it must reshape nothing: the bare metric names stay, no
    per-agent or population name appears, and the update sees the whole wave."""
    module = _training_script()

    record, seen = _run_population_main(
        module, monkeypatch, tmp_path / "single", population=1, games=2
    )

    assert seen == [None]
    assert record["entropy"] == 0.2
    assert not [name for name in record if name.startswith("population")]
    assert not [name for name in record if name.startswith("agent")]


def test_a_population_partitions_its_wave_across_the_members_exactly_once() -> None:
    """Every row belongs to exactly one member, and the union is the whole wave --
    otherwise a member trains on another's trajectories or the wave loses rows to
    no update at all, and both read as an ordinary run."""
    module = _training_script()
    population = 4
    games = population * (population - 1)
    agents = population_pairings(population, games).reshape(-1)

    rows = module._population_row_partition(agents, population)

    assert len(rows) == population
    union = np.concatenate(rows)
    assert sorted(union.tolist()) == list(range(agents.size))
    assert len(set(union.tolist())) == union.size
    # Each member holds one seat of every game it plays and no member's rows are
    # contiguous, which is why the update takes indices rather than a sub-batch.
    assert {index.size for index in rows} == {2 * (population - 1)}
    assert any(np.diff(index).max() > 1 for index in rows)

    with pytest.raises(ValueError, match="not covered by agents"):
        module._population_row_partition(np.array([0, 1, 2, 2], dtype=np.int64), 2)


def test_the_disagreement_gate_separates_converged_members_from_distinct_ones() -> None:
    """The failure this exists for reads healthy everywhere else: converged members
    score 0.5 against each other by symmetry, so entropy, KL, epoch fraction and
    the money curve all stay in bounds while the wave carries no gradient."""
    module = _training_script()
    wave = _population_wave(module, games=12, population=4, seed=3)
    forward_args, masks, active, codes = module._population_state_sample(
        wave, np.random.default_rng(0), torch.device("cpu")
    )

    distinct = _distinct_actors(_TINY_CONFIG, 4)
    identical = _distinct_actors(_TINY_CONFIG, 4)
    for member in identical[1:]:
        member.load_state_dict(identical[0].state_dict())

    reference = module.mean_off_diagonal(
        module._population_disagreement(distinct, forward_args, masks, active, codes)
    )
    collapsed = module.mean_off_diagonal(
        module._population_disagreement(identical, forward_args, masks, active, codes)
    )

    # Members running different programs disagree on most decisions; copies of one
    # set of weights run the same program and disagree on none.
    assert reference > 0.5
    assert collapsed == 0.0
    module._gate_population_disagreement(reference, reference)
    with pytest.raises(RuntimeError, match="converged into each other"):
        module._gate_population_disagreement(collapsed, reference)
    # The floor is a share of the recorded start, so a run may lose most of its
    # diversity before the gate calls it collapse.
    floor = module.POPULATION_DISAGREEMENT_FLOOR_FRACTION * reference
    module._gate_population_disagreement(floor, reference)
    with pytest.raises(RuntimeError, match="converged into each other"):
        module._gate_population_disagreement(floor * 0.99, reference)
    # A population that agreed everywhere at iteration 0 would make the floor a
    # share of zero, which admits everything: the gate refuses to be switched off.
    with pytest.raises(RuntimeError, match="agree on every sampled decision"):
        module._gate_population_disagreement(0.0, 0.0)


def test_members_starting_from_the_same_weights_are_rejected(monkeypatch, tmp_path) -> None:
    """Four agents built from one checkpoint are numerically identical, which makes
    every one of their first games a mirror scoring 0.5, so the run would train on
    a wave with no gradient in it. Both spellings of that mistake must fail: the
    same path twice, and two paths holding the same weights."""
    module = _training_script()
    artifact = _actor_artifact(
        tmp_path / "bc-actor.pt", _distinct_actors(_TINY_CONFIG, 1)[0], _TINY_CONFIG
    )
    twin = tmp_path / "bc-actor-copy.pt"
    twin.write_bytes(artifact.read_bytes())

    repeated = _population_arguments(tmp_path / "repeated", population=2, games=2)
    repeated.extend(("--init-actor-from", str(artifact), "--init-actor-from", str(artifact)))
    monkeypatch.setattr(sys, "argv", repeated)
    with pytest.raises(ValueError, match="different artifact per agent"):
        module.main()

    # Distinct paths, so validation cannot see it: only the artifacts' digests can.
    copied = _population_arguments(tmp_path / "copied", population=2, games=2)
    copied.extend(("--init-actor-from", str(artifact), "--init-actor-from", str(twin)))
    monkeypatch.setattr(sys, "argv", copied)
    with pytest.raises(ValueError, match="different weights per agent"):
        module.main()


def test_a_population_checkpoint_round_trips_and_the_bump_refuses_a_stale_one(
    monkeypatch, tmp_path
) -> None:
    """The payload has to carry every member: restoring one and training the rest
    from their fresh initialization would report a resumed run. A version-10
    payload holds four top-level state dicts and no members at all, which is why
    reading it as a population is refused rather than half-interpreted."""
    from kaggriculture.inference import POPULATION_CHECKPOINT_KEY
    from kaggriculture.training import CHECKPOINT_FORMAT_VERSION, TrainingAgent, load_checkpoint

    module = _training_script()
    population = 3
    games = population * (population - 1)
    run_dir = tmp_path / "population"

    artifacts = tuple(
        _actor_artifact(tmp_path / f"member-{agent}.pt", actor, _TINY_CONFIG)
        for agent, actor in enumerate(_distinct_actors(_TINY_CONFIG, population))
    )
    record, seen = _run_population_main(
        module,
        monkeypatch,
        run_dir,
        population=population,
        games=games,
        initial_actors=artifacts,
    )

    assert len(seen) == population
    assert sorted(np.concatenate(seen).tolist()) == list(range(2 * games))
    # Per member, so one collapsed member is visible as itself rather than as a
    # third of an average that still reads healthy.
    assert {record[f"agent{agent}_entropy"] for agent in range(population)} == {0.2}
    assert record["population_disagreement_mean"] > 0.0
    assert record["population_disagreement_floor"] == record["population_disagreement_mean"]

    payload = torch.load(run_dir / "latest.pt", weights_only=False)
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    members = payload[POPULATION_CHECKPOINT_KEY]
    assert len(members) == population
    assert all(
        set(member)
        == {"actor", "critic", "actor_optimizer", "critic_optimizer", "orientation"}
        for member in members
    )
    # Each member carries the orientation its index assigns, so a resume reads
    # the same label space the wave was collected under without re-deriving it.
    assert [member["orientation"] for member in members] == [
        int(code) for code in row_orientations(np.arange(population))
    ]
    assert not any(name in payload for name in ("actor", "critic", "actor_optimizer"))

    config = _TINY_CONFIG
    architecture = resolve_architecture({"architecture": CONV_ENTITY})
    restored = [
        TrainingAgent(architecture.actor_class(config), architecture.critic_class(config))
        for _ in range(population)
    ]
    reloaded = load_checkpoint(run_dir / "latest.pt", restored, device=torch.device("cpu"))
    assert reloaded["iteration"] == 1
    assert reloaded["population_disagreement_reference"] == record["population_disagreement_floor"]
    # Measured at the first actor-active iteration and carried, so a resume
    # judges the members against where they started rather than against however
    # far they have already sharpened.
    assert reloaded["policy_entropy_reference"] == [0.2] * population
    for member, stored in zip(restored, members, strict=True):
        assert all(
            torch.equal(value, stored["actor"][name])
            for name, value in member.actor.state_dict().items()
        )
    # Distinct members, which is the whole point of the population: one restored
    # set of weights standing in for all three would pass every other assertion.
    signatures = {
        tuple(round(float(value.sum()), 6) for value in member.actor.state_dict().values())
        for member in restored
    }
    assert len(signatures) == population

    # The strongest statement of the round trip: the loop itself continues from the
    # payload. Every population-shaped resume field is read on this path -- one
    # parity baseline per member and the iteration-0 disagreement reference, which
    # must be restored rather than re-measured after the members have moved.
    resumed_dir = tmp_path / "resumed"
    resumed_record, resumed_rows = _run_population_main(
        module,
        monkeypatch,
        resumed_dir,
        population=population,
        games=games,
        iterations=2,
        resume=run_dir / "latest.pt",
    )
    assert len(resumed_rows) == population
    assert resumed_record["iteration"] == 2
    assert (
        resumed_record["population_disagreement_floor"] == record["population_disagreement_floor"]
    )
    assert (resumed_dir / "checkpoint-000002.pt").is_file()

    stale = tmp_path / "stale.pt"
    torch.save({**payload, "format_version": CHECKPOINT_FORMAT_VERSION - 1}, stale)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(stale, restored, device=torch.device("cpu"))
