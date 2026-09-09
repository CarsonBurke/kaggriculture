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

    assert (args.games, args.league_games) == (128, 64)
    # Self-play contributes both learner seats; a league game contributes one.
    assert args.games * 2 == 4 * args.league_games
    assert (args.league_active_opponents, args.league_historical_opponents) == (2, 6)
    assert args.league_active_pool_size == 16
    assert (args.league_builtin_opponents, args.league_builtin_lanes) == ("", 0)
    assert (args.epochs, args.critic_epochs) == (1, 1)
    assert args.critic_lr == pytest.approx(2.5e-4)
    assert args.minibatch_size == 4096
    # An unflagged run is exactly the family's dataclass configuration, which
    # is what a warm-start artifact and the calibration benchmark both carry.
    assert model_config_from_args(resolve_architecture(args.architecture), args) == ModelConfig()
    # Entropy is telemetry only; the training CLI has no bonus coefficient.
    assert not hasattr(args, "entropy_coefficient")
    assert args.gamma == pytest.approx(0.997)
    assert args.actor_gae_lambda == pytest.approx(1.0 - 1.0 / (0.05 * 719.0))
    assert args.critic_gae_lambda == pytest.approx(1.0)
    assert not hasattr(args, "gae_lambda")
    assert args.target_kl == PpoConfig.target_kl
    assert args.checkpoint_seconds == 420.0
    assert not hasattr(args, "checkpoint_every")
    assert not args.deterministic_training
    assert not any(
        (
            args.structured_decision_coefficient,
            args.structured_patch_coefficient,
            args.structured_economy_coefficient,
            args.structured_opponent_summary_coefficient,
            args.structured_opponent_patch_coefficient,
            args.structured_critic_latent_coefficient,
            args.structured_critic_value_coefficient,
        )
    )
    assert not hasattr(args, "structured_actor_gradient_ratio")
    module._validate_args(args)
    for rejected in (299.0, 601.0, float("nan")):
        args.checkpoint_seconds = rejected
        with pytest.raises(ValueError, match="checkpoint seconds"):
            module._validate_args(args)
    args.checkpoint_seconds = 420.0

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


def test_structured_auxiliary_cli_is_typed_population_safe_and_resume_bound(
    monkeypatch, tmp_path
) -> None:
    module = _training_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--run-dir",
            str(tmp_path),
            "--architecture",
            "structured",
            "--population",
            "2",
            "--games",
            "2",
            "--league-games",
            "0",
            "--structured-decision-coefficient",
            "0.5",
            "--structured-opponent-summary-coefficient",
            "0.5",
            "--structured-opponent-patch-coefficient",
            "0.5",
            "--structured-critic-latent-coefficient",
            "0.75",
            "--structured-critic-value-coefficient",
            "0.25",
            "--structured-critic-horizon",
            "3",
            "--deterministic-training",
        ],
    )
    args = module.parse_args()
    module._validate_args(args)

    assert args.structured_decision_horizon == 2
    assert args.structured_patch_horizon == 1
    assert args.structured_critic_horizon == 3
    assert args.structured_critic_latent_coefficient == pytest.approx(0.75)
    assert args.structured_critic_value_coefficient == pytest.approx(0.25)
    assert module._training_data_config(args, torch.device("cpu"))["deterministic_training"]

    args.architecture = CONV_ENTITY
    with pytest.raises(ValueError, match="require --architecture structured"):
        module._validate_args(args)
    args.architecture = STRUCTURED
    args.structured_critic_horizon = 0
    with pytest.raises(ValueError, match="critic horizon must be positive"):
        module._validate_args(args)


def test_deterministic_training_configures_pytorch_and_cudnn(monkeypatch) -> None:
    module = _training_script()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        module._configure_training_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert module.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        module._configure_training_determinism(False)


def test_recovery_checkpoint_timer_uses_injected_monotonic_clock() -> None:
    module = _training_script()
    now = [100.0]
    timer = module.RecoveryCheckpointTimer(420.0, clock=lambda: now[0])

    now[0] = 519.999
    assert not timer.due()
    now[0] = 520.0
    assert timer.due()
    timer.committed()
    now[0] = 939.999
    assert not timer.due()
    now[0] = 940.0
    assert timer.due()


def test_population_defaults_drop_the_frozen_lane(monkeypatch, tmp_path) -> None:
    """`--population 4` must not inherit the single-learner's 128+64 mix.

    That mix is not a valid population wave (128 is not a multiple of 12, and
    64 frozen games are a league the collector refuses), so leaving those
    defaults implicit would make every unflagged population launch fail.
    """
    module = _training_script()
    monkeypatch.setattr(
        sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path), "--population", "4"]
    )

    args = module.parse_args()

    assert args.population == 4
    assert args.games == 4 * 3 * 13
    assert args.league_games == 0


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


def test_balanced_opponent_assignments_are_reproducible_and_seat_balanced() -> None:
    module = _training_script()
    first = module._balanced_assignments(11, 4, np.random.default_rng(7), seed_start=17)
    second = module._balanced_assignments(11, 4, np.random.default_rng(7), seed_start=17)

    np.testing.assert_array_equal(first, second)
    counts = np.bincount(first, minlength=4)
    assert counts.max() - counts.min() == 1
    seats = (17 + np.arange(11)) % 2
    for opponent in range(4):
        opponent_seats = np.bincount(seats[first == opponent], minlength=2)
        assert abs(int(opponent_seats[0]) - int(opponent_seats[1])) <= 1
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
    builtin_rates = {f"builtin_{name}": 0.5 for name in module.BUILTIN_OPPONENTS}
    assert module._validate_league_score_rates(builtin_rates) == builtin_rates
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


def test_league_score_rate_blend_decays_all_stale_evidence() -> None:
    module = _training_script()
    rates = {"00000001": 1.0}

    module._blend_league_score_rates(rates, {"builtin_starter": 1.0})

    # A first measurement blends against the unmeasured prior of 0.5, so one
    # perfect wave cannot retire a freshly met opponent.
    assert rates["builtin_starter"] == pytest.approx(0.75)
    assert rates["00000001"] == pytest.approx(0.975)

    module._blend_league_score_rates(rates, {"builtin_starter": 1.0})
    mastered = rates["builtin_starter"]
    for _ in range(20):
        module._blend_league_score_rates(rates, {})

    # Learner drift invalidates both immutable built-ins and frozen snapshots.
    assert 0.5 < rates["builtin_starter"] < mastered
    assert rates["builtin_starter"] == pytest.approx(0.5 + (mastered - 0.5) * 0.95**20)
    assert rates["00000001"] < 0.975


def test_external_eval_launcher_acknowledges_complete_events_in_fifo_order(
    monkeypatch, tmp_path
) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval=True,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        external_eval_seed_start=4_000_000,
        episode_steps=720,
        run_dir=tmp_path,
        population=1,
    )
    launched: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            launched.append(command)
            self.command = command
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self) -> int:
            artifact = Path(self.command[self.command.index("--artifact") + 1])
            iteration = int(self.command[self.command.index("--iteration") + 1])
            digest = module.file_sha256(artifact)
            row = {
                "event": "external_eval",
                "iteration": iteration,
                "artifact": artifact.name,
                "artifact_sha256": digest,
                "agent": None,
                "opponent": "starter",
                "games": 4,
                "completed_games": 4,
                "seed_start": 4_000_000,
                "seed_count": 2,
            }
            marker = {
                "event": "external_eval_complete",
                "iteration": iteration,
                "artifact": artifact.name,
                "artifact_sha256": digest,
                "members": [None],
                "opponents": ["starter"],
                "records": 1,
            }
            with (tmp_path / "metrics-external.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.write(json.dumps(marker) + "\n")
            self.returncode = 0
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)

    assert (
        module._maybe_launch_external_eval(args, tmp_path / "checkpoint-000000.pt", 0, None) is None
    )
    checkpoints = []
    for iteration in (10, 20, 30, 40):
        checkpoint = tmp_path / f"checkpoint-{iteration:06d}.pt"
        checkpoint.write_bytes(str(iteration).encode())
        checkpoints.append(checkpoint)

    process = module._maybe_launch_external_eval(args, checkpoints[0], 10, None)
    assert isinstance(process, FakeProcess)
    command = launched[0]
    assert command[command.index("--artifact") + 1] == str(checkpoints[0])
    assert command[command.index("--agents") + 1] == ""
    assert command[command.index("--iteration") + 1] == "10"
    assert command[command.index("--opponents") + 1] == "starter"
    assert command[command.index("--output") + 1] == str(tmp_path / "metrics-external.jsonl")

    assert module._maybe_launch_external_eval(args, checkpoints[1], 20, process) is process
    assert module._maybe_launch_external_eval(args, checkpoints[2], 30, process) is process
    drained = module._maybe_launch_external_eval(
        args, checkpoints[3], 40, process, wait_for_slot=True
    )
    assert drained is None
    assert [command[command.index("--iteration") + 1] for command in launched] == [
        "10",
        "20",
        "30",
        "40",
    ]
    assert not (tmp_path / module._EXTERNAL_EVAL_PENDING).exists()

    disabled = SimpleNamespace(**{**vars(args), "external_eval": False})
    assert module._maybe_launch_external_eval(disabled, checkpoints[2], 30, None) is None


def test_a_population_is_probed_from_its_committed_checkpoint(monkeypatch, tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval=True,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        external_eval_seed_start=4_000_000,
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
    checkpoint = tmp_path / "checkpoint-000010.pt"
    module._maybe_launch_external_eval(args, checkpoint, 10, None)
    command = launched[0]
    assert command[command.index("--artifact") + 1] == str(checkpoint)
    assert command[command.index("--agents") + 1] == "0,1,2,3"


def test_external_eval_launch_failure_is_retried_from_the_durable_fifo(
    monkeypatch,
    tmp_path,
) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval=True,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        external_eval_seed_start=4_000_000,
        episode_steps=720,
        run_dir=tmp_path,
        population=1,
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fork failed")),
    )
    checkpoint = tmp_path / "checkpoint-000010.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert module._maybe_launch_external_eval(args, checkpoint, 10, None) is None
    pending_path = tmp_path / module._EXTERNAL_EVAL_PENDING
    assert pending_path.is_file()

    resumed = SimpleNamespace(
        **{key: value for key, value in vars(args).items() if key != "_kaggriculture_pending_evals"}
    )
    launched: list[list[str]] = []

    class FakeProcess:
        returncode = None

        def __init__(self, command, **_kwargs):
            self.command = command
            launched.append(command)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    process = module._maybe_launch_external_eval(resumed, None, 0, None)

    assert isinstance(process, FakeProcess)
    assert launched[0][launched[0].index("--artifact") + 1] == str(checkpoint)
    assert launched[0][launched[0].index("--iteration") + 1] == "10"
    # Launching is not acknowledgement; an interrupted worker remains durable.
    assert pending_path.is_file()
    process.returncode = 1
    replacement = module._maybe_launch_external_eval(resumed, None, 0, process)
    assert isinstance(replacement, FakeProcess) and replacement is not process
    assert len(launched) == 2
    assert pending_path.is_file()


def test_final_external_eval_fails_closed_without_completion_record(monkeypatch, tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval=True,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        external_eval_seed_start=4_000_000,
        episode_steps=720,
        run_dir=tmp_path,
        population=1,
    )
    checkpoint = tmp_path / "checkpoint-000010.pt"
    checkpoint.write_bytes(b"checkpoint")

    class MissingCompletionProcess:
        returncode = None

        def __init__(self, *_args, **_kwargs):
            pass

        def poll(self):
            return self.returncode

        def wait(self):
            self.returncode = 0
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", MissingCompletionProcess)
    process = module._maybe_launch_external_eval(args, checkpoint, 10, None)
    with pytest.raises(RuntimeError, match="without a matching completion record"):
        module._maybe_launch_external_eval(args, None, 10, process, wait_for_slot=True)
    assert (tmp_path / module._EXTERNAL_EVAL_PENDING).is_file()


def test_external_eval_opponent_resolution_degrades_instead_of_blocking(capsys, tmp_path) -> None:
    module = _training_script()
    missing = tmp_path / "gone.py"
    args = SimpleNamespace(
        external_eval=True,
        external_eval_opponents=f"starter,{missing},",
    )
    module._resolve_external_eval_opponents(args)
    assert args.external_eval
    assert args.external_eval_opponents == "starter"
    assert "dropped" in capsys.readouterr().err

    args = SimpleNamespace(external_eval=True, external_eval_opponents=str(missing))
    module._resolve_external_eval_opponents(args)
    assert not args.external_eval
    assert "disabled" in capsys.readouterr().err


def test_training_data_config_captures_rollout_semantics(monkeypatch, tmp_path) -> None:
    module = _training_script()
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path)])
    args = module.parse_args()

    config = module._training_data_config(args, module._device("cpu"))

    assert config["games"] == 128
    assert config["league_games"] == 64
    assert config["update_compile_mode"] == "default"
    assert config["device_type"] == "cpu"
    # The collector-owned CUDA graph over the Inductor-fused forward: fusion
    # removes the eager kernel count, and capture removes the launch overhead.
    assert config["rollout_forward_mode"] == "inductor_graph"
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
    """The collector-owned CUDA graph over the Inductor-fused forward measured a
    3.90 s steady rollout median against the eager-kernel graph's 6.72 s at
    production shape, with lower update-replay drift."""
    module = _training_script()

    def parsed(*flags: str):
        monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--run-dir", str(tmp_path), *flags])
        return module.parse_args()

    default = parsed()
    assert default.rollout_forward_mode == "inductor_graph"
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
        current_iteration=1,
        model_config=model_config,
    )

    assert {
        ref.iteration: snapshot_sha256(ref.path) for ref in module.list_actor_snapshots(destination)
    } == manifest

    save_actor_snapshot(destination, actor, 2)
    save_actor_snapshot(destination, actor, 3)
    invalid = {**validated, 1: "0" * 64}
    with pytest.raises(ValueError, match="digest mismatch"):
        module._restore_league_archive(
            checkpoint=checkpoint,
            destination=destination,
            manifest=invalid,
            current_iteration=1,
            model_config=model_config,
        )
    assert [ref.iteration for ref in module.list_actor_snapshots(destination)] == [0, 1, 2, 3]

    module._restore_league_archive(
        checkpoint=checkpoint,
        destination=destination,
        manifest=validated,
        current_iteration=1,
        model_config=model_config,
    )
    assert [ref.iteration for ref in module.list_actor_snapshots(destination)] == [0, 1]

    sparse_destination = tmp_path / "sparse" / "league"
    sparse_manifest = {0: manifest[0]}
    module._restore_league_archive(
        checkpoint=checkpoint,
        destination=sparse_destination,
        manifest=sparse_manifest,
        current_iteration=2,
        model_config=model_config,
    )
    save_actor_snapshot(sparse_destination, actor, 1)
    with pytest.raises(ValueError, match="missing from the checkpoint manifest"):
        module._restore_league_archive(
            checkpoint=checkpoint,
            destination=sparse_destination,
            manifest=sparse_manifest,
            current_iteration=2,
            model_config=model_config,
        )


def test_league_manifest_accepts_sparse_warmup_history_but_requires_the_anchor() -> None:
    module = _training_script()

    sparse = {0: "a" * 64, 4: "b" * 64}
    assert module._validate_league_manifest(sparse, current_iteration=4) == sparse
    with pytest.raises(ValueError, match="iteration zero"):
        module._validate_league_manifest({2: "b" * 64}, current_iteration=2)
    with pytest.raises(ValueError, match="digest"):
        module._validate_league_manifest({0: "not-a-digest"}, current_iteration=0)


def test_orphan_checkpoint_matching_ignores_only_volatile_metrics() -> None:
    module = _training_script()
    original = {
        "iteration": 2,
        "next_seed": 9,
        "actor": {"weight": torch.tensor([1.0])},
        "metrics": {"elapsed_hours": 0.1, "iteration_seconds": 20.0},
    }
    replayed = {
        **original,
        "metrics": {"elapsed_hours": 0.2, "iteration_seconds": 21.0},
    }
    assert module._checkpoint_recovery_values_equal(original, replayed)
    assert not module._checkpoint_recovery_values_equal(original, {**replayed, "next_seed": 10})


def test_metrics_journal_rolls_back_to_a_verified_checkpoint_boundary(tmp_path: Path) -> None:
    module = _training_script()
    path = tmp_path / "metrics.jsonl"
    records = [
        {"iteration": 1, "value_loss": 1.0},
        {"iteration": 2, "value_loss": 0.5},
        {"iteration": 3, "value_loss": 0.25},
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    module._rollback_metrics_journal(path, 1, records[0])

    assert path.read_text(encoding="utf-8") == json.dumps(records[0], sort_keys=True) + "\n"

    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match recovery state"):
        module._rollback_metrics_journal(path, 1, {"iteration": 1, "value_loss": 9.0})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_main_writes_complete_manifests_and_portably_resumes(
    monkeypatch,
    tmp_path,
) -> None:
    module = _training_script()
    run_provenance = module.run_provenance_from_decision(
        {
            "source_identity": module.source_identity(),
            "rollout_forward_mode": "inductor_graph",
            "update_compile_mode": "default",
            "eager_report_sha256": "a" * 64,
            "eager_report_size_bytes": 100,
            "mixed_report_sha256": "b" * 64,
            "mixed_report_size_bytes": 110,
            "compiled_report_sha256": "c" * 64,
            "compiled_report_size_bytes": 120,
            "minimum_compile_speedup": 1.05,
            "attributed_knob_speedups": {
                "rollout_forward_mode": 1.1,
                "update_compile_mode": 1.1,
            },
        }
    )
    monkeypatch.setattr(module, "_load_run_provenance", lambda *_args, **_kwargs: run_provenance)

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    rollout = SimpleNamespace(state_count=1, trajectories=2)
    collected_gammas: list[float] = []

    def collect(*args, **kwargs):
        collected_gammas.append(kwargs["gamma"])
        return rollout

    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(module, "collect_mixed_play_rust", collect)
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
            "--no-bfloat16",
            "--gamma",
            "0.91",
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
    assert numbered["run_provenance"] == run_provenance
    assert not (source_run / "latest.pt").is_symlink()
    assert (source_run / "latest.pt").stat().st_ino == (
        source_run / "checkpoint-000001.pt"
    ).stat().st_ino

    # A crash or manual cleanup that removes the immutable name while its
    # hard-linked latest alias survives is repaired without reserialization.
    (source_run / "checkpoint-000001.pt").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(source_run, 1, source_run / "latest.pt"),
    )
    module.main()
    repaired = torch.load(source_run / "checkpoint-000001.pt", weights_only=False)
    assert repaired["league_snapshot_manifest"] == latest["league_snapshot_manifest"]
    assert (source_run / "latest.pt").stat().st_ino == (
        source_run / "checkpoint-000001.pt"
    ).stat().st_ino

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
    assert collected_gammas and set(collected_gammas) == {0.91}


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
        lambda *args, **kwargs: SimpleNamespace(state_count=1, trajectories=2),
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

    args.critic_warmup_iterations = module.MAX_CRITIC_WARMUP_ITERATIONS + 1
    with pytest.raises(ValueError, match="readiness deadline"):
        module._validate_args(args)

    args.critic_warmup_iterations = 0
    args.resume = tmp_path / "latest.pt"
    with pytest.raises(ValueError, match="fresh run"):
        module._validate_args(args)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--run-dir",
            str(tmp_path),
            "--init-actor-from",
            str(tmp_path / "bc-actor.pt"),
        ],
    )
    defaulted = module.parse_args()
    module._validate_args(defaulted)
    assert defaulted.critic_warmup_iterations == module.DEFAULT_CRITIC_WARMUP_ITERATIONS == 5


def test_adaptive_critic_warmup_uses_prior_wave_ev_and_has_a_hard_deadline() -> None:
    module = _training_script()

    active, reason = module._critic_warmup_decision(
        iteration=4,
        minimum=5,
        complete=False,
        previous_evs=[0.9],
    )
    assert active and reason == "minimum_iterations"

    active, reason = module._critic_warmup_decision(
        iteration=5,
        minimum=5,
        complete=False,
        previous_evs=[0.09],
    )
    assert active and reason == "waiting_for_monte_carlo_ev"

    active, reason = module._critic_warmup_decision(
        iteration=6,
        minimum=5,
        complete=False,
        previous_evs=[0.10, 0.35],
    )
    assert not active and reason == "monte_carlo_ev_ready"
    assert module._critic_warmup_decision(
        iteration=20,
        minimum=5,
        complete=True,
        previous_evs=[-1.0],
    ) == (False, "complete")

    with pytest.raises(RuntimeError, match="within 40 iterations"):
        module._critic_warmup_decision(
            iteration=module.MAX_CRITIC_WARMUP_ITERATIONS,
            minimum=5,
            complete=False,
            previous_evs=[0.099],
        )


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
            "bc_provenance": {
                "teacher": {"label": "public-v27"},
                "datasets": [{"train_seeds": [1, 2], "holdout_seeds": [3]}],
            },
            "seed_usage": [{"domain": "bc", "start": 1, "count": 3}],
        },
        artifact,
    )

    record = module._load_initial_actor(
        artifact, FarmActor(config), CONV_ENTITY, config, torch.device("cpu")
    )
    record["critic_warmup_iterations"] = 15
    record["critic_warmup_state"] = {
        "complete": False,
        "last_monte_carlo_explained_variance": [0.04],
    }

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
    assert payload["initial_actor"]["critic_warmup_state"] == {
        "complete": False,
        "last_monte_carlo_explained_variance": [0.04],
    }
    assert module._validate_critic_warmup_state(payload["initial_actor"], population=1) == (
        15,
        False,
        [0.04],
    )
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
            "bc_provenance": {
                "teacher": {"label": "public-v27"},
                "datasets": [{"train_seeds": [1, 2], "holdout_seeds": [3]}],
            },
            "seed_usage": [{"domain": "bc", "start": 1, "count": 3}],
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
    assert any(row["domain"] == "bc" and row["start"] == 1 for row in provenance["seed_usage"])
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
    # A short KL-clipped epoch and a sharp policy used to kill the run.
    # Improvement vs public-v27 is the ranking; those are not stop conditions.
    module._gate_update_metrics(
        {**healthy, "actor_updates": 1, "kl_early_stop": 1, "max_approx_kl": 0.0857},
        warmup_active=False,
    )
    module._gate_update_metrics({**healthy, "entropy": 0.0}, warmup_active=False)
    module._gate_update_metrics(
        {**healthy, "entropy": MINIMUM_POLICY_ENTROPY * 0.99}, warmup_active=False
    )
    module._gate_update_metrics(
        {**healthy, "actor_updates": 0, "kl_early_stop": 1}, warmup_active=True
    )
    module._gate_update_metrics({**healthy, "entropy": 0.0}, warmup_active=True)


def test_sharp_clone_entropy_is_telemetry_not_a_stop_condition() -> None:
    """A faithful BC clone may begin sharp while already playing well."""
    module = _training_script()
    healthy = {
        "first_minibatch_approx_kl": 0.0,
        "value_target_saturated_fraction": 0.0,
        "actor_updates": 113,
        "actor_minibatches_intended": 113,
        "max_approx_kl": 0.0,
        "entropy": 0.0,
    }

    module._gate_update_metrics(healthy, warmup_active=False)
    assert module._validate_policy_entropy_reference(0.0) == 0.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        module._validate_policy_entropy_reference(-1e-9)


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
    with pytest.raises(ValueError, match="finite and nonnegative"):
        module._validate_entropy_references([-0.1, None, 0.29], population=3)


def test_structured_persistence_diagnostics_use_each_fresh_wave() -> None:
    module = _training_script()
    metrics = {
        "structured_preupdate_combined": 2.0,
        "structured_preupdate_decision": 4.0,
        "structured_preupdate_patch": 0.5,
        "structured_preupdate_economy": 0.5,
        "structured_preupdate_opponent_summary": 0.06,
        "structured_preupdate_opponent_patches": 0.03,
        "structured_critic_preupdate_combined": 3.0,
        "structured_critic_preupdate_latent": 2.0,
        "structured_critic_preupdate_value": 1.0,
    }
    metrics.update(
        {
            name.replace("preupdate_", "preupdate_persistence_"): value / 2
            for name, value in tuple(metrics.items())
        }
    )
    for kind, prefix in (
        ("actor", "structured_persistence_"),
        ("critic", "structured_critic_persistence_"),
    ):
        measured = module._structured_persistence_diagnostics(metrics, kind=kind)
        assert measured[f"{prefix}combined_ratio"] == 2.0
        assert measured[f"{prefix}combined_informative"] == 1
        assert all(name.endswith(("_ratio", "_informative")) for name in measured)

    # Changing decoder scales next wave must change the comparison immediately,
    # without historical references or predictor-quality admission state.
    metrics["structured_preupdate_persistence_opponent_patches"] = 0.06
    actor = module._structured_persistence_diagnostics(metrics, kind="actor")
    assert actor["structured_persistence_opponent_patches_ratio"] == 0.5
    assert actor["structured_persistence_decision_ratio"] == 2.0


def test_zero_decoder_diagnostics_become_informative_on_fresh_waves() -> None:
    module = _training_script()

    def wave(value: float, baseline: float) -> dict[str, float]:
        return {
            "structured_critic_preupdate_combined": 0.5 + value,
            "structured_critic_preupdate_latent": 0.5,
            "structured_critic_preupdate_value": value,
            "structured_critic_preupdate_persistence_combined": 1.0 + baseline,
            "structured_critic_preupdate_persistence_latent": 1.0,
            "structured_critic_preupdate_persistence_value": baseline,
        }

    for value in (0.0, 0.01, 0.0):
        measured = module._structured_persistence_diagnostics(wave(value, 0.0), kind="critic")
        assert measured["structured_critic_persistence_value_informative"] == 0
        assert measured["structured_critic_persistence_value_ratio"] == 1.0
        json.dumps(measured, allow_nan=False)
    # There is no arbitrary loss-scale floor: tiny positive baselines carry
    # genuine information, while a worse-than-persistence ratio remains useful.
    measured = module._structured_persistence_diagnostics(wave(1e-9, 2e-9), kind="critic")
    assert measured["structured_critic_persistence_value_informative"] == 1
    assert measured["structured_critic_persistence_value_ratio"] == 0.5
    measured = module._structured_persistence_diagnostics(wave(0.1, 0.09), kind="critic")
    assert measured["structured_critic_persistence_value_ratio"] == pytest.approx(10 / 9)
    json.dumps(measured, allow_nan=False)

    for value, baseline in (
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (1.0, 1e-320),
    ):
        with pytest.raises(FloatingPointError, match="non-finite structured critic persistence"):
            module._structured_persistence_diagnostics(wave(value, baseline), kind="critic")


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


def _actor_artifact(
    path: Path,
    actor: torch.nn.Module,
    config: ModelConfig | StructuredConfig,
    *,
    architecture: str = CONV_ENTITY,
) -> Path:
    """One exported actor artifact of the shape a BC clone writes."""
    from kaggriculture.inference import ACTOR_ARTIFACT_FORMAT_VERSION
    from kaggriculture.provenance import source_identity

    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "architecture": architecture,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 0,
            "source_identity": source_identity(),
            "seed_usage": [],
        },
        path,
    )
    return path


_TINY_CONFIG = ModelConfig(
    cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
)


def _population_arguments(
    run_dir: Path,
    *,
    population: int,
    games: int,
    iterations: int = 1,
    architecture: str = CONV_ENTITY,
):
    arguments = [
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
        "--device",
        "cpu",
        "--model-dim",
        "16",
        "--attention-heads",
        "2",
        "--no-bfloat16",
    ]
    if architecture == CONV_ENTITY:
        arguments.extend(("--cnn-width", "8", "--cnn-blocks", "1", "--transformer-layers", "3"))
    else:
        arguments.extend(("--architecture", architecture))
    return arguments


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
    architecture: str = CONV_ENTITY,
    critic_warmup_iterations: int = 0,
    update_fn=None,
    extra_arguments: tuple[str, ...] = (),
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
            "monte_carlo_explained_variance": 0.2,
        }

    wave = _population_wave(module, games=games, population=max(population, 2))
    if population == 1:
        wave.agents = np.zeros(wave.agents.size, dtype=np.int64)

    def collect(*args, **kwargs):
        assert kwargs["gamma"] == pytest.approx(0.997)
        return wave

    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(module, "collect_population_play_rust", collect)
    monkeypatch.setattr(module, "collect_mixed_play_rust", collect)
    monkeypatch.setattr(module, "slice_trajectories", lambda batch, start, stop: batch)
    monkeypatch.setattr(module, "rollout_diagnostics", lambda batch: {})
    monkeypatch.setattr(module, "update_replay_parity", lambda *a, **k: _parity_metrics(module))
    monkeypatch.setattr(module, "update_ppo", update if update_fn is None else update_fn)
    arguments = _population_arguments(
        run_dir,
        population=population,
        games=games,
        iterations=iterations,
        architecture=architecture,
    )
    for artifact in initial_actors:
        arguments.extend(("--init-actor-from", str(artifact)))
    if initial_actors:
        arguments.extend(("--critic-warmup-iterations", str(critic_warmup_iterations)))
    if resume is not None:
        arguments.extend(("--resume", str(resume)))
    arguments.extend(extra_arguments)
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    record = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    return record, seen


def test_runner_enters_joint_training_despite_poor_predictor_persistence(
    monkeypatch, tmp_path
) -> None:
    from kaggriculture.structured import StructuredActor

    module = _training_script()
    config = StructuredConfig(model_dim=16, attention_heads=2)
    artifact = _actor_artifact(
        tmp_path / "initial.pt",
        StructuredActor(config),
        config,
        architecture=STRUCTURED,
    )
    phases: list[tuple[int | None, bool, bool]] = []

    def update(*args, actor_epochs=None, **kwargs):
        phases.append(
            (
                actor_epochs,
                kwargs["structured_actor_auxiliary"],
                kwargs["structured_critic_auxiliary"],
            )
        )
        metrics = {
            "actor_updates": 0 if actor_epochs == 0 else 1,
            "actor_minibatches_intended": 1,
            "critic_updates": 1,
            "first_minibatch_approx_kl": 0.0,
            "value_target_saturated_fraction": 0.0,
            "entropy": 0.2,
            # Prior-wave evidence releases the actor after the warmup floor.
            "monte_carlo_explained_variance": 0.2,
        }
        for prefix, fields in (
            ("structured_preupdate_", module._STRUCTURED_ACTOR_PERSISTENCE_FIELDS),
            ("structured_critic_preupdate_", module._STRUCTURED_CRITIC_PERSISTENCE_FIELDS),
        ):
            for name in fields:
                metrics[f"{prefix}{name}"] = 10.0
                metrics[f"{prefix}persistence_{name}"] = 1.0
        return metrics

    run_dir = tmp_path / "joint"
    _run_population_main(
        module,
        monkeypatch,
        run_dir,
        population=1,
        games=2,
        iterations=3,
        architecture=STRUCTURED,
        initial_actors=(artifact,),
        critic_warmup_iterations=1,
        update_fn=update,
        extra_arguments=(
            "--structured-decision-coefficient",
            "0.5",
            "--structured-critic-latent-coefficient",
            "0.5",
        ),
    )
    # This is the actual runner's adaptive warmup transition, not independent
    # calls to a boolean helper: a bad fresh-wave ratio neither delays release
    # nor revokes joint learning on the following iteration.
    assert phases == [(0, False, True), (None, True, True), (None, True, True)]
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    updates = [record for record in records if "critic_warmup_active" in record]
    assert [record["critic_warmup_active"] for record in updates] == [1, 0, 0]
    for record in updates:
        assert record["structured_persistence_combined_ratio"] == 10.0
        assert record["structured_critic_persistence_combined_ratio"] == 10.0


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
    forward_args, masks, active = module._population_state_sample(
        wave, np.random.default_rng(0), torch.device("cpu")
    )

    distinct = _distinct_actors(_TINY_CONFIG, 4)
    identical = _distinct_actors(_TINY_CONFIG, 4)
    for member in identical[1:]:
        member.load_state_dict(identical[0].state_dict())

    reference = module.mean_off_diagonal(
        module._population_disagreement(distinct, forward_args, masks, active)
    )
    collapsed = module.mean_off_diagonal(
        module._population_disagreement(identical, forward_args, masks, active)
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
    """Four agents built from one checkpoint are numerically identical, which
    removes the population diversity the run is configured to preserve. Both
    spellings of that mistake must fail: the same path twice, and two paths
    holding the same weights."""
    module = _training_script()
    artifact = _actor_artifact(
        tmp_path / "bc-actor.pt", _distinct_actors(_TINY_CONFIG, 1)[0], _TINY_CONFIG
    )
    twin = tmp_path / "bc-actor-copy.pt"
    twin.write_bytes(artifact.read_bytes())

    repeated = _population_arguments(tmp_path / "repeated", population=2, games=2)
    repeated.extend(
        (
            "--init-actor-from",
            str(artifact),
            "--init-actor-from",
            str(artifact),
            "--critic-warmup-iterations",
            "0",
        )
    )
    monkeypatch.setattr(sys, "argv", repeated)
    with pytest.raises(ValueError, match="different artifact per agent"):
        module.main()

    # Distinct paths, so validation cannot see it: only the artifacts' digests can.
    copied = _population_arguments(tmp_path / "copied", population=2, games=2)
    copied.extend(
        (
            "--init-actor-from",
            str(artifact),
            "--init-actor-from",
            str(twin),
            "--critic-warmup-iterations",
            "0",
        )
    )
    monkeypatch.setattr(sys, "argv", copied)
    with pytest.raises(ValueError, match="different weights per agent"):
        module.main()


def test_finite_warm_start_cannot_exit_before_actor_release(monkeypatch, tmp_path) -> None:
    module = _training_script()
    artifact = _actor_artifact(
        tmp_path / "bc-actor.pt",
        _distinct_actors(_TINY_CONFIG, 1)[0],
        _TINY_CONFIG,
    )
    run_dir = tmp_path / "short"

    with pytest.raises(RuntimeError, match="final iteration before the critic satisfied"):
        _run_population_main(
            module,
            monkeypatch,
            run_dir,
            population=1,
            games=2,
            iterations=1,
            initial_actors=(artifact,),
        )

    checkpoint = torch.load(run_dir / "latest.pt", weights_only=False)
    assert checkpoint["iteration"] == 1
    assert checkpoint["initial_actor"]["critic_warmup_state"] == {
        "complete": False,
        "last_monte_carlo_explained_variance": [0.2],
    }


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
        iterations=2,
    )

    assert len(seen) == 2 * population
    for wave_rows in (seen[:population], seen[population:]):
        assert sorted(np.concatenate(wave_rows).tolist()) == list(range(2 * games))
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
        set(member) == {"actor", "critic", "actor_optimizer", "critic_optimizer", "orientation"}
        for member in members
    )
    # Evaluation plays the real board. Training cycles frames per game,
    # so a member has no private code to resume under.
    assert [member["orientation"] for member in members] == [0] * population

    assert not any(name in payload for name in ("actor", "critic", "actor_optimizer"))

    config = _TINY_CONFIG
    architecture = resolve_architecture({"architecture": CONV_ENTITY})
    restored = [
        TrainingAgent(architecture.actor_class(config), architecture.critic_class(config))
        for _ in range(population)
    ]
    reloaded = load_checkpoint(run_dir / "latest.pt", restored, device=torch.device("cpu"))
    assert reloaded["iteration"] == 2
    assert reloaded["population_disagreement_reference"] == record["population_disagreement_floor"]
    assert record["critic_warmup_active"] == 0
    assert record["critic_warmup_reason"] == "monte_carlo_ev_ready"
    assert record["critic_warmup_ready_members"] == population
    # The first wave is critic-only even with a zero configured minimum; the
    # second uses that fresh-wave EV to release every actor.
    assert reloaded["policy_entropy_reference"] == [0.2] * population
    assert reloaded["initial_actor"]["critic_warmup_state"] == {
        "complete": True,
        "last_monte_carlo_explained_variance": [0.2] * population,
    }
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
        iterations=3,
        resume=run_dir / "latest.pt",
    )
    assert len(resumed_rows) == population
    assert resumed_record["iteration"] == 3
    assert (
        resumed_record["population_disagreement_floor"] == record["population_disagreement_floor"]
    )
    assert (resumed_dir / "checkpoint-000003.pt").is_file()
    resumed_payload = torch.load(resumed_dir / "latest.pt", weights_only=False)
    assert resumed_payload["policy_entropy_reference"] == [0.2] * population
    assert resumed_payload["initial_actor"]["critic_warmup_state"]["complete"] is True
    assert resumed_record["critic_warmup_active"] == 0
    assert resumed_record["critic_warmup_reason"] == "complete"
    assert resumed_record["critic_warmup_ready_members"] == population

    stale = tmp_path / "stale.pt"
    torch.save({**payload, "format_version": CHECKPOINT_FORMAT_VERSION - 1}, stale)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(stale, restored, device=torch.device("cpu"))
