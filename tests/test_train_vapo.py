from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kaggriculture.league import (
    SnapshotRef,
    SnapshotSelection,
    save_actor_snapshot,
    snapshot_sha256,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.modelargs import model_config_from_args
from kaggriculture.registry import CONV_ENTITY, STRUCTURED, resolve_architecture
from kaggriculture.structured import StructuredConfig


def _training_script():
    path = Path(__file__).parents[1] / "scripts" / "train_vapo.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_train_vapo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_defaults_prioritize_fresh_games_and_diverse_league(monkeypatch, tmp_path) -> None:
    module = _training_script()
    monkeypatch.setattr(sys, "argv", ["train_vapo.py", "--run-dir", str(tmp_path)])

    args = module.parse_args()

    assert (args.games, args.league_games) == (112, 96)
    assert (args.league_active_opponents, args.league_historical_opponents) == (2, 2)
    assert args.league_active_pool_size == 16
    assert args.epochs == 1
    assert args.minibatch_size == 2048
    # An unflagged run is exactly the family's dataclass configuration, which
    # is what a warm-start artifact and the calibration benchmark both carry.
    assert model_config_from_args(resolve_architecture(args.architecture), args) == ModelConfig()
    assert not hasattr(args, "entropy_coefficient")
    assert args.gamma == 1.0
    assert args.actor_gae_lambda == pytest.approx(1.0 - 1.0 / (0.05 * 719.0))
    assert not hasattr(args, "gae_lambda")
    assert args.target_kl == 0.03
    assert args.checkpoint_every == 5
    module._validate_args(args)

    args.gamma = 0.99
    with pytest.raises(ValueError, match=r"require --gamma 1\.0"):
        module._validate_args(args)


def test_model_flags_are_family_scoped_and_default_to_the_family_configuration(
    monkeypatch, tmp_path
) -> None:
    """Warm starting compares model configurations for equality, so an
    unflagged run must build the family default and a foreign flag must fail
    loudly instead of being silently dropped."""
    module = _training_script()

    monkeypatch.setattr(
        sys, "argv", ["train_vapo.py", "--run-dir", str(tmp_path), "--architecture", STRUCTURED]
    )
    args = module.parse_args()
    structured = resolve_architecture(STRUCTURED)
    assert model_config_from_args(structured, args) == StructuredConfig()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_vapo.py",
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
            "train_vapo.py",
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
        sys, "argv", ["train_vapo.py", "--run-dir", str(tmp_path), "--latents", "16"]
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
        ["train_vapo.py", "--run-dir", str(tmp_path), "--league-games", "2"],
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
        final_money=np.asarray([100.0, 50.0, 80.0, 90.0]),
        opponent_money=np.asarray([90.0, 60.0, 80.0, 20.0]),
    )
    assignments = np.asarray([0, 0, 1, 1])
    selections = [
        SnapshotSelection(SnapshotRef(2, tmp_path / "historical.pt"), "historical"),
        SnapshotSelection(SnapshotRef(9, tmp_path / "active.pt"), "active"),
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
    assert measured_rates == {2: 0.5, 9: 0.75}


def test_disabled_league_selection_does_not_advance_training_rng(tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        league_games=0,
        league_active_opponents=2,
        league_historical_opponents=2,
        league_active_pool_size=16,
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
    assert module._validate_league_score_rates({3: 0.25, 7: 1.0}) == {3: 0.25, 7: 1.0}
    for invalid in (
        None,
        [(3, 0.25)],
        {True: 0.5},
        {-1: 0.5},
        {3: 1},
        {3: float("nan")},
        {3: 1.5},
        {3: -0.1},
    ):
        with pytest.raises(ValueError):
            module._validate_league_score_rates(invalid)


def test_league_score_rate_blend_seeds_from_prior_and_decays_unmeasured() -> None:
    module = _training_script()
    rates = {1: 1.0}

    module._blend_league_score_rates(rates, {2: 1.0})

    # A first measurement blends against the unmeasured prior of 0.5, so one
    # perfect wave can never pin an estimate at exactly 1.0 and hard-retire a
    # freshly met opponent.
    assert rates[2] == pytest.approx(0.75)
    # Opponents that were not sampled decay toward the prior, keeping
    # retirement provisional instead of permanent.
    assert rates[1] == pytest.approx(0.975)

    module._blend_league_score_rates(rates, {2: 0.25})
    assert rates[2] == pytest.approx(0.5)


def test_external_eval_launcher_respects_cadence_and_running_worker(monkeypatch, tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval_every=10,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        episode_steps=720,
        run_dir=tmp_path,
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
    assert command[command.index("--snapshot") + 1].endswith("league-actor-00000010.pt")
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


def test_external_eval_launch_failure_never_kills_training(monkeypatch, tmp_path) -> None:
    module = _training_script()
    args = SimpleNamespace(
        external_eval_every=10,
        external_eval_opponents="starter",
        external_eval_seeds=2,
        episode_steps=720,
        run_dir=tmp_path,
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
    monkeypatch.setattr(sys, "argv", ["train_vapo.py", "--run-dir", str(tmp_path)])
    args = module.parse_args()

    config = module._training_data_config(args, module._device("cpu"))

    assert config["games"] == 112
    assert config["league_games"] == 96
    assert config["compile_models"] is False
    assert config["device_type"] == "cpu"
    args.games += 1
    assert module._training_data_config(args, module._device("cpu")) != config


def test_calibration_provenance_binds_initial_command_but_remains_portable(
    monkeypatch, tmp_path
) -> None:
    module = _training_script()
    identity = module.source_identity()
    initial_argv = ["train_vapo.py", "--run-dir", str(tmp_path / "initial")]
    decision = {
        "source_identity": identity,
        "compile_models": False,
        "eager_report_sha256": "a" * 64,
        "eager_report_size_bytes": 100,
        "compiled_report_sha256": "b" * 64,
        "compiled_report_size_bytes": 120,
        "minimum_compile_speedup": 1.05,
        "measured_compile_speedup": 1.0,
        "training_command": [
            sys.executable,
            str((Path(__file__).parents[1] / "scripts" / "train_vapo.py").resolve()),
            *initial_argv[1:],
        ],
    }
    path = tmp_path / "calibration-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", initial_argv)

    provenance = module._load_run_provenance(path, identity, bind_command=True)

    monkeypatch.setattr(sys, "argv", ["train_vapo.py", "--run-dir", str(tmp_path / "portable")])
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
        lambda *args, **kwargs: {"update_replay_max_ratio_error": 0.0},
    )
    monkeypatch.setattr(
        module,
        "update_vapo",
        lambda *args, **kwargs: {
            "actor_updates": 1,
            "critic_updates": 1,
            "first_minibatch_approx_kl": 0.0,
        },
    )

    def arguments(run_dir: Path, iterations: int, resume: Path | None = None) -> list[str]:
        values = [
            "train_vapo.py",
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


def test_warm_start_flags_validate_freshness_and_sign(monkeypatch, tmp_path) -> None:
    module = _training_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_vapo.py",
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
        monkeypatch.setattr(sys, "argv", ["train_vapo.py", "--run-dir", str(tmp_path), *flags])
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
    from kaggriculture.provenance import source_identity
    from kaggriculture.training import CHECKPOINT_FORMAT_VERSION, checkpoint_payload
    from kaggriculture.vapo import VapoConfig

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
        actor_state={},
        critic_state={},
        actor_optimizer_state={},
        critic_optimizer_state={},
        model_config=config,
        vapo_config=VapoConfig(epochs=1, minibatch_size=4, use_bfloat16=False),
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
