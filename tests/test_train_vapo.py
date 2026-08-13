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
    assert (
        args.cnn_width,
        args.cnn_blocks,
        args.model_dim,
        args.transformer_layers,
        args.attention_heads,
        args.ffn_multiplier,
        args.quantity_rank,
    ) == (48, 2, 96, 7, 4, 4, 32)
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
        SnapshotSelection(SnapshotRef(0, tmp_path / "initial.pt"), "initial"),
        SnapshotSelection(SnapshotRef(9, tmp_path / "active.pt"), "active"),
    ]

    diagnostics = module._league_opponent_diagnostics(league, assignments, selections)

    assert diagnostics["league_opponent_00000000_category"] == "initial"
    assert diagnostics["league_opponent_00000000_games"] == 2
    assert diagnostics["league_opponent_00000000_score_rate"] == 0.5
    assert diagnostics["league_opponent_00000009_category"] == "active"
    assert diagnostics["league_opponent_00000009_games"] == 2
    assert diagnostics["league_opponent_00000009_score_rate"] == 0.75


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

    assert module._select_league_opponents(args, refs, 1, generator) == []
    assert generator.random() == reference.random()


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

    rollout = SimpleNamespace(states=1)
    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(module, "collect_mixed_play_rust", lambda *args, **kwargs: rollout)
    monkeypatch.setattr(module, "slice_trajectories", lambda batch, start, stop: batch)
    monkeypatch.setattr(module, "rollout_diagnostics", lambda batch: {})
    monkeypatch.setattr(
        module,
        "update_vapo",
        lambda *args, **kwargs: {"actor_updates": 1, "critic_updates": 1},
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
