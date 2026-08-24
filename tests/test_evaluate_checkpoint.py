from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kaggriculture.inference import CHECKPOINT_FORMAT_VERSION
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.provenance import source_identity


def _load_evaluator():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(module, seed: int, seat: int, reward: float, opponent_reward: float, **changes):
    values = {
        "seed": seed,
        "candidate_seat": seat,
        "candidate_reward": reward,
        "opponent_reward": opponent_reward,
        "candidate_status": "DONE",
        "opponent_status": "DONE",
        "environment_done": True,
        "steps": 720,
        "expected_steps": 720,
        "elapsed_seconds": 0.1,
        "action_seconds": (0.001, 0.002),
    }
    values.update(changes)
    return module.GameResult(**values)


def test_summary_clusters_confidence_intervals_by_seed_pair() -> None:
    evaluator = _load_evaluator()
    rows = [
        _result(evaluator, 1, 0, 10.0, 0.0),
        _result(evaluator, 1, 1, 10.0, 0.0),
        _result(evaluator, 2, 0, 0.0, 10.0),
        _result(evaluator, 2, 1, 0.0, 10.0),
    ]

    summary = evaluator.summarize(rows, seed_count=2)

    assert summary["valid_for_selection"] is True
    assert summary["games"] == 4
    assert summary["complete_seat_pairs"] == 2
    assert summary["seed_clusters"] == 2
    assert summary["wins"] == 2
    assert summary["losses"] == 2
    assert summary["score_rate"] == 0.5
    assert summary["score_rate_95ci"] == [0.0, 1.0]
    assert summary["margin_95ci"] == pytest.approx([-19.6, 19.6])
    assert [row["seed"] for row in summary["seed_cluster_statistics"]] == [1, 2]


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_status": "ERROR"},
        {"opponent_status": "ACTIVE", "environment_done": False},
        {"steps": 719},
        {"candidate_reward": float("nan")},
    ],
)
def test_invalid_game_blocks_all_selection_metrics(changes) -> None:
    evaluator = _load_evaluator()
    valid = _result(evaluator, 3, 0, 100.0, 50.0)
    invalid = _result(evaluator, 3, 1, 100.0, 50.0, **changes)

    summary = evaluator.summarize([valid, invalid], seed_count=1)

    assert invalid.complete is False
    assert summary["valid_for_selection"] is False
    assert summary["games_completed"] == 1
    assert summary["invalid_games"] == 1
    assert summary["complete_seat_pairs"] == 0
    assert summary["score_rate"] is None
    assert summary["score_rate_95ci"] is None
    assert summary["mean_margin"] is None


def test_worker_reuses_one_loaded_model_across_actions_and_paired_games(monkeypatch) -> None:
    evaluator = _load_evaluator()
    counters = {"loads": 0, "actions": 0}

    class FakeAgent:
        def __call__(self, _observation):
            counters["actions"] += 1
            return {"farmer": ["PASS"], "hands": [], "market": []}

    def fake_checkpoint_agent(*_args, **_kwargs):
        counters["loads"] += 1
        return FakeAgent()

    class FakeEnvironment:
        def __init__(self) -> None:
            self.done = True
            final = [
                SimpleNamespace(reward=100.0, status="DONE"),
                SimpleNamespace(reward=50.0, status="DONE"),
            ]
            self.steps = [final] * 720

        def run(self, players) -> None:
            candidate = next(player for player in players if callable(player))
            for _ in range(3):
                candidate({"player": 0})

    monkeypatch.setattr(evaluator, "CheckpointAgent", fake_checkpoint_agent)
    monkeypatch.setattr(evaluator, "_make_environment", lambda *_args: FakeEnvironment())

    evaluator._initialize_worker("artifact.pt", "cpu", 1, "pass", None)

    pair = evaluator._run_seed_pair(9)

    assert counters == {"loads": 1, "actions": 6}
    assert all(result.complete for result in pair)
    assert [len(result.action_seconds) for result in pair] == [3, 3]


def test_run_game_records_non_done_status_as_an_explicit_error(monkeypatch) -> None:
    evaluator = _load_evaluator()
    evaluator._WORKER_AGENT = lambda _observation: {}
    evaluator._WORKER_OPPONENT = "pass"

    class ErrorEnvironment:
        def __init__(self) -> None:
            self.done = True
            self.steps = [
                [
                    SimpleNamespace(reward=None, status="ERROR"),
                    SimpleNamespace(reward=0.0, status="DONE"),
                ]
            ]

        @staticmethod
        def run(_players) -> None:
            print("agent diagnostic")

    monkeypatch.setattr(evaluator, "_make_environment", lambda *_args: ErrorEnvironment())

    result = evaluator._run_game(evaluator.GameSpec(seed=4, candidate_seat=0))

    assert result.complete is False
    assert "candidate status is ERROR" in result.error
    assert "expected 720" in result.error
    assert "agent diagnostic" in result.captured_output


def test_public_v27_alias_resolves_to_fixed_file(monkeypatch, tmp_path: Path) -> None:
    import kaggriculture.opponents as opponents

    evaluator = _load_evaluator()
    opponent = tmp_path / "v27.py"
    opponent.write_text("def agent(obs): return {}\n", encoding="utf-8")
    monkeypatch.setattr(opponents, "PUBLIC_V27_OPPONENT", opponent)

    label, resolved = evaluator.normalize_opponent("v27")

    assert label == "public-v27"
    assert resolved == str(opponent.resolve())


def test_screening_defaults_to_32_paired_v27_seed_clusters(monkeypatch) -> None:
    evaluator = _load_evaluator()
    monkeypatch.setattr(sys, "argv", ["evaluate_checkpoint.py", "--artifact", "model.pt"])

    args = evaluator.parse_args()

    assert args.seeds == 32
    assert args.opponent == "v27"


def test_artifact_validation_accepts_saved_training_checkpoint(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)
    checkpoint = tmp_path / "checkpoint-000017.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "critic": {"ignored_by_evaluation": torch.tensor(1.0)},
            "iteration": 17,
            "source_identity": source_identity(),
        },
        checkpoint,
    )

    provenance = evaluator._artifact_provenance(checkpoint)

    assert provenance["path"] == str(checkpoint)
    assert provenance["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert provenance["iteration"] == 17
    assert len(provenance["sha256"]) == 64
    assert provenance["source_identity"] == source_identity()


def test_finalist_selection_evidence_binds_exact_promoted_bytes(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    artifact = tmp_path / "best.pt"
    artifact.write_bytes(b"checkpoint")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    provenance = {
        "path": str(artifact.resolve()),
        "sha256": digest,
        "source_identity": source_identity(),
        "run_provenance": None,
    }
    report = tmp_path / "selection.json"
    report.write_text(
        json.dumps(
            {
                "valid_for_selection": True,
                "best_output": "/a/different/machine/best.pt",
                "best_output_sha256": digest,
                "source_identity": source_identity(),
                "run_provenance": None,
                "seed_start": 10_000_000,
                "seed_count": 32,
                "opponent_provenance": {
                    "public-v27": {
                        "kind": "python_file",
                        "path": "/var/tmp/v27.py",
                        "sha256": "a" * 64,
                        "size_bytes": 100,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    binding = evaluator._selection_provenance(report, provenance)

    assert binding["best_output_sha256"] == digest
    assert len(binding["sha256"]) == 64
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        evaluator._selection_provenance(report, provenance | {"sha256": "0" * 64})


def test_json_rendering_is_strict_finite_and_accepts_numpy_scalars() -> None:
    evaluator = _load_evaluator()

    rendered = evaluator.render_json({"value": np.float32(1.25), "count": np.int64(3)})

    assert json.loads(rendered) == {"count": 3, "value": 1.25}
    with pytest.raises(FloatingPointError, match=r"payload\.value"):
        evaluator.render_json({"value": float("inf")})


def test_game_payload_preserves_original_field_names() -> None:
    evaluator = _load_evaluator()
    result = _result(evaluator, 5, 1, 17.0, 12.0)

    payload = evaluator._game_payload(result)

    assert payload["seat"] == payload["candidate_seat"] == 1
    assert payload["reward"] == payload["candidate_reward"] == 17.0
    assert payload["status"] == payload["candidate_status"] == "DONE"
