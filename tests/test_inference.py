from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
from kaggle_environments import make

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.inference import (
    ACTOR_ARTIFACT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION,
    LEGACY_CHECKPOINT_FORMAT_VERSIONS,
    CheckpointAgent,
    _cpu_portable_structured_state,
    actor_artifact_from_checkpoint,
    cpu_portable_actor_artifact,
    load_actor_artifact,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.provenance import (
    _identity_digest,
    is_legacy_run_provenance,
    run_provenance_from_decision,
    source_identity,
)
from kaggriculture.structured import StructuredActor, StructuredConfig


@pytest.mark.parametrize(
    "checkpoint_version",
    [
        ACTOR_ARTIFACT_FORMAT_VERSION,
        *sorted(LEGACY_CHECKPOINT_FORMAT_VERSIONS),
        CHECKPOINT_FORMAT_VERSION,
    ],
)
def test_actor_artifact_round_trip(tmp_path: Path, checkpoint_version: int) -> None:
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)
    artifact = actor_artifact_from_checkpoint(
        {
            "format_version": checkpoint_version,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 3,
            "source_identity": source_identity(),
        }
    )
    path = tmp_path / "model.pt"
    torch.save(artifact, path)

    restored, metadata = load_actor_artifact(path)

    assert metadata["iteration"] == 3
    assert metadata["format_version"] == ACTOR_ARTIFACT_FORMAT_VERSION
    for expected, actual in zip(actor.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def _checkpoint_agent_with_ranked_actions(
    tmp_path: Path, priorities: tuple[UnitAction, ...]
) -> CheckpointAgent:
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)
    with torch.no_grad():
        actor.unit_head[-1].weight.zero_()
        actor.unit_head[-1].bias.fill_(-100.0)
        for rank, action in enumerate(priorities):
            actor.unit_head[-1].bias[action] = 10.0 * (len(priorities) - rank)
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-100.0)
        actor.market_kind.bias[MarketKind.STOP] = 10.0
    artifact = actor_artifact_from_checkpoint(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "source_identity": source_identity(),
        }
    )
    path = tmp_path / "model.pt"
    torch.save(artifact, path)
    return CheckpointAgent(path)


def test_checkpoint_agent_preserves_same_tile_dig_then_plant(tmp_path: Path) -> None:
    agent = _checkpoint_agent_with_ranked_actions(
        tmp_path, (UnitAction.PLANT_WHEAT, UnitAction.DIG)
    )
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 13})
    observation = environment.reset(2)[0].observation
    farm = observation["farms"][0]
    x, y = farm["farmer"]
    farm["tiles"][y][x] = {"kind": "WEED"}
    farm["hands"] = [[x, y]]
    observation["private"]["inventories"].append({})
    observation["private"]["seeds"]["WHEAT"] = 1

    action = agent(observation)
    following = environment.step([action, {}])[0].observation

    assert action["farmer"] == ["DIG"]
    assert action["hands"] == [["PLANT", "WHEAT"]]
    planted = following["farms"][0]["tiles"][y][x]
    assert planted["kind"] == "PLANT"
    assert planted["crop"] == "WHEAT"
    assert following["private"]["seeds"]["WHEAT"] == 0


@pytest.mark.parametrize(
    ("choice", "command", "position", "carried_wheat"),
    [
        (UnitAction.NORTH, ["NORTH"], [4, 3], 0),
        (UnitAction.PICKUP_WHEAT_1, ["PICKUP", "WHEAT", 1], [4, 4], 1),
    ],
)
def test_checkpoint_agent_can_move_or_use_shed_while_standing_on_weeds(
    tmp_path: Path,
    choice: UnitAction,
    command: list,
    position: list[int],
    carried_wheat: int,
) -> None:
    agent = _checkpoint_agent_with_ranked_actions(tmp_path, (choice, UnitAction.DIG))
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 13})
    observation = environment.reset(2)[0].observation
    observation["farms"][0]["farmer"] = [4, 4]
    observation["farms"][0]["tiles"][4][4] = {"kind": "WEED"}
    observation["private"]["shed"]["WHEAT"] = 1
    observation["private"]["inventories"][0] = {}

    action = agent.act_many([observation])[0]
    following = environment.step([action, {}])[0].observation

    assert action["farmer"] == command
    assert following["farms"][0]["farmer"] == position
    assert following["farms"][0]["tiles"][4][4]["kind"] == "WEED"
    assert following["private"]["inventories"][0].get("WHEAT", 0) == carried_wheat
    assert following["private"]["shed"]["WHEAT"] == 1 - carried_wheat


def test_actor_export_strips_all_critic_and_predictor_recovery_state() -> None:
    config = ModelConfig(
        cnn_width=8,
        cnn_blocks=1,
        model_dim=16,
        transformer_layers=3,
        attention_heads=2,
    )
    actor = FarmActor(config)
    training_only = {
        "critic",
        "actor_optimizer",
        "critic_optimizer",
        "structured_dynamics",
        "structured_dynamics_optimizer",
        "structured_critic_dynamics",
        "structured_critic_dynamics_optimizer",
    }
    training_state = {key: {"training_only": torch.tensor(1.0)} for key in training_only}
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": config.to_dict(),
        "source_identity": source_identity(),
    }
    checkpoints = (
        ({**metadata, "actor": actor.state_dict(), **training_state}, None),
        (
            {
                **metadata,
                "agents": [
                    {"actor": actor.state_dict(), **training_state},
                    {"actor": actor.state_dict(), **training_state},
                ],
            },
            0,
        ),
    )

    for checkpoint, agent in checkpoints:
        artifact = actor_artifact_from_checkpoint(checkpoint, agent=agent)
        assert training_only.isdisjoint(artifact)
        assert set(artifact) == {
            "format_version",
            "architecture",
            "model_config",
            "actor",
            "iteration",
            "metrics",
            "source_identity",
            "run_provenance",
            "orientation",
        }


def test_fused_structured_artifact_rejects_cpu_inference(tmp_path: Path) -> None:
    config = StructuredConfig(
        model_dim=128,
        attention_heads=8,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
        fused_mlp=True,
    )
    actor = StructuredActor(config)
    path = tmp_path / "fused.pt"
    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "architecture": "structured",
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "source_identity": source_identity(),
        },
        path,
    )

    with pytest.raises(ValueError, match="require CUDA inference"):
        load_actor_artifact(path, device="cpu")


def test_portable_structured_import_tolerates_triton_without_tensor_descriptor() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['triton.tools.tensor_descriptor'] = None; "
                "import kaggriculture.structured"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_fused_structured_artifact_exports_to_cpu_linear_layout(tmp_path: Path) -> None:
    config = StructuredConfig(
        model_dim=128,
        attention_heads=8,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
        fused_mlp=True,
    )
    fused = StructuredActor(config)
    artifact = cpu_portable_actor_artifact(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": "structured",
            "model_config": config.to_dict(),
            "actor": fused.state_dict(),
            "iteration": 7,
            "source_identity": source_identity(),
        }
    )
    assert all(tensor.device.type == "cpu" for tensor in artifact["actor"].values())
    path = tmp_path / "portable.pt"
    torch.save(artifact, path)
    portable, metadata = load_actor_artifact(path, device="cpu")

    assert metadata["model_config"]["fused_mlp"] is False
    assert metadata["iteration"] == 7
    portable_state = portable.state_dict()
    for name, value in fused.state_dict().items():
        if name.endswith(".ffn.up_weight"):
            prefix = name.removesuffix("up_weight")
            assert torch.equal(portable_state[f"{prefix}input.weight"], value)
            assert torch.count_nonzero(portable_state[f"{prefix}input.bias"]) == 0
        elif name.endswith(".ffn.down_weight"):
            prefix = name.removesuffix("down_weight")
            assert torch.equal(portable_state[f"{prefix}output.weight"], value.T)
            assert torch.count_nonzero(portable_state[f"{prefix}output.bias"]) == 0
        else:
            assert torch.equal(portable_state[name], value)


def test_fused_structured_state_rejects_hybrid_portable_keys() -> None:
    config = StructuredConfig(
        model_dim=128,
        attention_heads=8,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
        fused_mlp=True,
    )
    state = dict(StructuredActor(config).state_dict())
    state["trunk.farm_local.0.ffn.input.weight"] = torch.zeros(256, 128)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        _cpu_portable_structured_state(config.to_dict(), state)


@pytest.mark.parametrize(
    "checkpoint_version",
    [*sorted(LEGACY_CHECKPOINT_FORMAT_VERSIONS), CHECKPOINT_FORMAT_VERSION],
)
def test_full_checkpoint_loads_directly_as_actor(tmp_path: Path, checkpoint_version: int) -> None:
    config = ModelConfig(
        cnn_width=8,
        cnn_blocks=1,
        model_dim=16,
        transformer_layers=3,
        attention_heads=2,
    )
    actor = FarmActor(config)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": checkpoint_version,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "source_identity": source_identity(),
        },
        path,
    )

    restored, metadata = load_actor_artifact(path)

    assert metadata["format_version"] == checkpoint_version
    for expected, actual in zip(actor.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("calibrated", [False, True])
def test_submission_bundle_is_isolated_complete_and_within_action_timeout(
    tmp_path: Path,
    calibrated: bool,
) -> None:
    config = ModelConfig()
    actor = FarmActor(config)
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-50.0)
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 50.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[:, -1] = 50.0
    checkpoint = tmp_path / "checkpoint.pt"
    archive = tmp_path / "submission.tar.gz"
    run_provenance = None
    if calibrated:
        run_provenance = run_provenance_from_decision(
            {
                "source_identity": source_identity(),
                "rollout_forward_mode": "eager",
                "update_compile_mode": "eager",
                "eager_report_sha256": "a" * 64,
                "eager_report_size_bytes": 100,
                "mixed_report_sha256": "c" * 64,
                "mixed_report_size_bytes": 110,
                "compiled_report_sha256": "b" * 64,
                "compiled_report_size_bytes": 120,
                "minimum_compile_speedup": 1.05,
                "attributed_knob_speedups": {
                    "rollout_forward_mode": 1.0,
                    "update_compile_mode": 1.0,
                },
            },
        )
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 17,
            "source_identity": source_identity(),
            "run_provenance": run_provenance,
            "seed_usage": [],
        },
        checkpoint,
    )
    repository = Path(__file__).parents[1]
    evaluation = tmp_path / "evaluation.json"
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    opponent_provenance = {
        "kind": "python_file",
        "path": "/var/tmp/public-v27.py",
        "sha256": "c" * 64,
        "size_bytes": 100,
    }
    evaluation.write_text(
        json.dumps(
            _evaluation_protocol(
                {
                    "valid_for_selection": True,
                    "device": "cpu",
                    "opponent_label": "public-v27",
                    "seed_count": 128,
                    "seed_start": 12_000_000,
                    "paired_seats": True,
                    "summary": {"score_rate": 0.75},
                    "selection_provenance": {
                        "best_output_sha256": checkpoint_digest,
                        "sha256": "d" * 64,
                        "run_provenance": run_provenance,
                        "screening_seed_start": 10_000_000,
                        "screening_seed_count": 32,
                        "opponent_provenance": {
                            "public-v27": {
                                "kind": "python_file",
                                "sha256": "c" * 64,
                                "size_bytes": 100,
                            }
                        },
                    },
                    "opponent_provenance": opponent_provenance,
                    "artifact_provenance": {
                        "sha256": checkpoint_digest,
                        "source_identity": source_identity(),
                        "run_provenance": run_provenance,
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    starter_evaluation = tmp_path / "starter.json"
    starter_evaluation.write_text(
        json.dumps(
            _evaluation_protocol(
                {
                    "valid_for_selection": True,
                    "device": "cpu",
                    "opponent_label": "starter",
                    "seed_count": 64,
                    "paired_seats": True,
                    "summary": {"score_rate": 1.0},
                    "artifact_provenance": {
                        "sha256": checkpoint_digest,
                        "source_identity": source_identity(),
                        "run_provenance": run_provenance,
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "build_submission.py"),
            "--checkpoint",
            str(checkpoint),
            "--evaluation-report",
            str(evaluation),
            "--builtin-evaluation-report",
            str(starter_evaluation),
            "--output",
            str(archive),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    required = {
        "main.py",
        "model.pt",
        "evaluation.json",
        "manifest.json",
        "kaggriculture/__init__.py",
        "kaggriculture/actions.py",
        "kaggriculture/constants.py",
        "kaggriculture/encoding.py",
        "kaggriculture/inference.py",
        "kaggriculture/evaluation.py",
        "kaggriculture/model.py",
        "kaggriculture/orientation.py",
        "kaggriculture/policy.py",
        "kaggriculture/provenance.py",
        "kaggriculture/registry.py",
        "kaggriculture/structured.py",
        "kaggriculture/tokens.py",
        "kaggriculture/triton_mlp.py",
    }
    with tarfile.open(archive, "r:gz") as bundle:
        assert set(bundle.getnames()) == required
        bundle.extractall(tmp_path / "extracted", filter="data")
    # Keep the actor-only artifact compact without constraining worthwhile
    # policy capacity to the former CNN's incidental five-megabyte footprint.
    assert archive.stat().st_size < 8_000_000

    probe = r"""
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import kaggriculture
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

assert Path(kaggriculture.__file__).resolve().is_relative_to(root)
main_path = root / "main.py"
raw_agent = get_last_callable(main_path.read_text(encoding="utf-8"), path=str(main_path))
environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 991})
observation = environment.reset(2)[0].observation
farm = observation["farms"][0]
farm["hands"] = [[4 + index % 2, 4 + index // 2 % 2] for index in range(15)]
farm["money"] = 1_000_000_000
farm["hires_today"] = 15
observation["private"]["inventories"] = [{} for _ in range(16)]

elapsed = []
for _ in range(5):
    started = time.perf_counter()
    action = raw_agent(observation)
    elapsed.append(time.perf_counter() - started)
assert len(action["hands"]) == 15
assert len(action["market"]) == 10
assert action["market"] == [["BUY_SEED", "WHEAT", 100]] * 10
assert max(elapsed) < 1.0
print(json.dumps({"max_action_seconds": max(elapsed), "action": action}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path / "extracted")],
        cwd=tmp_path / "extracted",
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Kaggle's import path may log framework diagnostics to stdout before the
    # probe result, so treat the final non-empty line as the machine payload.
    result = json.loads([line for line in completed.stdout.splitlines() if line][-1])
    assert result["max_action_seconds"] < 1.0


def _build_submission_module():
    path = Path(__file__).parents[1] / "scripts" / "build_submission.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_build_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_submission_module():
    path = Path(__file__).parents[1] / "scripts" / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_validate_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_summary(start: int, count: int, score: float) -> dict:
    from kaggriculture.evaluation import SCORE_CONFIDENCE, bounded_mean_interval

    return {
        "score_rate": score,
        "score_rate_95ci": list(bounded_mean_interval([score] * count)),
        "score_confidence": SCORE_CONFIDENCE,
        "seed_cluster_statistics": [
            {"seed": seed, "score_rate": score, "mean_margin": 0.0}
            for seed in range(start, start + count)
        ],
    }


def _evaluation_protocol(payload: dict) -> dict:
    from kaggriculture.evaluation import seed_protocol

    payload["artifact_provenance"]["seed_usage"] = []
    finalist = payload["opponent_label"] == "public-v27"
    start = payload.setdefault("seed_start", 12_000_000 if finalist else 4_000_000)
    count = payload["seed_count"]
    payload["summary"] = _score_summary(start, count, payload["summary"]["score_rate"])
    usage = []
    if finalist:
        selection = payload["selection_provenance"]
        selection["statistical_selection"] = {
            "protocol": "archive_screening_then_untouched_finalist",
            "candidate_count": 2,
            "finalist_required": True,
            "candidate_frozen_before_finalist": True,
        }
        selection["seed_protocol"] = seed_protocol(
            "screening",
            selection["screening_seed_start"],
            selection["screening_seed_count"],
            usage=[],
        )
        usage.append(selection["seed_protocol"]["evaluation"])
    payload["seed_protocol"] = seed_protocol(
        "finalist" if finalist else "development",
        start,
        count,
        usage=usage,
    )
    return payload


def _submission_inputs(tmp_path: Path) -> tuple[Path, dict, dict]:
    """A checkpoint and the two reports a submission needs, all mutually bound."""
    config = ModelConfig()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": FarmActor(config).state_dict(),
            "iteration": 5,
            "source_identity": source_identity(),
            "run_provenance": None,
            "seed_usage": [],
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    provenance = {
        "sha256": digest,
        "source_identity": source_identity(),
        "run_provenance": None,
    }
    finalist = {
        "valid_for_selection": True,
        "device": "cpu",
        "opponent_label": "public-v27",
        "seed_count": 128,
        "seed_start": 12_000_000,
        "paired_seats": True,
        "summary": {"score_rate": 0.75},
        "selection_provenance": {
            "best_output_sha256": digest,
            "sha256": "d" * 64,
            "run_provenance": None,
            "screening_seed_start": 10_000_000,
            "screening_seed_count": 32,
            "opponent_provenance": {
                "public-v27": {"kind": "python_file", "sha256": "c" * 64, "size_bytes": 100}
            },
        },
        "opponent_provenance": {
            "kind": "python_file",
            "path": "/var/tmp/public-v27.py",
            "sha256": "c" * 64,
            "size_bytes": 100,
        },
        "artifact_provenance": provenance,
    }
    starter = {
        "valid_for_selection": True,
        "device": "cpu",
        "opponent_label": "starter",
        "seed_count": 64,
        "paired_seats": True,
        "summary": {"score_rate": 1.0},
        "artifact_provenance": provenance,
    }
    return checkpoint, _evaluation_protocol(finalist), _evaluation_protocol(starter)


def test_submission_reports_bind_the_exported_population_member(tmp_path: Path) -> None:
    builder = _build_submission_module()
    checkpoint, finalist, starter = _submission_inputs(tmp_path)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    finalist["artifact_provenance"]["agent"] = 0
    assert builder._evaluation_agent(json.dumps(finalist).encode()) == 0

    with pytest.raises(ValueError, match="different population member"):
        builder._load_evaluation(
            json.dumps(finalist).encode(),
            digest,
            source_identity(),
            0.0,
            1,
        )
    with pytest.raises(ValueError, match="different population member"):
        builder._load_builtin_evaluation(
            json.dumps(starter).encode(),
            digest,
            source_identity(),
            0.0,
            1,
            1,
        )


def test_submission_refuses_an_agent_that_loses(tmp_path: Path) -> None:
    """The provenance gate cannot see strength, and both shipped finalists prove it.

    `evaluations/vapo-lv2-iter415-finalist-v27.json` and its `vapo-main` sibling
    each record `score_rate` 0.0 with 0 wins over 256 seats while stamped
    `valid_for_selection: True`. That flag means the evaluation ran, so a gate
    reading only provenance packaged agents that never won a game.
    """
    build_submission = _build_submission_module()
    checkpoint, finalist, starter = _submission_inputs(tmp_path)
    finalist_path = tmp_path / "finalist.json"
    starter_path = tmp_path / "starter.json"
    output = tmp_path / "submission.tar.gz"

    def write(finalist_payload: dict, starter_payload: dict) -> None:
        finalist_path.write_text(json.dumps(finalist_payload), encoding="utf-8")
        starter_path.write_text(json.dumps(starter_payload), encoding="utf-8")

    def attempt(**overrides) -> dict:
        return build_submission.build(
            checkpoint,
            finalist_path,
            output,
            builtin_evaluation_reports=[starter_path],
            minimum_score_rate=0.5,
            minimum_builtin_score_rate=0.9,
            minimum_builtin_seed_count=32,
            **overrides,
        )

    write(finalist, starter)
    manifest = attempt()
    assert manifest["evaluation"]["score_rate"] == 0.75
    assert manifest["strength_gate"]["builtin_score_rates"] == {"starter": 1.0}

    write({**finalist, "device": "cuda"}, starter)
    with pytest.raises(ValueError, match="finalist evaluation must run on CPU"):
        attempt()
    write(finalist, {**starter, "device": "cuda"})
    with pytest.raises(ValueError, match="built-in evaluation must run on CPU"):
        attempt()

    # The exact historical failure: every seat lost, provenance immaculate.
    write({**finalist, "summary": _score_summary(finalist["seed_start"], 128, 0.0)}, starter)
    with pytest.raises(ValueError, match="below the required"):
        attempt()

    # Losing to the carrot-loop heuristic must block a submission on its own,
    # even when the public-v27 number is healthy.
    write(finalist, {**starter, "summary": _score_summary(starter["seed_start"], 64, 0.4)})
    with pytest.raises(ValueError, match="against the built-in starter"):
        attempt()

    # A tiny sample is not evidence of beating it.
    write(finalist, {**starter, "seed_count": 4})
    with pytest.raises(ValueError, match="seed clusters, below the required"):
        attempt()

    # An aborted evaluation writes a null rate, which must not read as zero or crash.
    write({**finalist, "summary": {"score_rate": None}}, starter)
    with pytest.raises(ValueError, match="complete seed-cluster panel"):
        attempt()

    write(finalist, starter)
    with pytest.raises(ValueError, match="requires an evaluation against the built-in starter"):
        build_submission.build(
            checkpoint,
            finalist_path,
            output,
            builtin_evaluation_reports=[],
            minimum_score_rate=0.5,
        )


def test_submission_rejects_a_finalist_without_selection_provenance(tmp_path: Path) -> None:
    builder = _build_submission_module()
    checkpoint, finalist, _starter = _submission_inputs(tmp_path)
    finalist["selection_provenance"] = None
    with pytest.raises(ValueError, match="checkpoint-selection provenance"):
        builder._load_evaluation(
            json.dumps(finalist).encode(),
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            source_identity(),
            0.0,
            None,
        )


def test_submission_validator_rejects_internally_consistent_cuda_evaluation(
    tmp_path: Path,
) -> None:
    builder = _build_submission_module()
    validator = _validate_submission_module()
    checkpoint, finalist, starter = _submission_inputs(tmp_path)
    finalist_path = tmp_path / "finalist.json"
    starter_path = tmp_path / "starter.json"
    archive_path = tmp_path / "submission.tar.gz"
    finalist_path.write_text(json.dumps(finalist), encoding="utf-8")
    starter_path.write_text(json.dumps(starter), encoding="utf-8")
    builder.build(
        checkpoint,
        finalist_path,
        archive_path,
        builtin_evaluation_reports=[starter_path],
        minimum_score_rate=0.5,
        minimum_builtin_score_rate=0.9,
        minimum_builtin_seed_count=16,
    )

    root = tmp_path / "tampered"
    root.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(root, filter="data")
    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["device"] = "cuda"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    evaluation_digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["sha256"] = evaluation_digest
    manifest["files"]["evaluation.json"] = evaluation_digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered_archive = tmp_path / "cuda-evaluated-submission.tar.gz"
    with tarfile.open(tampered_archive, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root), recursive=False)

    destination = tmp_path / "rejected"
    destination.mkdir()
    with pytest.raises(ValueError, match="did not run on Kaggle's CPU backend"):
        validator._extract(tampered_archive, destination)


def test_submission_equivalence_keeps_evaluations_bound_to_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _build_submission_module()
    checkpoint, finalist, starter = _submission_inputs(tmp_path)
    finalist_path = tmp_path / "finalist.json"
    starter_path = tmp_path / "starter.json"
    witness_path = tmp_path / "equivalence.json"
    finalist_path.write_text(json.dumps(finalist), encoding="utf-8")
    starter_path.write_text(json.dumps(starter), encoding="utf-8")
    artifact_source = source_identity()
    candidate_files = {
        **artifact_source["files"],
        "tests/equivalence-sentinel.py": "0" * 64,
    }
    candidate_source = {
        "format_version": artifact_source["format_version"],
        "sha256": _identity_digest(candidate_files),
        "files": candidate_files,
    }
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    witness = {
        "artifact_sha256": checkpoint_digest,
        "expected_identity": artifact_source["sha256"],
        "candidate_identity": candidate_source["sha256"],
        "surfaces": {
            name: {"equal": True, "reference": name, "candidate": name}
            for name in ("observations", "masks", "logits")
        },
        "games": 1,
        "steps": 1,
        "failures": [],
    }
    witness_path.write_text(json.dumps(witness), encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "require_source_identity",
        lambda *_args, **_kwargs: candidate_source,
    )

    manifest = builder.build(
        checkpoint,
        finalist_path,
        tmp_path / "submission.tar.gz",
        builtin_evaluation_reports=[starter_path],
        minimum_score_rate=0.5,
        minimum_builtin_score_rate=0.9,
        minimum_builtin_seed_count=16,
        inference_equivalence=witness_path,
    )

    assert manifest["source_identity"] == candidate_source
    assert manifest["evaluation"]["score_rate"] == finalist["summary"]["score_rate"]
    extracted = tmp_path / "equivalent-validated"
    extracted.mkdir()
    _, validated_manifest = _validate_submission_module()._extract(
        tmp_path / "submission.tar.gz",
        extracted,
    )
    assert validated_manifest == manifest


def test_weights_load_across_the_provenance_bump_but_do_not_export(tmp_path: Path) -> None:
    """Loading weights and carrying a calibration claim forward are different
    operations, and only the second needs the claim to be interpretable.

    `load_actor_artifact` is the read path for deliberately cross-tree work --
    `--init-actor-from`, replay viewing, behavior audits -- none of which reads
    run provenance. Refusing those over a pre-split calibration record would
    reject good weights for a field the caller never touches. Export is the
    opposite case: it copies provenance into a submission, where an
    uninterpretable claim would be asserted as though it were recoverable.
    """
    config = ModelConfig(cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3)
    actor = FarmActor(config)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": config.to_dict(),
        "actor": actor.state_dict(),
        "source_identity": source_identity(),
        # A well-formed record from before the rollout/update split.
        "run_provenance": {"format_version": 1, "sha256": "a" * 64, "calibration": {}},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    restored, payload = load_actor_artifact(path)
    for expected, actual in zip(actor.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
    # The stale record is dropped from the loader's view, not rewritten on disk.
    assert payload["run_provenance"]["format_version"] == 1

    # Version 2 is dropped on the same read path for a different reason: it did
    # split the decision per knob, but its per-phase speedups were differenced
    # across a pair of runs that moved both knobs at once, so each phase's ratio
    # carries whatever drift that pair happened to have. The configuration that
    # isolates a single knob was never run, so those numbers cannot be
    # re-attributed after the fact and the record is rejected, not migrated.

    with pytest.raises(ValueError, match="carries superseded calibration provenance"):
        actor_artifact_from_checkpoint(checkpoint)
    # Every superseded version is refused at the same boundary, so a checkpoint
    # written by an earlier tree cannot export a decision whose evidence no
    # longer substantiates it. Version 3's rollout knob is a boolean whose
    # only "on" value was `cudagraphs`, so the speedup it certifies belongs to a
    # mode measurement rejects -- 5.309 ms against eager's 4.907 ms on the
    # isolated collection forward -- and no mode can be recovered from `False`.
    # Version 4's update knob is the same defect one phase over: its `true`
    # names all four compiled modes at once, so nothing in the record says which
    # one the speedup beside it was measured under.
    for superseded in (2, 3, 4):
        assert is_legacy_run_provenance(
            {"format_version": superseded, "sha256": "a" * 64, "calibration": {}}
        )
        with pytest.raises(ValueError, match="carries superseded calibration provenance"):
            actor_artifact_from_checkpoint(
                {
                    **checkpoint,
                    "run_provenance": {
                        "format_version": superseded,
                        "sha256": "a" * 64,
                        "calibration": {},
                    },
                }
            )

    # Corruption is not leniency's business: only a well-formed older version
    # is dropped, and anything else still raises on the read path.
    torch.save({**checkpoint, "run_provenance": {"nonsense": True}}, path)
    with pytest.raises(ValueError, match="invalid schema"):
        load_actor_artifact(path)


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4])
def test_actor_artifact_rejects_incompatible_format(tmp_path: Path, version) -> None:
    path = tmp_path / "model.pt"
    torch.save({"format_version": version}, path)

    with pytest.raises(ValueError, match="unsupported actor artifact format"):
        load_actor_artifact(path)


def _bundle(root: Path, main_source: str) -> Path:
    """Assemble a submission bundle by hand, so `main.py` can be made faulty."""
    builder = _load_build_submission()
    package = root / "kaggriculture"
    package.mkdir(parents=True)
    (root / "main.py").write_text(main_source, encoding="utf-8")
    config = ModelConfig()
    actor = FarmActor(config)
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-50.0)
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 50.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[:, -1] = 50.0
    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 3,
            "source_identity": source_identity(),
            "run_provenance": None,
        },
        root / "model.pt",
    )
    for name in builder.PACKAGE_FILES:
        shutil.copy2(Path(__file__).parents[1] / "src" / "kaggriculture" / name, package / name)
    return root


def _load_build_submission():
    spec = importlib.util.spec_from_file_location(
        "build_submission_under_test",
        Path(__file__).parents[1] / "scripts" / "build_submission.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_bundle_smoke_test_refuses_an_agent_that_never_acts(tmp_path: Path) -> None:
    # The expensive failure is quiet: `kaggle_environments` swallows an agent
    # exception, seats PASS for the rest of the episode, and reports DONE with the
    # starting bank intact. On the leaderboard that is indistinguishable from a
    # merely weak submission, and it costs a day of submission budget to learn.
    builder = _load_build_submission()

    playing = _bundle(tmp_path / "playing", builder.MAIN)
    result = builder._smoke_test(playing)
    assert result["status"] == "DONE"
    assert result["submitted"] == builder._SMOKE_STEPS - 1
    assert result["acting"] > 0

    with pytest.raises(ValueError, match="passed on every step"):
        builder._smoke_test(_bundle(tmp_path / "silent", "def agent(obs):\n    return {}\n"))
    with pytest.raises(ValueError, match="failed to run"):
        builder._smoke_test(
            _bundle(tmp_path / "raising", "def agent(obs):\n    raise ValueError('bad')\n")
        )
    # The trap the entrypoint template exists to avoid: `getfullargspec` counts
    # `self`, so a callable object is invoked with two arguments, the TypeError is
    # swallowed, and the seat submits nothing for the whole episode.
    with pytest.raises(ValueError, match="failed to run"):
        builder._smoke_test(
            _bundle(
                tmp_path / "callable",
                "from pathlib import Path\n"
                "import kaggriculture\n"
                "from kaggriculture.inference import CheckpointAgent\n"
                "agent = CheckpointAgent(\n"
                "    Path(kaggriculture.__file__).resolve().parent.parent / 'model.pt'\n"
                ")\n",
            )
        )


def _load_build_plan_submission():
    spec = importlib.util.spec_from_file_location(
        "build_plan_submission_under_test",
        Path(__file__).parents[1] / "scripts" / "build_plan_submission.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_plan_patch_cancels_exactly_the_named_steps(tmp_path: Path) -> None:
    # The failure mode is silent and expensive: a patch that emits
    # `frozenset(220, 224)` raises at import, and one that emits a truthy scalar
    # would cancel the market on every step. Either ships an agent that is not the
    # one the search measured, and the archive still looks well formed.
    builder = _load_build_plan_submission()
    source = tmp_path / "plan.py"
    source.write_text(
        "def _get(obs, key, default=None):\n"
        "    return obs.get(key, default)\n"
        "\n"
        "\n"
        "def agent(obs, configuration=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': [['SELL', 'WHEAT', 1]]}\n",
        encoding="utf-8",
    )
    patched = tmp_path / "main.py"
    patched.write_text(builder.patched_source(source, (2, 5)), encoding="utf-8")

    module = builder._load(patched, "patched_plan_under_test")
    assert frozenset({2, 5}) == module._CANCELLED_MARKET_STEPS
    cancelled = [step for step in range(8) if module.agent({"step": step})["market"] == []]
    assert cancelled == [2, 5]
    # The rest of the action is the plan's own, so the edit cannot be credited
    # with a change it did not make.
    assert module.agent({"step": 3})["market"] == [["SELL", "WHEAT", 1]]

    with pytest.raises(SystemExit, match="already patched"):
        builder.patched_source(patched, (2, 5))
    bare = tmp_path / "bare.py"
    bare.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no top-level agent"):
        builder.patched_source(bare, (2,))


def test_the_plan_patch_shrinks_only_priced_purchases(tmp_path: Path) -> None:
    # Reducing a sell would stop the plan banking its harvest, and reducing a bare
    # `HIRE` would corrupt an order the engine reads positionally. Both are silent:
    # the agent still returns a well formed action and still plays 720 steps.
    builder = _load_build_plan_submission()
    source = tmp_path / "plan.py"
    source.write_text(
        "def _get(obs, key, default=None):\n"
        "    return obs.get(key, default)\n"
        "\n"
        "\n"
        "def agent(obs, configuration=None):\n"
        "    return {\n"
        "        'farmer': ['PASS'],\n"
        "        'hands': [],\n"
        "        'market': [\n"
        "            ['HIRE'],\n"
        "            ['BUY_SEED', 'MELON', 7],\n"
        "            ['SELL', 'WHEAT', 9],\n"
        "            ['BUY_PRODUCT', 'WHEAT', 1],\n"
        "        ],\n"
        "    }\n",
        encoding="utf-8",
    )
    patched = tmp_path / "main.py"
    patched.write_text(builder.patched_source(source, (5,), ((3, 2),)), encoding="utf-8")

    module = builder._load(patched, "reduced_plan_under_test")
    assert module._REDUCED_BUY_STEPS == {3: 2}
    assert module.agent({"step": 3})["market"] == [
        ["HIRE"],
        ["BUY_SEED", "MELON", 5],
        ["SELL", "WHEAT", 9],
        # Floored at one: the engine has no smaller order, so the alternative to a
        # shrink this deep is a cancellation, which is a different edit.
        ["BUY_PRODUCT", "WHEAT", 1],
    ]
    assert module.agent({"step": 4})["market"] == [
        ["HIRE"],
        ["BUY_SEED", "MELON", 7],
        ["SELL", "WHEAT", 9],
        ["BUY_PRODUCT", "WHEAT", 1],
    ]
    assert module.agent({"step": 5})["market"] == []

    with pytest.raises(SystemExit, match="both cancelled and reduced"):
        builder.patched_source(source, (3,), ((3, 2),))
