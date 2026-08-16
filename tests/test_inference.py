from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch

from kaggriculture.actions import MarketKind
from kaggriculture.inference import (
    ACTOR_ARTIFACT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION,
    LEGACY_CHECKPOINT_FORMAT_VERSIONS,
    actor_artifact_from_checkpoint,
    load_actor_artifact,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.provenance import (
    is_legacy_run_provenance,
    run_provenance_from_decision,
    source_identity,
)


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
                "compile_rollout": False,
                "compile_update": False,
                "eager_report_sha256": "a" * 64,
                "eager_report_size_bytes": 100,
                "mixed_report_sha256": "c" * 64,
                "mixed_report_size_bytes": 110,
                "compiled_report_sha256": "b" * 64,
                "compiled_report_size_bytes": 120,
                "minimum_compile_speedup": 1.05,
                "attributed_knob_speedups": {"compile_rollout": 1.0, "compile_update": 1.0},
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
            {
                "valid_for_selection": True,
                "opponent_label": "public-v27",
                "seed_count": 128,
                "seed_start": 20_000_000,
                "paired_seats": True,
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
        "kaggriculture/model.py",
        "kaggriculture/policy.py",
        "kaggriculture/provenance.py",
        "kaggriculture/registry.py",
        "kaggriculture/structured.py",
        "kaggriculture/tokens.py",
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
    assert is_legacy_run_provenance({"format_version": 2, "sha256": "a" * 64, "calibration": {}})

    with pytest.raises(ValueError, match="carries superseded calibration provenance"):
        actor_artifact_from_checkpoint(checkpoint)
    # And version 2 is refused at the same boundary, so a checkpoint written by
    # the immediately preceding tree cannot export a decision whose evidence no
    # longer substantiates it.
    with pytest.raises(ValueError, match="carries superseded calibration provenance"):
        actor_artifact_from_checkpoint(
            {
                **checkpoint,
                "run_provenance": {"format_version": 2, "sha256": "a" * 64, "calibration": {}},
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
