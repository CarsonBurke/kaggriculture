"""Prevent incomplete evidence or collapsed clones from launching long PPO jobs."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "scripts/admit_core_training.py"
    spec = importlib.util.spec_from_file_location("admit_core_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.admission


@pytest.fixture
def evidence(tmp_path):
    artifact = tmp_path / "clone.pt"
    artifact.write_bytes(b"immutable checkpoint fixture")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    identity = {"sha256": "source"}
    reports = {
        mode: {
            "complete": True,
            "decoding": mode,
            "source_identity": identity,
            "games_per_opponent": 256,
            "artifacts": {
                "control": {
                    "path": str(artifact),
                    "sha256": digest,
                    "source_identity": identity,
                    "panels": {
                        opponent: {
                            "games": [
                                {"seed": seed, "seat": seed % 2, "score": 1.0, "money": 80000}
                                for seed in range(256)
                            ]
                        }
                        for opponent in ("starter", "scripted-v27", "scripted-v16")
                    },
                }
            },
        }
        for mode in ("argmax", "sampled")
    }
    return reports, artifact


def test_competent_clone_is_admitted(gate, evidence):
    reports, artifact = evidence
    assert gate(reports, variant="control", artifact=artifact, source_digest="source")["admitted"]


def test_current_native_panel_is_admitted(gate, evidence):
    reports, artifact = evidence
    for report in reports.values():
        del report["artifacts"]["control"]["panels"]["scripted-v16"]
    assert gate(reports, variant="control", artifact=artifact, source_digest="source")["admitted"]


@pytest.mark.parametrize("opponent", ["starter", "scripted-v16"])
def test_missing_or_mismatched_opponents_are_rejected(gate, evidence, opponent):
    reports, artifact = evidence
    del reports["sampled"]["artifacts"]["control"]["panels"][opponent]
    with pytest.raises(ValueError, match="opponents"):
        gate(reports, variant="control", artifact=artifact, source_digest="source")


def test_sampled_collapse_blocks_otherwise_strong_greedy_clone(gate, evidence):
    reports, artifact = evidence
    rows = reports["sampled"]["artifacts"]["control"]["panels"]["scripted-v16"]["games"]
    for row in rows[:100]:
        row["money"] = 50
    result = gate(reports, variant="control", artifact=artifact, source_digest="source")
    assert not result["admitted"]
    assert result["failures"] == ["sampled scripted-v16 bankrupt tail above 25%"]


@pytest.mark.parametrize("defect", ["incomplete", "changed_weights", "duplicate", "nonfinite"])
def test_invalid_evidence_fails_closed(gate, evidence, defect):
    reports, artifact = evidence
    rows = reports["sampled"]["artifacts"]["control"]["panels"]["starter"]["games"]
    if defect == "incomplete":
        reports["sampled"]["complete"] = False
    elif defect == "changed_weights":
        artifact.write_bytes(b"different checkpoint")
    elif defect == "duplicate":
        rows[1] = dict(rows[0])
    else:
        rows[0]["money"] = float("nan")
    with pytest.raises(ValueError):
        gate(reports, variant="control", artifact=artifact, source_digest="source")
