from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggriculture.provenance import (
    freeze_source,
    require_source_identity,
    run_provenance_from_decision,
    source_identity,
    validate_run_provenance,
    validate_source_identity,
)


def _minimal_source(root: Path) -> None:
    for relative, contents in {
        "pyproject.toml": "[project]\nname='test'\n",
        "uv.lock": "version = 1\n",
        "src/kaggriculture/module.py": "VALUE = 1\n",
        "scripts/train.py": "print('train')\n",
        "rust/kagg_env/Cargo.toml": "[package]\nname='test'\n",
        "rust/kagg_env/Cargo.lock": "version = 4\n",
        "rust/kagg_env/pyproject.toml": "[build-system]\n",
        "rust/kagg_env/src/lib.rs": "pub fn value() -> u8 { 1 }\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def test_source_identity_detects_content_and_path_changes(tmp_path: Path) -> None:
    _minimal_source(tmp_path)
    original = source_identity(tmp_path)

    require_source_identity(original, tmp_path)
    (tmp_path / "scripts/train.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"scripts/train\.py"):
        require_source_identity(original, tmp_path)

    malformed = dict(original)
    malformed["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="aggregate"):
        validate_source_identity(malformed)


def test_freeze_source_is_exact_read_only_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _minimal_source(source)
    expected = source_identity(source)
    destination = tmp_path / "snapshots" / expected["sha256"]

    assert freeze_source(destination, source) == expected
    assert freeze_source(destination, source) == expected
    assert source_identity(destination) == expected
    assert json.loads((destination / ".source-identity.json").read_text()) == expected
    assert (destination / "scripts/train.py").stat().st_mode & 0o222 == 0
    changed = destination / "scripts/train.py"
    changed.chmod(0o644)
    with pytest.raises(PermissionError, match="permissions are mutable"):
        freeze_source(destination, source)


def test_run_provenance_is_portable_canonical_and_tamper_evident(tmp_path: Path) -> None:
    _minimal_source(tmp_path)
    identity = source_identity(tmp_path)
    provenance = run_provenance_from_decision(
        {
            "source_identity": identity,
            "eager_report_sha256": "a" * 64,
            "eager_report_size_bytes": 100,
            "compiled_report_sha256": "b" * 64,
            "compiled_report_size_bytes": 120,
            "compile_models": True,
            "minimum_compile_speedup": 1.05,
            "measured_compile_speedup": 1.2,
            "training_command": ["intentionally", "excluded"],
            "run_dir": "/also/excluded",
        }
    )

    assert validate_run_provenance(provenance, required=True) == provenance
    assert "training_command" not in provenance["calibration"]
    tampered = provenance | {"calibration": provenance["calibration"] | {"compile_models": False}}
    with pytest.raises(ValueError, match=r"contradicts|digest"):
        validate_run_provenance(tampered, required=True)
