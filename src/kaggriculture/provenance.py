"""Content-addressed source provenance for training and submission artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_IDENTITY_FORMAT_VERSION = 1
RUN_PROVENANCE_FORMAT_VERSION = 1
_ROOT_FILES = ("pyproject.toml", "uv.lock")
_SOURCE_ROOTS = ("src/kaggriculture", "scripts", "rust/kagg_env")
_EXCLUDED_DIRECTORIES = frozenset(("__pycache__", "target"))
_EXCLUDED_SUFFIXES = frozenset((".pyc", ".pyo"))


def repository_root() -> Path:
    """Return the checkout or frozen-source root containing this package."""
    return Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    """Hash one regular file without following a mutable logical identity."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source provenance requires a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_paths(root: Path) -> list[Path]:
    root = root.resolve()
    paths = {root / name for name in _ROOT_FILES}
    for relative_root in _SOURCE_ROOTS:
        source_root = root / relative_root
        if not source_root.is_dir():
            raise FileNotFoundError(f"source provenance root is missing: {source_root}")
        for path in source_root.rglob("*"):
            relative = path.relative_to(source_root)
            if any(part in _EXCLUDED_DIRECTORIES for part in relative.parts):
                continue
            if path.is_file() and path.suffix not in _EXCLUDED_SUFFIXES:
                paths.add(path)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = sorted(map(str, missing))
        raise FileNotFoundError(f"source provenance inputs are missing: {rendered}")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _identity_digest(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"kaggriculture-source-v{SOURCE_IDENTITY_FORMAT_VERSION}\0".encode())
    for relative, content_digest in sorted(files.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(content_digest))
    return digest.hexdigest()


def validate_source_identity(value: object) -> dict[str, Any]:
    """Validate and normalize a serialized source identity."""
    if not isinstance(value, dict) or set(value) != {"format_version", "sha256", "files"}:
        raise ValueError("source identity has an invalid schema")
    if value["format_version"] != SOURCE_IDENTITY_FORMAT_VERSION:
        raise ValueError(
            f"unsupported source identity format: {value['format_version']}; "
            f"expected {SOURCE_IDENTITY_FORMAT_VERSION}"
        )
    files = value["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("source identity must contain a non-empty file manifest")
    normalized_files: dict[str, str] = {}
    for relative, digest in files.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("source identity contains an invalid path")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError(f"source identity contains an unsafe path: {relative!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"source identity contains an invalid digest for {relative}")
        normalized_files[relative] = digest
    expected = _identity_digest(normalized_files)
    if value["sha256"] != expected:
        raise ValueError("source identity aggregate digest does not match its file manifest")
    return {
        "format_version": SOURCE_IDENTITY_FORMAT_VERSION,
        "sha256": expected,
        "files": dict(sorted(normalized_files.items())),
    }


def _hex_digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _positive_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be numeric")
    converted = float(value)
    if not converted > 0.0 or converted == float("inf"):
        raise ValueError(f"{context} must be finite and positive")
    return converted


def _run_digest(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def validate_run_provenance(value: object, *, required: bool = False) -> dict[str, Any] | None:
    """Validate the calibration decision embedded into production checkpoints."""
    if value is None and not required:
        return None
    if not isinstance(value, dict) or set(value) != {
        "format_version",
        "sha256",
        "source_identity",
        "calibration",
    }:
        raise ValueError("run provenance has an invalid schema")
    if value["format_version"] != RUN_PROVENANCE_FORMAT_VERSION:
        raise ValueError(f"unsupported run provenance format: {value['format_version']}")
    identity = validate_source_identity(value["source_identity"])
    calibration = value["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "eager_report",
        "compiled_report",
        "compile_models",
        "minimum_compile_speedup",
        "measured_compile_speedup",
    }:
        raise ValueError("run provenance calibration has an invalid schema")
    reports = {}
    for name in ("eager_report", "compiled_report"):
        report = calibration[name]
        if not isinstance(report, dict) or set(report) != {"sha256", "size_bytes"}:
            raise ValueError(f"run provenance {name} has an invalid schema")
        if type(report["size_bytes"]) is not int or report["size_bytes"] <= 0:
            raise ValueError(f"run provenance {name} size must be a positive integer")
        reports[name] = {
            "sha256": _hex_digest(report["sha256"], f"run provenance {name} digest"),
            "size_bytes": report["size_bytes"],
        }
    if type(calibration["compile_models"]) is not bool:
        raise ValueError("run provenance compile decision must be boolean")
    minimum_speedup = _positive_number(
        calibration["minimum_compile_speedup"],
        "minimum compile speedup",
    )
    measured_speedup = _positive_number(
        calibration["measured_compile_speedup"],
        "measured compile speedup",
    )
    if calibration["compile_models"] != (measured_speedup >= minimum_speedup):
        raise ValueError("run provenance compile decision contradicts measured speedup")
    normalized = {
        "format_version": RUN_PROVENANCE_FORMAT_VERSION,
        "source_identity": identity,
        "calibration": {
            **reports,
            "compile_models": calibration["compile_models"],
            "minimum_compile_speedup": minimum_speedup,
            "measured_compile_speedup": measured_speedup,
        },
    }
    expected = _run_digest(normalized)
    if value["sha256"] != expected:
        raise ValueError("run provenance digest does not match its canonical calibration")
    return normalized | {"sha256": expected}


def run_provenance_from_decision(decision: object) -> dict[str, Any]:
    """Extract portable, path-independent calibration evidence from a launch decision."""
    if not isinstance(decision, dict):
        raise ValueError("calibration decision must be an object")
    payload = {
        "format_version": RUN_PROVENANCE_FORMAT_VERSION,
        "source_identity": decision.get("source_identity"),
        "calibration": {
            "eager_report": {
                "sha256": decision.get("eager_report_sha256"),
                "size_bytes": decision.get("eager_report_size_bytes"),
            },
            "compiled_report": {
                "sha256": decision.get("compiled_report_sha256"),
                "size_bytes": decision.get("compiled_report_size_bytes"),
            },
            "compile_models": decision.get("compile_models"),
            "minimum_compile_speedup": decision.get("minimum_compile_speedup"),
            "measured_compile_speedup": decision.get("measured_compile_speedup"),
        },
    }
    payload["sha256"] = _run_digest(payload)
    validated = validate_run_provenance(payload, required=True)
    assert validated is not None
    return validated


def source_identity(root: Path | None = None) -> dict[str, Any]:
    """Hash every Python, native, build, and dependency input used by the pipeline."""
    resolved_root = repository_root() if root is None else Path(root).resolve()
    files = {
        path.relative_to(resolved_root).as_posix(): file_sha256(path)
        for path in _source_paths(resolved_root)
    }
    return validate_source_identity(
        {
            "format_version": SOURCE_IDENTITY_FORMAT_VERSION,
            "sha256": _identity_digest(files),
            "files": files,
        }
    )


def require_source_identity(expected: object, root: Path | None = None) -> dict[str, Any]:
    """Require the current checkout to exactly match a serialized identity."""
    normalized = validate_source_identity(expected)
    current = source_identity(root)
    if current != normalized:
        expected_files = normalized["files"]
        current_files = current["files"]
        changed = sorted(
            relative
            for relative in set(expected_files) | set(current_files)
            if expected_files.get(relative) != current_files.get(relative)
        )
        preview = ", ".join(changed[:8])
        suffix = " ..." if len(changed) > 8 else ""
        raise ValueError(
            "source tree does not match the bound artifact identity "
            f"({normalized['sha256']} != {current['sha256']}): {preview}{suffix}"
        )
    return current


def freeze_source(destination: Path, root: Path | None = None) -> dict[str, Any]:
    """Atomically materialize a read-only source tree with a verified identity."""
    source_root = repository_root() if root is None else Path(root).resolve()
    identity = source_identity(source_root)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(destination)
        require_source_identity(identity, destination)
        identity_path = destination / ".source-identity.json"
        if (
            not identity_path.is_file()
            or validate_source_identity(json.loads(identity_path.read_text(encoding="utf-8")))
            != identity
        ):
            raise ValueError("frozen source identity record is missing or inconsistent")
        actual_files = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and path.name != ".source-identity.json"
        }
        if actual_files != set(identity["files"]):
            raise ValueError("frozen source contains files outside its source identity")
        paths = sorted(destination.rglob("*"), reverse=True)
        for path in paths:
            expected_mode = 0o555 if path.is_dir() else 0o444
            if path.stat().st_mode & 0o777 != expected_mode:
                raise PermissionError(f"frozen source permissions are mutable: {path}")
        if destination.stat().st_mode & 0o777 != 0o555:
            raise PermissionError(f"frozen source root permissions are mutable: {destination}")
        return identity

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    installed = False
    try:
        for relative in identity["files"]:
            source = source_root / relative
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        require_source_identity(identity, temporary)
        (temporary / ".source-identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in sorted(temporary.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
        installed = True
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)
    return identity
