"""Build and load the in-process batched Rust Kaggriculture simulator."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE_NAME = "_kagg_env"
_DYNAMIC_LIBRARY_SUFFIXES = frozenset({".dll", ".dylib", ".pyd", ".so"})
_LOAD_LOCK = threading.RLock()
_MODULE_CACHE: dict[bool, ModuleType] = {}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _local_crate() -> Path | None:
    crate = _repository_root() / "rust" / "kagg_env"
    return crate if (crate / "Cargo.toml").is_file() else None


def _profile_name(release: bool) -> str:
    return "release" if release else "debug"


def _fallback_artifacts(crate: Path, release: bool) -> list[Path]:
    """Return likely artifacts without invoking Cargo, for ``build=False``."""
    profile = _profile_name(release)
    configured = os.environ.get("CARGO_TARGET_DIR")
    target = Path(configured).expanduser() if configured else crate / "target"
    if not target.is_absolute():
        target = crate / target
    names = ("lib_kagg_env.so", "lib_kagg_env.dylib", "_kagg_env.dll", "_kagg_env.pyd")
    direct = [target / profile / name for name in names]
    cross_compiled = [path for name in names for path in target.glob(f"*/{profile}/{name}")]
    return direct + sorted(cross_compiled, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def _cargo_diagnostic(events: list[dict[str, Any]]) -> str:
    rendered = [
        str(message)
        for event in events
        if event.get("reason") == "compiler-message"
        and (message := event.get("message", {}).get("rendered"))
    ]
    return "".join(rendered).strip()


def _build_native(crate: Path, release: bool) -> Path:
    """Build the cdylib and return the exact artifact path reported by Cargo."""
    manifest = crate / "Cargo.toml"
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(manifest),
        "--lib",
        "--message-format=json-render-diagnostics",
    ]
    if release:
        command.append("--release")
    try:
        completed = subprocess.run(
            command,
            cwd=crate,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"cannot build {_MODULE_NAME}: Cargo was not found; install a Rust toolchain "
            f"or call load_native(build=False) with an installed extension"
        ) from error

    events: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    if completed.returncode != 0:
        details = "\n".join(
            part for part in (_cargo_diagnostic(events), completed.stderr.strip()) if part
        )
        suffix = f"\n{details}" if details else ""
        raise RuntimeError(
            f"Cargo failed to build {_MODULE_NAME} from {manifest} "
            f"(exit status {completed.returncode}){suffix}"
        )

    artifact: Path | None = None
    resolved_manifest = manifest.resolve()
    for event in events:
        target = event.get("target", {})
        manifest_path = event.get("manifest_path")
        if (
            event.get("reason") != "compiler-artifact"
            or target.get("name") != _MODULE_NAME
            or "cdylib" not in target.get("crate_types", ())
            or not manifest_path
            or Path(manifest_path).resolve() != resolved_manifest
        ):
            continue
        for filename in event.get("filenames", ()):
            candidate = Path(filename)
            if candidate.suffix in _DYNAMIC_LIBRARY_SUFFIXES:
                artifact = candidate
                break
    if artifact is None or not artifact.is_file():
        raise RuntimeError(
            f"Cargo reported a successful build for {manifest} but did not report a usable cdylib"
        )
    return artifact


def _load_path(path: Path) -> ModuleType:
    if not path.is_file():
        raise ImportError(f"native extension artifact does not exist: {path}")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Python cannot create an extension loader for native artifact {path}")
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:
        raise ImportError(f"failed to load native extension from {path}: {error}") from error
    return module


def _load_installed() -> ModuleType:
    try:
        return importlib.import_module(_MODULE_NAME)
    except ImportError as error:
        raise ImportError(
            f"{_MODULE_NAME} is unavailable: no local Rust crate/artifact was selected and the "
            "extension is not importable; use an editable checkout with Cargo installed or "
            "install a package containing the native extension"
        ) from error


def _validate_module(module: ModuleType, origin: str) -> ModuleType:
    if getattr(module, "BatchEnv", None) is None:
        raise ImportError(f"native extension loaded from {origin} does not expose BatchEnv")
    return module


def load_native(*, build: bool = True, release: bool = True) -> ModuleType:
    """Return the process-cached native extension.

    In an editable checkout, the first call with ``build=True`` asks Cargo to build the local
    cdylib. Cargo, rather than timestamp heuristics, determines whether every build input is
    current and reports the exact artifact path (including custom target directories). Subsequent
    calls return the same initialized module because native extensions cannot be safely unloaded
    or hot-reloaded; restart the process after editing Rust sources.

    With ``build=False``, an existing local artifact is preferred and an installed ``_kagg_env``
    module is the fallback. Calls are serialized within the process; Cargo supplies the build lock
    between processes.
    """
    with _LOAD_LOCK:
        cached = _MODULE_CACHE.get(release)
        if cached is not None:
            return cached

        crate = _local_crate()
        if crate is not None and build:
            artifact = _build_native(crate, release)
            module = _validate_module(_load_path(artifact), str(artifact))
        elif crate is not None:
            artifact = next(
                (
                    candidate
                    for candidate in _fallback_artifacts(crate, release)
                    if candidate.is_file()
                ),
                None,
            )
            if artifact is None:
                module = _validate_module(_load_installed(), "installed module")
            else:
                module = _validate_module(_load_path(artifact), str(artifact))
        else:
            module = _validate_module(_load_installed(), "installed module")

        _MODULE_CACHE[release] = module
        return module
