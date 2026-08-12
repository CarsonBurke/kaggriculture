"""Build and load the in-process batched Rust Kaggriculture simulator."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _crate_sources(crate: Path) -> list[Path]:
    return [
        *(path for path in (crate / "Cargo.toml", crate / "Cargo.lock") if path.exists()),
        *sorted((crate / "src").rglob("*.rs")),
    ]


def _artifact_is_stale(artifact: Path, sources: list[Path]) -> bool:
    if not artifact.is_file():
        return True
    modified = artifact.stat().st_mtime_ns
    return any(source.stat().st_mtime_ns > modified for source in sources)


def _load_path(path: Path) -> ModuleType | None:
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_kagg_env", path)
    if spec is None or spec.loader is None:
        return None
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        return None
    return module


def load_native(*, build: bool = True, release: bool = True) -> ModuleType:
    """Load the native extension, rebuilding it when Rust sources are newer."""
    crate = _repository_root() / "rust" / "kagg_env"
    profile = "release" if release else "debug"
    artifact = crate / "target" / profile / "lib_kagg_env.so"
    sources = _crate_sources(crate)
    if build and sources and _artifact_is_stale(artifact, sources):
        command = ["cargo", "build"]
        if release:
            command.append("--release")
        subprocess.run(command, cwd=crate, check=True)
    module = _load_path(artifact)
    if module is None:
        module = importlib.import_module("_kagg_env")
    batch_env = getattr(module, "BatchEnv", None)
    if batch_env is None:
        raise ImportError("_kagg_env does not expose BatchEnv")
    return module
