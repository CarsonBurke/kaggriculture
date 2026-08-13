from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from kaggriculture import rust_env


@pytest.fixture(autouse=True)
def clear_native_cache() -> Iterator[None]:
    rust_env._MODULE_CACHE.clear()
    yield
    rust_env._MODULE_CACHE.clear()


def fake_module() -> ModuleType:
    module = ModuleType("_kagg_env")
    module.BatchEnv = object  # type: ignore[attr-defined]
    return module


def local_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    crate = tmp_path / "rust" / "kagg_env"
    crate.mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]\nname='kagg_env'\nversion='0.1.0'\n")
    monkeypatch.setattr(rust_env, "_repository_root", lambda: tmp_path)
    return crate


def test_repeated_and_concurrent_loads_build_and_initialize_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = local_checkout(tmp_path, monkeypatch)
    artifact = crate / "target" / "release" / "lib_kagg_env.so"
    module = fake_module()
    build_calls = 0
    load_calls = 0

    def build(_crate: Path, release: bool) -> Path:
        nonlocal build_calls
        assert _crate == crate
        assert release
        build_calls += 1
        time.sleep(0.01)
        return artifact

    def load(path: Path) -> ModuleType:
        nonlocal load_calls
        assert path == artifact
        load_calls += 1
        return module

    monkeypatch.setattr(rust_env, "_build_native", build)
    monkeypatch.setattr(rust_env, "_load_path", load)
    with ThreadPoolExecutor(max_workers=16) as executor:
        loaded = list(executor.map(lambda _: rust_env.load_native(), range(64)))

    assert all(candidate is module for candidate in loaded)
    assert rust_env.load_native(build=False) is module
    assert build_calls == 1
    assert load_calls == 1


def test_build_true_delegates_staleness_and_artifact_location_to_cargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = local_checkout(tmp_path, monkeypatch)
    artifact = tmp_path / "configured-target" / "release" / "lib_kagg_env.so"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    event = {
        "reason": "compiler-artifact",
        "manifest_path": str(crate / "Cargo.toml"),
        "target": {"name": "_kagg_env", "crate_types": ["cdylib", "rlib"]},
        "filenames": [str(artifact), str(artifact.with_suffix(".rlib"))],
    }
    seen_command: list[str] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen_command.extend(command)
        assert kwargs["cwd"] == crate
        return subprocess.CompletedProcess(command, 0, json.dumps(event), "")

    monkeypatch.setattr(rust_env.subprocess, "run", run)
    assert rust_env._build_native(crate, release=True) == artifact
    assert seen_command[:2] == ["cargo", "build"]
    assert "--manifest-path" in seen_command
    assert "--lib" in seen_command
    assert "--release" in seen_command
    assert "--message-format=json-render-diagnostics" in seen_command


def test_cargo_failure_preserves_compiler_and_process_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = local_checkout(tmp_path, monkeypatch)
    event = {
        "reason": "compiler-message",
        "message": {"rendered": "error[E0001]: precise compiler failure\n"},
    }

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 101, json.dumps(event), "cargo failed summary")

    monkeypatch.setattr(rust_env.subprocess, "run", run)
    with pytest.raises(RuntimeError) as caught:
        rust_env._build_native(crate, release=False)
    message = str(caught.value)
    assert "exit status 101" in message
    assert "precise compiler failure" in message
    assert "cargo failed summary" in message
    assert str(crate / "Cargo.toml") in message


def test_missing_cargo_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = local_checkout(tmp_path, monkeypatch)

    def run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("cargo")

    monkeypatch.setattr(rust_env.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Cargo was not found"):
        rust_env._build_native(crate, release=True)


def test_build_false_honors_custom_target_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_checkout(tmp_path, monkeypatch)
    target = tmp_path / "custom-target"
    artifact = target / "debug" / "lib_kagg_env.so"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    module = fake_module()
    monkeypatch.setenv("CARGO_TARGET_DIR", str(target))
    monkeypatch.setattr(rust_env, "_load_path", lambda path: module if path == artifact else None)
    monkeypatch.setattr(
        rust_env,
        "_build_native",
        lambda *_: pytest.fail("build=False invoked Cargo"),
    )

    assert rust_env.load_native(build=False, release=False) is module


def test_installed_extension_is_fresh_clone_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rust_env, "_repository_root", lambda: tmp_path)
    module = fake_module()
    monkeypatch.setattr(rust_env.importlib, "import_module", lambda name: module)
    assert rust_env.load_native() is module
    assert rust_env.load_native() is module


def test_missing_installed_extension_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rust_env, "_repository_root", lambda: tmp_path)

    def missing(name: str) -> ModuleType:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(rust_env.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="editable checkout with Cargo installed") as caught:
        rust_env.load_native()
    assert isinstance(caught.value.__cause__, ImportError)


def test_corrupt_local_artifact_reports_path_and_original_failure(tmp_path: Path) -> None:
    artifact = tmp_path / "lib_kagg_env.so"
    artifact.write_bytes(b"not a shared library")
    with pytest.raises(ImportError, match=str(artifact)) as caught:
        rust_env._load_path(artifact)
    assert caught.value.__cause__ is not None


def test_module_contract_is_validated() -> None:
    module = ModuleType("_kagg_env")
    with pytest.raises(ImportError, match="does not expose BatchEnv"):
        rust_env._validate_module(module, "test module")
