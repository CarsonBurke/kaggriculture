from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from kaggriculture import rust_env


@pytest.fixture(autouse=True)
def clear_native_cache() -> Iterator[None]:
    rust_env._MODULE_CACHE.clear()
    yield
    rust_env._MODULE_CACHE.clear()


def test_the_toolchain_is_reported_when_present_and_absent_without_failing(monkeypatch) -> None:
    # Recorded into a run's configuration at startup, so a machine running an
    # installed extension with no Rust toolchain must degrade to "unknown"
    # rather than take the run down before it begins.
    monkeypatch.setattr(
        rust_env.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "rustc 1.94.0-nightly\n", ""),
    )
    assert rust_env.toolchain_identity() == "rustc 1.94.0-nightly"

    monkeypatch.setattr(
        rust_env.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("no rustc")),
    )
    assert rust_env.toolchain_identity() is None

    # A toolchain that runs but fails, or answers with nothing, is not an
    # identity either -- recording an empty string would read as a measurement.
    for completed in (
        subprocess.CompletedProcess([], 1, "rustc 1.94.0-nightly\n", ""),
        subprocess.CompletedProcess([], 0, "   \n", ""),
    ):
        monkeypatch.setattr(rust_env.subprocess, "run", lambda *a, _result=completed, **k: _result)
        assert rust_env.toolchain_identity() is None


def fake_module() -> ModuleType:
    module = ModuleType("_kagg_env")
    module.BatchEnv = object  # type: ignore[attr-defined]
    from kaggriculture.tokens import OBSERVATION_SCHEMA_VERSION

    module.OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION
    return module


PINNED_TOOLCHAIN = "nightly-2025-12-13"


def local_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    crate = tmp_path / "rust" / "kagg_env"
    crate.mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]\nname='kagg_env'\nversion='0.1.0'\n")
    (tmp_path / "rust-toolchain.toml").write_text(f'[toolchain]\nchannel = "{PINNED_TOOLCHAIN}"\n')
    monkeypatch.setattr(rust_env, "_repository_root", lambda: tmp_path)
    return crate


def with_pinned_rustup(cargo: Any) -> Any:
    """Answer the toolchain check, and leave every other call to the caller.

    `_build_native` verifies the pinned toolchain before it compiles anything,
    so a bare `subprocess.run` stub would have the check swallow the cargo
    answer meant for the build.
    """

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:1] == ["rustup"]:
            return subprocess.CompletedProcess(
                command,
                0,
                f"{PINNED_TOOLCHAIN}-x86_64-unknown-linux-gnu (directory override)\n",
                "",
            )
        return cargo(command, **kwargs)

    return run


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

    monkeypatch.setattr(rust_env.subprocess, "run", with_pinned_rustup(run))
    assert rust_env._build_native(crate, release=True) == artifact
    assert seen_command[:2] == ["cargo", "build"]
    assert "--manifest-path" in seen_command
    assert "--lib" in seen_command
    assert "--release" in seen_command
    assert "--message-format=json-render-diagnostics" in seen_command
    target_index = seen_command.index("--target-dir")
    assert seen_command[target_index + 1] == str(crate / "target")


def test_frozen_snapshot_builds_in_writable_user_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = local_checkout(tmp_path, monkeypatch)
    (tmp_path / ".source-identity.json").write_text(
        json.dumps(
            {
                "files": {
                    "rust-toolchain.toml": "toolchain",
                    "rust/kagg_env/Cargo.toml": "manifest",
                    "rust/kagg_env/src/lib.rs": "source",
                }
            }
        )
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    target = rust_env._cargo_target_dir(crate)
    artifact = target / "release" / "lib_kagg_env.so"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    event = {
        "reason": "compiler-artifact",
        "manifest_path": str(crate / "Cargo.toml"),
        "target": {"name": "_kagg_env", "crate_types": ["cdylib"]},
        "filenames": [str(artifact)],
    }
    seen_command: list[str] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen_command.extend(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(event), "")

    monkeypatch.setattr(rust_env.subprocess, "run", with_pinned_rustup(run))

    assert rust_env._build_native(crate, release=True) == artifact
    target_index = seen_command.index("--target-dir")
    assert seen_command[target_index + 1] == str(target)
    assert not target.is_relative_to(tmp_path / "rust")


def test_frozen_cargo_cache_is_shared_only_by_identical_rust_inputs(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        root.mkdir()
    shared = {
        "rust-toolchain.toml": "toolchain",
        "rust/kagg_env/Cargo.toml": "manifest",
        "rust/kagg_env/src/lib.rs": "source",
    }
    (roots[0] / ".source-identity.json").write_text(
        json.dumps({"files": {**shared, "scripts/train_ppo.py": "old"}})
    )
    (roots[1] / ".source-identity.json").write_text(
        json.dumps({"files": {**shared, "scripts/train_ppo.py": "new"}})
    )

    assert rust_env._cargo_cache_key(roots[0]) == rust_env._cargo_cache_key(roots[1])

    (roots[1] / ".source-identity.json").write_text(
        json.dumps({"files": {**shared, "rust/kagg_env/src/lib.rs": "changed"}})
    )
    assert rust_env._cargo_cache_key(roots[0]) != rust_env._cargo_cache_key(roots[1])


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

    monkeypatch.setattr(rust_env.subprocess, "run", with_pinned_rustup(run))
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

    monkeypatch.setattr(rust_env.subprocess, "run", with_pinned_rustup(run))
    with pytest.raises(RuntimeError, match="Cargo was not found"):
        rust_env._build_native(crate, release=True)


def test_a_toolchain_that_is_not_the_pin_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rust-toolchain.toml is hashed into source_identity, so a checkpoint's
    # provenance asserts which compiler built its simulator. rustup lets
    # RUSTUP_TOOLCHAIN, `cargo +toolchain` and directory overrides outrank the
    # file, and none of those appear in the identity -- so an unverified pin
    # states something the run cannot back up. Building anyway is the one
    # outcome that makes the recorded provenance false rather than absent.
    crate = local_checkout(tmp_path, monkeypatch)

    def cargo(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the build ran under an unverified toolchain")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:1] == ["rustup"]:
            return subprocess.CompletedProcess(
                command, 0, "stable-x86_64-unknown-linux-gnu (overridden by environment)\n", ""
            )
        return cargo(command, **kwargs)

    monkeypatch.setattr(rust_env.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=r"active Rust toolchain is 'stable"):
        rust_env._build_native(crate, release=True)

    # And a toolchain that cannot be established at all is not a pass either.
    # Anything short of a positive match has to stop the build, because the
    # identity records the pin either way.
    def absent(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("rustup")

    monkeypatch.setattr(rust_env.subprocess, "run", absent)
    with pytest.raises(RuntimeError, match="rustup could not be run"):
        rust_env._build_native(crate, release=True)

    for stdout, returncode, expected in (
        ("", 1, "rustup reported no active toolchain"),
        ("   \n", 0, "active Rust toolchain is ''"),
    ):
        monkeypatch.setattr(
            rust_env.subprocess,
            "run",
            lambda *a, _out=stdout, _rc=returncode, **k: subprocess.CompletedProcess(
                [], _rc, _out, ""
            ),
        )
        with pytest.raises(RuntimeError, match=expected):
            rust_env._build_native(crate, release=True)


def test_the_pin_that_is_enforced_is_the_one_in_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rust-toolchain.toml is what source_identity hashes, so the build must be
    # checked against that file's contents and not against a constant that
    # merely agrees with it today. Written with a channel nothing else in the
    # suite uses, so a check that ignored the file could not accidentally pass.
    crate = local_checkout(tmp_path, monkeypatch)
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.89.0"\n')
    assert rust_env._pinned_toolchain() == "1.89.0"

    monkeypatch.setattr(
        rust_env.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, f"{PINNED_TOOLCHAIN}-x86_64-unknown-linux-gnu\n", ""
        ),
    )
    with pytest.raises(RuntimeError, match=r"pins '1\.89\.0'"):
        rust_env._build_native(crate, release=True)

    # A prefix is not a match. rustup answers with the full triple, so the pin
    # is legitimately a prefix of it -- but only across the separator. Pin
    # `1.8` against an active `1.89.0` is the case a bare startswith accepts
    # and this must not: two different compilers, one of them recorded.
    monkeypatch.setattr(
        rust_env.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "1.89.0-x86_64-unknown-linux-gnu\n", ""),
    )
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.8"\n')
    with pytest.raises(RuntimeError, match=r"pins '1\.8'"):
        rust_env._verify_pinned_toolchain(crate)

    # And the genuine prefix, the one rustup actually produces, still passes.
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.89.0"\n')
    rust_env._verify_pinned_toolchain(crate)


def test_a_tree_without_the_pin_says_so_instead_of_raising_an_errno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The likeliest cause is a frozen snapshot or worktree taken before the pin
    # joined the source identity. Those cannot be repaired in place, and a bare
    # ENOENT on a path the reader has never heard of does not say that.
    crate = local_checkout(tmp_path, monkeypatch)
    (tmp_path / "rust-toolchain.toml").unlink()
    with pytest.raises(RuntimeError, match="must be re-frozen"):
        rust_env._build_native(crate, release=True)

    # `path = ` is a legal toolchain table with no channel in it at all.
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\npath = "/opt/rust"\n')
    with pytest.raises(RuntimeError, match="cannot read the pinned Rust toolchain"):
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
    module.BatchEnv = object
    with pytest.raises(ImportError, match="stale observation schema"):
        rust_env._validate_module(module, "test module")
