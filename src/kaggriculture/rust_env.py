"""Build and load the in-process batched Rust Kaggriculture simulator."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import threading
import tomllib
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


def _cargo_cache_key(root: Path) -> str:
    """Key frozen builds by Rust inputs, not by unrelated Python edits."""
    identity_path = root / ".source-identity.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        files = identity["files"]
        rust_inputs = {
            name: digest
            for name, digest in files.items()
            if name == "rust-toolchain.toml" or name.startswith("rust/")
        }
        if rust_inputs:
            encoded = json.dumps(rust_inputs, sort_keys=True, separators=(",", ":")).encode()
            return hashlib.sha256(encoded).hexdigest()[:16]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def _cargo_target_dir(crate: Path) -> Path:
    """Resolve a writable Cargo cache without mutating frozen source trees."""
    configured = os.environ.get("CARGO_TARGET_DIR")
    if configured:
        target = Path(configured).expanduser()
        return target if target.is_absolute() else crate / target

    root = _repository_root()
    if not (root / ".source-identity.json").is_file() and os.access(crate, os.W_OK):
        return crate / "target"

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    return cache_root / "kaggriculture" / "cargo-target" / _cargo_cache_key(root)


def _fallback_artifacts(crate: Path, release: bool) -> list[Path]:
    """Return likely artifacts without invoking Cargo, for ``build=False``."""
    profile = _profile_name(release)
    target = _cargo_target_dir(crate)
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


def _pinned_toolchain() -> str:
    pin = _repository_root() / "rust-toolchain.toml"
    try:
        with pin.open("rb") as stream:
            return str(tomllib.load(stream)["toolchain"]["channel"])
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        # A bare errno here is actively misleading, because the likeliest cause
        # is a tree that predates the pin -- a frozen snapshot or a worktree
        # taken before `rust-toolchain.toml` joined the source identity. Those
        # cannot be repaired in place and have to be re-frozen or rebased, which
        # is not something an ENOENT conveys. `path = ` is also a legal
        # toolchain table with no channel at all, so a KeyError is reachable
        # without the file being damaged.
        raise RuntimeError(
            f"cannot read the pinned Rust toolchain from {pin}: {error!r}; the file is part "
            f"of the source identity, so a tree without a usable one predates the pin and "
            f"must be re-frozen rather than built in place"
        ) from error


def _verify_pinned_toolchain(crate: Path) -> None:
    """Fail unless the compiler about to run is the one the identity records.

    `rust-toolchain.toml` is hashed into `source_identity`, so a checkpoint's
    provenance asserts which rustc built its simulator. That assertion is only
    as good as the file being obeyed, and rustup lets `RUSTUP_TOOLCHAIN`, a
    `cargo +toolchain` invocation, or a directory override outrank it -- none
    of which appear anywhere in the identity. Verifying here, against the same
    working directory the build uses, is what turns the pin from a request into
    a fact.
    """
    expected = _pinned_toolchain()
    try:
        completed = subprocess.run(
            ["rustup", "show", "active-toolchain"],
            cwd=crate,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(
            f"cannot verify the pinned Rust toolchain {expected!r}: rustup could not be run "
            f"({error!r}), so nothing enforces rust-toolchain.toml and the recorded source "
            f"identity would misstate which compiler built {_MODULE_NAME}"
        ) from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot verify the pinned Rust toolchain {expected!r}: "
            f"{completed.stderr.strip() or 'rustup reported no active toolchain'}"
        )
    active = completed.stdout.split(maxsplit=1)[0] if completed.stdout.split() else ""
    # Anchored on the separator rather than a bare prefix. rustup answers with
    # the full triple, `nightly-2025-12-13-x86_64-...`, so the pin is a prefix
    # by design -- but a bare startswith would also accept `1.75.0` for a pin
    # of `1.7`, which is the one comparison this must never get wrong.
    if not (active == expected or active.startswith(f"{expected}-")):
        raise RuntimeError(
            f"the active Rust toolchain is {active!r} but rust-toolchain.toml pins "
            f"{expected!r}; the source identity records the pin, so building here would "
            f"stamp checkpoints with a compiler that did not produce them "
            f"(check RUSTUP_TOOLCHAIN and any `rustup override` for this directory)"
        )


def _build_native(crate: Path, release: bool) -> Path:
    """Build the cdylib and return the exact artifact path reported by Cargo."""
    _verify_pinned_toolchain(crate)
    manifest = crate / "Cargo.toml"
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(manifest),
        "--lib",
        "--message-format=json-render-diagnostics",
        "--target-dir",
        str(_cargo_target_dir(crate)),
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


def toolchain_identity() -> str | None:
    """Return the Rust compiler that builds the crate, or None if it is absent.

    Recorded beside a run's configuration because nothing else captures it.
    `uv.lock` is hashed into source_identity and pins torch down to wheel
    digests, so the Python side of the simulator's numerics is bound by the
    identity itself; Cargo.lock pins dependency crates rather than the
    compiler.

    The binding for rustc is now `rust-toolchain.toml`, which is hashed into
    the identity and enforced by `_verify_pinned_toolchain` before any build.
    This stays because the two answer different questions: the pin says which
    toolchain was demanded, and this says which compiler actually replied. They
    can differ -- a pin names a channel, not a build, so `nightly-2025-12-13`
    resolved to `1.94.0-nightly (fa5eda19b)` here and a rustup that re-resolves
    it would not announce itself. Note also that this reports the compiler, not
    RUSTFLAGS or the target CPU.
    """
    try:
        completed = subprocess.run(
            ["rustc", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    version = completed.stdout.strip()
    return version if completed.returncode == 0 and version else None


def _validate_module(module: ModuleType, origin: str) -> ModuleType:
    if getattr(module, "BatchEnv", None) is None:
        raise ImportError(f"native extension loaded from {origin} does not expose BatchEnv")
    from kaggriculture.tokens import OBSERVATION_SCHEMA_VERSION

    if getattr(module, "OBSERVATION_SCHEMA_VERSION", None) != OBSERVATION_SCHEMA_VERSION:
        raise ImportError(
            f"native extension loaded from {origin} has stale observation schema; rebuild required"
        )
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
