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
# Bumped to 2 when the compile decision split into a rollout knob and an
# update knob. A version-1 record names a single `compile_models` whose
# per-phase meaning is unrecoverable -- the run it describes could not have
# measured the phases separately -- so it is rejected rather than migrated.
#
# Bumped to 3 when the calibration stopped differencing two runs that change
# both knobs at once. A version-2 record carries per-phase ratios taken across
# a pair whose knobs both moved, so a phase's ratio is contaminated by whatever
# between-run drift the pair happened to have -- and the conv calibration
# measured exactly that, a 4.9% drift on a rollout phase neither run compiled,
# against an 8.0% "speedup" credited to the rollout knob. The evidence a
# version-2 record retains cannot be re-attributed after the fact, because the
# configuration that isolates each knob was never run, so it is rejected rather
# than migrated.
RUN_PROVENANCE_FORMAT_VERSION = 3
#: The speedup floor a knob must clear to be enabled, measured on the whole
#: iteration rather than on the knob's own phase: a knob that halves a phase
#: worth 2% of an iteration has not earned the compile. It lives here rather
#: than in the launcher because provenance.py is what re-derives each decision
#: from the speedup recorded beside it, and a validator that reads the
#: threshold out of the record it is checking has verified nothing. The
#: launcher imports this; provenance.py cannot import the launcher, since this
#: module ships inside the submission bundle and scripts/ does not.
MINIMUM_COMPILE_SPEEDUP = 1.05


def is_legacy_run_provenance(value: object) -> bool:
    """Whether a record is a well-formed provenance from an older format.

    Distinguishes "predates the current schema" from "corrupt", so a read path
    can drop the first without also silently accepting the second.
    """
    return (
        isinstance(value, Mapping)
        and type(value.get("format_version")) is int
        and 0 < value["format_version"] < RUN_PROVENANCE_FORMAT_VERSION
    )


# `rust-toolchain.toml` earns its place here the same way `uv.lock` does: the
# native extension is build output and deliberately outside the identity, so
# the pinned compiler is the only record of what produced it. Without it a
# toolchain update changes every rollout's binary while the identity, and so
# every checkpoint's provenance, stays byte-identical.
_ROOT_FILES = ("pyproject.toml", "uv.lock", "rust-toolchain.toml")
_SOURCE_ROOTS = ("src/kaggriculture", "scripts", "rust/kagg_env")
# Build output is derived from the source rather than an input to it, and it
# cannot be hashed as though it were. Cargo's target directory embeds the
# absolute path it was built at -- 82 files under one such tree contain this
# machine's home directory -- along with the exact compiler in
# .rustc_info.json. An identity covering that is machine-, path- and
# toolchain-cache-dependent, so a clean checkout of the same commit on another
# machine can never reproduce it, which is the opposite of what a
# content-addressed identity is for.
#
# It is also unsound in a way that bites a single machine, because the identity
# is computed once at startup while rollout.load_native() shells out to `cargo
# build` at the first rollout. A run whose build was stale at launch stamps its
# checkpoints with the pre-build identity, mutates the tree, and then fails
# require_source_identity against itself on resume -- a wedge the run inflicts
# on itself with no external trigger.
#
# A cache directory inside a source root is an error, never a prune. Cargo
# writes CACHEDIR.TAG into every target directory, as do the pytest, ruff and
# mypy caches, so the tag identifies that whole class -- but *acting* on it by
# pruning is a silent drop, and by the argument below a silent drop is the
# failure that cannot be recovered from. Skipping a tagged directory would
# delete whatever it holds from the identity with nothing to say so, and there
# is no contents test that could make that safe: a cargo target tree genuinely
# contains Rust, since build scripts emit `build/*/out/*.rs` and this tree
# carries three from serde and target-lexicon.
#
# Scoping the prune to the crate was the first attempt and it was not enough.
# Cargo compiles `benches/` and `examples/` too, so protecting only `src/` and
# `tests/` left real crate source droppable by one stray file. The rule that
# actually holds is the simple one: build output does not belong inside a
# source root at all. Cargo's conventional location is excluded by path before
# the tag is ever consulted, so nothing legitimate reaches this check, and
# anything that does is a misconfigured CARGO_TARGET_DIR worth stopping for.
#
# Excluded by path rather than by bare name: as a name it also swallowed
# `scripts/target/`, which has nothing to do with cargo. `__pycache__` stays a
# name because it is one wherever it appears, and holds only the bytecode of
# files already hashed.
_EXCLUDED_DIRECTORY_PATHS = ("rust/kagg_env/target",)
_EXCLUDED_DIRECTORY_NAMES = frozenset(("__pycache__",))
_CACHE_DIRECTORY_TAG = "CACHEDIR.TAG"
# Files that survive the directory pruning above are classified, not defaulted.
# Neither a denylist nor an allowlist is safe on its own, because they fail in
# opposite directions and only one of the two failures is visible. A denylist
# silently *hashes* whatever it has not been taught to exclude, which makes the
# identity machine-dependent -- bad, but it announces itself the first time a
# clean checkout fails require_source_identity. A bare allowlist silently
# *drops* whatever it has not been taught to include, and that is strictly
# worse: two trees that differ in a real input then share one identity, so the
# provenance claim is false rather than merely unreproducible, and nothing
# anywhere fails to say so.
#
# So the union is closed and an unclassified file is an error. The cost is that
# adding a source file of a new kind stops the next launch until someone says
# which side it belongs on, and that is the right prompt rather than a chore:
# it is exactly the question the identity exists to answer. `.gitignore` is
# named on the ignored side because it steers git and nothing else -- it is not
# read by the build, the simulator, or training.
_HASHED_SUFFIXES = frozenset((".py", ".rs", ".toml", ".lock"))
_IGNORED_SOURCE_FILES = frozenset(("rust/kagg_env/.gitignore",))


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


def _raise_walk_error(error: OSError) -> None:
    raise error


def _source_paths(root: Path) -> list[Path]:
    root = root.resolve()
    paths = {root / name for name in _ROOT_FILES}
    unclassified: list[Path] = []
    tagged_inside_source: list[Path] = []
    linked_directories: list[Path] = []
    for relative_root in _SOURCE_ROOTS:
        source_root = root / relative_root
        if not source_root.is_dir():
            raise FileNotFoundError(f"source provenance root is missing: {source_root}")
        # Pruned in place while walking, so an excluded directory is never
        # descended into rather than being filtered out file by file after the
        # fact. Matching on the directory instead of on each path's parts also
        # keeps a *file* named `target` from being dropped for sharing a name
        # with a directory nobody wants.
        #
        # `onerror` is not optional here. os.walk's documented default is to
        # ignore whatever scandir raises, so an unreadable directory -- wrong
        # permissions, a dropped mount, a concurrent delete -- contributes no
        # files and no error, and the identity silently loses that subtree.
        for directory, subdirectories, names in os.walk(source_root, onerror=_raise_walk_error):
            current = Path(directory)
            retained: list[str] = []
            for name in subdirectories:
                child = current / name
                relative = child.relative_to(root).as_posix()
                if name in _EXCLUDED_DIRECTORY_NAMES or relative in _EXCLUDED_DIRECTORY_PATHS:
                    continue
                # os.walk does not follow directory symlinks, so a linked-in
                # source directory is retained here and then never descended
                # into -- everything under it leaves the identity without a
                # word. Following it instead would invite cycles and would hash
                # content from outside the tree, so the honest answer is to
                # refuse the arrangement rather than to silently half-support it.
                if child.is_symlink():
                    linked_directories.append(child)
                    continue
                if (child / _CACHE_DIRECTORY_TAG).is_file():
                    tagged_inside_source.append(child)
                    continue
                retained.append(name)
            subdirectories[:] = retained
            for name in names:
                path = current / name
                if path.relative_to(root).as_posix() in _IGNORED_SOURCE_FILES:
                    continue
                if Path(name).suffix not in _HASHED_SUFFIXES:
                    unclassified.append(path)
                    continue
                # A dangling symlink is not an input and never can be: there is
                # nothing behind it to hash. Editors manufacture these -- Emacs
                # names its lock `.#module.py`, which carries a `.py` suffix --
                # and aborting a launch because a source file is open in a
                # buffer would be absurd. A symlink that does resolve is a
                # different matter and still refused, loudly, by file_sha256.
                if path.is_symlink() and not path.exists():
                    continue
                paths.add(path)
    if tagged_inside_source:
        rendered = sorted(path.relative_to(root).as_posix() for path in tagged_inside_source)
        raise ValueError(
            "source provenance found a cache directory inside the source tree; move it out "
            "rather than letting it prune source out of the identity silently, and check "
            f"CARGO_TARGET_DIR if cargo put it there: {rendered}"
        )
    if linked_directories:
        rendered = sorted(path.relative_to(root).as_posix() for path in linked_directories)
        raise ValueError(
            "source provenance cannot hash a symlinked directory; os.walk does not follow "
            "one, so its contents would leave the identity without a word: "
            f"{rendered}"
        )
    if unclassified:
        rendered = sorted(path.relative_to(root).as_posix() for path in unclassified)
        raise ValueError(
            "source provenance cannot classify these files; hash them by adding their "
            "suffix to _HASHED_SUFFIXES, or exclude them by naming them in "
            f"_IGNORED_SOURCE_FILES: {rendered}"
        )
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


def _normalized_run_provenance(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize everything a run provenance digest covers.

    Shared by the validator and the producer so both digest the same bytes.
    Keeping them separate meant the producer hashed the decision's raw values
    while the validator hashed the normalized ones, and `_positive_number`
    coerces to float -- so an integral `minimum_compile_speedup` of 2 rendered
    as `2` on one side and `2.0` on the other and the record accused itself of
    tampering. Editing the shipped threshold to a whole number would have been
    enough to break every calibrated launch that way.
    """
    identity = validate_source_identity(value["source_identity"])
    calibration = value["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "eager_report",
        "mixed_report",
        "compiled_report",
        "compile_rollout",
        "compile_update",
        "minimum_compile_speedup",
        "attributed_knob_speedups",
    }:
        raise ValueError("run provenance calibration has an invalid schema")
    reports = {}
    # Three reports, not two. Each knob is attributed against the neighbouring
    # report that isolates it -- the pair whose configurations differ in that
    # knob and nothing else -- so the chain from all-eager to all-compiled has
    # to be carried whole. Dropping the middle report would leave a record
    # whose speedups cannot be recomputed from the evidence it names.
    for name in ("eager_report", "mixed_report", "compiled_report"):
        report = calibration[name]
        if not isinstance(report, dict) or set(report) != {"sha256", "size_bytes"}:
            raise ValueError(f"run provenance {name} has an invalid schema")
        if type(report["size_bytes"]) is not int or report["size_bytes"] <= 0:
            raise ValueError(f"run provenance {name} size must be a positive integer")
        reports[name] = {
            "sha256": _hex_digest(report["sha256"], f"run provenance {name} digest"),
            "size_bytes": report["size_bytes"],
        }
    minimum_speedup = _positive_number(
        calibration["minimum_compile_speedup"],
        "minimum compile speedup",
    )
    # The record supplies both sides of the derivation below -- the measured
    # speedup and the threshold it is compared against -- so internal
    # consistency alone certifies nothing. Left free, a threshold of 1e-9 flips
    # both knobs on with every check green and no fabricated timing anywhere,
    # which would make the derivation ceremony rather than verification.
    #
    # The floor is the shipped constant, so a record can only ever be stricter
    # than current policy, never laxer. Allowing stricter is the deliberate
    # half: a decision made under a higher bar stays valid when the bar is
    # lowered, whereas re-deriving against equality would retroactively
    # invalidate decisions that were correct when they were made.
    if minimum_speedup < MINIMUM_COMPILE_SPEEDUP:
        raise ValueError(
            "run provenance minimum compile speedup must be at least "
            f"{MINIMUM_COMPILE_SPEEDUP}; {minimum_speedup} would certify a phase "
            "the shipped policy rejects"
        )
    # Each knob must be derivable from the speedup recorded beside it. That is
    # the whole point of carrying the measurement into the checkpoint: a
    # decision nobody can recompute from the evidence is an assertion, not
    # provenance. Two knobs mean two derivations, and a knob named in one place
    # but not the other is a schema error rather than a default.
    #
    # The speedup recorded is the knob's attributed effect on the whole
    # iteration: the isolating pair's iteration budget with that knob's phase,
    # and only that phase, moved to its measured value under the knob. A raw
    # per-phase ratio would flatter a knob whose phase is a small share of the
    # iteration, and a raw total ratio across the pair would credit the knob
    # with the pair's drift on phases it does not touch.
    measured = calibration["attributed_knob_speedups"]
    knobs = ("compile_rollout", "compile_update")
    if not isinstance(measured, dict) or set(measured) != set(knobs):
        raise ValueError("run provenance attributed knob speedups have an invalid schema")
    speedups = {}
    for knob in knobs:
        if type(calibration[knob]) is not bool:
            raise ValueError(f"run provenance {knob} decision must be boolean")
        speedups[knob] = _positive_number(measured[knob], f"attributed {knob} speedup")
        if calibration[knob] != (speedups[knob] >= minimum_speedup):
            raise ValueError(f"run provenance {knob} decision contradicts measured speedup")
    return {
        "format_version": RUN_PROVENANCE_FORMAT_VERSION,
        "source_identity": identity,
        "calibration": {
            **reports,
            **{knob: calibration[knob] for knob in knobs},
            "minimum_compile_speedup": minimum_speedup,
            "attributed_knob_speedups": speedups,
        },
    }


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
    normalized = _normalized_run_provenance(value)
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
            **{
                f"{name}_report": {
                    "sha256": decision.get(f"{name}_report_sha256"),
                    "size_bytes": decision.get(f"{name}_report_size_bytes"),
                }
                for name in ("eager", "mixed", "compiled")
            },
            "compile_rollout": decision.get("compile_rollout"),
            "compile_update": decision.get("compile_update"),
            "minimum_compile_speedup": decision.get("minimum_compile_speedup"),
            "attributed_knob_speedups": decision.get("attributed_knob_speedups"),
        },
    }
    normalized = _normalized_run_provenance(payload)
    return normalized | {"sha256": _run_digest(normalized)}


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
