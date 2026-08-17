from __future__ import annotations

import json
import shutil
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
        "rust-toolchain.toml": '[toolchain]\nchannel = "nightly-2025-12-13"\n',
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


def _build_output(root: Path, target: str) -> None:
    """Write a cargo-shaped target directory, cache tag included."""
    for relative, contents in {
        f"{target}/CACHEDIR.TAG": "Signature: 8a477f597d28d172789f06886806bc55\n",
        f"{target}/release/lib_kagg_env.so": "built\n",
        f"{target}/release/.fingerprint/kagg_env/invoked.timestamp": "stamped\n",
        f"{target}/release/.fingerprint/kagg_env/dep-lib-kagg_env.d": f"{root}/absolute\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def test_the_identity_covers_source_and_nothing_a_build_produces(tmp_path: Path) -> None:
    # Two failures, one rule. Cargo's target directory embeds the absolute path
    # it was built at and the exact compiler, so an identity covering it is
    # machine- and toolchain-cache-dependent and no clean checkout of the same
    # commit reproduces it. And because the identity is computed once at
    # startup while rollout.load_native() shells out to `cargo build` at the
    # first rollout, a run whose build was stale at launch would stamp its
    # checkpoints with the pre-build identity, mutate the tree, and then fail
    # require_source_identity against itself -- wedged by its own first
    # iteration with nothing external having touched the checkout.
    _minimal_source(tmp_path)
    original = source_identity(tmp_path)

    # Stated as the whole set rather than as a list of names to exclude. A
    # denylist only ever covers the instances already known to have caused
    # trouble, which is how a stray `artifacts/cargo-target` got hashed while
    # `target` was excluded; asserting the set fails on any future addition
    # whatever it is called.
    assert set(original["files"]) == {
        "pyproject.toml",
        "uv.lock",
        # The compiler pin is source even though what it pins is not: the native
        # extension is build output and excluded below, so this file is the only
        # thing in the identity that distinguishes two trees whose Rust differs
        # solely in which rustc compiled it.
        "rust-toolchain.toml",
        "src/kaggriculture/module.py",
        "scripts/train.py",
        "rust/kagg_env/Cargo.toml",
        "rust/kagg_env/Cargo.lock",
        "rust/kagg_env/pyproject.toml",
        "rust/kagg_env/src/lib.rs",
    }

    # Cargo's conventional directory is excluded by path, so an ordinary
    # `cargo build` leaves the identity alone.
    _build_output(tmp_path, "rust/kagg_env/target")
    (tmp_path / "src/kaggriculture/__pycache__").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/kaggriculture/__pycache__/module.cpython-313.pyc").write_text(
        "bytecode\n", encoding="utf-8"
    )

    assert source_identity(tmp_path) == original
    require_source_identity(original, tmp_path)

    # A rebuild rewrites that output; the source it was built from did not move.
    (tmp_path / "rust/kagg_env/target/release/lib_kagg_env.so").write_text(
        "rebuilt\n", encoding="utf-8"
    )
    assert source_identity(tmp_path) == original
    require_source_identity(original, tmp_path)

    # Build output anywhere *else* under a source root is a misconfiguration
    # rather than something to absorb quietly -- the stray this repository
    # actually grew, and a name nothing has seen before. Pruning them on their
    # cache tag is what let a tag delete real source, so both now stop the
    # identity and say where to look.
    for target in (
        "rust/kagg_env/artifacts/cargo-target",
        "rust/kagg_env/build/whatever-someone-sets-next",
    ):
        _build_output(tmp_path, target)
        with pytest.raises(ValueError, match=r"cache directory inside the source tree"):
            source_identity(tmp_path)
        shutil.rmtree(tmp_path / target)

    assert source_identity(tmp_path) == original

    # The crate's own sources are still inputs, so a real edit beside the
    # excluded output must still be caught.
    (tmp_path / "rust/kagg_env/src/lib.rs").write_text(
        "pub fn value() -> u8 { 2 }\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"rust/kagg_env/src/lib\.rs"):
        require_source_identity(original, tmp_path)


def test_an_unclassified_file_stops_the_identity_instead_of_being_guessed_at(
    tmp_path: Path,
) -> None:
    # The asymmetry this protects against. Hashing a file that should not be
    # hashed announces itself -- the identity stops reproducing and the next
    # clean checkout fails loudly. Dropping a file that should be hashed does
    # not: two trees that differ in a real input agree on their identity, every
    # checkpoint stamped under either one claims the other's provenance, and no
    # check anywhere fires. Only the second failure is silent, so the classifier
    # refuses to guess rather than defaulting either way.
    _minimal_source(tmp_path)
    original = source_identity(tmp_path)

    # A plausible future input rather than a contrived one: type stubs are read
    # by nothing at runtime, a JSON table is read by everything, and the suffix
    # alone cannot tell you which -- which is the point of asking.
    (tmp_path / "src/kaggriculture/schedule.json").write_text('{"steps": 719}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"cannot classify.*schedule\.json"):
        source_identity(tmp_path)

    # Naming it on the ignored side is the other half of the answer, and it
    # restores exactly the identity that existed before the file did.
    (tmp_path / "src/kaggriculture/schedule.json").unlink()
    (tmp_path / "rust/kagg_env/.gitignore").write_text("/target\n", encoding="utf-8")
    assert source_identity(tmp_path) == original
    require_source_identity(original, tmp_path)


def _cache_tag(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "CACHEDIR.TAG").write_text(
        "Signature: 8a477f597d28d172789f06886806bc55\n", encoding="utf-8"
    )


def test_a_cache_tag_cannot_prune_source_out_of_the_identity(tmp_path: Path) -> None:
    # The tag rule is the one place a *directory* can disappear, and pruning is
    # a silent drop -- the failure the file classifier above refuses to make.
    # One dropped tag would otherwise delete a whole package from the identity
    # with nothing to say so, which is the same false-provenance case: two
    # trees whose Python genuinely differs, agreeing on one digest.
    _minimal_source(tmp_path)
    original = source_identity(tmp_path)

    package = tmp_path / "src/kaggriculture/subpkg"
    package.mkdir(parents=True)
    (package / "critical.py").write_text("VALUE = 1\n", encoding="utf-8")
    with_package = source_identity(tmp_path)
    assert "src/kaggriculture/subpkg/critical.py" in with_package["files"]

    _cache_tag(package)
    with pytest.raises(ValueError, match=r"cache directory inside the source tree"):
        source_identity(tmp_path)

    (package / "CACHEDIR.TAG").unlink()
    assert source_identity(tmp_path) == with_package

    # Nowhere under the crate is exempt either. Scoping the prune to the crate
    # and protecting `src/` and `tests/` was the first attempt at this rule and
    # it was not enough: cargo compiles `benches/` and `examples/` too, so a
    # stray tag in either still deleted real crate source. Rather than grow the
    # list of protected names until it happens to be complete, build output is
    # simply not allowed inside a source root -- cargo's conventional `target/`
    # never reaches this check because it is excluded by path first.
    for crate_directory in ("src", "tests", "benches", "examples", "srcextra"):
        tagged = tmp_path / "rust/kagg_env" / crate_directory
        tagged.mkdir(parents=True, exist_ok=True)
        (tagged / "code.rs").write_text("// crate source\n", encoding="utf-8")
        _cache_tag(tagged)
        with pytest.raises(ValueError, match=rf"rust/kagg_env/{crate_directory}"):
            source_identity(tmp_path)
        (tagged / "CACHEDIR.TAG").unlink()
        (tagged / "code.rs").unlink()

    assert source_identity(tmp_path) == with_package
    (package / "critical.py").unlink()
    package.rmdir()
    assert source_identity(tmp_path) == original


def test_a_symlinked_source_directory_is_refused_rather_than_walked_past(tmp_path: Path) -> None:
    # os.walk does not follow directory symlinks, so this one is retained by
    # the pruning loop and then never descended into: every file under it
    # leaves the identity silently. Following it instead would invite cycles
    # and hash content from outside the tree, so the arrangement is refused.
    _minimal_source(tmp_path)
    outside = tmp_path / "outside_the_tree"
    outside.mkdir()
    (outside / "important.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "src/kaggriculture/linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match=r"symlinked directory.*src/kaggriculture/linked"):
        source_identity(tmp_path)


def test_a_symlink_to_a_real_source_file_is_refused_rather_than_hashed(tmp_path: Path) -> None:
    # The companion to the dangling-lock skip below: that one is waved through
    # because there is nothing behind it, and this one must not be, because
    # hashing it would bind the identity to a path rather than to content.
    # Without this the loud path is one edit from becoming the silent one.
    _minimal_source(tmp_path)
    (tmp_path / "src/kaggriculture/alias.py").symlink_to(tmp_path / "src/kaggriculture/module.py")
    with pytest.raises(ValueError, match=r"regular non-symlink file"):
        source_identity(tmp_path)


def test_cargos_conventional_target_is_pruned_by_path_not_by_bare_name(tmp_path: Path) -> None:
    # Untagged on purpose. The suite's other build-output trees all carry a
    # CACHEDIR.TAG, so the tag rule masked this one entirely: deleting the name
    # exclusion left every test passing while `rust/kagg_env/target` -- the
    # directory an ordinary `cargo build` actually creates -- went into the
    # identity.
    _minimal_source(tmp_path)
    original = source_identity(tmp_path)
    generated = tmp_path / "rust/kagg_env/target/release"
    generated.mkdir(parents=True)
    (generated / "build_script.rs").write_text("// generated\n", encoding="utf-8")
    assert source_identity(tmp_path) == original

    # Scoped to the crate, because as a bare name it also swallowed directories
    # that have nothing to do with cargo.
    helper = tmp_path / "scripts/target"
    helper.mkdir(parents=True)
    (helper / "helper.py").write_text("HELPER = 1\n", encoding="utf-8")
    assert "scripts/target/helper.py" in source_identity(tmp_path)["files"]


def test_a_directory_the_walk_cannot_read_fails_instead_of_vanishing(tmp_path: Path) -> None:
    # os.walk's default is to swallow whatever scandir raises, so without an
    # onerror the subtree contributes nothing and reports nothing.
    _minimal_source(tmp_path)
    locked = tmp_path / "src/kaggriculture/locked"
    locked.mkdir(parents=True)
    (locked / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    locked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            source_identity(tmp_path)
    finally:
        locked.chmod(0o755)

    # A source root is the case only `onerror` covers. Every *sub*directory is
    # probed for a cache tag before the walk descends, and that probe raises on
    # its own -- Path.is_file() swallows ENOENT and friends but propagates
    # EACCES -- so a subdirectory is guarded twice over. The root of the walk
    # is never probed, and without onerror it comes back empty and quiet.
    root = tmp_path / "scripts"
    root.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            source_identity(tmp_path)
    finally:
        root.chmod(0o755)


def test_an_editors_dangling_lock_is_not_mistaken_for_a_missing_input(tmp_path: Path) -> None:
    # Emacs names its lock `.#<file>`, so a buffer open on `module.py` leaves a
    # dangling symlink called `.#module.py` -- a `.py` suffix with nothing
    # behind it. It is not an input and cannot become one, and aborting a
    # launch because a file is open in an editor would be indefensible.
    _minimal_source(tmp_path)
    original = source_identity(tmp_path)
    (tmp_path / "src/kaggriculture/.#module.py").symlink_to("someone@host.1234:1700000000")
    assert source_identity(tmp_path) == original


def test_a_source_file_is_not_excluded_for_sharing_a_name_with_a_build_directory(
    tmp_path: Path,
) -> None:
    # The exclusion applies to directories, not to every component of a path,
    # so a module that happens to be called target.py is source like any other.
    _minimal_source(tmp_path)
    (tmp_path / "scripts/target.py").write_text("print('target')\n", encoding="utf-8")

    assert "scripts/target.py" in source_identity(tmp_path)["files"]


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
            "mixed_report_sha256": "c" * 64,
            "mixed_report_size_bytes": 110,
            "compiled_report_sha256": "b" * 64,
            "compiled_report_size_bytes": 120,
            "rollout_forward_mode": "eager",
            "compile_update": True,
            "minimum_compile_speedup": 1.05,
            "attributed_knob_speedups": {"rollout_forward_mode": 0.5, "compile_update": 1.2},
            "training_command": ["intentionally", "excluded"],
            "run_dir": "/also/excluded",
        }
    )

    assert validate_run_provenance(provenance, required=True) == provenance
    assert "training_command" not in provenance["calibration"]
    # Either knob, moved on its own, must contradict the speedup recorded
    # beside it or fail the digest. A decision that survives being moved is
    # a decision the evidence does not actually constrain. The rollout knob is
    # moved to another mode rather than negated: it is mode-valued, `not "eager"`
    # is not a configuration anything could have run, and what has to be caught
    # is a mode that compiles sitting beside a 0.5x measurement.
    for knob, moved in (("rollout_forward_mode", "inductor"), ("compile_update", False)):
        calibration = provenance["calibration"]
        tampered = provenance | {"calibration": calibration | {knob: moved}}
        with pytest.raises(ValueError, match=r"contradicts|digest"):
            validate_run_provenance(tampered, required=True)


def test_an_integral_threshold_round_trips_instead_of_reading_as_tampering(
    tmp_path: Path,
) -> None:
    """The producer and the validator must digest the same canonical form.

    They did not: the producer hashed the decision's raw values while the
    validator hashed the normalized ones, and normalization coerces to float.
    An integral `minimum_compile_speedup` of 2 therefore rendered as `2` on one
    side and `2.0` on the other, and a record accused itself of tampering over
    a type. The landmine was that tightening the shipped constant to a whole
    number -- a one-character, entirely reasonable edit -- would have broken
    every calibrated launch with a corruption message.
    """
    _minimal_source(tmp_path)
    provenance = run_provenance_from_decision(
        {
            "source_identity": source_identity(tmp_path),
            "eager_report_sha256": "a" * 64,
            "eager_report_size_bytes": 100,
            "mixed_report_sha256": "c" * 64,
            "mixed_report_size_bytes": 110,
            "compiled_report_sha256": "b" * 64,
            "compiled_report_size_bytes": 120,
            "rollout_forward_mode": "eager",
            "compile_update": True,
            "minimum_compile_speedup": 2,
            "attributed_knob_speedups": {"rollout_forward_mode": 0.5, "compile_update": 3},
        }
    )

    assert validate_run_provenance(provenance, required=True) == provenance
    calibration = provenance["calibration"]
    assert isinstance(calibration["minimum_compile_speedup"], float)
    assert isinstance(calibration["attributed_knob_speedups"]["compile_update"], float)


@pytest.mark.parametrize("threshold", [0.4, 0.999, 1e-300])
def test_a_threshold_below_one_cannot_certify_a_phase_measured_slower(
    tmp_path: Path,
    threshold: float,
) -> None:
    """The record supplies both sides of the derivation, so internal
    consistency alone certifies nothing.

    Lowering the declared threshold flips a knob while leaving every measured
    value untouched and every other check green: here the genuine 0.5x rollout
    measurement -- compilation losing badly -- would be recorded as compiled.
    No timing has to be fabricated for this, which is what makes a floor the
    right shape of fix rather than more cross-checking.
    """
    _minimal_source(tmp_path)
    decision = {
        "source_identity": source_identity(tmp_path),
        "eager_report_sha256": "a" * 64,
        "eager_report_size_bytes": 100,
        "mixed_report_sha256": "c" * 64,
        "mixed_report_size_bytes": 110,
        "compiled_report_sha256": "b" * 64,
        "compiled_report_size_bytes": 120,
        "rollout_forward_mode": "inductor",
        "compile_update": True,
        "minimum_compile_speedup": threshold,
        "attributed_knob_speedups": {"rollout_forward_mode": 0.5, "compile_update": 2.3333},
    }

    # Pinned to the whole floor, not a prefix of it: `1\.0` is a substring of
    # "1.05", so the old pattern matched the shipped message by accident and
    # would go on matching if the constant were quietly lowered back to 1.0 --
    # which is precisely the regression this test is here to catch.
    with pytest.raises(ValueError, match=r"must be at least 1\.05;"):
        validate_run_provenance(run_provenance_from_decision(decision), required=True)


@pytest.mark.parametrize("value", [False, True, None, "", 1])
def test_a_rollout_knob_that_is_not_a_mode_is_rejected_rather_than_read_as_eager(
    tmp_path: Path,
    value: object,
) -> None:
    """This is what the bump to format version 4 is for.

    A format-3 record carries a boolean where the mode now goes, and `False` is
    not a mode: reading it as `eager` would let a record that never stated one
    validate as though it had, and `True` meant `cudagraphs` -- the one mode
    measurement rejects, at 5.309 ms against eager's 4.907 ms on the isolated
    collection forward. The knob has to be a mode name or nothing.
    """
    _minimal_source(tmp_path)
    decision = {
        "source_identity": source_identity(tmp_path),
        "eager_report_sha256": "a" * 64,
        "eager_report_size_bytes": 100,
        "mixed_report_sha256": "c" * 64,
        "mixed_report_size_bytes": 110,
        "compiled_report_sha256": "b" * 64,
        "compiled_report_size_bytes": 120,
        "rollout_forward_mode": value,
        "compile_update": True,
        "minimum_compile_speedup": 1.05,
        "attributed_knob_speedups": {"rollout_forward_mode": 0.5, "compile_update": 1.2},
    }

    with pytest.raises(ValueError, match="must be a collection forward mode"):
        run_provenance_from_decision(decision)
