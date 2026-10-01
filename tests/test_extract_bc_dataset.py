from __future__ import annotations

import importlib.util
import json
import sys
import time
import zlib
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from kaggriculture.demonstrations import (
    DemonstrationError,
    project_demonstration,
    verify_round_trip,
)


def _load_extractor():
    path = Path(__file__).parents[1] / "scripts" / "extract_bc_dataset.py"
    spec = importlib.util.spec_from_file_location("extract_bc_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_failed_projection_cancels_the_seeds_that_have_not_started() -> None:
    """One unrepresentable step must abort the whole extraction promptly.

    The dataset would otherwise be biased by the seeds the ledger silently
    dropped, and an operator waiting on a thousand-seed run needs the
    traceback now rather than after every remaining game has played out. The
    cancellation has to be issued from the collecting thread: asking the pool
    to do it via `shutdown(cancel_futures=True)` does not work, because the
    shutdown that runs while the exception unwinds withdraws the request
    before the pool's manager thread acts on it.

    Driving bare futures rather than a live pool keeps the queued/started
    split exact -- with a real executor a worker races to pick up the next
    seed the instant the first one fails.
    """
    extractor = _load_extractor()
    failed: Future = Future()
    failed.set_exception(DemonstrationError("step 3 seat 0: action is not representable"))
    queued: list[Future] = [Future() for _ in range(5)]
    pending = {failed: 0, **{future: seed for seed, future in enumerate(queued, start=1)}}

    with pytest.raises(DemonstrationError, match="not representable"):
        extractor.collect_extractions(pending, time.perf_counter())

    assert all(future.cancelled() for future in queued)


def test_successful_extraction_returns_every_submitted_seat() -> None:
    extractor = _load_extractor()
    pending: dict[Future, int] = {}
    for seed in range(4):
        future: Future = Future()
        future.set_result([{"seed": seed, "seat": 0}, {"seed": seed, "seat": 1}])
        pending[future] = seed

    episodes = extractor.collect_extractions(pending, time.perf_counter())

    assert sorted((record["seed"], record["seat"]) for record in episodes) == [
        (seed, seat) for seed in range(4) for seat in (0, 1)
    ]


def test_teacher_sits_both_seats_against_a_distinct_opponent() -> None:
    """A clone has to see the farm from both sides, not just seat 0."""
    extractor = _load_extractor()
    assert extractor.teacher_jobs("v16", "starter") == (
        ("v16", "starter", (0,)),
        ("starter", "v16", (1,)),
    )
    assert extractor.teacher_jobs("starter", "starter") == (("starter", "starter", (0, 1)),)


def test_failed_generation_commit_preserves_committed_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extractor = _load_extractor()
    output = tmp_path / "dataset"
    output.mkdir()
    old_archive = output / "episode-old.npz"
    old_archive.write_bytes(b"old generation")
    old_manifest = {"episodes": [{"file": old_archive.name}]}
    (output / "manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")

    staging = tmp_path / ".dataset.staging-test"
    staging.mkdir()
    new_archive = staging / "episode-new.npz"
    new_archive.write_bytes(b"new generation")
    manifest = {
        "episodes": [
            {
                "file": new_archive.name,
                "sha256": extractor.file_sha256(new_archive),
            }
        ]
    }

    def fail_manifest(*_args, **_kwargs) -> None:
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(extractor, "_write_manifest_atomic", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest failure"):
        extractor.commit_dataset_generation(staging, output, manifest)

    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == old_manifest
    assert old_archive.read_bytes() == b"old generation"


def test_resume_requires_matching_configuration_and_archive_digests(tmp_path: Path) -> None:
    extractor = _load_extractor()
    output = tmp_path / "dataset"
    output.mkdir()
    configuration = {
        "format_version": extractor.DATASET_FORMAT_VERSION,
        "teacher": {"label": "teacher", "sha256": "teacher-digest"},
        "opponent": {"label": "opponent", "sha256": "opponent-digest"},
        "episode_steps": 720,
        "seed_start": 4,
        "episode_count": 1,
        "extractor_source_identity": "source-digest",
    }
    records = []
    for seat in (0, 1):
        archive = output / f"episode-00000004-seat{seat}.npz"
        archive.write_bytes(f"seat {seat}".encode())
        records.append(
            {
                "file": archive.name,
                "seed": 4,
                "seat": seat,
                "sha256": extractor.file_sha256(archive),
            }
        )
    (output / "manifest.json").write_text(
        json.dumps({**configuration, "episodes": records}),
        encoding="utf-8",
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    recovered = extractor._load_resumable_records(output, staging, configuration)
    assert {(record["seed"], record["seat"]) for record in recovered} == {(4, 0), (4, 1)}

    mismatched = {
        **configuration,
        "teacher": {"label": "other", "sha256": "other-digest"},
    }
    with pytest.raises(ValueError, match="provenance does not match"):
        extractor._load_resumable_records(output, staging, mismatched)
    # A recovery corpus resumed as a clean one: the key the clean run would not
    # write still describes the committed archives.
    (output / "manifest.json").write_text(
        json.dumps(
            {**configuration, "perturbation": {"rate": 0.02, "seed": 0}, "episodes": records}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance does not match"):
        extractor._load_resumable_records(output, staging, configuration)
    (output / "manifest.json").write_text(
        json.dumps({**configuration, "episodes": records}), encoding="utf-8"
    )

    (output / records[0]["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        extractor._load_resumable_records(output, staging, configuration)


def test_recovery_labels_are_the_teachers_actions_while_deviations_execute() -> None:
    """DART: the archive labels every state with the teacher's own action.

    The executed action differs from it exactly at the perturbed steps, each
    executed deviation is itself legal under the projection ledger and changes
    one decision, and the archived game stays the executed one.
    """
    extractor = _load_extractor()
    steps, recoveries = extractor._play_episode(
        "starter",
        "starter",
        3,
        720,
        recovery_seats=(0,),
        perturbation_rate=0.1,
        perturbation_seed=1,
    )
    record = recoveries[0]
    assert len(record.labels) == len(record.perturbed) == 719
    perturbed = np.flatnonzero(record.perturbed)
    assert 30 < perturbed.size < 120

    for step in range(719):
        observation = extractor._seat_observation(steps, step, 0)
        executed = extractor._json_form(steps[step + 1][0].action)
        if not record.perturbed[step]:
            assert executed == record.labels[step]
            continue
        deviation = project_demonstration(observation, executed)
        verify_round_trip(observation, executed, deviation)
        teacher = project_demonstration(
            observation, record.labels[step], redundant_fertilize_as_pass=True
        )
        changed_units = int((deviation.unit_actions != teacher.unit_actions).sum())
        added_orders = len(deviation.canonical_action["market"]) - len(
            teacher.canonical_action["market"]
        )
        assert (changed_units, added_orders) in {(1, 0), (0, 1)}

    arrays = extractor.extract_episode(steps, 0, episode_steps=720, recovery=record)
    np.testing.assert_array_equal(arrays["perturbed"], record.perturbed)
    raw = json.loads(zlib.decompress(arrays["raw_json_zlib"]))
    assert raw["teacher_actions"] == record.labels
    assert raw["actions"] == [extractor._json_form(steps[t + 1][0].action) for t in range(719)]
    for step in perturbed:
        teacher = project_demonstration(
            extractor._seat_observation(steps, int(step), 0),
            record.labels[step],
            redundant_fertilize_as_pass=True,
        )
        np.testing.assert_array_equal(arrays["unit_actions"][step], teacher.unit_actions)
        np.testing.assert_array_equal(arrays["market_kinds"][step], teacher.market_kinds)
        np.testing.assert_array_equal(arrays["market_quantities"][step], teacher.market_quantities)


def test_a_failing_teacher_aborts_the_recovery_game(monkeypatch) -> None:
    """An agent error would otherwise be played as PASS to a DONE finish."""
    extractor = _load_extractor()

    def fail(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(extractor, "perturb_action", fail)
    with pytest.raises(DemonstrationError, match="step 0: ValueError: boom"):
        extractor._play_episode(
            "starter", "starter", 3, 720, recovery_seats=(0,), perturbation_rate=0.99
        )


def test_a_misaligned_recovery_record_is_rejected() -> None:
    extractor = _load_extractor()
    record = extractor.RecoveryRecord(labels=[{"farmer": ["PASS"]}], perturbed=[False])
    with pytest.raises(DemonstrationError, match="misaligned"):
        record.label(0, {"farmer": ["NORTH"]}, 0)
    with pytest.raises(DemonstrationError, match="no teacher action"):
        record.label(1, {"farmer": ["PASS"]}, 0)
    deviated = extractor.RecoveryRecord(labels=[{"farmer": ["PASS"]}], perturbed=[True])
    with pytest.raises(DemonstrationError, match="equals the teacher's at a perturbed"):
        deviated.label(0, {"farmer": ["PASS"]}, 0)
