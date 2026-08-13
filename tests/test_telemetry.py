from __future__ import annotations

import json
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from kaggriculture.telemetry import TensorboardMirror, migrate_jsonl_to_tensorboard


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    accumulator = EventAccumulator(str(log_dir)).Reload()
    return [(event.step, event.value) for event in accumulator.Scalars(tag)]


def test_training_jsonl_migration_is_idempotent_and_rebuilds_after_append(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    records = [
        {"iteration": 1, "loss": 0.5, "label": "ignored"},
        {"iteration": 2, "loss": 0.25},
    ]
    _write_jsonl(journal, records)

    first = migrate_jsonl_to_tensorboard(journal, log_dir)
    second = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert first.rebuilt
    assert not second.rebuilt
    assert _scalars(log_dir, "loss") == [(1, 0.5), (2, 0.25)]

    records.append({"iteration": 3, "loss": 0.125})
    _write_jsonl(journal, records)
    updated = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert updated.rebuilt
    assert _scalars(log_dir, "loss") == [(1, 0.5), (2, 0.25), (3, 0.125)]


def test_migration_repairs_a_corrupted_event_file_from_jsonl(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(journal, [{"iteration": 1, "score": 0.75}])
    migrate_jsonl_to_tensorboard(journal, log_dir)
    event_file = next(log_dir.glob("events.out.tfevents.*"))
    with event_file.open("ab") as stream:
        stream.write(b"corruption")

    repaired = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert repaired.rebuilt
    assert _scalars(log_dir, "score") == [(1, 0.75)]


def test_migration_rejects_a_destination_containing_its_source(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    log_dir.mkdir()
    journal = log_dir / "metrics.jsonl"
    _write_jsonl(journal, [{"iteration": 1, "score": 0.75}])

    with pytest.raises(ValueError, match="cannot contain"):
        migrate_jsonl_to_tensorboard(journal, log_dir)


def test_migration_rejects_a_symlink_destination(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "tensorboard"
    symlink.symlink_to(target, target_is_directory=True)
    _write_jsonl(journal, [{"iteration": 1, "score": 0.75}])

    with pytest.raises(ValueError, match="cannot be a symlink"):
        migrate_jsonl_to_tensorboard(journal, symlink)


def test_migration_rejects_missing_source_and_unowned_destination(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        migrate_jsonl_to_tensorboard(tmp_path / "missing.jsonl", tmp_path / "tensorboard")

    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "important"
    log_dir.mkdir()
    important = log_dir / "do-not-delete.txt"
    important.write_text("valuable", encoding="utf-8")
    _write_jsonl(journal, [{"iteration": 1, "score": 0.75}])

    with pytest.raises(ValueError, match="not owned"):
        migrate_jsonl_to_tensorboard(journal, log_dir)
    assert important.read_text(encoding="utf-8") == "valuable"


def test_migration_recovers_only_a_torn_final_record(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    journal.write_bytes(b'{"iteration":1,"score":0.75}\n{"iteration":2')

    migrated = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert migrated.records == 1
    assert _scalars(log_dir, "score") == [(1, 0.75)]

    journal.write_bytes(b'{"iteration":1}\n{"broken"\n{"iteration":3}\n')
    with pytest.raises(ValueError, match="invalid metrics record"):
        migrate_jsonl_to_tensorboard(journal, log_dir)


def test_failed_rebuild_preserves_the_current_mirror(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(journal, [{"iteration": 1, "score": 0.75}])
    migrate_jsonl_to_tensorboard(journal, log_dir)

    def fail_to_create_writer(path: Path):
        raise OSError(f"simulated failure in {path}")

    with pytest.raises(OSError, match="simulated failure"):
        migrate_jsonl_to_tensorboard(
            journal,
            log_dir,
            force=True,
            writer_factory=fail_to_create_writer,
        )

    assert _scalars(log_dir, "score") == [(1, 0.75)]


def test_live_mirror_recovers_a_committed_record(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "loss": 0.5}
    second = {"iteration": 2, "loss": 0.25}
    _write_jsonl(journal, [first])
    mirror = TensorboardMirror(journal, log_dir)
    _write_jsonl(journal, [first, second])

    mirror.record(second)
    mirror.close()

    assert _scalars(log_dir, "loss") == [(1, 0.5), (2, 0.25)]
    assert not migrate_jsonl_to_tensorboard(journal, log_dir).rebuilt


def test_live_mirror_survives_an_equal_length_torn_tail_replacement(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "loss": 0.5}
    second = {"iteration": 2, "loss": 0.25}
    _write_jsonl(journal, [first])
    with journal.open("ab") as stream:
        stream.write(json.dumps(second, sort_keys=True).encode("utf-8") + b"X")
    mirror = TensorboardMirror(journal, log_dir)
    torn_size = journal.stat().st_size

    # The journal appender truncates the torn suffix and commits a complete
    # record of exactly the same byte length, so file size alone cannot reveal
    # the change to the incremental mirror state.
    _write_jsonl(journal, [first, second])
    assert journal.stat().st_size == torn_size

    mirror.record(second)
    mirror.close()

    assert _scalars(log_dir, "loss") == [(1, 0.5), (2, 0.25)]


def test_live_mirror_repairs_corruption_before_recording(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "loss": 0.5}
    second = {"iteration": 2, "loss": 0.25}
    _write_jsonl(journal, [first])
    mirror = TensorboardMirror(journal, log_dir)
    event_file = next(log_dir.glob("events.out.tfevents.*"))
    with event_file.open("ab") as stream:
        stream.write(b"corruption")
    _write_jsonl(journal, [first, second])

    mirror.record(second)
    mirror.close()

    assert _scalars(log_dir, "loss") == [(1, 0.5), (2, 0.25)]


def test_live_mirror_self_heals_after_a_writer_failure(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "loss": 0.5}
    second = {"iteration": 2, "loss": 0.25}
    _write_jsonl(journal, [first])
    failed = False

    class FailingWriter:
        def __init__(self, writer: SummaryWriter) -> None:
            self.writer = writer

        def add_scalar(self, *args, **kwargs) -> None:
            raise OSError("simulated event-file failure")

        def add_text(self, *args, **kwargs) -> None:
            self.writer.add_text(*args, **kwargs)

        def flush(self) -> None:
            self.writer.flush()

        def close(self) -> None:
            self.writer.close()

    def factory(path: Path):
        nonlocal failed
        writer = SummaryWriter(path)
        if path.resolve() == log_dir.resolve() and not failed:
            failed = True
            return FailingWriter(writer)
        return writer

    mirror = TensorboardMirror(journal, log_dir, writer_factory=factory)
    _write_jsonl(journal, [first, second])

    mirror.record(second)
    mirror.close()

    assert failed
    assert _scalars(log_dir, "loss") == [(1, 0.5), (2, 0.25)]


def test_benchmark_jsonl_uses_separate_mode_and_batch_series(tmp_path: Path) -> None:
    journal = tmp_path / "rollout.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(
        journal,
        [
            {"event": "configuration", "compile_models": True, "games": [16]},
            {
                "event": "repeat",
                "games": 16,
                "repeat": 0,
                "complete_games_per_second": 4.5,
            },
            {
                "event": "batch_summary",
                "games": 16,
                "steady_state_complete_games_per_second": 5.0,
            },
        ],
    )

    migrate_jsonl_to_tensorboard(journal, log_dir)

    assert _scalars(log_dir, "rollout/compiled/games_16/complete_games_per_second") == [(0, 4.5)]
    assert _scalars(
        log_dir,
        "rollout/compiled/batch_summary/steady_state_complete_games_per_second",
    ) == [(16, 5.0)]
