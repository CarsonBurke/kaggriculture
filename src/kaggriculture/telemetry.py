"""Crash-recoverable JSONL journals mirrored into TensorBoard event logs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

TENSORBOARD_MIRROR_FORMAT_VERSION = 1
_MANIFEST_NAME = ".kaggriculture-tensorboard.json"


class SummaryWriterLike(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def add_text(self, tag: str, text_string: str, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


WriterFactory = Callable[[Path], SummaryWriterLike]


@dataclass(frozen=True)
class JournalSnapshot:
    path: Path
    sha256: str
    size_bytes: int
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MigrationResult:
    log_dir: Path
    records: int
    rebuilt: bool
    source_sha256: str


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant in metrics journal: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in metrics journal: {key}")
        result[key] = value
    return result


def _parse_journal_lines(
    contents: bytes, path: Path, first_line_number: int = 1
) -> tuple[dict[str, Any], ...]:
    """Strictly parse complete journal lines, ignoring only a torn final suffix."""
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"metrics journal is not UTF-8: {path}") from error
    if text and not text.endswith("\n"):
        text = text.rpartition("\n")[0]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=first_line_number):
        if not line:
            raise ValueError(f"metrics journal contains a blank record: {path}:{line_number}")
        try:
            record = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid metrics record at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"metrics record is not an object: {path}:{line_number}")
        records.append(record)
    return tuple(records)


def read_jsonl_snapshot(path: Path) -> JournalSnapshot:
    """Read one atomic journal snapshot, recovering only a torn final suffix."""
    path = Path(path).expanduser().resolve()
    contents = path.read_bytes() if path.exists() else b""
    digest = hashlib.sha256(contents).hexdigest()
    records = _parse_journal_lines(contents, path)
    return JournalSnapshot(path, digest, len(contents), records)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _event_files(log_dir: Path) -> dict[str, dict[str, int | str]]:
    if not log_dir.exists():
        return {}
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(log_dir.rglob("events.out.tfevents.*")):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"TensorBoard event path is not a regular file: {path}")
        relative = path.relative_to(log_dir).as_posix()
        result[relative] = {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
    return result


def _manifest_payload(snapshot: JournalSnapshot, log_dir: Path) -> dict[str, Any]:
    return {
        "format_version": TENSORBOARD_MIRROR_FORMAT_VERSION,
        "source": {
            "name": snapshot.path.name,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "records": len(snapshot.records),
        },
        "event_files": _event_files(log_dir),
    }


def _write_manifest(snapshot: JournalSnapshot, log_dir: Path) -> None:
    _write_manifest_payload(_manifest_payload(snapshot, log_dir), log_dir)


def _write_manifest_payload(payload: dict[str, Any], log_dir: Path) -> None:
    destination = log_dir / _MANIFEST_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=log_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _mirror_is_current(snapshot: JournalSnapshot, log_dir: Path) -> bool:
    manifest_path = log_dir / _MANIFEST_NAME
    if not log_dir.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = _manifest_payload(snapshot, log_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return manifest == expected


def _manifest_event_files_match(log_dir: Path) -> bool:
    manifest_path = log_dir / _MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("format_version") == TENSORBOARD_MIRROR_FORMAT_VERSION and manifest.get(
            "event_files"
        ) == _event_files(log_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _require_owned_or_empty_directory(destination: Path) -> None:
    if not destination.exists():
        return
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"TensorBoard destination is not a safe directory: {destination}")
    if not any(destination.iterdir()):
        return
    owner = destination / _MANIFEST_NAME
    if not owner.is_file() or owner.is_symlink():
        raise ValueError(
            "refusing to replace a nonempty directory not owned by the TensorBoard mirror: "
            f"{destination}"
        )


def _default_writer_factory(log_dir: Path) -> SummaryWriterLike:
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise FloatingPointError(f"non-finite TensorBoard scalar: {value}")
    return converted


#: Step fields a training journal may be keyed by, in precedence order. VAPO
#: counts iterations and behavior cloning counts epochs; both are a single
#: monotonic step index over a flat record of scalars, and neither carries the
#: `event` tag that marks a benchmark record.
_TRAINING_STEP_FIELDS = ("iteration", "epoch")


def training_step_field(record: dict[str, Any]) -> str | None:
    """The step field of a training journal record, or None if it is not one.

    A record that reaches TensorBoard without matching here is mirrored as a
    JSON text blob rather than as scalars, which is the right fallback for
    benchmark metadata and useless for a training curve -- so every training
    journal has to be recognized here or its curves silently do not appear.
    """
    if record.get("event") is not None:
        return None
    return next(
        (name for name in _TRAINING_STEP_FIELDS if type(record.get(name)) is int),
        None,
    )


def _configuration_context(configuration: dict[str, Any]) -> tuple[str, str]:
    mode = "compiled" if configuration.get("compile_models") is True else "eager"
    kind = "vapo" if "self_play_game_counts" in configuration else "rollout"
    return kind, mode


def _benchmark_context(records: tuple[dict[str, Any], ...]) -> tuple[str, str]:
    configuration = next(
        (record for record in records if record.get("event") == "configuration"),
        {},
    )
    return _configuration_context(configuration)


def _write_record(
    writer: SummaryWriterLike,
    record: dict[str, Any],
    record_index: int,
    context: tuple[str, str],
) -> None:
    step_field = training_step_field(record)
    if step_field is not None:
        step = int(record[step_field])
        for name, value in record.items():
            scalar = _number(value)
            if name != step_field and scalar is not None:
                writer.add_scalar(name, scalar, step)
        return

    event = record.get("event")
    kind, mode = context
    if event in {"iteration", "repeat"}:
        games = record.get("self_play_games", record.get("games"))
        step = record.get("repeat", record_index)
        if type(games) is not int or type(step) is not int:
            raise ValueError(f"benchmark {event} record lacks integer games/repeat fields")
        prefix = f"{kind}/{mode}/games_{games}"
        ignored = {"event", "repeat", "games", "self_play_games"}
    elif event == "batch_summary":
        games = record.get("self_play_games", record.get("games"))
        if type(games) is not int:
            raise ValueError("benchmark batch summary lacks an integer game count")
        step = games
        prefix = f"{kind}/{mode}/batch_summary"
        ignored = {"event", "games", "self_play_games"}
    else:
        writer.add_text(
            f"{kind}/{mode}/metadata/{event or 'record'}",
            "```json\n" + json.dumps(record, indent=2, sort_keys=True) + "\n```",
            record_index,
        )
        return

    for name, value in record.items():
        scalar = _number(value)
        if name not in ignored and scalar is not None:
            writer.add_scalar(f"{prefix}/{name}", scalar, step)


def _write_records(writer: SummaryWriterLike, snapshot: JournalSnapshot) -> None:
    context = _benchmark_context(snapshot.records)
    for index, record in enumerate(snapshot.records):
        _write_record(writer, record, index, context)
    writer.flush()


def _replace_derived_directory(staging: Path, destination: Path) -> None:
    _require_owned_or_empty_directory(destination)
    backup: Path | None = None
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"TensorBoard destination is not a directory: {destination}")
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.previous.", dir=destination.parent)
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def migrate_jsonl_to_tensorboard(
    journal_path: Path,
    log_dir: Path,
    *,
    force: bool = False,
    allow_missing_journal: bool = False,
    writer_factory: WriterFactory = _default_writer_factory,
) -> MigrationResult:
    """Idempotently rebuild a TensorBoard mirror from canonical JSONL bytes."""
    selected_journal = Path(journal_path).expanduser()
    if not selected_journal.exists() and not allow_missing_journal:
        raise FileNotFoundError(selected_journal)
    snapshot = read_jsonl_snapshot(selected_journal)
    selected_destination = Path(log_dir).expanduser()
    if selected_destination.is_symlink():
        raise ValueError(f"TensorBoard destination cannot be a symlink: {selected_destination}")
    destination = selected_destination.resolve()
    if destination.parent == destination or destination in snapshot.path.parents:
        raise ValueError("TensorBoard destination cannot contain the source journal")
    _require_owned_or_empty_directory(destination)
    if not force and _mirror_is_current(snapshot, destination):
        return MigrationResult(destination, len(snapshot.records), False, snapshot.sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    installed = False
    try:
        writer = writer_factory(staging)
        try:
            _write_records(writer, snapshot)
        finally:
            writer.close()
        _write_manifest(snapshot, staging)
        _replace_derived_directory(staging, destination)
        installed = True
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)
    return MigrationResult(destination, len(snapshot.records), True, snapshot.sha256)


class TensorboardMirror:
    """Live TensorBoard writer that repairs itself from its JSONL source journal.

    The journal and TensorBoard event files are both append-only, so per-record
    integrity tracking reads and hashes only the appended bytes. Any anomaly
    (shrinkage, torn suffix, parse failure, manifest mismatch) falls back to
    the full-read verification and rebuild paths.
    """

    def __init__(
        self,
        journal_path: Path,
        log_dir: Path,
        *,
        writer_factory: WriterFactory = _default_writer_factory,
    ) -> None:
        self.journal_path = Path(journal_path).expanduser().resolve()
        selected_log_dir = Path(log_dir).expanduser()
        self.writer_factory = writer_factory
        migrate_jsonl_to_tensorboard(
            self.journal_path,
            selected_log_dir,
            allow_missing_journal=True,
            writer_factory=self.writer_factory,
        )
        self.log_dir = selected_log_dir.resolve()
        self._open_writer()

    def _reset_journal_state(self) -> None:
        self._journal_sha = hashlib.sha256()
        self._journal_size = 0
        self._journal_records = 0
        self._last_record: dict[str, Any] | None = None
        self._configuration: dict[str, Any] = {}
        self._event_hashes: dict[str, tuple[Any, int]] = {}

    def _absorb_journal_records(self, records: tuple[dict[str, Any], ...]) -> None:
        self._journal_records += len(records)
        if records:
            self._last_record = records[-1]
        if not self._configuration:
            self._configuration = next(
                (record for record in records if record.get("event") == "configuration"),
                {},
            )

    def _reload_journal_state(self) -> None:
        self._reset_journal_state()
        contents = self.journal_path.read_bytes() if self.journal_path.exists() else b""
        if not contents.endswith(b"\n"):
            # Track only the newline-terminated prefix. A torn suffix may later
            # be truncated and replaced by an append of identical length, which
            # byte size alone cannot distinguish from an unchanged file.
            cut = contents.rfind(b"\n")
            contents = contents[: cut + 1] if cut >= 0 else b""
        records = _parse_journal_lines(contents, self.journal_path)
        self._journal_sha.update(contents)
        self._journal_size = len(contents)
        self._absorb_journal_records(records)

    def _extend_journal_state(self) -> bool:
        """Absorb appended journal bytes; False demands a full state reload."""
        size = self.journal_path.stat().st_size if self.journal_path.exists() else 0
        if size < self._journal_size:
            return False
        if size == self._journal_size:
            return True
        with self.journal_path.open("rb") as stream:
            stream.seek(self._journal_size)
            appended = stream.read(size - self._journal_size)
        if len(appended) != size - self._journal_size or not appended.endswith(b"\n"):
            return False
        try:
            records = _parse_journal_lines(
                appended, self.journal_path, first_line_number=self._journal_records + 1
            )
        except ValueError:
            return False
        self._journal_sha.update(appended)
        self._journal_size = size
        self._absorb_journal_records(records)
        return True

    def _current_event_files(self) -> dict[str, dict[str, int | str]]:
        """Hash event files incrementally, rehashing only on shrinkage."""
        result: dict[str, dict[str, int | str]] = {}
        for path in sorted(self.log_dir.rglob("events.out.tfevents.*")):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"TensorBoard event path is not a regular file: {path}")
            relative = path.relative_to(self.log_dir).as_posix()
            cached = self._event_hashes.get(relative)
            size = path.stat().st_size
            if cached is None or size < cached[1]:
                hasher, offset = hashlib.sha256(), 0
            else:
                hasher, offset = cached
            if size > offset:
                with path.open("rb") as stream:
                    stream.seek(offset)
                    while chunk := stream.read(1 << 20):
                        hasher.update(chunk)
                        offset += len(chunk)
            self._event_hashes[relative] = (hasher, offset)
            result[relative] = {"sha256": hasher.hexdigest(), "size_bytes": offset}
        return result

    def _manifest_matches(self) -> bool:
        manifest_path = self.log_dir / _MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return (
                manifest.get("format_version") == TENSORBOARD_MIRROR_FORMAT_VERSION
                and manifest.get("event_files") == self._current_event_files()
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _open_writer(self) -> None:
        writer = self.writer_factory(self.log_dir)
        try:
            writer.flush()
            _write_manifest(read_jsonl_snapshot(self.journal_path), self.log_dir)
        except BaseException:
            with suppress(Exception):
                writer.close()
            raise
        self.writer = writer
        self._reload_journal_state()

    def _repair(self) -> None:
        with suppress(Exception):
            self.writer.close()
        migrate_jsonl_to_tensorboard(
            self.journal_path,
            self.log_dir,
            force=True,
            allow_missing_journal=True,
            writer_factory=self.writer_factory,
        )
        self._open_writer()

    def record(self, payload: dict[str, Any]) -> None:
        """Mirror the journal's newly committed final record, repairing on failure."""
        if not self._extend_journal_state():
            self._reload_journal_state()
        if not self._journal_records or self._last_record != payload:
            raise ValueError("TensorBoard payload is not the committed final JSONL record")
        if not self._manifest_matches():
            self._repair()
            return
        try:
            context = _configuration_context(self._configuration)
            _write_record(self.writer, payload, self._journal_records - 1, context)
            self.writer.flush()
            _write_manifest_payload(
                {
                    "format_version": TENSORBOARD_MIRROR_FORMAT_VERSION,
                    "source": {
                        "name": self.journal_path.name,
                        "sha256": self._journal_sha.hexdigest(),
                        "size_bytes": self._journal_size,
                        "records": self._journal_records,
                    },
                    "event_files": self._current_event_files(),
                },
                self.log_dir,
            )
        except Exception:
            self._repair()

    def close(self) -> None:
        try:
            if not _manifest_event_files_match(self.log_dir):
                self._repair()
            self.writer.flush()
            self.writer.close()
            snapshot = read_jsonl_snapshot(self.journal_path)
            _write_manifest(snapshot, self.log_dir)
        except Exception:
            with suppress(Exception):
                self.writer.close()
            migrate_jsonl_to_tensorboard(
                self.journal_path,
                self.log_dir,
                force=True,
                allow_missing_journal=True,
                writer_factory=self.writer_factory,
            )
