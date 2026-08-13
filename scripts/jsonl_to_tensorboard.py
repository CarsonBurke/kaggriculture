#!/usr/bin/env python3
"""Idempotently migrate training or benchmark JSONL journals to TensorBoard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture.telemetry import migrate_jsonl_to_tensorboard, read_jsonl_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="root for one TensorBoard run per input (defaults beside each journal)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _destination(journal: Path, output_root: Path | None) -> Path:
    snapshot = read_jsonl_snapshot(journal)
    training = bool(snapshot.records) and all(
        record.get("event") is None and type(record.get("iteration")) is int
        for record in snapshot.records
    )
    if output_root is not None:
        return output_root.expanduser().resolve() / journal.stem
    if training and journal.name == "metrics.jsonl":
        return journal.expanduser().resolve().parent / "tensorboard"
    return journal.expanduser().resolve().parent / "tensorboard" / journal.stem


def main() -> None:
    args = parse_args()
    journals: list[Path] = []
    destinations: list[Path] = []
    for journal in args.journals:
        selected = journal.expanduser()
        if selected.is_symlink() or not selected.is_file():
            raise ValueError(f"journal is not a regular file: {selected}")
        resolved = selected.resolve()
        journals.append(resolved)
        destinations.append(_destination(resolved, args.output_root))
    if len(set(destinations)) != len(destinations):
        raise ValueError("multiple journals resolve to the same TensorBoard destination")

    results = []
    for journal, destination in zip(journals, destinations, strict=True):
        result = migrate_jsonl_to_tensorboard(
            journal,
            destination,
            force=args.force,
        )
        results.append(
            {
                "journal": str(journal.expanduser().resolve()),
                "log_dir": str(result.log_dir),
                "records": result.records,
                "rebuilt": result.rebuilt,
                "source_sha256": result.source_sha256,
            }
        )
    print(json.dumps({"migrations": results}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
