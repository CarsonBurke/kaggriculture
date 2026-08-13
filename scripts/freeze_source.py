#!/usr/bin/env python3
"""Create a content-addressed, read-only source snapshot for an ML job DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture.provenance import freeze_source, source_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/source-snapshots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = source_identity()
    destination = args.output_root.expanduser().resolve() / identity["sha256"]
    freeze_source(destination)
    print(
        json.dumps(
            {"source_root": str(destination), "source_identity": identity},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
