#!/usr/bin/env python3
"""Regenerate the matched BC corpora with the current action space and both seats."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from kaggriculture.actions import N_UNIT_ACTIONS

DATASETS = (
    ("data/bc-v16-current-mirror-64", "public-v16", 0),
    ("data/bc-v16-current-starter-64", "starter", 1_000_000),
    ("data/bc-v16-current-pass-64", "pass", 2_000_000),
    ("data/bc-v16-current-random-64", "random", 3_000_000),
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative, opponent, seed_start in DATASETS:
        output = root / relative
        command = [
            sys.executable,
            "-u",
            str(root / "scripts/extract_bc_dataset.py"),
            "--output-dir",
            str(output),
            "--episodes",
            "64",
            "--seed-start",
            str(seed_start),
            "--teacher",
            "public-v16",
            "--opponent",
            opponent,
            "--workers",
            "8",
            # A committed manifest marks a corpus to resume; otherwise start one.
            *(["--resume"] if (output / "manifest.json").is_file() else []),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)

        manifest = json.loads((output / "manifest.json").read_text())
        episodes = manifest["episodes"]
        if len(episodes) != 128:
            raise RuntimeError(f"{output}: expected 128 paired-seat archives, got {len(episodes)}")
        seats_by_seed: dict[int, set[int]] = {}
        for record in episodes:
            seats_by_seed.setdefault(int(record["seed"]), set()).add(int(record["seat"]))
            with np.load(output / record["file"]) as archive:
                width = int(archive["unit_masks"].shape[-1])
            if width != N_UNIT_ACTIONS:
                raise RuntimeError(
                    f"{output / record['file']}: unit mask width {width}, expected {N_UNIT_ACTIONS}"
                )
        if len(seats_by_seed) != 64 or any(seats != {0, 1} for seats in seats_by_seed.values()):
            raise RuntimeError(f"{output}: every one of 64 seeds must contain both teacher seats")
        print(
            f"validated {output}: 64 seeds, both seats, {N_UNIT_ACTIONS} unit actions",
            flush=True,
        )


if __name__ == "__main__":
    main()
