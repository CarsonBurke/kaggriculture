#!/usr/bin/env python3
"""Run the shared official/native state, encoding, and reward parity oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rust.kagg_env.tests.parity_oracle import (  # noqa: E402
    MAX_TRANSITIONS,
    assert_official_native_parity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8, help="Number of consecutive games")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=MAX_TRANSITIONS)
    parser.add_argument("--factor-seed", type=int, default=20260812)
    parser.add_argument("--mode", choices=("pass", "random"), default="random")
    parser.add_argument("--debug-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = assert_official_native_parity(
        games=args.seeds,
        steps=args.steps,
        seed_start=args.seed_start,
        factor_seed=args.factor_seed,
        mode=args.mode,
        build=True,
        release=not args.debug_build,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
