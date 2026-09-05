"""Focused, full-episode production parity gates.

These tests expect the release native extension and pinned public-v27 artifact to
already exist. They deliberately do not build either prerequisite.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rust.kagg_env.tests.parity_oracle import assert_official_native_parity  # noqa: E402
from scripts.audit_v27_parity import MAX_TRANSITIONS, assert_v27_parity  # noqa: E402


def test_full_production_episode_matches_official_oracle() -> None:
    report = assert_official_native_parity(
        games=1,
        steps=MAX_TRANSITIONS,
        seed_start=0,
        factor_seed=20260812,
        mode="random",
        build=False,
    )

    assert report["transitions_per_game"] == MAX_TRANSITIONS
    assert report["joint_transitions"] == MAX_TRANSITIONS
    assert report["submitted_cases_exercised"] is True


def test_scripted_v27_matches_pinned_agent_for_full_episode() -> None:
    report = assert_v27_parity(
        games=1,
        episodes=1,
        seed_start=90_001,
        steps=MAX_TRANSITIONS,
        build=False,
    )

    assert report["actions_compared"] == 2 * MAX_TRANSITIONS
    assert len(report["episodes"]) == 3
