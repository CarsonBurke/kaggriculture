"""Analytical checks for the no-training terminal utility comparison."""

import importlib.util
import math
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/compare_terminal_rewards.py"
_spec = importlib.util.spec_from_file_location("compare_terminal_rewards", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("own,other", [(0, 0), (0, 100000), (101000, 100000), (1e12, 1)])
def test_paused_reward_is_saturated_regularized_log_ratio(own, other):
    values = _module.rewards(own, other)
    transformed = math.tanh(math.tanh(values["log_ratio_starting_bank"] / 2) / 0.01)
    assert values["paused_run_soft"] == pytest.approx(transformed, abs=1e-12)
    for key, value in values.items():
        assert math.isfinite(value)
        assert _module.rewards(other, own)[key] == pytest.approx(-value)


def test_log_ratio_can_prefer_lower_win_rate():
    frequent = _module.summarize([(110000, 100000)] * 90 + [(90000, 100000)] * 10)
    large = _module.summarize([(200000, 100000)] * 60 + [(90000, 100000)] * 40)
    assert frequent["score_rate"] > large["score_rate"]
    assert (
        frequent["rewards"]["paused_run_soft"]["mean"] > large["rewards"]["paused_run_soft"]["mean"]
    )
    assert (
        frequent["rewards"]["log_ratio_starting_bank"]["mean"]
        < large["rewards"]["log_ratio_starting_bank"]["mean"]
    )


def test_nonnegative_finite_bank_contract():
    for value in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            _module.rewards(value, 0)


def test_saturating_soft_reward_can_trade_certain_win_for_risk():
    safe = _module.summarize([(100001, 100000)] * 100)
    risky = _module.summarize([(200000, 100000)] * 99 + [(0, 100000)])
    audit = _module.score_preference_audit({"safe": safe, "risky": risky})
    for name in ("signed_log1p_log_ratio", "signed_sqrt_log_ratio", "near_sign_log_ratio"):
        assert audit[name]["utility_maximizers"] == ["risky"]
        assert audit[name]["worst_score_regret_against_available_arms"] == pytest.approx(0.01)
    assert audit["win_loss"]["utility_maximizers"] == ["safe"]
    assert audit["win_loss_plus_001_bounded_margin"]["utility_maximizers"] == ["safe"]
