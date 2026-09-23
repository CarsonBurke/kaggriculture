"""Model-free rollout field preparation shared by standalone critic diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def critic_replay_arrays(rollout: Any) -> dict[str, np.ndarray]:
    """Include shared masks needed by critic inputs and replay row counting.

    ``RolloutBatch.states`` contains architecture-specific observations. The
    own-unit activity mask and actions live in the shared rollout fields, so
    staging observations alone cannot satisfy the critic replay contract.
    """
    return {
        **rollout.states,
        "unit_active": rollout.unit_active,
        "unit_actions": rollout.unit_actions,
    }


def terminal_outcomes(rollout: Any) -> np.ndarray:
    """Read native win/draw/loss outcomes without comparing rounded bank balances."""
    rewards = np.asarray(rollout.rewards)
    valid = np.asarray(rollout.valid)
    if (
        rollout.reward_mode != "terminal-outcome"
        or rewards.ndim != 2
        or not all(rewards.shape)
        or rewards.shape != valid.shape
        or not valid.all()
        or not np.isfinite(rewards).all()
        or np.any(rewards[:, :-1] != 0)
        or not np.isin(rewards[:, -1], (-1.0, 0.0, 1.0)).all()
    ):
        raise ValueError("native terminal outcomes require complete finite win/draw/loss rewards")
    return rewards[:, -1].copy()
