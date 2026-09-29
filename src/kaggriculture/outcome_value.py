"""Outcome-only critic supervision and pre-update match-score calibration.

Class order is loss, draw, win. These targets describe completed games, never
rounded value bootstraps or money margins. Calibration is measured on behavior
predictions before fitting the collected wave; it is not held-out Elo evidence.
"""

from __future__ import annotations

import numpy as np
from torch import Tensor


def validate_outcome_objective(reward_mode: str, gamma: float, critic_lambda: float) -> None:
    if reward_mode != "terminal-outcome" or gamma != 1.0 or critic_lambda != 1.0:
        raise ValueError(
            "WDL critic requires terminal-outcome rewards, gamma 1, and critic GAE lambda 1; "
            "disable it (--wdl-value false) for other objectives"
        )


def validate_terminal_outcome_rewards(rewards: np.ndarray, valid: np.ndarray) -> None:
    """Require one outcome at each trajectory's last valid step; allow padding."""
    if rewards.ndim != 2 or rewards.shape != valid.shape or not rewards.size:
        raise ValueError("WDL rewards and validity must be matching nonempty trajectory arrays")
    valid = valid.astype(bool)
    lengths = valid.sum(axis=1)
    if np.any(lengths == 0) or np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("WDL trajectories require a nonempty contiguous valid prefix")
    last = np.arange(rewards.shape[1])[None, :] == lengths[:, None] - 1
    if np.any(rewards[valid & ~last] != 0) or not np.isin(rewards[last], (-1, 0, 1)).all():
        raise ValueError("WDL requires zero intermediate rewards and a terminal loss/draw/win")


def outcome_value_loss(logits: Tensor, targets: Tensor, *, validate: bool = True) -> Tensor:
    """Unsmoothed categorical cross entropy, with no gradients through labels."""
    if logits.shape != (*targets.shape, 3):
        raise ValueError("WDL logits must have three classes per target")
    if validate and not bool(((targets == -1) | (targets == 0) | (targets == 1)).all()):
        raise ValueError("WDL targets must be exact completed-game outcomes -1, 0, or 1")
    labels = targets.detach().long() + 1
    return -logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def match_score_calibration(values: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """State-weighted diagnostics for expected score, usable for either critic.

    Preserve out-of-range predictions in errors rather than hiding them by
    clipping. Only bucket assignment clips. Draws have score 0.5, so this MSE
    is not the three-class Brier score and cannot identify draw calibration.
    """
    if values.shape != outcomes.shape or not values.size:
        raise ValueError("calibration needs matching nonempty prediction and outcome arrays")
    if not np.isfinite(values).all() or not np.isin(outcomes, (-1, 0, 1)).all():
        raise ValueError("calibration requires finite predictions and exact outcomes")
    predictions = (values.astype(np.float64).ravel() + 1.0) / 2.0
    scores = (outcomes.astype(np.float64).ravel() + 1.0) / 2.0
    buckets = np.clip((predictions * 10).astype(np.int64), 0, 9)
    counts = np.bincount(buckets, minlength=10)
    errors = np.bincount(buckets, weights=predictions - scores, minlength=10)
    return {
        "behavior_match_score_mse": float(np.mean((predictions - scores) ** 2)),
        "behavior_match_score_bias": float(np.mean(predictions - scores)),
        "behavior_match_score_calibration_error": float(
            np.abs(errors[counts > 0]).sum() / predictions.size
        ),
        "behavior_match_score_out_of_range_fraction": float(
            np.mean((predictions < 0) | (predictions > 1))
        ),
    }
