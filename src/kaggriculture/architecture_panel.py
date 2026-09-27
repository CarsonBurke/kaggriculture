"""Fixed development strength/calibration panel and a resumable plateau guard."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture.evaluation import DEVELOPMENT_SEED_START

PANEL_POLICY = {
    "version": 1,
    "seed_start": DEVELOPMENT_SEED_START,
    "games_per_opponent": 64,
    "opponents": ["starter", "scripted-v27"],
    "decoding": ["sampled", "argmax"],
    "sampling_seed": 20260928,
    "calibration_decoding": "sampled",
    "ema_alpha": 0.5,
    "warmup_actor_waves": 150,
    "patience_actor_waves": 100,
    "material_score_gain": 0.01,
    "material_mse_reduction": 0.01,
    "initialization_score_drop": 0.05,
}


class ArchitecturePanelGuard:
    """Cull persistent loss of fixed-panel strength only after both signals stall."""

    def __init__(self, interval: int, state: dict[str, Any] | None = None, *, iteration: int = 0):
        if type(interval) is not int or interval < 1:
            raise ValueError("architecture panel interval must be a positive integer")
        self.interval = interval
        blank = {
            "iteration": 0,
            "actor_waves": 0,
            "last_evaluated_wave": 0,
            "last_improvement_wave": 0,
            "evaluations": 0,
            "initial_score": None,
            "score_ema": None,
            "critic_mse_ema": None,
            "best_score_ema": None,
            "best_critic_mse_ema": None,
            "best_score": None,
            "best_checkpoint": None,
            "culled": False,
        }
        if state is None:
            if iteration:
                raise ValueError("resume checkpoint is missing architecture panel state")
            self.state = blank
            return
        if not isinstance(state, dict) or state.keys() != blank.keys():
            raise ValueError("invalid architecture panel state fields")
        self.state = copy.deepcopy(state)
        for name in (
            "iteration",
            "actor_waves",
            "last_evaluated_wave",
            "last_improvement_wave",
            "evaluations",
        ):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"invalid architecture panel {name}")
        if (
            state["iteration"] != iteration
            or state["actor_waves"] > iteration
            or state["last_evaluated_wave"] > state["actor_waves"]
            or state["last_improvement_wave"] > state["last_evaluated_wave"]
            or state["last_evaluated_wave"] != state["actor_waves"] // interval * interval
            or state["last_improvement_wave"] % interval
            or state["evaluations"] != 1 + state["last_evaluated_wave"] // interval
            or type(state["culled"]) is not bool
        ):
            raise ValueError("architecture panel state does not match checkpoint boundary")
        for name in ("initial_score", "score_ema", "best_score_ema", "best_score"):
            value = state[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid architecture panel {name}")
        for name in ("critic_mse_ema", "best_critic_mse_ema"):
            value = state[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid architecture panel {name}")
        if (
            state["evaluations"] < 1
            or not isinstance(state["best_checkpoint"], str)
            or not Path(state["best_checkpoint"]).is_file()
            or Path(state["best_checkpoint"]).is_symlink()
        ):
            raise ValueError("architecture panel best evaluated checkpoint is missing")
        if state["culled"] and not (
            state["actor_waves"] >= PANEL_POLICY["warmup_actor_waves"]
            and state["actor_waves"] - state["last_improvement_wave"]
            >= PANEL_POLICY["patience_actor_waves"]
            and state["score_ema"]
            <= state["initial_score"] - PANEL_POLICY["initialization_score_drop"]
        ):
            raise ValueError("architecture panel cull lacks sustained deterioration evidence")

    @property
    def culled(self) -> bool:
        return self.state["culled"]

    @property
    def due(self) -> bool:
        return (
            self.state["evaluations"] == 0
            or self.state["actor_waves"] - self.state["last_evaluated_wave"] >= self.interval
        )

    def advance(self, iteration: int, *, actor_active: bool) -> None:
        if iteration != self.state["iteration"] + 1:
            raise ValueError("architecture panel requires consecutive completed iterations")
        self.state["iteration"] = iteration
        self.state["actor_waves"] += int(actor_active)

    def observe(self, panel: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        score = float(panel["score_rate"])
        calibration = float(panel["critic_mse"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("architecture panel score must be finite and in [0,1]")
        if not math.isfinite(calibration) or calibration < 0:
            raise ValueError("architecture panel critic MSE must be finite and nonnegative")
        state = self.state
        if not self.due:
            raise ValueError("architecture panel was evaluated before its actor-wave boundary")
        first = state["evaluations"] == 0
        alpha = PANEL_POLICY["ema_alpha"]
        state["score_ema"] = score if first else alpha * score + (1 - alpha) * state["score_ema"]
        state["critic_mse_ema"] = (
            calibration if first else alpha * calibration + (1 - alpha) * state["critic_mse_ema"]
        )
        improved = False
        for current, best, threshold, direction in (
            ("score_ema", "best_score_ema", "material_score_gain", 1),
            ("critic_mse_ema", "best_critic_mse_ema", "material_mse_reduction", -1),
        ):
            value = state[current]
            if state[best] is None or direction * (value - state[best]) >= PANEL_POLICY[threshold]:
                state[best] = value
                improved = True
        if first:
            state["initial_score"] = score
        if state["best_score"] is None or score > state["best_score"]:
            state["best_score"] = score
            state["best_checkpoint"] = str(checkpoint.resolve())
        wave = state["actor_waves"]
        if improved:
            state["last_improvement_wave"] = wave
        state["last_evaluated_wave"] = wave
        state["evaluations"] += 1
        state["culled"] = bool(
            wave >= PANEL_POLICY["warmup_actor_waves"]
            and wave - state["last_improvement_wave"] >= PANEL_POLICY["patience_actor_waves"]
            and state["score_ema"]
            <= state["initial_score"] - PANEL_POLICY["initialization_score_drop"]
        )
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)


def evaluate_architecture_panel(
    actor,
    critic,
    *,
    compile_mode: str,
    calibration_actor=None,
) -> dict[str, Any]:
    """Native compiled BF16 strength plus V(s) calibration on unseen fixed seeds.

    When deployment uses averaged weights, pass the live actor separately:
    its critic must be calibrated on returns from its own sampled policy.

    Sampling generators are fixed, and the surrounding training RNG is restored.
    Each seed's learner seat is its parity, giving 32 games in each seat per
    opponent/mode. No gradient or optimizer update uses these development rows.
    """
    import torch

    from kaggriculture.critic_diagnostics import terminal_outcomes
    from kaggriculture.ppo import _stage_tensor, replay_behavior_values
    from kaggriculture.registry import architecture_of_config
    from kaggriculture.rollout import collect_mixed_play_rust

    device = next(actor.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_bf16_supported() or compile_mode == "eager":
        raise ValueError("architecture panel requires compiled CUDA BF16")
    architecture = architecture_of_config(actor.config).name
    calibration_actor = actor if calibration_actor is None else calibration_actor
    actor_training, critic_training = actor.training, critic.training
    calibration_training = calibration_actor.training
    numpy_rng = np.random.get_state()
    panels = []
    targets_sum = targets_squared = residual_squared = 0.0
    state_count = 0
    try:
        actor.eval()
        critic.eval()
        calibration_actor.eval()
        with torch.random.fork_rng(devices=[device]), torch.no_grad():
            for mode in PANEL_POLICY["decoding"]:
                for opponent in PANEL_POLICY["opponents"]:
                    collection = dict(
                        self_play_games=0,
                        league_games=PANEL_POLICY["games_per_opponent"],
                        builtin_lanes=(opponent,),
                        seed_start=PANEL_POLICY["seed_start"],
                        episode_steps=720,
                        deterministic=mode == "argmax",
                        temperature=1.0,
                        reward_mode="terminal-outcome",
                        sampling_seed=PANEL_POLICY["sampling_seed"],
                        forward_mode="inductor_graph",
                        forward_autocast=True,
                    )
                    rollout = collect_mixed_play_rust(actor, **collection)
                    outcomes = terminal_outcomes(rollout).astype(np.float64)
                    expected_seeds = np.arange(
                        PANEL_POLICY["seed_start"], PANEL_POLICY["seed_start"] + len(outcomes)
                    )
                    if (
                        rollout.valid.shape != (PANEL_POLICY["games_per_opponent"], 719)
                        or not np.array_equal(rollout.episode_seeds, expected_seeds)
                        or not np.array_equal(rollout.seats, expected_seeds % 2)
                        or rollout.learner_stochastic != (mode == "sampled")
                    ):
                        raise ValueError("architecture panel collector violated its fixed protocol")
                    if mode == PANEL_POLICY["calibration_decoding"]:
                        calibration = (
                            rollout
                            if calibration_actor is actor
                            else collect_mixed_play_rust(calibration_actor, **collection)
                        )
                        terminal_outcomes(calibration)
                        if (
                            calibration.valid.shape != rollout.valid.shape
                            or not np.all(calibration.valid)
                            or not np.array_equal(calibration.episode_seeds, expected_seeds)
                            or not np.array_equal(calibration.seats, expected_seeds % 2)
                            or not calibration.learner_stochastic
                        ):
                            raise ValueError("critic calibration violated its live-policy protocol")
                        staged = {
                            name: _stage_tensor(value, device)
                            for name, value in calibration.states.items()
                        }
                        staged |= {
                            name: _stage_tensor(getattr(calibration, name), device)
                            for name in ("unit_active", "unit_actions")
                        }
                        values = (
                            replay_behavior_values(
                                critic,
                                architecture,
                                staged,
                                compile_mode=compile_mode,
                                autocast_enabled=True,
                            )
                            .cpu()
                            .numpy()
                            .reshape(calibration.valid.shape)
                            .astype(np.float64)
                        )
                        if not np.isfinite(values).all():
                            raise FloatingPointError(
                                "architecture panel critic predictions are nonfinite"
                            )
                        paid = calibration.rewards[:, -1].astype(np.float64)
                        targets = np.broadcast_to(paid[:, None], values.shape)
                        state_count += values.size
                        targets_sum += float(targets.sum())
                        targets_squared += float(np.square(targets).sum())
                        residual_squared += float(np.square(values - targets).sum())
                        del staged, values, targets, calibration
                    panels.append(
                        {
                            "decoding": mode,
                            "opponent": opponent,
                            "score_rate": float(((outcomes + 1) / 2).mean()),
                            "outcomes": outcomes.tolist(),
                            "seeds": expected_seeds.tolist(),
                            "seats": rollout.seats.tolist(),
                        }
                    )
                    del rollout
    finally:
        np.random.set_state(numpy_rng)
        actor.train(actor_training)
        critic.train(critic_training)
        calibration_actor.train(calibration_training)
    total_variation = targets_squared - targets_sum * targets_sum / state_count
    return {
        "score_rate": float(np.mean([panel["score_rate"] for panel in panels])),
        "monte_carlo_r_squared": (
            1 - residual_squared / total_variation if total_variation > 0 else None
        ),
        "critic_mse": residual_squared / state_count,
        "critic_states": state_count,
        "critic_policy": "live" if calibration_actor is not actor else "deployment",
        "panels": panels,
    }
