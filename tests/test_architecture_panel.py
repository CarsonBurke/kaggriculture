"""Fixed-panel guard state machine and mocked trainer publication contracts."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from test_train_ppo import _run_population_main, _training_script

from kaggriculture.architecture_panel import PANEL_POLICY, ArchitecturePanelGuard


def _panel(score=0.6, calibration=0.2):
    return {"score_rate": score, "critic_mse": calibration, "monte_carlo_r_squared": None}


def _guard(tmp_path):
    checkpoint = tmp_path / "checkpoint-000000.pt"
    checkpoint.touch()
    guard = ArchitecturePanelGuard(25)
    guard.observe(_panel(), checkpoint)
    return guard


def _advance(guard, wave, tmp_path, *, score=0.4, calibration=0.2, actor_active=True):
    guard.advance(wave, actor_active=actor_active)
    if guard.due:
        path = tmp_path / f"checkpoint-{wave:06d}.pt"
        path.touch()
        guard.observe(_panel(score, calibration), path)


def test_panel_requires_persistent_decline_and_both_signals_to_stall(tmp_path):
    guard = _guard(tmp_path)
    for wave in range(1, 150):
        _advance(guard, wave, tmp_path)
        assert not guard.culled
    _advance(guard, 150, tmp_path)
    assert guard.culled
    assert guard.state["evaluations"] == 7
    assert guard.state["best_checkpoint"].endswith("checkpoint-000000.pt")


@pytest.mark.parametrize("improvement", ["score", "calibration"])
def test_either_panel_improvement_resets_full_patience(tmp_path, improvement):
    guard = _guard(tmp_path)
    for wave in range(1, 126):
        _advance(
            guard,
            wave,
            tmp_path,
            score=0.95 if wave == 125 and improvement == "score" else 0.4,
            calibration=0.1 if wave == 125 and improvement == "calibration" else 0.2,
        )
    assert guard.state["last_improvement_wave"] == 125
    for wave in range(126, 225):
        _advance(guard, wave, tmp_path)
        assert not guard.culled
    _advance(guard, 225, tmp_path)
    assert guard.culled


def test_plateau_without_strength_loss_does_not_cull(tmp_path):
    guard = _guard(tmp_path)
    for wave in range(1, 301):
        _advance(guard, wave, tmp_path, score=0.6)
    assert not guard.culled


def test_constant_outcome_panels_can_cull_with_finite_heldout_mse(tmp_path):
    guard = _guard(tmp_path)
    for wave in range(1, 151):
        _advance(guard, wave, tmp_path, score=0.0, calibration=0.2)
    # _panel deliberately reports undefined R-squared for zero-variance outcomes.
    assert guard.culled
    assert guard.state["critic_mse_ema"] == pytest.approx(0.2)


@pytest.mark.parametrize("mse", [float("nan"), float("inf"), -0.1])
def test_guard_rejects_nonfinite_or_negative_mse(tmp_path, mse):
    guard = ArchitecturePanelGuard(25)
    with pytest.raises(ValueError, match="critic MSE"):
        guard.observe(_panel(calibration=mse), tmp_path / "checkpoint-000000.pt")


def test_warmup_does_not_age_panel_clock_and_resume_preserves_exact_decision(tmp_path):
    guard = _guard(tmp_path)
    for wave in range(1, 41):
        _advance(guard, wave, tmp_path, actor_active=False)
    assert guard.state["actor_waves"] == 0
    assert guard.state["evaluations"] == 1
    for wave in range(41, 116):
        _advance(guard, wave, tmp_path)
    state = guard.snapshot()
    resumed = ArchitecturePanelGuard(25, state, iteration=115)
    state["score_ema"] = 1.0
    assert resumed.state != state
    for wave in range(116, 191):
        _advance(guard, wave, tmp_path)
        _advance(resumed, wave, tmp_path)
        assert guard.state == resumed.state
    assert guard.culled


@pytest.mark.parametrize(
    "field,value",
    [
        ("actor_waves", 1),
        ("iteration", 4),
        ("score_ema", float("nan")),
        ("critic_mse_ema", float("inf")),
        ("best_checkpoint", "missing.pt"),
        ("evaluations", 0),
        ("culled", 1),
    ],
)
def test_resume_rejects_missing_or_invalid_guard_evidence(tmp_path, field, value):
    state = _guard(tmp_path).snapshot()
    state[field] = value
    with pytest.raises(ValueError):
        ArchitecturePanelGuard(25, state, iteration=0)


def test_panel_flag_is_opt_in_and_resume_bound(monkeypatch, tmp_path):
    module = _training_script()
    base = ["train_ppo.py", "--run-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base)
    args = module.parse_args()
    assert args.architecture_panel == 0
    assert module._training_data_config(args, torch.device("cpu"))["architecture_panel"] is None
    monkeypatch.setattr(sys, "argv", [*base, "--architecture-panel", "25"])
    args = module.parse_args()
    module._validate_args(args)
    policy = module._training_data_config(args, torch.device("cpu"))["architecture_panel"]
    assert policy == {**PANEL_POLICY, "interval_actor_waves": 25}
    for key, value in (
        ("architecture_panel", -1),
        ("device", "cpu"),
        ("autocull", True),
        ("population", 2),
        ("update_compile_mode", "eager"),
    ):
        changed = copy.copy(args)
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            module._validate_args(changed)


def test_panel_milestones_commit_and_culled_resume_cannot_restart(monkeypatch, tmp_path, capsys):
    module = _training_script()
    monkeypatch.setitem(PANEL_POLICY, "warmup_actor_waves", 4)
    monkeypatch.setitem(PANEL_POLICY, "patience_actor_waves", 2)
    # Exercise trainer state/publication on mocked waves, never an actual CPU
    # model forward; CUDA-only validation is covered independently above.
    validate = module._validate_args

    def validate_mocked(args):
        interval = args.architecture_panel
        args.architecture_panel = 0
        try:
            validate(args)
        finally:
            args.architecture_panel = interval

    monkeypatch.setattr(module, "_validate_args", validate_mocked)
    evaluations = []

    def evaluate(*args, **kwargs):
        evaluations.append(1)
        return _panel(0.8 if len(evaluations) == 1 else 0.4)

    monkeypatch.setattr(module, "evaluate_architecture_panel", evaluate)
    for resume in (None, tmp_path / "latest.pt"):
        with pytest.raises(SystemExit) as stopped:
            _run_population_main(
                module,
                monkeypatch,
                tmp_path,
                population=1,
                games=2,
                iterations=8,
                resume=resume,
                extra_arguments=("--architecture-panel", "1"),
            )
        assert stopped.value.code == 75
        checkpoint = torch.load(tmp_path / "latest.pt", weights_only=False)
        state = checkpoint["metrics"]["architecture_panel_state"]
        assert checkpoint["iteration"] == 4
        assert state["culled"]
        assert Path(state["best_checkpoint"]).is_file()
        assert len(evaluations) == 5
        assert all((tmp_path / f"checkpoint-{step:06d}.pt").is_file() for step in range(5))
        assert not (tmp_path / "checkpoint-000005.pt").exists()
        assert "fixed_panel_strength_and_calibration_stalled" in capsys.readouterr().out


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("separate_live_actor", [False, True])
def test_fixed_panel_protocol_rng_modes_and_sampled_only_calibration(
    monkeypatch, invalid, separate_live_actor
):
    from types import SimpleNamespace

    import numpy as np

    import kaggriculture.ppo as ppo
    import kaggriculture.rollout as rollout_module
    from kaggriculture.architecture_panel import evaluate_architecture_panel
    from kaggriculture.entity import EntityConfig

    class Network:
        config = EntityConfig()

        def __init__(self, training):
            self.training = training

        def parameters(self):
            return iter((SimpleNamespace(device=torch.device("cuda:0")),))

        def eval(self):
            self.training = False

        def train(self, value):
            self.training = value

    actor, critic = Network(True), Network(False)
    live_actor = Network(True) if separate_live_actor else actor
    collected, replayed = [], []
    fork_rng = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda devices: fork_rng(devices=[]))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    def collect(policy, **kwargs):
        assert not torch.is_grad_enabled()
        assert not actor.training and not critic.training
        assert not live_actor.training
        assert kwargs["self_play_games"] == 0
        assert kwargs["league_games"] == 64
        assert kwargs["forward_mode"] == "inductor_graph"
        assert kwargs["forward_autocast"] is True
        assert kwargs["reward_mode"] == "terminal-outcome"
        collected.append(kwargs)
        np.random.random()
        torch.rand(1)
        rewards = np.zeros((64, 719), dtype=np.float32)
        rewards[:, -1] = 1 if kwargs["deterministic"] else np.tile([-1, 1], 32)
        if separate_live_actor and policy is live_actor:
            # This policy loses every game; deployment's sampled policy does not.
            rewards[:, -1] = -1
        if invalid:
            rewards[0, -1] = np.nan
        seeds = np.arange(kwargs["seed_start"], kwargs["seed_start"] + 64)
        return SimpleNamespace(
            reward_mode="terminal-outcome",
            rewards=rewards,
            valid=np.ones_like(rewards, dtype=bool),
            episode_seeds=seeds,
            seats=seeds % 2,
            learner_stochastic=not kwargs["deterministic"],
            states={},
            unit_active=np.ones((64, 719, 16), dtype=bool),
            unit_actions=np.zeros((64, 719, 16), dtype=np.int8),
        )

    def values(*args, **kwargs):
        assert not collected[-1]["deterministic"]
        assert kwargs["compile_mode"] == "default" and kwargs["autocast_enabled"]
        replayed.append(1)
        return torch.full((64 * 719,), -1.0) if separate_live_actor else torch.zeros(64 * 719)

    monkeypatch.setattr(rollout_module, "collect_mixed_play_rust", collect)
    monkeypatch.setattr(ppo, "_stage_tensor", lambda array, device: torch.from_numpy(array))
    monkeypatch.setattr(ppo, "replay_behavior_values", values)
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    if invalid:
        with pytest.raises(ValueError, match="native terminal outcomes"):
            evaluate_architecture_panel(
                actor, critic, compile_mode="default", calibration_actor=live_actor
            )
    else:
        result = evaluate_architecture_panel(
            actor, critic, compile_mode="default", calibration_actor=live_actor
        )
        assert result["score_rate"] == 0.75
        assert result["monte_carlo_r_squared"] == (None if separate_live_actor else 0.0)
        assert result["critic_mse"] == (0.0 if separate_live_actor else 1.0)
        assert result["critic_policy"] == ("live" if separate_live_actor else "deployment")
        assert result["critic_states"] == 2 * 64 * 719
        assert len(collected) == (6 if separate_live_actor else 4) and len(replayed) == 2
        assert {call["builtin_lanes"] for call in collected} == {("starter",), ("scripted-v27",)}
        assert len({call["sampling_seed"] for call in collected}) == 1
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == numpy_state[0] and observed_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(observed_numpy[1], numpy_state[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    assert actor.training is True and critic.training is False
    assert live_actor.training is True
