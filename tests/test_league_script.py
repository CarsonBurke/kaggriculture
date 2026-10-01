from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kaggriculture import opponents
from kaggriculture.registry import CONV_ENTITY
from kaggriculture.script_opponents import ScriptOpponent


def _training_script():
    path = Path(__file__).parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_train_ppo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent_file(tmp_path: Path, name: str = "rival") -> Path:
    path = tmp_path / name / "main.py"
    path.parent.mkdir()
    path.write_text("def agent(observation):\n    return {}\n")
    return path


def _parsed(module, monkeypatch, tmp_path: Path, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ppo.py", "--run-dir", str(tmp_path / "run"), "--architecture-panel", "0", *extra],
    )
    return module.parse_args()


def test_script_lanes_default_to_the_league_reference_agents(monkeypatch, tmp_path) -> None:
    references = tmp_path / "references"
    references.mkdir()
    for name in opponents.REFERENCE_AGENTS:
        (references / f"{name}.py").write_text("def agent(observation):\n    return {}\n")
    monkeypatch.setattr(opponents, "REFERENCE_AGENT_DIR", references)
    module = _training_script()
    default = _parsed(module, monkeypatch, tmp_path)
    module._validate_args(default)
    assert default.league_script_opponent == list(opponents.LEAGUE_REFERENCE_AGENTS)
    assert default.league_script_games == 8 * len(opponents.LEAGUE_REFERENCE_AGENTS)
    assert (default.league_builtin_opponents, default.league_builtin_lanes) == ("", 0)
    assert module._wave_games(default, 1) == (
        default.games + default.league_games + default.league_script_games
    )
    recorded = module._training_data_config(default, torch.device("cpu"))
    assert [lane["name"] for lane in recorded["league_script_opponents"]] == list(
        opponents.LEAGUE_REFERENCE_AGENTS
    )
    population = _parsed(module, monkeypatch, tmp_path, "--population", "2")
    assert (population.league_script_opponent, population.league_script_games) == ([], 0)


def test_script_lane_flags_are_recorded_when_set(monkeypatch, tmp_path) -> None:
    module = _training_script()
    default = _parsed(module, monkeypatch, tmp_path, "--league-script-games", "0")
    module._validate_args(default)
    assert (default.league_script_opponent, default.league_script_games) == ([], 0)

    path = _agent_file(tmp_path)
    args = _parsed(
        module,
        monkeypatch,
        tmp_path,
        "--league-script-opponent",
        f"rival={path}",
        "--league-script-games",
        "16",
    )
    module._validate_args(args)
    # On top of the league budget, which keeps its recipe.
    assert args.league_games == default.league_games
    assert module._wave_games(args, 1) == args.games + args.league_games + 16
    recorded = module._training_data_config(args, torch.device("cpu"))
    assert recorded["league_script_games"] == 16
    assert recorded["league_script_opponents"] == [
        {"name": "rival", "sha256": ScriptOpponent.from_path("rival", path).sha256}
    ]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--league-script-games", "4"), "set together"),
        (("--league-script-opponent", "rival={path}"), "set together"),
        (
            (
                "--league-script-opponent",
                "rival={path}",
                "--league-script-opponent",
                "rival={path}",
                "--league-script-games",
                "4",
            ),
            "distinct",
        ),
        (
            (
                "--league-script-opponent",
                "rival={path}",
                "--league-script-opponent",
                "other={path}",
                "--league-script-games",
                "1",
            ),
            "cover every script opponent",
        ),
        (("--league-script-opponent", "rival={path}", "--league-script-games", "-1"), "negative"),
        (
            (
                "--league-script-opponent",
                "rival={path}",
                "--league-script-games",
                "4",
                "--league-script-workers",
                "0",
            ),
            "workers",
        ),
    ],
)
def test_script_lane_launch_errors(monkeypatch, tmp_path, extra, message) -> None:
    module = _training_script()
    path = _agent_file(tmp_path)
    args = _parsed(module, monkeypatch, tmp_path, *(value.format(path=path) for value in extra))
    with pytest.raises(ValueError, match=message):
        module._validate_args(args)


def test_a_population_refuses_script_lanes(monkeypatch, tmp_path) -> None:
    module = _training_script()
    path = _agent_file(tmp_path)
    args = _parsed(
        module,
        monkeypatch,
        tmp_path,
        "--population",
        "2",
        "--games",
        "2",
        "--league-script-opponent",
        f"rival={path}",
        "--league-script-games",
        "2",
    )
    with pytest.raises(ValueError, match="script"):
        module._validate_args(args)


@pytest.mark.parametrize("reward_mode", ["terminal-bank", "shaped"])
def test_script_diagnostics_report_each_opponent_apart_from_selected_lanes(
    tmp_path, reward_mode
) -> None:
    module = _training_script()
    league = SimpleNamespace(
        final_money=np.asarray([100.0, 50.0, 80.0, 90.0, 200.0, 10.0]),
        opponent_money=np.asarray([90.0, 60.0, 80.0, 20.0, 30.0, 40.0]),
        reward_mode=reward_mode,
    )
    # Two selected lanes (0, 1), then the script lanes 2 and 3.
    assignments = np.asarray([0, 2, 1, 3, 2, 2])
    first = ScriptOpponent.from_path("first", _agent_file(tmp_path, "first"))
    second = ScriptOpponent.from_path("second", _agent_file(tmp_path, "second"))
    statistics = {"seats": 4, "agent_errors": 1, "wait_seconds": 0.5}

    diagnostics = module._league_script_diagnostics(
        league, assignments, 2, [first, second], statistics
    )

    assert diagnostics["league_script_seats"] == 4
    assert diagnostics["league_script_agent_errors"] == 1
    assert diagnostics["league_script_wait_seconds"] == 0.5
    # Games 1, 4, 5: a loss, a win and a loss.
    assert diagnostics["league_script_first_games"] == 3
    assert diagnostics["league_script_first_score_rate"] == pytest.approx(1 / 3)
    assert diagnostics["league_script_first_own_bank"] == pytest.approx(260.0 / 3)
    assert diagnostics["league_script_first_opponent_bank"] == pytest.approx(130.0 / 3)
    assert diagnostics["league_script_first_mean_margin"] == pytest.approx(130.0 / 3)
    assert diagnostics["league_script_second_games"] == 1
    assert diagnostics["league_script_second_score_rate"] == 1.0
    assert diagnostics["league_script_second_mean_margin"] == 70.0


def test_a_wave_with_no_selectable_lane_still_plays_its_script_games(monkeypatch, tmp_path) -> None:
    # Iteration 0 of a snapshot-only league has no eligible lane, so the wave
    # is self-play plus the script lane alone: shorter than the arena, which
    # is sized for every league game. It must collect into the matching prefix.
    module = _training_script()
    run_provenance = module.run_provenance_from_decision(
        {
            "source_identity": module.source_identity(),
            "rollout_forward_mode": "inductor_graph",
            "update_compile_mode": "default",
            "eager_report_sha256": "a" * 64,
            "eager_report_size_bytes": 100,
            "mixed_report_sha256": "b" * 64,
            "mixed_report_size_bytes": 110,
            "compiled_report_sha256": "c" * 64,
            "compiled_report_size_bytes": 120,
            "minimum_compile_speedup": 1.05,
            "attributed_knob_speedups": {
                "rollout_forward_mode": 1.1,
                "update_compile_mode": 1.1,
            },
        }
    )
    monkeypatch.setattr(module, "_load_run_provenance", lambda *_args, **_kwargs: run_provenance)

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    def parity(*args, **kwargs) -> dict[str, float]:
        metrics: dict[str, float] = {"update_replay_max_kl": 0.0}
        metrics["update_replay_max_tail_fraction"] = 0.0
        for component in module.PARITY_COMPONENTS:
            metrics[f"update_replay_{component}_kl"] = 0.0
            metrics[f"update_replay_{component}_tail_fraction"] = 0.0
            metrics[f"update_replay_{component}_active_count"] = 1
        return metrics

    collected: list[int] = []
    collect = module.collect_mixed_play_rust

    def recording_collect(*args, **kwargs):
        collected.append(kwargs["league_games"])
        return collect(*args, **kwargs)

    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(module, "collect_mixed_play_rust", recording_collect)
    monkeypatch.setattr(module, "update_replay_parity", parity)
    monkeypatch.setattr(
        module,
        "update_ppo",
        lambda *args, **kwargs: {
            "actor_updates": 1,
            "actor_minibatches_intended": 1,
            "critic_updates": 1,
            "first_minibatch_component_kl": 0.0,
            "value_target_saturated_fraction": 0.0,
            "entropy": 0.2,
        },
    )
    path = _agent_file(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--run-dir",
            str(run_dir),
            "--iterations",
            "1",
            "--games",
            "1",
            "--league-games",
            "2",
            "--league-active-opponents",
            "1",
            "--league-historical-opponents",
            "1",
            "--league-builtin-lanes",
            "0",
            "--league-script-opponent",
            f"rival={path}",
            "--league-script-games",
            "2",
            "--league-script-workers",
            "1",
            "--device",
            "cpu",
            "--architecture",
            CONV_ENTITY,
            "--architecture-panel",
            "0",
            "--cnn-width",
            "8",
            "--cnn-blocks",
            "1",
            "--no-bfloat16",
        ],
    )

    module.main()

    assert collected == [2]
    (metrics,) = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics["league_script_rival_games"] == 2
    # Both seats act at every one of the episode's 719 native steps.
    assert metrics["league_script_seat_steps"] == 2 * 719
    assert metrics["league_script_agent_errors"] == 0
    assert 0 < metrics["league_script_slowest_reply_seconds"] < 30
