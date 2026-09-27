"""A DONE final row is insufficient evidence that both policies executed."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def evaluator():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_public_agents.py"
    spec = importlib.util.spec_from_file_location("public_agent_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode():
    farms = [SimpleNamespace(money=4000.0), SimpleNamespace(money=3000.0)]
    steps = [
        [
            SimpleNamespace(
                status="ACTIVE",
                action={"farmer": ["PASS"], "hands": [], "market": []},
                observation=SimpleNamespace(farms=farms),
                reward=farms[seat].money,
            )
            for seat in (0, 1)
        ]
        for _ in range(720)
    ]
    for row in steps[-1]:
        row.status = "DONE"
    return SimpleNamespace(steps=steps, done=True)


def test_rejects_midgame_error_overwritten_at_terminal(evaluator):
    env = episode()
    env.steps[350][0].status = "ERROR"
    with pytest.raises(RuntimeError, match="Agent failure"):
        evaluator.validate_episode(env, {})


def test_rejects_last_turn_missing_action_despite_done(evaluator):
    env = episode()
    env.steps[-1][0].action = None
    with pytest.raises(RuntimeError, match="Missing or invalid"):
        evaluator.validate_episode(env, {})


def test_replay_requires_exact_recorded_banks(evaluator):
    env = episode()
    evaluator.validate_episode(env, {"expected_money": [4000, 3000]})
    with pytest.raises(RuntimeError, match="Replay reproduction failed"):
        evaluator.validate_episode(env, {"expected_money": [3999, 3000]})


def test_opposite_seats_share_one_seed_cluster(evaluator):
    rows = [
        {"agents": ["a", "b"], "seed": 1, "money": [4, 3]},
        {"agents": ["b", "a"], "seed": 1, "money": [4, 3]},
        {"agents": ["a", "b"], "seed": 2, "money": [4, 3]},
        {"agents": ["b", "a"], "seed": 2, "money": [4, 3]},
    ]
    result = evaluator.summarize(rows)[0]
    assert result["seed_clusters"] == 2
    assert result["score"] == 0.5
    assert result["score_cluster_standard_error"] == 0
    assert result["seat_scores"] == [1.0, 0.0]


def test_score_keeps_draws_distinct_from_wins(evaluator):
    rows = [
        {"agents": ["a", "b"], "seed": 1, "money": [4, 3]},
        {"agents": ["b", "a"], "seed": 1, "money": [4, 4]},
    ]
    result = evaluator.summarize(rows)[0]
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 0)
    assert result["score"] == 0.75
