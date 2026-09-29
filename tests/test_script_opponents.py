from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    MarketKind,
    UnitAction,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS, SEED_COST
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.rollout import EXTERNAL_AGENT_CODE, collect_mixed_play_rust
from kaggriculture.rust_env import load_native
from kaggriculture.script_opponents import (
    SCRIPT_AGENT_CONFIGURATION,
    ScriptAgentPool,
    ScriptOpponent,
    parse_script_opponent,
    shaped_observation,
    structify,
)

HORIZON = 719

# Logs every call as `namespace player step calls seed`, and buys three wheat
# seeds on its namespace's first call only: a namespace reused across seats or
# waves would skip the purchase, and one shared by two seats would count
# their calls together.
_LOGGING_AGENT = """
import os
import uuid

NAMESPACE = uuid.uuid4().hex
calls = 0


def agent(observation, configuration):
    global calls
    with open(os.environ["SCRIPT_AGENT_LOG"], "a") as log:
        log.write(
            f"{NAMESPACE} {observation.player} {observation.step} {calls} "
            f"{configuration.seed} {configuration.episodeSteps}\\n"
        )
    calls += 1
    market = [["BUY_SEED", "WHEAT", 3]] if calls == 1 else []
    return {"farmer": ["PASS"], "hands": [], "market": market}
"""

_RAISING_AGENT = """
def agent(observation):
    raise RuntimeError("broken agent")
"""


def _agent(tmp_path: Path, name: str, source: str) -> ScriptOpponent:
    path = tmp_path / name / "main.py"
    path.parent.mkdir()
    path.write_text(source)
    return ScriptOpponent.from_path(name, path)


def _actor() -> FarmActor:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return FarmActor(config)


def _log_rows(path: Path) -> dict[str, list[tuple[int, int, int, str, int]]]:
    rows: dict[str, list[tuple[int, int, int, str, int]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        namespace, player, step, calls, seed, episode_steps = line.split()
        rows[namespace].append((int(player), int(step), int(calls), seed, int(episode_steps)))
    return rows


def test_external_agent_code_mirrors_the_native_binding() -> None:
    assert load_native().EXTERNAL_AGENT_CODE == EXTERNAL_AGENT_CODE


def _step(environment, codes: np.ndarray) -> dict:
    rows = codes.size
    sampled = environment.sample_buffers()
    market = np.zeros((rows, MAX_MARKET_ORDERS), dtype=np.float32)
    environment.sample_and_step_into(
        np.zeros((rows, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.float32),
        np.zeros((rows, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.float32),
        np.zeros((rows, MAX_MARKET_ORDERS, 1), dtype=np.float32),
        np.zeros((1, N_MARKET_KINDS, 1), dtype=np.float32),
        np.zeros((1, N_QUANTITIES, 1), dtype=np.float32),
        np.zeros((1, N_MARKET_KINDS, N_QUANTITIES), dtype=np.float32),
        np.zeros(rows, dtype=np.uint16),
        np.zeros((rows, MAX_UNITS), dtype=np.float32),
        market,
        market,
        np.ones(rows, dtype=np.bool_),
        np.ones(rows, dtype=np.float32),
        codes,
        sampled,
    )
    return sampled


def _staged(environment, rows: list[int], *, kind: int = 0, unit: int = 0) -> None:
    count = len(rows)
    kinds = np.zeros((count, MAX_MARKET_ORDERS), dtype=np.uint8)
    kinds[:, 0] = kind
    units = np.full((count, MAX_UNITS), unit, dtype=np.uint8)
    environment.set_external_actions(
        np.asarray(rows, dtype=np.int64),
        units,
        kinds,
        np.zeros((count, MAX_MARKET_ORDERS), dtype=np.uint8),
    )


def test_staged_external_actions_play_exactly_the_coded_rows_once() -> None:
    native = load_native()
    environment = native.BatchEnv(np.asarray([5, 6], dtype=np.uint64))
    codes = np.asarray([0, EXTERNAL_AGENT_CODE, 0, 0], dtype=np.uint8)

    # Coded external with nothing staged, and staged but not coded, both refuse.
    with pytest.raises(ValueError, match="no action was staged"):
        _step(environment, codes)
    _staged(environment, [1])
    with pytest.raises(ValueError, match="coded 0"):
        _step(environment, np.zeros(4, dtype=np.uint8))
    # Nothing else may step past a staged row either.
    with pytest.raises(ValueError, match="staged external action"):
        environment.step_factors(
            np.zeros((2, 2, MAX_UNITS), dtype=np.uint8),
            np.zeros((2, 2, MAX_MARKET_ORDERS), dtype=np.uint8),
            np.zeros((2, 2, MAX_MARKET_ORDERS), dtype=np.uint8),
            external=True,
        )

    # A HIRE on row 1 is what that seat plays, with zero policy statistics.
    environment.reset(np.asarray([5, 6], dtype=np.uint64))
    hire = int(MarketKind.HIRE)
    _staged(environment, [1], kind=hire)
    sampled = _step(environment, codes)
    assert int(np.asarray(sampled["market_kinds"])[1, 0]) == hire
    np.testing.assert_array_equal(np.asarray(sampled["unit_logprobs"])[1], 0.0)
    snapshot = json.loads(environment.snapshot_json(0, False))
    assert len(snapshot["farms"][1]["hands"]) == 1
    assert len(snapshot["farms"][0]["hands"]) == 0
    # Consumed: the next step needs a fresh staging.
    with pytest.raises(ValueError, match="no action was staged"):
        _step(environment, codes)


def test_external_action_staging_validates_rows_and_values() -> None:
    environment = load_native().BatchEnv(np.asarray([5], dtype=np.uint64))
    with pytest.raises(IndexError):
        _staged(environment, [2])
    with pytest.raises(ValueError, match="already staged"):
        _staged(environment, [0, 0])
    with pytest.raises(ValueError, match="unit action"):
        _staged(environment, [0], unit=N_UNIT_ACTIONS)
    with pytest.raises(ValueError, match="market kind"):
        _staged(environment, [0], kind=N_MARKET_KINDS)
    # A refused call stages nothing, so a clean retry succeeds.
    _staged(environment, [0], unit=int(UnitAction.PASS))
    with pytest.raises(ValueError, match="already staged"):
        _staged(environment, [0])


def test_worker_structify_and_shaping_match_the_official_runner() -> None:
    from kaggle_environments import make
    from kaggle_environments.utils import structify as official_structify

    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 17})
    official = json.loads(json.dumps(environment.reset(2)[0].observation))
    snapshot = json.loads(load_native().BatchEnv(np.asarray([17], np.uint64)).snapshot_json(0))
    shaped = shaped_observation(snapshot, 0)
    assert shaped == official
    assert structify(shaped) == official_structify(official)
    assert structify(shaped).farms[0].money == official["farms"][0]["money"]
    # The runner's configuration, which never carries the episode seed.
    assert {**environment.configuration, "seed": None} == SCRIPT_AGENT_CONFIGURATION
    quirk = {"items": 1, "kept": [{"items": 2, "x": 3}]}
    assert structify(quirk) == official_structify(quirk) == {"kept": [{"x": 3}]}


def test_script_opponent_specs_pin_the_file(tmp_path) -> None:
    opponent = _agent(tmp_path, "logger", _LOGGING_AGENT)
    parsed = parse_script_opponent(f"logger={opponent.path}")
    assert parsed == opponent and parsed.key == "script_logger"
    for spec in ("logger", f"={opponent.path}", "logger="):
        with pytest.raises(ValueError):
            parse_script_opponent(spec)
    with pytest.raises(ValueError):
        ScriptOpponent.from_path("bad name", opponent.path)
    with pytest.raises(FileNotFoundError):
        ScriptOpponent.from_path("missing", tmp_path / "missing.py")


def test_script_lanes_play_fresh_namespaces_every_step_of_every_wave(tmp_path, monkeypatch) -> None:
    log = tmp_path / "calls.log"
    monkeypatch.setenv("SCRIPT_AGENT_LOG", str(log))
    logger = _agent(tmp_path, "logger", _LOGGING_AGENT)
    broken = _agent(tmp_path, "broken", _RAISING_AGENT)
    actor = _actor()
    # Game 0 is self-play; league games 1..4 meet pass, logger, broken, logger.
    assignments = np.asarray([0, 1, 2, 1], dtype=np.int64)
    with ScriptAgentPool([broken, logger], workers=2) as pool:
        batches = [
            collect_mixed_play_rust(
                actor,
                self_play_games=1,
                league_games=4,
                opponent_indices=assignments,
                builtin_lanes=("pass",),
                script_lanes=(logger, broken),
                script_pool=pool,
                seed_start=1_000,
                sampling_seed=3,
                forward_mode="eager",
            )
            for _ in range(2)
        ]
        statistics = pool.take_statistics()

    # Two waves, each with three script seats that played every step.
    assert statistics["seats"] == 6
    assert statistics["seat_steps"] == 6 * HORIZON
    assert statistics["agent_errors"] == 2 * HORIZON
    assert statistics["projection_errors"] == 0
    rows = _log_rows(log)
    assert len(rows) == 4
    for calls in rows.values():
        players = {player for player, *_ in calls}
        assert len(players) == 1
        assert [step for _, step, *_ in calls] == list(range(HORIZON))
        assert [count for _, _, count, *_ in calls] == list(range(HORIZON))
        assert {(seed, steps) for *_, seed, steps in calls} == {("None", 720)}

    first, second = batches
    np.testing.assert_array_equal(first.final_money, second.final_money)
    np.testing.assert_array_equal(first.opponent_money, second.opponent_money)
    # Learner rows are stored for every league game, on seed-parity seats.
    np.testing.assert_array_equal(first.seats[2:], first.episode_seeds[2:] % 2)
    league_opponent_money = first.opponent_money[2:]
    starting = 3_000.0
    # The pass seat and the raising agent (played as PASS) never spend; each
    # logger seat bought its three seeds exactly once.
    assert league_opponent_money[0] == starting
    assert league_opponent_money[2] == starting
    np.testing.assert_array_equal(league_opponent_money[[1, 3]], starting - 3 * SEED_COST["WHEAT"])


def test_script_lanes_refuse_missing_pool_and_learner_rows(tmp_path) -> None:
    logger = _agent(tmp_path, "logger", _LOGGING_AGENT)
    actor = _actor()
    with pytest.raises(ValueError, match="script agent pool"):
        collect_mixed_play_rust(
            actor,
            league_games=2,
            script_lanes=(logger,),
            seed_start=0,
            forward_mode="eager",
        )
    with pytest.raises(ValueError, match="even number"):
        collect_mixed_play_rust(
            actor,
            league_games=1,
            builtin_lanes=("pass",),
            paired_league_seats=True,
            seed_start=0,
            forward_mode="eager",
        )


def test_paired_league_seats_play_each_seed_from_both_seats() -> None:
    batch = collect_mixed_play_rust(
        _actor(),
        self_play_games=1,
        league_games=4,
        builtin_lanes=("pass",),
        seed_start=500,
        paired_league_seats=True,
        deterministic=True,
        forward_mode="eager",
    )
    np.testing.assert_array_equal(batch.episode_seeds, [500, 500, 501, 501, 502, 502])
    np.testing.assert_array_equal(batch.seats, [0, 1, 0, 1, 0, 1])


def test_interleaved_segments_share_workers_and_discard_abandoned_requests(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SCRIPT_AGENT_LOG", str(tmp_path / "calls.log"))
    logger = _agent(tmp_path, "logger", _LOGGING_AGENT)
    native = load_native()
    first_environment = native.BatchEnv(np.asarray([1, 2, 3], dtype=np.uint64))
    second_environment = native.BatchEnv(np.asarray([4, 5], dtype=np.uint64))
    codes = np.zeros(6, dtype=np.uint8)
    codes[[1, 2, 5]] = EXTERNAL_AGENT_CODE
    with ScriptAgentPool([logger], workers=2) as pool:
        first = pool.segment(
            first_environment,
            game_indices=np.asarray([0, 1, 2]),
            players=np.asarray([1, 0, 1]),
            opponents=np.zeros(3, dtype=np.int64),
        )
        second = pool.segment(
            second_environment,
            game_indices=np.asarray([0]),
            players=np.asarray([0]),
            opponents=np.zeros(1, dtype=np.int64),
        )
        first.request()
        second.request()
        # Staged in the opposite order: each worker's replies to the first
        # segment wait in the stash while the second collects its own.
        second.stage()
        first.stage()
        _step(first_environment, codes)
        with pytest.raises(RuntimeError, match="requested"):
            first.stage()
        # A segment abandoned mid-request leaves nothing for the next one.
        second.request()
        second.finish()
        third = pool.segment(
            second_environment,
            game_indices=np.asarray([1]),
            players=np.asarray([1]),
            opponents=np.zeros(1, dtype=np.int64),
        )
        third.request()
        third.stage()
        first.finish()
        third.finish()
        assert not pool._stash
        assert pool.take_statistics()["seats"] == 5


_HANGING_AGENT = """
import time


def agent(observation):
    time.sleep(3600)
"""

_EXITING_AGENT = """
import os


def agent(observation):
    os._exit(3)
"""


def _one_seat(pool: ScriptAgentPool):
    environment = load_native().BatchEnv(np.asarray([7], dtype=np.uint64))
    return pool.segment(
        environment,
        game_indices=np.asarray([0]),
        players=np.asarray([1]),
        opponents=np.zeros(1, dtype=np.int64),
    )


def test_a_hung_agent_is_killed_and_reported_where_it_hung(tmp_path) -> None:
    hanging = _agent(tmp_path, "hanging", _HANGING_AGENT)
    with ScriptAgentPool([hanging], workers=1, timeout=0.5) as pool:
        segment = _one_seat(pool)
        segment.request()
        with pytest.raises(RuntimeError, match=r"sent nothing for 0\.5 s") as raised:
            segment.stage()
        # faulthandler's dump names the agent's frame.
        assert 'main.py", line 6 in agent' in str(raised.value)
        assert pool.closed
        assert all(process.poll() is not None for process in pool._processes)
        # Releasing the seats of a dead pool neither raises nor hides the error.
        segment.finish()
        with pytest.raises(RuntimeError, match="closed"):
            _one_seat(pool)


def test_a_worker_that_dies_fails_the_wave_with_its_own_error(tmp_path) -> None:
    exiting = _agent(tmp_path, "exiting", _EXITING_AGENT)
    with ScriptAgentPool([exiting], workers=1) as pool:
        segment = _one_seat(pool)
        segment.request()
        with pytest.raises(RuntimeError, match="worker 0 exited"):
            segment.stage()
        segment.finish()
        assert pool._processes[0].returncode == 3
