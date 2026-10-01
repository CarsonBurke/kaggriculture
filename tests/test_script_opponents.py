from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from kaggriculture.constants import (
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRIVATE_ITEMS,
    SEED_COST,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.rollout import EXTERNAL_AGENT_CODE, collect_mixed_play_rust
from kaggriculture.rust_env import load_native
from kaggriculture.script_opponents import (
    SCRIPT_AGENT_CONFIGURATION,
    UNIT_COMMAND_ACTION,
    UNIT_COMMAND_PICKUP,
    UNIT_COMMAND_PLACE,
    ScriptAgentPool,
    ScriptOpponent,
    SubmittedActionError,
    encode_submitted_turn,
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


def _staged(environment, turns: dict[int, dict]) -> None:
    encoded = [encode_submitted_turn(action) for action in turns.values()]
    environment.set_submitted_actions(
        np.asarray(list(turns), dtype=np.int64),
        np.asarray([len(units) for units, _ in encoded], dtype=np.int64),
        np.concatenate([units for units, _ in encoded]).reshape(-1, 3),
        np.stack([orders for _, orders in encoded]).reshape(-1, MAX_MARKET_ORDERS, 2),
    )


_PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def test_staged_external_actions_play_exactly_the_coded_rows_once() -> None:
    native = load_native()
    environment = native.BatchEnv(np.asarray([5, 6], dtype=np.uint64))
    codes = np.asarray([0, EXTERNAL_AGENT_CODE, 0, 0], dtype=np.uint8)

    # Coded external with nothing staged, and staged but not coded, both refuse.
    with pytest.raises(ValueError, match="no action was staged"):
        _step(environment, codes)
    _staged(environment, {1: _PASS})
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
    _staged(environment, {1: {"market": [["HIRE"]]}})
    sampled = _step(environment, codes)
    np.testing.assert_array_equal(np.asarray(sampled["unit_logprobs"])[1], 0.0)
    np.testing.assert_array_equal(np.asarray(sampled["market_kind_logprobs"])[1], 0.0)
    snapshot = json.loads(environment.snapshot_json(0, False))
    assert len(snapshot["farms"][1]["hands"]) == 1
    assert len(snapshot["farms"][0]["hands"]) == 0
    # Consumed: the next step needs a fresh staging.
    with pytest.raises(ValueError, match="no action was staged"):
        _step(environment, codes)


def test_submitted_action_staging_validates_rows_and_values() -> None:
    environment = load_native().BatchEnv(np.asarray([5], dtype=np.uint64))
    no_orders = np.zeros((1, MAX_MARKET_ORDERS, 2), dtype=np.uint32)

    def stage(units, *, orders=no_orders, counts=None, rows=(0,)) -> None:
        units = np.asarray(units, dtype=np.uint32).reshape(-1, 3)
        environment.set_submitted_actions(
            np.asarray(rows, dtype=np.int64),
            np.asarray([len(units)] if counts is None else counts, dtype=np.int64),
            units,
            orders,
        )

    with pytest.raises(IndexError):
        _staged(environment, {2: _PASS})
    with pytest.raises(ValueError, match="already staged"):
        stage(
            [[UNIT_COMMAND_ACTION, 0, 0]] * 2,
            counts=[1, 1],
            rows=[0, 0],
            orders=np.zeros((2, MAX_MARKET_ORDERS, 2), dtype=np.uint32),
        )
    with pytest.raises(ValueError, match="action"):
        stage([[UNIT_COMMAND_ACTION, N_UNIT_ACTIONS, 0]])
    # PICKUP and PLACE carry their own quantity, never a fixed factor code.
    with pytest.raises(ValueError, match="fixes its quantity"):
        stage([[UNIT_COMMAND_ACTION, int(UnitAction.PICKUP_WHEAT_1), 0]])
    with pytest.raises(ValueError, match="item"):
        stage([[UNIT_COMMAND_PICKUP, len(PRIVATE_ITEMS), 1]])
    with pytest.raises(ValueError, match="form"):
        stage([[3, 0, 0]])
    with pytest.raises(ValueError, match="claims"):
        stage([[UNIT_COMMAND_ACTION, 0, 0]], counts=[2])
    with pytest.raises(ValueError, match="unit counts claim"):
        stage([[UNIT_COMMAND_ACTION, 0, 0]] * 2, counts=[1])
    bad_orders = no_orders.copy()
    bad_orders[0, 3] = (N_MARKET_KINDS, 1)
    with pytest.raises(ValueError, match="kind"):
        stage([[UNIT_COMMAND_ACTION, 0, 0]], orders=bad_orders)
    bad_orders[0, 3] = (int(MarketKind.SELL_WHEAT), 0)
    with pytest.raises(ValueError, match="quantity 0"):
        stage([[UNIT_COMMAND_ACTION, 0, 0]], orders=bad_orders)
    # A refused call stages nothing, so a clean retry succeeds.
    _staged(environment, {0: _PASS})
    with pytest.raises(ValueError, match="already staged"):
        _staged(environment, {0: _PASS})


def test_unit_command_forms_mirror_the_native_binding() -> None:
    native = load_native()
    assert (
        native.UNIT_COMMAND_ACTION,
        native.UNIT_COMMAND_PICKUP,
        native.UNIT_COMMAND_PLACE,
    ) == (UNIT_COMMAND_ACTION, UNIT_COMMAND_PICKUP, UNIT_COMMAND_PLACE)


def test_submitted_turns_keep_what_the_factored_space_cannot_carry() -> None:
    units, orders = encode_submitted_turn(
        {
            "farmer": ["PICKUP", "COW", 5],
            "hands": [
                ["PLACE", "WHEAT", 3],
                ["PLANT", "WHEAT"],
                ["PICKUP", "WHEAT"],
                ["PLACE", "GOOSE", -2],
                ["FLY"],
                "PASS",
            ],
            "market": [
                ["SELL", "WHEAT", 0],
                ["SELL", "WHEAT", 250],
                ["SELL", "BREAD", 3],
                ["BUY_SEED", "WHEAT", "2"],
                ["BUY_LAND", "ignored"],
                [],
                ["HIRE"],
            ],
        }
    )
    cow = PRIVATE_ITEMS.index("COW")
    np.testing.assert_array_equal(
        units,
        [
            [UNIT_COMMAND_PICKUP, cow, 5],
            [UNIT_COMMAND_PLACE, 0, 3],
            [UNIT_COMMAND_ACTION, int(UnitAction.PLANT_WHEAT), 0],
            [UNIT_COMMAND_PICKUP, 0, 1],
            [UNIT_COMMAND_PLACE, PRIVATE_ITEMS.index("GOOSE"), 0],
            [UNIT_COMMAND_ACTION, int(UnitAction.PASS), 0],
            [UNIT_COMMAND_ACTION, int(UnitAction.PASS), 0],
        ],
    )
    # Unread orders stay holes in their own slots, so later orders keep the
    # opponent orders they share a quote with.
    expected = np.zeros((MAX_MARKET_ORDERS, 2), dtype=np.uint32)
    expected[1] = (int(MarketKind.SELL_WHEAT), 250)
    expected[3] = (int(MarketKind.BUY_SEED_WHEAT), 2)
    expected[4] = (int(MarketKind.BUY_LAND), 0)
    expected[6] = (int(MarketKind.HIRE), 0)
    np.testing.assert_array_equal(orders, expected)

    # Only the first maxMarketOrdersPerTurn orders are read; non-list parts
    # are the interpreter's defaults.
    _, orders = encode_submitted_turn({"market": [["HIRE"]] * 12})
    assert (orders[:, 0] == int(MarketKind.HIRE)).all()
    units, orders = encode_submitted_turn({"farmer": "NORTH", "hands": "x", "market": "y"})
    np.testing.assert_array_equal(units, [[UNIT_COMMAND_ACTION, int(UnitAction.PASS), 0]])
    assert not orders.any()


@pytest.mark.parametrize(
    "action",
    [
        {"farmer": ["PICKUP", "WHEAT", "many"]},
        {"hands": [["PLACE", "EGG", None]]},
        {"farmer": [["NORTH"]]},
        {"farmer": ["PLANT", ["WHEAT"]]},
        {"market": [["SELL", ["WHEAT"], 1]]},
        {"market": [["SELL", "WHEAT", float("inf")]]},
    ],
)
def test_arguments_the_interpreter_raises_on_are_refused(action) -> None:
    with pytest.raises(SubmittedActionError):
        encode_submitted_turn(action)


def _parity_oracle():
    path = Path(__file__).parents[1] / "rust" / "kagg_env" / "tests" / "parity_oracle.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_parity_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FUZZ_QUANTITIES = (-1, 0, 1, 2, 3, 4, 5, 7, 16, 40, 101, 250, "3", 2.5)
_FUZZ_OPERATIONS = (
    "PASS",
    "DROP",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
)


def _fuzz_action(rng: random.Random, observation: Any) -> dict:
    """A turn full of what the factored space cannot express, plus ordinary play.

    Its money goes round trips through wheat and fertilizer, whose quotes both
    seats move, so which of their orders share a slot decides who pays or
    earns what. Seeds, animals and hands, which never sell back, it buys only
    while it has money to spare, and only wheat seeds by the hundred.
    """
    farm = observation.farms[observation.player]
    spare = farm.money > 1_200

    def quantity() -> list:
        return [] if rng.random() < 0.1 else [rng.choice(_FUZZ_QUANTITIES)]

    def unit() -> Any:
        roll = rng.random()
        if roll < 0.35:
            return [rng.choice(("NORTH", "SOUTH", "EAST", "WEST"))]
        if roll < 0.5:
            return ["PICKUP", rng.choice(PRIVATE_ITEMS), *quantity()]
        if roll < 0.65:
            return ["PLACE", rng.choice(PRIVATE_ITEMS), *quantity()]
        if roll < 0.75:
            # Mostly one crop, so turns regularly ask for more seeds than held.
            return ["PLANT", rng.choice(("WHEAT", "WHEAT", "CARROT", "BREAD"))]
        if roll < 0.95:
            return [rng.choice(_FUZZ_OPERATIONS)]
        return rng.choice((["FLY"], [], "PASS", ["PICKUP"], ["PLACE"]))

    def order() -> Any:
        roll = rng.random()
        if roll < 0.15:
            return rng.choice(([], ["SELL", "WHEAT"], ["SELL", "BREAD", 3], "SELL"))
        if roll < 0.3 and spare:
            if len(farm.hands) < 2 and rng.random() < 0.5:
                return ["HIRE"]
            if rng.random() < 0.5:
                # Cheap enough to fill past the factored space's 100.
                return ["BUY_SEED", "WHEAT", rng.choice(_FUZZ_QUANTITIES)]
            operation, items = rng.choice((("BUY_SEED", CROPS), ("BUY_ANIMAL", ANIMALS)))
            return [operation, rng.choice(items), rng.randrange(1, 4)]
        operation = rng.choice(("SELL", "BUY_PRODUCT") if farm.money > 800 else ("SELL",))
        return [operation, rng.choice(("WHEAT", "FERTILIZER")), rng.choice(_FUZZ_QUANTITIES)]

    return {
        "farmer": unit(),
        # Sometimes more hand commands than hands: they still count toward PLANT demand.
        "hands": [unit() for _ in range(len(farm.hands) + rng.randrange(2))],
        "market": [order() for _ in range(rng.randrange(13))],
    }


@pytest.mark.parametrize("seed", [20260929, 20260930, 20260931, 20260932])
def test_submitted_turns_match_the_official_interpreter_for_a_full_episode(seed) -> None:
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

    oracle = _parity_oracle()
    official = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    official.reset(2)
    environment = load_native().BatchEnv(np.asarray([seed], dtype=np.uint64))
    codes = np.full(2, EXTERNAL_AGENT_CODE, dtype=np.uint8)
    rng = random.Random(seed)
    transitions = 0
    while not official.done:
        observations = [state.observation for state in official.state]
        # The starter half the time keeps seat 1's farm stocked; the other half
        # both queues are fuzz, pairing their holes and oversized orders.
        actions = [
            _fuzz_action(rng, observations[0]),
            starter_agent(observations[1])
            if rng.random() < 0.5
            else _fuzz_action(rng, observations[1]),
        ]
        _staged(environment, dict(enumerate(actions)))
        _step(environment, codes)
        official.step(actions)
        transitions += 1
        difference = oracle._first_difference(
            oracle._official_snapshot(official), json.loads(environment.snapshot_json(0))
        )
        assert difference is None, f"transition {transitions}: {difference}"
    assert transitions == HORIZON


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
    assert statistics["action_errors"] == 0
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
