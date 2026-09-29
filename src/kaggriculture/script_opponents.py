"""Python agent files as opponents inside the native batched engine.

A scripted Kaggle agent (a `main.py` whose last callable is the policy) cannot
run inside Rust, but it can play a seat of a native `BatchEnv` game: each step
the engine's state is serialized, shaped into the official observation, handed
to the agent, and the agent's action dict is projected back into factor codes
(`project_demonstration`) that the next native step plays under the
interpreter's submitted-dict rules. The agents run in worker processes so their
pure-Python cost overlaps the device forward instead of stalling it.

Kaggle executes every episode's agent source into a fresh namespace, and agents
keep per-episode state in module globals (demand-advance4's routing, sale
reservations and terminal plans). Each game seat here therefore owns its own
namespace, built with Kaggle's last-callable rule and never shared or reused.

Parity with the official engine is near exact rather than exact: the one
projection relabel (`deposit_all_products`: a partial shed deposit becomes a
full one) moves demand-advance4's final banks by about 0.1% at most
(`scripts/evaluate_script_native.py --parity-seeds` measures it against
`kaggle_environments` directly).

Two further departures from the official runner are deliberate and counted.
An agent that raises, or a turn `project_demonstration` cannot express, plays
PASS for that whole turn (the runner would forfeit the seat, or play the
representable part); both tallies reach the journal. The configuration omits
the runner's `__raw_path__` key, which no fielded agent reads.

A worker's request carries every seat it holds in the segment, one state
snapshot each (about 3 KB at the first step, 16 KB at the largest). Past the
socket buffer, about 200 KB or a dozen seats per worker, the host's send waits
for the worker to drain its previous request, and the two segments' agent work
stops overlapping; `wait_seconds` shows it. More workers is the remedy.
"""

from __future__ import annotations

import contextlib
import faulthandler
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS

#: The configuration an official agent receives (kaggle-environments 1.32.7).
#: Agents read it: demand-advance4 refuses its terminal planners unless the
#: board/turn/shed/order values match. The runner never shows agents the
#: episode seed, even when the environment was made with one.
SCRIPT_AGENT_CONFIGURATION: dict[str, Any] = {
    "episodeSteps": 720,
    "seed": None,
    "actTimeout": 1,
    "runTimeout": 1200,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
}
#: The official observation carries the agent's remaining overage budget. The
#: native wave has no clock, so every step reports the untouched allowance.
_REMAINING_OVERAGE_SECONDS = 60
#: How long the host waits on one worker reply before declaring the worker
#: hung. A reply covers every seat the worker holds in a segment, and may queue
#: behind the other segment's request, so this is far above an agent's
#: per-step budget: it catches an agent that never returns, not a slow one.
DEFAULT_REPLY_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ScriptOpponent:
    """One agent file fielded as an opponent, pinned by content digest."""

    name: str
    path: Path
    sha256: str

    @property
    def key(self) -> str:
        """Journal/diagnostic key, disjoint from snapshot and built-in keys."""
        return f"script_{self.name}"

    @classmethod
    def from_path(cls, name: str, path: Path | str) -> ScriptOpponent:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"script opponent does not exist: {resolved}")
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"script opponent name must be alphanumeric/-/_: {name!r}")
        return cls(name, resolved, hashlib.sha256(resolved.read_bytes()).hexdigest())


def parse_script_opponent(spec: str) -> ScriptOpponent:
    """Parse ``NAME=PATH``."""
    name, separator, path = spec.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"script opponent must be NAME=PATH, got {spec!r}")
    return ScriptOpponent.from_path(name.strip(), path.strip())


class Struct(dict):
    """`kaggle_environments.utils.Struct`: a dict whose keys are also attributes.

    Vendored with `structify` so a worker needs neither `kaggle_environments`,
    whose import alone costs ~150 MB of game registrations per process, nor
    anything else outside this module and the projection. Faithful to its
    quirks, including silently dropping an ``items`` key.
    """

    def __init__(self, **entries: Any) -> None:
        entries = {key: value for key, value in entries.items() if key != "items"}
        dict.__init__(self, entries)
        self.__dict__.update(entries)

    def __setattr__(self, attr: str, value: Any) -> None:
        self.__dict__[attr] = value
        self[attr] = value


def structify(value: Any) -> Any:
    """`kaggle_environments.utils.structify`: deep-copy into `Struct`s and lists."""
    if isinstance(value, list):
        return [structify(item) for item in value]
    if isinstance(value, dict):
        return Struct(**{key: structify(item) for key, item in value.items()})
    return value


def shaped_observation(snapshot: dict[str, Any], player: int) -> dict[str, Any]:
    """Shape one native snapshot into the observation an official seat receives."""
    return {
        "player": player,
        "step": snapshot["step"],
        "day": snapshot["day"],
        "hour": snapshot["hour"],
        "farms": snapshot["farms"],
        "private": snapshot["privates"][player],
        "market": snapshot["market"],
        "town": snapshot["town"],
        "remainingOverageTime": _REMAINING_OVERAGE_SECONDS,
    }


def load_script_agent(code: Any, path: Path) -> Callable[..., Any]:
    """Execute agent source as `kaggle_environments.agent.get_last_callable` does.

    An empty namespace, the agent's directory importable while it executes,
    and the last callable it defines is the agent.
    """
    namespace: dict[str, Any] = {}
    sys.path.append(str(path.parent))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
    finally:
        sys.path.remove(str(path.parent))
    callables = [value for value in namespace.values() if callable(value)]
    if not callables:
        raise ValueError(f"agent file defines no callable: {path}")
    return callables[-1]


def _empty_statistics() -> dict[str, float | int]:
    """Zeroed script-seat totals.

    Seats and seat-steps played; steps whose agent raised or whose action could
    not be projected (both then play PASS); the per-step slowest worker's
    compute, summed, and its single slowest reply, the headroom under the reply
    timeout; and the host time spent blocked on the workers.
    """
    return {
        "seats": 0,
        "seat_steps": 0,
        "agent_errors": 0,
        "projection_errors": 0,
        "worker_seconds": 0.0,
        "slowest_reply_seconds": 0.0,
        "wait_seconds": 0.0,
    }


_PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


@dataclass
class _WorkerAgents:
    """A worker's compiled sources, live per-seat agents and prebuilt spares."""

    opponents: tuple[ScriptOpponent, ...]
    codes: list[Any] = field(default_factory=list)
    #: (segment tag, slot) -> (opponent, agent) for every seat in play.
    live: dict[tuple[int, int], tuple[int, Callable[..., Any]]] = field(default_factory=dict)
    spares: dict[int, list[Callable[..., Any]]] = field(default_factory=dict)
    # The most seats of each opponent ever live at once. `end` refills spares
    # to it, so steady-state waves never execute agent source on the hot path.
    demand: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for opponent in self.opponents:
            source = opponent.path.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if digest != opponent.sha256:
                raise ValueError(f"script opponent {opponent.name} changed on disk: {digest}")
            self.codes.append(compile(source, str(opponent.path), "exec"))

    def begin(self, tag: int, seats: Sequence[tuple[int, int]]) -> None:
        for slot, opponent in seats:
            spares = self.spares.get(opponent)
            agent = (
                spares.pop()
                if spares
                else load_script_agent(self.codes[opponent], self.opponents[opponent].path)
            )
            self.live[(tag, slot)] = (opponent, agent)
        in_play: dict[int, int] = {}
        for opponent, _ in self.live.values():
            in_play[opponent] = in_play.get(opponent, 0) + 1
        for opponent, count in in_play.items():
            self.demand[opponent] = max(self.demand.get(opponent, 0), count)

    def end(self, tag: int) -> None:
        for key in [key for key in self.live if key[0] == tag]:
            del self.live[key]

    def refill(self) -> None:
        for opponent, wanted in self.demand.items():
            spares = self.spares.setdefault(opponent, [])
            while len(spares) < wanted:
                spares.append(
                    load_script_agent(self.codes[opponent], self.opponents[opponent].path)
                )


def _call_agent(agent: Callable[..., Any], observation: Any, configuration: Any) -> Any:
    arguments = (observation, configuration)
    code = getattr(agent, "__code__", None)
    if code is not None:
        arguments = arguments[: code.co_argcount]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return agent(*arguments)


def _worker_main(connection: Connection, stack_dump: TextIO) -> None:
    from kaggriculture.demonstrations import DemonstrationError, project_demonstration

    # The host signals a hung worker before killing it; the dump says where
    # the agent was stuck.
    faulthandler.register(signal.SIGUSR1, file=stack_dump, all_threads=False)
    agents = _WorkerAgents(connection.recv())
    connection.send(("ready",))
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        kind = message[0]
        if kind == "close":
            return
        if kind == "begin":
            agents.begin(*message[1:])
            continue
        if kind == "end":
            agents.end(message[1])
            # Idle time between waves pays for the next wave's namespaces.
            agents.refill()
            continue
        if kind != "act":
            raise ValueError(f"unknown script worker message {kind!r}")
        _, tag, requests = message
        started = time.perf_counter()
        count = len(requests)
        units = np.zeros((count, MAX_UNITS), dtype=np.uint8)
        kinds = np.zeros((count, MAX_MARKET_ORDERS), dtype=np.uint8)
        quantities = np.zeros((count, MAX_MARKET_ORDERS), dtype=np.uint8)
        agent_errors = 0
        projection_errors = 0
        for index, (slot, player, snapshot_json) in enumerate(requests):
            observation = shaped_observation(json.loads(snapshot_json), player)
            try:
                # Structified copies, as the runner hands every agent its own:
                # the projection below still reads the untouched observation.
                action = _call_agent(
                    agents.live[(tag, slot)][1],
                    structify(observation),
                    structify(SCRIPT_AGENT_CONFIGURATION),
                )
                if not isinstance(action, dict):
                    raise TypeError(f"agent returned {type(action).__name__}, not a dict")
            except Exception:
                # The official runner would forfeit this seat. A training lane
                # instead plays PASS and reports the count: one bad step must
                # not end a 30-minute run, but it must never go unseen either.
                agent_errors += 1
                action = _PASS_ACTION
            try:
                projected = project_demonstration(observation, action, deposit_all_products=True)
            except DemonstrationError:
                projection_errors += 1
                projected = project_demonstration(observation, _PASS_ACTION)
            units[index] = projected.unit_actions
            kinds[index] = projected.market_kinds
            quantities[index] = projected.market_quantities
        connection.send(
            (
                "acted",
                tag,
                units,
                kinds,
                quantities,
                agent_errors,
                projection_errors,
                time.perf_counter() - started,
            )
        )


class ScriptAgentPool:
    """Persistent worker processes playing script seats for native waves.

    Workers start once and serve every wave. Each is a fresh interpreter
    running this module, never a fork (the trainer holds CUDA state) and never
    a `multiprocessing` spawn, which would re-import the parent's `__main__`
    and with it torch -- about 650 MB per worker. Each wave segment claims
    seats with :meth:`segment`; seats spread round-robin over the workers, and
    a segment's per-step requests are tagged so two interleaved segments can
    share the pool.
    """

    def __init__(
        self,
        opponents: Sequence[ScriptOpponent],
        workers: int,
        *,
        timeout: float = DEFAULT_REPLY_TIMEOUT_SECONDS,
    ) -> None:
        if not opponents:
            raise ValueError("a script agent pool needs at least one opponent")
        if workers < 1:
            raise ValueError("a script agent pool needs at least one worker")
        if not timeout > 0:
            raise ValueError("a script agent pool needs a positive reply timeout")
        names = [opponent.name for opponent in opponents]
        if len(set(names)) != len(names):
            raise ValueError("script opponent names must be distinct")
        self.opponents = tuple(opponents)
        self.timeout = float(timeout)
        self.closed = False
        self._connections: list[Connection] = []
        self._processes: list[subprocess.Popen[bytes]] = []
        self._stack_dumps: list[Any] = []
        # Agents are single-threaded Python; a worker never needs a BLAS pool.
        environment = {
            **os.environ,
            **dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), "1"),
        }
        try:
            for _ in range(workers):
                parent, child = socket.socketpair()
                stack_dump = tempfile.TemporaryFile()  # noqa: SIM115 -- closed by _terminate
                self._stack_dumps.append(stack_dump)
                with child:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            __name__,
                            str(child.fileno()),
                            str(stack_dump.fileno()),
                        ],
                        pass_fds=(child.fileno(), stack_dump.fileno()),
                        env=environment,
                    )
                self._processes.append(process)
                self._connections.append(Connection(parent.detach()))
                self._connections[-1].send(self.opponents)
            for worker in range(workers):
                self._receive_raw(worker, expect="ready")
        except BaseException:
            self._terminate()
            raise
        self._next_tag = 0
        self._next_worker = 0
        # Responses that arrived for another segment while one was waiting.
        self._stash: dict[tuple[int, int], tuple[Any, ...]] = {}
        self._abandoned: set[int] = set()
        self._statistics = _empty_statistics()

    @property
    def workers(self) -> int:
        return len(self._connections)

    def _receive_raw(self, worker: int, *, expect: str) -> tuple[Any, ...]:
        # Any failure below leaves the protocol out of step, so it takes the
        # whole pool down: later calls fail fast instead of reading replies
        # meant for requests that were never answered.
        connection = self._connections[worker]
        try:
            ready = connection.poll(self.timeout)
            message = connection.recv() if ready else None
        except (EOFError, OSError) as error:
            self._terminate()
            raise RuntimeError(
                f"script agent worker {worker} exited; its traceback is on stderr"
            ) from error
        if message is None:
            stack = self._dump_stack(worker)
            # It will not read the close either, so it gets no grace period.
            self._processes[worker].kill()
            self._terminate()
            raise RuntimeError(
                f"script agent worker {worker} sent nothing for {self.timeout:g} s and was "
                f"killed; its stack was:\n{stack}"
            )
        if message[0] != expect:
            self._terminate()
            raise RuntimeError(f"script agent worker sent {message[0]!r}, expected {expect!r}")
        return message

    def _dump_stack(self, worker: int) -> str:
        process = self._processes[worker]
        stack_dump = self._stack_dumps[worker]
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGUSR1)
        # faulthandler writes the dump from its signal handler in many small
        # writes; read once the file has stopped growing, or after a second.
        deadline = time.monotonic() + 1.0
        size, settled_since = 0, time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.01)
            current = os.fstat(stack_dump.fileno()).st_size
            if current != size:
                size, settled_since = current, time.monotonic()
            elif size and time.monotonic() - settled_since >= 0.1:
                break
        stack_dump.seek(0)
        return stack_dump.read().decode(errors="replace") or "(no stack dump)"

    def _receive(self, worker: int, tag: int) -> tuple[Any, ...]:
        stashed = self._stash.pop((worker, tag), None)
        if stashed is not None:
            return stashed
        while True:
            message = self._receive_raw(worker, expect="acted")
            received_tag = message[1]
            if received_tag == tag:
                return message
            if received_tag not in self._abandoned:
                self._stash[(worker, received_tag)] = message

    def segment(
        self,
        environment: Any,
        *,
        game_indices: np.ndarray,
        players: np.ndarray,
        opponents: np.ndarray,
    ) -> ScriptSegment:
        """Claim fresh seats for one BatchEnv's script games."""
        if self.closed:
            raise RuntimeError("the script agent pool is closed")
        tag = self._next_tag
        self._next_tag += 1
        count = len(game_indices)
        workers = (self._next_worker + np.arange(count)) % self.workers
        self._next_worker = (self._next_worker + count) % self.workers
        for worker in range(self.workers):
            seats = [
                (int(slot), int(opponents[slot])) for slot in np.flatnonzero(workers == worker)
            ]
            if seats:
                self._send(worker, ("begin", tag, seats))
        return ScriptSegment(
            self,
            environment,
            tag=tag,
            game_indices=np.asarray(game_indices, dtype=np.int64),
            players=np.asarray(players, dtype=np.int64),
            workers=workers,
        )

    def _send(self, worker: int, message: tuple[Any, ...]) -> None:
        if self.closed:
            raise RuntimeError("the script agent pool is closed")
        try:
            self._connections[worker].send(message)
        except OSError as error:
            self._terminate()
            raise RuntimeError(
                f"script agent worker {worker} exited; its traceback is on stderr"
            ) from error

    def take_statistics(self) -> dict[str, float | int]:
        """Totals over the segments finished since the last call, then reset."""
        statistics, self._statistics = self._statistics, _empty_statistics()
        return statistics

    def close(self) -> None:
        if self.closed:
            return
        for connection in self._connections:
            with contextlib.suppress(OSError):
                connection.send(("close",))
        self._terminate()

    def _terminate(self) -> None:
        if self.closed:
            return
        self.closed = True
        # A closed socket reads as EOF, on which an idle worker exits.
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stack_dump in self._stack_dumps:
            stack_dump.close()

    def __enter__(self) -> ScriptAgentPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class ScriptSegment:
    """The script seats of one native BatchEnv for one wave.

    Call :meth:`request` whenever the environment reaches a new state and
    :meth:`stage` right before the native step that must play those actions;
    everything between the two calls overlaps the agents' own work.
    """

    pool: ScriptAgentPool
    environment: Any
    tag: int
    game_indices: np.ndarray
    players: np.ndarray
    workers: np.ndarray
    rows: np.ndarray = field(init=False)
    pending: bool = False
    finished: bool = False
    steps: int = 0
    agent_errors: int = 0
    projection_errors: int = 0
    #: Summed per-step maximum worker compute, its largest single value, and
    #: host time blocked waiting.
    worker_seconds: float = 0.0
    slowest_reply_seconds: float = 0.0
    wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.rows = self.game_indices * 2 + self.players
        self._groups = [
            (worker, np.flatnonzero(self.workers == worker))
            for worker in range(self.pool.workers)
            if np.any(self.workers == worker)
        ]

    def request(self) -> None:
        """Send every script seat the environment's current state."""
        if self.pending:
            raise RuntimeError("script actions were requested twice without staging")
        snapshots = [self.environment.snapshot_json(int(game), False) for game in self.game_indices]
        # Pending before any send: if a send fails partway, the replies of the
        # workers that did get the request are still discarded on finish.
        self.pending = True
        for worker, slots in self._groups:
            self.pool._send(
                worker,
                (
                    "act",
                    self.tag,
                    [(int(slot), int(self.players[slot]), snapshots[slot]) for slot in slots],
                ),
            )

    def stage(self) -> None:
        """Collect the requested actions and stage them for the next native step."""
        if not self.pending:
            raise RuntimeError("script actions were staged before being requested")
        count = len(self.rows)
        units = np.zeros((count, MAX_UNITS), dtype=np.uint8)
        kinds = np.zeros((count, MAX_MARKET_ORDERS), dtype=np.uint8)
        quantities = np.zeros((count, MAX_MARKET_ORDERS), dtype=np.uint8)
        started = time.perf_counter()
        slowest = 0.0
        for worker, slots in self._groups:
            _, _, worker_units, worker_kinds, worker_quantities, agent, projection, seconds = (
                self.pool._receive(worker, self.tag)
            )
            units[slots] = worker_units
            kinds[slots] = worker_kinds
            quantities[slots] = worker_quantities
            self.agent_errors += int(agent)
            self.projection_errors += int(projection)
            slowest = max(slowest, float(seconds))
        self.wait_seconds += time.perf_counter() - started
        self.worker_seconds += slowest
        self.slowest_reply_seconds = max(self.slowest_reply_seconds, slowest)
        self.pending = False
        self.steps += 1
        self.environment.set_external_actions(self.rows, units, kinds, quantities)

    def finish(self) -> None:
        """Release the seats; an abandoned in-flight request is discarded."""
        if self.finished:
            return
        self.finished = True
        if self.pending:
            self.pool._abandoned.add(self.tag)
            for key in [key for key in self.pool._stash if key[1] == self.tag]:
                del self.pool._stash[key]
        # Best effort: on a dead pool the error that ended the wave is the one
        # worth raising, not a broken pipe from releasing its seats.
        if not self.pool.closed:
            for worker, _ in self._groups:
                with contextlib.suppress(OSError):
                    self.pool._connections[worker].send(("end", self.tag))
        totals = self.pool._statistics
        totals["seats"] += len(self.rows)
        totals["seat_steps"] += len(self.rows) * self.steps
        totals["agent_errors"] += self.agent_errors
        totals["projection_errors"] += self.projection_errors
        totals["worker_seconds"] += self.worker_seconds
        totals["slowest_reply_seconds"] = max(
            totals["slowest_reply_seconds"], self.slowest_reply_seconds
        )
        totals["wait_seconds"] += self.wait_seconds


if __name__ == "__main__":
    # A pool worker: `ScriptAgentPool` passes its end of a socket pair.
    _worker_main(Connection(int(sys.argv[1])), open(int(sys.argv[2]), "w"))  # noqa: SIM115
