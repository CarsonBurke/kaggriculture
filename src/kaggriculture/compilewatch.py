"""Account for every `torch.compile` entry a run pays for, and refuse late ones.

Update minibatch and rollout row counts are fixed by configuration. Frozen
league execution additionally specializes on the assigned neural lane count
and padded width. The collector compiles only the layout a wave actually uses,
before opening its captured step loop; its persistent ensemble caches reuse
previously seen layouts instead of precompiling an unvisited shape family.
The update still warms the released actor's forward and backward while the
actor is frozen (`ppo.py::_warm_actor_update_graphs`).

First-use league layouts can therefore compile after wave one. The caller
withholds changed-layout and warmup-transition waves from the settled-wave
contract; this module distinguishes the events it records:

* A **recompile** is a guard failure -- a frame already compiled for one set of
  input properties met inputs that violate its guards, so something varies that
  was supposed to be constant. `CompilationMetrics.cache_size` is the number of
  entries that frame already had, so it is positive exactly here, and Dynamo's
  `recompiles` logging artifact carries the reason, which is why that artifact
  is enabled and captured rather than reconstructed.
* A **late first compile** (`cache_size == 0`) is a previously unseen frame or
  shape. This is expected when the selected league layout changes, but a fault
  when the caller holds the wave to the settled-shape contract.

Both are only faults once a wave can be held to the contract: the caller
decides that boundary, and passes only settled waves to `check`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch._dynamo.utils import get_compilation_metrics, set_compilation_metrics_limit

__all__ = ["CompileDriftError", "CompileEvent", "CompileWatch"]

#: Loggers Dynamo writes its recompilation reasons to.
_RECOMPILE_LOGGERS = ("torch._dynamo.guards", "torch._dynamo.convert_frame")

#: Dynamo keeps compilation metrics in a bounded deque and drops the oldest
#: entries. The default 64 is smaller than one cold iteration's compile count,
#: and a dropped entry is an unaccounted compile, so the buffer is widened to
#: hold a whole run's worth.
_METRICS_LIMIT = 8192


class CompileDriftError(RuntimeError):
    """A frame compiled after the run's shapes were supposed to be settled."""


@dataclass(frozen=True)
class CompileEvent:
    name: str
    filename: str
    lineno: int
    #: Cache entries the frame already had. Zero is a first compile.
    cache_size: int
    seconds: float

    @property
    def recompile(self) -> bool:
        return self.cache_size > 0

    def describe(self) -> str:
        kind = "recompile" if self.recompile else "compile"
        return (
            f"{kind} {self.name} at {self.filename}:{self.lineno} "
            f"(cache_size={self.cache_size}, {self.seconds:.2f}s)"
        )


class _ReasonHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


class CompileWatch:
    """Per-phase accounting of compiles, with late ones optionally fatal.

    Install once per process. `drain` returns the events since the previous
    call, so a caller reads one wave at a time and can attribute compile cost to
    the wave that paid it instead of discovering it as an unexplained spike.
    """

    def __init__(self) -> None:
        set_compilation_metrics_limit(_METRICS_LIMIT)
        torch._logging.set_logs(recompiles=True)
        self._handler = _ReasonHandler()
        for name in _RECOMPILE_LOGGERS:
            logging.getLogger(name).addHandler(self._handler)
        self._cursor = len(get_compilation_metrics())

    def drain(self) -> tuple[tuple[CompileEvent, ...], tuple[str, ...]]:
        """Compile events and captured guard-failure reasons since the last call."""
        metrics = get_compilation_metrics()
        fresh = metrics[self._cursor :]
        self._cursor = len(metrics)
        reasons = tuple(self._handler.records)
        self._handler.records.clear()
        events = tuple(
            CompileEvent(
                name=str(entry.co_name),
                filename=str(entry.co_filename),
                lineno=int(entry.co_firstlineno or 0),
                cache_size=int(entry.cache_size or 0),
                seconds=(entry.duration_us or 0) / 1e6,
            )
            for entry in fresh
        )
        return events, reasons

    def check(self, events: tuple[CompileEvent, ...], reasons: tuple[str, ...]) -> None:
        """Raise if a settled wave compiled anything at all.

        The recompiles among the offenders carry Dynamo's guard-failure reasons;
        a late first compile has none to carry, so the frame and the seconds it
        cost are the whole report, and the fix is a warmup that reaches it.
        """
        if not events:
            return
        detail = "; ".join(event.describe() for event in events)
        recompiles = sum(1 for event in events if event.recompile)
        because = " | ".join(reason.replace("\n", " ") for reason in reasons)
        raise CompileDriftError(
            f"{len(events)} compilation(s) ({recompiles} recompile(s)) after the shapes "
            f"were supposed to settle: {detail}. "
            f"Guard failures: {because or 'not captured'}"
        )

    def close(self) -> None:
        for name in _RECOMPILE_LOGGERS:
            logging.getLogger(name).removeHandler(self._handler)
