from __future__ import annotations

import importlib.util
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from kaggriculture.demonstrations import DemonstrationError


def _load_extractor():
    path = Path(__file__).parents[1] / "scripts" / "extract_bc_dataset.py"
    spec = importlib.util.spec_from_file_location("extract_bc_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_failed_projection_cancels_the_seeds_that_have_not_started() -> None:
    """One unrepresentable step must abort the whole extraction promptly.

    The dataset would otherwise be biased by the seeds the ledger silently
    dropped, and an operator waiting on a thousand-seed run needs the
    traceback now rather than after every remaining game has played out. The
    cancellation has to be issued from the collecting thread: asking the pool
    to do it via `shutdown(cancel_futures=True)` does not work, because the
    shutdown that runs while the exception unwinds withdraws the request
    before the pool's manager thread acts on it.

    Driving bare futures rather than a live pool keeps the queued/started
    split exact -- with a real executor a worker races to pick up the next
    seed the instant the first one fails.
    """
    extractor = _load_extractor()
    failed: Future = Future()
    failed.set_exception(DemonstrationError("step 3 seat 0: action is not representable"))
    queued: list[Future] = [Future() for _ in range(5)]
    pending = {failed: 0, **{future: seed for seed, future in enumerate(queued, start=1)}}

    with pytest.raises(DemonstrationError, match="not representable"):
        extractor.collect_extractions(pending, time.perf_counter())

    assert all(future.cancelled() for future in queued)


def test_successful_extraction_returns_every_submitted_seat() -> None:
    extractor = _load_extractor()
    pending: dict[Future, int] = {}
    for seed in range(4):
        future: Future = Future()
        future.set_result([{"seed": seed, "seat": 0}, {"seed": seed, "seat": 1}])
        pending[future] = seed

    episodes = extractor.collect_extractions(pending, time.perf_counter())

    assert sorted((record["seed"], record["seat"]) for record in episodes) == [
        (seed, seat) for seed in range(4) for seat in (0, 1)
    ]


def test_teacher_sits_both_seats_against_a_distinct_opponent() -> None:
    """A clone has to see the farm from both sides, not just seat 0."""
    extractor = _load_extractor()
    assert extractor.teacher_jobs("v16", "starter") == (
        ("v16", "starter", (0,)),
        ("starter", "v16", (1,)),
    )
    assert extractor.teacher_jobs("starter", "starter") == (("starter", "starter", (0, 1)),)
