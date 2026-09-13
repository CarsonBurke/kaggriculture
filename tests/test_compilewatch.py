from __future__ import annotations

import pytest
import torch

from kaggriculture.compilewatch import CompileDriftError, CompileWatch


@pytest.fixture
def watch():
    instance = CompileWatch()
    try:
        yield instance
    finally:
        instance.close()


def test_a_late_first_compile_is_also_a_fault(watch: CompileWatch) -> None:
    """A settled wave has to compile nothing, not merely avoid guard failures.

    A changed league layout may pay its first compile, but the caller excludes
    that wave from this check. With the layout held fixed, a new frame means a
    warmup missed work and costs the same seconds as a recompile. Reporting
    only guard failures would silently accept that cold work in settled waves.
    """

    @torch.compile(dynamic=False)
    def only_once(x: torch.Tensor) -> torch.Tensor:
        return x.sin().sum()

    only_once(torch.randn(4))
    only_once(torch.randn(4))

    events, reasons = watch.drain()

    assert [event.cache_size for event in events] == [0]
    assert not any(event.recompile for event in events)
    with pytest.raises(CompileDriftError, match="0 recompile"):
        watch.check(events, reasons)


def test_a_wave_that_compiles_nothing_passes(watch: CompileWatch) -> None:
    """The steady state is the common case and must never raise."""
    watch.check((), ())


def test_a_guard_failure_is_reported_with_its_reason(watch: CompileWatch) -> None:
    """The shape that varied has to be named, or the abort is not actionable."""

    @torch.compile(dynamic=False)
    def resized(x: torch.Tensor) -> torch.Tensor:
        return x.cos().sum()

    resized(torch.randn(4))
    resized(torch.randn(8))

    events, reasons = watch.drain()

    assert [event.recompile for event in events] == [False, True]
    with pytest.raises(CompileDriftError, match="size mismatch"):
        watch.check(events, reasons)


def test_draining_twice_does_not_repeat_events(watch: CompileWatch) -> None:
    """Per-wave accounting needs each compile attributed to one wave only."""

    @torch.compile(dynamic=False)
    def drained(x: torch.Tensor) -> torch.Tensor:
        return x.tanh().sum()

    drained(torch.randn(4))
    first, _ = watch.drain()
    second, second_reasons = watch.drain()

    assert len(first) == 1
    assert second == ()
    assert second_reasons == ()
