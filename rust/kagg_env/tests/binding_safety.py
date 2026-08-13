"""Crash and transaction-safety regressions for the native NumPy binding.

Run from the repository root after a release build:
  PYTHONPATH=.venv/lib/python3.13/site-packages:src .venv/bin/python \
    rust/kagg_env/tests/binding_safety.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

import numpy as np

from kaggriculture.rust_env import load_native

BATCH = 2
ROWS = BATCH * 2
RANK = 3


def sampler_inputs() -> list[np.ndarray]:
    return [
        np.zeros((ROWS, 16, 59), dtype=np.float32),
        np.zeros((ROWS, 10, 22), dtype=np.float32),
        np.zeros((ROWS, 10, RANK), dtype=np.float32),
        np.zeros((1, 22, RANK), dtype=np.float32),
        np.zeros((1, 100, RANK), dtype=np.float32),
        np.zeros((1, 22, 100), dtype=np.float32),
        np.zeros(ROWS, dtype=np.uint16),
        np.zeros((ROWS, 16), dtype=np.float32),
        np.zeros((ROWS, 10), dtype=np.float32),
        np.zeros((ROWS, 10), dtype=np.float32),
        np.zeros(ROWS, dtype=np.bool_),
        np.ones(ROWS, dtype=np.float32),
    ]


def strided_like(array: np.ndarray) -> np.ndarray:
    shape = (*array.shape[:-1], array.shape[-1] * 2)
    result = np.zeros(shape, dtype=array.dtype)[..., ::2]
    assert result.shape == array.shape
    assert not result.flags.c_contiguous
    return result


def step_of(environment: object) -> int:
    return int(json.loads(environment.snapshot_json(0))["step"])


def child_noncontiguous_inputs() -> None:
    native = load_native(release=True)
    seeds = np.arange(BATCH, dtype=np.uint64)
    names = (
        "unit_logits",
        "market_kind_logits",
        "market_quantity_context",
        "quantity_kind_gate",
        "quantity_values",
        "quantity_bias",
        "head_ids",
        "unit_draws",
        "market_kind_draws",
        "market_quantity_draws",
        "deterministic_rows",
        "temperatures",
    )
    for index, name in enumerate(names):
        environment = native.BatchEnv(seeds)
        inputs = sampler_inputs()
        inputs[index] = strided_like(inputs[index])
        before = step_of(environment)
        try:
            environment.sample_and_step_into(*inputs, environment.sample_buffers())
        except ValueError as error:
            assert "C-contiguous" in str(error), (name, error)
        else:
            raise AssertionError(f"strided {name} was accepted")
        assert step_of(environment) == before, name


def assert_output_rejected_without_step(
    mutate: Callable[[dict[str, np.ndarray]], None],
) -> None:
    native = load_native(release=True)
    environment = native.BatchEnv(np.arange(BATCH, dtype=np.uint64))
    output = environment.sample_buffers()
    mutate(output)
    before = step_of(environment)
    try:
        environment.sample_and_step_into(*sampler_inputs(), output)
    except (KeyError, RuntimeError, TypeError, ValueError):
        pass
    else:
        raise AssertionError("malformed output buffers were accepted")
    assert step_of(environment) == before


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--child-noncontiguous":
        child_noncontiguous_inputs()
        return

    child = subprocess.run(
        [sys.executable, __file__, "--child-noncontiguous"],
        check=False,
        capture_output=True,
        text=True,
    )
    if child.returncode != 0:
        raise AssertionError(
            f"strided-input subprocess failed with {child.returncode}:\n"
            f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
        )

    assert_output_rejected_without_step(lambda output: output.pop("potentials"))
    assert_output_rejected_without_step(
        lambda output: output.__setitem__("potentials", np.zeros(BATCH + 1, dtype=np.float32))
    )
    assert_output_rejected_without_step(
        lambda output: output.__setitem__("potentials", np.zeros(BATCH, dtype=np.float64))
    )

    def make_readonly(output: dict[str, np.ndarray]) -> None:
        output["potentials"].flags.writeable = False

    assert_output_rejected_without_step(make_readonly)
    assert_output_rejected_without_step(
        lambda output: output.__setitem__(
            "market_kinds",
            np.zeros((ROWS, 20), dtype=np.uint8)[:, ::2],
        )
    )
    assert_output_rejected_without_step(
        lambda output: output.__setitem__("market_quantities", output["market_kinds"])
    )
    print("binding safety: 12 strided inputs + 6 malformed/aliased outputs rejected pre-step")


if __name__ == "__main__":
    main()
