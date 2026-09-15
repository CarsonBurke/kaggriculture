#!/usr/bin/env python3
"""Time the actor forward and backward with and without the opponent farm, paired.

The whole-iteration benchmark cannot answer this question cleanly, for two
reasons that both disappear here.

It measures two different policies. Turning the opponent farm off changes the
actions, so the arms roll out different games and replay different amounts of
work: the shipped pair differs by 5.7% in replayed active unit rows, which is
itself a large fraction of the effect being measured. Here every arm sees one
identical batch, so a time difference is a speed difference and nothing else.

Its arms also run in separate processes minutes apart. That turned out to
matter more than the within-arm spread suggested: `structured_critic_auxiliary_
seconds` must be invariant to this flag, because `private_columns` forces the
critic's opponent path on regardless, and it still moved 10.9% between two
runs that should have agreed (2.587 s against 2.868 s). The arms here are
interleaved inside one process, in rotating order, so they share clock state,
allocator state, and host load.

The third arm is a null: a second, independently constructed opponent-farm
actor, identical in configuration to the first. Whatever difference it shows
against the first arm is this harness's own noise, so the on/off gap is
credible only to the extent the null gap is smaller. An A/B without an A/A is
how the 10.9% above got mistaken for signal.

What this does not measure: the critic, the environment, and the optimizer, all
of which are unchanged by the flag and none of which get faster. Scale the
result by the actor's share of the update before quoting an iteration number.
"""

from __future__ import annotations

import argparse
import json
import statistics

import numpy as np
import torch
from torch.utils._pytree import tree_flatten

from kaggriculture.ppo import actor_forward_args
from kaggriculture.registry import STRUCTURED
from kaggriculture.rollout import _state_field_specs
from kaggriculture.structured import StructuredActor, StructuredConfig
from kaggriculture.tokens import MAX_UNITS


def _states(rows: int, generator: np.random.Generator) -> dict[str, np.ndarray]:
    """One batch in the exact layout and dtype the update stages.

    Values are random rather than zero because zeros are not a neutral choice
    for a timing run: they make every gate and mask degenerate, and a padded
    attention over an all-zero batch can take a different path than a real one.
    Integer fields are kept inside their observed range by construction, since
    an out-of-range index would fault an embedding lookup rather than slow it.
    """
    states: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in _state_field_specs(STRUCTURED).items():
        if np.issubdtype(dtype, np.floating):
            states[name] = generator.standard_normal((rows, *shape)).astype(dtype)
        elif dtype == np.bool_:
            states[name] = generator.integers(0, 2, (rows, *shape)).astype(dtype)
        else:
            states[name] = generator.integers(0, 2, (rows, *shape)).astype(dtype)
    states["unit_active"] = np.ones((rows, MAX_UNITS), dtype=np.bool_)
    return states


def _objective(outputs) -> torch.Tensor:
    """A scalar that depends on every floating-point output.

    Summing rather than applying the real losses keeps the backward graph the
    same shape as training's while leaving out per-head weighting that would
    differ between arms for reasons unrelated to the trunk.
    """
    flat, _ = tree_flatten(outputs)
    terms = [value.float().sum() for value in flat if isinstance(value, torch.Tensor)]
    return torch.stack(terms).sum()


def _time_step(model, forward_args, dtype: torch.dtype) -> tuple[float, float]:
    """Milliseconds in the forward and in the backward of one step."""
    model.zero_grad(set_to_none=True)
    start = torch.cuda.Event(enable_timing=True)
    middle = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        loss = _objective(model(*forward_args))
    middle.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(middle), middle.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6400, help="states per legacy D80 minibatch")
    parser.add_argument("--repeats", type=int, default=20, help="timed steps per arm")
    parser.add_argument("--warmup", type=int, default=4, help="untimed steps per arm")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--report", type=argparse.FileType("w"))
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark measures device time and needs CUDA")
    device = torch.device("cuda")
    dtype = getattr(torch, arguments.dtype)
    torch.manual_seed(arguments.seed)

    # Historical D80 tile-memory ablation, independent of the production family.
    base = StructuredConfig(
        model_dim=80,
        attention_heads=4,
        attention_kv_heads=2,
        ffn_multiplier=2,
        farm_blocks=2,
        opponent_latents=8,
        latents=32,
        core_layers=8,
        quantity_rank=32,
        input_reinject_layers=(1, 2, 3, 4, 5, 6, 7, 8),
        core_skip_source=3,
        core_skip_target=6,
        mudd_lite=True,
        fuse_market_decoder=True,
        global_modulation=True,
        zero_init_branches=False,
    ).to_dict()
    # The null arm is a second `on` actor. Ordering puts it last so that any
    # monotone drift in machine state over the run counts against the null gap
    # rather than hiding inside it.
    specs = (("on", True), ("off", False), ("on-null", True))
    models = {}
    for name, reads_opponent in specs:
        config = StructuredConfig(**{**base, "actor_opponent_farm": reads_opponent})
        with device:
            model = StructuredActor(config)
        model.train()
        models[name] = (
            model if arguments.no_compile else torch.compile(model, mode=arguments.compile_mode)
        )

    generator = np.random.default_rng(arguments.seed)
    forward_args = actor_forward_args(STRUCTURED, _states(arguments.rows, generator), device)

    for name in models:
        for _ in range(arguments.warmup):
            _time_step(models[name], forward_args, dtype)

    samples: dict[str, list[tuple[float, float]]] = {name: [] for name in models}
    order = list(models)
    for repeat in range(arguments.repeats):
        # Rotate so no arm always runs first in a repeat; a fixed order lets
        # one arm absorb whatever the previous arm left in the caches.
        for name in order[repeat % len(order) :] + order[: repeat % len(order)]:
            samples[name].append(_time_step(models[name], forward_args, dtype))

    def summarize(name: str) -> dict[str, float]:
        forward = [value for value, _ in samples[name]]
        backward = [value for _, value in samples[name]]
        total = [value + other for value, other in samples[name]]
        return {
            "forward_ms_median": statistics.median(forward),
            "backward_ms_median": statistics.median(backward),
            "total_ms_median": statistics.median(total),
            "total_ms_min": min(total),
            "total_ms_spread_fraction": (max(total) - min(total)) / statistics.median(total),
            "parameters": sum(
                parameter.numel()
                for parameter in (
                    models[name]._orig_mod if hasattr(models[name], "_orig_mod") else models[name]
                ).parameters()
            ),
        }

    report = {
        "rows": arguments.rows,
        "repeats": arguments.repeats,
        "dtype": arguments.dtype,
        "compile_mode": None if arguments.no_compile else arguments.compile_mode,
        "device_name": torch.cuda.get_device_name(0),
        "arms": {name: summarize(name) for name in models},
    }
    reference = report["arms"]["on"]["total_ms_median"]
    report["fraction_vs_on"] = {
        name: report["arms"][name]["total_ms_median"] / reference - 1.0 for name in models
    }
    text = json.dumps(report, indent=1)
    print(text)
    if arguments.report is not None:
        arguments.report.write(text + "\n")


if __name__ == "__main__":
    main()
