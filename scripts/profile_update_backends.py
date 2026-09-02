"""Measure the update phase across Inductor compile modes, with peak memory.

The update phase is the larger half of an iteration, and it is the one phase
that never got the treatment collection did: `_cached_update_callable` compiles
with Inductor's DEFAULT mode, so the forward+backward keeps every kernel launch
and every unfused GEMM epilogue. Its docstring rejects CUDA graphs outright, on
the argument that graph-capturing forward+backward "would permanently pin every
minibatch's activations in private pools" -- an assertion with no number behind
it, and the identical reasoning that turned out to be wrong for collection,
where the shipped graph backend was slower than not compiling at all.

So this measures both halves of that claim on the original benchmark schedule:
wall clock AND peak allocated bytes, over the real `_actor_minibatch_terms` and
`_critic_minibatch_loss` including their backwards, the gradient clip, and the
fused optimizer step.

The measured baseline has 73 actor minibatches and 292 critic minibatches per
iteration (epochs=1, critic_epochs=4, minibatch_size=2048), alternating so the
two compiled graphs interleave exactly as they do in `update_ppo`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from kaggriculture.actions import N_QUANTITIES
from kaggriculture.model import (
    BOARD_CHANNELS,
    BOARD_SIZE,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    N_MARKET_KINDS,
    N_UNIT_ACTIONS,
    UNIT_FEATURES,
    DistributionalCritic,
    FarmActor,
)
from kaggriculture.ppo import (
    UPDATE_COMPILE_MODES,
    PpoConfig,
    _actor_minibatch_terms,
    _critic_minibatch_loss,
    _optimizer_step,
)


def _minibatch(rows: int, device: torch.device) -> dict[str, Any]:
    """One production-shaped minibatch, resident and reused across repeats."""
    generator = torch.Generator(device=device.type).manual_seed(20260817)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=device, generator=generator)

    unit_masks = rand(rows, MAX_UNITS, N_UNIT_ACTIONS) > 0.2
    kind_masks = rand(rows, MAX_MARKET_ORDERS, N_MARKET_KINDS) > 0.2
    quantity_masks = rand(rows, MAX_MARKET_ORDERS, N_QUANTITIES) > 0.2
    # Every categorical needs at least one legal action or the log-softmax over
    # its masked logits is -inf and the surrogate is not finite.
    unit_masks[..., 0] = True
    kind_masks[..., 0] = True
    quantity_masks[..., 0] = True
    return {
        "unit_actions": torch.zeros(rows, MAX_UNITS, dtype=torch.long, device=device),
        "market_kinds": torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.long, device=device),
        "market_quantities": torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.long, device=device),
        "unit_masks": unit_masks,
        "kind_masks": kind_masks,
        "quantity_masks": quantity_masks,
        "unit_active": (rand(rows, MAX_UNITS) > 0.3).float(),
        "kind_active": (rand(rows, MAX_MARKET_ORDERS) > 0.5).float(),
        "quantity_active": (rand(rows, MAX_MARKET_ORDERS) > 0.5).float(),
        "old_unit": torch.full((rows, MAX_UNITS), -2.0, device=device),
        "old_kind": torch.full((rows, MAX_MARKET_ORDERS), -1.5, device=device),
        "old_quantity": torch.full((rows, MAX_MARKET_ORDERS), -1.5, device=device),
        "advantages": torch.randn(rows, device=device, generator=generator),
        "value_targets": torch.randn(rows, device=device, generator=generator) * 0.3,
        "board": torch.randn(
            rows, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device, generator=generator
        ),
        "global_features": torch.randn(rows, GLOBAL_FEATURES, device=device, generator=generator),
        "units": rand(rows, MAX_UNITS, UNIT_FEATURES),
        "unit_positions": torch.randint(
            0, BOARD_SIZE, (rows, MAX_UNITS, 2), device=device, generator=generator
        ),
        "critic_features": torch.randn(rows, CRITIC_FEATURES, device=device, generator=generator),
    }


def _compiled(function: Any, mode: str) -> Any:
    if mode == "eager":
        return function
    return torch.compile(function, mode=mode, fullgraph=True, dynamic=False)


def _run_mode(
    mode: str, batch: dict[str, Any], config: PpoConfig, device: torch.device, args
) -> dict[str, float]:
    """Time one full update phase's minibatch schedule under one compile mode."""
    torch._dynamo.reset()
    torch.manual_seed(4242)
    actor = FarmActor().to(device)
    critic = DistributionalCritic().to(device)
    actor.train()
    critic.train()
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.actor_learning_rate, fused=True
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=config.critic_learning_rate, fused=True
    )
    actor_terms = _compiled(_actor_minibatch_terms, mode)
    critic_loss_fn = _compiled(_critic_minibatch_loss, mode)
    autocast = config.use_bfloat16

    actor_args = (batch["board"], batch["global_features"], batch["units"], batch["unit_positions"])
    critic_args = (batch["board"], batch["critic_features"])

    def one_minibatch(run_actor: bool) -> None:
        if run_actor:
            actor_optimizer.zero_grad(set_to_none=True)
            policy_sum, _entropy_sum, _kl_sum, _clipped = actor_terms(
                actor,
                batch["unit_actions"],
                batch["market_kinds"],
                batch["market_quantities"],
                batch["unit_masks"],
                batch["kind_masks"],
                batch["quantity_masks"],
                batch["unit_active"],
                batch["kind_active"],
                batch["quantity_active"],
                batch["old_unit"],
                batch["old_kind"],
                batch["old_quantity"],
                batch["advantages"],
                config.clip_low,
                config.clip_high,
                autocast,
                *actor_args,
            )
            (-policy_sum / 2048).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_gradient_norm)
            _optimizer_step(actor_optimizer, config.actor_learning_rate, config.lr_warmup_steps)
        critic_optimizer.zero_grad(set_to_none=True)
        value_loss, _predicted = critic_loss_fn(
            critic, batch["value_targets"], autocast, *critic_args
        )
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_gradient_norm)
        _optimizer_step(critic_optimizer, config.critic_learning_rate, config.lr_warmup_steps)

    # Warm up every graph variant that the schedule will replay, so compilation
    # never lands inside a measured repeat.
    compile_started = time.perf_counter()
    one_minibatch(True)
    one_minibatch(False)
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - compile_started

    samples: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for index in range(args.critic_minibatches):
            one_minibatch(index < args.actor_minibatches)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return {
        "update_seconds_median": statistics.median(samples),
        "update_seconds_min": min(samples),
        "compile_seconds": compile_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--actor-minibatches", type=int, default=73)
    parser.add_argument("--critic-minibatches", type=int, default=292)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default=",".join(UPDATE_COMPILE_MODES))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the update phase is only meaningful on the accelerator")
    torch.set_float32_matmul_precision("high")
    config = PpoConfig(epochs=1, critic_epochs=4, minibatch_size=args.minibatch_size)
    batch = _minibatch(args.minibatch_size, device)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "minibatch_size": args.minibatch_size,
        "actor_minibatches": args.actor_minibatches,
        "critic_minibatches": args.critic_minibatches,
        "use_bfloat16": config.use_bfloat16,
        "modes": {},
    }
    for mode in args.modes.split(","):
        if mode not in UPDATE_COMPILE_MODES:
            raise ValueError(f"unknown update compile mode {mode!r}")
        measured = _run_mode(mode, batch, config, device, args)
        report["modes"][mode] = measured
        print(
            f"{mode:28s} update={measured['update_seconds_median']:7.3f} s"
            f"  peak_alloc={measured['peak_allocated_gib']:6.2f} GiB"
            f"  peak_resv={measured['peak_reserved_gib']:6.2f} GiB"
            f"  compile={measured['compile_seconds']:7.2f} s",
            flush=True,
        )

    baseline = report["modes"].get("default", {}).get("update_seconds_median")
    if baseline:
        print("\nspeedup against today's default-mode Inductor:")
        for mode, measured in report["modes"].items():
            print(f"  {mode:28s} {baseline / measured['update_seconds_median']:5.3f}x")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
