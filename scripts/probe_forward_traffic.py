#!/usr/bin/env python3
"""Account every byte a structured forward moves, by operator and by call site.

The production model is tiny and narrow -- 80 channels, 4 heads, 8 core layers --
so its arithmetic intensity is far below what the device wants. Measuring that
claim needs bytes, not FLOPs, and bytes are what a wall-clock profile hides: a
kernel that reads 200 MB and writes 200 MB looks the same in a trace as one that
reads 20 MB, and only the first is worth attacking.

This runs the forward under `FakeTensorMode`, so no memory is allocated and no
device is touched -- it can run beside a training job without contending for
anything. A dispatch interceptor sums the footprint of every operator's inputs
and outputs, and module hooks attribute each operator to the innermost named
call site, which is what turns "attention is 19% of the time" into "the opponent
farm's blocks are 19% of the bytes".

Two numbers make the result actionable. The compute floor is the forward's
multiply-accumulates at the device's dense BF16 rate; the traffic floor is its
bytes at the device's measured bandwidth. Their ratio is how far from
compute-bound the model is, and therefore whether a faster kernel or fewer bytes
is the correct next move.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten

from kaggriculture.ppo import _critic_batch_args, actor_forward_args
from kaggriculture.production import production_model_config
from kaggriculture.registry import STRUCTURED
from kaggriculture.rollout import _state_field_specs
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredCritic
from kaggriculture.tokens import MAX_UNITS

#: Call sites worth separating. Each is a module path prefix inside the trunk;
#: an operator is charged to the longest prefix that is currently on the stack,
#: so `trunk.farm_local.0.attention` bills to `farm_local` rather than to the
#: whole trunk. Everything outside them lands in `other`.
_CALL_SITES = (
    "trunk.tiles",
    "trunk.farm_local",
    "trunk.opponent_summary",
    "trunk.units",
    "trunk.economy",
    "trunk.latent_read",
    "trunk.core",
    "unit_decoder",
    "unit_local_decoder",
    "market_decoder",
)

#: Operators that return a view of storage their caller already paid for. They
#: cost no kernel, and charging them would roughly double the measured total --
#: a transposed `[6400, 200, 80]` activation is not re-read because its strides
#: changed. Kept as an explicit list because `FakeTensor` does not always
#: populate `_base`, so aliasing cannot be detected structurally.
_VIEWS = (
    "view",
    "_unsafe_view",
    "reshape",
    "permute",
    "transpose",
    "expand",
    "unbind",
    "squeeze",
    "unsqueeze",
    "detach",
    "alias",
    "as_strided",
    "slice",
    "split",
    "narrow",
    "select.int",
    "t.default",
    "contiguous",
)

#: Bytes an operator moves that are not worth separating one by one. The buckets
#: are chosen so each names a distinct fix: casts are removed by keeping a dtype,
#: reductions by fusing, matmuls by shrinking a dimension.
_CATEGORIES = {
    "cast": ("_to_copy", "copy_"),
    "matmul": ("mm.", "bmm", "addmm", "matmul", "linear", "_scaled_dot_product"),
    "norm": ("rms_norm", "layer_norm", "mean.dim", "var_mean"),
    "gather": ("embedding", "index_select", "index.", "gather"),
}


def _category(name: str) -> str:
    for category, needles in _CATEGORIES.items():
        if any(needle in name for needle in needles):
            return category
    return "pointwise"


def _footprint(values) -> int:
    flat, _ = tree_flatten(values)
    return sum(
        value.numel() * value.element_size() for value in flat if isinstance(value, torch.Tensor)
    )


def _multiply_accumulates(name: str, arguments, result) -> int:
    """Fused multiply-accumulates for the operators that have any."""
    flat, _ = tree_flatten(arguments)
    tensors = [value for value in flat if isinstance(value, torch.Tensor)]
    if "addmm" in name and len(tensors) >= 3:
        left, right = tensors[1], tensors[2]
    elif ("mm" in name or "matmul" in name) and len(tensors) >= 2:
        left, right = tensors[0], tensors[1]
    else:
        return 0
    if left.ndim < 2 or right.ndim < 2:
        return 0
    batch = int(np.prod(left.shape[:-2])) if left.ndim > 2 else 1
    return batch * left.shape[-2] * left.shape[-1] * right.shape[-1]


def _moves_bytes(name: str, result) -> bool:
    """Whether an operator costs a kernel that reads and writes memory.

    Two things masquerade as traffic. Metadata queries such as `prim.device`
    dispatch like operators and touch nothing, and views hand back storage the
    caller already paid for.
    """
    if not name.startswith("aten."):
        return False
    operator = name.removeprefix("aten.")
    if any(operator.startswith(view) for view in _VIEWS):
        return False
    flat, _ = tree_flatten(result)
    return any(isinstance(value, torch.Tensor) for value in flat)


class _TrafficCounter(TorchDispatchMode):
    def __init__(self, call_site: list[str]) -> None:
        super().__init__()
        self.call_site = call_site
        self.bytes_by_category: dict[str, int] = defaultdict(int)
        self.bytes_by_site: dict[str, int] = defaultdict(int)
        self.bytes_by_operator: dict[str, int] = defaultdict(int)
        self.multiply_accumulates = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        name = str(func)
        self.multiply_accumulates += _multiply_accumulates(name, args, result)
        if _moves_bytes(name, result):
            moved = _footprint(args) + _footprint(kwargs) + _footprint(result)
            self.bytes_by_category[_category(name)] += moved
            self.bytes_by_site[self.call_site[-1] if self.call_site else "other"] += moved
            self.bytes_by_operator[name] += moved
        return result


def _attach_site_hooks(model: torch.nn.Module, stack: list[str]) -> None:
    """Maintain a call-site stack across the forward.

    Both hooks must return `None`. A forward hook that returns a value replaces
    the module's output, so `lambda ...: stack.pop()` silently substitutes the
    popped string for the module's tensor and the failure surfaces many frames
    later as an attribute error on `str`.
    """

    def push(site: str):
        def hook(_module, _inputs) -> None:
            stack.append(site)

        return hook

    def pop(_module, _inputs, _output) -> None:
        stack.pop()

    for name, module in model.named_modules():
        # `trunk.farm_local` and `trunk.core` are `ModuleList`s, which never run
        # a forward, so an exact-name match would silently drop the two largest
        # call sites. Tagging descendants instead charges `trunk.farm_local.1
        # .ffn` to `farm_local`, and the innermost tag on the stack still wins
        # for anything nested inside a second listed site.
        site = next(
            (site for site in _CALL_SITES if name == site or name.startswith(f"{site}.")), None
        )
        if site is not None:
            module.register_forward_pre_hook(push(site))
            module.register_forward_hook(pop)


def _states(rows: int) -> dict[str, np.ndarray]:
    """One batch of zeroed states in the exact layout the update stages.

    Shapes and dtypes come from the rollout's own field table rather than from
    constants restated here, because a probe that measures a layout the trainer
    does not use measures nothing.
    """
    states = {
        name: np.zeros((rows, *shape), dtype=dtype)
        for name, (shape, dtype) in _state_field_specs(STRUCTURED).items()
    }
    states["unit_active"] = np.ones((rows, MAX_UNITS), dtype=np.bool_)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6400, help="states in one minibatch forward")
    parser.add_argument("--model", choices=("actor", "critic"), default="actor")
    parser.add_argument("--bandwidth-gbps", type=float, default=1792.0, help="device HBM bandwidth")
    parser.add_argument(
        "--bf16-tflops", type=float, default=209.5, help="device dense BF16 matmul rate"
    )
    # Autocast is deliberately not a mode here. What it does to traffic is
    # exactly the question: it keeps fp32 parameters and inserts a cast per
    # consumer, so its cost is the gap between these two arms plus whatever it
    # forces back into fp32 (`rms_norm`, notably).
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    # CUDA is the default because the dispatcher picks different kernels for it:
    # on CPU, SDPA falls back to the math decomposition, which materializes the
    # whole score matrix and reports traffic no fused kernel ever performs. This
    # allocates nothing -- `FakeTensorMode` intercepts every operator before it
    # reaches an allocator -- so it is safe to run beside a training job.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=argparse.FileType("w"))
    arguments = parser.parse_args()

    config = StructuredConfig(**production_model_config())
    dtype = getattr(torch, arguments.dtype)
    # Built at the target dtype rather than converted into it: under
    # `FakeTensorMode` a later `Module.to` cannot swap parameter storage.
    torch.set_default_dtype(dtype)
    with FakeTensorMode(allow_non_fake_inputs=True):
        device = torch.device(arguments.device)
        # Constructed on the device, not moved to it: under `FakeTensorMode`
        # parameter storage cannot be swapped after the fact.
        with device:
            model = (
                StructuredActor(config) if arguments.model == "actor" else StructuredCritic(config)
            )
        model.eval()
        stack: list[str] = []
        _attach_site_hooks(model, stack)
        states = _states(arguments.rows)
        actor_args = actor_forward_args(STRUCTURED, states, device)
        if arguments.model == "actor":
            forward_args: tuple = actor_args
        else:
            forward_args = _critic_batch_args(
                STRUCTURED,
                {name: torch.from_numpy(value).to(device=device) for name, value in states.items()},
                slice(None),
                actor_args=actor_args,
            )
        flat, spec = tree_flatten(forward_args)
        forward_args = tree_unflatten(
            [
                value.to(dtype)
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value
                for value in flat
            ],
            spec,
        )
        counter = _TrafficCounter(stack)
        with counter, torch.no_grad():
            model(*forward_args)

    total = sum(counter.bytes_by_category.values())
    traffic_floor_ms = total / (arguments.bandwidth_gbps * 1e9) * 1e3
    compute_floor_ms = 2 * counter.multiply_accumulates / (arguments.bf16_tflops * 1e12) * 1e3
    report = {
        "model": arguments.model,
        "rows": arguments.rows,
        "dtype": arguments.dtype,
        "device": arguments.device,
        "gflop": 2 * counter.multiply_accumulates / 1e9,
        "traffic_gib": total / 2**30,
        "compute_floor_ms": compute_floor_ms,
        "traffic_floor_ms": traffic_floor_ms,
        "memory_bound_ratio": traffic_floor_ms / compute_floor_ms if compute_floor_ms else None,
        "bytes_by_category": {
            name: value / 2**30
            for name, value in sorted(counter.bytes_by_category.items(), key=lambda item: -item[1])
        },
        "bytes_by_call_site": {
            name: value / 2**30
            for name, value in sorted(counter.bytes_by_site.items(), key=lambda item: -item[1])
        },
        "top_operators": {
            name: value / 2**30
            for name, value in sorted(counter.bytes_by_operator.items(), key=lambda item: -item[1])[
                :12
            ]
        },
    }
    text = json.dumps(report, indent=1)
    print(text)
    if arguments.report is not None:
        arguments.report.write(text + "\n")


if __name__ == "__main__":
    main()
