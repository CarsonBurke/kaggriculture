#!/usr/bin/env python3
"""Does compiling the clone loss change the loss, or the step it produces?

The cost probe timed `_clone_loss` under `torch.compile` and never checked its
value. Inductor is free to reorder floating point, so a 2x speedup is only
shippable if the number it produces is the same number.

A first pass compared the largest gradient element and found 2.9% on a conv
bottleneck weight, which is NOT interpretable on its own: a conv weight gradient
is a sum over batch x spatial (204,800 terms at batch 2048) whose terms largely
cancel at initialization, so reordering a bf16 accumulation can move the result
by a large fraction of a small number. Max-element difference cannot tell that
apart from a real bug.

So this measures the quantities an optimizer actually consumes -- cosine
similarity and relative L2, per tensor and globally -- and it measures them
twice: under the shipped bf16 autocast and again in pure fp32. If the fp32 run
is clean and the bf16 run is not, the cause is bf16 accumulation order, which
eager already has and which the trainer already accepts everywhere else. If fp32
is also dirty, inductor is miscompiling and the flag does not ship.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gradients(actor: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().double().clone()
        for name, parameter in actor.named_parameters()
        if parameter.grad is not None
    }


def _compare(
    reference: dict[str, torch.Tensor], other: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Cosine and relative L2, per tensor and over the concatenated gradient."""
    assert reference.keys() == other.keys()
    worst_cosine, worst_cosine_name = 1.0, ""
    worst_l2, worst_l2_name = 0.0, ""
    for name, value in reference.items():
        a, b = value.flatten(), other[name].flatten()
        denominator = (a.norm() * b.norm()).clamp_min(1e-300)
        cosine = float((a @ b) / denominator)
        relative = float((b - a).norm() / a.norm().clamp_min(1e-300))
        if cosine < worst_cosine:
            worst_cosine, worst_cosine_name = cosine, name
        if relative > worst_l2:
            worst_l2, worst_l2_name = relative, name
    flat_reference = torch.cat([value.flatten() for value in reference.values()])
    flat_other = torch.cat([other[name].flatten() for name in reference])
    return {
        "global_cosine": float(
            (flat_reference @ flat_other)
            / (flat_reference.norm() * flat_other.norm()).clamp_min(1e-300)
        ),
        "global_relative_l2": float(
            (flat_other - flat_reference).norm() / flat_reference.norm().clamp_min(1e-300)
        ),
        "worst_tensor_cosine": worst_cosine,
        "worst_tensor_cosine_parameter": worst_cosine_name,
        "worst_tensor_relative_l2": worst_l2,
        "worst_tensor_relative_l2_parameter": worst_l2_name,
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    device = torch.device("cuda")
    sys.path.insert(0, str(REPO / "src"))
    trainer = _load("kaggriculture_train_bc", REPO / "scripts" / "train_bc.py")
    from kaggriculture.registry import resolve_architecture

    architecture = resolve_architecture("entity-cnn")
    config = architecture.config_class()
    train_split, _, _ = trainer.load_dataset(
        [REPO / "data" / "bc-v16-mirror-512"],
        architecture="entity-cnn",
        holdout_seeds=1,
        seeds_per_dataset=8,
        encode_workers=2,
    )

    torch.manual_seed(0)
    actor = architecture.actor_class(config).to(device)
    batch = 2048
    indices = torch.arange(batch)
    rows = {
        name: trainer._batch_tensor(value, indices).to(device)
        for name, value in train_split.staged.items()
    }
    whole = slice(None)
    actor_args = trainer._actor_batch_args("entity-cnn", rows, whole)
    factors = {
        "unit_actions": trainer._batch_tensor(rows["unit_actions"], whole, torch.long),
        "market_kinds": trainer._batch_tensor(rows["market_kinds"], whole, torch.long),
        "market_quantities": trainer._batch_tensor(rows["market_quantities"], whole, torch.long),
        "unit_masks": rows["unit_masks"],
        "market_kind_masks": rows["market_kind_masks"],
        "market_quantity_masks": rows["market_quantity_masks"],
        "unit_active": rows["unit_active"],
        "market_active": rows["market_active"],
        "market_quantity_active": rows["market_quantity_active"],
    }

    report: dict[str, object] = {"probe": "compile-parity", "batch_size": batch}
    for autocast in (True, False):
        precision = "bf16-autocast" if autocast else "fp32"
        actor.zero_grad(set_to_none=True)
        eager_loss = trainer._clone_loss(actor, actor_args, factors, autocast)
        eager_loss.backward()
        eager_grads = _gradients(actor)
        # A second eager pass on the same inputs: the floor this comparison is
        # measured against. Anything at or below it is not attributable to
        # compilation, because eager does not reproduce it either.
        actor.zero_grad(set_to_none=True)
        eager_value = float(eager_loss.detach())
        again = trainer._clone_loss(actor, actor_args, factors, autocast)
        again.backward()
        cell: dict[str, object] = {
            "eager_loss": eager_value,
            "eager_rerun": _compare(eager_grads, _gradients(actor)),
        }
        for mode in ("default", "max-autotune-no-cudagraphs"):
            compiled = torch.compile(trainer._clone_loss, mode=mode)
            actor.zero_grad(set_to_none=True)
            loss = compiled(actor, actor_args, factors, autocast)
            loss.backward()
            cell[mode] = {
                "loss": float(loss.detach()),
                "loss_relative_difference": abs(float(loss.detach()) - eager_value)
                / max(abs(eager_value), 1e-12),
                **_compare(eager_grads, _gradients(actor)),
            }
        report[precision] = cell
        print(json.dumps({precision: cell}, indent=1, sort_keys=True), flush=True)

    output = REPO / "artifacts" / "probes" / "compile-parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
