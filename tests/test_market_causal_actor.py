"""Market-only causal decoder contracts, including recorded-action replay."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from kaggriculture.causal_actor import CausalActor, CausalChoice, CausalConfig, CausalReplay
from kaggriculture.device_ledger import DeviceLedger
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredInputs


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(
                    os.environ.get("KAGG_CAUSAL_CUDA") != "1",
                    reason="GPU contracts run through mlq",
                ),
            ],
        ),
    ],
)
def test_market_only_causal_generation_matches_teacher_replay_and_native_masks(device):
    torch.manual_seed(50231)
    env = load_native().BatchEnv(np.array([50231], dtype=np.uint64))
    # Builtin policy has stocked products and makes a legal sell by this point.
    for _ in range(72):
        actions = env.builtin_actions(np.full(2, 4, dtype=np.uint8))
        env.step_factors(
            *(
                actions[name].reshape(1, 2, -1)
                for name in ("unit_actions", "market_kinds", "market_quantities")
            )
        )
    actions = env.builtin_actions(np.full(2, 4, dtype=np.uint8))
    assert np.any(actions["market_kinds"] >= 13)
    encoded = env.structured()
    index_fields = {"tile_categorical", "unit_categorical", "unit_tile_gather"}
    inputs = StructuredInputs(
        *(
            torch.as_tensor(
                encoded[name], device=device, dtype=torch.int64 if name in index_fields else None
            )
            for name in StructuredInputs._fields
        )
    )
    packed = torch.as_tensor(env.policy_ledger(), device=device)
    actor = CausalActor(
        CausalConfig(parallel_unit_decode=True),
        ledger=DeviceLedger.from_native(device, minimum=-20000, maximum=30000),
    ).to(device)
    rows = packed.shape[0]
    choice = CausalChoice(
        packed,
        torch.as_tensor(actions["unit_actions"], dtype=torch.int64, device=device),
        torch.as_tensor(actions["market_kinds"], dtype=torch.int64, device=device),
        torch.as_tensor(actions["market_quantities"], dtype=torch.int64, device=device),
        torch.rand(rows, 36, device=device),
        torch.ones(rows, device=device),
        torch.tensor([False, True], device=device),
    )
    generated = actor(inputs, choice)
    assert generated.unit_active.any()
    assert generated.market_active.any()
    selling = (generated.market_kinds >= 13) & generated.market_quantity_active
    assert selling.any()
    assert (generated.market_quantity_masks[..., 1:].any(-1) & selling).any()
    replay = CausalReplay(
        packed,
        generated.unit_actions,
        generated.market_kinds,
        generated.market_quantities,
    )
    teacher = actor(inputs, replay)
    for field in (
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "unit_masks",
        "market_kind_masks",
        "market_quantity_masks",
        "unit_active",
        "market_active",
        "market_quantity_active",
    ):
        torch.testing.assert_close(
            getattr(generated, field), getattr(teacher, field), rtol=0, atol=0
        )
    torch.testing.assert_close(
        generated.factor_logprobs, teacher.factor_logprobs, atol=2e-5, rtol=2e-5
    )
    for field in ("unit_logits", "market_kind_logits", "market_quantity_context"):
        torch.testing.assert_close(
            getattr(generated, field), getattr(teacher, field), atol=2e-3, rtol=2e-3
        )
    generated_quantity_logits = actor.quantity_logits(
        generated.market_quantity_context,
        generated.market_kinds,
        generated.market_quantity_masks,
    )
    teacher_quantity_logits = actor.quantity_logits(
        teacher.market_quantity_context,
        teacher.market_kinds,
        teacher.market_quantity_masks,
    )
    torch.testing.assert_close(
        generated_quantity_logits, teacher_quantity_logits, atol=2e-3, rtol=2e-3
    )
    factors = [
        getattr(generated, name).detach().cpu().numpy().astype(np.uint8).reshape(1, 2, -1)
        for name in ("unit_actions", "market_kinds", "market_quantities")
    ]
    oracle = env.factor_masks(*factors)
    for name, expected in oracle.items():
        np.testing.assert_array_equal(getattr(generated, name).detach().cpu().numpy(), expected)
    loss = -teacher.factor_logprobs.sum(1).mean()
    loss.backward()
    assert actor.unit_head[-1].weight.grad is not None
    assert actor.market_kind.weight.grad is not None
    assert actor.market_quantity_context.weight.grad.abs().sum() > 0
    assert actor.market_quantity_value.weight.grad.abs().sum() > 0
