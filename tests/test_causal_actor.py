"""Compiled BF16 causal policy contracts. Run through the shared ML queue."""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch

from kaggriculture.causal_actor import (
    CausalActor,
    CausalChoice,
    CausalConfig,
    CausalReplay,
    causal_choose,
    causal_distribution,
)
from kaggriculture.device_ledger import DeviceLedger
from kaggriculture.model import policy_compile_options
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredInputs

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        os.environ.get("KAGG_CAUSAL_CUDA") != "1", reason="compiled GPU contracts require mlq"
    ),
]


def test_causal_distribution_matches_reference_gradient():
    torch.manual_seed(61904)
    logits = torch.randn(4, 100, device="cuda", requires_grad=True)
    mask = (
        torch.arange(100, device="cuda")[None]
        < torch.tensor([1, 7, 32, 100], device="cuda")[:, None]
    )
    selected = torch.tensor([0, 6, 17, 91], device="cuda")
    active = torch.tensor([True, True, False, True], device="cuda")
    temperatures = torch.tensor([1.0, 0.8, 1.2, 1.5], device="cuda")
    weights = torch.randn(4, 2, device="cuda")

    chosen, entropy = causal_distribution(logits, mask, selected, active, temperatures)
    actual_gradient = torch.autograd.grad(
        (torch.stack((chosen, entropy), -1) * weights).sum(), logits
    )[0]
    reference_logprob = (logits / temperatures[:, None]).masked_fill(~mask, -1e9).log_softmax(-1)
    reference_chosen = torch.where(
        active, reference_logprob.gather(-1, selected[:, None]).squeeze(-1), 0
    )
    reference_entropy = torch.where(
        active, -(reference_logprob.exp() * reference_logprob).sum(-1), 0
    )
    reference_gradient = torch.autograd.grad(
        (torch.stack((reference_chosen, reference_entropy), -1) * weights).sum(), logits
    )[0]
    torch.testing.assert_close(chosen, reference_chosen, rtol=0, atol=0)
    torch.testing.assert_close(entropy, reference_entropy, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-5, atol=1e-6)


def test_causal_choose_matches_reference_for_sampling_and_recorded_actions():
    logits = torch.arange(40, device="cuda", dtype=torch.float32).reshape(4, 10) / 7
    mask = (
        torch.arange(10, device="cuda")[None] < torch.tensor([1, 4, 7, 10], device="cuda")[:, None]
    )
    recorded = torch.tensor([-1, -1, 9, 8], device="cuda")
    uniform = torch.tensor([0.7, 0.35, 0.8, 0.6], device="cuda")
    temperatures = torch.tensor([1.0, 0.8, 1.2, 1.5], device="cuda")
    deterministic = torch.tensor([False, True, False, False], device="cuda")
    actual = causal_choose(logits, mask, recorded, uniform, temperatures, deterministic)

    scaled = (logits / temperatures[:, None]).masked_fill(~mask, -1e9)
    cdf = scaled.softmax(-1).cumsum(-1)
    last = torch.where(mask, torch.arange(mask.shape[-1], device="cuda"), 0).amax(-1)
    sampled = torch.minimum((cdf <= uniform[:, None]).sum(-1), last)
    sampled = torch.where(deterministic, scaled.argmax(-1), sampled)
    proposed = torch.where(recorded >= 0, recorded, sampled).long()
    expected = torch.where(mask.gather(1, proposed[:, None]).squeeze(1), proposed, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shared_mask", [False, True])
def test_causal_custom_ops_vmap_matches_per_lane_and_distribution_gradient(shared_mask):
    torch.manual_seed(61905)
    lanes, rows, choices = 3, 4, 17
    logits = torch.randn(lanes, rows, choices, device="cuda", requires_grad=True)
    mask = (
        torch.arange(choices, device="cuda")[None]
        < torch.tensor([1, 5, 12, 17], device="cuda")[:, None]
    )
    masks = mask if shared_mask else mask.expand(lanes, -1, -1).clone()
    mask_dim = None if shared_mask else 0
    selected = torch.tensor([0, 4, 10, 16], device="cuda").expand(lanes, -1)
    active = torch.tensor([True, False, True, True], device="cuda").expand(lanes, -1)
    temperatures = torch.linspace(0.8, 1.4, lanes * rows, device="cuda").reshape(lanes, rows)
    uniform = torch.rand(lanes, rows, device="cuda")
    deterministic = torch.tensor([False, True, False, True], device="cuda").expand(lanes, -1)
    recorded = torch.tensor([-1, -1, 10, -1], device="cuda").expand(lanes, -1)

    chosen, entropy = torch.vmap(causal_distribution, in_dims=(0, mask_dim, 0, 0, 0))(
        logits, masks, selected, active, temperatures
    )
    expected = [
        causal_distribution(
            logits[index],
            mask if shared_mask else masks[index],
            selected[index],
            active[index],
            temperatures[index],
        )
        for index in range(lanes)
    ]
    torch.testing.assert_close(chosen, torch.stack([part[0] for part in expected]))
    torch.testing.assert_close(entropy, torch.stack([part[1] for part in expected]))
    weights = torch.randn(lanes, rows, 2, device="cuda")
    actual_gradient = torch.autograd.grad(
        (torch.stack((chosen, entropy), -1) * weights).sum(), logits, retain_graph=True
    )[0]
    reference_gradient = torch.autograd.grad(
        (
            torch.stack(
                (
                    torch.stack([part[0] for part in expected]),
                    torch.stack([part[1] for part in expected]),
                ),
                -1,
            )
            * weights
        ).sum(),
        logits,
    )[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-5, atol=1e-6)

    actual_choice = torch.vmap(causal_choose, in_dims=(0, mask_dim, 0, 0, 0, 0))(
        logits.detach(), masks, recorded, uniform, temperatures, deterministic
    )
    expected_choice = torch.stack(
        [
            causal_choose(
                logits[index].detach(),
                mask if shared_mask else masks[index],
                recorded[index],
                uniform[index],
                temperatures[index],
                deterministic[index],
            )
            for index in range(lanes)
        ]
    )
    torch.testing.assert_close(actual_choice, expected_choice, rtol=0, atol=0)


@pytest.fixture(scope="module")
def bundle():
    torch.manual_seed(61903)
    env = load_native().BatchEnv(np.array([1701, 1702], dtype=np.uint64))
    for _ in range(72):
        actions = env.builtin_actions(np.full(4, 4, dtype=np.uint8))
        env.step_factors(
            *(
                actions[name].reshape(2, 2, -1)
                for name in ("unit_actions", "market_kinds", "market_quantities")
            )
        )
    encoded = env.structured()
    index_fields = {"tile_categorical", "unit_categorical", "unit_tile_gather"}
    inputs = StructuredInputs(
        *(
            torch.as_tensor(
                encoded[name], device="cuda", dtype=torch.int64 if name in index_fields else None
            )
            for name in StructuredInputs._fields
        )
    )
    packed = torch.as_tensor(env.policy_ledger(), device="cuda")
    rules = DeviceLedger.from_native("cuda", minimum=-20000, maximum=30000)
    actor = CausalActor(CausalConfig(), ledger=rules).cuda()
    assert copy.deepcopy(actor).device_ledger is rules
    choice = CausalChoice(
        packed,
        torch.full((4, 16), -1, device="cuda", dtype=torch.int64),
        torch.full((4, 10), -1, device="cuda", dtype=torch.int64),
        torch.full((4, 10), -1, device="cuda", dtype=torch.int64),
        torch.rand(4, 36, device="cuda"),
        torch.ones(4, device="cuda"),
        torch.tensor([False, True, False, True], device="cuda"),
    )
    compiled = torch.compile(
        actor, fullgraph=True, dynamic=False, options=policy_compile_options("default")
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = compiled(inputs, choice)
        # Capture-backed production callers must also support stable repeat replay.
        repeated = compiled(inputs, choice)
    for name in ("unit_actions", "market_kinds", "market_quantities", "factor_logprobs"):
        torch.testing.assert_close(
            getattr(generated, name), getattr(repeated, name), rtol=0, atol=0
        )
    return env, actor, compiled, inputs, choice, generated


def test_generated_factors_have_exact_native_masks_and_sampling_density(bundle):
    env, actor, _, _, choice, generated = bundle
    factors = [
        getattr(generated, name).cpu().numpy().astype(np.uint8).reshape(2, 2, -1)
        for name in ("unit_actions", "market_kinds", "market_quantities")
    ]
    oracle = env.factor_masks(*factors)
    for name, expected in oracle.items():
        np.testing.assert_array_equal(getattr(generated, name).cpu().numpy(), expected)
    assert generated.factor_logprobs.isfinite().all()
    assert generated.factor_entropies.isfinite().all()
    # Saved behavior log probabilities are computed from the exact emitted heads.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Compile the standalone quantity readout too: no eager model path.
        qhead = torch.compile(
            actor.quantity_logits,
            fullgraph=True,
            dynamic=False,
            options=policy_compile_options("default"),
        )
        quantities = qhead(generated.market_quantity_context, generated.market_kinds)
    parts = []
    for logits, masks, actions, active in (
        (
            generated.unit_logits,
            generated.unit_masks,
            generated.unit_actions,
            generated.unit_active,
        ),
        (
            generated.market_kind_logits,
            generated.market_kind_masks,
            generated.market_kinds,
            generated.market_active,
        ),
        (
            quantities,
            generated.market_quantity_masks,
            generated.market_quantities,
            generated.market_quantity_active,
        ),
    ):
        lp = (
            (logits.float() / choice.temperatures[:, None, None])
            .masked_fill(~masks, -1e9)
            .log_softmax(-1)
        )
        parts.append(torch.where(active, lp.gather(-1, actions[..., None]).squeeze(-1), 0))
    expected = torch.cat((parts[0], torch.stack(parts[1:], -1).flatten(1)), 1)
    torch.testing.assert_close(generated.factor_logprobs, expected, rtol=0, atol=0.002)


def test_parallel_teacher_force_matches_cached_decoding_and_has_gradients(bundle):
    _, actor, compiled, inputs, choice, generated = bundle
    replay = CausalReplay(
        choice.packed_ledger,
        generated.unit_actions,
        generated.market_kinds,
        generated.market_quantities,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        teacher = compiled(inputs, replay)
        loss = -teacher.factor_logprobs.sum(1).mean()
    for name in (
        "unit_masks",
        "market_kind_masks",
        "market_quantity_masks",
        "unit_active",
        "market_active",
        "market_quantity_active",
    ):
        torch.testing.assert_close(getattr(teacher, name), getattr(generated, name), rtol=0, atol=0)
    difference = (teacher.factor_logprobs.detach() - generated.factor_logprobs).abs()
    print(
        "causal cached/parallel max factor log-prob difference",
        difference.max().item(),
        "mean joint difference",
        (teacher.factor_logprobs.detach().sum(1) - generated.factor_logprobs.sum(1))
        .abs()
        .mean()
        .item(),
    )
    torch.testing.assert_close(
        teacher.factor_logprobs, generated.factor_logprobs, rtol=0, atol=0.004
    )
    loss.backward()
    missing = [name for name, parameter in actor.named_parameters() if parameter.grad is None]
    assert not missing, missing
    assert all(parameter.grad.isfinite().all() for parameter in actor.parameters())
    assert actor.trunk.previous_action.weight.grad.abs().sum() > 0
    assert actor.trunk.ledger[0].weight.grad.abs().sum() > 0


def test_parallel_decoder_cannot_read_future_actions(bundle):
    _, _, compiled, inputs, choice, generated = bundle
    replay = CausalReplay(
        choice.packed_ledger,
        generated.unit_actions,
        generated.market_kinds,
        generated.market_quantities,
    )
    changed = generated.market_quantities.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 100
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original = compiled(inputs, replay)
        alternate = compiled(inputs, replay._replace(market_quantities=changed))
    # Last quantity has no causal successors and cannot alter any neural head.
    for name in ("unit_logits", "market_kind_logits", "market_quantity_context"):
        torch.testing.assert_close(
            getattr(original, name), getattr(alternate, name), rtol=0, atol=0
        )
