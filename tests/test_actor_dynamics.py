"""Head-space recurrence, frozen final projections, and successor eligibility."""

import numpy as np
import pytest
import torch
from torch import nn

from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS, MarketKind
from kaggriculture.actor_dynamics import ActorDynamics, actor_horizon_loss, actor_window_loss
from kaggriculture.constants import BOARD_SIZE, MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.latent_dynamics import DecodeHeads, DecodeMasks, _decision_kl
from kaggriculture.structured import (
    StructuredActor,
    StructuredConfig,
    StructuredDecisionBelief,
    StructuredInputs,
)
from kaggriculture.structured_dynamics import (
    PersistenceDynamics,
    ShuffledActionDynamics,
    structured_horizon_plan,
)


class _TensorTransition:
    """Pure tensor recurrence: no actor or predictor model executes on CPU."""

    def __init__(self):
        self.scale = torch.tensor(0.2, requires_grad=True)

    def __call__(self, belief, unit_actions, *context):
        return StructuredDecisionBelief(
            *(
                value + self.scale * (value.sin() + unit_actions[:, :1, None].float())
                for value in belief
            )
        )


def _tensor_case(episodes, steps):
    rows = len(steps)
    generator = torch.Generator().manual_seed(829)
    belief = StructuredDecisionBelief(
        *(
            torch.randn(rows, slots, 4, generator=generator).requires_grad_()
            for slots in (MAX_UNITS, MAX_MARKET_ORDERS)
        )
    )
    inputs = StructuredInputs(
        *(torch.zeros(rows, 1, dtype=torch.long) for _ in StructuredInputs._fields)
    )
    inputs = inputs._replace(unit_active=torch.ones(rows, MAX_UNITS, dtype=torch.bool))
    factors = {
        "episode_index": torch.tensor(episodes),
        "step": torch.tensor(steps),
        "unit_actions": torch.arange(rows).remainder(5).reshape(rows, 1),
        "market_kinds": torch.zeros(rows, 1, dtype=torch.long),
        "market_quantities": torch.zeros(rows, 1, dtype=torch.long),
        "market_active": torch.ones(rows, MAX_MARKET_ORDERS, dtype=torch.bool),
    }
    return belief, inputs, factors


@pytest.mark.parametrize("ancestry", ["complete", "broken", "empty"])
def test_head_dense_compact_window_match_independent_recursive_loss_and_gradients(ancestry):
    episodes = np.repeat(np.arange(2), 3)
    steps = np.tile(np.arange(3), 2)
    if ancestry == "broken":
        # Row two matches row zero's endpoint but cannot repair row one's edge.
        episodes[1] = 7
    elif ancestry == "empty":
        steps[:] = 0
    belief, inputs, factors = _tensor_case(episodes, steps)
    dynamics = _TensorTransition()
    plan = structured_horizon_plan(episodes, steps, 2)
    results = [
        actor_horizon_loss(
            dynamics,
            None,
            belief,
            inputs,
            factors,
            decision_horizon=0,
            latent_horizon=2,
            plan=selection,
        )
        for selection in (None, plan)
    ]
    results.append(
        actor_window_loss(
            dynamics, None, belief, inputs, factors, decision_horizon=0, latent_horizon=2
        )
    )
    per_horizon = [[], []]
    eligible_count = 0
    for start in (0, 3):
        for source in range(start, start + 2):
            predicted = torch.cat(tuple(value[source] for value in belief), dim=0)
            for offset in range(1, start + 3 - source):
                target = source + offset
                if episodes[target] != episodes[source] or steps[target] != steps[source] + offset:
                    break
                action = factors["unit_actions"][target - 1].float()
                predicted = predicted + dynamics.scale * (predicted.sin() + action)
                per_horizon[offset - 1].append(
                    nn.functional.smooth_l1_loss(
                        predicted,
                        torch.cat(tuple(value[target].detach() for value in belief), dim=0),
                    )
                )
                eligible_count += 1
    expected = (
        sum(
            torch.stack(values).mean() if values else sum(value.sum() for value in belief) * 0
            for values in per_horizon
        )
        / 2
    )
    parameters = (*belief, dynamics.scale)
    reference_gradients = torch.autograd.grad(
        expected, parameters, allow_unused=True, retain_graph=True
    )
    for terms in results:
        torch.testing.assert_close(terms.latent, expected)
        assert terms.eligible.item() == eligible_count / 2
        gradients = torch.autograd.grad(
            terms.latent, parameters, allow_unused=True, retain_graph=True
        )
        # Empty ancestry still keeps a differentiable zero through the predictor.
        for actual, reference in zip(gradients, reference_gradients, strict=True):
            if reference is None and actual is not None:
                assert actual.count_nonzero() == 0
            else:
                torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], results[2])


def test_windows_never_connect_adjacent_groups_even_when_metadata_is_contiguous():
    belief, inputs, factors = _tensor_case([0] * 6, list(range(6)))
    dynamics = _TensorTransition()
    window = actor_window_loss(
        dynamics, None, belief, inputs, factors, decision_horizon=0, latent_horizon=2
    )
    isolated = {**factors, "episode_index": torch.arange(6) // 3, "step": torch.arange(6) % 3}
    expected = actor_horizon_loss(
        dynamics, None, belief, inputs, isolated, decision_horizon=0, latent_horizon=2
    )
    torch.testing.assert_close(window, expected)
    assert window.eligible == 3


@pytest.mark.parametrize("availability", [(False, True, True), (True, False, True)])
def test_head_latent_targets_require_unbroken_unit_survival(availability):
    belief, inputs, factors = _tensor_case([0, 0, 0], [0, 1, 2])
    active = inputs.unit_active.clone()
    active[:, 0] = torch.tensor(availability)
    inputs = inputs._replace(unit_active=active)
    expected = []
    for offset in (1, 2):
        errors = []
        for source in range(3 - offset):
            target = source + offset
            surviving = active[source : target + 1].all(dim=0)
            errors.extend(
                (
                    nn.functional.smooth_l1_loss(
                        belief.unit_decisions[source, surviving],
                        belief.unit_decisions[target, surviving].detach(),
                        reduction="none",
                    ).flatten(),
                    nn.functional.smooth_l1_loss(
                        belief.market_decisions[source],
                        belief.market_decisions[target].detach(),
                        reduction="none",
                    ).flatten(),
                )
            )
        expected.append(torch.cat(errors).mean())
    reference = torch.stack(expected).mean()
    plan = structured_horizon_plan(np.zeros(3, dtype=np.int64), np.arange(3), 2)
    results = [
        actor_horizon_loss(
            PersistenceDynamics(),
            None,
            belief,
            inputs,
            factors,
            decision_horizon=0,
            latent_horizon=2,
            plan=selection,
        )
        for selection in (None, plan)
    ]
    results.append(
        actor_window_loss(
            PersistenceDynamics(),
            None,
            belief,
            inputs,
            factors,
            decision_horizon=0,
            latent_horizon=2,
        )
    )
    for terms in results:
        torch.testing.assert_close(terms.latent, reference)


def test_head_controls_keep_joint_actions_paired_without_rng():
    class JointTransition:
        def __call__(self, belief, units, kinds, quantities, *context):
            return StructuredDecisionBelief(
                *(value + (units + 2 * kinds + 3 * quantities)[:, :1, None] for value in belief)
            )

    belief, inputs, factors = _tensor_case([0, 0, 1, 1], [0, 1, 0, 1])
    factors["unit_actions"] = torch.tensor([[1], [0], [3], [0]])
    factors["market_kinds"] = factors["unit_actions"] + 1
    factors["market_quantities"] = factors["unit_actions"] + 2
    belief = StructuredDecisionBelief(*(torch.zeros_like(value) for value in belief))
    transition = JointTransition()
    actions = tuple(factors[name] for name in ("unit_actions", "market_kinds", "market_quantities"))
    for row in (0, 2):
        successor = transition(
            StructuredDecisionBelief(*(value[row : row + 1] for value in belief)),
            *(value[row : row + 1] for value in actions),
        )
        for value, target in zip(belief, successor, strict=True):
            value[row + 1] = target[0]
    rng = torch.get_rng_state().clone()

    def loss(model):
        return actor_horizon_loss(
            model, None, belief, inputs, factors, decision_horizon=0, latent_horizon=1
        ).latent

    assert loss(transition) == 0
    assert loss(PersistenceDynamics()) > 0
    assert loss(ShuffledActionDynamics(transition)) > 0
    torch.testing.assert_close(
        ShuffledActionDynamics(transition)(belief, *actions),
        transition(belief, *(value.roll(1, 0) for value in actions)),
    )
    assert torch.equal(rng, torch.get_rng_state())


@pytest.fixture
def cuda_case():
    torch.manual_seed(159)
    config = StructuredConfig(
        model_dim=32,
        attention_heads=2,
        attention_kv_heads=2,
        latents=4,
        core_layers=1,
        farm_blocks=1,
        quantity_rank=4,
    )
    actor = StructuredActor(config).cuda()
    with torch.no_grad():
        actor.market_quantity_kind_gate.weight.normal_()
        actor.market_quantity_value.weight.normal_()
    rows = 6
    active = torch.zeros(rows, MAX_UNITS, dtype=torch.bool, device="cuda")
    active[:, 0] = True
    active[1:3, 1] = True  # Newborn state is unavailable on its incoming edge.
    values = {name: torch.zeros(rows, 1, device="cuda") for name in StructuredInputs._fields}
    values.update(
        unit_active=active,
        unit_categorical=torch.zeros(rows, MAX_UNITS, 4, dtype=torch.long, device="cuda"),
        unit_tile_gather_valid=torch.ones(rows, MAX_UNITS, 5, dtype=torch.bool, device="cuda"),
    )
    inputs = StructuredInputs(**values)
    belief = StructuredDecisionBelief(
        *(
            torch.randn(rows, slots, config.model_dim, device="cuda", requires_grad=True)
            for slots in (MAX_UNITS, MAX_MARKET_ORDERS)
        )
    )
    market_active = torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.bool, device="cuda")
    market_active[:, :2] = True
    quantity_active = market_active.clone()
    quantity_active[:, 1] = False
    kinds = torch.zeros(rows, MAX_MARKET_ORDERS, dtype=torch.long, device="cuda")
    kinds[:, 0] = MarketKind.BUY_SEED_WHEAT
    factors = {
        "episode_index": torch.arange(rows, device="cuda") // 3,
        "step": torch.arange(rows, device="cuda") % 3,
        "unit_actions": torch.zeros(rows, MAX_UNITS, dtype=torch.long, device="cuda"),
        "market_kinds": kinds,
        "market_quantities": torch.zeros_like(kinds),
        "unit_active": active,
        "market_active": market_active,
        "market_quantity_active": quantity_active,
        "unit_masks": active.unsqueeze(-1).expand(-1, -1, N_UNIT_ACTIONS),
        "market_kind_masks": market_active.unsqueeze(-1).expand(-1, -1, N_MARKET_KINDS),
        "market_quantity_masks": quantity_active.unsqueeze(-1).expand(-1, -1, N_QUANTITIES),
    }
    return actor, belief, inputs, factors


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("objectives", [(0, 2), (2, 0), (2, 2)])
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_actor_losses_train_source_heads_and_predictor_without_teacher_gradients(
    cuda_case, objectives
):
    actor, belief, inputs, factors = cuda_case
    dynamics = ActorDynamics(actor.config).cuda()
    decision_horizon, latent_horizon = objectives
    terms = actor_window_loss(
        dynamics,
        actor,
        belief,
        inputs,
        factors,
        decision_horizon=decision_horizon,
        latent_horizon=latent_horizon,
    )
    (terms.latent + terms.decision).backward()
    for value in belief:
        assert value.grad[:2].abs().sum() > 0
        assert value.grad[[2, 5]].count_nonzero() == 0
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in dynamics.parameters()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_joint_action_padding_is_inert_but_stop_and_active_heads_train():
    torch.manual_seed(915)
    dynamics = ActorDynamics(StructuredConfig(model_dim=32, latents=4)).cuda()
    belief = StructuredDecisionBelief(
        *(
            torch.randn(2, slots, 32, device="cuda", requires_grad=True)
            for slots in (MAX_UNITS, MAX_MARKET_ORDERS)
        )
    )
    active = torch.zeros(2, MAX_UNITS, dtype=torch.bool, device="cuda")
    active[:, :2] = True
    units = torch.zeros(2, MAX_UNITS, dtype=torch.long, device="cuda")
    kinds = torch.zeros(2, MAX_MARKET_ORDERS, dtype=torch.long, device="cuda")
    kinds[:, 0] = MarketKind.HIRE
    quantities = torch.zeros_like(kinds)
    categorical = torch.zeros(2, MAX_UNITS, 4, dtype=torch.long, device="cuda")
    encoded = []

    def retain_actions(_module, _inputs, output):
        for value in output:
            value.retain_grad()
            encoded.append(value)

    hook = dynamics.action.register_forward_hook(retain_actions)
    predicted = dynamics(belief, units, kinds, quantities, categorical, active)
    hook.remove()
    other_categorical = categorical.clone()
    other_categorical[:, 2:, 2:] = BOARD_SIZE - 1
    other_kinds = kinds.clone()
    other_kinds[:, 2:] = MarketKind.BUY_SEED_WHEAT
    other_quantities = quantities.clone()
    other_quantities[:, 2:] = N_QUANTITIES - 1
    other = dynamics(
        belief,
        units.masked_fill(~active, N_UNIT_ACTIONS - 1),
        other_kinds,
        other_quantities,
        other_categorical,
        active,
    )
    torch.testing.assert_close(other, predicted, rtol=0, atol=0)
    sum(value.square().sum() for value in predicted).backward()
    assert belief.unit_decisions.grad[~active].count_nonzero() == 0
    assert (belief.unit_decisions.grad[active].abs().sum(-1) > 0).all()
    assert (belief.market_decisions.grad.abs().sum(-1) > 0).all()
    assert encoded[0].grad[~active].count_nonzero() == 0
    assert encoded[0].grad[active].abs().sum() > 0
    assert encoded[1].grad[:, 2:].count_nonzero() == 0
    assert encoded[1].grad[:, 1].abs().sum() > 0
    changed_units = units.clone()
    changed_units[:, 1] = N_UNIT_ACTIONS - 1
    changed = dynamics(belief, changed_units, kinds, quantities, categorical, active)
    assert any(not torch.allclose(a, b) for a, b in zip(changed, predicted, strict=True))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_decoded_dense_compact_windows_match_surviving_head_reference(cuda_case):
    actor, belief, inputs, factors = cuda_case
    plan = structured_horizon_plan(np.repeat(np.arange(2), 3), np.tile(np.arange(3), 2), 2)
    plan = type(plan)(*(value.cuda() for value in plan))
    results = [
        actor_horizon_loss(
            PersistenceDynamics(),
            actor,
            belief,
            inputs,
            factors,
            decision_horizon=2,
            latent_horizon=2,
            plan=selection,
        )
        for selection in (None, plan)
    ]
    results.append(
        actor_window_loss(
            PersistenceDynamics(),
            actor,
            belief,
            inputs,
            factors,
            decision_horizon=2,
            latent_horizon=2,
        )
    )
    heads = DecodeHeads.from_actor(actor, normalized_units=True)
    expected = []
    for offset in (1, 2):
        source = torch.tensor(
            [start + position for start in (0, 3) for position in range(3 - offset)], device="cuda"
        )
        target = source + offset
        student = heads.decode(torch.cat(tuple(value[source] for value in belief), dim=1))
        teacher = heads.decode(torch.cat(tuple(value[target].detach() for value in belief), dim=1))
        masks = DecodeMasks(*(factors[name][target] for name in DecodeMasks._fields))
        surviving = inputs.unit_active[source].clone()
        for step in range(1, offset + 1):
            surviving &= inputs.unit_active[source + step]
        masks = masks._replace(unit_active=masks.unit_active & surviving)
        pairs = (
            (student.unit_logits, teacher.unit_logits, masks.unit_masks, masks.unit_active),
            (
                student.market_kind_logits,
                teacher.market_kind_logits,
                masks.market_kind_masks,
                masks.market_active,
            ),
            (
                heads.quantity_logits(student.market_quantity_context, masks.market_kinds),
                heads.quantity_logits(teacher.market_quantity_context, masks.market_kinds),
                masks.market_quantity_masks,
                masks.market_quantity_active,
            ),
        )
        errors = [
            _decision_kl(predicted, truth, mask, active.float())
            for predicted, truth, mask, active in pairs
        ]
        expected.append(sum(error for error, _ in errors) / sum(count for _, count in errors))
    expected_decision = torch.stack(expected).mean()
    for terms in results:
        torch.testing.assert_close(terms.decision, expected_decision, rtol=0.02, atol=2e-5)
    # Compact padding and shrinking windows can select different CUDA kernels.
    for terms in results[1:]:
        torch.testing.assert_close(terms, results[0], rtol=0.02, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_exact_successor_heads_have_zero_kl_without_reencoding(cuda_case):
    actor, belief, inputs, factors = cuda_case

    class ExactSuccessor:
        def __call__(self, source, *actions):
            return StructuredDecisionBelief(
                *(value[torch.tensor([1, 2, 2, 4, 5, 5], device="cuda")] for value in belief)
            )

    def forbidden_trunk(*_args):
        raise AssertionError("auxiliary decoding must not reencode observations")

    hook = actor.trunk.register_forward_pre_hook(forbidden_trunk)
    try:
        terms = actor_horizon_loss(
            ExactSuccessor(), actor, belief, inputs, factors, decision_horizon=1, latent_horizon=1
        )
    finally:
        hook.remove()
    torch.testing.assert_close(terms.latent, torch.zeros_like(terms.latent), rtol=0, atol=0)
    torch.testing.assert_close(terms.decision, torch.zeros_like(terms.decision), rtol=0, atol=1e-7)
