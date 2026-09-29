from __future__ import annotations

import copy

import pytest
import torch

from kaggriculture.optim import (
    NorMuon,
    polar_express,
    route_parameters,
)
from kaggriculture.ppo import PpoConfig, make_optimizers
from kaggriculture.production import PRODUCTION_ARCHITECTURE, production_model_config
from kaggriculture.registry import ENTITY_ATTENTION, resolve_architecture
from kaggriculture.structured import StructuredConfig
from kaggriculture.structured_dynamics import StructuredDynamics

SHAPES = ((96, 96), (384, 96), (96, 384), (48, 432), (22, 100))


def _decaying_spectrum(shape: tuple[int, int], seed: int = 0) -> torch.Tensor:
    """A matrix with a steep spectrum, which is what a real gradient looks like."""

    generator = torch.Generator().manual_seed(seed)
    left, values, right = torch.linalg.svd(
        torch.randn(*shape, generator=generator), full_matrices=False
    )
    return (left * torch.linspace(1.0, 0.01, values.numel()) ** 2) @ right


def _exact_polar(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(matrix.double(), full_matrices=False)
    return left @ right


def _model(seed: int = 0) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Linear(32, 8))


def _optimizer(model: torch.nn.Module, **overrides: float) -> NorMuon:
    settings: dict[str, float] = {"learning_rate": 1e-2, "adam_learning_rate": 1e-3}
    settings.update(overrides)
    return NorMuon(
        [parameter for parameter in model.parameters() if parameter.ndim >= 2],
        [parameter for parameter in model.parameters() if parameter.ndim < 2],
        **settings,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_polar_express_recovers_the_direction_of_an_exact_svd_polar_factor(
    shape: tuple[int, int],
) -> None:
    matrix = _decaying_spectrum(shape)
    approximation = polar_express(matrix).double()
    reference = _exact_polar(matrix)
    cosine = (approximation * reference).sum() / (approximation.norm() * reference.norm())
    # Five iterations at this cushion do not reach an exact orthogonal factor,
    # and are not meant to: the coefficients trade the smallest singular
    # directions away rather than amplify what is noise there.
    assert cosine > 0.94


@pytest.mark.parametrize("shape", SHAPES)
def test_polar_express_flattens_the_spectrum_without_exploding_it(
    shape: tuple[int, int],
) -> None:
    values = torch.linalg.svdvals(polar_express(_decaying_spectrum(shape)).float())
    # An exact polar factor has every singular value at one. The iteration's
    # cushion overshoots slightly, and that bound is what keeps a step from
    # being larger than the learning rate claims.
    assert values.max() < 1.2


@pytest.mark.parametrize("shape", SHAPES)
def test_polar_express_is_invariant_to_the_scale_of_its_input(shape: tuple[int, int]) -> None:
    """Small PPO momenta must have the same update direction as larger copies."""
    matrix = _decaying_spectrum(shape) * 1e-6
    once = polar_express(matrix)
    scaled = polar_express(matrix * 1000.0)
    assert (once - scaled).norm() / once.norm() < 1e-4


def test_polar_express_handles_both_orientations_consistently() -> None:
    matrix = _decaying_spectrum((384, 96))
    upright = polar_express(matrix)
    transposed = polar_express(matrix.mT.contiguous())
    assert torch.allclose(upright, transposed.mT, atol=1e-5)


def test_polar_express_rejects_a_matrix_without_two_dimensions() -> None:
    with pytest.raises(ValueError, match="two dimensions"):
        polar_express(torch.zeros(8))


def test_polar_express_preserves_zero_matrices_in_a_mixed_batch() -> None:
    matrix = _decaying_spectrum((22, 100)) * 1e-6
    batch = torch.stack((torch.zeros_like(matrix), matrix))
    result = polar_express(batch)
    assert torch.equal(result[0], torch.zeros_like(matrix))
    torch.testing.assert_close(result[1], polar_express(matrix), rtol=1e-4, atol=1e-5)


def test_a_normuon_step_is_invariant_to_the_gradient_scale() -> None:
    steps = []
    for scale in (1.0, 1000.0):
        model = _model()
        optimizer = _optimizer(model)
        before = torch.cat(
            [p.detach().reshape(-1) for p in model.parameters() if p.ndim >= 2]
        ).clone()
        inputs = torch.randn(64, 16, generator=torch.Generator().manual_seed(7))
        ((model(inputs) ** 2).mean() * scale).backward()
        optimizer.step()
        after = torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.ndim >= 2])
        steps.append(after - before)
    assert (steps[0] - steps[1]).norm() / steps[0].norm() < 1e-3


def test_the_relative_step_size_of_a_square_matrix_is_the_learning_rate() -> None:
    """What makes a NorMuon rate transferable, and what an Adam rate is not.

    The orthogonalized update has Frobenius norm `lr * sqrt(min(rows, cols))`
    against a weight norm near `sqrt(fan_out)`, so for a square matrix the
    learning rate IS the fraction of itself the layer moves per step. The
    schedule comments in `ppo.py` depend on this identity holding.
    """

    layer = torch.nn.Linear(96, 96, bias=False)
    initial = layer.weight.detach().clone()
    optimizer = NorMuon([layer.weight], [], learning_rate=1e-2, adam_learning_rate=1e-3)
    layer.weight.grad = torch.randn_like(layer.weight)
    optimizer.step()
    relative = (layer.weight.detach() - initial).norm() / initial.norm()
    assert 0.5e-2 < relative < 2e-2


def test_the_shape_multiplier_keeps_a_tall_matrix_from_taking_a_smaller_step() -> None:
    relatives = []
    for rows, columns in ((96, 96), (384, 96)):
        layer = torch.nn.Linear(columns, rows, bias=False)
        initial = layer.weight.detach().clone()
        optimizer = NorMuon([layer.weight], [], learning_rate=1e-2, adam_learning_rate=1e-3)
        layer.weight.grad = torch.randn_like(layer.weight)
        optimizer.step()
        relatives.append(((layer.weight.detach() - initial).norm() / initial.norm()).item())
    assert relatives[1] == pytest.approx(relatives[0], rel=0.35)


def test_a_found_inf_step_leaves_parameters_moments_and_the_counter_untouched() -> None:
    model = _model()
    optimizer = _optimizer(model)
    inputs = torch.randn(64, 16)
    (model(inputs) ** 2).mean().backward()
    optimizer.step()

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    state_before = copy.deepcopy(
        {
            index: {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in optimizer.state[parameter].items()
            }
            for index, parameter in enumerate(model.parameters())
        }
    )
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    optimizer.grad_scale = None
    optimizer.found_inf = torch.ones(())
    try:
        optimizer.step()
    finally:
        del optimizer.grad_scale, optimizer.found_inf

    for before, parameter in zip(parameters_before, model.parameters(), strict=True):
        assert torch.equal(before, parameter.detach())
        assert torch.isfinite(parameter).all()
    for index, parameter in enumerate(model.parameters()):
        for key, value in optimizer.state[parameter].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(state_before[index][key], value)
            else:
                assert state_before[index][key] == value


def test_a_zero_found_inf_step_applies_exactly_what_an_ungated_step_would() -> None:
    outcomes = []
    for gated in (False, True):
        model = _model()
        optimizer = _optimizer(model)
        inputs = torch.randn(64, 16, generator=torch.Generator().manual_seed(3))
        (model(inputs) ** 2).mean().backward()
        if gated:
            optimizer.grad_scale = None
            optimizer.found_inf = torch.zeros(())
        optimizer.step()
        if gated:
            del optimizer.grad_scale, optimizer.found_inf
        outcomes.append(torch.cat([p.detach().reshape(-1) for p in model.parameters()]))
    assert torch.allclose(outcomes[0], outcomes[1], atol=1e-7)


def test_the_optimizer_state_survives_a_save_and_load(tmp_path) -> None:
    """Resume must reproduce the next step exactly, which is how training resumes.

    The state_dict is round-tripped through `torch.save`, deliberately: torch's
    `load_state_dict` hands back the *same* state tensors when dtype and device
    already match, so loading a live `state_dict()` aliases the donor's buffers
    and any later step by either optimizer would corrupt the other. Every real
    resume comes off disk, where the tensors are fresh.
    """

    inputs = torch.randn(64, 16, generator=torch.Generator().manual_seed(11))
    original = _model()
    original_optimizer = _optimizer(original)
    (original(inputs) ** 2).mean().backward()
    original_optimizer.step()

    path = tmp_path / "optimizer.pt"
    torch.save({"model": original.state_dict(), "optimizer": original_optimizer.state_dict()}, path)
    payload = torch.load(path, weights_only=False)

    resumed = _model()
    resumed_optimizer = _optimizer(resumed)
    resumed.load_state_dict(payload["model"])
    resumed_optimizer.load_state_dict(payload["optimizer"])

    for model, optimizer in ((original, original_optimizer), (resumed, resumed_optimizer)):
        for parameter in model.parameters():
            parameter.grad = None
        (model(inputs) ** 2).mean().backward()
        optimizer.step()

    for left, right in zip(original.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(left.detach(), right.detach())


def test_the_optimizer_descends_a_regression_it_can_solve() -> None:
    model = _model()
    optimizer = _optimizer(model)
    inputs = torch.randn(64, 16, generator=torch.Generator().manual_seed(5))
    targets = torch.randn(64, 8, generator=torch.Generator().manual_seed(6))
    losses = []
    for _ in range(100):
        for parameter in model.parameters():
            parameter.grad = None
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8


def test_a_parameter_without_a_gradient_is_skipped_rather_than_stepped() -> None:
    layer = torch.nn.Linear(32, 16)
    optimizer = NorMuon([layer.weight], [layer.bias], learning_rate=1e-2, adam_learning_rate=1e-3)
    before = layer.weight.detach().clone()
    layer.bias.grad = torch.randn_like(layer.bias)
    optimizer.step()
    assert torch.equal(before, layer.weight.detach())
    assert optimizer.state[layer.weight] == {}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("weight_decay", [0.0, 0.005])
def test_adam_matches_reference_with_missing_gradients_gates_and_resume(
    device: str,
    weight_decay: float,
) -> None:
    # CUDA/no-decay exercises native fusion; the other cases retain cautious
    # decay and CPU arithmetic. Distinct histories must not share a timestep.
    expected = [
        torch.linspace(-0.4, 0.3, 7),
        torch.linspace(-0.2, 0.4, 10).reshape(2, 5),
        torch.tensor(0.15),
    ]
    parameters = [torch.nn.Parameter(value.to(device).clone()) for value in expected]
    optimizer = NorMuon(
        [],
        parameters,
        learning_rate=0.01,
        adam_learning_rate=0.003,
        adam_betas=(0.8, 0.95),
        adam_epsilon=1e-12,
        adam_weight_decay=weight_decay,
    )
    reference_states: list[dict[str, torch.Tensor]] = [{}, {}, {}]
    resumed_parameters = None
    resumed_optimizer = None
    schedule = [
        (2.0, (float("inf"), float("nan"), None)),
        (0.0, (0.03, -0.04, None)),
        (None, (0.0, None, 2e-12)),
        (float("nan"), (float("nan"), None, float("inf"))),
        (0.0, (-3e-12, 0.02, 0.0)),
        (0.0, (None, 0.0, -0.03)),
        (-2.0, (float("inf"), float("nan"), float("inf"))),
        (None, (0.01, -0.03, None)),
    ]
    for index, (gate, magnitudes) in enumerate(schedule):
        learning_rate = 0.003 * (index + 1) / len(schedule)
        skipped = gate is not None and gate != 0
        gradients = [
            None if magnitude is None else torch.full_like(value, magnitude)
            for value, magnitude in zip(expected, magnitudes, strict=True)
        ]
        # Independent per-parameter reference: the original FP32 lerp and
        # corrected moments, including quadratic, sign-cautious decay.
        for value, gradient, state in zip(expected, gradients, reference_states, strict=True):
            if gradient is None:
                continue
            if not state:
                state.update(
                    step=torch.zeros(()),
                    exp_avg=torch.zeros_like(value),
                    exp_avg_sq=torch.zeros_like(value),
                )
            if skipped:
                continue
            state["step"].add_(1)
            state["exp_avg"].lerp_(gradient, 0.2)
            state["exp_avg_sq"].lerp_(gradient.square(), 0.05)
            bias1 = (1 - 0.8 ** state["step"]).clamp_min(1e-12)
            bias2 = (1 - 0.95 ** state["step"]).clamp_min(1e-12)
            update = (state["exp_avg"] / bias1) / ((state["exp_avg_sq"] / bias2).sqrt() + 1e-12)
            decay = ((update * value) > 0) * value * (weight_decay * learning_rate**2)
            value.sub_(update * learning_rate + decay)

        instances = [(parameters, optimizer)]
        if resumed_optimizer is not None:
            instances.append((resumed_parameters, resumed_optimizer))
        for selected_parameters, selected_optimizer in instances:
            selected_optimizer.param_groups[0]["lr"] = learning_rate
            before_parameters = [parameter.detach().clone() for parameter in selected_parameters]
            before_state = copy.deepcopy(selected_optimizer.state_dict())
            for parameter, gradient in zip(selected_parameters, gradients, strict=True):
                parameter.grad = None if gradient is None else gradient.to(device).clone()
            if gate is not None:
                # Non-binary, double-precision and NaN gates must all retain
                # the public nonzero-means-skip contract of the eager path.
                selected_optimizer.found_inf = torch.tensor(
                    gate, device=device, dtype=torch.float64
                )
            selected_optimizer.step()
            if gate is not None:
                del selected_optimizer.found_inf
            for position, (parameter, value, gradient, state) in enumerate(
                zip(selected_parameters, expected, gradients, reference_states, strict=True)
            ):
                torch.testing.assert_close(parameter.detach().cpu(), value, rtol=1e-5, atol=1e-7)
                actual = selected_optimizer.state[parameter]
                assert actual.keys() == state.keys()
                for key, reference in state.items():
                    assert actual[key].dtype == torch.float32
                    assert actual[key].shape == reference.shape
                    torch.testing.assert_close(
                        actual[key].cpu(),
                        reference,
                        rtol=0 if key == "step" else 1e-5,
                        atol=0 if key == "step" else 1e-30,
                    )
                if gradient is not None:
                    torch.testing.assert_close(
                        parameter.grad.cpu(),
                        gradient,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                if skipped:
                    assert torch.equal(parameter, before_parameters[position])
                    previous = before_state["state"].get(position, {})
                    for key, previous_value in previous.items():
                        assert torch.equal(actual[key], previous_value)
        if resumed_optimizer is not None:
            for original, resumed in zip(parameters, resumed_parameters, strict=True):
                assert torch.equal(original, resumed)
                for key, value in optimizer.state[original].items():
                    assert torch.equal(value, resumed_optimizer.state[resumed][key])
        if index == 4:
            resumed_parameters = [
                torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters
            ]
            resumed_optimizer = NorMuon(
                [],
                resumed_parameters,
                learning_rate=0.01,
                adam_learning_rate=0.003,
            )
            resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_adam_does_not_sanitize_an_applied_nonfinite_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3], device="cuda"))
    parameter.grad = torch.tensor([float("nan"), float("inf"), 0.25], device="cuda")
    optimizer = NorMuon([], [parameter], learning_rate=0.01, adam_learning_rate=0.003)
    optimizer.found_inf = torch.zeros((), device="cuda")
    optimizer.step()
    assert not torch.isfinite(parameter[:2]).any()
    assert torch.isfinite(parameter[2])
    for key in ("exp_avg", "exp_avg_sq"):
        assert not torch.isfinite(optimizer.state[parameter][key][:2]).any()
    assert optimizer.state[parameter]["step"].item() == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_adam_keeps_bias_correction_floor_when_beta_rounds_to_one() -> None:
    outcomes = []
    for device in ("cpu", "cuda"):
        parameter = torch.nn.Parameter(torch.tensor([0.1, -0.2], device=device))
        optimizer = NorMuon(
            [],
            [parameter],
            learning_rate=0.01,
            adam_learning_rate=0.003,
            adam_betas=(1.0 - 1e-10, 0.95),
        )
        parameter.grad = torch.tensor([0.25, -0.5], device=device)
        optimizer.step()
        outcomes.append(
            (
                parameter.detach().cpu(),
                {key: value.cpu() for key, value in optimizer.state[parameter].items()},
            )
        )
    torch.testing.assert_close(outcomes[1][0], outcomes[0][0])
    for key, value in outcomes[0][1].items():
        torch.testing.assert_close(outcomes[1][1][key], value)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_adam_skip_preserves_a_large_checkpoint_counter() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.1, -0.2], device="cuda"))
    optimizer = NorMuon([], [parameter], learning_rate=0.01, adam_learning_rate=0.003)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["step"].fill_(2**24)
    before = parameter.detach().clone()
    state_before = copy.deepcopy(optimizer.state[parameter])
    parameter.grad.fill_(float("inf"))
    optimizer.found_inf = torch.ones((), device="cuda")
    optimizer.step()
    assert torch.equal(parameter, before)
    for key, value in state_before.items():
        assert torch.equal(optimizer.state[parameter][key], value)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", 0.0),
        ("adam_learning_rate", -1.0),
        ("momentum", 1.0),
        ("beta2", -0.1),
    ),
)
def test_unusable_hyperparameters_are_rejected(field: str, value: float) -> None:
    layer = torch.nn.Linear(8, 8)
    settings: dict[str, float] = {"learning_rate": 1e-2, "adam_learning_rate": 1e-3}
    settings[field] = value
    with pytest.raises(ValueError):
        NorMuon([layer.weight], [layer.bias], **settings)


def test_a_vector_routed_to_the_matrix_set_is_rejected() -> None:
    layer = torch.nn.Linear(8, 8)
    with pytest.raises(ValueError, match="two dimensions"):
        NorMuon([layer.bias], [], learning_rate=1e-2, adam_learning_rate=1e-3)


def test_an_optimizer_needs_at_least_one_parameter() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        NorMuon([], [], learning_rate=1e-2, adam_learning_rate=1e-3)


def _production_modules(
    family: str = PRODUCTION_ARCHITECTURE,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    architecture = resolve_architecture(family)
    config = (
        architecture.config_class(**production_model_config())
        if family == PRODUCTION_ARCHITECTURE
        else architecture.config_class()
    )
    return architecture.actor_class(config), architecture.critic_class(config)


# Production's family and the entity-attention trunk it wraps, whose query banks
# and attention the `lejepa` critic reads but does not own.
ROUTED_FAMILIES = pytest.mark.parametrize("family", [PRODUCTION_ARCHITECTURE, ENTITY_ATTENTION])


@ROUTED_FAMILIES
def test_every_production_parameter_is_routed_exactly_once(family) -> None:
    for module in _production_modules(family):
        matrices, vectors, multipliers = route_parameters(module)
        assert len(multipliers) == len(vectors)
        routed = {id(parameter) for parameter in matrices} | {
            id(parameter) for parameter in vectors
        }
        assert len(matrices) + len(vectors) == len(routed)
        assert routed == {id(parameter) for parameter in module.parameters()}


@ROUTED_FAMILIES
def test_production_lookup_and_head_roles_are_not_hidden_matrices(family) -> None:
    for module in _production_modules(family):
        matrices, vectors, _ = route_parameters(module)
        matrix_ids = {id(parameter) for parameter in matrices}
        vector_ids = {id(parameter) for parameter in vectors}
        for name, child in module.named_modules():
            if isinstance(child, torch.nn.Embedding):
                assert id(child.weight) in vector_ids, name
        if hasattr(module, "value_query"):
            assert id(module.value_query) in vector_ids
            assert id(module.value_head.weight) in vector_ids
        else:
            assert id(module.unit_head[-1].weight) in vector_ids
            assert id(module.market_kind.weight) in vector_ids
            assert id(module.market_quantity_bias) in vector_ids
        # A `lejepa` critic owns no trunk: it reads the actor's backbone, which
        # wraps the entity trunk its world model trains, through its own rounds.
        if not hasattr(module, "trunk"):
            assert id(module.pool_attention.query.weight) in matrix_ids
            continue
        trunk = getattr(module.trunk, "trunk", module.trunk)
        for name in ("opponent_queries", "latent_queries", "market_queries"):
            query = getattr(trunk, name, None)
            if query is not None:
                parameter = query.weight if isinstance(query, torch.nn.Embedding) else query
                assert id(parameter) in vector_ids
        core = trunk.core[0]
        attention = core.attention if hasattr(core, "attention") else core.self_attention
        assert id(attention.query.weight) in matrix_ids


def test_convolution_weights_remain_hidden_matrices() -> None:
    # Production is structured and currently has no convolution. Pin the
    # standard Muon flattening convention independently so a later CNN route
    # cannot regress when the production architecture changes.
    convolution = torch.nn.Conv2d(2, 3, kernel_size=3)
    matrices, vectors, _ = route_parameters(convolution)
    assert any(parameter is convolution.weight for parameter in matrices)
    assert any(parameter is convolution.bias for parameter in vectors)


def test_structured_transition_lookup_tables_stay_on_adam() -> None:
    dynamics = StructuredDynamics(
        StructuredConfig(model_dim=16, attention_heads=2, ffn_multiplier=1)
    )
    _, vectors, _ = route_parameters(dynamics)
    vector_ids = {id(parameter) for parameter in vectors}
    for module in dynamics.modules():
        if isinstance(module, torch.nn.Embedding):
            assert id(module.weight) in vector_ids


def test_lookup_updates_ignore_other_rows_even_with_a_tied_projection() -> None:
    module = torch.nn.Module()
    module.projection = torch.nn.Linear(4, 3, bias=False)
    module.vocabulary = torch.nn.Embedding(3, 4)
    module.projection.weight = module.vocabulary.weight
    isolated = copy.deepcopy(module)
    optimizers = [
        NorMuon(*route_parameters(model), learning_rate=1e-2, adam_learning_rate=1e-3)
        for model in (module, isolated)
    ]
    gradient = torch.tensor([[1.0, 2.0, -1.0, 0.5], [3.0, -1.0, 2.0, 1.0], [2.0, 1.0, 3.0, -1.0]])
    for scale in (1.0, -0.5):
        module.vocabulary.weight.grad = gradient * scale
        isolated.vocabulary.weight.grad = gradient * scale
        isolated.vocabulary.weight.grad[1:].zero_()
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(
            module.vocabulary.weight[0], isolated.vocabulary.weight[0], rtol=0, atol=0
        )


@ROUTED_FAMILIES
def test_make_optimizers_builds_normuon_for_both_networks_by_default(family) -> None:
    actor, critic = _production_modules(family)
    config = PpoConfig()
    assert config.optimizer == "normuon"
    assert (config.actor_learning_rate, config.critic_learning_rate) == (5e-5, 1.5e-4)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, config)
    for module, optimizer, learning_rate in (
        (actor, actor_optimizer, 5e-5),
        (critic, critic_optimizer, 1.5e-4),
    ):
        assert isinstance(optimizer, NorMuon)
        rates = {
            (group["kind"], group.get("lr_multiplier")): group["lr"]
            for group in optimizer.param_groups
        }
        adam_rate = learning_rate * PpoConfig().adam_learning_rate_ratio
        assert rates[("normuon", None)] == pytest.approx(learning_rate)
        assert rates[("adam", 1.0)] == pytest.approx(adam_rate)
        # A `lejepa` backbone steps in the world model's optimizer, not these.
        if hasattr(getattr(module, "trunk", None), "opponent_queries"):
            # Query banks initialized at 0.02 RMS take 0.02 of the shared Adam
            # rate, so every Adam parameter moves by the same fraction of itself.
            assert rates[("adam", 0.02)] == pytest.approx(adam_rate * 0.02)
        # `_optimizer_step` drives the warmup through this metadata.
        for group in optimizer.param_groups:
            assert group["base_lr"] == pytest.approx(group["lr"])
            assert group["warmup_step"] == 0


def test_the_critic_step_can_be_gated_on_the_device_under_normuon() -> None:
    # `update_ppo` asks for this capability rather than inferring it from a
    # fused AdamW, and a false answer would cost a host synchronization per
    # minibatch or an ungated poisoned step.
    actor, critic = _production_modules()
    _, critic_optimizer = make_optimizers(actor, critic, PpoConfig())
    assert getattr(critic_optimizer, "supports_found_inf", False)


def test_the_adamw_optimizer_remains_available_for_comparison() -> None:
    actor, critic = _production_modules()
    actor_optimizer, _ = make_optimizers(actor, critic, PpoConfig(optimizer="adamw"))
    assert isinstance(actor_optimizer, torch.optim.AdamW)


def test_an_unknown_optimizer_name_is_rejected() -> None:
    actor, critic = _production_modules()
    with pytest.raises(ValueError, match="unsupported optimizer"):
        make_optimizers(actor, critic, PpoConfig(optimizer="lion"))


def _matrix_step(weight_decay: float, learning_rate: float = 1e-2) -> tuple[torch.Tensor, ...]:
    """One NorMuon step on a square matrix; returns the weight before and after.

    Square on purpose: the shape multiplier is then exactly 1, so the step is
    the bare rate and the expected decay factor is `lr * lr * wd` with nothing
    from `_shape_learning_rate_multiplier` folded in.
    """

    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(96, 96) * 0.1)
    parameter.grad = _decaying_spectrum((96, 96), seed=1)
    optimizer = NorMuon(
        [parameter],
        [],
        learning_rate=learning_rate,
        adam_learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    start = parameter.detach().clone()
    optimizer.step()
    return start, parameter.detach().clone()


def test_cautious_decay_only_shrinks_weights_the_step_was_already_shrinking() -> None:
    learning_rate, decay = 1e-2, 1.2
    start, plain = _matrix_step(0.0, learning_rate)
    start_again, decayed = _matrix_step(decay, learning_rate)
    assert torch.equal(start, start_again)
    # The applied step is observed rather than recomputed: `start - plain` is
    # `step * direction`, so its sign against the weight is the predicate the
    # update uses, without this test restating the algebra it checks.
    applied = start - plain
    difference = plain - decayed
    term = decay * learning_rate * learning_rate * start
    # The term is proportional to the weight and is recovered by subtracting two
    # fp32 results of order 1e-3, so on the smallest weights it sinks under that
    # subtraction's rounding. Pin the exact value where it stands clear of it,
    # and pin the gate itself everywhere.
    resolved = start.abs() > start.abs().median()
    ratio = difference[resolved] / term[resolved]
    untouched = ratio.abs() < 1e-3
    decayed_fully = (ratio - 1.0).abs() < 1e-3
    assert (untouched | decayed_fully).all()
    assert untouched.any() and decayed_fully.any()
    # Exactly untouched wherever the step was already growing the weight, which
    # is the whole content of "cautious".
    grew = applied * start < -1e-8 * start.abs()
    assert grew.any()
    assert (difference[grew] == 0.0).all()
    # The decay's own contribution always points toward zero. Stated on the
    # contribution rather than on the resulting weight, because a step that
    # overshoots zero can leave `decayed` further from it than `plain` while
    # the decay term itself still pointed inward.
    assert (difference * start >= 0).all()


def test_cautious_decay_on_the_adam_half_is_quadratic_in_the_rate() -> None:
    learning_rate, decay = 1e-2, 0.005

    def step(weight_decay: float) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(3)
        parameter = torch.nn.Parameter(torch.randn(128) * 0.1)
        parameter.grad = torch.randn(128) * 0.05
        optimizer = NorMuon(
            [],
            [parameter],
            learning_rate=learning_rate,
            adam_learning_rate=learning_rate,
            adam_weight_decay=weight_decay,
        )
        start = parameter.detach().clone()
        optimizer.step()
        return start, parameter.detach().clone()

    start, plain = step(0.0)
    _, decayed = step(decay)
    shrinking = ((start - plain) * start) > 0
    assert shrinking.any() and not shrinking.all()
    torch.testing.assert_close(decayed[~shrinking], plain[~shrinking], rtol=0.0, atol=0.0)
    expected = plain[shrinking] - decay * learning_rate * learning_rate * start[shrinking]
    torch.testing.assert_close(decayed[shrinking], expected)


def test_a_skipped_minibatch_decays_nothing() -> None:
    # Decay is written against the gated step rather than the bare rate, so a
    # non-finite gradient must leave the weights untouched -- not merely
    # un-stepped, which decay applied separately would quietly violate.
    torch.manual_seed(0)
    matrix = torch.nn.Parameter(torch.randn(96, 96) * 0.1)
    vector = torch.nn.Parameter(torch.randn(128) * 0.1)
    matrix.grad = torch.randn(96, 96)
    vector.grad = torch.randn(128)
    optimizer = NorMuon(
        [matrix],
        [vector],
        learning_rate=1e-2,
        adam_learning_rate=1e-2,
        weight_decay=1.2,
        adam_weight_decay=0.005,
    )
    before = (matrix.detach().clone(), vector.detach().clone())
    optimizer.found_inf = torch.ones((), dtype=torch.float64)
    optimizer.step()
    del optimizer.found_inf
    assert torch.equal(matrix.detach(), before[0])
    assert torch.equal(vector.detach(), before[1])


@pytest.mark.parametrize("value", [-1e-9, float("nan"), float("inf")])
def test_an_unusable_weight_decay_is_rejected(value: float) -> None:
    parameter = torch.nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError, match="weight decay must be finite and non-negative"):
        NorMuon(
            [parameter],
            [],
            learning_rate=1e-3,
            adam_learning_rate=1e-3,
            weight_decay=value,
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("weight_decay", [0.0, 1.2])
@pytest.mark.parametrize("shape", [(24, 24), (48, 16), (16, 48)])
def test_batching_a_shape_group_steps_each_matrix_as_if_it_were_alone(
    shape: tuple[int, int],
    weight_decay: float,
    device: str,
) -> None:
    """One optimizer over many same-shaped matrices must not couple them.

    The matrix half stacks every matrix of one shape into a single Polar
    Express and a single variance reduction, which is only legitimate because
    both reduce over the trailing two dimensions alone. If either ever grew a
    reduction across the batch, the step a matrix takes would start depending
    on which other parameters happened to share its shape -- so compare a group
    of five against five optimizers holding one matrix each. Gated packed
    gradients must match ungated reference steps, including after a skipped
    non-finite minibatch, a NaN skip gate, a zero gradient, and a missing gradient.
    """

    torch.manual_seed(11)
    count = 5
    together = [torch.nn.Parameter(torch.randn(shape, device=device)) for _ in range(count)]
    apart = [torch.nn.Parameter(parameter.detach().clone()) for parameter in together]
    grouped = NorMuon(
        together,
        [],
        learning_rate=1e-2,
        adam_learning_rate=1e-2,
        weight_decay=weight_decay,
    )
    separate = [
        NorMuon(
            [parameter],
            [],
            learning_rate=1e-2,
            adam_learning_rate=1e-2,
            weight_decay=weight_decay,
        )
        for parameter in apart
    ]

    generator = torch.Generator().manual_seed(12)
    for iteration, gate in enumerate((None, 0.0, 2.0, float("nan"), 0.0)):
        skipped = gate is not None and gate != 0
        original_gradients = []
        before = [parameter.detach().clone() for parameter in together]
        before_state = [copy.deepcopy(grouped.state[parameter]) for parameter in together]
        for index in range(count):
            gradient = (
                None
                if iteration == 4 and index == 2
                else torch.full(shape, float("inf"), device=device)
                if skipped
                else torch.zeros(shape, device=device)
                if iteration == 1 and index == 0
                else torch.randn(shape, generator=generator).to(device)
            )
            together[index].grad = gradient
            apart[index].grad = None if gradient is None else gradient.clone()
            original_gradients.append(None if gradient is None else gradient.clone())
        if gate is not None:
            grouped.found_inf = torch.tensor(gate, device=device)
        grouped.step()
        if gate is not None:
            del grouped.found_inf
        if not skipped:
            for optimizer in separate:
                optimizer.step()

        for index in range(count):
            torch.testing.assert_close(together[index], apart[index], rtol=1e-5, atol=1e-6)
            for key in ("momentum", "second_moment"):
                torch.testing.assert_close(
                    grouped.state[together[index]][key],
                    separate[index].state[apart[index]][key],
                    rtol=1e-5,
                    atol=1e-6,
                )
                if skipped:
                    assert torch.equal(
                        grouped.state[together[index]][key], before_state[index][key]
                    )
            if skipped:
                assert torch.equal(together[index], before[index])
            if original_gradients[index] is not None:
                assert torch.equal(together[index].grad, original_gradients[index])


def test_a_mixed_shape_group_batches_only_what_shares_a_shape() -> None:
    """Shapes that appear once still step, and identically to a lone optimizer."""

    torch.manual_seed(13)
    shapes = [(24, 24), (24, 24), (32, 8), (8, 32)]
    together = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
    apart = [torch.nn.Parameter(parameter.detach().clone()) for parameter in together]
    grouped = NorMuon(together, [], learning_rate=1e-2, adam_learning_rate=1e-2)
    separate = [
        NorMuon([parameter], [], learning_rate=1e-2, adam_learning_rate=1e-2) for parameter in apart
    ]

    generator = torch.Generator().manual_seed(14)
    for _ in range(2):
        for index, shape in enumerate(shapes):
            gradient = torch.randn(shape, generator=generator)
            together[index].grad = gradient
            apart[index].grad = gradient.clone()
        grouped.step()
        for optimizer in separate:
            optimizer.step()

    for index in range(len(shapes)):
        torch.testing.assert_close(together[index], apart[index], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("optimizer_kind", ["normuon", "adamw"])
def test_critic_head_rate_changes_only_readout_updates_and_resumes(optimizer_kind) -> None:
    from kaggriculture.ppo import _optimizer_step
    from kaggriculture.structured import StructuredActor, StructuredCritic

    model_config = StructuredConfig(
        model_dim=16, attention_heads=2, ffn_multiplier=1, latents=4, core_layers=1
    )
    actor = StructuredActor(model_config)
    critic = StructuredCritic(model_config)
    reference = copy.deepcopy(critic)
    config = PpoConfig(
        optimizer=optimizer_kind, critic_head_learning_rate=8.75e-5, lr_warmup_steps=2
    )
    _, optimizer = make_optimizers(actor, critic, config)
    _, reference_optimizer = make_optimizers(
        actor, reference, PpoConfig(optimizer=optimizer_kind, lr_warmup_steps=2)
    )

    def step(model, selected_optimizer):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        _optimizer_step(selected_optimizer, config.critic_learning_rate, config.lr_warmup_steps)

    step(critic, optimizer)
    step(reference, reference_optimizer)
    for (name, parameter), (_, expected) in zip(
        critic.named_parameters(), reference.named_parameters(), strict=True
    ):
        if not name.startswith("value_head."):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    # The readout starts at zero; its first Adam step includes half-rate warmup.
    torch.testing.assert_close(
        critic.value_head.weight,
        torch.full_like(critic.value_head.weight, -8.75e-5 / 2),
        rtol=1e-4,
        atol=1e-8,
    )
    assert not torch.equal(critic.value_head.weight, reference.value_head.weight)

    resumed = copy.deepcopy(critic)
    _, resumed_optimizer = make_optimizers(actor, resumed, config)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(critic, optimizer)
    step(resumed, resumed_optimizer)
    for parameter, expected in zip(critic.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        critic.value_head.weight,
        torch.full_like(critic.value_head.weight, -8.75e-5 * 1.5),
        rtol=1e-4,
        atol=1e-8,
    )
