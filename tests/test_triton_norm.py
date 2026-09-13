"""Contracts for the opaque CUDA RMS normalization boundary."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kaggriculture.triton_norm import _rms_norm, rms_norm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def _isolated_normalization_compilations():
    # Dtype, strides, vmap and empty rows intentionally require different graphs.
    torch._dynamo.reset_code(rms_norm.__code__)
    yield
    torch._dynamo.reset_code(rms_norm.__code__)


def _reference(inputs: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    values = inputs.double()
    normalized = values * (values.square().mean(-1, keepdim=True) + eps).rsqrt()
    if weight is not None:
        normalized = normalized * weight.double()
    return normalized.to(inputs.dtype)


def _assert_close(actual: Tensor, expected: Tensor) -> None:
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("affine", [False, True])
def test_compiled_forward_and_backward_match_high_precision(
    input_dtype: torch.dtype, weight_dtype: torch.dtype, affine: bool
) -> None:
    torch.manual_seed(91)
    # Non-power-of-two width exercises the masked reduction. Leading strides
    # cannot be flattened to rows without a copy by a reshape-based kernel.
    inputs = torch.randn(3, 5, 37, device="cuda", dtype=input_dtype).transpose(0, 1)
    inputs.requires_grad_()
    weight = (
        torch.randn(37, device="cuda", dtype=weight_dtype, requires_grad=True) if affine else None
    )
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_() if weight is not None else None
    gradient = torch.randn_like(inputs)

    compiled = torch.compile(rms_norm, fullgraph=True)
    actual = compiled(inputs, weight, 1e-5)
    expected = _reference(reference_inputs, reference_weight, 1e-5)
    assert actual.is_contiguous()
    _assert_close(actual, expected)
    actual.backward(gradient)
    expected.backward(gradient)
    _assert_close(inputs.grad, reference_inputs.grad)
    if weight is not None:
        _assert_close(weight.grad, reference_weight.grad)


@pytest.mark.parametrize("layout", ["expanded", "column_slice", "column_transpose", "vector"])
def test_strides_are_respected_without_changing_output_layout(layout: str) -> None:
    torch.manual_seed(92)
    if layout == "expanded":
        inputs = torch.randn(1, 4, 19, device="cuda").expand(3, -1, -1)
    elif layout == "column_slice":
        inputs = torch.randn(3, 4, 38, device="cuda")[..., 1::2]
    elif layout == "column_transpose":
        inputs = torch.randn(3, 19, 4, device="cuda").transpose(-1, -2)
    else:
        inputs = torch.randn(38, device="cuda")[1::2]
    inputs.requires_grad_()
    weight = torch.randn(38, device="cuda")[1::2].requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    actual = rms_norm(inputs, weight, 0.03)
    expected = _reference(reference_inputs, reference_weight, 0.03)
    assert actual.is_contiguous()
    _assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    _assert_close(inputs.grad, reference_inputs.grad)
    _assert_close(weight.grad, reference_weight.grad)


@pytest.mark.parametrize("mapping", ["both", "inputs", "weights", "no_affine"])
def test_compiled_vmap_preserves_distinct_lanes_and_shared_gradients(mapping: str) -> None:
    torch.manual_seed(93)
    # Nonleading mapped dimensions and distinct affine lanes expose an
    # incorrect row-to-weight offset or accidental summation across actors.
    # Eleven rows per lane cross a packed CTA boundary, leaving one final row.
    inputs = torch.randn(11, 3, 23, device="cuda", requires_grad=True)
    weight = torch.randn(23, 3, device="cuda", requires_grad=True)
    input_dim = 1
    weight_dim = 1
    if mapping == "inputs":
        weight = weight[:, 0].detach().requires_grad_()
        weight_dim = None
    elif mapping == "weights":
        inputs = inputs[:, 0].detach().requires_grad_()
        input_dim = None
    elif mapping == "no_affine":
        weight = None
        weight_dim = None
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_() if weight is not None else None
    mapped = torch.vmap(rms_norm, in_dims=(input_dim, weight_dim, None))
    compiled = torch.compile(mapped, fullgraph=True)
    actual = compiled(inputs, weight, 1e-4)
    expected = torch.stack(
        [
            _reference(
                reference_inputs.select(input_dim, lane)
                if input_dim is not None
                else reference_inputs,
                reference_weight.select(weight_dim, lane)
                if weight_dim is not None
                else reference_weight,
                1e-4,
            )
            for lane in range(3)
        ]
    )
    _assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    _assert_close(inputs.grad, reference_inputs.grad)
    if weight is not None:
        _assert_close(weight.grad, reference_weight.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [20, 80, 320])
def test_compiled_rows_are_bitwise_independent_of_batch_and_autograd(
    dtype: torch.dtype, width: int
) -> None:
    torch.manual_seed(94)
    # Repetition shifts rows between positions in full and partial packed CTAs.
    rows = torch.randn(5, 7, width, device="cuda", dtype=dtype)
    weight = torch.randn(width, device="cuda", dtype=dtype, requires_grad=True)
    compiled = torch.compile(rms_norm, fullgraph=True)
    with torch.no_grad():
        baseline = compiled(rows, weight, 1e-5)
        _assert_close(baseline, _reference(rows, weight, 1e-5))
        repeated = compiled(rows.repeat(16, 1, 1), weight, 1e-5)
    training_rows = rows.detach().requires_grad_()
    training = compiled(training_rows, weight, 1e-5)
    assert torch.equal(baseline, training)
    assert torch.equal(baseline.repeat(16, 1, 1), repeated)
    # Exercise the compiled backward as well as its forward specialization.
    training.sum().backward()


@pytest.mark.parametrize("affine", [False, True])
def test_custom_op_schema_fake_layout_and_autograd(affine: bool) -> None:
    inputs = torch.randn(2, 3, 17, device="cuda").transpose(0, 1).requires_grad_()
    weight = torch.randn(17, device="cuda", requires_grad=True) if affine else None
    _output, inverse = _rms_norm(inputs, weight, 1e-5)
    assert not inverse.requires_grad
    torch.library.opcheck(_rms_norm, (inputs, weight, 1e-5))


def test_empty_leading_dimension_has_well_defined_affine_gradient() -> None:
    inputs = torch.empty(0, 19, device="cuda", requires_grad=True)
    weight = torch.randn(19, device="cuda", requires_grad=True)
    output = torch.compile(rms_norm, fullgraph=True)(inputs, weight, 1e-5)
    assert output.shape == inputs.shape
    output.sum().backward()
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
