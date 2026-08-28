"""NorMuon: spectrally normalized matrix updates with a low-rank second moment.

Ported from `modded-nanogpt`'s `NorMuonAndAdam`, keeping the parts that are
about optimization and dropping the parts that are about pretraining a large
language model across eight GPUs.

What is kept, and why each piece is here:

  * **Polar Express** orthogonalization (arXiv 2505.16932).  Muon replaces
    Adam's element-wise adaptive scaling with the polar factor of the
    momentum-averaged gradient: for `M = U S V^T` the update direction is
    `U V^T`, which keeps the singular DIRECTIONS and discards the singular
    VALUES.  Polar Express is a fixed five-step odd-polynomial iteration that
    approximates that factor without an SVD.  The coefficients are
    `modded-nanogpt`'s, computed for five iterations at safety factor 2e-2 and
    cushion 2, and are meaningless if the iteration count changes.  We run the
    iteration in fp32 rather than the reference's bf16; `polar_express` records
    the measurement behind that.

  * **Nesterov momentum** in fp32 ahead of the orthogonalization, the standard
    Muon formulation.

  * **NorMuon's low-rank second moment** (arXiv 2510.05491).  Plain Muon leaves
    the orthogonalized update's rows at wildly different scales; NorMuon keeps
    an Adafactor-style row (or column) second moment and equalizes them, then
    rescales so the matrix's Frobenius norm is exactly what plain Muon would
    have applied.  It is variance reduction that cannot change the step's size.

  * **The shape learning-rate multiplier** `max(1, rows/cols) ** 0.5`, which
    makes a step's effect on the layer's output independent of its aspect
    ratio.

  * **Cautious weight decay.**  The decay term joins the update only where the
    update and the parameter already share a sign -- where the gradient step is
    itself shrinking that weight -- so decay never opposes the gradient.  The
    reference scales it quadratically in the rate, `lr^2 * wd` for Adam and
    `lr^2 * lr_mul * wd` for matrices, which is why it can carry `wd = 1.2` on
    matrices: at its 0.023 rate the per-step factor is 6.3e-4.  Off by default
    here, because a PPO trust region is a statement about the policy the update
    replays and decay moves weights the surrogate never asked to move.
    Pretraining is the case it was written for, and `train_bc.py` turns it on.

What is deliberately NOT ported:

  * **All communication.**  Replicated, sharded, and sparse gradient reduction,
    parameter banks, and the reduce/work orders exist to overlap eight ranks.
    We train on one GPU, so every one of those paths is dead weight here.

  * **bfloat16 parameters with mantissa tracking.**  That trick stores a bf16
    parameter beside a uint16 low half, so the forward reads half the bytes
    while the update keeps fp32 precision.  It buys bandwidth, and bandwidth is
    not what binds us: the actor holds 1.42M parameters (5.4 MiB fp32) and its
    forward was measured launch-gap bound, 4.9 ms of summed kernel time inside
    12.94 ms of wall clock.  Halving a 5.4 MiB read saves microseconds and costs
    two extra kernels plus bit manipulation on every parameter of every step.
    Parameters stay fp32; the forward keeps its bf16 autocast.

An important interaction with the rest of this trainer: Polar Express opens by
dividing its input by that input's Frobenius norm, so a NorMuon step is
invariant to any uniform rescaling of the gradient.  `max_gradient_norm` is
documented in `ppo.py` as clipping on 100% of updates, which makes Adam's
applied step `lr * g / ||g||` and its effective learning rate vary about
tenfold between minibatches.  For matrices under NorMuon that variation is
gone -- not clipped, but algebraically absent.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "ADAM_PARAMETER_ROLES",
    "POLAR_EXPRESS_COEFFICIENTS",
    "NorMuon",
    "polar_express",
    "route_parameters",
]


# Parameters whose input or output side is an index rather than a feature, plus
# the gates. `modded-nanogpt` keeps exactly these on Adam -- `embed`, `lm_head`,
# the value embeddings, the gates and scalars -- and gives the spectral
# treatment only to hidden matrices. The reason is that Muon's premise is a
# statement about a matrix acting on a feature space: an embedding table's rows
# are independent lookups, so orthogonalizing across them mixes unrelated
# directions, and a logit head's rows are per-action scores whose relative
# magnitudes ARE the output rather than an artifact of conditioning.
#
# Named by dotted-path component, so a rename shows up as a routing test
# failure rather than a silent demotion to Adam.
ADAM_PARAMETER_ROLES: frozenset[str] = frozenset(
    {
        # Learned queries and type embeddings: rows are lookups.
        "unit_slots",
        "market_queries",
        "token_types",
        # Output heads: rows are per-action logits.
        "unit_head",
        "market_kind",
        "market_quantity_value",
        "market_quantity_bias",
        "value_head",
        # Gate.
        "market_quantity_kind_gate",
        # Training-only transition embeddings: every row is an independent
        # action, slot, coordinate, type, or position lookup.
        "unit_action",
        "unit_slot",
        "unit_row",
        "unit_column",
        "unit_active",
        "market_quantity",
        "market_slot",
        "action_type",
        "type_identity",
        "position_identity",
    }
)


def route_parameters(module: torch.nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    """Split a module's parameters into the NorMuon and Adam sets.

    Everything with two or more dimensions is a matrix for NorMuon unless its
    name names a role in `ADAM_PARAMETER_ROLES`; everything one-dimensional --
    normalization gains, biases, learned scalars -- goes to Adam, which is
    where Muon is not defined and not wanted.
    """

    matrices: list[Tensor] = []
    vectors: list[Tensor] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        role = ADAM_PARAMETER_ROLES.isdisjoint(name.split("."))
        if parameter.ndim >= 2 and role:
            matrices.append(parameter)
        else:
            vectors.append(parameter)
    return matrices, vectors


# Computed by `modded-nanogpt` for num_iters=5, safety_factor=2e-2, cushion=2.
# The tuple length IS the iteration count; a different length is a different
# polynomial and these coefficients no longer approximate the polar factor.
POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express(matrices: Tensor) -> Tensor:
    """Approximate the polar factor of each matrix in a batch.

    `matrices` is `(..., rows, columns)`.  The iteration is written for the
    wide orientation because it forms the smaller Gram matrix there, so a tall
    input is transposed on the way in and back on the way out; the polar factor
    of a transpose is the transpose of the polar factor, so that costs nothing
    but two views.

    The leading division makes the spectral norm at most one, which is what
    the coefficients assume; without it the polynomial diverges.  It also makes
    the whole function invariant to a uniform rescaling of its input, which is
    what removes gradient-clipping's tenfold effective-rate variation from
    every matrix that goes through here.

    **This runs in fp32 where the reference runs bf16, and that is measured,
    not preferred.**  The iteration is chaotic near degenerate singular values:
    the polar factor of a matrix with two close singular values is not unique,
    so an input perturbation the size of bf16's 8-bit mantissa moves the output
    a finite distance.  Feeding the same matrix scaled by 1000 -- which is
    algebraically the same problem, since the first line divides the scale back
    out -- moves a bf16 step by 4.7% to 15.7% of its own length across our
    shapes, against 3e-6 in fp32, and bf16 also scores *worse* against an exact
    float64 SVD polar factor (cos 0.970-0.975 versus 0.980).  The reference
    accepts that because its matrices are large enough for the weight read to
    be the binding cost.  Ours are at most 384x96 and the forward is launch-gap
    bound, so the halved read buys nothing and the noise costs reproducibility:
    two identical runs would take different steps.
    """

    if matrices.ndim < 2:
        raise ValueError("polar_express needs at least two dimensions")
    transposed = matrices.size(-2) > matrices.size(-1)
    x = matrices.mT if transposed else matrices
    x = x.float()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * (1.0 + 2e-2) + 1e-6)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = x @ x.mT
        # b*A + c*A@A, the odd polynomial's inner term.
        inner = torch.addmm if gram.ndim == 2 else torch.baddbmm
        combined = inner(gram, gram, gram, beta=b, alpha=c)
        x = a * x + combined @ x
    return x.mT if transposed else x


def _shape_learning_rate_multiplier(rows: int, columns: int) -> float:
    """Make a step's effect independent of the matrix's aspect ratio."""

    return max(1.0, rows / columns) ** 0.5


class NorMuon(torch.optim.Optimizer):
    """NorMuon for matrices, Adam for everything else, in one optimizer.

    Parameters are routed by `matrix_parameters` and `vector_parameters` rather
    than by inspecting shapes here, because the routing is a modelling
    decision: `modded-nanogpt` keeps embeddings and the output head on Adam,
    and only hidden matrices get the spectral treatment.

    A matrix parameter of more than two dimensions -- a convolution kernel --
    is flattened to `(out_channels, -1)`, the standard Muon convention.

    `found_inf`, the skip signal `GradScaler` and the fused optimizers use, is
    honoured device-side: a nonzero value leaves parameters, both moment
    buffers, and the step counter exactly as they were, without the host
    learning which way it went.  Gating is written with `torch.where` and not a
    multiply so that a non-finite gradient is never multiplied by zero, which
    would launder an infinity into a NaN.
    """

    # Asked by `update_ppo` instead of inferring gateability from `fused`.
    supports_found_inf = True

    #: The `GradScaler` skip protocol, set by the caller immediately around a
    #: `step()` and deleted afterwards so an absent skip stays distinguishable
    #: from a decided one. Declared here because it is a real part of this
    #: class's interface, not an attribute smuggled in from outside.
    found_inf: Tensor | None
    grad_scale: Tensor | None

    def __init__(
        self,
        matrix_parameters: Iterable[Tensor],
        vector_parameters: Iterable[Tensor],
        *,
        learning_rate: float,
        adam_learning_rate: float,
        momentum: float = 0.95,
        beta2: float = 0.9,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        adam_epsilon: float = 1e-10,
        weight_decay: float = 0.0,
        adam_weight_decay: float = 0.0,
    ) -> None:
        matrices = [parameter for parameter in matrix_parameters]
        vectors = [parameter for parameter in vector_parameters]
        for parameter in matrices:
            if parameter.ndim < 2:
                raise ValueError(
                    f"NorMuon needs at least two dimensions, got {tuple(parameter.shape)}"
                )
        for name, value in (
            ("learning rate", learning_rate),
            ("adam learning rate", adam_learning_rate),
            ("adam epsilon", adam_epsilon),
        ):
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (
            ("weight decay", weight_decay),
            ("adam weight decay", adam_weight_decay),
        ):
            # Written as a bounded interval so NaN is rejected by the same
            # comparison that rejects an infinity.
            if not 0.0 <= value < float("inf"):
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        for name, value in (
            ("momentum", momentum),
            ("beta2", beta2),
            ("adam beta1", adam_betas[0]),
            ("adam beta2", adam_betas[1]),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {value}")

        groups: list[dict[str, Any]] = []
        if matrices:
            groups.append(
                {
                    "params": matrices,
                    "kind": "normuon",
                    "lr": learning_rate,
                    "base_lr": learning_rate,
                    "warmup_step": 0,
                    "momentum": momentum,
                    "beta2": beta2,
                    "weight_decay": weight_decay,
                }
            )
        if vectors:
            groups.append(
                {
                    "params": vectors,
                    "kind": "adam",
                    "lr": adam_learning_rate,
                    "base_lr": adam_learning_rate,
                    "warmup_step": 0,
                    "betas": tuple(adam_betas),
                    "eps": adam_epsilon,
                    "weight_decay": adam_weight_decay,
                }
            )
        if not groups:
            raise ValueError("NorMuon needs at least one parameter")
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Read the skip signal the way the fused optimizers do, so
        # `_optimizer_step` can drive this class unchanged.
        found_inf: Tensor | None = getattr(self, "found_inf", None)
        for group in self.param_groups:
            if group["kind"] == "normuon":
                self._normuon_group(group, found_inf)
            else:
                self._adam_group(group, found_inf)
        return loss

    def _normuon_group(self, group: dict[str, Any], found_inf: Tensor | None) -> None:
        momentum = float(group["momentum"])
        beta2 = float(group["beta2"])
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            rows = parameter.shape[0]
            flat_parameter = parameter.view(rows, -1)
            flat_gradient = gradient.view(rows, -1).float()
            columns = flat_parameter.shape[1]
            state = self.state[parameter]
            if not state:
                state["momentum"] = torch.zeros_like(flat_parameter, dtype=torch.float32)
                reduced_dimension = -1 if rows >= columns else -2
                second_shape = (rows, 1) if reduced_dimension == -1 else (1, columns)
                state["second_moment"] = flat_parameter.new_zeros(second_shape, dtype=torch.float32)
                state["reduced_dimension"] = reduced_dimension
            buffer = state["momentum"]
            reduced_dimension = state["reduced_dimension"]

            if found_inf is None:
                blend = 1.0 - momentum
                safe_gradient = flat_gradient
            else:
                # A skipped minibatch must not move the buffer at all, and its
                # gradient may be non-finite, so select rather than scale.
                applied = found_inf == 0
                safe_gradient = torch.where(applied, flat_gradient, 0.0)
                blend = torch.where(applied, 1.0 - momentum, 0.0)
            buffer.lerp_(safe_gradient, blend)
            nesterov = safe_gradient.lerp(buffer, momentum)

            direction = polar_express(nesterov)
            direction = self._reduce_variance(
                direction, state["second_moment"], beta2, reduced_dimension, found_inf
            )
            step = learning_rate * _shape_learning_rate_multiplier(rows, columns)
            if found_inf is not None:
                step = torch.where(found_inf == 0, step, 0.0)
            if not weight_decay:
                flat_parameter.add_(direction.to(flat_parameter.dtype) * -step)
            else:
                # `step` carries both the shape multiplier and the skip gate, so
                # the reference's `lr^2 * lr_mul * wd` and its "a skipped
                # minibatch changes nothing" both follow from writing the decay
                # against it rather than against the bare rate.
                decay = weight_decay * learning_rate * step
                shrinking = (direction * flat_parameter) >= 0
                update = direction * step + flat_parameter * shrinking * decay
                flat_parameter.sub_(update.to(flat_parameter.dtype))

    @staticmethod
    def _reduce_variance(
        direction: Tensor,
        second_moment: Tensor,
        beta2: float,
        reduced_dimension: int,
        found_inf: Tensor | None,
    ) -> Tensor:
        """Equalize the orthogonalized update's rows without resizing the step.

        The three-line version of the algebra: `scale` normalizes each row by
        its running root-mean-square, and the ratio that follows restores the
        matrix's Frobenius norm to the value it had before that normalization.
        NorMuon therefore changes the update's DIRECTION only -- its length is
        whatever plain Muon would have applied.
        """

        mean_square = direction.float().square().mean(dim=reduced_dimension, keepdim=True)
        reduced_size = direction.size(reduced_dimension)
        norm_before = mean_square.sum(dim=(-2, -1), keepdim=True).mul(reduced_size).sqrt()
        blend = 1.0 - beta2 if found_inf is None else torch.where(found_inf == 0, 1.0 - beta2, 0.0)
        second_moment.lerp_(mean_square.to(second_moment.dtype), blend)
        scale = second_moment.clamp_min(1e-10).rsqrt()
        norm_after = (
            (mean_square * reduced_size)
            .mul(scale.float().square())
            .sum(dim=(-2, -1), keepdim=True)
            .sqrt()
        )
        return direction * (scale * (norm_before / norm_after.clamp_min(1e-10))).type_as(direction)

    def _adam_group(self, group: dict[str, Any], found_inf: Tensor | None) -> None:
        beta1, beta2 = group["betas"]
        epsilon = float(group["eps"])
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            state = self.state[parameter]
            if not state:
                state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            if found_inf is None:
                applied_scalar: Tensor | float = 1.0
                safe_gradient = gradient.float()
            else:
                applied = found_inf == 0
                applied_scalar = applied.to(state["step"].dtype)
                safe_gradient = torch.where(applied, gradient.float(), 0.0)
            # The step counter is device-side so a skipped minibatch leaves
            # bias correction where it was without a host read.
            state["step"] += applied_scalar
            count = state["step"]
            first, second = state["exp_avg"], state["exp_avg_sq"]
            blend1 = (1.0 - beta1) * applied_scalar
            blend2 = (1.0 - beta2) * applied_scalar
            first.lerp_(safe_gradient, blend1)
            second.lerp_(safe_gradient.square(), blend2)
            bias1 = 1.0 - beta1**count
            bias2 = 1.0 - beta2**count
            # A never-stepped parameter has zero moments and zero corrections;
            # clamping the denominators keeps that case at an exact no-op.
            update = (first / bias1.clamp_min(1e-12)) / (
                (second / bias2.clamp_min(1e-12)).sqrt() + epsilon
            )
            step = learning_rate if found_inf is None else learning_rate * applied_scalar
            if not weight_decay:
                parameter.add_((update * -step).to(parameter.dtype))
            else:
                # Quadratic in the rate here too, which is why the reference's
                # 0.005 bites only on the tables it gives a large `lr_mul`.
                decay = weight_decay * learning_rate * step
                shrinking = (update * parameter) > 0
                decayed = update * step + parameter * shrinking * decay
                parameter.sub_(decayed.to(parameter.dtype))
