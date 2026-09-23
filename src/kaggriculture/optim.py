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
    iteration in float32 tensors; backend matmul precision remains controlled
    by the caller.

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
invariant to any uniform rescaling of the gradient. PPO therefore leaves the
policy and critic gradients unclipped and applies `nextlat_max_gradient_norm`
only to the actor-side and critic-side NextLat predictors. The categorical
value head remains Adam-managed; unlike predictor parameters, it has no
auxiliary gradient safeguard to clip.

That invariance covers the matrices and stops there, which is why Adam's epsilon
is 1e-20 here and not `modded-nanogpt`'s 1e-10. Epsilon is only negligible
against the second moments a trainer actually produces, and a PPO surrogate's
are nothing like a language model's. Measured on the actor optimizer state of
`runs/production-value-softcap-p100-20260910`, per-element `sqrt(v_hat)` at the
tenth percentile is 1.9e-9 and `market_quantity_bias` sits at a median of
1.07e-11 by wave 100 -- two orders BELOW a 1e-10 epsilon. Those elements are not
Adam-stepped at all: their update is `m_hat / eps`, proportional to the gradient
instead of normalized by it, so their effective learning rate is the advantage
scale. It tightens as the critic fits and advantages shrink:
`market_quantity_bias` runs 0.69 -> 0.59 -> 0.42 -> 0.38 mean
`sqrt(v_hat)/(sqrt(v_hat) + eps)` across waves 41, 59, 78 and 100, and the count
of Adam parameters under 0.95 goes 1 -> 2 -> 5 -> 7 over the same waves. The
rarely-sampled action rows are hit hardest and first, which makes an action's
disappearance self-sealing: sampled less, smaller second moment, more epsilon
attenuation, updated less. The ratio `m_hat / sqrt(v_hat)` is bounded near one
for any decaying-moment schedule, and a parameter with no gradient history has
both moments exactly zero, so shrinking epsilon to 1e-20 changes nothing except
removing that floor.
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


# Lookup rows and output-head rows are not feature-to-feature maps. Follow the
# reference's separation of embeddings/readouts from hidden matrix projections.
# Embedding weights are recognized by module ownership below, including tied
# weights. Raw learned queries and output heads need explicit role names.
ADAM_PARAMETER_ROLES: frozenset[str] = frozenset(
    {
        # Learned queries stored directly as Parameters rather than Embeddings.
        "opponent_queries",
        "latent_queries",
        "value_query",
        # Output heads: rows are per-action logits.
        "unit_head",
        "market_kind",
        "market_quantity_bias",
        "value_head",
        "reward_head",
        "forecast_head",
        "plan_head",
    }
)


#: Attribute a module sets to declare Adam rate multipliers for parameters it
#: deliberately initializes away from unit RMS. Maps its own attribute name to
#: the multiplier.
_MULTIPLIER_ATTRIBUTE = "adam_learning_rate_multipliers"


def _declared_multipliers(module: torch.nn.Module) -> dict[int, float]:
    """Collect the Adam rate multipliers the module tree declares.

    Adam's step is an ABSOLUTE per-element displacement, so two parameters at
    different scales under one rate move by entirely different fractions of
    themselves. Neither reference leaves that to chance: `modded-nanogpt`
    carries an explicit `lr_mul` on every Adam-routed role in its parameter
    table (`train_gpt.py:2026-2050`, 0.01 on `smear_gate` through 75 on the
    embedding tables), and `../NextLat` exposes the same idea as the
    `_get_param_lr_overrides` hook (`models/model_base.py:150-159`). This
    trainer had no such mechanism, and `trunk.opponent_queries` at 0.02 RMS was
    taking fifty times the relative step of the unit-RMS query banks beside it.
    """

    multipliers: dict[int, float] = {}
    for child in module.modules():
        declared = getattr(child, _MULTIPLIER_ATTRIBUTE, None)
        if declared is None:
            continue
        for name, multiplier in declared.items():
            parameter = getattr(child, name, None)
            if not isinstance(parameter, torch.nn.Parameter):
                raise ValueError(
                    f"{type(child).__name__}.{name} is not a parameter but declares a "
                    "learning-rate multiplier"
                )
            value = float(multiplier)
            if not 0.0 < value < float("inf"):
                raise ValueError(
                    f"{type(child).__name__}.{name} declares a non-positive or non-finite "
                    f"learning-rate multiplier {multiplier}"
                )
            multipliers[id(parameter)] = value
    return multipliers


def route_parameters(
    module: torch.nn.Module,
    *,
    exclude: Iterable[Tensor] = (),
) -> tuple[list[Tensor], list[Tensor], list[float]]:
    """Split a module's parameters into the NorMuon and Adam sets.

    Embedding weights use Adam regardless of module names or weight sharing.
    Named lookup/readout roles and one-dimensional gains, biases, and gates
    also use Adam. Other parameters with two or more dimensions use NorMuon.

    The third result is each Adam parameter's rate multiplier, one per entry of
    the second. A NorMuon step is already invariant to the matrix's scale --
    Polar Express divides by the input's Frobenius norm -- so a multiplier on a
    matrix would describe nothing, and declaring one is an error.

    `exclude` drops named parameters from the routing without changing how the
    rest are routed. It exists for the `lejepa` family, whose actor carries the
    shared world-model backbone as a submodule so that one flat state dict still
    serializes the deployed model, while a different optimizer owns it. Routing
    is by parameter name and module type, so a subset of one module routes
    exactly as it would have inside it -- which is the property that makes the
    split safe to state here rather than by rebuilding the module tree.
    """

    excluded = {id(parameter) for parameter in exclude}
    embedding_parameters = {
        id(child.weight) for child in module.modules() if isinstance(child, torch.nn.Embedding)
    }
    declared = _declared_multipliers(module)
    matrices: list[Tensor] = []
    vectors: list[Tensor] = []
    multipliers: list[float] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or id(parameter) in excluded:
            continue
        role = ADAM_PARAMETER_ROLES.isdisjoint(name.split("."))
        if parameter.ndim >= 2 and id(parameter) not in embedding_parameters and role:
            if id(parameter) in declared:
                raise ValueError(
                    f"{name} routes to NorMuon, whose step is scale-invariant, and cannot "
                    "carry an Adam learning-rate multiplier"
                )
            matrices.append(parameter)
        else:
            vectors.append(parameter)
            multipliers.append(declared.get(id(parameter), 1.0))
    return matrices, vectors, multipliers


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


def _polar_express_wide_batch(matrices: Tensor) -> Tensor:
    """Evaluate the fixed polynomial for a rank-3 batch in wide orientation."""
    x = matrices.float()
    denominator = x.norm(dim=(-2, -1), keepdim=True) * (1.0 + 2e-2)
    x = x / torch.where(denominator == 0, 1.0, denominator)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = torch.bmm(x, x.mT)
        combined = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = a * x + torch.bmm(combined, x)
    return x


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

    The polynomial uses float32 tensors; CUDA multiplication accuracy still
    follows PyTorch's process-wide matmul precision setting. Normalization
    guards only a zero denominator: adding a fixed epsilon changes the spectrum
    at the small momentum magnitudes PPO actually produces, making the update
    depend on gradient scale again.
    """

    if matrices.ndim < 2:
        raise ValueError("polar_express needs at least two dimensions")
    transposed = matrices.size(-2) > matrices.size(-1)
    oriented = matrices.mT if transposed else matrices
    leading_shape = oriented.shape[:-2]
    rows, columns = oriented.shape[-2:]
    batched = oriented.reshape(-1, rows, columns)
    result = _polar_express_wide_batch(batched).reshape(*leading_shape, rows, columns)
    return result.mT if transposed else result


# The optimizer owns many distinct matrix shapes. Compile one rank-3,
# dynamically-shaped wide-orientation kernel rather than specializing the public
# wrapper: static specialization exhausted Dynamo's eight-entry recompile cache
# on the production actor before its first optimizer step. Keeping rank and
# orientation outside the graph leaves only matrix extents dynamic.
_polar_express_compiled = torch.compile(_polar_express_wide_batch, dynamic=True, fullgraph=True)


def _polar_factor(matrices: Tensor) -> Tensor:
    if not matrices.is_cuda:
        return polar_express(matrices)
    if matrices.ndim < 2:
        raise ValueError("polar_express needs at least two dimensions")
    transposed = matrices.size(-2) > matrices.size(-1)
    oriented = matrices.mT if transposed else matrices
    leading_shape = oriented.shape[:-2]
    rows, columns = oriented.shape[-2:]
    batched = oriented.reshape(-1, rows, columns)
    result = _polar_express_compiled(batched).reshape(*leading_shape, rows, columns)
    return result.mT if transposed else result


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
    learning which way it went. Matrix gradients are selected before arithmetic;
    compatible Adam groups use the native fused kernel's early-return gate.
    Neither path multiplies a non-finite gradient by zero.
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
        vector_learning_rate_multipliers: Iterable[float] | None = None,
        *,
        learning_rate: float,
        adam_learning_rate: float,
        momentum: float = 0.95,
        beta2: float = 0.9,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        adam_epsilon: float = 1e-20,
        weight_decay: float = 0.0,
        adam_weight_decay: float = 0.0,
    ) -> None:
        matrices = [parameter for parameter in matrix_parameters]
        vectors = [parameter for parameter in vector_parameters]
        multipliers = (
            [1.0] * len(vectors)
            if vector_learning_rate_multipliers is None
            else [float(multiplier) for multiplier in vector_learning_rate_multipliers]
        )
        if len(multipliers) != len(vectors):
            raise ValueError(
                f"got {len(multipliers)} learning-rate multipliers for {len(vectors)} "
                "Adam parameters"
            )
        for multiplier in multipliers:
            if not 0.0 < multiplier < float("inf"):
                raise ValueError(
                    f"adam learning rate multiplier must be positive and finite, got {multiplier}"
                )
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
        # One group per distinct multiplier, which is how the reference's
        # `lr_mul` column resolves too. Descending order keeps the shared rate
        # -- the group holding the value head and every gain and bias -- first.
        by_multiplier: dict[float, list[Tensor]] = {}
        for parameter, multiplier in zip(vectors, multipliers, strict=True):
            by_multiplier.setdefault(multiplier, []).append(parameter)
        for multiplier, members in sorted(by_multiplier.items(), reverse=True):
            rate = adam_learning_rate * multiplier
            groups.append(
                {
                    "params": members,
                    "kind": "adam",
                    "lr": rate,
                    "base_lr": rate,
                    "lr_multiplier": multiplier,
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
        """Step every matrix in the group in as few launches as its shapes allow.

        The arithmetic is exactly the per-parameter form this replaces; only the
        launch structure differs, and it has to. The model's matrices are tiny --
        the largest is 320x80 -- so the ~28 kernels one `polar_express` call
        issues each spend longer being launched than doing work, and a
        parameter-at-a-time loop over the actor's 110 matrices costs roughly
        4800 launches for a step the optimizer takes 342 times an iteration.
        Nothing had ever measured that: the update profilers time fused AdamW,
        while production runs this. It was 12.24 s of a 38.9 s update, the
        single largest line item after the forward and backward themselves.

        The shapes are what make it collapse. The production actor's 110
        matrices occupy 19 distinct shapes, 85 of them in four groups, so one
        batched Polar Express per shape replaces up to 47 sequential ones. On an
        RTX 5090, production actor and critic (`probe_optimizer_ab`):

            step        reference    batched   speedup   per iteration
            actor       34.087 ms   6.582 ms     5.18x   3.886 -> 0.750 s
            critic      36.650 ms   8.850 ms     4.14x   8.356 -> 2.018 s

        Gated steps pack gradients into the same shape batches the polar factor
        needs anyway. One selection per shape replaces one per parameter, and
        the packed storage becomes the Nesterov batch in place.

        Two properties make the batching exact rather than approximate.
        `polar_express` normalizes and iterates per matrix over the trailing two
        dimensions, so stacking same-shaped matrices into a leading batch
        dimension computes each one's polar factor independently.
        `_reduce_variance` reduces over `reduced_dimension` with `keepdim` and
        sums only over the trailing two, so it is batch-safe for the same
        reason -- and `reduced_dimension` follows from the shape, so it is
        necessarily uniform within a shape group.
        """

        momentum = float(group["momentum"])
        beta2 = float(group["beta2"])
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])

        flat_parameters: list[Tensor] = []
        gradients: list[Tensor] = []
        buffers: list[Tensor] = []
        second_moments: list[Tensor] = []
        reduced_dimensions: list[int] = []
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            rows = parameter.shape[0]
            flat_parameter = parameter.view(rows, -1)
            columns = flat_parameter.shape[1]
            state = self.state[parameter]
            if not state:
                state["momentum"] = torch.zeros_like(flat_parameter, dtype=torch.float32)
                reduced_dimension = -1 if rows >= columns else -2
                second_shape = (rows, 1) if reduced_dimension == -1 else (1, columns)
                state["second_moment"] = flat_parameter.new_zeros(second_shape, dtype=torch.float32)
                state["reduced_dimension"] = reduced_dimension
            flat_parameters.append(flat_parameter)
            gradients.append(gradient.view(rows, -1).float())
            buffers.append(state["momentum"])
            second_moments.append(state["second_moment"])
            reduced_dimensions.append(state["reduced_dimension"])
        if not flat_parameters:
            return

        shape_groups: dict[tuple[int, int], list[int]] = {}
        for index, flat_parameter in enumerate(flat_parameters):
            key = (flat_parameter.shape[0], flat_parameter.shape[1])
            shape_groups.setdefault(key, []).append(index)
        packed_nesterovs: dict[tuple[int, int], Tensor] = {}

        if found_inf is None:
            blend: Tensor | float = 1.0 - momentum
            torch._foreach_lerp_(buffers, gradients, blend)
            nesterovs = torch._foreach_lerp(gradients, buffers, momentum)
        else:
            # A skipped minibatch must not move the buffer at all, and its
            # gradient may be non-finite, so select rather than scale: scaling
            # would turn an infinity into a NaN the buffer then keeps forever.
            applied = found_inf == 0
            skipped = ~applied
            blend = torch.where(applied, 1.0 - momentum, 0.0)
            safe_by_index: dict[int, Tensor] = {}
            for shape, members in shape_groups.items():
                if len(members) == 1:
                    # A single selection already owns its output; stacking a
                    # singleton first would add a copy with nothing to amortize.
                    packed = torch.where(applied, gradients[members[0]], 0.0).unsqueeze(0)
                else:
                    packed = torch.stack([gradients[index] for index in members])
                    packed.masked_fill_(skipped, 0.0)
                packed_nesterovs[shape] = packed
                for index, gradient in zip(members, packed.unbind(0), strict=True):
                    safe_by_index[index] = gradient
            nesterovs = [safe_by_index[index] for index in range(len(gradients))]
            torch._foreach_lerp_(buffers, nesterovs, blend)
            # These views own packed copies, never parameter.grad storage.
            torch._foreach_lerp_(nesterovs, buffers, momentum)

        reduced: dict[int, Tensor] = {}
        for shape, members in shape_groups.items():
            reduced_dimension = reduced_dimensions[members[0]]
            if len(members) == 1:
                index = members[0]
                reduced[index] = self._reduce_variance(
                    _polar_factor(nesterovs[index]),
                    second_moments[index],
                    beta2,
                    reduced_dimension,
                    found_inf,
                )
                continue
            # `_reduce_variance` advances `second_moment` in place, so the
            # stacked copy has to be written back; one `_foreach_copy_` returns
            # the whole group's running estimates to their own state entries.
            stacked_second = torch.stack([second_moments[index] for index in members])
            group_directions = self._reduce_variance(
                _polar_factor(
                    packed_nesterovs[shape]
                    if found_inf is not None
                    else torch.stack([nesterovs[index] for index in members])
                ),
                stacked_second,
                beta2,
                reduced_dimension,
                found_inf,
            )
            torch._foreach_copy_(
                [second_moments[index] for index in members], list(stacked_second.unbind(0))
            )
            for offset, index in enumerate(members):
                reduced[index] = group_directions[offset]
        directions = [reduced[index] for index in range(len(flat_parameters))]

        steps = [
            learning_rate * _shape_learning_rate_multiplier(*flat_parameter.shape)
            for flat_parameter in flat_parameters
        ]
        updates = torch._foreach_mul(directions, steps)
        if weight_decay:
            for flat_parameter, direction, update, step in zip(
                flat_parameters, directions, updates, steps, strict=True
            ):
                # `step` carries the shape multiplier, and the skip gate below
                # scales the finished update, so the reference's
                # `lr^2 * lr_mul * wd` and its "a skipped minibatch changes
                # nothing" both still follow from the decay written against it.
                decay = weight_decay * learning_rate * step
                shrinking = (direction * flat_parameter) >= 0
                update.add_(flat_parameter * shrinking * decay)
        if found_inf is not None:
            # Gating the finished update rather than each shape's step keeps the
            # skip one launch instead of one per parameter, and it is exact:
            # every term here is finite even on a skipped step, because the
            # direction descends from the untouched buffer, not the gradient.
            torch._foreach_mul_(updates, (found_inf == 0).to(updates[0].dtype))
        torch._foreach_sub_(flat_parameters, updates)

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
        """Use native fused Adam for the production FP32, no-decay CUDA groups.

        The kernel consumes one device-side counter per parameter and fuses
        moments, bias corrections and updates in one multi-tensor operation.
        The only ungated companion operation advances those counters. Native Adam's
        decay is not our quadratic cautious decay, so that branch retains its
        reference arithmetic, as do CPU and incompatible dtype/layout groups.

        No optimizer object or compiled graph wraps this call: warmup changes
        the scalar rate without recompilation, and the existing state tensors
        remain the complete checkpoint representation.
        """

        beta1, beta2 = group["betas"]
        epsilon = float(group["eps"])
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])

        parameters: list[Tensor] = []
        gradients: list[Tensor] = []
        counts: list[Tensor] = []
        firsts: list[Tensor] = []
        seconds: list[Tensor] = []
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            state = self.state[parameter]
            if not state:
                state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            parameters.append(parameter)
            gradients.append(gradient)
            counts.append(state["step"])
            firsts.append(state["exp_avg"])
            seconds.append(state["exp_avg_sq"])
        if not parameters:
            return

        device = parameters[0].device
        if (
            not weight_decay
            and device.type == "cuda"
            # Native bias corrections are not clamped. A beta rounding to one
            # in FP32 needs the reference's denominator floor instead.
            and beta1 < 1.0 - 2.0**-25
            and beta2 < 1.0 - 2.0**-25
            and all(
                tensor.device == device and tensor.dtype == torch.float32 and tensor.is_contiguous()
                for tensors in (parameters, gradients, firsts, seconds, counts)
                for tensor in tensors
            )
            and all(count.ndim == 0 for count in counts)
            and (found_inf is None or (found_inf.device == device and found_inf.numel() == 1))
        ):
            native_found_inf = None
            increment: Tensor | float = 1.0
            if found_inf is not None:
                # The native kernel only skips exactly 1, whereas our public
                # gate skips every nonzero value (including NaN). Normalize on
                # device, preserving the caller's gate and gradient storage.
                native_found_inf = (found_inf != 0).to(dtype=torch.float32)
                increment = 1.0 - native_found_inf
            # Unlike torch.optim's increment-and-rollback, adding zero on a
            # skip also preserves counters at the FP32 integer precision limit.
            torch._foreach_add_(counts, increment)
            torch._fused_adam_(
                parameters,
                gradients,
                firsts,
                seconds,
                [],
                counts,
                amsgrad=False,
                lr=learning_rate,
                beta1=beta1,
                beta2=beta2,
                weight_decay=0.0,
                eps=epsilon,
                maximize=False,
                grad_scale=None,
                found_inf=native_found_inf,
            )
            return

        if found_inf is None:
            applied_scalar: Tensor | float = 1.0
            safe_gradients = [gradient.float() for gradient in gradients]
        else:
            applied = found_inf == 0
            applied_scalar = applied.to(counts[0].dtype)
            safe_gradients = [torch.where(applied, gradient.float(), 0.0) for gradient in gradients]
        # The step counter is device-side so a skipped minibatch leaves bias
        # correction where it was without a host read.
        torch._foreach_add_(counts, applied_scalar)
        blend1 = (1.0 - beta1) * applied_scalar
        blend2 = (1.0 - beta2) * applied_scalar
        torch._foreach_lerp_(firsts, safe_gradients, blend1)
        torch._foreach_lerp_(seconds, torch._foreach_mul(safe_gradients, safe_gradients), blend2)

        # Every count is zero-dimensional, so the corrections themselves batch
        # into single kernels even though applying them to the moments cannot.
        bias1 = torch._foreach_pow(beta1, counts)
        torch._foreach_neg_(bias1)
        torch._foreach_add_(bias1, 1.0)
        bias2 = torch._foreach_pow(beta2, counts)
        torch._foreach_neg_(bias2)
        torch._foreach_add_(bias2, 1.0)
        # A never-stepped parameter has zero moments and zero corrections;
        # clamping the denominators keeps that case at an exact no-op.
        torch._foreach_clamp_min_(bias1, 1e-12)
        torch._foreach_clamp_min_(bias2, 1e-12)
        denominators = torch._foreach_div(seconds, bias2)
        torch._foreach_sqrt_(denominators)
        torch._foreach_add_(denominators, epsilon)

        updates = torch._foreach_div(firsts, bias1)
        torch._foreach_div_(updates, denominators)
        step = learning_rate if found_inf is None else learning_rate * applied_scalar
        if not weight_decay:
            torch._foreach_mul_(updates, step)
            torch._foreach_sub_(parameters, updates)
            return
        # Quadratic in the rate here too, which is why the reference's 0.005
        # bites only on the tables it gives a large `lr_mul`.
        decay = weight_decay * learning_rate * step
        # The cautious mask and the decay both read the pre-step parameter and
        # the unscaled update, so the addends are formed before the scaling.
        # Adding them as a separate term rather than folding the decay into the
        # update keeps this branch bit-identical to the one above wherever the
        # mask is false, which is what makes "cautious decay changes nothing
        # where it should not" an exact statement rather than an approximate one.
        addends = [
            ((update * parameter) > 0) * parameter * decay
            for parameter, update in zip(parameters, updates, strict=True)
        ]
        torch._foreach_mul_(updates, step)
        torch._foreach_add_(updates, addends)
        torch._foreach_sub_(parameters, updates)
