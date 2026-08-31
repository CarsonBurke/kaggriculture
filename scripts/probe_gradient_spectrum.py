#!/usr/bin/env python3
"""Measure whether spectral orthogonalization improves this system's real gradients.

Muon replaces Adam's element-wise adaptive scaling with an approximate spectral
normalization of the momentum matrix: for a 2D weight whose momentum-averaged
gradient is `M = U S V^T`, the update direction is the polar factor
`P(M) = U V^T`, which keeps the singular DIRECTIONS and discards the singular
VALUES.  Whether that helps is not a matter of taste.  It is a property of the
gradient's per-direction signal-to-noise ratio, and the published RL results
split along exactly that line: "When Does Muon Help Agentic Reinforcement
Learning?" (arXiv 2607.16169v1) reports Muon FAILING under single-turn RLVR with
episode-level outcome-only advantages and HELPING under GiGPO's dense
step-level credit, and its Equation 4 gives the criterion.  Writing a noisy
gradient as `Ghat = Gstar + E` with per-singular-direction signal `s_i` and
noise `sigma_i`,

    E<P(Ghat), Gstar>_F = sum_i s_i * (2 * Phi(s_i / sigma_i) - 1)

so flattening the spectrum pays only to the extent that the WEAK directions'
SIGNS are reliable.  That paper measures neither gradient SNR nor update
spectra; it names both as open work.  This script measures them here.

What is measured, on the real update path and nothing else:

  * `Gstar`, the full-batch policy-loss gradient over one staged rollout,
    accumulated through `_actor_minibatch_terms` at production minibatch size,
    autocast and compile mode, with behavior likelihoods supplied by
    `replay_behavior_logprobs` exactly as `update_ppo` supplies them.  This is
    the best available estimate of the true gradient at this operating point.
    No optimizer is constructed and no step is ever taken.
  * `Ghat_n`, the mean gradient over `n` disjoint minibatches, for several
    effective averaging widths `n`.  Width 20 is the reading of Muon's default
    momentum 0.95 as `1 / (1 - 0.95) = 20`; the variance-equivalent sample size
    of a normalized EMA is `(1 + 0.95) / (1 - 0.95) = 39`, so 20 is the more
    pessimistic (noisier) proxy of the two and both are worth passing.
  * `cos_raw = cos(Ghat_n, Gstar)` and `cos_polar = cos(P(Ghat_n), Gstar)` under
    the Frobenius inner product, where `P` is the EXACT polar factor from a
    float64 SVD, plus `cos_polar_ns5` and `cos_polar_polar_express` from the
    two Newton-Schulz approximations Muon implementations actually run, so the
    verdict can be checked against the approximation as well as the ideal.
  * `alignment_gain = cos_polar - cos_raw`, the headline.  POSITIVE means
    orthogonalization moves the update CLOSER to the true gradient on this
    system's gradients and Muon's premise holds here.  NEGATIVE FALSIFIES THE
    PREMISE for this system: the polar factor would be spending step budget on
    directions whose signs are noise, which is the low-SNR failure mode.
  * The Equation 4 inputs themselves: `s_i` from `Gstar`, the across-minibatch
    standard deviation `sigma_i` of `diag(U^T Ghat V)`, their ratio by rank, and
    the spectral decay and stable rank of `Gstar` - flat spectra are where
    orthogonalization changes little and steep ones where it changes most.

Three diagnostics exist so that a sign cannot be read naively, because a
negative `alignment_gain` has two possible causes and they mean different
things:

  * `spectrum.polar_ceiling`, the value `cos(P(Gstar), Gstar)` that
    orthogonalization would score on a NOISELESS gradient.  It equals one only
    for a perfectly flat spectrum, so a matrix whose `Gstar` spectrum is steep
    pays a fixed alignment cost for flattening however clean its gradient is,
    and `cos_polar` cannot exceed this number.  A negative gain that merely
    reproduces the ceiling is spectral anisotropy, not a noise verdict.
  * `signal_to_noise_split`, the same per-direction SNR taken with the singular
    basis built from one half of the minibatches and the statistics from the
    other.  The specified estimator reads `s_i` off `Gstar`'s own singular
    values, and `Gstar` is a finite-sample gradient whose singular values carry
    the top of the noise spectrum too, which inflates `s_i / sigma_i`.  The
    split basis removes that circularity and instead shrinks the signal, so the
    two bracket the truth.
  * `reference_quality`, the cosine between two independent half-batch
    gradients and the signal energy share it implies for `Gstar`.  This is what
    says whether the reference deserves to be called a signal at all.

Linear weights and Conv2d weights are reported in separate groups.  Conv
gradients are reshaped to `(out_channels, in_channels * kh * kw)`, the standard
Muon convention, because that reshape is a modelling choice a reader may not
accept and it should be judgeable on its own.

MEASURED, and the premise is FALSIFIED here.  Four configurations -- the BC
actor with a freshly initialized critic, the BC actor with the iteration-12
critic, the iteration-12 actor and critic together, and a doubled 224-game wave
-- at 112 or 224 games, 316 minibatches each, averaging width 20:

    group   cos_raw   cos_polar   gain     gain_ns5   gain_holdout   ceiling
    conv    0.37-0.53  0.28-0.40  -0.09..-0.14  worse  -0.017..-0.038  0.75-0.78
    linear  0.38-0.53  0.18-0.25  -0.20..-0.30  worse  -0.009..-0.064  0.47-0.50

Every gain is negative, in every group, in every configuration, for the exact
polar factor AND for the Newton-Schulz approximation Muon actually runs, AND for
the holdout reference that removes the self-alignment circularity.  The three
diagnostics above say this is a noise verdict rather than anisotropy: for linear
weights the noiseless `polar_ceiling` of 0.47-0.50 is itself BELOW the 0.51-0.53
that the raw gradient already achieves, so orthogonalization is capped under the
baseline structurally and not merely degraded by noise.

The Equation 4 inputs explain it.  `signal_to_noise_split_fraction_above_one` is
0.000 in every group of every configuration: with the circularity removed, NO
singular direction carries signal above its across-minibatch noise.  The
gradients are also far from flat -- stable rank 2.7-3.3 out of 96 for linear
weights, top-over-median singular value 48-62 -- so flattening spends the step
budget on a tail that is entirely noise.  That is precisely the low-SNR,
episode-level regime arXiv 2607.16169 reports Muon failing in, and the reward
this probe measured was a terminal money difference at `gamma = 1.0`, the same
undiscounted finite-horizon objective used by current zero-sum training, with a
measured raw advantage standard deviation of 0.13-0.16.

A side measurement that does not concern Muon: `partitions[*].
minibatch_fraction_above_clip` is 1.000 in all 1264 minibatches, so
`PpoConfig.max_gradient_norm` binds on every step.  See its comment in
`kaggriculture.ppo`.

What this CANNOT tell you: it measures the PREMISE of Muon on one operating
point's gradients, not the end-to-end effect of training with Muon.  It says
nothing about Muon's step-size geometry, its weight-decay interaction, or how
the operating point moves once the optimizer changes it.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from kaggriculture.inference import load_actor_artifact
from kaggriculture.ppo import (
    UPDATE_COMPILE_MODES,
    PpoConfig,
    _actor_batch_args,
    _actor_minibatch_terms,
    _balanced_minibatch_slices,
    _batch_tensor,
    _cached_update_callable,
    _device_compile_mode,
    _explained_variance,
    _stage_tensor,
    _validate_staged_action_masks,
    prepare_advantages,
    replay_behavior_logprobs,
    replay_behavior_values,
)
from kaggriculture.production import (
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_ROLLOUT_BFLOAT16,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_TEMPERATURE,
    PRODUCTION_UPDATE_COMPILE_MODE,
    production_ppo_config,
)
from kaggriculture.provenance import source_identity
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import (
    ROLLOUT_FORWARD_MODES,
    allocate_rollout_storage,
    collect_self_play_rust,
)

#: Muon's default momentum. `1 / (1 - beta)` is the weight-sum reading of its
#: averaging width and `(1 + beta) / (1 - beta)` the variance-equivalent sample
#: size of the normalized EMA; the report states both against the widths run.
MUON_MOMENTUM = 0.95

#: Keller Jordan's standard Muon Newton-Schulz quintic, as one coefficient
#: triple applied for every one of the five steps. The reference implementation
#: in modded-nanogpt no longer runs this: it runs the Polar Express schedule
#: below through custom Triton kernels. Both are measured, because the shipped
#: verdict must not depend on which approximation a reader has in mind.
NEWTON_SCHULZ_COEFFICIENTS = ((3.4445, -4.7750, 2.0315),) * 5
NEWTON_SCHULZ_SAFETY = 1.0
NEWTON_SCHULZ_EPSILON = 1e-7

#: Polar Express (arXiv 2505.16932) as configured in
#: `/home/marvin/Documents/repositories/modded-nanogpt/train_gpt.py`
#: (`polar_express_coeffs`, num_iters=5, safety_factor=2e-2, cushion=2), with
#: that function's pre-scaling. The iteration is reproduced in plain torch
#: rather than through its fused Triton kernels and distributed machinery; the
#: arithmetic is the same `X <- a X + (b A + c A A) X` with `A = X X^T`.
POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
POLAR_EXPRESS_SAFETY = 1.0 + 2e-2
POLAR_EXPRESS_EPSILON = 1e-6

DEFAULT_AVERAGING_WIDTHS = "1,2,4,8,20"
#: The width the verdict line is read at, per the momentum reading above.
HEADLINE_WIDTH = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor",
        type=Path,
        default=Path("runs/bc4-v27-conv/bc-actor.pt"),
        help="actor artifact defining the operating point; the spectrum is a property of it",
    )
    parser.add_argument(
        "--critic-checkpoint",
        type=Path,
        help=(
            "training checkpoint supplying fitted critic weights. The advantages this "
            "gradient is built from are GAE residuals against the critic, so a randomly "
            "initialized critic measures a different gradient; the report always states "
            "which was used and the critic's explained variance on this rollout"
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=PRODUCTION_SELF_PLAY_GAMES,
        help="self-play games; both seats are learner trajectories (states = games * 2 * horizon)",
    )
    parser.add_argument("--episode-steps", type=int, default=PRODUCTION_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--averaging-widths",
        default=DEFAULT_AVERAGING_WIDTHS,
        help=(
            "comma-separated effective averaging widths n; Ghat_n is the mean gradient "
            "over n disjoint minibatches"
        ),
    )
    parser.add_argument(
        "--shuffles",
        type=int,
        default=4,
        help=(
            "independent minibatch partitions of the same rollout. Draws inside one "
            "partition are disjoint; more partitions buy more draws at the wide n where "
            "one partition supports only a couple, at one extra backward pass each"
        ),
    )
    parser.add_argument(
        "--rollout-forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        default=PRODUCTION_ROLLOUT_FORWARD_MODE,
    )
    parser.add_argument(
        "--rollout-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=PRODUCTION_ROLLOUT_BFLOAT16,
    )
    parser.add_argument(
        "--update-compile-mode",
        choices=UPDATE_COMPILE_MODES,
        default=PRODUCTION_UPDATE_COMPILE_MODE,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="path for the JSON report")
    return parser.parse_args()


def _averaging_widths(specification: str) -> tuple[int, ...]:
    widths = tuple(sorted({int(part) for part in specification.split(",") if part.strip()}))
    if not widths or widths[0] < 1:
        raise ValueError("averaging widths must be a comma-separated list of positive integers")
    return widths


@dataclass(frozen=True)
class TrackedMatrix:
    """One Muon-eligible weight, with its flat offset in the gradient snapshot."""

    name: str
    kind: str
    shape: tuple[int, ...]
    rows: int
    columns: int
    offset: int

    @property
    def numel(self) -> int:
        return self.rows * self.columns


def tracked_matrices(actor: nn.Module) -> tuple[list[nn.Parameter], list[TrackedMatrix]]:
    """Every `nn.Linear` and `nn.Conv2d` weight, in one flat contiguous layout.

    Conv weights are recorded as `(out_channels, in_channels * kh * kw)`, which
    is what Muon does to a convolution, and are tagged so the report can judge
    them apart from the genuinely 2D matrices.
    """
    parameters: list[nn.Parameter] = []
    matrices: list[TrackedMatrix] = []
    offset = 0
    for name, module in actor.named_modules():
        if isinstance(module, nn.Linear):
            kind = "linear"
        elif isinstance(module, nn.Conv2d):
            kind = "conv"
        else:
            continue
        weight = module.weight
        rows = int(weight.shape[0])
        matrices.append(
            TrackedMatrix(
                name=name,
                kind=kind,
                shape=tuple(int(size) for size in weight.shape),
                rows=rows,
                columns=weight.numel() // rows,
                offset=offset,
            )
        )
        parameters.append(weight)
        offset += weight.numel()
    if not matrices:
        raise ValueError("actor exposes no linear or convolutional weights")
    return parameters, matrices


def _flat_gradient(parameters: list[nn.Parameter]) -> Tensor:
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("a tracked weight received no gradient from the policy loss")
    return torch.cat([parameter.grad.reshape(-1) for parameter in parameters])


@dataclass(frozen=True)
class GradientContext:
    """Everything one policy-gradient minibatch needs, resolved once."""

    actor: nn.Module
    architecture: str
    staged: dict[str, Tensor]
    config: PpoConfig
    autocast_enabled: bool
    actor_terms: Any
    component_counts: np.ndarray
    parameters: list[nn.Parameter]

    def policy_objective(self, host_indices: np.ndarray, indices: Tensor) -> tuple[Tensor, int]:
        """The clipped surrogate sum and its component count for one minibatch.

        Identical to `update_ppo`'s actor minibatch: the same compiled callable,
        the same staged tensors and dtypes, the same host-side component count.
        Behavior likelihoods come from `replay_behavior_logprobs`, so at
        unchanged weights the importance ratio starts at one and this is the
        gradient the update's first actor step would actually see.
        """
        staged = self.staged
        policy_sum, _entropy, _kl, _clipped = self.actor_terms(
            self.actor,
            _batch_tensor(staged["unit_actions"], indices, torch.long),
            _batch_tensor(staged["market_kinds"], indices, torch.long),
            _batch_tensor(staged["market_quantities"], indices, torch.long),
            _batch_tensor(staged["unit_masks"], indices, torch.bool),
            _batch_tensor(staged["market_kind_masks"], indices, torch.bool),
            _batch_tensor(staged["market_quantity_masks"], indices, torch.bool),
            _batch_tensor(staged["unit_active"], indices, torch.float32),
            _batch_tensor(staged["market_active"], indices, torch.float32),
            _batch_tensor(staged["market_quantity_active"], indices, torch.float32),
            _batch_tensor(staged["old_unit_logprobs"], indices, torch.float32),
            _batch_tensor(staged["old_market_kind_logprobs"], indices, torch.float32),
            _batch_tensor(staged["old_market_quantity_logprobs"], indices, torch.float32),
            _batch_tensor(staged["advantages"], indices, torch.float32),
            self.config.clip_low,
            self.config.clip_high,
            self.autocast_enabled,
            *_actor_batch_args(self.architecture, staged, indices),
        )
        return policy_sum, max(1, int(self.component_counts[host_indices].sum()))


def full_batch_gradient(context: GradientContext, order: np.ndarray) -> Tensor:
    """The exact full-batch policy gradient, accumulated minibatch by minibatch.

    Normalized once by the total component count rather than per minibatch, so
    this is the gradient of the whole rollout's policy loss and does not depend
    on how the rollout happens to be partitioned. `Gstar` in the report.
    """
    device = context.staged["unit_actions"].device
    ordered = torch.from_numpy(order).to(device=device)
    total = max(1, int(context.component_counts[order].sum()))
    context.actor.zero_grad(set_to_none=True)
    for batch_slice in _balanced_minibatch_slices(order.size, context.config.minibatch_size):
        policy_sum, _count = context.policy_objective(order[batch_slice], ordered[batch_slice])
        (-policy_sum / total).backward()
    gradient = _flat_gradient(context.parameters).double()
    context.actor.zero_grad(set_to_none=True)
    return gradient


def minibatch_gradients(
    context: GradientContext, order: np.ndarray
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-minibatch policy gradients over one partition of the rollout.

    Each row is `grad(-policy_sum / component_count)` for one minibatch: the
    quantity a Muon momentum buffer would accumulate, one optimizer step's worth
    of gradient. Returned unclipped; `update_ppo` renormalizes each minibatch by
    `clip_grad_norm_`, which is a per-minibatch rescale that would change the
    direction of an average of several, so the norms are reported alongside for
    the reader to judge how much of the batch that bound would have touched.
    """
    device = context.staged["unit_actions"].device
    ordered = torch.from_numpy(order).to(device=device)
    slices = _balanced_minibatch_slices(order.size, context.config.minibatch_size)
    total = sum(parameter.numel() for parameter in context.parameters)
    stack = torch.empty((len(slices), total), dtype=torch.float32, device=device)
    counts = torch.empty(len(slices), dtype=torch.float64, device=device)
    for index, batch_slice in enumerate(slices):
        context.actor.zero_grad(set_to_none=True)
        policy_sum, count = context.policy_objective(order[batch_slice], ordered[batch_slice])
        (-policy_sum / count).backward()
        stack[index] = _flat_gradient(context.parameters)
        counts[index] = count
    context.actor.zero_grad(set_to_none=True)
    return stack, counts, stack.norm(dim=1).double()


def _cosine(left: Tensor, right: Tensor) -> Tensor:
    """Frobenius cosine similarity, batched over any leading dimensions."""
    inner = (left * right).sum(dim=(-2, -1))
    scale = left.flatten(-2).norm(dim=-1) * right.flatten(-2).norm(dim=-1)
    return inner / scale.clamp_min(torch.finfo(inner.dtype).tiny)


def polar_factor(matrix: Tensor) -> Tensor:
    """The exact polar factor `U V^T` from a float64 SVD.

    float64 throughout: the whole measurement is a difference between two
    cosines that can be a few thousandths wide, and an fp32 SVD of a
    near-singular gradient is not reliably that accurate.
    """
    left, _values, right = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right


def newton_schulz(
    matrix: Tensor,
    coefficients: tuple[tuple[float, float, float], ...],
    *,
    safety: float,
    epsilon: float,
) -> Tensor:
    """Muon's Newton-Schulz orthogonalization, in bf16 exactly as Muon runs it."""
    iterate = matrix.bfloat16()
    tall = iterate.shape[-2] > iterate.shape[-1]
    if tall:
        iterate = iterate.mT
    norms = iterate.flatten(-2).norm(dim=-1)[..., None, None]
    iterate = iterate / (norms * safety + epsilon)
    for first, second, third in coefficients:
        gram = iterate @ iterate.mT
        iterate = first * iterate + (second * gram + third * (gram @ gram)) @ iterate
    return (iterate.mT if tall else iterate).double()


@dataclass
class WidthAccumulator:
    """Cosines at one averaging width, pooled over draws and partitions."""

    cos_raw: list[float] = field(default_factory=list)
    cos_polar: list[float] = field(default_factory=list)
    cos_polar_ns5: list[float] = field(default_factory=list)
    cos_polar_polar_express: list[float] = field(default_factory=list)
    cos_raw_holdout: list[float] = field(default_factory=list)
    cos_polar_holdout: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        gains = [polar - raw for polar, raw in zip(self.cos_polar, self.cos_raw, strict=True)]
        holdout_gains = [
            polar - raw
            for polar, raw in zip(self.cos_polar_holdout, self.cos_raw_holdout, strict=True)
        ]
        return {
            "draws": len(self.cos_raw),
            "cos_raw": _moments(self.cos_raw),
            "cos_polar": _moments(self.cos_polar),
            "cos_polar_ns5": _moments(self.cos_polar_ns5),
            "cos_polar_polar_express": _moments(self.cos_polar_polar_express),
            "alignment_gain": _moments(gains),
            "alignment_gain_ns5": _moments(
                [polar - raw for polar, raw in zip(self.cos_polar_ns5, self.cos_raw, strict=True)]
            ),
            "cos_raw_holdout": _moments(self.cos_raw_holdout),
            "cos_polar_holdout": _moments(self.cos_polar_holdout),
            "alignment_gain_holdout": _moments(holdout_gains),
        }


def _moments(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


@dataclass
class ProjectionMoments:
    """Streamed first and second moments of `diag(U^T G V)` over minibatches."""

    total: Tensor
    square_total: Tensor
    count: int = 0

    @classmethod
    def zeros(cls, rank: int, device: torch.device) -> ProjectionMoments:
        return cls(
            total=torch.zeros(rank, dtype=torch.float64, device=device),
            square_total=torch.zeros(rank, dtype=torch.float64, device=device),
        )

    def observe(self, projected: Tensor) -> None:
        self.total += projected.sum(dim=0)
        self.square_total += projected.square().sum(dim=0)
        self.count += projected.shape[0]

    def noise(self) -> Tensor:
        """Across-minibatch standard deviation per direction: Equation 4's sigma_i."""
        if self.count < 2:
            raise ValueError("per-direction noise needs at least two minibatches")
        mean = self.total / self.count
        variance = (self.square_total - self.count * mean.square()) / (self.count - 1)
        return variance.clamp_min(0.0).sqrt()

    def signal(self) -> Tensor:
        """Across-minibatch mean per direction: the signal in an independent basis."""
        return self.total / max(self.count, 1)


def _snr_report(signal: Tensor, noise: Tensor) -> dict[str, float]:
    tiny = torch.finfo(noise.dtype).tiny
    ratio = signal.abs() / noise.clamp_min(tiny)
    rank = ratio.numel()
    return {
        "top_rank": float(ratio[0]),
        "median_rank": float(ratio[rank // 2]),
        "bottom_rank": float(ratio[-1]),
        "fraction_above_one": float((ratio > 1.0).double().mean()),
        "signal_top_rank": float(signal[0]),
        "signal_median_rank": float(signal[rank // 2]),
        "signal_bottom_rank": float(signal[-1]),
        "noise_top_rank": float(noise[0]),
        "noise_median_rank": float(noise[rank // 2]),
        "noise_bottom_rank": float(noise[-1]),
    }


def _eq4_cosine(signal: Tensor, noise: Tensor, width: int) -> float:
    """Equation 4's expected inner product, normalized into a cosine.

    `sigma_i` shrinks by `sqrt(width)` because the draw averages that many
    independent minibatches, and the normalization is exact: `||P(Ghat)||_F` is
    `sqrt(rank)` for any polar factor.
    """
    tiny = torch.finfo(noise.dtype).tiny
    ratio = signal.abs() / noise.clamp_min(tiny) * math.sqrt(width)
    inner = float((signal.abs() * (2.0 * torch.special.ndtr(ratio) - 1.0)).sum())
    denominator = math.sqrt(signal.numel()) * max(float(signal.norm()), float(tiny))
    return inner / denominator


@dataclass
class MatrixAccumulator:
    """Per-matrix state: the reference SVD plus pooled draw and noise moments."""

    matrix: TrackedMatrix
    reference: Tensor
    left: Tensor
    singular_values: Tensor
    right: Tensor
    widths: dict[int, WidthAccumulator]
    moments: ProjectionMoments
    split_moments: ProjectionMoments
    split_half_cosine: list[float] = field(default_factory=list)

    @classmethod
    def build(
        cls, matrix: TrackedMatrix, reference: Tensor, widths: tuple[int, ...]
    ) -> MatrixAccumulator:
        left, singular_values, right = torch.linalg.svd(reference, full_matrices=False)
        rank = singular_values.numel()
        return cls(
            matrix=matrix,
            reference=reference,
            left=left,
            singular_values=singular_values,
            right=right,
            widths={width: WidthAccumulator() for width in widths},
            moments=ProjectionMoments.zeros(rank, reference.device),
            split_moments=ProjectionMoments.zeros(rank, reference.device),
        )

    def observe(self, gradients: Tensor, counts: Tensor) -> None:
        """Fold one partition's per-minibatch gradients into every statistic."""
        # Diagonal of U^T G V per minibatch, in `Gstar`'s own basis: the
        # per-singular-direction coordinate whose across-minibatch spread is
        # Equation 4's sigma_i, and the estimator the contract specifies.
        self.moments.observe(torch.einsum("ai,mab,ib->mi", self.left, gradients, self.right))
        self._observe_split(gradients)

        minibatches = gradients.shape[0]
        weighted_total = (counts[:, None, None] * gradients).sum(dim=0)
        count_total = counts.sum()
        for width, accumulator in self.widths.items():
            draws = minibatches // width
            if draws < 1:
                continue
            grouped = gradients[: draws * width].reshape(draws, width, *gradients.shape[1:])
            estimate = grouped.mean(dim=1)
            # A held-out reference for the same draw: the full batch minus the
            # draw's own minibatches. `Gstar` contains every draw by
            # construction, which flatters both cosines through shared noise;
            # this variant shares none, and the headline is a difference of two
            # cosines against the same reference either way.
            drawn_counts = counts[: draws * width].reshape(draws, width)
            drawn_weighted = (grouped * drawn_counts[..., None, None]).sum(dim=1)
            remainder = (count_total - drawn_counts.sum(dim=1)).clamp_min(1.0)
            holdout = (weighted_total - drawn_weighted) / remainder[:, None, None]

            polar = polar_factor(estimate)
            quintic = newton_schulz(
                estimate,
                NEWTON_SCHULZ_COEFFICIENTS,
                safety=NEWTON_SCHULZ_SAFETY,
                epsilon=NEWTON_SCHULZ_EPSILON,
            )
            express = newton_schulz(
                estimate,
                POLAR_EXPRESS_COEFFICIENTS,
                safety=POLAR_EXPRESS_SAFETY,
                epsilon=POLAR_EXPRESS_EPSILON,
            )
            accumulator.cos_raw += _cosine(estimate, self.reference).tolist()
            accumulator.cos_polar += _cosine(polar, self.reference).tolist()
            accumulator.cos_polar_ns5 += _cosine(quintic, self.reference).tolist()
            accumulator.cos_polar_polar_express += _cosine(express, self.reference).tolist()
            accumulator.cos_raw_holdout += _cosine(estimate, holdout).tolist()
            accumulator.cos_polar_holdout += _cosine(polar, holdout).tolist()

    def _observe_split(self, gradients: Tensor) -> None:
        """Per-direction moments in a basis independent of the gradients measured.

        The specified estimator reads `s_i` off `Gstar`'s own singular values,
        and `Gstar` is a finite-sample gradient: its singular values carry the
        top of the noise spectrum as well as the signal, which inflates `s_i`
        and therefore inflates `s_i / sigma_i`. Splitting the partition in two
        removes that circularity -- the basis comes from the odd minibatches and
        every statistic from the even ones -- at the cost of a noisier basis,
        which shrinks the measured signal rather than inflating it. So the two
        together bracket the true per-direction SNR.
        """
        if gradients.shape[0] < 4:
            return
        odd = gradients[1::2]
        even = gradients[0::2]
        left, _values, right = torch.linalg.svd(odd.mean(dim=0), full_matrices=False)
        self.split_moments.observe(torch.einsum("ai,mab,ib->mi", left, even, right))
        self.split_half_cosine.append(float(_cosine(odd.mean(dim=0), even.mean(dim=0))))

    def report(self) -> dict[str, Any]:
        values = self.singular_values
        noise = self.moments.noise()
        tiny = torch.finfo(noise.dtype).tiny
        rank = values.numel()
        frobenius = float(values.square().sum().sqrt())
        split_signal = self.split_moments.signal()
        split_noise = (
            self.split_moments.noise() if self.split_moments.count > 1 else torch.ones_like(values)
        )
        # cos(P(Gstar), Gstar): what orthogonalization scores on a NOISELESS
        # gradient. It is 1 only for a perfectly flat spectrum, so any matrix
        # with a steep spectrum pays a fixed alignment cost for flattening no
        # matter how clean the gradient is, and the measured `cos_polar` cannot
        # exceed it. Without this number a negative alignment gain cannot be
        # attributed between spectral anisotropy and low SNR.
        ceiling = float(values.sum()) / (math.sqrt(rank) * max(frobenius, float(tiny)))
        widths: dict[str, Any] = {}
        for width, accumulator in sorted(self.widths.items()):
            if not accumulator.cos_raw:
                # A width wider than the partition supports draws no samples.
                continue
            summary = accumulator.summary()
            summary["eq4_predicted_cos_polar"] = _eq4_cosine(values, noise, width)
            summary["eq4_predicted_cos_polar_split"] = (
                _eq4_cosine(split_signal, split_noise, width)
                if self.split_moments.count > 1
                else math.nan
            )
            summary["eq4_predicted_gain"] = (
                summary["eq4_predicted_cos_polar"] - summary["cos_raw"]["mean"]
            )
            summary["cos_polar_minus_ceiling"] = summary["cos_polar"]["mean"] - ceiling
            widths[str(width)] = summary
        # Two independent half-batch gradients agree only to the extent that
        # `Gstar` itself is signal: for a shared truth plus independent noise,
        # their cosine c implies a signal energy share of 2c / (1 + c) in the
        # full batch. This is what says whether the reference deserves the name.
        half_cosine = float(np.mean(self.split_half_cosine)) if self.split_half_cosine else math.nan
        return {
            "name": self.matrix.name,
            "kind": self.matrix.kind,
            "shape": list(self.matrix.shape),
            "matrix_shape": [self.matrix.rows, self.matrix.columns],
            "parameters": self.matrix.numel,
            "in_transformer": self.matrix.name.startswith("transformer."),
            "rank": rank,
            "minibatches_observed": self.moments.count,
            "spectrum": {
                "top": float(values[0]),
                "median": float(values[rank // 2]),
                "min": float(values[-1]),
                "top_over_median": float(values[0] / values[rank // 2].clamp_min(tiny)),
                "top_over_min": float(values[0] / values[-1].clamp_min(tiny)),
                "stable_rank": float(values.square().sum() / values[0].square().clamp_min(tiny)),
                "frobenius": frobenius,
                "polar_ceiling": ceiling,
            },
            "reference_quality": {
                "half_batch_cosine": half_cosine,
                "signal_energy_fraction": 2.0 * half_cosine / (1.0 + half_cosine),
            },
            "signal_to_noise": _snr_report(values, noise),
            "signal_to_noise_split": _snr_report(split_signal, split_noise),
            "widths": widths,
        }


def _parameter_weighted(values: list[float], weights: list[int]) -> float:
    """Mean of `values` weighted by parameter count, which is Muon's own unit."""
    weighted = np.asarray(weights, dtype=np.float64)
    return float((np.asarray(values, dtype=np.float64) * weighted).sum() / weighted.sum())


def _weighted_group(reports: list[dict[str, Any]], widths: tuple[int, ...]) -> dict[str, Any]:
    """Parameter-count-weighted aggregate over one group of matrices."""
    if not reports:
        return {}
    weights = [int(report["parameters"]) for report in reports]
    aggregate: dict[str, Any] = {
        "matrices": len(reports),
        "parameters": sum(weights),
        "widths": {},
    }
    metrics = (
        "cos_raw",
        "cos_polar",
        "cos_polar_ns5",
        "cos_polar_polar_express",
        "alignment_gain",
        "alignment_gain_ns5",
        "alignment_gain_holdout",
    )
    for width in widths:
        key = str(width)
        present = [report for report in reports if key in report["widths"]]
        if not present:
            continue
        subset = [int(report["parameters"]) for report in present]
        summaries = [report["widths"][key] for report in present]
        aggregate["widths"][key] = {
            "matrices": len(present),
            "draws": min(summary["draws"] for summary in summaries),
            **{
                metric: _parameter_weighted(
                    [summary[metric]["mean"] for summary in summaries], subset
                )
                for metric in metrics
            },
            "eq4_predicted_cos_polar": _parameter_weighted(
                [summary["eq4_predicted_cos_polar"] for summary in summaries], subset
            ),
            "eq4_predicted_cos_polar_split": _parameter_weighted(
                [summary["eq4_predicted_cos_polar_split"] for summary in summaries], subset
            ),
            "cos_polar_minus_ceiling": _parameter_weighted(
                [summary["cos_polar_minus_ceiling"] for summary in summaries], subset
            ),
            "matrices_with_positive_gain": sum(
                1 for summary in summaries if summary["alignment_gain"]["mean"] > 0.0
            ),
        }
    aggregate |= {
        "signal_to_noise_fraction_above_one": _parameter_weighted(
            [report["signal_to_noise"]["fraction_above_one"] for report in reports], weights
        ),
        "signal_to_noise_split_fraction_above_one": _parameter_weighted(
            [report["signal_to_noise_split"]["fraction_above_one"] for report in reports], weights
        ),
        "signal_to_noise_median_rank": _parameter_weighted(
            [report["signal_to_noise"]["median_rank"] for report in reports], weights
        ),
        "signal_to_noise_split_median_rank": _parameter_weighted(
            [report["signal_to_noise_split"]["median_rank"] for report in reports], weights
        ),
        "stable_rank": _parameter_weighted(
            [report["spectrum"]["stable_rank"] for report in reports], weights
        ),
        "polar_ceiling": _parameter_weighted(
            [report["spectrum"]["polar_ceiling"] for report in reports], weights
        ),
        "top_over_median_singular_value": _parameter_weighted(
            [report["spectrum"]["top_over_median"] for report in reports], weights
        ),
        "reference_signal_energy_fraction": _parameter_weighted(
            [report["reference_quality"]["signal_energy_fraction"] for report in reports], weights
        ),
    }
    return aggregate


def _stage_rollout(rollout: Any, device: torch.device) -> dict[str, Tensor]:
    """Stage exactly the fields `update_ppo` stages, in the same dtypes."""
    staged = {name: _stage_tensor(array, device) for name, array in rollout.states.items()}
    staged |= {
        "unit_actions": _stage_tensor(rollout.unit_actions, device),
        "market_kinds": _stage_tensor(rollout.market_kinds, device),
        "market_quantities": _stage_tensor(rollout.market_quantities, device),
        "unit_masks": _stage_tensor(rollout.unit_masks, device),
        "market_kind_masks": _stage_tensor(rollout.market_kind_masks, device),
        "market_quantity_masks": _stage_tensor(rollout.market_quantity_masks, device),
        "unit_active": _stage_tensor(rollout.unit_active, device),
        "market_active": _stage_tensor(rollout.market_active, device),
        "market_quantity_active": _stage_tensor(rollout.market_quantity_active, device),
    }
    return staged


def _load_critic(
    args: argparse.Namespace, architecture: Any, model_config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    critic = architecture.build_critic(model_config).to(device)
    provenance: dict[str, Any] = {"source": "initialized", "iteration": None}
    if args.critic_checkpoint is not None:
        checkpoint = torch.load(args.critic_checkpoint, map_location=device, weights_only=False)
        if "critic" not in checkpoint:
            raise SystemExit(f"{args.critic_checkpoint} carries no critic weights")
        if checkpoint.get("model_config") != model_config:
            raise SystemExit("critic checkpoint model configuration does not match the actor's")
        critic.load_state_dict(checkpoint["critic"])
        provenance = {
            "source": str(args.critic_checkpoint),
            "iteration": int(checkpoint.get("iteration", 0)),
        }
    critic.requires_grad_(False)
    return critic, provenance


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    # The gradient being measured is a bf16-autocast, Inductor-compiled
    # gradient. A CPU run would measure a different one at a hundredth of the
    # speed and report it under the same name.
    if device.type != "cuda":
        raise SystemExit(f"the gradient spectrum must be measured on cuda, not {device.type}")
    widths = _averaging_widths(args.averaging_widths)
    if args.shuffles < 1:
        raise SystemExit("at least one minibatch partition is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    actor, payload = load_actor_artifact(args.actor, device=device)
    architecture = resolve_architecture(payload)
    model_config = payload["model_config"]
    config = PpoConfig(**production_ppo_config(update_compile_mode=args.update_compile_mode))
    critic, critic_provenance = _load_critic(args, architecture, model_config, device)

    arena = allocate_rollout_storage(
        architecture.name,
        args.games * 2,
        args.episode_steps - 1,
        pin_memory=True,
    )
    rollout = collect_self_play_rust(
        actor,
        games=args.games,
        seed_start=args.seed,
        episode_steps=args.episode_steps,
        temperature=PRODUCTION_TEMPERATURE,
        sampling_seed=args.seed ^ 0x5EED,
        forward_mode=args.rollout_forward_mode,
        forward_autocast=args.rollout_bfloat16,
        storage=arena,
    )

    flat_valid = rollout.valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    component_counts = (
        rollout.unit_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_quantity_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
    )
    minibatches = len(_balanced_minibatch_slices(valid_indices.size, config.minibatch_size))
    # Four is the floor for the split-half diagnostics, and a rollout that
    # cannot supply a draw at the widest requested averaging width measures
    # nothing at the width the verdict is read at.
    if minibatches < 4:
        raise SystemExit(
            f"{valid_indices.size} valid states partition into {minibatches} minibatches; "
            "at least four are required"
        )
    if max(widths) > minibatches:
        raise SystemExit(
            f"averaging width {max(widths)} exceeds the {minibatches} minibatches this rollout "
            "supplies; raise --games or lower --averaging-widths"
        )
    staged = _stage_rollout(rollout, device)
    _validate_staged_action_masks(staged, torch.from_numpy(flat_valid).to(device))

    autocast_enabled = config.use_bfloat16
    compile_mode = _device_compile_mode(config.update_compile_mode, device)
    behavior_values = (
        replay_behavior_values(
            critic,
            architecture.name,
            staged,
            compile_mode=compile_mode,
            autocast_enabled=autocast_enabled,
        )
        .cpu()
        .numpy()
        .reshape(rollout.rewards.shape)
    )
    staged.update(
        replay_behavior_logprobs(
            actor,
            architecture.name,
            staged,
            valid_indices,
            minibatch_size=config.minibatch_size,
            autocast_enabled=autocast_enabled,
            compile_mode=compile_mode,
        )
    )
    prepared = prepare_advantages(rollout, behavior_values, config)
    staged["advantages"] = torch.from_numpy(prepared.advantages.reshape(-1)).to(device)
    actor.train()

    parameters, matrices = tracked_matrices(actor)
    context = GradientContext(
        actor=actor,
        architecture=architecture.name,
        staged=staged,
        config=config,
        autocast_enabled=autocast_enabled,
        actor_terms=_cached_update_callable(
            actor, "_kaggriculture_update_terms", _actor_minibatch_terms, compile_mode
        ),
        component_counts=component_counts,
        parameters=parameters,
    )

    reference = full_batch_gradient(context, valid_indices)
    accumulators = [
        MatrixAccumulator.build(
            matrix,
            reference[matrix.offset : matrix.offset + matrix.numel].view(
                matrix.rows, matrix.columns
            ),
            widths,
        )
        for matrix in matrices
    ]

    generator = np.random.default_rng(args.seed)
    partitions: list[dict[str, Any]] = []
    for partition in range(args.shuffles):
        order = generator.permutation(valid_indices)
        stack, counts, norms = minibatch_gradients(context, order)
        weighted = (stack.T @ counts.float()).double() / counts.sum()
        partitions.append(
            {
                "partition": partition,
                "minibatches": int(stack.shape[0]),
                "reference_agreement": float(
                    torch.dot(weighted, reference)
                    / (weighted.norm() * reference.norm()).clamp_min(1e-300)
                ),
                "minibatch_gradient_norm_median": float(norms.median()),
                "minibatch_gradient_norm_min": float(norms.min()),
                "minibatch_gradient_norm_max": float(norms.max()),
                "minibatch_fraction_above_clip": float(
                    (norms > config.max_gradient_norm).double().mean()
                ),
            }
        )
        for accumulator in accumulators:
            matrix = accumulator.matrix
            gradients = stack[:, matrix.offset : matrix.offset + matrix.numel].view(
                -1, matrix.rows, matrix.columns
            )
            accumulator.observe(gradients.double(), counts)
        del stack, counts, norms, weighted
        torch.cuda.empty_cache()
        print(
            f"partition {partition}: {partitions[-1]['minibatches']} minibatches, "
            f"reference agreement {partitions[-1]['reference_agreement']:.6f}",
            flush=True,
        )

    reports = [accumulator.report() for accumulator in accumulators]
    groups = {
        "linear": _weighted_group([r for r in reports if r["kind"] == "linear"], widths),
        "conv": _weighted_group([r for r in reports if r["kind"] == "conv"], widths),
        "linear_transformer": _weighted_group(
            [r for r in reports if r["kind"] == "linear" and r["in_transformer"]], widths
        ),
    }
    headline = str(HEADLINE_WIDTH) if HEADLINE_WIDTH in widths else str(widths[-1])
    verdict: dict[str, Any] = {
        "width": int(headline),
        "momentum_equivalent_widths": {
            "weight_sum": 1.0 / (1.0 - MUON_MOMENTUM),
            "variance_equivalent": (1.0 + MUON_MOMENTUM) / (1.0 - MUON_MOMENTUM),
        },
    }
    for name, group in groups.items():
        if not group or headline not in group["widths"]:
            continue
        measured = group["widths"][headline]
        verdict |= {
            f"{name}_alignment_gain": measured["alignment_gain"],
            f"{name}_alignment_gain_ns5": measured["alignment_gain_ns5"],
            f"{name}_alignment_gain_holdout": measured["alignment_gain_holdout"],
            f"{name}_cos_raw": measured["cos_raw"],
            f"{name}_cos_polar": measured["cos_polar"],
            f"{name}_polar_ceiling": group["polar_ceiling"],
            f"{name}_draws": measured["draws"],
        }

    report = {
        "configuration": {
            "actor": str(args.actor),
            "actor_iteration": int(payload.get("iteration", 0)),
            "architecture": architecture.name,
            "model_config": model_config,
            "critic": critic_provenance,
            "games": args.games,
            "episode_steps": args.episode_steps,
            "seed": args.seed,
            "shuffles": args.shuffles,
            "averaging_widths": list(widths),
            "minibatch_size": config.minibatch_size,
            "use_bfloat16": config.use_bfloat16,
            "update_compile_mode": config.update_compile_mode,
            "rollout_forward_mode": args.rollout_forward_mode,
            "rollout_bfloat16": args.rollout_bfloat16,
            "clip_low": config.clip_low,
            "clip_high": config.clip_high,
            "max_gradient_norm": config.max_gradient_norm,
            "actor_gae_lambda": config.actor_gae_lambda,
            "gamma": config.gamma,
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "source_identity": source_identity(),
        },
        "rollout": {
            "trajectories": int(rollout.valid.shape[0]),
            "horizon": int(rollout.valid.shape[1]),
            "states": int(flat_valid.size),
            "valid_states": int(valid_indices.size),
            "active_components": int(component_counts[valid_indices].sum()),
            "minibatches": minibatches,
            "raw_advantage_mean": prepared.raw_advantage_mean,
            "raw_advantage_std": prepared.raw_advantage_std,
            "critic_explained_variance": _explained_variance(
                prepared.value_targets, behavior_values, rollout.valid
            ),
            "critic_monte_carlo_explained_variance": _explained_variance(
                prepared.monte_carlo_returns, behavior_values, rollout.valid
            ),
        },
        "reference_gradient": {
            "frobenius_norm": float(reference.norm()),
            "tracked_parameters": int(reference.numel()),
            "matrices": len(matrices),
        },
        "partitions": partitions,
        "groups": groups,
        "verdict": verdict,
        "matrices": reports,
    }

    for name, group in groups.items():
        if not group or headline not in group["widths"]:
            continue
        measured = group["widths"][headline]
        for width in widths:
            row = group["widths"].get(str(width))
            if row is None:
                continue
            print(
                f"{name:>18s} n={width:<3d} draws {row['draws']:>3d}"
                f"  cos_raw {row['cos_raw']:+.6f}  cos_polar {row['cos_polar']:+.6f}"
                f"  gain {row['alignment_gain']:+.6f}"
                f"  ns5 {row['cos_polar_ns5']:+.6f}"
                f"  express {row['cos_polar_polar_express']:+.6f}"
                f"  holdout gain {row['alignment_gain_holdout']:+.6f}",
                flush=True,
            )
        print(
            f"\n{name}: n={headline} over {group['matrices']} matrices, "
            f"{group['parameters']} parameters"
            f"\n  ALIGNMENT GAIN {measured['alignment_gain']:+.6f}"
            f"  (cos_raw {measured['cos_raw']:+.6f} -> cos_polar {measured['cos_polar']:+.6f})"
            f"\n  positive-gain matrices {measured['matrices_with_positive_gain']}"
            f"/{measured['matrices']}"
            f"  noiseless polar ceiling {group['polar_ceiling']:+.6f}"
            f"  cos_polar - ceiling {measured['cos_polar_minus_ceiling']:+.6f}"
            f"\n  eq4 predicted cos_polar {measured['eq4_predicted_cos_polar']:+.6f}"
            f" (split-basis {measured['eq4_predicted_cos_polar_split']:+.6f})"
            f"\n  s_i/sigma_i > 1 fraction {group['signal_to_noise_fraction_above_one']:.4f}"
            f" (split-basis {group['signal_to_noise_split_fraction_above_one']:.4f})"
            f"  median-rank s/sigma {group['signal_to_noise_median_rank']:.4f}"
            f" (split {group['signal_to_noise_split_median_rank']:.4f})"
            f"\n  stable rank {group['stable_rank']:.2f}"
            f"  s_1/s_median {group['top_over_median_singular_value']:.2f}"
            f"  Gstar signal energy share {group['reference_signal_energy_fraction']:.4f}",
            flush=True,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
