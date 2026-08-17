"""PPO masked-token update with length-adaptive GAE and lambda-return targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    distributional_value_loss,
)
from kaggriculture.policy import component_logprobs, component_selected_logprobs
from kaggriculture.registry import CONV_ENTITY
from kaggriculture.rollout import RolloutBatch
from kaggriculture.structured import StructuredActor, StructuredCritic, StructuredInputs

Critic = DistributionalCritic | StructuredCritic
Actor = FarmActor | StructuredActor

#: VAPO's length-adaptive GAE constant, the one piece of that paper this update
#: still takes: it sets lambda from the horizon rather than from a tuned guess.
LENGTH_ADAPTIVE_GAE_ALPHA = 0.05
COMPETITION_ACTION_STEPS = EPISODE_STEPS - 1
DEFAULT_ACTOR_GAE_LAMBDA = 1.0 - 1.0 / (LENGTH_ADAPTIVE_GAE_ALPHA * COMPETITION_ACTION_STEPS)

# Numerics gates shared by the calibration benchmark, the training launcher's
# expected configuration, and the production training loop.
#
# The sampling-path versus update-replay divergence is bounded as a KL, not as
# a worst component. The quantity at risk is off-policy sampling bias — actions
# were drawn from the rollout forward while the gradient treats them as drawn
# from the update forward — which is a divergence between two distributions,
# taken under the one that did the sampling. A maximum over active components
# does not estimate that: it is an extreme-value statistic whose expectation
# grows with the component count, so the same policy fails or passes depending
# on rollout size, and it is dominated by actions whose behavior probability is
# too small to carry weight in any expectation. The earlier 5e-2 bound came
# from sweeping a randomly initialized actor, whose near-uniform heads have a
# thin tail; a behavior-cloned actor measures 20.68 on the identical code path
# with no defect present, which is what falsified the premise that only bugs
# reach that magnitude.
#
# The estimator is the k3 form `expm1(d) - d`, averaged over each head's active
# components, matching `_clipped_surrogate_sums` exactly so the two numbers are
# at least computed the same way. They are not, however, interchangeable as
# budgets: `target_kl` bounds staleness that the importance ratio *corrects
# for*, which costs variance and leaves the estimator consistent, whereas this
# divergence is uncorrected — nothing multiplies by p/q. Reading a bias budget
# off a variance budget would be a category error, so this bound is not derived
# from target_kl. Taking the maximum over heads rather than pooling keeps a
# single-head defect from being diluted by the unit head, which supplies most
# components.
#
# A mean alone would be the wrong single gate, because the bugs this audit
# exists to catch are characteristically localized — an off-by-one on the final
# minibatch, one seat, one action type, one mask path — and a mean dilutes them
# by the corrupted fraction. At production scale a 20% likelihood error confined
# to the last minibatch moves the mean by well under the bound while being an
# unambiguous defect. So the tail is gated too, by the *fraction* of sampled
# actions the two paths disagree about by more than
# UPDATE_REPLAY_TAIL_LOGPROB, rather than by how far the worst one strayed.
# Counting is what makes this stable: an indicator mean is bounded in [0, 1] and
# has finite variance no matter how heavy the log-ratio tail is, whereas a
# maximum — and, under a sufficiently heavy tail, the k3 mean itself — is
# dominated by a handful of order statistics and swings across seeds. A bug
# touching a fraction f of components registers as a tail fraction of about f,
# so the bound reads directly as the smallest detectable corrupted share.
#
# Both bounds are set from measurement on a behavior-cloned actor at production
# rollout size, because that is the sharp-policy regime the earlier calibration
# never sampled. Holding one rollout fixed and varying only autocast attributes
# the divergence almost entirely to bf16 in the update forward: per-component KL
# 3.0e-3 under bf16 against 2.2e-5 in fp32, a factor of 138, with backend and
# minibatch shape accounting for only the fp32 remainder. It is a bulk effect,
# not a tail one — 7.7% of unit components disagree by more than 0.05 nats where
# a randomly initialized actor has none at all — and it scales with how sharp
# the policy is, from 4e-7 at initialization to 2.1e-3 for the clone.
#
# That divergence is accepted rather than removed, and the reason is structural
# rather than a matter of magnitude. The importance ratio is unaffected, since
# `replay_behavior_logprobs` recomputes the behavior side through the same bf16
# forward; what remains is off-policy sampling bias. Writing d = log q - log p
# for the update and sampling heads, the bias in the surrogate is
# -E_p[expm1(d) * A]. Both heads normalize over the same masked support, so
# E_p[exp(d)] = 1 and therefore E[expm1(d)] = 0 *within each state*; the
# advantage is a per-state scalar (see `expanded_advantage` below), so it
# factors out of that inner expectation and the bias in the surrogate objective
# is exactly zero, not merely small. What survives is the gradient bias,
# A * E_p[grad log q], which does not vanish because the score varies with the
# action. Advantages are normalized to zero mean and unit variance and are
# essentially uncorrelated with a rounding pattern, so this cancels across
# states as well, but it is systematic where sampling noise is not, and
# systematic error accumulates linearly against the noise's square root.
# That is the argument for auditing it repeatedly through a run (see
# REPLAY_PARITY_AUDIT_INTERVAL) rather than assuming a one-time pass holds.
#
# Note also that the magnitude is easy to understate, and the honest reference
# point is what the update actually moves rather than what it is permitted to.
# KL is second order in d, so a per-component KL of 2.5e-3 means std[d] =
# sqrt(2 * KL) = 0.071. Measured across four runs, realized approx_kl has a
# median of 2.3e-3 per iteration — std 0.068. The uncorrected divergence is
# therefore about as wide as the corrected one, not a fraction of it; comparing
# instead against target_kl's 0.245 would report 29% and comparing the KLs
# directly would report 8%, and both flatter the number by measuring it against
# a trust region the update never reaches. Removing the divergence would mean an fp32
# update forward and forfeiting the bf16 speedup. MAX_UPDATE_REPLAY_KL is set
# at roughly 2.6x the worst production measurement of the clone: 1.77e-3,
# 1.87e-3 and 1.91e-3 over three waves drawn from disjoint environments, a
# spread of 1.08x, which is what makes a fixed bound meaningful at all.
#
# The materiality threshold must sit above the numerical noise floor of the
# precision actually in use, or it measures rounding instead of defects, and
# that floor has to be measured at the threshold rather than extrapolated to it.
# Swept on a production wave of the behavior clone, the share of components bf16
# rounding alone pushes past a threshold falls off far more slowly than an
# exponential — nearer a factor of three per half nat than a factor of two, and
# slowing as the threshold rises:
#
#     0.247 nats 9.8e-3 | 0.50 2.7e-3 | 0.75 1.1e-3 | 1.00 5.1e-4
#      1.25 nats 2.8e-4 | 1.50 1.6e-4 | 2.00 5.7e-5 | 2.50 1.9e-5
#      3.00 nats 1.1e-5 | 4.00 3.3e-6
#
# For contrast a randomly initialized actor has *zero* components past even
# 0.247 nats on any head, which is why the earlier calibration could never have
# predicted any of this.
#
# Picking the pair is more constrained than it looks, and the constraint is
# worth writing down because it decides how much this gate can ever be trusted
# to do. Call the KL gate's remaining budget B — about 2.5e-3, conservatively,
# after the clone's own ~1.9e-3. A defect touching a share f of components at
# log-likelihood error d already fails the KL gate when f * k3(d) > B, so the
# tail gate only earns its place on defects with f above the bound but
# f * k3(d) still under B, which caps the useful bound at B / k3(threshold).
# The bound must also clear the measured floor. Both at once are governed by
# floor(t) * k3(t), and what that product does is decided by a race: the floor
# decays faster than k3 grows from half a nat through 2.5, so the product
# improves, from 4.0e-4 at half a nat to 3.7e-4 at one, 2.5e-4 at two and
# 1.7e-4 at 2.5 — and then the two rates cross, 1.81x against k3's 1.85x over
# the half nat from 2.5 to 3, and the product goes flat within 4% through three
# and four nats. So 2.5 nats is the knee, and the knee is exactly where those
# rates meet. Below it the product is still sliding and the gate is weaker for
# no reason; above it there is nothing further to gain on this metric, and the
# floor grows too sparse to calibrate against — five components at four nats
# cannot support a three-wave spread estimate, and this whole method depends on
# a *measured* floor. At no threshold does bf16 turn this into a strong
# independent detector; the honest most it buys is an extension of the mean's
# reach to defects too concentrated for a mean to see.
#
# 2.5 nats with a 2e-4 bound is pinned from both sides. Above, by redundancy:
# the ceiling is the KL gate's remaining budget over k3(2.5), which is 2.9e-4
# against the conservative B and 3.6e-4 against the clone's actual 1.9e-3, so a
# looser bound would be decoration. Below, by the false-abort rate on the
# *smallest* head, which is what governs it because the gated statistic is a max
# over heads. As fractions the heads are comparable; as counts they are not. At
# production the quantity head carries 75k active components against the unit
# head's 1.45M, so the same 2e-4 bound is a count of 15 there and 290 there.
# Measured tail counts over three waves are 29/43/46 on the unit head and 0/1/1
# on the quantity head: the bound sits 40 Poisson sigma out on the former and
# still needs a 15-against-0.67 excursion on the latter. Over-dispersion is what
# could spoil that, and its mechanism is components within one state sharing
# logits and therefore sharing their rounding — which is why the unit head, at
# 6.3 active components per state, measures 2.1x over-dispersed (three waves
# cannot exclude more, so read Poisson as a floor on the noise rather than a
# model of it), and why the quantity head, at 0.33 per state, has almost nothing
# to cluster with. Tightening to 1e-4 halves both counts and 0/1/1 does not
# support it.
#
# What the gate uniquely covers is f in (2.0e-4, 3.6e-4], roughly 300 to 520
# components — severe corruption too small in extent for the mean to notice.
# That window is not a fixed property of the pair: its ceiling is the KL gate's
# *remaining* budget, so it narrows as the run's own divergence drifts up and
# closes altogether once that reaches 3.3e-3, about 44% of the drift available
# between the clone and the bound. Past that point the tail statistic is
# redundant as a gate and earns its keep as a numerics-regime monitor in
# telemetry, which is the honest description of most of its working life.
#
# What that leaves the KL gate alone responsible for is stated plainly: the
# canonical localized defect, one confined to the final minibatch, covers 2048
# of 320 * 719 = 230,080 states, a 0.89% share, and the KL gate binds there at
# d <= 0.666. So such a defect is visible from a 1.95x likelihood error up,
# where the original 0.247-nat threshold caught it from 1.28x. That band, 1.28x
# to 1.95x on a last-minibatch-sized defect, is what the bf16 update forward
# costs in detection power. The budget is computed against the unit head's
# baseline, the largest of the three, so the other heads are caught earlier and
# this is the worst case.
#
# MAX_FIRST_MINIBATCH_KL governs a different comparison from the three bounds
# above it, and conflating the two is what made the previous value wrong. The
# bounds above measure the rollout's *sampling* likelihoods against the update
# replay. The every-iteration gate never sees the sampling likelihoods: by the
# time the minibatch loop runs, `update_ppo` has overwritten them with
# `replay_behavior_logprobs`, so its ratio is replay against update forward and
# its residual is only a separate compiled graph plus shuffle-dependent batch
# composition. `_replay_to_update_minibatch_kl` measures that quantity, and the
# audit gates this constant against it.
#
# The old 1e-4 came from ~7e-8 measured on a randomly initialized actor, whose
# heads have no confident components at all. A cloned actor has many, and rare
# catastrophic cancellation on near-zero-probability components dominates the
# k3 mean; production training measured 1.98e-3 on the clone's first actor
# minibatch and the gate stopped a 500-iteration run there. Calibrating a
# numerical bound against random init and applying it to a clone has now failed
# three times.
MAX_UPDATE_REPLAY_KL = 5e-3
UPDATE_REPLAY_TAIL_LOGPROB = 2.5
MAX_UPDATE_REPLAY_TAIL_FRACTION = 2e-4
# Measured on the cloned conv actor at production scale, worst of the 113
# minibatches in each of three disjoint waves:
#
#   4.634e-3   7.338e-3   4.060e-2
#
# Unlike its sampling-path siblings, which reproduce to 8% across waves, this
# spans 8.8x. It is an extreme value over a distribution whose mass sits near
# 2e-3 -- production training drew 1.98e-3 on its own first minibatch -- with a
# tail from rare catastrophic cancellation on near-zero-probability components.
# The gate draws one minibatch per iteration, so a 500-iteration run takes 500
# draws where this audit takes 339, and a bound that merely cleared the typical
# draw would fire on a healthy tail with near-certainty.
#
# 1.1e-1 is 2.6x the worst of those 339, the margin MAX_UPDATE_REPLAY_KL takes
# over its own worst wave. It reads loose for a numerical residual, and it is:
# the tail, not the bound, is what is loose. A real staging or replay desync
# moves every minibatch rather than one, so it clears 1.1e-1 by orders of
# magnitude, and `update_replay_mean_minibatch_kl` is the statistic to watch for
# one arriving gradually.
MAX_FIRST_MINIBATCH_KL = 1.1e-1
#: Any shuffle reproduces the gate's residual, so the audit fixes one and the
#: measurement stays comparable between waves and between trees.
_REPLAY_AUDIT_SHUFFLE_SEED = 20260815

#: Share of value targets the critic support may saturate before the run is
#: stopped. The lambda-return adds the critic's own prediction to the reward, so
#: no support width contains it by construction and a small saturated share is
#: ordinary critic error escaping the outermost atom. What this catches is the
#: degenerate end: a critic collapsed onto the edge atom, whose every target is
#: then clipped to a constant label that holds it there.
#:
#: The bound cannot be read off that description, because a categorical critic's
#: mean is itself bounded by its support -- the runaway saturates a share, never
#: the whole batch. Measured over 32 production-length trajectories whose
#: potential is a bounded random walk, with the critic pinned at a constant:
#:
#:     V     0.00  1.00  1.50  2.00  2.10  2.15  2.20
#:     share 0.000 0.000 0.000 0.007 0.072 0.186 0.374
#:
#: So a critic pinned at the outermost atom, the worst state reachable, reaches
#: 0.374 and any bound at or above that is inert. Anything a working critic
#: produces sits at zero with the whole 0.2 of headroom to spare, which leaves a
#: wide band to place this in; 0.05 is an order of magnitude above the first
#: non-zero reading and still fires well before the collapse completes.
MAX_VALUE_TARGET_SATURATED_FRACTION = 0.05


@dataclass(frozen=True)
class PpoConfig:
    # Learning rates follow CleanRL's PPO reference (2.5e-4, Adam eps 1e-5)
    # for both networks; CleanRL additionally anneals linearly to zero, which
    # this pipeline deliberately does not adopt (warmup then constant).
    actor_learning_rate: float = 2.5e-4
    critic_learning_rate: float = 2.5e-4
    lr_warmup_steps: int = 32
    # No weight decay: with decay the AdamW update is not scale-invariant and
    # steadily shrinks norm gains and biases, and the CleanRL reference runs
    # plain Adam. Zero makes AdamW identical to Adam.
    weight_decay: float = 0.0
    epochs: int = 4
    # Total epochs for the critic; the actor participates only in the first
    # `epochs` of them, so values above `epochs` are critic-only refits over
    # the same rollout. None matches the actor epoch count. The actor's trust
    # region binds near one pass per state, but a single critic pass leaves
    # explained variance oscillating and starves rarely-visited states of
    # value estimates.
    critic_epochs: int | None = None
    minibatch_size: int = 2048
    # DAPO's Clip-Higher band, as eps_low 0.2 and eps_high 0.28 rather than
    # PPO's symmetric 0.2 either side. The asymmetry exists to stop entropy
    # collapse: a symmetric band clips a low-probability action's upside at the
    # same ratio as a high-probability one's, which is a much tighter bound on
    # its absolute probability, so exploration dies faster than it should.
    clip_low: float = 0.80
    clip_high: float = 1.28
    # VAPO's lambda_policy = 1 - 1 / (alpha * length), with alpha=0.05 and the
    # competition's fixed 719-action horizon. The critic shares it: VAPO's
    # decoupled lambda-one critic answers a sparse terminal-reward setting
    # where the only unbiased signal is the whole trajectory, and this
    # environment's reward is a dense potential difference at every one of the
    # 719 transitions instead.
    actor_gae_lambda: float = DEFAULT_ACTOR_GAE_LAMBDA
    gamma: float = 1.0
    max_gradient_norm: float = 1.0
    target_kl: float = 0.08
    # BF16 autocast for both update-path forwards. The actor's importance
    # ratio starts at one because `replay_behavior_logprobs` recomputes the
    # behavior side through the update path's forward at the same precision;
    # log_softmax stays fp32 under autocast either way.
    use_bfloat16: bool = True
    # Compile the update-path forward/backward with Inductor. Fusion collapses
    # the launch-bound logprob/surrogate math into a few large kernels while
    # keeping memory eager-like, unlike CUDA-graph capture whose per-minibatch
    # forward+backward recordings pin multiple GiB of activation pools. The
    # resulting importance-ratio drift against stored behavior likelihoods is
    # gated end to end by `update_replay_parity`.
    compile_update: bool = True


@dataclass(frozen=True)
class AdvantageBatch:
    advantages: np.ndarray
    value_targets: np.ndarray
    # The undiscounted suffix return. Nothing trains on it: it is the target the
    # critic used to fit, kept as the one measurement of critic quality the
    # critic cannot move. Explained variance against `value_targets` is scored
    # against a target that contains the prediction, so it improves when the
    # critic merely agrees with itself; against this it does not.
    monte_carlo_returns: np.ndarray
    # Location and scale of the advantages before normalization. The normalized
    # array is zero-mean and unit-variance by construction, so measuring it
    # reports the normalizer rather than the rollout; the scale here is the
    # actual size of the advantage signal, which shrinks as the critic starts
    # explaining the return.
    raw_advantage_mean: float
    raw_advantage_std: float


def generalized_advantage_and_targets(
    rewards: Tensor,
    values: Tensor,
    valid: Tensor,
    actor_gae_lambda: float = DEFAULT_ACTOR_GAE_LAMBDA,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Compute lambda-GAE advantages and the matching lambda-return targets.

    The critic target is the standard PPO return, `advantage + value`, which is
    the lambda-return under the same lambda the actor uses. It is unbiased
    wherever the value function is exact and trades the remaining bias for a
    variance reduction that grows with the horizon: over 719 transitions the
    lambda-one Monte Carlo suffix return accumulates the noise of every later
    action into every earlier state's target, and the dense potential-difference
    reward makes that trade lopsided -- almost all of the return is already
    observable within the lambda's ~36-step effective window.
    """
    if rewards.shape != values.shape or valid.shape != values.shape:
        raise ValueError("rewards, values, and valid mask must have the same shape")
    if values.ndim != 2:
        raise ValueError("values must be [trajectories, time]")
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not math.isfinite(actor_gae_lambda) or not 0.0 <= actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")
    if values.size(1) == 0:
        return torch.zeros_like(values), torch.zeros_like(values)

    valid_mask = valid.bool()
    if bool((~torch.isfinite(rewards) & valid_mask).any()):
        raise ValueError("valid rewards must be finite")
    if bool((~torch.isfinite(values) & valid_mask).any()):
        raise ValueError("valid values must be finite")
    # Invalid padding is semantically absent. Select it away before arithmetic
    # because IEEE NaN multiplied by a zero mask remains NaN and could otherwise
    # contaminate the preceding valid suffix.
    rewards = torch.where(valid_mask, rewards, torch.zeros_like(rewards))
    values = torch.where(valid_mask, values, torch.zeros_like(values))
    valid = valid_mask.to(values.dtype)
    zero_column = torch.zeros_like(values[:, :1])
    next_values = torch.cat((values[:, 1:], zero_column), dim=1)
    next_valids = torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), dim=1)
    deltas = rewards + gamma * next_values * next_valids - values
    running_advantage = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
    advantage_columns: list[Tensor] = []
    for step in range(values.size(1) - 1, -1, -1):
        next_valid = next_valids[:, step]
        running_advantage = (
            deltas[:, step] + gamma * actor_gae_lambda * running_advantage * next_valid
        ) * valid[:, step]
        advantage_columns.append(running_advantage)
    advantages = torch.stack(advantage_columns[::-1], dim=1).to(values.dtype)
    # Padding was selected away from both terms above, so the target stays
    # exactly zero there rather than picking up a stale value prediction.
    value_targets = advantages + values
    return advantages, value_targets


def _validate_config(config: PpoConfig) -> None:
    for name, value in (
        ("actor learning rate", config.actor_learning_rate),
        ("critic learning rate", config.critic_learning_rate),
        ("max gradient norm", config.max_gradient_norm),
        ("target KL", config.target_kl),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and non-negative")
    if config.epochs < 1 or config.minibatch_size < 1:
        raise ValueError("epochs and minibatch size must be positive")
    if config.critic_epochs is not None and config.critic_epochs < config.epochs:
        raise ValueError("critic epochs cannot be fewer than actor epochs")
    if config.lr_warmup_steps < 0:
        raise ValueError("LR warmup steps cannot be negative")
    if not 0.0 < config.clip_low < 1.0 < config.clip_high:
        raise ValueError("clip interval must straddle one")
    if config.gamma != 1.0:
        raise ValueError("Kaggriculture bank-delta rewards require undiscounted gamma=1")
    if not math.isfinite(config.actor_gae_lambda) or not 0.0 <= config.actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")


def _leading_tensor(args: tuple[Any, ...]) -> Tensor:
    head = args[0]
    while isinstance(head, tuple):
        head = head[0]
    return head


def _actor_batch_args(
    architecture: str, staged: dict[str, Tensor], indices: Tensor | slice
) -> tuple[Any, ...]:
    """Build one minibatch of actor forward arguments from staged storage.

    The returned tuple is splatted directly into the actor's forward, so its
    arity is a property of the architecture; the compiled update callables
    therefore take these arguments last, after every fixed factor tensor.
    """
    if architecture == CONV_ENTITY:
        return (
            _batch_tensor(staged["board"], indices, torch.float32),
            _batch_tensor(staged["global_features"], indices, torch.float32),
            _batch_tensor(staged["units"], indices, torch.float32),
            _batch_tensor(staged["unit_positions"], indices, torch.long),
        )
    return (
        StructuredInputs(
            tile_categorical=_batch_tensor(staged["tile_categorical"], indices, torch.long),
            tile_continuous=_batch_tensor(staged["tile_continuous"], indices, torch.float32),
            unit_categorical=_batch_tensor(staged["unit_categorical"], indices, torch.long),
            unit_continuous=_batch_tensor(staged["unit_continuous"], indices, torch.float32),
            unit_active=_batch_tensor(staged["unit_active"], indices, torch.bool),
            unit_tile_gather=_batch_tensor(staged["unit_tile_gather"], indices, torch.long),
            unit_tile_gather_valid=_batch_tensor(
                staged["unit_tile_gather_valid"], indices, torch.bool
            ),
            products=_batch_tensor(staged["products"], indices, torch.float32),
            crops=_batch_tensor(staged["crops"], indices, torch.float32),
            farms=_batch_tensor(staged["farms"], indices, torch.float32),
            town=_batch_tensor(staged["town"], indices, torch.float32),
        ),
    )


def _critic_batch_args(
    architecture: str, staged: dict[str, Tensor], indices: Tensor | slice
) -> tuple[Any, ...]:
    """Build one minibatch of critic forward arguments from staged storage.

    The structured centralized critic reads the actor's viewpoint with the
    opponent's private economy columns concatenated onto the product and crop
    tokens, plus the opponent's unit tokens as attention context.
    """
    if architecture == CONV_ENTITY:
        return (
            _batch_tensor(staged["board"], indices, torch.float32),
            _batch_tensor(staged["critic_features"], indices, torch.float32),
        )
    (actor_inputs,) = _actor_batch_args(architecture, staged, indices)
    inputs = actor_inputs._replace(
        products=torch.cat(
            (
                actor_inputs.products,
                _batch_tensor(staged["critic_products"], indices, torch.float32),
            ),
            dim=-1,
        ),
        crops=torch.cat(
            (actor_inputs.crops, _batch_tensor(staged["critic_crops"], indices, torch.float32)),
            dim=-1,
        ),
    )
    return (
        inputs,
        _batch_tensor(staged["opponent_unit_categorical"], indices, torch.long),
        _batch_tensor(staged["opponent_unit_continuous"], indices, torch.float32),
        _batch_tensor(staged["opponent_unit_active"], indices, torch.bool),
    )


def _replayed_value_chunk(
    critic: Critic,
    autocast_enabled: bool,
    *critic_args: Any,
) -> Tensor:
    """One critic value forward over a chunk of stored state features.

    Runs under the same autocast state as the critic's training minibatches.
    The values feed only GAE advantages, whose tolerance is far looser than
    the ~1e-3 value shift BF16 introduces, and `critic.value` reduces the
    distributional head in fp32 either way.
    """
    with torch.autocast(
        device_type=_leading_tensor(critic_args).device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        critic_logits = critic(*critic_args)
    return critic.value(critic_logits)


@torch.inference_mode()
def replay_behavior_values(
    critic: Critic,
    architecture: str,
    staged: dict[str, Tensor],
    *,
    # Transient fp32 activations scale with the chunk. At production model
    # size 16384 rows would add several GiB right when the staged rollout
    # already occupies the device; 4096 keeps the pass large enough to stay
    # bandwidth-bound without that spike.
    chunk_size: int = 4096,
    compile_model: bool = False,
    autocast_enabled: bool = False,
) -> Tensor:
    """Replay behavior-time value predictions from stored state features.

    The critic is untouched between rollout collection and its first
    optimizer step of the update, so replaying the stored features through
    the critic reproduces the collection-time predictions without paying one
    small synchronous critic forward per environment step. Must run before
    the update mutates the critic. This full-batch pass is half the update's
    wall clock when run eagerly, so on CUDA it routes through the same
    Inductor compilation and autocast state as the rest of the update path.
    """
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    rows = staged["unit_actions"].shape[0]
    device = staged["unit_actions"].device
    forward = (
        _cached_update_callable(critic, "_kaggriculture_value_replay", _replayed_value_chunk)
        if compile_model and device.type == "cuda"
        else _replayed_value_chunk
    )
    was_training = critic.training
    critic.eval()
    try:
        values = [
            forward(
                critic,
                autocast_enabled,
                *_critic_batch_args(architecture, staged, slice(start, start + chunk_size)),
            )
            for start in range(0, rows, chunk_size)
        ]
    finally:
        critic.train(was_training)
    return torch.cat(values).float()


def prepare_advantages(
    rollout: RolloutBatch, values: np.ndarray, config: PpoConfig
) -> AdvantageBatch:
    _validate_config(config)
    if values.shape != rollout.rewards.shape:
        raise ValueError("behavior values must match the rollout reward shape")
    rewards = torch.from_numpy(rollout.rewards).float()
    values = torch.from_numpy(values).float()
    valid = torch.from_numpy(rollout.valid).float()
    advantages, targets = generalized_advantage_and_targets(
        rewards,
        values,
        valid,
        actor_gae_lambda=config.actor_gae_lambda,
        gamma=config.gamma,
    )
    selected = advantages[valid.bool()]
    if selected.numel() == 0:
        raise ValueError("rollout contains no valid states")
    raw_mean = selected.mean()
    raw_std = selected.std(unbiased=False)
    normalized = (advantages - raw_mean) / raw_std.clamp_min(1e-6)
    normalized *= valid
    # Lambda one against a zero reference: the residuals telescope, so the
    # advantage is the exact suffix return and the same recurrence yields it.
    monte_carlo = generalized_advantage_and_targets(
        rewards,
        torch.zeros_like(values),
        valid,
        actor_gae_lambda=1.0,
        gamma=config.gamma,
    )[1]
    return AdvantageBatch(
        advantages=normalized.numpy(),
        value_targets=targets.numpy(),
        monte_carlo_returns=monte_carlo.numpy(),
        raw_advantage_mean=float(raw_mean),
        raw_advantage_std=float(raw_std),
    )


def make_optimizers(
    actor: Actor, critic: Critic, config: PpoConfig
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    _validate_config(config)
    actor_device = next(actor.parameters()).device
    critic_device = next(critic.parameters()).device
    if actor_device != critic_device:
        raise ValueError("actor and critic must use the same device")
    fused = actor_device.type == "cuda"
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.actor_learning_rate,
        eps=1e-5,
        weight_decay=config.weight_decay,
        fused=fused,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        eps=1e-5,
        weight_decay=config.weight_decay,
        fused=fused,
    )
    for optimizer, base_lr in (
        (actor_optimizer, config.actor_learning_rate),
        (critic_optimizer, config.critic_learning_rate),
    ):
        for group in optimizer.param_groups:
            # Optimizer param-group metadata is checkpointed by PyTorch, so the
            # schedule resumes exactly without a separate scheduler object.
            group["base_lr"] = base_lr
            group["warmup_step"] = 0
    return actor_optimizer, critic_optimizer


def _stage_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    flat = torch.from_numpy(array.reshape((-1, *array.shape[2:])))
    # Pinned rollout arenas upload asynchronously; stream ordering keeps the
    # copies safe because every consumer runs on the same stream.
    return flat.to(device=device, non_blocking=flat.is_pinned())


def _batch_tensor(
    staged: Tensor, indices: Tensor | slice, dtype: torch.dtype | None = None
) -> Tensor:
    selected = staged[indices] if isinstance(indices, slice) else staged.index_select(0, indices)
    return selected if dtype is None or selected.dtype == dtype else selected.to(dtype=dtype)


def _balanced_minibatch_slices(sample_count: int, maximum_size: int) -> tuple[slice, ...]:
    """Partition an epoch into near-equal, nonempty minibatches.

    A short final tail would otherwise receive a full optimizer step despite
    its mean loss containing fewer samples. Balancing keeps every sample's
    per-epoch influence approximately equal without dropping any states.
    """
    if sample_count < 1 or maximum_size < 1:
        raise ValueError("sample count and maximum minibatch size must be positive")
    batch_count = math.ceil(sample_count / maximum_size)
    base_size, larger_batches = divmod(sample_count, batch_count)
    slices: list[slice] = []
    start = 0
    for batch in range(batch_count):
        size = base_size + int(batch < larger_batches)
        slices.append(slice(start, start + size))
        start += size
    return tuple(slices)


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    warmup_steps: int,
    found_inf: Tensor | None = None,
) -> None:
    """Advance the warmup schedule and step, optionally skipping on the device.

    `found_inf` is the fused optimizer's own skip signal, the mechanism
    `GradScaler` uses: a nonzero value leaves the parameters, both moment
    buffers, and the per-parameter step counter untouched. Passing a device
    predicate therefore expresses "this minibatch was never applied" without
    the host ever learning which way it went, so no synchronization is needed.
    Callers use it only where a nonzero value is fatal to the run anyway, since
    the warmup counter below advances regardless of the outcome.
    """
    for group in optimizer.param_groups:
        step = int(group.get("warmup_step", 0)) + 1
        group["warmup_step"] = step
        group_base_lr = float(group.get("base_lr", base_learning_rate))
        scale = min(step / warmup_steps, 1.0) if warmup_steps else 1.0
        group["lr"] = group_base_lr * scale
    if found_inf is None:
        optimizer.step()
        return
    # Set, step, delete -- the same lifecycle `GradScaler` uses. The optimizer
    # reads these through `getattr(..., None)`, so leaving them behind would
    # make an absent skip indistinguishable from a decided one to any other
    # caller, and a stale `grad_scale` of None faults a real scaler's arithmetic.
    optimizer.grad_scale = None
    optimizer.found_inf = found_inf
    try:
        optimizer.step()
    finally:
        del optimizer.grad_scale
        del optimizer.found_inf


def _replayed_component_logprobs(
    actor: Actor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
    autocast_enabled: bool,
    *actor_args: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Replay stored actions through the update-path policy forward.

    This defines the current-likelihood side of the PPO importance ratio in
    every actor minibatch. The behavior side is produced at unchanged weights
    by `_replayed_selected_logprobs`, which runs the identical forward,
    masking, fp32 log_softmax, and gather math minus the entropy branch, so
    the ratio starts at one up to numerics. The residual — separate Inductor
    graphs when compiled, and minibatch composition that differs by the
    update loop's shuffle (row counts differ by at most one) — is observed
    directly by the `first_minibatch_approx_kl` metric and gated at
    `MAX_FIRST_MINIBATCH_KL`. Autocast keeps log_softmax in fp32 by policy,
    so the returned log-likelihoods are full precision either way.
    """
    with torch.autocast(
        device_type=unit_actions.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        actor_output = actor(*actor_args)
        return component_logprobs(
            actor_output,
            actor.quantity_logits(actor_output.market_quantity_context, market_kinds),
            unit_actions,
            market_kinds,
            market_quantities,
            unit_masks,
            kind_masks,
            quantity_masks,
            validate_masks=False,
        )


def _replayed_selected_logprobs(
    actor: Actor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
    autocast_enabled: bool,
    *actor_args: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    """Entropy-free `_replayed_component_logprobs` for full-batch replays.

    The behavior replay and the parity audit sweep every valid state but use
    only the gathered log-likelihoods, so this variant skips the per-head
    entropy reductions the minibatch objective needs for its metrics.
    """
    with torch.autocast(
        device_type=unit_actions.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        actor_output = actor(*actor_args)
        return component_selected_logprobs(
            actor_output,
            actor.quantity_logits(actor_output.market_quantity_context, market_kinds),
            unit_actions,
            market_kinds,
            market_quantities,
            unit_masks,
            kind_masks,
            quantity_masks,
            validate_masks=False,
        )


@torch.no_grad()
def replay_behavior_logprobs(
    actor: Actor,
    architecture: str,
    staged: dict[str, Tensor],
    valid_indices: np.ndarray,
    *,
    minibatch_size: int,
    autocast_enabled: bool,
    compile_model: bool = False,
) -> dict[str, Tensor]:
    """Recompute behavior log-likelihoods through the update-path forward.

    The rollout path samples actions from logits produced by a differently
    compiled (and differently batched) forward, so its recorded likelihoods
    differ from the update path's by kernel-selection noise. Recomputing them
    here at unchanged weights, with the update path's forward and logprob
    math at the update's precision, starts the importance ratio at one up to
    numerics — which is what makes a reduced-precision update forward legal.
    (The match is not bit-exact: the update loop shuffles its balanced
    minibatches so a row's batch size can differ by one, and the compiled
    replay and minibatch graphs are separate Inductor artifacts. That
    residual is measured by `first_minibatch_approx_kl` and gated at
    `MAX_FIRST_MINIBATCH_KL`.) The rollout-vs-update divergence becomes an
    off-policy sampling bias instead of a ratio error; `update_replay_parity`
    measures exactly that divergence.

    Must run before the first actor optimizer step. `torch.no_grad` rather
    than inference mode: the outputs are later gathered inside the autograd
    minibatch graph, which inference tensors do not permit.
    """
    if minibatch_size < 1:
        raise ValueError("minibatch size must be positive")
    if valid_indices.size == 0:
        raise ValueError("rollout contains no valid states")
    device = staged["unit_actions"].device
    replay = (
        _cached_update_callable(actor, "_kaggriculture_logprob_replay", _replayed_selected_logprobs)
        if compile_model and device.type == "cuda"
        else _replayed_selected_logprobs
    )
    rows = staged["unit_actions"].shape[0]
    replayed = {
        "old_unit_logprobs": torch.zeros(
            (rows, staged["unit_actions"].shape[1]), dtype=torch.float32, device=device
        ),
        "old_market_kind_logprobs": torch.zeros(
            (rows, staged["market_kinds"].shape[1]), dtype=torch.float32, device=device
        ),
        "old_market_quantity_logprobs": torch.zeros(
            (rows, staged["market_quantities"].shape[1]), dtype=torch.float32, device=device
        ),
    }
    ordered = torch.from_numpy(valid_indices).to(device=device)
    for batch_slice in _balanced_minibatch_slices(valid_indices.size, minibatch_size):
        indices = ordered[batch_slice]
        unit_logprobs, kind_logprobs, quantity_logprobs = replay(
            actor,
            _batch_tensor(staged["unit_actions"], indices, torch.long),
            _batch_tensor(staged["market_kinds"], indices, torch.long),
            _batch_tensor(staged["market_quantities"], indices, torch.long),
            _batch_tensor(staged["unit_masks"], indices, torch.bool),
            _batch_tensor(staged["market_kind_masks"], indices, torch.bool),
            _batch_tensor(staged["market_quantity_masks"], indices, torch.bool),
            autocast_enabled,
            *_actor_batch_args(architecture, staged, indices),
        )
        for name, values in (
            ("old_unit_logprobs", unit_logprobs),
            ("old_market_kind_logprobs", kind_logprobs),
            ("old_market_quantity_logprobs", quantity_logprobs),
        ):
            replayed[name].index_copy_(0, indices, values.float())
    return replayed


def _cached_update_callable(module: torch.nn.Module, attribute: str, function):
    """Compile an update-path computation with Inductor, cached per module.

    Rollout collection deliberately uses the fusion-free cudagraphs backend,
    but graph-capturing the update's forward+backward would permanently pin
    every minibatch's activations in private pools. Inductor keeps memory
    eager-like and instead removes launch overhead by fusing the elementwise
    logprob/surrogate math; the numeric drift fusion introduces is bounded by
    the `update_replay_parity` gate. The compiled wrapper is attached outside
    the module hierarchy so checkpoints stay clean.
    """
    compiled = getattr(module, attribute, None)
    if compiled is None:
        compiled = torch.compile(function, fullgraph=True, dynamic=False)
        object.__setattr__(module, attribute, compiled)
    return compiled


def _actor_minibatch_terms(
    actor: Actor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
    unit_active: Tensor,
    kind_active: Tensor,
    quantity_active: Tensor,
    old_unit: Tensor,
    old_kind: Tensor,
    old_quantity: Tensor,
    advantages: Tensor,
    clip_low: float,
    clip_high: float,
    autocast_enabled: bool,
    *actor_args: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One actor minibatch: policy replay plus clipped surrogate reductions.

    Returns device-side (policy objective sum, entropy sum, k3 KL sum, clipped
    count). Normalization by the per-minibatch component count stays outside
    so the varying host integer never enters the captured graph.
    """
    (
        new_unit,
        new_kind,
        new_quantity,
        unit_entropy,
        kind_entropy,
        quantity_entropy,
    ) = _replayed_component_logprobs(
        actor,
        unit_actions,
        market_kinds,
        market_quantities,
        unit_masks,
        kind_masks,
        quantity_masks,
        autocast_enabled,
        *actor_args,
    )
    device = unit_actions.device
    policy_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    kl_sum = torch.zeros((), device=device)
    clipped_sum = torch.zeros((), device=device)
    for new, old, active, entropy in (
        (new_unit, old_unit, unit_active, unit_entropy),
        (new_kind, old_kind, kind_active, kind_entropy),
        (new_quantity, old_quantity, quantity_active, quantity_entropy),
    ):
        component_objective, component_kl, component_clipped = _clipped_surrogate_sums(
            new,
            old,
            advantages,
            active,
            clip_low,
            clip_high,
        )
        policy_sum = policy_sum + component_objective
        entropy_sum = entropy_sum + (entropy.detach() * active).sum()
        kl_sum = kl_sum + component_kl
        clipped_sum = clipped_sum + component_clipped
    return policy_sum, entropy_sum, kl_sum, clipped_sum


def _critic_minibatch_loss(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
) -> tuple[Tensor, Tensor]:
    """One critic minibatch: the distributional loss and the mean it implies.

    The predicted mean rides along because the forward that produced the logits
    is the only place it is free. Scoring the critic's fit to its own target
    otherwise costs a second full-rollout replay -- 230k states at the
    measured 42k states/s, about 15% of an iteration -- to recover numbers the
    update already computed and threw away.
    """
    with torch.autocast(
        device_type=value_targets.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        critic_logits = critic(*critic_args)
    loss = distributional_value_loss(
        critic_logits,
        value_targets,
        critic.support,
        sigma_ratio=critic.config.value_sigma_ratio,
        validate=False,
    ).mean()
    return loss, critic.value(critic_logits).detach()


# `torch.no_grad` rather than inference mode, and the choice is load-bearing
# twice over. Dynamo specializes on the grad context, so an inference-mode
# audit compiles a *second* entry per minibatch shape on the same code object
# `replay_behavior_logprobs` already compiled under no_grad. That doubles the
# cache from four entries to eight against a per-code-object limit of eight,
# which a `fullgraph=True` region overruns as a hard failure rather than a
# fallback. Sharing the context also means the audit executes literally the
# graph the update executes, which is the whole point of an audit that exists
# to compare against it.
@torch.no_grad()
def update_replay_parity(
    actor: Actor,
    rollout: RolloutBatch,
    *,
    minibatch_size: int,
    compile_model: bool = False,
    autocast_enabled: bool = False,
) -> dict[str, float | int]:
    """Measure rollout-sampling versus update-replay likelihood divergence.

    Runs the same staging, minibatch slicing, and policy forward as
    `update_ppo`'s behavior replay and compares its log-likelihoods with the
    likelihoods the rollout sampler actually drew actions from. Since
    `replay_behavior_logprobs` pins the update's importance ratio to one at
    unchanged weights by construction, this difference no longer enters the
    objective as ratio error; it instead bounds the off-policy sampling bias
    between the distribution actions were drawn from and the distribution the
    gradient assumes. Pass the production `use_bfloat16` flag so the audited
    path is the deployed one.

    Two statistics are gated, both means over active components and therefore
    both invariant to rollout size. `update_replay_max_kl` is the k3 divergence
    estimator, an estimate of KL(sampling policy || update policy) under the
    sampling distribution — the bias itself. `update_replay_max_tail_fraction`
    is the share of sampled actions the two paths disagree about by more than
    `UPDATE_REPLAY_TAIL_LOGPROB`, which is what catches a localized defect that
    a mean would dilute. The per-head maxima are retained as diagnostics — they
    locate the worst component when something does go wrong — but they are
    extreme values over hundreds of thousands of samples, so they grow with the
    component count and are not thresholds.
    """
    if minibatch_size < 1:
        raise ValueError("minibatch size must be positive")
    device = next(actor.parameters()).device
    flat_valid = rollout.valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    if valid_indices.size == 0:
        raise ValueError("rollout contains no valid states")
    staged = {name: _stage_tensor(array, device) for name, array in rollout.states.items()}
    staged |= {
        name: _stage_tensor(getattr(rollout, name), device)
        for name in (
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "unit_masks",
            "market_kind_masks",
            "market_quantity_masks",
            "unit_active",
            "market_active",
            "market_quantity_active",
            "old_unit_logprobs",
            "old_market_kind_logprobs",
            "old_market_quantity_logprobs",
        )
    }
    ordered = torch.from_numpy(valid_indices).to(device=device)
    compile_enabled = compile_model and device.type == "cuda"
    replay = (
        _cached_update_callable(actor, "_kaggriculture_logprob_replay", _replayed_selected_logprobs)
        if compile_enabled
        else _replayed_selected_logprobs
    )
    maximum_logprob_error = dict.fromkeys(("unit", "kind", "quantity"), 0.0)
    maximum_ratio_error = dict.fromkeys(("unit", "kind", "quantity"), 0.0)
    active_counts = dict.fromkeys(("unit", "kind", "quantity"), 0)
    # Accumulated in float64 on the host: the per-head sums run to hundreds of
    # thousands of terms whose individual magnitudes are near the float32
    # rounding floor, which is precisely the regime where a float32 running sum
    # loses the quantity being measured.
    kl_sums = dict.fromkeys(("unit", "kind", "quantity"), 0.0)
    tail_counts = dict.fromkeys(("unit", "kind", "quantity"), 0)
    # Joint over all three heads on one minibatch, reported as a diagnostic for
    # how far a 2048-row slice strays from the full-batch mean. It is *not* the
    # every-iteration gate's statistic; see `_replay_to_update_minibatch_kl`.
    worst_minibatch_kl = 0.0
    for batch_slice in _balanced_minibatch_slices(valid_indices.size, minibatch_size):
        indices = ordered[batch_slice]
        minibatch_kl_sum = 0.0
        minibatch_active = 0
        market_kinds = _batch_tensor(staged["market_kinds"], indices, torch.long)
        replayed = replay(
            actor,
            _batch_tensor(staged["unit_actions"], indices, torch.long),
            market_kinds,
            _batch_tensor(staged["market_quantities"], indices, torch.long),
            _batch_tensor(staged["unit_masks"], indices, torch.bool),
            _batch_tensor(staged["market_kind_masks"], indices, torch.bool),
            _batch_tensor(staged["market_quantity_masks"], indices, torch.bool),
            autocast_enabled,
            *_actor_batch_args(rollout.architecture, staged, indices),
        )
        for name, new_logprobs, old_key, active_key in (
            ("unit", replayed[0], "old_unit_logprobs", "unit_active"),
            ("kind", replayed[1], "old_market_kind_logprobs", "market_active"),
            ("quantity", replayed[2], "old_market_quantity_logprobs", "market_quantity_active"),
        ):
            active = _batch_tensor(staged[active_key], indices, torch.bool)
            active_count = int(active.sum())
            active_counts[name] += active_count
            if not active_count:
                continue
            difference = (
                new_logprobs[active].float()
                - _batch_tensor(staged[old_key], indices, torch.float32)[active]
            )
            if not bool(torch.isfinite(difference).all()):
                raise FloatingPointError(f"non-finite {name} update replay difference")
            maximum_logprob_error[name] = max(
                maximum_logprob_error[name], float(difference.abs().max())
            )
            maximum_ratio_error[name] = max(
                maximum_ratio_error[name], float((difference.exp() - 1.0).abs().max())
            )
            # k3, the same estimator the objective's own KL uses. Both heads
            # normalize over the same masked support, so E[exp(d)] is one and
            # this is exactly unbiased for KL(sampling || update) rather than
            # merely a proxy. The plain -d mean is unbiased too but can go
            # negative; k3 cannot, which is what lets it be compared against a
            # one-sided bound. It is not the lower-variance of the two here —
            # under a heavy log-ratio tail k3 is the more variable one — so the
            # tail statistics below, not this mean, carry bug detection.
            # Promote to float64 before expm1: `expm1(d) - d` is d^2/2 to
            # leading order and cancels catastrophically in float32.
            widened = difference.double()
            head_kl_sum = float((torch.expm1(widened) - widened).sum())
            kl_sums[name] += head_kl_sum
            minibatch_kl_sum += head_kl_sum
            minibatch_active += active_count
            tail_counts[name] += int((widened.abs() > UPDATE_REPLAY_TAIL_LOGPROB).sum())
        if minibatch_active:
            worst_minibatch_kl = max(worst_minibatch_kl, minibatch_kl_sum / minibatch_active)
    first_minibatch_kl, mean_first_minibatch_kl = _replay_to_update_minibatch_kl(
        actor,
        rollout,
        staged,
        valid_indices,
        minibatch_size=minibatch_size,
        compile_model=compile_model,
        autocast_enabled=autocast_enabled,
    )
    total_active = sum(active_counts.values())
    total_kl_sum = sum(kl_sums.values())
    replay_kl = {
        name: kl_sums[name] / active_counts[name] if active_counts[name] else 0.0
        for name in kl_sums
    }
    tail_fraction = {
        name: tail_counts[name] / active_counts[name] if active_counts[name] else 0.0
        for name in tail_counts
    }
    return {
        **{
            f"update_replay_{name}_logprob_max_abs_error": value
            for name, value in maximum_logprob_error.items()
        },
        **{
            f"update_replay_{name}_ratio_max_abs_error": value
            for name, value in maximum_ratio_error.items()
        },
        **{f"update_replay_{name}_kl": value for name, value in replay_kl.items()},
        **{f"update_replay_{name}_tail_fraction": value for name, value in tail_fraction.items()},
        **{f"update_replay_{name}_active_count": value for name, value in active_counts.items()},
        "update_replay_max_ratio_error": max(maximum_ratio_error.values()),
        "update_replay_max_kl": max(replay_kl.values()),
        "update_replay_max_tail_fraction": max(tail_fraction.values()),
        # A component-weighted mean of the three heads, so it can never exceed
        # the largest of them and MAX_UPDATE_REPLAY_KL bounds it by
        # construction. Reported so the relationship is visible rather than
        # merely true.
        "update_replay_joint_kl": total_kl_sum / total_active if total_active else 0.0,
        # The same sampling-versus-replay mean on the worst single minibatch.
        # Diagnostic only: it says how much a 2048-row slice of a heavy-tailed
        # per-component distribution strays from the full-batch mean, which is
        # what makes the tail statistics rather than the mean the bug detector.
        "update_replay_minibatch_kl": worst_minibatch_kl,
        # The statistic `MAX_FIRST_MINIBATCH_KL` actually bounds. It is a
        # different comparison from every number above, which is easy to miss:
        # those measure the *sampling* likelihoods against the update replay,
        # while the every-iteration gate never sees the sampling likelihoods at
        # all -- `update_ppo` overwrites them with the replay, so its ratio is
        # replay against update forward and its residual is only compiled-graph
        # and shuffle-dependent batch composition.
        "update_replay_first_minibatch_kl": first_minibatch_kl,
        # The same residual averaged over every minibatch instead of maximized.
        # It is the stable half of the pair: the maximum is an extreme value over
        # a distribution whose mass sits orders of magnitude below its tail, so
        # the mean is what shows a real desync arriving, and the gap between the
        # two is what shows the tail is only a tail.
        "update_replay_mean_minibatch_kl": mean_first_minibatch_kl,
    }


def _replay_to_update_minibatch_kl(
    actor: Actor,
    rollout: RolloutBatch,
    staged: dict[str, Tensor],
    valid_indices: np.ndarray,
    *,
    minibatch_size: int,
    compile_model: bool,
    autocast_enabled: bool,
) -> tuple[float, float]:
    """Worst and mean per-minibatch KL between behavior replay and update forward.

    This is the quantity `MAX_FIRST_MINIBATCH_KL` bounds, and it is not any of
    the sampling-versus-replay numbers this module's other statistics report.
    `update_ppo` replaces the rollout's sampling likelihoods with a replay
    through the update path, so its importance ratio starts at one by
    construction and the residual its gate observes comes only from a separate
    compiled graph and a shuffled batch composition. Auditing a sampling-path
    number against that bound compares two different quantities that happen to
    sit at a similar magnitude on a cloned actor.

    Both sides are therefore produced the way the update produces them: the
    replay through `replay_behavior_logprobs`, the comparison through the same
    `_actor_minibatch_terms` callable and the same permuted slicing. Advantages
    are zero because the k3 sum does not depend on them, and every minibatch of
    one epoch is measured rather than only the first, since the gate draws one
    at random each iteration and the worst draw is the one that has to clear.
    """
    device = next(actor.parameters()).device
    compile_enabled = compile_model and device.type == "cuda"
    # The k3 sum ignores both the advantages and the clip bounds, so the
    # defaults stand in for a config this audit is not otherwise given.
    clip = PpoConfig()
    replayed = replay_behavior_logprobs(
        actor,
        rollout.architecture,
        staged,
        valid_indices,
        minibatch_size=minibatch_size,
        autocast_enabled=autocast_enabled,
        compile_model=compile_model,
    )
    staged = staged | replayed
    flat_valid_size = rollout.valid.size
    flat_component_counts = (
        rollout.unit_active.reshape(flat_valid_size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_active.reshape(flat_valid_size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_quantity_active.reshape(flat_valid_size, -1).sum(axis=1, dtype=np.int64)
    )
    terms = (
        _cached_update_callable(actor, "_kaggriculture_update_terms", _actor_minibatch_terms)
        if compile_enabled
        else _actor_minibatch_terms
    )
    # The permutation only has to be *a* shuffle, not the training run's: the
    # residual comes from minibatches being composed differently than the replay
    # composed them, and any shuffle does that.
    shuffled = np.random.default_rng(_REPLAY_AUDIT_SHUFFLE_SEED).permutation(valid_indices)
    shuffled_device = torch.from_numpy(shuffled).to(device=device)
    zero_advantages = torch.zeros(minibatch_size, dtype=torch.float32, device=device)
    worst = 0.0
    total = 0.0
    measured = 0
    for batch_slice in _balanced_minibatch_slices(shuffled.size, minibatch_size):
        host_indices = shuffled[batch_slice]
        indices = shuffled_device[batch_slice]
        component_count = max(1, int(flat_component_counts[host_indices].sum()))
        # Grad is re-enabled inside this no_grad audit deliberately. Dynamo
        # specializes on the grad context, so measuring under no_grad would both
        # compile a second entry for a code object already at two of its eight
        # and -- far worse for an audit whose subject is compiled-graph
        # divergence -- execute a different graph from the one the update runs.
        # No backward follows, so nothing frees the activations on its own and
        # they are dropped explicitly at the end of the loop body instead. That
        # placement is load-bearing: rebinding here would evaluate the next
        # minibatch's forward *before* releasing this one's graph, holding two
        # at once, and the update this mirrors only ever holds one.
        with torch.enable_grad():
            policy_sum, entropy_sum, kl_sum, clipped_sum = terms(
                actor,
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
                zero_advantages[: indices.numel()],
                clip.clip_low,
                clip.clip_high,
                autocast_enabled,
                *_actor_batch_args(rollout.architecture, staged, indices),
            )
        minibatch_kl = float(kl_sum.detach().double()) / component_count
        del policy_sum, entropy_sum, kl_sum, clipped_sum
        worst = max(worst, minibatch_kl)
        total += minibatch_kl
        measured += 1
    return worst, total / measured if measured else 0.0


def _clipped_surrogate_sums(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    active: Tensor,
    clip_low: float,
    clip_high: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the PPO token objective, non-negative k3 KL, and clipped count."""
    log_ratio = torch.where(active.bool(), new_logprobs.float() - old_logprobs.float(), 0.0)
    expanded_advantage = advantages.float()[:, None]
    effective_log_ratio = torch.where(
        expanded_advantage >= 0.0,
        log_ratio.clamp_max(math.log(clip_high)),
        log_ratio.clamp_min(math.log(clip_low)),
    )
    objective_sum = (effective_log_ratio.exp() * expanded_advantage * active).sum()
    approximate_kl_sum = ((torch.expm1(log_ratio) - log_ratio) * active).sum()
    clipped_sum = (
        ((log_ratio < math.log(clip_low)) | (log_ratio > math.log(clip_high))).to(active.dtype)
        * active
    ).sum()
    return objective_sum, approximate_kl_sum, clipped_sum


def _target_correlation(targets: np.ndarray, predictions: np.ndarray, valid: np.ndarray) -> float:
    """Pearson correlation of critic predictions with their targets.

    Separates the two ways explained variance goes negative. Predictions that
    are uncorrelated noise and predictions that rank states correctly but at
    the wrong scale or offset both score below zero on explained variance;
    they score near zero and near one respectively here, and the fixes are not
    the same. Returns 0.0 when either side is constant, since a correlation
    with a constant is undefined rather than absent.
    """
    selected_targets = targets[valid]
    selected_predictions = predictions[valid]
    if selected_targets.size < 2:
        return 0.0
    target_deviation = selected_targets - selected_targets.mean()
    prediction_deviation = selected_predictions - selected_predictions.mean()
    denominator = float(
        np.sqrt(float((target_deviation**2).sum()) * float((prediction_deviation**2).sum()))
    )
    if denominator < 1e-12:
        return 0.0
    return float((target_deviation * prediction_deviation).sum()) / denominator


def _epoch_value_losses(marks: list[tuple[Tensor, int]]) -> tuple[float, float]:
    """Mean critic loss over the first and last epoch, from cumulative marks.

    A single epoch is its own first and last. Both are returned as 0.0 when the
    critic did not run, which is not a loss of zero but the absence of one, and
    matches how the other critic statistics report an update that never happened.
    """
    if not marks:
        return 0.0, 0.0
    first_total, first_states = marks[0]
    first = float(first_total) / max(1, first_states)
    if len(marks) == 1:
        return first, first
    last_total, last_states = marks[-1]
    previous_total, previous_states = marks[-2]
    span = last_states - previous_states
    last = float(last_total - previous_total) / max(1, span)
    return first, last


def _explained_variance(targets: np.ndarray, predictions: np.ndarray, valid: np.ndarray) -> float:
    selected_targets = targets[valid]
    selected_predictions = predictions[valid]
    variance = float(np.var(selected_targets))
    if variance < 1e-12:
        return 0.0
    return 1.0 - float(np.var(selected_targets - selected_predictions)) / variance


def _fit_moments(device: torch.device) -> dict[str, Tensor]:
    """Zeroed float64 accumulators for one critic epoch's fit statistics."""
    return {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in ("target", "target_square", "residual", "residual_square")
    }


def _accumulate_fit_moments(sums: dict[str, Tensor], targets: Tensor, predictions: Tensor) -> None:
    """Stream one minibatch's target and residual moments into float64 sums."""
    residuals = targets - predictions
    sums["target"] += targets.sum()
    sums["target_square"] += targets.square().sum()
    sums["residual"] += residuals.sum()
    sums["residual_square"] += residuals.square().sum()


def _fit_explained_variance(sums: dict[str, Tensor], states: int) -> float:
    """Explained variance of the critic's own regression, from streamed sums.

    The two explained variances taken from the pre-update replay cannot answer
    whether the regression worked. Against the suffix return the critic is
    scored on a quantity it never fits, and under potential-shaped rewards with
    gamma one that return is the terminal outcome minus the current potential,
    so most of its variance is the game's coin flip and no critic can explain
    it. Against the lambda-return the residual is identically the advantage --
    the target is `advantages + values` and the prediction is those same
    `values` -- so the number rises whenever the critic's predictions merely
    gain variance, agreeing with themselves.

    This one is neither: the targets are fixed before the update and the
    predictions are the critic's own, so a critic that is fitting what it was
    asked to fit drives this up and one that is not cannot, whatever its
    predictions do. It is reported twice, once per scoring epoch, because a
    single reading cannot separate fitting from memorizing -- see the two
    accumulators in `update_ppo`.

    Population variances, matching `_explained_variance`'s `np.var`, taken from
    float64 running sums rather than a retained per-state array.
    """
    if states < 2:
        return 0.0
    count = float(states)
    target_mean = float(sums["target"]) / count
    target_variance = float(sums["target_square"]) / count - target_mean * target_mean
    if target_variance < 1e-12:
        return 0.0
    residual_mean = float(sums["residual"]) / count
    residual_variance = float(sums["residual_square"]) / count - residual_mean * residual_mean
    return 1.0 - residual_variance / target_variance


def _validate_staged_action_masks(staged: dict[str, Tensor], valid: Tensor) -> None:
    """Validate stored categorical support in one staged accelerator pass."""
    flags: list[Tensor] = []
    messages: list[str] = []
    for name, masks_key, actions_key in (
        ("unit", "unit_masks", "unit_actions"),
        ("market kind", "market_kind_masks", "market_kinds"),
        ("market quantity", "market_quantity_masks", "market_quantities"),
    ):
        masks = staged[masks_key]
        actions = staged[actions_key].long()
        if masks.shape[:-1] != actions.shape:
            raise ValueError(f"{name} action and mask shapes do not align")
        categories = masks.shape[-1]
        valid_rows = valid.view(valid.shape[0], *([1] * (actions.ndim - 1)))
        selected = torch.gather(masks, -1, actions.clamp(0, categories - 1).unsqueeze(-1)).squeeze(
            -1
        )
        flags.extend(
            (
                (((actions < 0) | (actions >= categories)) & valid_rows).any(),
                (~masks.any(dim=-1) & valid_rows).any(),
                (~selected & valid_rows).any(),
            )
        )
        messages.extend(
            (
                f"{name} action is outside its categorical support",
                f"{name} mask has no valid category",
                f"{name} action is masked out",
            )
        )
    # One aggregated host readback replaces per-field synchronizing checks.
    failures = torch.stack(flags).cpu()
    for failed, message in zip(failures.tolist(), messages, strict=True):
        if failed:
            raise ValueError(message)


def update_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: PpoConfig,
    *,
    generator: np.random.Generator,
    actor_epochs: int | None = None,
) -> dict[str, float | int]:
    """Replay one rollout with asymmetric, per-component clipped policy updates.

    `actor_epochs` overrides the actor's participation for this call only —
    used by the warm-start critic-first phase, where a freshly initialized
    critic must fit before its advantages are allowed to push a pretrained
    actor. Zero runs a critic-only refit; the behavior-likelihood replay is
    skipped entirely because nothing consumes it.
    """
    _validate_config(config)
    if actor_epochs is None:
        actor_epochs = config.epochs
    elif not 0 <= actor_epochs <= config.epochs:
        raise ValueError("actor epoch override must lie within the configured epochs")
    device = next(actor.parameters()).device
    if next(critic.parameters()).device != device:
        raise ValueError("actor and critic must use the same device")
    flat_valid = rollout.valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    flat_component_counts = (
        rollout.unit_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_quantity_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
    )

    # The complete rollout is reused for several PPO epochs. Stage every array
    # on the accelerator once; repeated NumPy advanced indexing otherwise makes
    # a new host copy and host-to-device transfer for every field/minibatch.
    architecture = rollout.architecture
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
    # Stored categorical support is validated in one staged pass; repeated
    # NumPy sweeps over the multi-gigabyte host rollout would stall the update.
    _validate_staged_action_masks(staged, torch.from_numpy(flat_valid).to(device))
    # Behavior values for GAE are replayed here from the staged features at
    # full batch instead of one small synchronous critic forward per rollout
    # step. The critic still holds exactly the behavior weights at this point.
    autocast_enabled = config.use_bfloat16 and device.type == "cuda"
    compile_enabled = config.compile_update and device.type == "cuda"
    behavior_values = (
        replay_behavior_values(
            critic,
            architecture,
            staged,
            compile_model=config.compile_update,
            autocast_enabled=autocast_enabled,
        )
        .cpu()
        .numpy()
        .reshape(rollout.rewards.shape)
    )
    # Behavior likelihoods are recomputed through the update path itself (same
    # callable, precision, and minibatch partitioning as the loop below), not
    # taken from the rollout's sampling-path logits. The stored rollout
    # likelihoods remain the sampling ground truth that `update_replay_parity`
    # audits this replay against.
    if actor_epochs > 0:
        staged.update(
            replay_behavior_logprobs(
                actor,
                architecture,
                staged,
                valid_indices,
                minibatch_size=config.minibatch_size,
                autocast_enabled=autocast_enabled,
                compile_model=config.compile_update,
            )
        )
    actor.train()
    critic.train()
    prepared = prepare_advantages(rollout, behavior_values, config)
    valid_value_targets = prepared.value_targets[rollout.valid]
    value_support = critic.support.detach().float().cpu().numpy()
    support_widths = np.diff(value_support)
    if (
        not np.isfinite(value_support).all()
        or not (support_widths > 0).all()
        or not np.allclose(support_widths, support_widths[:1])
    ):
        raise ValueError("critic value support must be finite, increasing, and evenly spaced")
    if not np.isfinite(valid_value_targets).all():
        raise ValueError("value targets must be finite")
    # The lambda-return bootstraps off the critic's own prediction, so its range
    # is the reward range plus the critic's rather than the reward range alone;
    # no bounded support can contain it by construction, and every categorical
    # critic saturates the target at the outermost atom for exactly this reason.
    # Saturation is therefore measured rather than fatal: it is the critic's own
    # error escaping the support, and the fraction over time is the signal --
    # killing a run on one excursion would report the same fact by crashing.
    saturated = np.count_nonzero(
        (valid_value_targets < value_support[0]) | (valid_value_targets > value_support[-1])
    )
    value_targets = np.clip(prepared.value_targets, value_support[0], value_support[-1])
    staged["advantages"] = torch.from_numpy(prepared.advantages.reshape(-1)).to(device)
    staged["value_targets"] = torch.from_numpy(value_targets.reshape(-1)).to(device)
    totals = {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "actor_gradient_norm",
            "critic_gradient_norm",
        )
    }
    # Streamed moments of the scoring critic epochs' targets and residuals.
    # Kept on the device in float64 and reduced once at the end, so scoring the
    # regression costs four reductions over a minibatch already resident rather
    # than a retained copy of every prediction.
    first_fit_sums = _fit_moments(device)
    last_fit_sums = _fit_moments(device)
    first_fit_states = 0
    last_fit_states = 0
    total_states = 0
    actor_states = 0
    total_components = 0
    updates = 0
    actor_updates = 0
    completed_epochs = 0
    max_approx_kl = 0.0
    first_minibatch_kl = 0.0
    actor_terms = (
        _cached_update_callable(actor, "_kaggriculture_update_terms", _actor_minibatch_terms)
        if compile_enabled
        else _actor_minibatch_terms
    )
    critic_loss_fn = (
        _cached_update_callable(critic, "_kaggriculture_update_loss", _critic_minibatch_loss)
        if compile_enabled
        else _critic_minibatch_loss
    )
    stop_for_kl = False
    # Guard scalars leave the device through one pinned async copy per ACTOR
    # minibatch. A CUDA event scopes the host wait to that tiny copy, so the
    # KL/finiteness decisions overlap the already-queued critic backward
    # instead of serializing the stream after every actor forward.
    #
    # Critic-only minibatches take no such wait. The host wait exists to enforce
    # the actor's trust region inside the epoch that produced it, and on the
    # production schedule (epochs=1, critic_epochs=4) three of every four epochs
    # have no actor at all -- 219 of 292 minibatches were paying for a decision
    # with no branch to inform. Those gate the critic step with the fused
    # optimizer's own device-side skip and report finiteness at the epoch
    # boundary, which is the first point the outcome can change what runs next.
    guard_host = torch.empty(3, dtype=torch.float64, pin_memory=device.type == "cuda")
    guard_event = torch.cuda.Event() if device.type == "cuda" else None
    critic_nonfinite = torch.zeros((), dtype=torch.float64, device=device)
    # `found_inf` is a fused-implementation facility; the single-tensor and
    # foreach paths assert it is unused. Ask the optimizer that will receive the
    # skip whether it can honour one, rather than inferring it from the device
    # that happened to imply `fused` back in `make_optimizers`.
    critic_step_is_gateable = any(
        group.get("fused", False) for group in critic_optimizer.param_groups
    )

    critic_epochs = config.epochs if config.critic_epochs is None else config.critic_epochs
    # Cumulative marks at each epoch boundary. The reported `value_loss` averages
    # every critic epoch, so it cannot distinguish a critic that predicts fresh
    # rollouts well from one that merely memorizes each batch over four passes.
    # The first epoch is scored on states the critic has never been fit to and
    # the last on states it has seen three times; their difference is what
    # separates those, and both are deltas of a running device total rather than
    # a second accumulator.
    epoch_marks: list[tuple[Tensor, int]] = []
    for epoch_index in range(critic_epochs):
        shuffled = generator.permutation(valid_indices)
        shuffled_device = torch.from_numpy(shuffled).to(device=device)
        for batch_slice in _balanced_minibatch_slices(shuffled.size, config.minibatch_size):
            host_indices = shuffled[batch_slice]
            indices = shuffled_device[batch_slice]
            critic_args = _critic_batch_args(architecture, staged, indices)
            value_targets = _batch_tensor(staged["value_targets"], indices, torch.float32)
            states = indices.numel()

            run_actor = epoch_index < actor_epochs and not stop_for_kl
            if run_actor:
                # Component activity is immutable rollout metadata. Reducing it
                # on the host avoids a CUDA synchronization in every minibatch
                # merely to recover a denominator already known before staging.
                component_count = max(1, int(flat_component_counts[host_indices].sum()))
                unit_actions = _batch_tensor(staged["unit_actions"], indices, torch.long)
                market_kinds = _batch_tensor(staged["market_kinds"], indices, torch.long)
                market_quantities = _batch_tensor(staged["market_quantities"], indices, torch.long)
                unit_masks = _batch_tensor(staged["unit_masks"], indices, torch.bool)
                kind_masks = _batch_tensor(staged["market_kind_masks"], indices, torch.bool)
                quantity_masks = _batch_tensor(staged["market_quantity_masks"], indices, torch.bool)
                unit_active = _batch_tensor(staged["unit_active"], indices, torch.float32)
                kind_active = _batch_tensor(staged["market_active"], indices, torch.float32)
                quantity_active = _batch_tensor(
                    staged["market_quantity_active"], indices, torch.float32
                )
                old_unit = _batch_tensor(staged["old_unit_logprobs"], indices, torch.float32)
                old_kind = _batch_tensor(staged["old_market_kind_logprobs"], indices, torch.float32)
                old_quantity = _batch_tensor(
                    staged["old_market_quantity_logprobs"], indices, torch.float32
                )
                advantages = _batch_tensor(staged["advantages"], indices, torch.float32)

                actor_optimizer.zero_grad(set_to_none=True)
                # Behavior likelihoods were replayed above at unchanged
                # weights through the update path's logprob math, so the ratio
                # starts at one up to numerics (separate compiled graphs and
                # shuffle-dependent batch composition); the first-minibatch KL
                # metric observes that residual.
                policy_sum, entropy_sum, kl_sum, clipped_sum = actor_terms(
                    actor,
                    unit_actions,
                    market_kinds,
                    market_quantities,
                    unit_masks,
                    kind_masks,
                    quantity_masks,
                    unit_active,
                    kind_active,
                    quantity_active,
                    old_unit,
                    old_kind,
                    old_quantity,
                    advantages,
                    config.clip_low,
                    config.clip_high,
                    autocast_enabled,
                    *_actor_batch_args(architecture, staged, indices),
                )
                batch_kl = kl_sum.detach().double() / component_count
                policy_loss = -policy_sum / component_count
                entropy_mean = entropy_sum / component_count
                # Gradients are computed eagerly but the actor is mutated only
                # after the deferred trust-region check below, so the guard
                # semantics stay exact: a violating minibatch is never applied.
                policy_loss.backward()
                actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), config.max_gradient_norm
                ).detach()

            # Target KL constrains only the actor. Keep fitting the critic for
            # every configured epoch even after policy replay is frozen.
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss, predicted_values = critic_loss_fn(
                critic, value_targets, autocast_enabled, *critic_args
            )
            # The first and last critic epochs, accumulated separately. The
            # first scores every state before this update has fitted it, so it
            # reads out of sample; the last scores each one on its fourth pass.
            # A critic that is generalizing keeps the two together, one that is
            # memorizing the batch pulls them apart. The epochs between are
            # never mixed in: averaging predictions from weights three passes
            # apart reports a fit no single critic ever had.
            if epoch_index == 0 or epoch_index == critic_epochs - 1:
                targets = value_targets.double()
                predictions = predicted_values.double()
                if epoch_index == 0:
                    _accumulate_fit_moments(first_fit_sums, targets, predictions)
                    first_fit_states += states
                if epoch_index == critic_epochs - 1:
                    _accumulate_fit_moments(last_fit_sums, targets, predictions)
                    last_fit_states += states
            if run_actor:
                guard_values = torch.stack(
                    (
                        batch_kl,
                        policy_loss.detach().double(),
                        value_loss.detach().double(),
                    )
                )
                guard_host.copy_(guard_values, non_blocking=True)
                if guard_event is not None:
                    guard_event.record()
            value_loss.backward()
            critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), config.max_gradient_norm
            ).detach()

            if run_actor:
                # The event covers only the three-scalar copy, so this wait
                # overlaps the critic backward still executing on the stream.
                if guard_event is not None:
                    guard_event.synchronize()
                batch_kl_value, policy_loss_value, value_loss_value = guard_host.tolist()
                if updates == 0:
                    # At unchanged weights this KL is pure numerics: the drift
                    # between the behavior replay above and this minibatch
                    # forward.
                    first_minibatch_kl = batch_kl_value
                max_approx_kl = max(max_approx_kl, batch_kl_value)
                # Non-finite losses abort training; the already-queued backward
                # of a poisoned minibatch is never observed past this raise.
                if not math.isfinite(policy_loss_value):
                    raise FloatingPointError("non-finite policy loss")
                # The KL belongs to the policy that produced these gradients,
                # so enforce the trust region before mutating that policy.
                if batch_kl_value > config.target_kl:
                    stop_for_kl = True
                else:
                    _optimizer_step(
                        actor_optimizer,
                        config.actor_learning_rate,
                        config.lr_warmup_steps,
                    )
                    totals["policy_loss"] += policy_loss.detach().double() * component_count
                    totals["entropy"] += entropy_mean.detach().double() * component_count
                    totals["approx_kl"] += batch_kl * component_count
                    totals["clip_fraction"] += clipped_sum.detach().double()
                    totals["actor_gradient_norm"] += actor_gradient_norm * states
                    total_components += component_count
                    actor_states += states
                    actor_updates += 1
                if not math.isfinite(value_loss_value):
                    raise FloatingPointError("non-finite critic loss")
                critic_skip = None
            elif not critic_step_is_gateable:
                # Nothing to defer for: an unfused step cannot be skipped on the
                # device, and without a stream to stall the read costs nothing.
                if not math.isfinite(float(value_loss.detach())):
                    raise FloatingPointError("non-finite critic loss")
                critic_skip = None
            else:
                # The only decision this minibatch can inform is whether to abort,
                # and aborting cannot come sooner than the epoch boundary without
                # a host wait. Gate the step instead: a poisoned minibatch leaves
                # the critic, its moments, and its step counter untouched.
                # fp32, which is the only dtype the fused optimizer's skip accepts.
                critic_skip = (~torch.isfinite(value_loss.detach())).float()
                critic_nonfinite += critic_skip
            _optimizer_step(
                critic_optimizer,
                config.critic_learning_rate,
                config.lr_warmup_steps,
                found_inf=critic_skip,
            )

            totals["value_loss"] += value_loss.detach().double() * states
            totals["critic_gradient_norm"] += critic_gradient_norm * states
            total_states += states
            updates += 1
        epoch_marks.append((totals["value_loss"].clone(), total_states))
        # One host read per epoch, covering every critic-only minibatch it ran.
        # Their steps were already gated on the device, so the critic reaching
        # this line has never absorbed a non-finite loss.
        if critic_nonfinite.item():
            raise FloatingPointError("non-finite critic loss")
        completed_epochs += 1

    first_epoch_value_loss, last_epoch_value_loss = _epoch_value_losses(epoch_marks)
    metrics: dict[str, float | int] = {
        "updates": updates,
        "actor_updates": actor_updates,
        "epochs": completed_epochs,
        "states": rollout.state_count,
        "policy_loss": float(totals["policy_loss"] / max(1, total_components)),
        "value_loss": float(totals["value_loss"] / max(1, total_states)),
        "value_loss_first_epoch": first_epoch_value_loss,
        "value_loss_last_epoch": last_epoch_value_loss,
        "entropy": float(totals["entropy"] / max(1, total_components)),
        "approx_kl": float(totals["approx_kl"] / max(1, total_components)),
        "max_approx_kl": max_approx_kl,
        "first_minibatch_approx_kl": first_minibatch_kl,
        "kl_early_stop": int(stop_for_kl),
        "clip_fraction": float(totals["clip_fraction"] / max(1, total_components)),
        "actor_gradient_norm": float(totals["actor_gradient_norm"] / max(1, actor_states)),
        "critic_gradient_norm": float(totals["critic_gradient_norm"] / max(1, total_states)),
        "advantage_mean": prepared.raw_advantage_mean,
        "advantage_std": prepared.raw_advantage_std,
        "value_target_mean": float(prepared.value_targets[rollout.valid].mean()),
        "value_target_std": float(prepared.value_targets[rollout.valid].std()),
        "value_target_min": float(prepared.value_targets[rollout.valid].min()),
        "value_target_max": float(prepared.value_targets[rollout.valid].max()),
        # Every target statistic here, and the suffix-return and lambda-return
        # explained variances and the correlation below, are taken from the
        # unclipped return, so they all describe one quantity -- what the
        # target was, and how much of it the support could not hold.
        # `value_loss` and the two critic-fit explained variances are the
        # exceptions by necessity, since the critic regresses on the saturated
        # copy and scoring a fit against a target no bounded support can reach
        # would measure the support width. During a saturation episode the two
        # conventions diverge -- a fixed-error critic reads about +0.16 higher
        # on the clipped target at this fraction's run-killing bound -- and
        # this fraction is what says so.
        "value_target_saturated_fraction": float(saturated) / float(valid_value_targets.size),
        "actor_gae_lambda": config.actor_gae_lambda,
        "gamma": config.gamma,
        "actor_learning_rate": float(actor_optimizer.param_groups[0]["lr"]),
        "critic_learning_rate": float(critic_optimizer.param_groups[0]["lr"]),
        # Four explained variances against three targets, because one number
        # cannot carry the questions the run has been misread for want of
        # separating.
        #
        # Against the undiscounted suffix return. The critic does not regress
        # on it, and under this environment's potential-shaped reward with
        # gamma one the suffix return telescopes to the terminal outcome minus
        # the current potential -- so its variance is mostly the game's coin
        # flip, and a healthy critic still scores near zero here. Read the
        # correlation below, not this, for whether the critic knows anything
        # about how the game ends: this conflates that correlation with the
        # scale the critic was fitted at, which belongs to a different target.
        "monte_carlo_explained_variance": _explained_variance(
            prepared.monte_carlo_returns, behavior_values, rollout.valid
        ),
        # Against the target actually regressed on, from the pre-update
        # predictions -- which are inside that target, since it is
        # `advantages + values`. The residual is therefore identically the
        # advantage and this is exactly `1 - Var(A)/Var(G_lambda)`: the
        # conventional PPO reading, and one the critic can raise by merely
        # gaining prediction variance.
        "lambda_return_explained_variance": _explained_variance(
            prepared.value_targets, behavior_values, rollout.valid
        ),
        # Against the same target with predictions taken during the update, so
        # the residual is a fit error rather than an algebraic identity. These
        # two are what say whether the regression is working, and the gap
        # between them says whether it is generalizing rather than memorizing
        # the batch: the first epoch scores every state before this update has
        # fitted it, the last scores each one on its fourth pass. With a single
        # configured critic epoch they coincide, that epoch being both.
        "critic_fit_explained_variance_first_epoch": _fit_explained_variance(
            first_fit_sums, first_fit_states
        ),
        "critic_fit_explained_variance_last_epoch": _fit_explained_variance(
            last_fit_sums, last_fit_states
        ),
        # An explained variance alone cannot say why it is what it is, and the
        # suffix-return one was measured at -0.4 to -0.75 across a whole
        # 40-iteration critic warmup, flat, while the distributional
        # cross-entropy fell from 3.38 to 1.84. Those two are consistent with a
        # critic whose predicted distribution is sharpening while the mean taken
        # from it is not, and separating that from a mean that is simply
        # mis-scaled needs the predictions themselves. A collapsed critic shows
        # a near-zero prediction std against the target's; a mis-scaled one
        # shows a std of the wrong magnitude; a critic learning the right shape
        # but the wrong location shows a correlation near one with a displaced
        # mean. One scalar cannot be all three, so all three are recorded -- and
        # the correlation is taken against the suffix return because it is the
        # scale-free half of that reading: it is what the critic knows about how
        # the game ends, with the fitted scale divided out.
        "value_prediction_mean": float(behavior_values[rollout.valid].mean()),
        "value_prediction_std": float(behavior_values[rollout.valid].std()),
        "value_target_correlation": _target_correlation(
            prepared.monte_carlo_returns, behavior_values, rollout.valid
        ),
    }
    return metrics
