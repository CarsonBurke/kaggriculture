"""PPO masked-token update with VAPO's decoupled GAE and lambda-return targets."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from kaggriculture.constants import DEFAULT_REWARD_GAMMA, EPISODE_STEPS
from kaggriculture.latent_dynamics import DecodeContext, DecodeHeads, DecodeMasks
from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    distributional_value_loss,
)
from kaggriculture.optim import NorMuon, route_parameters
from kaggriculture.policy import component_logprobs, component_selected_logprobs
from kaggriculture.provenance import UNCOMPILED_UPDATE_COMPILE_MODE
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.rollout import RolloutBatch
from kaggriculture.structured import (
    StructuredActor,
    StructuredBelief,
    StructuredCritic,
    StructuredInputs,
    refresh_fused_mlp_fp8,
)
from kaggriculture.structured_dynamics import (
    StructuredDynamics,
    StructuredDynamicsTerms,
    structured_horizon_loss,
    structured_window_loss,
)

Critic = DistributionalCritic | StructuredCritic
Actor = FarmActor | StructuredActor

#: VAPO (arXiv:2504.05118) GAE alpha. Length-adaptive GAE would set
#: ``lambda_policy = 1 - 1 / (alpha * length)`` per sequence because LLM
#: responses vary. Kaggriculture episodes are a known 719-action horizon, so
#: the same formula is a constant rather than a per-row schedule.
VAPO_GAE_ALPHA = 0.05
COMPETITION_ACTION_STEPS = EPISODE_STEPS - 1
DEFAULT_ACTOR_GAE_LAMBDA = 1.0 - 1.0 / (VAPO_GAE_ALPHA * COMPETITION_ACTION_STEPS)
#: VAPO / VC-PPO decoupled GAE: the critic regresses on the unbiased
#: lambda-one return. Policy advantages keep the shorter actor lambda.
DEFAULT_CRITIC_GAE_LAMBDA = 1.0

#: Optimizers `make_optimizers` can build, named by `PpoConfig.optimizer`.
_OPTIMIZERS = ("normuon", "adamw")

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
#: stopped. VAPO decoupled GAE fits the critic on the lambda-one return, which
#: does not add the critic's own prediction, but a bounded categorical mean
#: still cannot represent a target outside the atoms. A small saturated share
#: is ordinary error escaping the outermost atom. What this catches is the
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

#: Mean policy entropy per active component, below which the policy is dead.
#:
#: Measured, not chosen from taste. A learning-rate sweep against the native
#: built-ins (`artifacts/probes/trust-cliff.json`) found a second failure mode
#: this pipeline could not see: at 1e-4 the actor converges within a dozen
#: iterations onto passing every turn -- entropy 0.000 nats, final money exactly
#: 3000, the starting bank untouched, and 0.000 score rate against every
#: built-in. Nothing in the telemetry called it: the epoch completes 100% of its
#: minibatches precisely because a deterministic policy has no KL movement to
#: bound, `actor_updates` reads 113 of 113, and money *rose* to its maximum.
#:
#: It is also terminal rather than a phase. The gradient of a clipped surrogate
#: comes from sampled alternatives, so a policy that samples nothing has no
#: signal with which to leave, and the run burns GPU-hours converged on the
#: reward's inaction basin -- passing scores -0.074 against `starter`, where
#: trading badly scores -0.75, so the objective genuinely prefers doing nothing
#: to farming incompetently and PPO is not misbehaving by finding that.
#:
#: The level is a ceiling on the floor, not the floor itself, because it does not
#: survive a warm start from a faithful clone. From-scratch runs that learned
#: measured 0.14 to 0.37 nats and collapsed ones 0.000 to 0.001, so 0.01 sat an
#: order of magnitude clear of both. The v16-KL clones then measured 0.00896 nats
#: at their first actor-active iteration -- below this level while holding NLL
#: 0.001 and 0.99996 unit accuracy, and beating `public-v27` in play. A faithful
#: clone of a sharp teacher legitimately starts sharper than any from-scratch run
#: ever gets, so an absolute floor calibrated on from-scratch entropy refuses the
#: strongest artifacts this project has. At 0.01 nats a component's top action
#: holds about 99.8% of the mass, which is past exploration for a policy that
#: arrived there by converging -- and unremarkable for one that was trained to
#: imitate.
MINIMUM_POLICY_ENTROPY = 0.01

#: Share of its own first actor-active entropy a policy must keep.
#:
#: Paired with the level above as `min(level, fraction * reference)`: whichever
#: is *lower* gates the run. A from-scratch run starting at 0.29 nats is judged
#: against 0.01 exactly as before, since a quarter of its start is looser; a
#: clone starting at 0.00896 is judged against 0.00224 rather than being refused
#: on its first update. `min` rather than `max` because a policy sharpening as it
#: converges is the expected trajectory, not a failure, and no measurement here
#: bounds how much of its starting spread a healthy run may spend -- so the
#: reference may only relax a level that has evidence behind it, never tighten
#: one that does not.
#:
#: The reference is measured at the first iteration the actor actually updates,
#: which is why the warmup exemption is load-bearing rather than cosmetic: a
#: frozen actor reports the entropy it was initialized with, and a critic-warmup
#: window would otherwise calibrate the floor against a policy that has not yet
#: taken a step.
POLICY_ENTROPY_FLOOR_FRACTION = 0.25

#: Entropy at or below which a policy is already collapsed, whatever it started
#: at. Collapsed runs measured 0.000 to 0.001 nats; 0.002 is twice the worst of
#: those. Its one job is to refuse a *reference* inside that range: a floor that
#: is a share of a collapsed start sits below the collapse and switches the gate
#: off for the remaining iterations, which is the failure the gate exists for
#: arriving before it can be measured.
MINIMUM_POLICY_ENTROPY_REFERENCE = 0.002

#: Share of its intended minibatches an actor epoch must actually apply.
#:
#: `actor_updates < 1` was already refused, and that bound is too weak by
#: exactly the amount that matters: a trust region mis-set against the policy's
#: sharpness stops the epoch after the first minibatch, not before it, so the
#: run reports one update and passes. That is what happened -- 66 iterations at
#: 1 of 113 minibatches, 0.9% of each wave, with `approx_kl` reading 6e-4
#: because it is a mean over the minibatches that stepped and almost none did.
#: Nothing in the telemetry looked wrong.
#:
#: The premise that set this at half was that a healthy iteration applies all of
#: them -- 113 of 113 at every rate whose movement fits inside `target_kl`, and
#: 113 of 113 for the 500 iterations of the run this replaces. That premise was
#: measured on waves the actor had almost no gradient on: self-play against its
#: own snapshots. Against opponents it cannot beat, the schedule that learns
#: fastest applies 89% of the epoch on average and 31% on its worst wave, so half
#: forbids the best configuration measured while the pathology it exists to catch
#: reads 1%.
#:
#: 0.15 keeps both properties. It is 2x below the worst wave of the shipped
#: schedule, which is the margin an unlucky draw needs, and 17x above the 0.009
#: that burned 66 iterations -- a gap no partial epoch can cross by accident.
#: The mismatch it guards against is still measurable at the far end of the same
#: sweep: 3.0e-4 against a 0.03 bound applies exactly 1 of 113 on every iteration.
MINIMUM_ACTOR_EPOCH_FRACTION = 0.15

#: Execution modes for the update path's forward+backward, as `torch.compile`
#: `mode=` values plus `eager` for not compiling at all. A boolean cannot name
#: this decision: the same mistake was already made on the collection side,
#: where the one mode a boolean could select turned out to be slower than eager
#: (`ROLLOUT_FORWARD_MODES`). `default` is Inductor's fusion without CUDA
#: graphs, `reduce-overhead` adds graph capture, and the `max-autotune` pair
#: separates benchmarked kernel selection from that capture so a win is
#: attributable to one of them rather than to both at once.
UPDATE_COMPILE_MODES = (
    "eager",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
# The off value lives in `provenance` rather than here, because the submission
# bundle ships `provenance.py` but not this module, so the run-provenance
# validator has to be able to recognise an uncompiled record without importing
# the training stack. `UNCOMPILED_ROLLOUT_FORWARD_MODE` sits there for exactly
# the same reason. Callers needing a boolean derive it as
# `mode != UNCOMPILED_UPDATE_COMPILE_MODE` at their single point of use rather
# than carrying a second switch that could contradict the first.


@dataclass(frozen=True)
class PpoConfig:
    # The critic follows CleanRL's PPO reference (2.5e-4, Adam eps 1e-5), which
    # this pipeline anneals to nothing -- warmup then constant -- rather than
    # linearly to zero. The tenfold NorMuon conversion that briefly sat here
    # (2.5e-3) was the same unmeasured unit change the actor tabulation
    # falsified; the critic never got its own sweep, and the 2.5e-3 step
    # clipped ~9x every minibatch on the economic population runs. The
    # reference rate is the one that has a citation.
    #
    # The actor does not, and cannot: 2.5e-4 is incompatible with `target_kl`
    # once the actor is warm-started from behavior cloning. A full epoch is 113
    # sequential updates over one 230,080-state wave, and a BC-cloned policy is
    # sharp, so it moves far more KL per unit of parameter movement than the
    # from-scratch policy the reference rate was inherited for.
    #
    # Measured three times, and only the third measurement decides it, because
    # the first two used proxies for the state the actor actually starts from.
    # Worst minibatch KL against the stored behavior, and the updates completed
    # of 113:
    #
    #   lr                    1e-12    3e-6    1e-5     1.5e-5   3e-5    2.5e-4
    #   iter-66, self-play    113      --      --       113      113     1
    #     max KL              1.44e-4  --      --       8.4e-3*  2.0e-2  --
    #   iter-66, league-mix   113      --      --       --       113     2
    #     max KL              1.40e-4  --      --       --       1.6e-2  --
    #   iter-40, league-mix   113      113     113      18       4       --
    #     max KL              3.73e-3  7.0e-3  1.58e-2  3.3e-2   3.3e-2  --
    #
    #   (* that column is 1.5e-5 on the first row only; the sweeps do not share
    #    every rate, since each narrowed around the previous one's answer.)
    #
    # The iteration-40 row is the one that counts: it is the checkpoint the actor
    # first updates from, with the critic its own 40 warmup iterations produced,
    # on the wave that ships. The iteration-66 rows understate movement 5.8x --
    # that actor had already taken 66 steps and carried a differently trained
    # critic -- and a run launched on their answer of 3.0e-5 stopped at 4-37 of
    # 113 on its first actor-active iteration.
    #
    # The 1e-12 control is why the readings above it mean anything: it completes
    # all 113 and reports the replay-parity floor, the divergence at unchanged
    # weights. That floor is 1.4e-4 on the trained actor and 3.73e-3 on this one
    # -- 26x larger and 12% of the whole budget -- because a sharp softmax
    # amplifies the same logit noise into far more KL. It is the clearest single
    # number for why a cloned policy cannot use a from-scratch policy's rate.
    #
    # 1.0e-5 was the largest rate whose whole epoch fits inside a 0.03 bound, and
    # shipping on that criterion was the mistake. Fitting the bound is necessary
    # and says nothing about learning: at 1.0e-5 the bound never even binds --
    # worst minibatch 0.024 against a 0.03 budget over 12 iterations -- so the
    # rate was tuned against a constraint that was not the constraint.
    #
    # What decides it is play. Every candidate below ran 12 to 16 iterations from
    # `checkpoint-000040`, drawing the same waves in the same order, on the
    # production lane shape (4 frozen snapshots beside the 3 native built-ins),
    # judged on score rate against `starter` -- the objective -- with entropy as
    # the liveness check and the applied share of the epoch as the operational
    # one (`scripts/probe_schedule_sweep.py`, `artifacts/probes/trust-*.json`):
    #
    #   lr      target_kl  epoch applied  entropy  money  score vs starter
    #   3.0e-6  0.03       100% / 100%    0.169     235   0.000
    #   1.0e-5  0.03        98% /  79%    0.206     367   0.026   <- shipped
    #   1.0e-5  0.10       100% / 100%    0.206     360   0.000
    #   3.0e-5  0.03        22% /   8%    0.291    1074   0.064
    #   3.0e-5  0.10        89% /  31%    0.311     826   0.051
    #   3.0e-5  0.30       100% / 100%    0.161    2383   0.000
    #   6.0e-5  0.30        99% /  87%    0.001    3000   0.000
    #   1.0e-4  0.03        15% /   1%    0.000    1833   0.000
    #   1.0e-4  0.10        49% /   3%    0.000    2993   0.000
    #   1.0e-4  0.30        90% /  34%    0.001    3000   0.000
    #   3.0e-4  0.03         1% /   1%    0.502     551   0.026
    #
    # Two failure modes bracket the answer. Below 3.0e-5 the policy barely moves
    # and never wins. At and above 6.0e-5 it converges onto passing every turn --
    # entropy 0.001, money exactly the untouched 3000 starting bank, and zero
    # score against everything, which is the reward's inaction basin and is
    # terminal (see MINIMUM_POLICY_ENTROPY). Money alone cannot rank these: it is
    # MAXIMIZED by the collapse, so a row is only readable with its entropy.
    #
    # 3.0e-5 is the whole viable band, and its two bounds are tied on the
    # objective: 0.064 against 0.051 is 5 wins against 4 over ~82 games. What
    # separates them is the data budget -- 0.03 discards 78% of every wave it
    # collects, 0.10 discards 11% -- so the bound moves to 0.10 and pays for
    # itself in rollout that is actually used. That reverses the earlier decision
    # to hold 0.03 and move the rate instead, on evidence the earlier decision did
    # not have: `clip_fraction` was already telling us the bound was stricter than
    # the clip band it backs up (0.036 at a barely-engaging 0.80/1.28), and a
    # 12-iteration trajectory at 0.10 ends with the highest entropy of any run
    # measured, rising rather than falling.
    #
    # The tenfold NorMuon conversion below was a hypothesis, not a measurement,
    # and the four-learner economic-reward run falsified it. Starting from the
    # same four clones after the same 40 critic-only iterations:
    #
    #   NorMuon lr  iteration  internal money  public-v27  public-v16
    #   3.0e-4     50         54,088          1/32        5/32
    #   3.0e-5     50         80,923          52/64       64/64
    #   1.0e-5     70         82,587          49/64       64/64
    #
    # External counts use both seats; the two lower-rate rows use 16 held-out
    # games per member and opponent. At 3.0e-4 the first ten actor updates erase
    # a strong clone. 3.0e-5 preserves the economy and wins the strongest
    # measured external row; 1.0e-5 survives longer but gives back three v27
    # games. The entropy floor stopped both conservative runs before a
    # deterministic update could be committed, with no entropy term in the
    # objective. This measurement supersedes the optimizer unit-conversion
    # projection while leaving the older Adam evidence above as history.
    actor_learning_rate: float = 3.0e-5
    critic_learning_rate: float = 2.5e-4
    lr_warmup_steps: int = 32
    # Which optimizer `make_optimizers` builds. `normuon` gives every hidden
    # matrix a spectrally normalized step (Polar Express + NorMuon's low-rank
    # second moment, ported from `modded-nanogpt` in `optim.py`) and leaves the
    # gains, biases, embeddings and logit heads on Adam; `adamw` is the prior
    # element-wise optimizer, kept so the two can be measured against each
    # other on play rather than argued about.
    #
    # The learning rates above mean DIFFERENT things under the two. Adam's step
    # is per-element, so a matrix's relative movement scales with its size; a
    # NorMuon step's Frobenius norm is `lr * sqrt(min(rows, cols))` against a
    # weight norm near `sqrt(fan_out)`, so `lr` IS the relative movement per
    # step. At Adam's 3.0e-5 a 96x96 layer moves about 3e-4 of its norm per
    # step, which is the anchor a NorMuon rate has to be swept around.
    optimizer: str = "normuon"
    # Adam's rate for the parameters NorMuon does not take -- the gains, biases,
    # embeddings and logit heads. Expressed as a multiple of the network's
    # NorMuon rate because Adam's step is absolute and NorMuon's relative, so
    # the ratio is the transferable quantity rather than a second absolute
    # number. 0.35 is `modded-nanogpt`'s own ratio, 0.008 over 0.023, and it
    # matters more than it looks: the heads are what most directly set the
    # predictions, and an earlier guess of 0.1 measurably under-fitted the
    # critic (explained variance 0.050 against AdamW's 0.33 on the same 64-state
    # regression) purely by starving the value head.
    adam_learning_rate_ratio: float = 0.35
    # Muon's defaults, unchanged: momentum 0.95 is the Nesterov coefficient
    # ahead of orthogonalization, and beta2 0.9 the low-rank second moment's
    # decay. `modded-nanogpt` ships both.
    normuon_momentum: float = 0.95
    normuon_beta2: float = 0.9
    epochs: int = 2
    # Total epochs for the critic; the actor participates only in the first
    # `epochs` of them, so values above `epochs` are critic-only refits over
    # the same rollout. None matches the actor epoch count.
    critic_epochs: int | None = None
    # Largest measured update batch with working headroom for the structured
    # n16 model. 2048 rows was 2.2-4.8% faster in the earlier throughput sweep,
    # but used roughly 12 GiB and left the device underfilled. 4096 uses roughly
    # 24 GiB; 8192 exceeds this 31.36-GiB GPU.
    minibatch_size: int = 4096
    # DAPO's Clip-Higher band, as eps_low 0.2 and eps_high 0.28 rather than
    # PPO's symmetric 0.2 either side. The asymmetry exists to stop entropy
    # collapse: a symmetric band clips a low-probability action's upside at the
    # same ratio as a high-probability one's, which is a much tighter bound on
    # its absolute probability, so exploration dies faster than it should.
    clip_low: float = 0.80
    clip_high: float = 1.28
    # The other half of the same fight, and until now the missing half: the
    # clip band above only bounds how fast exploration can be *removed*, while
    # nothing in the objective rewards keeping it. Measured on the run this
    # replaces, mean entropy per active component was 0.1395 nats at the first
    # actor-active iteration -- 3.4% of the unit head's ln(59) maximum, and the
    # entropy of a two-way choice taken 96.9% one way. That policy cannot find
    # a reward it has never sampled.
    #
    # Sampling temperature cannot supply the exploration instead: `rollout.py`
    # rejects any learner temperature other than 1.0, because the replay-parity
    # contract needs the update forward to reproduce the sampler's likelihoods.
    # So the policy's own entropy is the only exploration that exists. The
    # coefficient measured below is zero, leaving entropy as a liveness metric
    # rather than an objective term.
    #
    # Measured, and the answer is zero. Four coefficients ran 12 iterations each
    # from the same warm checkpoint on the same wave sequence, against the native
    # built-ins (`scripts/probe_schedule_sweep.py`, `artifacts/probes/entropy.json`).
    # The mechanism works and is monotone -- terminal entropy 0.294, 0.303, 0.324,
    # 0.371 nats at 0, 0.003, 0.01, 0.03 -- but it buys no play. Money against
    # `starter` over the last six iterations was 649 +/- 74 at zero against
    # 653 +/- 143 at 0.003, a difference of 4 on a standard error of 161, while
    # 0.01 and 0.03 were worse, and 0.03 much worse: 52-341 money over its last
    # six iterations, scoring 0.000 against `starter` in five of them.
    # That is a policy paying for noise.
    #
    # Discount-correct shaping uses this same gamma during collection:
    # nonterminal rewards are gamma * Phi(next) - Phi(current), while terminal
    # bank utility is paid separately. The fixed-horizon discounted return
    # therefore preserves the terminal objective without forcing gamma one.
    #
    # VAPO decoupled GAE (arXiv:2504.05118, following VC-PPO): the policy uses
    # a shorter lambda so advantages stay low-variance; the critic regresses on
    # the lambda-one return so long-horizon reward does not decay as lambda^t.
    # Length-adaptive GAE is not used -- every episode is COMPETITION_ACTION_STEPS
    # long, so actor_gae_lambda is VAPO's formula evaluated at that constant.
    actor_gae_lambda: float = DEFAULT_ACTOR_GAE_LAMBDA
    critic_gae_lambda: float = DEFAULT_CRITIC_GAE_LAMBDA
    gamma: float = DEFAULT_REWARD_GAMMA
    # Measured to bind on EVERY minibatch, which makes this the step-size
    # control and not a safety valve. `scripts/probe_gradient_spectrum.py` over
    # 1264 production-shaped minibatches in four configurations -- BC actor with
    # a fresh critic, BC actor with a trained critic, the iteration-12 actor and
    # critic together, and a doubled 224-game wave -- reports
    # `minibatch_fraction_above_clip` of 1.000 in every partition of every one.
    # Norm medians are 3.81-4.24 with a per-minibatch range of 2.22 to 21.37.
    #
    # So the applied step is `lr * g / ||g||` rather than `lr * g`: the effective
    # learning rate is about a quarter of `actor_learning_rate` and varies about
    # tenfold between minibatches, and `lr_warmup_steps` warms a quantity that
    # clipping then overrides. Whether 1.0 is the right value is a learning
    # question that only a training run answers, so it is left at the value every
    # measurement above was taken under rather than tuned against a proxy. What
    # is recorded here is that the constant is load-bearing: raising it changes
    # the step size on 100% of updates, not on the tail it reads as bounding.
    max_gradient_norm: float = 1.0
    # Restored to 0.03, the CleanRL-adjacent trust region this pipeline shipped
    # from scratch, on the population runs' own telemetry: four members at
    # actor_lr 3.0e-5 held approx_kl at 1e-4 to 2e-4 per iteration with the
    # epoch never stopping early (`runs/pop4-economic-*`), so at the rate the
    # launcher ships the region 0.03 does not bind and the wider 0.10 only
    # buys headroom for an excursion the measured surface does not produce.
    # The earlier tabulated raise to 0.10 was decided on a lane whose actor
    # started from a different BC artifact; it is history, not a constraint.
    target_kl: float = 0.03
    # BF16 autocast for both update-path forwards. The actor's importance
    # ratio starts at one because `replay_behavior_logprobs` recomputes the
    # behavior side through the update path's forward at the same precision;
    # log_softmax stays fp32 under autocast either way.
    use_bfloat16: bool = True
    # Execution mode for the update-path forward/backward, one of
    # `UPDATE_COMPILE_MODES`. Fusion collapses the launch-bound
    # logprob/surrogate math into a few large kernels; the modes above `default`
    # additionally capture CUDA graphs or benchmark kernel selection, and what
    # each one costs in importance-ratio drift against the stored behavior
    # likelihoods is gated end to end by `update_replay_parity`.
    update_compile_mode: str = "default"

    # Training-only typed NextLat objectives. All coefficients default to zero,
    # which leaves construction, optimizer membership, PPO ordering, checkpoint
    # shape, and inference artifacts on their historical paths.
    structured_decision_coefficient: float = 0.0
    structured_patch_coefficient: float = 0.0
    structured_economy_coefficient: float = 0.0
    structured_opponent_summary_coefficient: float = 0.0
    structured_opponent_patch_coefficient: float = 0.0
    structured_decision_horizon: int = 2
    structured_patch_horizon: int = 1
    # The predictor learns on every rollout, including critic-only warmup. It
    # owns an optimizer so its warmup cannot advance the actor optimizer's
    # schedule. The actor sees the auxiliary objective only after the trainer's
    # held-out gate opens, and even then this ratio caps the auxiliary gradient
    # against the PPO gradient before the ordinary global actor clip.
    structured_learning_rate: float | None = None
    structured_predictor_minibatch_size: int | None = None
    structured_actor_gradient_ratio: float = 0.0
    structured_gate_evaluation_windows: int = 4096
    structured_gate_combined_ratio: float = 0.85
    structured_gate_decision_ratio: float = 0.85
    structured_gate_opponent_summary_ratio: float = 0.95
    structured_gate_opponent_patch_ratio: float = 0.98
    structured_gate_patience: int = 2

    @property
    def resolved_structured_learning_rate(self) -> float:
        return (
            self.actor_learning_rate
            if self.structured_learning_rate is None
            else self.structured_learning_rate
        )

    @property
    def resolved_structured_predictor_minibatch_size(self) -> int:
        return (
            self.minibatch_size
            if self.structured_predictor_minibatch_size is None
            else self.structured_predictor_minibatch_size
        )

    @property
    def structured_auxiliary_active(self) -> bool:
        return any(
            (
                self.structured_decision_coefficient,
                self.structured_patch_coefficient,
                self.structured_economy_coefficient,
                self.structured_opponent_summary_coefficient,
                self.structured_opponent_patch_coefficient,
            )
        )


@dataclass(frozen=True)
class AdvantageBatch:
    advantages: np.ndarray
    value_targets: np.ndarray
    # Policy-lambda return ``A_policy + V``. The critic does not regress on it.
    # Explained variance against this target is the conventional PPO identity
    # ``1 - Var(A)/Var(G_lambda)``: it rises when the critic merely agrees with
    # itself. Monte Carlo explained variance against `monte_carlo_returns` is
    # the measurement the critic cannot move. Default critic_gae_lambda is 1, so
    # `value_targets` equal those suffix returns.
    policy_lambda_returns: np.ndarray
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
    gae_lambda: float = DEFAULT_ACTOR_GAE_LAMBDA,
    gamma: float = DEFAULT_REWARD_GAMMA,
) -> tuple[Tensor, Tensor]:
    """Compute lambda-GAE advantages and the matching lambda-return ``A + V``.

    Callers pick the lambda: the actor uses `PpoConfig.actor_gae_lambda`, the
    critic uses `PpoConfig.critic_gae_lambda` (VAPO's decoupled GAE, lambda one
    by default). This helper is the recurrence only.
    """
    if rewards.shape != values.shape or valid.shape != values.shape:
        raise ValueError("rewards, values, and valid mask must have the same shape")
    if values.ndim != 2:
        raise ValueError("values must be [trajectories, time]")
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("GAE lambda must be finite and in [0, 1]")
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
            deltas[:, step] + gamma * gae_lambda * running_advantage * next_valid
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
        ("adam learning rate ratio", config.adam_learning_rate_ratio),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.optimizer not in _OPTIMIZERS:
        raise ValueError(f"unsupported optimizer {config.optimizer!r}, want one of {_OPTIMIZERS}")
    for name, value in (
        ("normuon momentum", config.normuon_momentum),
        ("normuon beta2", config.normuon_beta2),
    ):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1)")
    if config.epochs < 1 or config.minibatch_size < 1:
        raise ValueError("epochs and minibatch size must be positive")
    if config.critic_epochs is not None and config.critic_epochs < config.epochs:
        raise ValueError("critic epochs cannot be fewer than actor epochs")
    if config.lr_warmup_steps < 0:
        raise ValueError("LR warmup steps cannot be negative")
    if not 0.0 < config.clip_low < 1.0 < config.clip_high:
        raise ValueError("clip interval must straddle one")
    if not math.isfinite(config.gamma) or not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not math.isfinite(config.actor_gae_lambda) or not 0.0 <= config.actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")
    if not math.isfinite(config.critic_gae_lambda) or not 0.0 <= config.critic_gae_lambda <= 1.0:
        raise ValueError("critic GAE lambda must be finite and in [0, 1]")
    coefficients = {
        "structured decision": config.structured_decision_coefficient,
        "structured patch": config.structured_patch_coefficient,
        "structured economy": config.structured_economy_coefficient,
        "structured opponent summary": config.structured_opponent_summary_coefficient,
        "structured opponent patch": config.structured_opponent_patch_coefficient,
    }
    for name, value in coefficients.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} coefficient must be finite and nonnegative")
    if config.structured_learning_rate is not None and (
        not math.isfinite(config.structured_learning_rate) or config.structured_learning_rate <= 0.0
    ):
        raise ValueError("structured learning rate must be finite and positive")
    if config.structured_predictor_minibatch_size is not None and (
        not isinstance(config.structured_predictor_minibatch_size, int)
        or isinstance(config.structured_predictor_minibatch_size, bool)
        or config.structured_predictor_minibatch_size < 1
    ):
        raise ValueError("structured predictor minibatch size must be a positive integer")
    if (
        not math.isfinite(config.structured_actor_gradient_ratio)
        or config.structured_actor_gradient_ratio < 0.0
    ):
        raise ValueError("structured actor gradient ratio must be finite and nonnegative")
    if config.structured_gate_evaluation_windows < 1:
        raise ValueError("structured gate evaluation windows must be positive")
    for name, value in (
        ("combined", config.structured_gate_combined_ratio),
        ("decision", config.structured_gate_decision_ratio),
        ("opponent summary", config.structured_gate_opponent_summary_ratio),
        ("opponent patch", config.structured_gate_opponent_patch_ratio),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"structured gate {name} ratio must be finite and in (0, 1]")
    if config.structured_gate_patience < 1:
        raise ValueError("structured gate patience must be positive")
    horizons = (
        config.structured_decision_horizon,
        config.structured_patch_horizon,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in horizons):
        raise ValueError("structured auxiliary horizons must be integers")
    if any(value < 0 for value in horizons):
        raise ValueError("structured auxiliary horizons cannot be negative")
    if config.structured_decision_coefficient and config.structured_decision_horizon < 1:
        raise ValueError("structured decision horizon must be positive when decision KL is active")
    if any(tuple(coefficients.values())[1:]) and config.structured_patch_horizon < 1:
        raise ValueError(
            "structured patch horizon must be positive when feature prediction is active"
        )


def _validate_structured_auxiliary_modules(
    actor: Actor,
    dynamics: StructuredDynamics | None,
    config: PpoConfig,
) -> None:
    active = config.structured_auxiliary_active
    if active and not isinstance(actor, StructuredActor):
        raise ValueError("structured auxiliary coefficients require a structured actor")
    if active != (dynamics is not None):
        state = "requires" if active else "does not admit"
        raise ValueError(f"structured auxiliary configuration {state} a dynamics predictor")
    if (
        dynamics is not None
        and next(dynamics.parameters()).device != next(actor.parameters()).device
    ):
        raise ValueError("actor and structured dynamics must use the same device")


def _leading_tensor(args: tuple[Any, ...]) -> Tensor:
    head = args[0]
    while isinstance(head, tuple):
        head = head[0]
    return head


#: Every actor forward argument, in order, as the state field it is taken from
#: and the dtype the forward requires. One table, because a second copy of it is
#: a staging bug waiting for the next architecture change: a wrong order or a
#: wrong dtype stays silent until the logits are wrong.
_ACTOR_FORWARD_FIELDS: dict[str, tuple[tuple[str, torch.dtype], ...]] = {
    CONV_ENTITY: (
        ("board", torch.float32),
        ("global_features", torch.float32),
        ("units", torch.float32),
        ("unit_positions", torch.long),
    ),
    STRUCTURED: (
        ("tile_categorical", torch.long),
        ("tile_continuous", torch.float32),
        ("unit_categorical", torch.long),
        ("unit_continuous", torch.float32),
        ("unit_active", torch.bool),
        ("unit_tile_gather", torch.long),
        ("unit_tile_gather_valid", torch.bool),
        ("products", torch.float32),
        ("crops", torch.float32),
        ("farms", torch.float32),
        ("town", torch.float32),
    ),
}


def _actor_forward_fields(architecture: str) -> tuple[tuple[str, torch.dtype], ...]:
    """The named architecture's forward fields, or a caller-actionable refusal."""
    try:
        return _ACTOR_FORWARD_FIELDS[architecture]
    except KeyError:
        known = ", ".join(sorted(_ACTOR_FORWARD_FIELDS))
        raise ValueError(f"unknown actor architecture {architecture!r}; known: {known}") from None


def _actor_forward_tuple(architecture: str, batched: dict[str, Tensor]) -> tuple[Any, ...]:
    """Assemble the actor's forward arguments from one batch of its fields.

    The returned tuple is splatted directly into the actor's forward, so its
    arity is a property of the architecture; the compiled update callables
    therefore take these arguments last, after every fixed factor tensor.
    """
    if architecture == CONV_ENTITY:
        return tuple(batched[name] for name, _dtype in _ACTOR_FORWARD_FIELDS[CONV_ENTITY])
    return (StructuredInputs(**batched),)


def _actor_batch_args(
    architecture: str, staged: dict[str, Tensor], indices: Tensor | slice
) -> tuple[Any, ...]:
    """Build one minibatch of actor forward arguments from staged storage."""
    return _actor_forward_tuple(
        architecture,
        {
            name: _batch_tensor(staged[name], indices, dtype)
            for name, dtype in _actor_forward_fields(architecture)
        },
    )


def actor_forward_args(
    architecture: str, states: Mapping[str, np.ndarray], device: torch.device
) -> tuple[Any, ...]:
    """Actor forward arguments for a batch of already-selected states.

    The field order and dtypes the update's own minibatches use, for callers
    holding host arrays already reduced to the rows they want -- scoring several
    actors on one shared sample of states, say -- rather than a staged
    whole-wave dict to be indexed per minibatch. `states` is keyed the way the
    update stages its fields: every architecture's `RolloutBatch.states`, plus
    the shared `unit_active` mask, which the structured actor reads and no
    architecture's state dict carries.
    """
    return _actor_forward_tuple(
        architecture,
        {
            name: torch.from_numpy(states[name]).to(device=device, dtype=dtype)
            for name, dtype in _actor_forward_fields(architecture)
        },
    )


def _critic_batch_args(
    architecture: str,
    staged: dict[str, Tensor],
    indices: Tensor | slice,
    *,
    actor_args: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Build one minibatch of critic forward arguments from staged storage.

    The structured centralized critic reads the actor's viewpoint with the
    opponent's private economy columns concatenated onto the product and crop
    tokens, plus the opponent's unit tokens as attention context.

    When the actor runs on the same minibatch, its already-gathered inputs are
    reused. They are immutable, non-gradient rollout tensors, so this removes a
    second index-select for every shared field, plus any dtype conversion that
    field needed, without coupling either network's autograd graph.
    """
    if architecture == CONV_ENTITY:
        board = (
            _batch_tensor(staged["board"], indices, torch.float32)
            if actor_args is None
            else actor_args[0]
        )
        return (
            board,
            _batch_tensor(staged["critic_features"], indices, torch.float32),
        )
    (actor_inputs,) = (
        _actor_batch_args(architecture, staged, indices) if actor_args is None else actor_args
    )
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
    # Staged rows to replay, in the order given, or every staged row. One
    # population member's critic is asked about that member's rows only.
    states: Tensor | None = None,
    # Transient fp32 activations scale with the chunk. At production model
    # size 16384 rows would add several GiB right when the staged rollout
    # already occupies the device; 4096 keeps the pass large enough to stay
    # bandwidth-bound without that spike.
    chunk_size: int = 4096,
    compile_mode: str = UNCOMPILED_UPDATE_COMPILE_MODE,
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

    `states` narrows the pass to those staged rows and returns one value per
    requested row rather than one per staged row. A population update owns a
    subset of the wave's trajectories, and every other row's prediction would
    be this member's critic reading a state its own policy never visited --
    which the advantage mask discards anyway, so the pass never computes it.
    """
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    row_count = staged["unit_actions"].shape[0] if states is None else int(states.numel())
    device = staged["unit_actions"].device
    forward = _cached_update_callable(
        critic,
        "_kaggriculture_value_replay",
        _replayed_value_chunk,
        _device_compile_mode(compile_mode, device),
    )
    chunks = [
        slice(start, start + chunk_size) if states is None else states[start : start + chunk_size]
        for start in range(0, row_count, chunk_size)
    ]
    was_training = critic.training
    critic.eval()
    try:
        values = [
            forward(critic, autocast_enabled, *_critic_batch_args(architecture, staged, chunk))
            for chunk in chunks
        ]
    finally:
        critic.train(was_training)
    return torch.cat(values).float()


def _owned_valid(rollout: RolloutBatch, rows: np.ndarray | None) -> np.ndarray:
    """The valid-state mask one update owns, restricted to `rows` when given.

    A population wave stores each game's two rows adjacently and they belong to
    two different members, so no storage order makes one member's rows a
    contiguous block. The partition is therefore a mask over the whole
    `(trajectories, steps)` grid instead of a sliced copy of the rollout, whose
    state arrays are the largest allocation in the process. Being a mask, it is
    indifferent to the order of `rows` and to repeats in it.
    """
    if rows is None:
        return rollout.valid
    owned = np.zeros(rollout.valid.shape, dtype=bool)
    owned[rows] = rollout.valid[rows]
    return owned


def _owned_behavior_values(
    critic: Critic,
    architecture: str,
    staged: dict[str, Tensor],
    rollout: RolloutBatch,
    rows: np.ndarray | None,
    *,
    compile_mode: str,
    autocast_enabled: bool,
) -> np.ndarray:
    """Behavior-time value predictions on the rollout grid, replayed for `rows`.

    Rows this update does not own stay exactly zero, which is what the owned
    mask makes of them in every consumer: GAE selects them away, and no metric
    reads them. Restricting the pass rather than the readings is what keeps a
    population's N per-member updates costing one whole-wave critic replay
    between them instead of N.
    """
    device = staged["unit_actions"].device
    horizon = rollout.horizon
    states = None
    if rows is not None:
        owned_rows = np.asarray(rows, dtype=np.int64).reshape(-1, 1)
        flat = (owned_rows * horizon + np.arange(horizon, dtype=np.int64)).reshape(-1)
        states = torch.from_numpy(flat).to(device=device)
    replayed = (
        replay_behavior_values(
            critic,
            architecture,
            staged,
            states=states,
            compile_mode=compile_mode,
            autocast_enabled=autocast_enabled,
        )
        .cpu()
        .numpy()
    )
    if rows is None:
        return replayed.reshape(rollout.rewards.shape)
    grid = np.zeros(rollout.rewards.shape, dtype=replayed.dtype)
    grid[rows] = replayed.reshape(-1, horizon)
    return grid


def prepare_advantages(
    rollout: RolloutBatch,
    values: np.ndarray,
    config: PpoConfig,
    *,
    rows: np.ndarray | None = None,
) -> AdvantageBatch:
    """VAPO decoupled GAE: policy advantages at actor lambda, critic targets at critic lambda.

    `rows` restricts every statistic to those trajectory rows: the location and
    scale the advantages are normalized by are that subset's own, and rows
    outside it come back exactly zero. That restriction is the reason a
    population wave partitions before it updates rather than after -- the
    normalizer is the batch's own advantage standard deviation, so a pooled
    batch would let one member's return scale set another member's step size.
    """
    _validate_config(config)
    if values.shape != rollout.rewards.shape:
        raise ValueError("behavior values must match the rollout reward shape")
    rewards = torch.from_numpy(rollout.rewards).float()
    values = torch.from_numpy(values).float()
    valid = torch.from_numpy(_owned_valid(rollout, rows)).float()
    advantages, policy_lambda_returns = generalized_advantage_and_targets(
        rewards,
        values,
        valid,
        gae_lambda=config.actor_gae_lambda,
        gamma=config.gamma,
    )
    _, critic_targets = generalized_advantage_and_targets(
        rewards,
        values,
        valid,
        gae_lambda=config.critic_gae_lambda,
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
        gae_lambda=1.0,
        gamma=config.gamma,
    )[1]
    return AdvantageBatch(
        advantages=normalized.numpy(),
        value_targets=critic_targets.numpy(),
        policy_lambda_returns=policy_lambda_returns.numpy(),
        monte_carlo_returns=monte_carlo.numpy(),
        raw_advantage_mean=float(raw_mean),
        raw_advantage_std=float(raw_std),
    )


def _initialize_optimizer_schedule(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        # Optimizer param-group metadata is checkpointed by PyTorch, so the
        # schedule resumes exactly without a separate scheduler object.
        group["base_lr"] = base_learning_rate
        group["warmup_step"] = 0


def make_optimizers(
    actor: Actor,
    critic: Critic,
    config: PpoConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    _validate_config(config)
    actor_device = next(actor.parameters()).device
    critic_device = next(critic.parameters()).device
    if actor_device != critic_device:
        raise ValueError("actor and critic must use the same device")
    if config.structured_auxiliary_active and not isinstance(actor, StructuredActor):
        raise ValueError("structured auxiliary coefficients require a structured actor")
    if config.optimizer == "normuon":
        # One learning rate per network drives both halves: the matrices under
        # NorMuon and the gains, biases and heads under Adam. They are not the
        # same quantity -- a NorMuon rate is a RELATIVE step, since the
        # orthogonalized update's Frobenius norm is `lr * sqrt(min(rows, cols))`
        # against a weight norm of about `sqrt(fan_out)`, while an Adam rate is
        # an ABSOLUTE per-element step. `adam_learning_rate_ratio` converts.
        actor_optimizer = NorMuon(
            *route_parameters(actor),
            learning_rate=config.actor_learning_rate,
            adam_learning_rate=config.actor_learning_rate * config.adam_learning_rate_ratio,
            momentum=config.normuon_momentum,
            beta2=config.normuon_beta2,
        )
        critic_optimizer = NorMuon(
            *route_parameters(critic),
            learning_rate=config.critic_learning_rate,
            adam_learning_rate=config.critic_learning_rate * config.adam_learning_rate_ratio,
            momentum=config.normuon_momentum,
            beta2=config.normuon_beta2,
        )
        return actor_optimizer, critic_optimizer
    fused = actor_device.type == "cuda"
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.actor_learning_rate,
        eps=1e-5,
        weight_decay=0.0,
        fused=fused,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        eps=1e-5,
        weight_decay=0.0,
        fused=fused,
    )
    _initialize_optimizer_schedule(actor_optimizer, config.actor_learning_rate)
    _initialize_optimizer_schedule(critic_optimizer, config.critic_learning_rate)
    return actor_optimizer, critic_optimizer


def make_structured_dynamics_optimizer(
    dynamics: StructuredDynamics,
    config: PpoConfig,
) -> torch.optim.Optimizer:
    """Build the predictor-only optimizer with an independent warmup clock."""
    _validate_config(config)
    learning_rate = config.resolved_structured_learning_rate
    if config.optimizer == "normuon":
        return NorMuon(
            *route_parameters(dynamics),
            learning_rate=learning_rate,
            adam_learning_rate=learning_rate * config.adam_learning_rate_ratio,
            momentum=config.normuon_momentum,
            beta2=config.normuon_beta2,
        )
    optimizer = torch.optim.AdamW(
        dynamics.parameters(),
        lr=learning_rate,
        eps=1e-5,
        weight_decay=0.0,
        fused=next(dynamics.parameters()).device.type == "cuda",
    )
    _initialize_optimizer_schedule(optimizer, learning_rate)
    return optimizer


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
    every actor minibatch. The behavior side uses the entropy-free replay at
    unchanged weights with an identical row partition. Autocast keeps
    log_softmax in fp32 by policy.
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
    compile_mode: str = UNCOMPILED_UPDATE_COMPILE_MODE,
) -> dict[str, Tensor]:
    """Recompute behavior likelihoods with the update precision and partition.

    The caller supplies the exact actor-epoch row order. The remaining
    inference-versus-training graph drift is measured by
    `first_minibatch_approx_kl`.
    """
    if minibatch_size < 1:
        raise ValueError("minibatch size must be positive")
    if valid_indices.size == 0:
        raise ValueError("rollout contains no valid states")
    device = staged["unit_actions"].device
    replay = _cached_update_callable(
        actor,
        "_kaggriculture_logprob_replay",
        _replayed_selected_logprobs,
        _device_compile_mode(compile_mode, device),
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


def _device_compile_mode(mode: str, device: torch.device) -> str:
    """Collapse a compile mode to `eager` on any device Inductor cannot serve.

    Every update-path entry point resolves the mode through here exactly once, so
    the CUDA test lives in one place instead of being repeated beside each
    `torch.compile` call. That repetition is what previously let a caller name a
    compiled mode while a stale second switch selected the eager path.
    """
    if mode not in UPDATE_COMPILE_MODES:
        raise ValueError(f"unknown update compile mode {mode!r}")
    return mode if device.type == "cuda" else UNCOMPILED_UPDATE_COMPILE_MODE


def _cached_update_callable(module: torch.nn.Module, attribute: str, function, mode: str):
    """Compile an update-path computation, cached per module AND per mode.

    The cache is keyed by mode because a run can measure several in one process
    -- the calibration chain and `update_replay_parity` both do -- and a single
    slot would hand back the first mode's artifact under a later mode's name,
    silently reporting one configuration's cost as another's.

    Which mode is worth its cost is a measured question, not an assumed one. The
    earlier version of this helper compiled with Inductor's default mode and
    justified refusing CUDA graphs on the claim that capturing the update's
    forward+backward "would permanently pin every minibatch's activations in
    private pools". That is false, and false in the opposite direction. Measured
    by `scripts/profile_update_backends.py` over its measured baseline schedule
    -- 73 actor and 292 critic minibatches at 2048 -- as wall clock / peak
    reserved / compile time:

        eager                        41.444 s   11.34 GiB     0.8 s
        default                      17.107 s   11.34 GiB    34.3 s
        reduce-overhead              17.026 s    8.44 GiB    29.2 s
        max-autotune                 16.152 s    8.43 GiB   522.6 s
        max-autotune-no-cudagraphs   16.377 s   16.76 GiB   204.4 s

    Compilation itself is worth 2.42x and is not optional. `max-autotune` is a
    net loss despite being fastest: 0.955 s per iteration over a 500-iteration
    run saves 8.0 minutes and costs 8.7 minutes compiling, and it is the same
    trap as the collection knob -- a mode that wins the microbenchmark and loses
    the run.

    The memory column above is a warmup artifact, not a footprint, and an earlier
    revision of this docstring drew a conclusion from it: that graph capture
    reserves 2.9 GiB less because the graph pool is reused where the caching
    allocator otherwise fragments. `scripts/sweep_update_batch.py` measures the
    same schedule with a full untimed schedule discarded before
    `reset_peak_memory_stats`, rather than a single warmup minibatch pair, and
    reads 8.45 GiB for `default` against 8.47 GiB for `reduce-overhead` -- equal.
    Its wall clock reproduces the table above to within 0.4%, so the two
    harnesses disagree only about memory, and the 11.34 GiB is first-schedule
    allocator growth that one warmup pair does not reach. Steady-state reserved
    is the same in both modes; `reduce-overhead`'s only measured advantage is a
    lower cold compile (29.0 s against 43.9 s at 4096 rows). Note also that
    `max_memory_allocated` cannot be compared across these modes at all: CUDA
    graph private pools are excluded from it, so capture reports 0.09 GiB
    allocated against 8.47 GiB reserved. Only reserved is comparable.

    What the modes cannot buy is launch overhead, because this path does not pay
    any. Wall clock equals summed device time to within 0.4% at 2048 and 4096
    rows in both compiled modes and in eager, and `reduce-overhead` removes 97.5%
    of the host launch submissions -- 906 launch API calls per actor minibatch
    down to 23 -- for a 0.2% change in wall clock. A compiled actor minibatch is
    906 kernels over 55.4 ms of device time, about 61 us each, so the GPU is
    saturated and the launches hide behind it. Compilation's 2.42x is fusion
    doing less total device work, not fewer launches: eager runs 1967 actor
    kernels over 143.6 ms against compiled's 907 over 56.2 ms. The consequence is
    that this phase shortens only by reducing device work -- fusion, precision,
    architecture, or fewer minibatches -- and not by launch-count engineering.

    The compiled wrapper is attached outside the module hierarchy so checkpoints
    stay clean.
    """
    if mode not in UPDATE_COMPILE_MODES:
        raise ValueError(f"unknown update compile mode {mode!r}")
    if mode == UNCOMPILED_UPDATE_COMPILE_MODE:
        return function
    cache = getattr(module, attribute, None)
    if cache is None:
        cache = {}
        object.__setattr__(module, attribute, cache)
    compiled = cache.get(mode)
    if compiled is None:
        compiled = torch.compile(function, mode=mode, fullgraph=True, dynamic=False)
        cache[mode] = compiled
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
    count). Normalization stays outside so host integers never enter the graph.

    Entropy and KL are telemetry, not objective terms. Detaching their sums
    inside this compiled region preserves their exact forward values while
    keeping their softmax-sized derivative branches and saved intermediates
    out of the actor backward.
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
        entropy_sum = entropy_sum + (entropy * active).sum()
        kl_sum = kl_sum + component_kl
        clipped_sum = clipped_sum + component_clipped
    return policy_sum, entropy_sum.detach(), kl_sum.detach(), clipped_sum


def _critic_logits_and_loss(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
) -> tuple[Tensor, Tensor]:
    """The shared critic forward and distributional objective."""
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
    return loss, critic_logits


def _critic_minibatch_objective(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
) -> Tensor:
    """One critic minibatch without the value telemetry scoring epochs need."""
    loss, _critic_logits = _critic_logits_and_loss(
        critic, value_targets, autocast_enabled, *critic_args
    )
    return loss


def _critic_minibatch_loss(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
) -> tuple[Tensor, Tensor]:
    """One critic minibatch: the distributional loss and the mean it implies.

    The predicted mean rides along because the forward that produced the logits
    is the only place it is free. Callers that do not score this prediction use
    `_critic_minibatch_objective`, avoiding the softmax and support reduction.
    """
    loss, critic_logits = _critic_logits_and_loss(
        critic, value_targets, autocast_enabled, *critic_args
    )
    return loss, critic.value(critic_logits).detach()


def _critic_minibatch_fit_terms(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
) -> tuple[Tensor, Tensor]:
    """Critic loss plus float64 target/residual moments for a scoring epoch.

    Keeping the moment computation inside the compiled region lets Inductor
    consume the predicted values where they are produced instead of returning
    a full minibatch and materializing five eager float64 intermediates.
    """
    loss, predictions = _critic_minibatch_loss(
        critic, value_targets, autocast_enabled, *critic_args
    )
    targets = value_targets.double()
    residuals = targets - predictions.double()
    moments = torch.stack(
        (
            targets.sum(),
            targets.square().sum(),
            residuals.sum(),
            residuals.square().sum(),
        )
    )
    return loss, moments


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
    compile_mode: str,
    autocast_enabled: bool,
    rows: np.ndarray | None = None,
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

    `rows` restricts the audit to those trajectory rows. In a population wave
    every row was sampled by its own member, so replaying the whole wave
    through one member's actor would measure the distance between two policies
    and report it as a staging defect.
    """
    if minibatch_size < 1:
        raise ValueError("minibatch size must be positive")
    device = next(actor.parameters()).device
    flat_valid = _owned_valid(rollout, rows).reshape(-1)
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
    behavior_replayed = {
        "old_unit_logprobs": torch.zeros(
            (staged["unit_actions"].shape[0], staged["unit_actions"].shape[1]),
            dtype=torch.float32,
            device=device,
        ),
        "old_market_kind_logprobs": torch.zeros(
            (staged["market_kinds"].shape[0], staged["market_kinds"].shape[1]),
            dtype=torch.float32,
            device=device,
        ),
        "old_market_quantity_logprobs": torch.zeros(
            (staged["market_quantities"].shape[0], staged["market_quantities"].shape[1]),
            dtype=torch.float32,
            device=device,
        ),
    }
    replay = _cached_update_callable(
        actor,
        "_kaggriculture_logprob_replay",
        _replayed_selected_logprobs,
        _device_compile_mode(compile_mode, device),
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
        for key, values in (
            ("old_unit_logprobs", replayed[0]),
            ("old_market_kind_logprobs", replayed[1]),
            ("old_market_quantity_logprobs", replayed[2]),
        ):
            behavior_replayed[key].index_copy_(0, indices, values.float())
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
        staged | behavior_replayed,
        valid_indices,
        minibatch_size=minibatch_size,
        compile_mode=compile_mode,
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
    compile_mode: str,
    autocast_enabled: bool,
) -> tuple[float, float]:
    """Worst and mean per-minibatch KL between behavior replay and update forward.

    This is the quantity `MAX_FIRST_MINIBATCH_KL` bounds, and it is not any of
    the sampling-versus-replay numbers this module's other statistics report.
    The first side is the no-grad update-path replay already produced by
    `update_replay_parity`; the comparison runs `_actor_minibatch_terms` over
    shuffled minibatches. This audit intentionally preserves the independent
    inference/training graphs whose numerical drift it measures. The optimizer
    itself uses a stricter same-training-graph replay.

    Advantages are zero because the k3 sum does not depend on them, and every
    minibatch of one epoch is measured rather than only the first.
    """
    device = next(actor.parameters()).device
    resolved_mode = _device_compile_mode(compile_mode, device)
    # The k3 sum ignores both the advantages and the clip bounds, so the
    # defaults stand in for a config this audit is not otherwise given.
    clip = PpoConfig()
    flat_valid_size = rollout.valid.size
    flat_component_counts = (
        rollout.unit_active.reshape(flat_valid_size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_active.reshape(flat_valid_size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_quantity_active.reshape(flat_valid_size, -1).sum(axis=1, dtype=np.int64)
    )
    # The audit runs while the collector actor is in eval mode; sharing its
    # compiled callable with the training update would freeze the BF16
    # inference branch and bypass the FP8 projections after actor.train().
    terms = _cached_update_callable(
        actor, "_kaggriculture_update_audit_terms", _actor_minibatch_terms, resolved_mode
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


_FIT_MOMENT_KEYS = ("target", "target_square", "residual", "residual_square")


def _fit_moment_mapping(sums: Tensor) -> dict[str, Tensor]:
    """Name a compiled fit accumulator's four scalar views."""
    if sums.shape != (len(_FIT_MOMENT_KEYS),):
        raise ValueError("fit moment vector has the wrong shape")
    return dict(zip(_FIT_MOMENT_KEYS, sums.unbind(), strict=True))


def _fit_explained_variance(sums: dict[str, Tensor], states: int) -> float:
    """Explained variance of the critic's own regression, from streamed sums.

    The two explained variances taken from the pre-update replay cannot answer
    whether the regression worked. Against the policy lambda-return the residual
    is identically the advantage -- that target is ``A_policy + V`` -- so the
    number rises whenever the critic's predictions merely gain variance. Against
    the Monte Carlo suffix the critic can fail while still looking good on the
    short-horizon identity. With decoupled GAE those two split: the critic
    fits the suffix, and the policy-lambda reading stays the identity.

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


_STRUCTURED_AUXILIARY_METRICS = (
    "decision",
    "decision_one",
    "decision_final",
    "decision_unit",
    "decision_market_kind",
    "decision_market_quantity",
    "patch",
    "patch_one",
    "patch_final",
    "patch_all",
    "patch_changed",
    "patch_unchanged",
    "economy",
    "opponent_summary",
    "opponent_patches",
    "eligible",
    "residual_ratio",
    "residual_own_patches",
    "residual_opponent_patches",
    "residual_opponent_summary",
    "residual_economy_entities",
    "residual_central_latents",
    "residual_unit_decisions",
    "residual_market_decisions",
)


def _structured_transition_order(
    valid: np.ndarray,
    rows: np.ndarray | None,
    horizon: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Shuffle complete contiguous transition windows without touching PPO RNG.

    Each returned row is a flat-index run ``[t, ..., t + horizon]`` from one
    trajectory. A run is admitted only when every state is valid, so neither a
    padding/terminal boundary nor a population row partition can be crossed.
    """
    if valid.ndim != 2:
        raise ValueError("structured transition geometry must be [trajectories, steps]")
    if horizon < 1:
        raise ValueError("structured transition horizon must be positive")
    selected = np.arange(valid.shape[0], dtype=np.int64) if rows is None else np.asarray(rows)
    if (
        selected.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or ((selected < 0) | (selected >= valid.shape[0])).any()
    ):
        raise ValueError("structured transition rows are outside rollout trajectories")
    selected = selected.astype(np.int64, copy=False)
    steps = valid.shape[1]
    source_steps = steps - horizon
    if source_steps < 1:
        raise ValueError("rollout contains no complete structured auxiliary transition")
    eligible = valid[selected, :source_steps].copy()
    for offset in range(1, horizon + 1):
        eligible &= valid[selected, offset : offset + source_steps]
    selected_row, step = np.nonzero(eligible)
    if step.size == 0:
        raise ValueError("rollout contains no complete structured auxiliary transition")
    starts = selected[selected_row] * steps + step
    offsets = np.arange(horizon + 1, dtype=np.int64)
    windows = starts[:, None] + offsets[None, :]
    return windows[generator.permutation(windows.shape[0])]


def _structured_auxiliary_terms(
    actor: StructuredActor,
    dynamics: StructuredDynamics,
    staged: dict[str, Tensor],
    indices: Tensor,
    *,
    steps_per_trajectory: int,
    config: PpoConfig,
    autocast_enabled: bool,
    actor_grad: bool,
    complete_windows: bool,
    belief_indices: Tensor | None = None,
    belief_inverse: Tensor | None = None,
) -> tuple[Tensor, StructuredDynamicsTerms]:
    """Evaluate the configured typed objective on complete transition runs."""
    (inputs,) = _actor_batch_args(STRUCTURED, staged, indices)
    if not isinstance(inputs, StructuredInputs):
        raise TypeError("structured auxiliary requires StructuredInputs")
    factors = {
        "unit_actions": _batch_tensor(staged["unit_actions"], indices, torch.long),
        "market_kinds": _batch_tensor(staged["market_kinds"], indices, torch.long),
        "market_quantities": _batch_tensor(staged["market_quantities"], indices, torch.long),
        "unit_masks": _batch_tensor(staged["unit_masks"], indices, torch.bool),
        "market_kind_masks": _batch_tensor(staged["market_kind_masks"], indices, torch.bool),
        "market_quantity_masks": _batch_tensor(
            staged["market_quantity_masks"], indices, torch.bool
        ),
        "unit_active": _batch_tensor(staged["unit_active"], indices, torch.bool),
        "market_active": _batch_tensor(staged["market_active"], indices, torch.bool),
        "market_quantity_active": _batch_tensor(
            staged["market_quantity_active"], indices, torch.bool
        ),
        "episode_index": torch.div(indices, steps_per_trajectory, rounding_mode="floor"),
        "step": indices.remainder(steps_per_trajectory),
    }
    decision_horizon = (
        config.structured_decision_horizon if config.structured_decision_coefficient else 0
    )
    state_active = any(
        (
            config.structured_patch_coefficient,
            config.structured_economy_coefficient,
            config.structured_opponent_summary_coefficient,
            config.structured_opponent_patch_coefficient,
        )
    )
    patch_horizon = config.structured_patch_horizon if state_active else 0
    with torch.autocast(
        device_type=indices.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        if actor_grad:
            if belief_indices is not None or belief_inverse is not None:
                raise ValueError("actor-gradient auxiliary cannot reuse detached beliefs")
            _, belief = actor.forward_with_belief(inputs)
        elif belief_indices is not None and belief_inverse is not None:
            (belief_inputs,) = _actor_batch_args(STRUCTURED, staged, belief_indices)
            if not isinstance(belief_inputs, StructuredInputs):
                raise TypeError("structured auxiliary requires StructuredInputs")
            with torch.no_grad():
                _, unique_belief = actor.forward_with_belief(belief_inputs)
            belief = StructuredBelief(*(value[belief_inverse] for value in unique_belief))
        elif belief_indices is None and belief_inverse is None:
            with torch.no_grad():
                _, belief = actor.forward_with_belief(inputs)
        else:
            raise ValueError("belief indices and inverse must be supplied together")
        decode = (
            DecodeContext(
                heads=DecodeHeads.from_actor(actor),
                masks=DecodeMasks(
                    unit_masks=factors["unit_masks"],
                    market_kind_masks=factors["market_kind_masks"],
                    market_quantity_masks=factors["market_quantity_masks"],
                    unit_active=factors["unit_active"],
                    market_active=factors["market_active"],
                    market_quantity_active=factors["market_quantity_active"],
                    market_kinds=factors["market_kinds"],
                ),
            )
            if decision_horizon
            else None
        )
        loss_function = structured_window_loss if complete_windows else structured_horizon_loss
        terms = loss_function(
            dynamics,
            belief,
            inputs,
            factors,
            decode=decode,
            decision_horizon=decision_horizon,
            patch_horizon=patch_horizon,
            own_patches_active=bool(config.structured_patch_coefficient),
            economy_active=bool(config.structured_economy_coefficient),
            opponent_summary_active=bool(config.structured_opponent_summary_coefficient),
            opponent_patches_active=bool(config.structured_opponent_patch_coefficient),
        )
        loss = (
            config.structured_decision_coefficient * terms.decision
            + config.structured_patch_coefficient * terms.patch
            + config.structured_economy_coefficient * terms.economy
            + config.structured_opponent_summary_coefficient * terms.opponent_summary
            + config.structured_opponent_patch_coefficient * terms.opponent_patches
        )
    return loss, terms


def _structured_auxiliary_horizon(config: PpoConfig) -> int:
    state_horizon = (
        config.structured_patch_horizon
        if any(
            (
                config.structured_patch_coefficient,
                config.structured_economy_coefficient,
                config.structured_opponent_summary_coefficient,
                config.structured_opponent_patch_coefficient,
            )
        )
        else 0
    )
    return max(
        config.structured_decision_horizon if config.structured_decision_coefficient else 0,
        state_horizon,
    )


def _structured_predictor_phase(
    actor: StructuredActor,
    dynamics: StructuredDynamics,
    dynamics_optimizer: torch.optim.Optimizer,
    staged: dict[str, Tensor],
    windows: np.ndarray,
    *,
    steps_per_trajectory: int,
    config: PpoConfig,
    autocast_enabled: bool,
    auxiliary_terms_fn: Any,
) -> dict[str, float | int]:
    """Measure then fit the predictor without changing or accumulating into the actor."""
    device = next(actor.parameters()).device
    predictor_minibatch_size = config.resolved_structured_predictor_minibatch_size
    windows_per_batch = max(1, predictor_minibatch_size // windows.shape[1])

    def batches(selected: np.ndarray) -> list[np.ndarray]:
        rank = {int(window[0]): position for position, window in enumerate(selected)}
        ordered = selected[np.argsort(selected[:, 0])]
        grouped = [
            ordered[batch_slice]
            for batch_slice in _balanced_minibatch_slices(ordered.shape[0], windows_per_batch)
        ]
        grouped.sort(key=lambda batch: min(rank[int(window[0])] for window in batch))
        return grouped

    def zero_totals() -> dict[str, Tensor]:
        return {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in _STRUCTURED_AUXILIARY_METRICS
        }

    def record(
        totals: dict[str, Tensor],
        terms: StructuredDynamicsTerms,
        weight: int,
    ) -> None:
        for name in _STRUCTURED_AUXILIARY_METRICS:
            totals[name] += getattr(terms, name).detach().double() * weight

    def indices_for(batch: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        host_indices = batch.reshape(-1)
        unique, inverse = np.unique(host_indices, return_inverse=True)
        return (
            torch.from_numpy(host_indices).to(device=device),
            torch.from_numpy(unique).to(device=device),
            torch.from_numpy(inverse).to(device=device),
        )

    evaluation_windows = windows[: config.structured_gate_evaluation_windows]
    preupdate_totals = zero_totals()
    evaluated = 0
    preupdate_loss_total = torch.zeros((), device=device, dtype=torch.float64)
    dynamics.eval()
    with torch.inference_mode():
        for batch in batches(evaluation_windows):
            indices, belief_indices, belief_inverse = indices_for(batch)
            loss, terms = auxiliary_terms_fn(
                actor,
                dynamics,
                staged,
                indices,
                steps_per_trajectory=steps_per_trajectory,
                config=config,
                autocast_enabled=autocast_enabled,
                actor_grad=False,
                complete_windows=True,
                belief_indices=belief_indices,
                belief_inverse=belief_inverse,
            )
            weight = batch.shape[0]
            preupdate_loss_total += loss.detach().double() * weight
            record(preupdate_totals, terms, weight)
            evaluated += weight

    # The actor heads decode predicted latents into decision targets. Freezing
    # their parameters still lets their Jacobian carry gradients from those
    # targets into the predictor, without allocating actor parameter gradients.
    actor_parameters = tuple(actor.parameters())
    actor_requires_grad = tuple(parameter.requires_grad for parameter in actor_parameters)
    for parameter in actor_parameters:
        parameter.requires_grad_(False)
    dynamics.train()
    refresh_fused_mlp_fp8(dynamics)
    training_totals = zero_totals()
    trained = 0
    gradient_norm_total = torch.zeros((), device=device, dtype=torch.float64)
    training_loss_total = torch.zeros((), device=device, dtype=torch.float64)
    training_updates = 0
    actor_states = 0
    predictor_nonfinite = torch.zeros((), device=device, dtype=torch.float32)
    predictor_step_is_gateable = getattr(dynamics_optimizer, "supports_found_inf", False) or any(
        group.get("fused", False) for group in dynamics_optimizer.param_groups
    )
    started = time.perf_counter()
    try:
        for batch in batches(windows):
            indices, belief_indices, belief_inverse = indices_for(batch)
            dynamics_optimizer.zero_grad(set_to_none=True)
            loss, terms = auxiliary_terms_fn(
                actor,
                dynamics,
                staged,
                indices,
                steps_per_trajectory=steps_per_trajectory,
                config=config,
                autocast_enabled=autocast_enabled,
                actor_grad=False,
                complete_windows=True,
                belief_indices=belief_indices,
                belief_inverse=belief_inverse,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                dynamics.parameters(), config.max_gradient_norm
            ).detach()
            finite = torch.isfinite(loss.detach()) & torch.isfinite(gradient_norm)
            if predictor_step_is_gateable:
                predictor_skip = (~finite).float()
                predictor_nonfinite += predictor_skip
            else:
                if not bool(finite):
                    raise FloatingPointError("non-finite structured predictor update")
                predictor_skip = None
            _optimizer_step(
                dynamics_optimizer,
                config.resolved_structured_learning_rate,
                config.lr_warmup_steps,
                found_inf=predictor_skip,
            )
            refresh_fused_mlp_fp8(dynamics, bootstrap_down=False)
            weight = batch.shape[0]
            record(training_totals, terms, weight)
            training_loss_total += loss.detach().double() * weight
            training_updates += 1
            gradient_norm_total += gradient_norm.double() * weight
            actor_states += belief_indices.numel()
            trained += weight
    finally:
        for parameter, requires_grad in zip(actor_parameters, actor_requires_grad, strict=True):
            parameter.requires_grad_(requires_grad)
    elapsed = time.perf_counter() - started
    if predictor_nonfinite.item():
        raise FloatingPointError("non-finite structured predictor update")

    metrics: dict[str, float | int] = {
        "structured_predictor_evaluation_windows": evaluated,
        "structured_predictor_training_windows": trained,
        "structured_predictor_updates": training_updates,
        "structured_predictor_minibatch_size": predictor_minibatch_size,
        "structured_predictor_gradient_norm": float(gradient_norm_total / max(1, trained)),
        "structured_predictor_actor_states": actor_states,
        "structured_predictor_seconds": elapsed,
        "structured_learning_rate": float(dynamics_optimizer.param_groups[0]["lr"]),
        "structured_preupdate_combined": float(preupdate_loss_total / max(1, evaluated)),
        "structured_predictor_combined": float(training_loss_total / max(1, trained)),
    }
    metrics.update(
        {
            f"structured_preupdate_{name}": float(total / max(1, evaluated))
            for name, total in preupdate_totals.items()
        }
    )
    metrics.update(
        {
            f"structured_predictor_{name}": float(total / max(1, trained))
            for name, total in training_totals.items()
        }
    )
    return metrics


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
    rows: np.ndarray | None = None,
    structured_dynamics: StructuredDynamics | None = None,
    structured_dynamics_optimizer: torch.optim.Optimizer | None = None,
    structured_actor_auxiliary: bool = False,
    auxiliary_generator: np.random.Generator | None = None,
) -> dict[str, float | int]:
    """Replay one rollout with asymmetric, per-component clipped policy updates.

    `actor_epochs` overrides the actor's participation for this call only —
    used by the warm-start critic-first phase, where a freshly initialized
    critic must fit before its advantages are allowed to push a pretrained
    actor. Zero runs a critic-only refit; the behavior-likelihood replay is
    skipped entirely because nothing consumes it.

    `rows` restricts the update to those trajectory rows -- a population's
    per-member partition. Every statistic below is then that subset's own, the
    advantage normalizer above all: pooled, one member's return scale would set
    another member's step size. The partition is a row index rather than a
    sliced rollout because a game's two rows belong to two different members,
    so no storage order makes one member's rows a contiguous block and slicing
    would copy the wave's state arrays.
    """
    _validate_config(config)
    _validate_structured_auxiliary_modules(actor, structured_dynamics, config)
    predictor_active = structured_dynamics is not None
    if predictor_active != (structured_dynamics_optimizer is not None):
        raise ValueError("active structured auxiliary requires exactly one predictor optimizer")
    if predictor_active != (auxiliary_generator is not None):
        raise ValueError(
            "active structured auxiliary requires exactly one independent auxiliary generator"
        )
    if structured_actor_auxiliary and (
        not predictor_active or config.structured_actor_gradient_ratio <= 0.0
    ):
        raise ValueError(
            "actor-side structured auxiliary requires an active predictor "
            "and positive gradient ratio"
        )
    if structured_dynamics is not None and structured_dynamics_optimizer is not None:
        actor_optimized = {
            id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]
        }
        dynamics_optimized = {
            id(parameter)
            for group in structured_dynamics_optimizer.param_groups
            for parameter in group["params"]
        }
        dynamics_parameters = {id(parameter) for parameter in structured_dynamics.parameters()}
        if actor_optimized & dynamics_parameters:
            raise ValueError("actor optimizer must not include dynamics parameters")
        if not dynamics_parameters <= dynamics_optimized:
            raise ValueError("predictor optimizer does not include every dynamics parameter")
    if actor_epochs is None:
        actor_epochs = config.epochs
    elif not 0 <= actor_epochs <= config.epochs:
        raise ValueError("actor epoch override must lie within the configured epochs")
    device = next(actor.parameters()).device
    if next(critic.parameters()).device != device:
        raise ValueError("actor and critic must use the same device")
    owned_valid = _owned_valid(rollout, rows)
    flat_valid = owned_valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    if valid_indices.size == 0:
        raise ValueError("rollout contains no valid states")
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
    compile_mode = _device_compile_mode(config.update_compile_mode, device)
    # The fused actor's train-mode FP8 path advances delayed activation scales
    # on every forward, making its likelihood depend on prior minibatches.
    # Eval mode retains gradients but selects the stateless fused BF16 path, so
    # PPO can replay a fixed behavior policy. The measured steady cost is small
    # relative to the correctness and memory failures of a duplicate FP8 graph.
    actor.eval()
    critic.train()
    refresh_fused_mlp_fp8(critic)
    # Predictor-only fitting must leave the actor with neither changed weights
    # nor stale gradient buffers, including throughout critic warmup.
    actor_optimizer.zero_grad(set_to_none=True)
    predictor_metrics: dict[str, float | int] = {}
    auxiliary_windows: np.ndarray | None = None
    structured_terms_fn: Any = None
    if predictor_active:
        assert isinstance(actor, StructuredActor)
        assert structured_dynamics is not None
        assert structured_dynamics_optimizer is not None
        assert auxiliary_generator is not None
        # The predictor's locality-aware batches have dynamic unique-row counts.
        # Compiling this path adds minutes of specialization latency without
        # improving its narrow attention/reduction workload.
        structured_terms_fn = _structured_auxiliary_terms
        auxiliary_windows = _structured_transition_order(
            rollout.valid,
            rows,
            _structured_auxiliary_horizon(config),
            auxiliary_generator,
        )
        predictor_metrics = _structured_predictor_phase(
            actor,
            structured_dynamics,
            structured_dynamics_optimizer,
            staged,
            auxiliary_windows,
            steps_per_trajectory=rollout.valid.shape[1],
            config=config,
            autocast_enabled=autocast_enabled,
            auxiliary_terms_fn=structured_terms_fn,
        )
    behavior_values = _owned_behavior_values(
        critic,
        architecture,
        staged,
        rollout,
        rows,
        compile_mode=compile_mode,
        autocast_enabled=autocast_enabled,
    )
    # Recompute behavior likelihoods in the exact row order reused by every
    # actor epoch. Matching the static partition removes batch-composition
    # numerics between the separate inference and grad-tracking compiled graphs.
    actor_order = generator.permutation(valid_indices) if actor_epochs > 0 else None
    if actor_order is not None:
        staged.update(
            replay_behavior_logprobs(
                actor,
                architecture,
                staged,
                actor_order,
                minibatch_size=config.minibatch_size,
                autocast_enabled=autocast_enabled,
                compile_mode=compile_mode,
            )
        )
    if structured_dynamics is not None:
        structured_dynamics.train()
    prepared = prepare_advantages(rollout, behavior_values, config, rows=rows)
    valid_value_targets = prepared.value_targets[owned_valid]
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
    # The critic target is the decoupled lambda-one return. A bounded
    # categorical mean still cannot represent a target outside its atoms, so
    # saturation is measured rather than fatal: it is error escaping the
    # support, and the fraction over time is the signal.
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
    # The compiled scoring callable reduces each minibatch directly into this
    # four-scalar vector, avoiding full-sized float64 temporaries and replacing
    # four eager accumulator launches with one vector addition.
    first_fit_sums = torch.zeros(len(_FIT_MOMENT_KEYS), device=device, dtype=torch.float64)
    last_fit_sums = torch.zeros_like(first_fit_sums)
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
    actor_auxiliary_active = bool(predictor_active and structured_actor_auxiliary and actor_epochs)
    auxiliary_totals = (
        {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in _STRUCTURED_AUXILIARY_METRICS
        }
        if actor_auxiliary_active
        else {}
    )
    auxiliary_updates = 0
    auxiliary_seconds = 0.0
    auxiliary_cursor = 0
    ppo_actor_gradient_norm_total = torch.zeros((), device=device, dtype=torch.float64)
    actor_auxiliary_raw_norm_total = torch.zeros((), device=device, dtype=torch.float64)
    actor_auxiliary_applied_norm_total = torch.zeros((), device=device, dtype=torch.float64)
    actor_auxiliary_scale_total = torch.zeros((), device=device, dtype=torch.float64)
    actor_terms = _cached_update_callable(
        actor, "_kaggriculture_update_terms", _actor_minibatch_terms, compile_mode
    )
    critic_objective_fn = _cached_update_callable(
        critic,
        "_kaggriculture_update_objective",
        _critic_minibatch_objective,
        compile_mode,
    )
    critic_fit_terms_fn = _cached_update_callable(
        critic,
        "_kaggriculture_update_fit_terms",
        _critic_minibatch_fit_terms,
        compile_mode,
    )
    stop_for_kl = False
    # Guard scalars leave the device through one pinned async copy per ACTOR
    # minibatch. A CUDA event scopes the host wait to that tiny copy, so the
    # KL/finiteness decisions overlap the already-queued critic backward
    # instead of serializing the stream after every actor forward.
    #
    # Critic-only minibatches take no such wait. The host wait exists to enforce
    # the actor's trust region inside the epoch that produced it. On the current
    # 230,080-state schedule (epochs=2, critic_epochs=4, minibatch_size=4096),
    # two of four epochs have no actor -- 114 of 228 minibatches need no host
    # decision. Those gate the critic step with the fused optimizer's own
    # device-side skip and report finiteness at the epoch boundary, which is the
    # first point the outcome can change what runs next.
    guard_host = torch.empty(
        4 if actor_auxiliary_active else 3,
        dtype=torch.float64,
        pin_memory=device.type == "cuda",
    )
    guard_event = torch.cuda.Event() if device.type == "cuda" else None
    critic_nonfinite = torch.zeros((), dtype=torch.float64, device=device)
    # `found_inf` is a fused-implementation facility, which `NorMuon` implements
    # directly and the single-tensor and foreach AdamW paths assert is unused.
    # Ask the optimizer that will receive the skip whether it can honour one,
    # rather than inferring it from the device that happened to imply `fused`.
    critic_step_is_gateable = getattr(critic_optimizer, "supports_found_inf", False) or any(
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
    # What a complete actor epoch would have applied, fixed before the loop so a
    # trust-region stop cannot shrink the denominator it is measured against.
    # Every actor epoch reuses the behavior replay's row partition, keeping the
    # importance-ratio baseline independent of shuffle composition. Critic-only
    # epochs keep independent permutations.
    actor_minibatches_intended = actor_epochs * len(
        _balanced_minibatch_slices(valid_indices.size, config.minibatch_size)
    )
    actor_zero = torch.zeros((), device=device, dtype=torch.float32)
    for epoch_index in range(critic_epochs):
        deferred_critic_finiteness = False
        shuffled = (
            actor_order
            if epoch_index < actor_epochs and actor_order is not None
            else generator.permutation(valid_indices)
        )
        shuffled_device = torch.from_numpy(shuffled).to(device=device)
        for batch_slice in _balanced_minibatch_slices(shuffled.size, config.minibatch_size):
            host_indices = shuffled[batch_slice]
            indices = shuffled_device[batch_slice]
            run_actor = epoch_index < actor_epochs and not stop_for_kl
            actor_args = _actor_batch_args(architecture, staged, indices) if run_actor else None
            critic_args = _critic_batch_args(architecture, staged, indices, actor_args=actor_args)
            value_targets = _batch_tensor(staged["value_targets"], indices, torch.float32)
            states = indices.numel()
            component_count = 0
            batch_kl = actor_zero
            policy_loss = actor_zero
            entropy_mean = actor_zero
            clipped_sum = actor_zero
            actor_gradient_norm = actor_zero
            ppo_gradient_norm = actor_zero
            combined_actor_loss = actor_zero
            auxiliary_loss = actor_zero
            auxiliary_terms: StructuredDynamicsTerms | None = None
            actor_auxiliary_raw_norm = actor_zero
            actor_auxiliary_applied_norm = actor_zero
            actor_auxiliary_scale = actor_zero
            auxiliary_started = 0.0
            auxiliary_start_event: torch.cuda.Event | None = None
            auxiliary_end_event: torch.cuda.Event | None = None
            if run_actor:
                assert actor_args is not None
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
                # Behavior likelihoods were replayed above at unchanged weights
                # with this exact partition and the same initial FP8 activation
                # scales. The first-minibatch metric observes only the remaining
                # inference-versus-training graph residual.
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
                    *actor_args,
                )
                batch_kl = kl_sum.detach().double() / component_count
                policy_loss = -policy_sum / component_count
                entropy_mean = entropy_sum / component_count
                # Release the PPO forward graph before building the predictor
                # graph. Gradients remain in the actor buffers, while peak
                # memory stays near one actor forward.
                policy_loss.backward()
                if actor_auxiliary_active:
                    ppo_gradients = [
                        parameter.grad
                        for parameter in actor.parameters()
                        if parameter.grad is not None
                    ]
                    ppo_gradient_norm = torch.nn.utils.get_total_norm(ppo_gradients).detach()
                    assert isinstance(actor, StructuredActor)
                    assert structured_dynamics is not None
                    assert auxiliary_windows is not None
                    window_count = max(
                        1,
                        math.ceil(states / auxiliary_windows.shape[1]),
                    )
                    selected_windows = (
                        np.arange(window_count, dtype=np.int64) + auxiliary_cursor
                    ) % auxiliary_windows.shape[0]
                    auxiliary_cursor = (auxiliary_cursor + window_count) % auxiliary_windows.shape[
                        0
                    ]
                    auxiliary_host_indices = auxiliary_windows[selected_windows].reshape(-1)
                    auxiliary_indices = torch.from_numpy(auxiliary_host_indices).to(device=device)
                    if device.type == "cuda":
                        auxiliary_start_event = torch.cuda.Event(enable_timing=True)
                        auxiliary_end_event = torch.cuda.Event(enable_timing=True)
                        auxiliary_start_event.record()
                    else:
                        auxiliary_started = time.perf_counter()
                    assert structured_terms_fn is not None
                    auxiliary_loss, auxiliary_terms = structured_terms_fn(
                        actor,
                        structured_dynamics,
                        staged,
                        auxiliary_indices,
                        steps_per_trajectory=rollout.valid.shape[1],
                        config=config,
                        autocast_enabled=autocast_enabled,
                        actor_grad=True,
                        complete_windows=True,
                    )
                    actor_parameters = tuple(actor.parameters())
                    auxiliary_gradients = torch.autograd.grad(
                        auxiliary_loss,
                        actor_parameters,
                        allow_unused=True,
                    )
                    present_auxiliary_gradients = [
                        gradient for gradient in auxiliary_gradients if gradient is not None
                    ]
                    actor_auxiliary_raw_norm = torch.nn.utils.get_total_norm(
                        present_auxiliary_gradients
                    ).detach()
                    actor_auxiliary_scale = torch.clamp(
                        (
                            config.structured_actor_gradient_ratio
                            * ppo_gradient_norm
                            / actor_auxiliary_raw_norm.clamp_min(torch.finfo(torch.float32).tiny)
                        ),
                        max=1.0,
                    )
                    if not torch.isfinite(actor_auxiliary_raw_norm):
                        actor_auxiliary_scale = torch.zeros_like(actor_auxiliary_scale)
                    actor_auxiliary_applied_norm = actor_auxiliary_raw_norm * actor_auxiliary_scale
                    for parameter, gradient in zip(
                        actor_parameters, auxiliary_gradients, strict=True
                    ):
                        if gradient is None:
                            continue
                        scaled_gradient = gradient * actor_auxiliary_scale
                        if parameter.grad is None:
                            parameter.grad = scaled_gradient
                        else:
                            parameter.grad.add_(scaled_gradient)
                    if auxiliary_end_event is not None:
                        auxiliary_end_event.record()
                    else:
                        auxiliary_seconds += time.perf_counter() - auxiliary_started
                combined_actor_loss = policy_loss.detach() + auxiliary_loss.detach()
                actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), config.max_gradient_norm
                ).detach()
                if not actor_auxiliary_active:
                    # `clip_grad_norm_` returns the same pre-clip total norm. In
                    # the ordinary PPO path, reuse it instead of launching a
                    # second complete parameter-gradient reduction solely for
                    # telemetry that is not even emitted without a predictor.
                    ppo_gradient_norm = actor_gradient_norm

            # Target KL constrains only the actor. Keep fitting the critic for
            # every configured epoch even after policy replay is frozen.
            critic_optimizer.zero_grad(set_to_none=True)
            # Only the first and last epochs score the critic's regression.
            # Middle epochs need the distributional objective alone; asking
            # `critic.value` for predictions that are immediately discarded
            # would add an fp32 softmax, support multiply, and reduction over
            # every state in those epochs.
            if epoch_index == 0 or epoch_index == critic_epochs - 1:
                value_loss, fit_moments = critic_fit_terms_fn(
                    critic, value_targets, autocast_enabled, *critic_args
                )
                if epoch_index == 0:
                    first_fit_sums += fit_moments
                    first_fit_states += states
                if epoch_index == critic_epochs - 1:
                    last_fit_sums += fit_moments
                    last_fit_states += states
            else:
                value_loss = critic_objective_fn(
                    critic, value_targets, autocast_enabled, *critic_args
                )
            if run_actor:
                guard_values = [
                    batch_kl,
                    policy_loss.detach().double(),
                    value_loss.detach().double(),
                ]
                if actor_auxiliary_active:
                    guard_values.append(combined_actor_loss.detach().double())
                guard_values_tensor = torch.stack(guard_values)
                guard_host.copy_(guard_values_tensor, non_blocking=True)
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
                guard_values_list = guard_host.tolist()
                batch_kl_value, policy_loss_value, value_loss_value = guard_values_list[:3]
                combined_actor_loss_value = (
                    guard_values_list[3] if actor_auxiliary_active else policy_loss_value
                )
                if auxiliary_start_event is not None and auxiliary_end_event is not None:
                    auxiliary_seconds += (
                        auxiliary_start_event.elapsed_time(auxiliary_end_event) / 1000.0
                    )
                if updates == 0:
                    # At unchanged weights this KL is pure numerics: the drift
                    # between the behavior replay above and this minibatch
                    # forward.
                    first_minibatch_kl = batch_kl_value
                else:
                    # Paired with `target_kl`, so it measures what the trust
                    # region governs. Mixing the numerical residual above into
                    # the same maximum would report rounding as policy movement.
                    max_approx_kl = max(max_approx_kl, batch_kl_value)
                # Non-finite losses abort training; the already-queued backward
                # of a poisoned minibatch is never observed past this raise.
                if not math.isfinite(policy_loss_value):
                    raise FloatingPointError("non-finite policy loss")
                if not math.isfinite(combined_actor_loss_value):
                    raise FloatingPointError("non-finite structured auxiliary loss")
                # The KL belongs to the policy that produced these gradients, so
                # enforce the trust region before mutating that policy.
                #
                # The first minibatch is exempt, and must be: at unchanged
                # weights its divergence is numerical residual between two
                # separately compiled graphs over differently composed batches,
                # not policy movement, so comparing it to a trust region reads
                # rounding as staleness. It is not a hypothetical -- the audit's
                # worst draw over 339 minibatches was 4.060e-2, above this
                # bound, which would latch the early stop before a single actor
                # step was taken and report a full iteration of `actor_updates:
                # 0` with a healthy policy. The residual has its own gate,
                # MAX_FIRST_MINIBATCH_KL, which `train_ppo` gives every
                # iteration and which raises rather than skipping, so a real
                # staging or replay desync still stops the run -- and stops it
                # with the right diagnosis instead of a silent trust-region hit.
                if updates > 0 and batch_kl_value > config.target_kl:
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
                    if predictor_active:
                        assert run_actor
                        ppo_actor_gradient_norm_total += ppo_gradient_norm.double() * states
                    actor_states += states
                    actor_updates += 1
                    if auxiliary_terms is not None:
                        for name in _STRUCTURED_AUXILIARY_METRICS:
                            auxiliary_totals[name] += (
                                getattr(auxiliary_terms, name).detach().double()
                            )
                        actor_auxiliary_raw_norm_total += actor_auxiliary_raw_norm.double() * states
                        actor_auxiliary_applied_norm_total += (
                            actor_auxiliary_applied_norm.double() * states
                        )
                        actor_auxiliary_scale_total += actor_auxiliary_scale.double() * states
                        auxiliary_updates += 1
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
                deferred_critic_finiteness = True
            _optimizer_step(
                critic_optimizer,
                config.critic_learning_rate,
                config.lr_warmup_steps,
                found_inf=critic_skip,
            )
            refresh_fused_mlp_fp8(critic, bootstrap_down=False)

            totals["value_loss"] += value_loss.detach().double() * states
            totals["critic_gradient_norm"] += critic_gradient_norm * states
            total_states += states
            updates += 1
        epoch_marks.append((totals["value_loss"].clone(), total_states))
        # Their steps were already gated on the device, so the critic reaching
        # this line has never absorbed a non-finite loss. If every minibatch was
        # checked through the actor guard or an ungateable optimizer, there is
        # no deferred result and therefore no device read to perform.
        if deferred_critic_finiteness and critic_nonfinite.item():
            raise FloatingPointError("non-finite critic loss")
        completed_epochs += 1

    first_epoch_value_loss, last_epoch_value_loss = _epoch_value_losses(epoch_marks)
    metrics: dict[str, float | int] = {
        "updates": updates,
        "actor_minibatches_intended": actor_minibatches_intended,
        "actor_updates": actor_updates,
        "epochs": completed_epochs,
        # The valid states this update trained on, which `rows` restricts to
        # one member's share of the wave.
        "states": valid_indices.size,
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
        "value_target_mean": float(prepared.value_targets[owned_valid].mean()),
        "value_target_std": float(prepared.value_targets[owned_valid].std()),
        "value_target_min": float(prepared.value_targets[owned_valid].min()),
        "value_target_max": float(prepared.value_targets[owned_valid].max()),
        # Every target statistic here, and the suffix-return explained variance
        # and the correlation below, are taken from the unclipped critic target
        # (the lambda-one return). `value_loss` and the two critic-fit explained
        # variances are the exceptions by necessity, since the critic regresses
        # on the saturated copy.
        "value_target_saturated_fraction": float(saturated) / float(valid_value_targets.size),
        "actor_gae_lambda": config.actor_gae_lambda,
        "critic_gae_lambda": config.critic_gae_lambda,
        "gamma": config.gamma,
        "actor_learning_rate": float(actor_optimizer.param_groups[0]["lr"]),
        "critic_learning_rate": float(critic_optimizer.param_groups[0]["lr"]),
        # Against the discounted suffix return, which decoupled GAE also uses
        # as the critic target. Pre-update predictions, so this is not a fit.
        "monte_carlo_explained_variance": _explained_variance(
            prepared.monte_carlo_returns, behavior_values, owned_valid
        ),
        # Against the policy lambda-return ``A + V``. Residual is identically
        # the advantage: ``1 - Var(A)/Var(G_lambda)``. The critic does not fit
        # this target.
        "lambda_return_explained_variance": _explained_variance(
            prepared.policy_lambda_returns, behavior_values, owned_valid
        ),
        # Against the critic target with predictions taken during the update, so
        # the residual is a fit error rather than an algebraic identity. These
        # two are what say whether the regression is working, and the gap
        # between them says whether it is generalizing rather than memorizing
        # the batch: the first epoch scores every state before this update has
        # fitted it, the last scores each one on its fourth pass. With a single
        # configured critic epoch they coincide, that epoch being both.
        "critic_fit_explained_variance_first_epoch": _fit_explained_variance(
            _fit_moment_mapping(first_fit_sums), first_fit_states
        ),
        "critic_fit_explained_variance_last_epoch": _fit_explained_variance(
            _fit_moment_mapping(last_fit_sums), last_fit_states
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
        "value_prediction_mean": float(behavior_values[owned_valid].mean()),
        "value_prediction_std": float(behavior_values[owned_valid].std()),
        "value_target_correlation": _target_correlation(
            prepared.monte_carlo_returns, behavior_values, owned_valid
        ),
    }
    metrics.update(predictor_metrics)
    if predictor_active:
        metrics["structured_actor_auxiliary_enabled"] = int(actor_auxiliary_active)
        metrics["ppo_actor_gradient_norm"] = float(
            ppo_actor_gradient_norm_total / max(1, actor_states)
        )
    if actor_auxiliary_active:
        metrics.update(
            {
                f"structured_actor_{name}": float(total / max(1, auxiliary_updates))
                for name, total in auxiliary_totals.items()
            }
        )
        metrics["structured_actor_auxiliary_updates"] = auxiliary_updates
        metrics["structured_actor_auxiliary_seconds"] = auxiliary_seconds
        metrics["structured_actor_auxiliary_raw_gradient_norm"] = float(
            actor_auxiliary_raw_norm_total / max(1, actor_states)
        )
        metrics["structured_actor_auxiliary_applied_gradient_norm"] = float(
            actor_auxiliary_applied_norm_total / max(1, actor_states)
        )
        metrics["structured_actor_auxiliary_scale"] = float(
            actor_auxiliary_scale_total / max(1, actor_states)
        )
    return metrics
