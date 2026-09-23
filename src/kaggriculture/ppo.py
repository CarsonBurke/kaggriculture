"""Masked-component PPO with independently configurable actor and critic GAE."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from kaggriculture.actor_dynamics import (
    ActorDynamics,
    ActorDynamicsTerms,
    actor_horizon_loss,
    actor_window_loss,
)
from kaggriculture.causal_actor import CausalActor, CausalReplay
from kaggriculture.constants import DEFAULT_REWARD_GAMMA, EPISODE_STEPS, STARTING_MONEY
from kaggriculture.economic_forecasting import (
    build_economic_forecast_targets,
    economic_forecast_loss,
)
from kaggriculture.entity import EntityActor, EntityCritic
from kaggriculture.lejepa import (
    JEPA_METRICS,
    JepaObjective,
    JepaPersistenceControl,
    JepaShuffledControl,
    JepaTerms,
    jepa_horizon_loss,
)
from kaggriculture.model import (
    ActorOutput,
    DistributionalCritic,
    FarmActor,
    distributional_value_loss,
    hl_gauss_value_targets,
    policy_compile_options,
    scalar_value_loss,
)
from kaggriculture.optim import NorMuon, route_parameters
from kaggriculture.policy import component_logprobs, component_selected_logprobs, mask_logits
from kaggriculture.provenance import UNCOMPILED_UPDATE_COMPILE_MODE
from kaggriculture.registry import (
    CONV_ENTITY,
    LEJEPA,
    STRUCTURED,
    architecture_of_config,
    resolve_architecture,
)
from kaggriculture.rollout import RolloutBatch, behavior_value_key
from kaggriculture.strategic_actor import PlanChoice, StrategicActor, StrategicOutput
from kaggriculture.structured import (
    JepaBelief,
    StructuredActor,
    StructuredBelief,
    StructuredCritic,
    StructuredCriticBelief,
    StructuredDecisionBelief,
    StructuredInputs,
    refresh_fused_mlp_fp8,
)
from kaggriculture.structured_dynamics import (
    PersistenceDynamics,
    StructuredCriticDynamics,
    StructuredCriticDynamicsTerms,
    StructuredHorizonPlan,
    structured_critic_horizon_loss,
    structured_critic_window_loss,
    structured_horizon_plan,
)
from kaggriculture.tokens import TILE_SLOT_CATEGORICAL

Critic = DistributionalCritic | StructuredCritic | EntityCritic
Actor = FarmActor | StructuredActor | EntityActor

#: VAPO temporal defaults: Monte Carlo critic targets, shorter actor GAE traces.
#: Actor lambda uses alpha=0.05 and the full game's EPISODE_STEPS - 1 transitions.
DEFAULT_ACTOR_GAE_LAMBDA = 1.0 - 1.0 / (0.05 * (EPISODE_STEPS - 1))
DEFAULT_CRITIC_GAE_LAMBDA = 1.0

#: Optimizers `make_optimizers` can build, named by `PpoConfig.optimizer`.
_OPTIMIZERS = ("normuon", "adamw")
#: Actor objectives `update_ppo` can fit, named by `PpoConfig.policy_objective`.
_POLICY_OBJECTIVES = ("clip", "tpo")
#: Sampled decisions at or above this likelihood carry no alternative to move
#: mass to, so a TPO target cannot be formed for them: single-valid-action
#: masks (a PASS-only unit) store an exact zero log-likelihood.
_TPO_SATURATED_LOGPROB = -1e-6

# Numerical audit gates retained pending measurements on the aligned sampling
# and per-component update paths. Replay is diagnostic only: PPO always uses
# the likelihoods stored by the distribution that sampled each action.
MAX_UPDATE_REPLAY_KL = 5e-3
UPDATE_REPLAY_TAIL_LOGPROB = 2.5
MAX_UPDATE_REPLAY_TAIL_FRACTION = 2e-4
# Gate the actual grad-tracking minibatch graph as well as full-rollout replay.
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
    # Promoted HL-Gauss VAPO terminal-outcome LR3 recipe.
    actor_learning_rate: float = 1.5e-4
    critic_learning_rate: float = 1.5e-4
    # Optional absolute LR for value_head only, independent of trunk Adam gains.
    critic_head_learning_rate: float | None = None
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
    epochs: int = 1
    # Total epochs for the critic; the actor participates only in the first
    # `epochs` of them, so values above `epochs` are critic-only refits over
    # the same rollout. None matches the actor epoch count. Production is 1/1:
    # a second same-wave critic pass memorized holdout, and a second actor pass
    # is a replay at a KL that does not bind.
    critic_epochs: int | None = None
    # Keep physical batches larger without consuming the last VRAM headroom.
    # Fixed-shape partitioning covers the full wave; the final batch pads with
    # zero weights. The size divides the production wave: 230,080 states make
    # 29 minibatches with 64 padded rows, where an 8192 ceiling made the same
    # 29 with 7,488 -- a 3.2% padded tail computed for nothing, and a final
    # step averaged over 704 states rather than a full batch.
    minibatch_size: int = 7936
    # Component scope clips each conditional decision independently; joint scope
    # clips the product of all active conditional probabilities in one state.
    policy_ratio_scope: str = "components"
    # "states" averages the surrogate sum over genuine states; component-count
    # reduction is only meaningful for independently clipped components.
    policy_loss_reduction: str = "states"
    # Which per-decision objective the actor fits. `clip` is the PPO surrogate
    # bounded by the band below. `tpo` is single-sample Target Policy
    # Optimization (Kaddour 2026, arXiv:2604.06159, Appendix C): the sampled
    # decision's rollout probability is tilted by exp(A / eta) to form a target
    # `logit q = logit p_old + A / eta`, and the policy is fit to it by the
    # Bernoulli KL on the sampled decision. Only the sampled coordinate carries
    # a score, so distributing the remaining mass by the current policy makes
    # the full-distribution target's gradient exact at the rollout snapshot
    # while needing nothing beyond the stored selected likelihoods. The paper's
    # within-group z-scoring is deliberately not applied: over a one-hot score
    # it discards |A| and saturates the target for any head this wide.
    policy_objective: str = "clip"
    # TPO tilt temperature over the batch-whitened advantage. Per-decision target
    # KL from the snapshot reaches 0.5 nats at |A| / eta = 3 on a 50% decision.
    tpo_eta: float = 1.0
    # Weight on KL(reference || policy), per active decision, against the frozen
    # initial actor (the behavior clone a warm start loads). Zero disables it.
    #
    # PPO's surrogate only moves a decision through the actions sampled there,
    # so a decision the policy already takes with probability ~1 receives no
    # gradient at all: nothing holds it in place while shared parameters move
    # under every other state's update. The opening purchase is such a decision
    # -- one state, identical in every game -- and in the first attached lejepa
    # run it drifted from the teacher's one cow to two over ten updates of
    # joint KL ~0.001, then bankrupted every game at once. The trust region could
    # not see it, being a mean over hundreds of thousands of decisions. The
    # forward KL restores exactly those decisions: its gradient on each state is
    # the gap between the reference's distribution and the policy's.
    reference_kl_coefficient: float = 0.0
    # DAPO's Clip-Higher band, as eps_low 0.2 and eps_high 0.28 rather than
    # PPO's symmetric 0.2 either side. The asymmetry exists to stop entropy
    # collapse: a symmetric band clips a low-probability action's upside at the
    # same ratio as a high-probability one's, which is a much tighter bound on
    # its absolute probability, so exploration dies faster than it should.
    #
    # Measured either way over 100 waves and it is not a lever: 1.20 plateaus
    # entropy at 0.210 from wave 65 where 1.28 reaches 0.233 still rising, and
    # money is 21.6k against 23.4k -- inside the spread two
    # identical-configuration runs show by wave 33, since
    # `deterministic_training` is off (`runs/production-lam1-{,clip12-}p100-20260911`).
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
    # Collection and PPO share gamma. At the default gamma one, terminal
    # win/loss/draw rewards retain the undiscounted game outcome.
    # The critic fits full Monte Carlo returns; the actor's shorter GAE trace
    # uses that critic to carry long-term credit with lower variance.
    actor_gae_lambda: float = DEFAULT_ACTOR_GAE_LAMBDA
    critic_gae_lambda: float = DEFAULT_CRITIC_GAE_LAMBDA
    gamma: float = DEFAULT_REWARD_GAMMA
    # Whitening the surrogate's advantages -- `(A - mean) / std` over the owned
    # valid states of the wave -- shipped here until `15e869c`, which removed it
    # inside a performance commit with no recorded rationale. Restoring it is a
    # flag rather than a default because the two halves of the transform are not
    # equally consequential, and the measurement says so:
    #
    #  * With actor auxiliary losses disabled, the scale half only reaches the
    #    update through Adam's epsilon. This invariance does not hold when only
    #    PPO is rescaled inside a PPO-plus-auxiliary sum: that changes the combined
    #    gradient direction and the effective auxiliary weight. Polar
    #    Express opens by dividing by its input's Frobenius norm
    #    (`optim.py:58-64`), so a NorMuon matrix step is exactly invariant to any
    #    uniform rescaling of the gradient. Adam is invariant only where epsilon
    #    is negligible against `sqrt(v_hat)`, which the reference's 1e-10 was not
    #    at these magnitudes; `optim.py` now ships 1e-20 and records the
    #    measurement. With that floor gone a per-wave rescale is a per-update
    #    constant the optimizer divides back out.
    #  * The mean half changes the direction, not the size: subtracting a common
    #    offset pushes every sampled action's logprob the same way, which is an
    #    entropy force with no counterpart in the clipped surrogate. Over waves
    #    24-100 of three runs the realized offset is small but sign-persistent
    #    late: mean advantage over advantage std averages -0.005 to -0.010 in the
    #    second half, positive in only 15-20 of 50 waves.
    normalize_advantages: bool = False
    # The PPO actor and critic gradients are intentionally unclipped. NorMuon
    # already normalizes matrix directions; the value head is Adam-managed.
    # Keep this threshold only for the actor/critic NextLat predictors, where
    # the auxiliary model has no PPO trust-region or value-fit safeguard.
    nextlat_max_gradient_norm: float = 1.0
    # Restored to 0.03, the CleanRL-adjacent trust region this pipeline shipped
    # from scratch, on the population runs' own telemetry: four members at
    # actor_lr 3.0e-5 held approx_kl at 1e-4 to 2e-4 per iteration with the
    # epoch never stopping early (`runs/pop4-economic-*`), so at the rate the
    # launcher ships the region 0.03 does not bind and the wider 0.10 only
    # buys headroom for an excursion the measured surface does not produce.
    # The earlier tabulated raise to 0.10 was decided on a lane whose actor
    # started from a different BC artifact; it is history, not a constraint.
    target_kl: float = 0.03
    # BF16 autocast for both update-path forwards. The importance ratio compares
    # the update likelihood with the rollout-stored sampler likelihood; an
    # unchanged actor is therefore near one to the precision of the two paths,
    # rather than being forced to one by an update-graph replay.
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
    # shape, and inference artifacts on their historical paths. When active,
    # minibatches are contiguous episode runs so h_t and h_{t+1} come from one
    # trunk forward, as in NextLat.
    # Actor terms sum independently normalized unit/kind/quantity head losses.
    structured_latent_coefficient: float = 0.0
    structured_decision_coefficient: float = 0.0
    structured_decision_horizon: int = 2
    # Predictor parameters live in a separate optimizer so they are not in the
    # actor/critic state dicts consumed by snapshots. They step on the same
    # minibatches as the trunk, adding the configured auxiliary losses directly.
    # The source remains attached whenever its model updates; successor targets
    # and auxiliary readout weights are always stop-gradient.
    structured_learning_rate: float | None = None
    structured_critic_learning_rate: float | None = None
    structured_critic_latent_coefficient: float = 0.0
    structured_critic_value_coefficient: float = 0.0
    structured_critic_horizon: int = 1
    structured_critic_gradient_balance: bool = False
    # LeJEPA world-model objective, used by the `lejepa` family in place of the
    # detached NextLat terms above. Two terms plus a reward, exactly as in
    # `../le-wm`: the next step's embedding is regressed with its gradient LIVE,
    # and SIGReg is what stops that from being solved by a constant encoder.
    # Both coefficients must be positive together -- an attached target with no
    # distributional constraint has a trivial global minimum -- and the pair is
    # mutually exclusive with the detached NextLat coefficients above, which
    # would be two self-predictive objectives fighting over one trunk.
    #
    # One set of coefficients and not two, because there is one backbone. The
    # critic's tower reads it detached and the actor's heads (by default) attached,
    # so this objective and the policy's are the only ones that train an encoder
    # anywhere in the `lejepa` family, and a second arm would be the same loss on
    # a second copy of the same function.
    jepa_prediction_coefficient: float = 0.0
    jepa_sigreg_coefficient: float = 0.0
    jepa_reward_coefficient: float = 0.0
    jepa_horizon: int = 1
    # Observable multi-horizon delta supervision, used only by the feed-forward
    # forecast critic. Its heads share the critic optimizer, never a predictor.
    economic_forecast_coefficient: float = 1.0

    @property
    def resolved_structured_learning_rate(self) -> float:
        return (
            self.actor_learning_rate
            if self.structured_learning_rate is None
            else self.structured_learning_rate
        )

    @property
    def resolved_structured_critic_learning_rate(self) -> float:
        return (
            self.critic_learning_rate
            if self.structured_critic_learning_rate is None
            else self.structured_critic_learning_rate
        )

    @property
    def jepa_active(self) -> bool:
        return bool(
            self.jepa_prediction_coefficient
            or self.jepa_sigreg_coefficient
            or self.jepa_reward_coefficient
        )

    @property
    def structured_actor_auxiliary_active(self) -> bool:
        return bool(
            self.structured_latent_coefficient
            or self.structured_decision_coefficient
            or self.jepa_active
        )

    @property
    def structured_critic_auxiliary_active(self) -> bool:
        # No LeJEPA term here: the world model is one arm on the actor's side of
        # the trainer, and the `lejepa` critic carries no encoder to regularize.
        return bool(
            self.structured_critic_latent_coefficient or self.structured_critic_value_coefficient
        )

    @property
    def structured_auxiliary_active(self) -> bool:
        return self.structured_actor_auxiliary_active or self.structured_critic_auxiliary_active


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
    # Location and scale of the advantages the surrogate sees.
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
    critic uses `PpoConfig.critic_gae_lambda` (one for Monte Carlo targets).
    This helper is the recurrence only.
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
        ("NextLat gradient norm", config.nextlat_max_gradient_norm),
        ("target KL", config.target_kl),
        ("adam learning rate ratio", config.adam_learning_rate_ratio),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.critic_head_learning_rate is not None and (
        not math.isfinite(config.critic_head_learning_rate)
        or config.critic_head_learning_rate <= 0.0
    ):
        raise ValueError("critic head learning rate must be finite and positive")
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
    if config.policy_loss_reduction not in ("components", "states"):
        raise ValueError("policy loss reduction must be 'components' or 'states'")
    if config.policy_ratio_scope not in ("components", "joint"):
        raise ValueError("policy ratio scope must be 'components' or 'joint'")
    if config.policy_ratio_scope == "joint" and config.policy_loss_reduction != "states":
        raise ValueError("joint policy ratio scope requires state policy loss reduction")
    if config.policy_objective not in _POLICY_OBJECTIVES:
        raise ValueError(
            f"unsupported policy objective {config.policy_objective!r}, "
            f"want one of {_POLICY_OBJECTIVES}"
        )
    if not math.isfinite(config.tpo_eta) or config.tpo_eta <= 0.0:
        raise ValueError("TPO eta must be finite and positive")
    if not math.isfinite(config.reference_kl_coefficient) or config.reference_kl_coefficient < 0:
        raise ValueError("reference KL coefficient must be finite and nonnegative")
    if not math.isfinite(config.gamma) or not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not math.isfinite(config.actor_gae_lambda) or not 0.0 <= config.actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")
    if not math.isfinite(config.critic_gae_lambda) or not 0.0 <= config.critic_gae_lambda <= 1.0:
        raise ValueError("critic GAE lambda must be finite and in [0, 1]")
    coefficients = {
        "economic forecast": config.economic_forecast_coefficient,
        "structured latent": config.structured_latent_coefficient,
        "structured decision": config.structured_decision_coefficient,
        "structured critic latent": config.structured_critic_latent_coefficient,
        "structured critic value": config.structured_critic_value_coefficient,
        "jepa prediction": config.jepa_prediction_coefficient,
        "jepa sigreg": config.jepa_sigreg_coefficient,
        "jepa reward": config.jepa_reward_coefficient,
    }
    for name, value in coefficients.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} coefficient must be finite and nonnegative")
    if config.structured_learning_rate is not None and (
        not math.isfinite(config.structured_learning_rate) or config.structured_learning_rate <= 0.0
    ):
        raise ValueError("structured learning rate must be finite and positive")
    if config.structured_critic_learning_rate is not None and (
        not math.isfinite(config.structured_critic_learning_rate)
        or config.structured_critic_learning_rate <= 0.0
    ):
        raise ValueError("structured critic learning rate must be finite and positive")
    horizons = (
        config.structured_decision_horizon,
        config.structured_critic_horizon,
        config.jepa_horizon,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in horizons):
        raise ValueError("structured auxiliary horizons must be integers")
    if any(value < 0 for value in horizons):
        raise ValueError("structured auxiliary horizons cannot be negative")
    if config.structured_actor_auxiliary_active and config.structured_decision_horizon < 1:
        raise ValueError(
            "structured decision horizon must be positive when actor auxiliary is active"
        )
    if config.structured_critic_auxiliary_active and config.structured_critic_horizon < 1:
        raise ValueError(
            "structured critic horizon must be positive when critic auxiliary is active"
        )
    _validate_jepa_coefficients(config)


def _validate_jepa_coefficients(config: PpoConfig) -> None:
    """Refuse the configurations the LeJEPA objective cannot survive.

    An attached prediction target without SIGReg is minimized exactly by a constant
    encoder, and the loss curve of a collapsing run is a clean descent to zero -- the
    failure is silent, and it destroys the backbone both towers read. Requiring the two
    together is therefore a correctness constraint, not a default: the policy gradient
    that also reaches the encoder constrains only what the decisions read, and the
    detached ablation removes even that. The detached NextLat coefficients are refused
    alongside it because they are a second self-predictive objective on the same
    representation with the opposite convention about where the gradient stops.
    """
    if not config.jepa_active:
        return
    if not (config.jepa_prediction_coefficient > 0.0 and config.jepa_sigreg_coefficient > 0.0):
        raise ValueError(
            "the LeJEPA objective needs positive prediction and SIGReg "
            "coefficients together; an attached target alone collapses"
        )
    detached = (
        config.structured_latent_coefficient,
        config.structured_decision_coefficient,
        config.structured_critic_latent_coefficient,
        config.structured_critic_value_coefficient,
    )
    if any(detached):
        raise ValueError("the LeJEPA and detached NextLat objectives are mutually exclusive")


def _validate_structured_auxiliary_modules(
    actor: Actor,
    dynamics: ActorDynamics | JepaObjective | None,
    config: PpoConfig,
) -> None:
    active = config.structured_actor_auxiliary_active
    if active and not architecture_of_config(actor.config).structured_inputs:
        raise ValueError("structured auxiliary coefficients require a structured actor")
    if active != (dynamics is not None):
        state = "requires" if active else "does not admit"
        raise ValueError(f"structured auxiliary configuration {state} a dynamics predictor")
    if config.jepa_active != isinstance(dynamics, JepaObjective):
        raise ValueError("the LeJEPA coefficients require exactly a LeJEPA objective")
    lejepa_family = architecture_of_config(actor.config).name == LEJEPA
    if isinstance(dynamics, JepaObjective) and not lejepa_family:
        raise ValueError("the LeJEPA objective requires the lejepa actor family")
    if lejepa_family and dynamics is not None and not isinstance(dynamics, JepaObjective):
        # `actor_belief_class` is keyed on the family, so a lejepa actor hands
        # back a four-field `JepaBelief` whatever predictor is attached. The
        # detached helpers splat a belief into a one- or two-field NamedTuple and
        # die on the first minibatch; refusing the pairing here is what keeps the
        # arms independently configurable without making the mix reachable.
        raise ValueError("the lejepa actor family admits only the LeJEPA objective")
    if (
        dynamics is not None
        and next(dynamics.parameters()).device != next(actor.parameters()).device
    ):
        raise ValueError("actor and structured dynamics must use the same device")


def _validate_structured_critic_auxiliary_modules(
    critic: Critic,
    dynamics: StructuredCriticDynamics | JepaObjective | None,
    config: PpoConfig,
) -> None:
    active = config.structured_critic_auxiliary_active
    # Both family refusals come before the generic active/module agreement
    # check, because both name the actual mistake. "This configuration does not
    # admit a critic predictor" is true of a LeJEPA objective handed to the
    # critic, and it is not what the caller needs to be told.
    if isinstance(dynamics, JepaObjective):
        # The world model is one arm and it hangs off the actor, whose trunk is
        # the shared backbone. A second objective handed to the critic would be
        # a second encoder's worth of projectors with no encoder behind them.
        raise ValueError("the LeJEPA objective belongs to the actor arm, not the critic")
    if architecture_of_config(critic.config).name == LEJEPA and dynamics is not None:
        raise ValueError("the lejepa critic reads the shared backbone and admits no predictor")
    if active and not architecture_of_config(critic.config).structured_inputs:
        raise ValueError("structured critic auxiliary coefficients require a structured critic")
    if active != (dynamics is not None):
        state = "requires" if active else "does not admit"
        raise ValueError(
            f"structured critic auxiliary configuration {state} a critic dynamics predictor"
        )
    if (
        dynamics is not None
        and next(dynamics.parameters()).device != next(critic.parameters()).device
    ):
        raise ValueError("critic and structured critic dynamics must use the same device")


#: The `role` of the world-model optimizer's parameter groups that hold the
#: shared backbone, whose warmup clock advances only when the backbone steps.
BACKBONE_ROLE = "backbone"


def backbone_parameters(actor: Actor) -> list[Tensor]:
    """The world-model parameters inside an actor, which its own optimizer must skip.

    Empty for every family but `lejepa`, so the callers below need no branch:
    a family whose trunk is trained by the policy owns all of itself.
    """
    getter = getattr(actor, "backbone_parameters", None)
    return [] if getter is None else list(getter())


def policy_owned_parameters(actor: Actor) -> list[Tensor]:
    """Everything in an actor that the policy objective is allowed to train."""
    getter = getattr(actor, "head_parameters", None)
    return list(actor.parameters() if getter is None else getter())


def _validate_optimizer_ownership(
    pairs: tuple[tuple[str, Any, torch.optim.Optimizer], ...],
) -> None:
    """Require exact, pairwise-disjoint ownership for every trainable parameter set.

    A set and not a module. Under the `lejepa` family one module is split across two
    owners: the shared backbone is a submodule of the actor -- which is what keeps
    league snapshots and inference bundles reading one flat actor state dict -- while
    the objective that trains it owns it, clips it and gates its step, even though the
    policy's gradient reaches it in the same backward. Stating ownership as an explicit
    parameter iterable is what lets that partition be checked as strictly as whole
    modules were: every parameter still has exactly one owner, and the claim is still
    verified in both directions rather than asserted in a docstring.
    """
    owned: list[tuple[str, set[int]]] = []
    for name, source, optimizer in pairs:
        parameters = source.parameters() if isinstance(source, torch.nn.Module) else source
        claimed = {id(parameter) for parameter in parameters}
        optimizer_parameters = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_parameters != claimed:
            raise ValueError(f"{name} optimizer must own exactly the {name} parameters")
        for other_name, other_parameters in owned:
            if claimed & other_parameters:
                raise ValueError(f"{name} and {other_name} parameters must be disjoint")
        owned.append((name, claimed))


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
        ("animals", torch.float32),
        ("crops", torch.float32),
        ("farms", torch.float32),
        ("town", torch.float32),
    ),
}


def _actor_forward_fields(architecture: str) -> tuple[tuple[str, torch.dtype], ...]:
    """The named architecture's forward fields, or a caller-actionable refusal."""
    family = resolve_architecture(architecture)
    return _ACTOR_FORWARD_FIELDS[STRUCTURED if family.structured_inputs else CONV_ENTITY]


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
    args = _actor_forward_tuple(
        architecture,
        {
            name: _batch_tensor(staged[name], indices, dtype)
            for name, dtype in _actor_forward_fields(architecture)
        },
    )
    if architecture == "strategic-plan" and "plan_indices" in staged:
        plan_indices = _batch_tensor(staged["plan_indices"], indices, torch.long)
        args += (
            PlanChoice(
                plan_indices,
                torch.zeros_like(plan_indices, dtype=torch.float32),
                torch.ones_like(plan_indices, dtype=torch.float32),
                torch.zeros_like(plan_indices, dtype=torch.bool),
                _batch_tensor(staged["old_plan_logprobs"], indices, torch.float32),
                _batch_tensor(staged["plan_active"], indices, torch.bool),
            ),
        )
    elif architecture == "causal-execution":
        args += (
            CausalReplay(
                _batch_tensor(staged["policy_ledger"], indices, torch.long),
                _batch_tensor(staged["unit_actions"], indices, torch.long),
                _batch_tensor(staged["market_kinds"], indices, torch.long),
                _batch_tensor(staged["market_quantities"], indices, torch.long),
            ),
        )
    return args


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
    actor_inputs = (
        _actor_forward_tuple(
            architecture,
            {
                name: _batch_tensor(staged[name], indices, dtype)
                for name, dtype in _actor_forward_fields(architecture)
            },
        )[0]
        if actor_args is None
        else actor_args[0]
    )
    inputs = actor_inputs._replace(
        products=torch.cat(
            (
                actor_inputs.products,
                _batch_tensor(staged["critic_products"], indices, torch.float32),
            ),
            dim=-1,
        ),
        animals=torch.cat(
            (actor_inputs.animals, _batch_tensor(staged["critic_animals"], indices, torch.float32)),
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
    include_entities: bool = False,
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
        if include_entities:
            if not isinstance(critic, StructuredCritic) or not getattr(
                critic.config, "per_entity_critic", False
            ):
                raise ValueError("entity replay requires a per-entity structured critic")
            global_logits, belief = critic.forward_with_belief(*critic_args)
            critic_logits = torch.cat(
                (global_logits[:, None], critic.decode_entity_belief(belief)), dim=1
            )
        else:
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
    include_entities: bool = False,
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

    With ``include_entities=True`` explicitly returns (rows, 1 + units + orders),
    global first. The default remains (rows,), even for an entity critic.
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
    # Every chunk is exactly `chunk_size` rows: the tail wraps to the leading
    # rows and the duplicates are dropped from the result. A short final chunk
    # was a second shape for this callable, and the wave's row count moves, so
    # the measured cost was an Inductor compile of the value replay for each new
    # remainder a run produced.
    chunk_count = math.ceil(row_count / chunk_size)
    wrapped = torch.arange(chunk_count * chunk_size, device=device) % row_count
    rows = wrapped if states is None else states.reshape(-1)[wrapped]
    chunks = [rows[start : start + chunk_size] for start in range(0, rows.numel(), chunk_size)]
    was_training = critic.training
    critic.eval()
    try:
        values = None
        for index, chunk in enumerate(chunks):
            _begin_update_graph_step(compile_mode, device)
            chunk_values = forward(
                critic,
                autocast_enabled,
                *_critic_batch_args(architecture, staged, chunk),
                include_entities=include_entities,
            )
            if values is None:
                values = chunk_values.new_empty((chunk_count * chunk_size, *chunk_values.shape[1:]))
            values[index * chunk_size : (index + 1) * chunk_size].copy_(chunk_values)
    finally:
        critic.train(was_training)
    assert values is not None
    return values[:row_count].float()


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
    include_entities: bool = False,
) -> np.ndarray:
    """Behavior-time value predictions on the rollout grid, replayed for `rows`.

    Rows this update does not own stay exactly zero, which is what the owned
    mask makes of them in every consumer: GAE selects them away, and no metric
    reads them. Restricting the pass rather than the readings is what keeps a
    population's N per-member updates costing one whole-wave critic replay
    between them instead of N.

    A wave whose collector already read these values (`collect_mixed_play_rust`
    with a `lejepa` critic) skips the replay while `behavior_value_key` still
    matches: the same critic, at the same weights, under the same autocast the
    replay would use. Anything else -- a critic stepped since, another
    precision, a population subset, per-entity values -- replays as before.
    """
    if (
        rows is None
        and not include_entities
        and rollout.behavior_values is not None
        and rollout.behavior_value_key == behavior_value_key(critic, autocast_enabled)
    ):
        return rollout.behavior_values
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
            include_entities=include_entities,
        )
        .cpu()
        .numpy()
    )
    shape = (*rollout.rewards.shape, *replayed.shape[1:])
    if rows is None:
        return replayed.reshape(shape)
    grid = np.zeros(shape, dtype=replayed.dtype)
    grid[rows] = replayed.reshape(-1, horizon, *replayed.shape[1:])
    return grid


def prepare_advantages(
    rollout: RolloutBatch,
    values: np.ndarray,
    config: PpoConfig,
    *,
    rows: np.ndarray | None = None,
) -> AdvantageBatch:
    """VAPO decoupled GAE: policy advantages at actor lambda, critic targets at critic lambda.

    `rows` restricts every statistic to those trajectory rows, and rows outside
    it come back exactly zero. A population wave partitions before it updates
    because a game's two seats belong to two members; no storage order makes
    one member's rows a contiguous block.
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
    raw_mean = float(selected.mean())
    raw_std = float(selected.std(unbiased=False))
    if config.normalize_advantages:
        # The wave's own statistics, over the states this update owns: a pooled
        # normalizer would let one population member's return scale set another
        # member's offset, and the partition is by seat, not by contiguous rows.
        advantages = (advantages - raw_mean) / max(raw_std, 1e-6)
    advantages = advantages * valid
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
        advantages=advantages.numpy(),
        value_targets=critic_targets.numpy(),
        policy_lambda_returns=policy_lambda_returns.numpy(),
        monte_carlo_returns=monte_carlo.numpy(),
        raw_advantage_mean=raw_mean,
        raw_advantage_std=raw_std,
    )


def prepare_entity_advantages(
    returns: np.ndarray,
    values: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """MAPPO whitening over active entities, counting each market order once.

    ``active`` already includes owned valid states; inactive NaNs are never
    read by the moments or returned to the actor. Both market heads reuse the
    one order advantage, not a separately normalized quantity baseline.
    """
    if values.shape != active.shape or returns.shape != values.shape[:-1]:
        raise ValueError("entity values and active masks must match the return grid")
    advantages = np.zeros_like(values, dtype=np.float32)
    np.subtract(returns[..., None], values, out=advantages, where=active)
    selected = advantages[active]
    if not selected.size or not np.isfinite(selected).all():
        raise ValueError("active entity advantages must be nonempty and finite")
    mean, std = float(selected.mean()), float(selected.std())
    advantages[active] = (selected - mean) / (std + 1e-5)
    return advantages, mean, std


def _initialize_optimizer_schedule(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        # Optimizer param-group metadata is checkpointed by PyTorch, so the
        # schedule resumes exactly without a separate scheduler object.
        group["base_lr"] = base_learning_rate
        group["warmup_step"] = 0


def _configure_critic_head_rate(
    optimizer: torch.optim.Optimizer, critic: Critic, learning_rate: float | None
) -> None:
    """Separate only the value readout while retaining optimizer and warmup state."""
    if learning_rate is None:
        return
    head = list(critic.value_head.parameters())
    head_ids = {id(parameter) for parameter in head}
    for group in optimizer.param_groups:
        if head_ids <= {id(parameter) for parameter in group["params"]}:
            remaining = [
                parameter for parameter in group["params"] if id(parameter) not in head_ids
            ]
            settings = dict(group)
            settings.update(params=head, lr=learning_rate, base_lr=learning_rate, role="value_head")
            if remaining:
                group["params"] = remaining
                optimizer.add_param_group(settings)
            else:
                group.update(settings)
            return
    raise ValueError("critic value head must belong to one optimizer group")


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
    if (
        config.structured_actor_auxiliary_active
        and not architecture_of_config(actor.config).structured_inputs
    ):
        raise ValueError("structured auxiliary coefficients require a structured actor")
    if (
        config.structured_critic_auxiliary_active
        and not architecture_of_config(critic.config).structured_inputs
    ):
        raise ValueError("structured critic auxiliary coefficients require a structured critic")
    # The `lejepa` actor carries the shared world-model backbone as a submodule
    # so one flat state dict still serializes the deployed model, but the
    # policy's optimizer does not own it: `make_structured_dynamics_optimizer`
    # does, beside the objective, and steps the policy's gradient into it with
    # the objective's. Everything below therefore steps the actor's heads alone.
    actor_owned = list(policy_owned_parameters(actor))
    if config.optimizer == "normuon":
        # One learning rate per network drives both halves: the matrices under
        # NorMuon and the gains, biases and heads under Adam. They are not the
        # same quantity -- a NorMuon rate is a RELATIVE step, since the
        # orthogonalized update's Frobenius norm is `lr * sqrt(min(rows, cols))`
        # against a weight norm of about `sqrt(fan_out)`, while an Adam rate is
        # an ABSOLUTE per-element step. `adam_learning_rate_ratio` converts.
        actor_optimizer = NorMuon(
            *route_parameters(actor, exclude=backbone_parameters(actor)),
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
        _configure_critic_head_rate(critic_optimizer, critic, config.critic_head_learning_rate)
        return actor_optimizer, critic_optimizer
    fused = actor_device.type == "cuda"
    actor_optimizer = torch.optim.AdamW(
        actor_owned,
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
    _configure_critic_head_rate(critic_optimizer, critic, config.critic_head_learning_rate)
    return actor_optimizer, critic_optimizer


def make_structured_dynamics_optimizer(
    dynamics: ActorDynamics | StructuredCriticDynamics | JepaObjective,
    config: PpoConfig,
    *,
    critic: bool | None = None,
    actor: Actor | None = None,
) -> torch.optim.Optimizer:
    """Build the predictor-only optimizer with an independent warmup clock.

    `critic` names which arm the detached predictor belongs to, since the two
    arms have separate configured rates. A `JepaObjective` is always the actor
    arm -- there is one world model -- and it is the one case where this
    optimizer owns more than the predictor: `actor` must be given, and its
    backbone steps here, beside the only loss that trains it.

    Together rather than under the actor's optimizer because they are one model,
    at one learning rate. The backbone's parameter groups are tagged
    `BACKBONE_ROLE` and keep a warmup clock of their own, because the backbone
    steps only when the policy does while the projector and the predictor step
    on every minibatch: a warmup wave spent fitting a fresh predictor against a
    frozen backbone must not use up the backbone's warmup, or its first policy
    steps would run at full rate beside heads that are still warming up.
    """
    _validate_config(config)
    if isinstance(dynamics, JepaObjective):
        if critic:
            raise ValueError("the LeJEPA objective is the actor arm; it has no critic arm")
        if actor is None:
            raise ValueError("the LeJEPA optimizer owns the shared backbone; pass the actor")
        if not backbone_parameters(actor):
            raise ValueError(
                "the LeJEPA objective trains a shared backbone, which this actor does not "
                "expose; the lejepa family is the only one that has one"
            )
        critic = False
    elif actor is not None:
        raise ValueError("only the LeJEPA objective's optimizer owns backbone parameters")
    elif critic is None:
        critic = isinstance(dynamics, StructuredCriticDynamics)
    owned = (
        list(dynamics.parameters()) + backbone_parameters(actor)
        if actor is not None
        else list(dynamics.parameters())
    )
    learning_rate = (
        config.resolved_structured_critic_learning_rate
        if critic
        else config.resolved_structured_learning_rate
    )
    if config.optimizer == "normuon":

        def normuon(module: torch.nn.Module) -> NorMuon:
            return NorMuon(
                *route_parameters(module),
                learning_rate=learning_rate,
                adam_learning_rate=learning_rate * config.adam_learning_rate_ratio,
                momentum=config.normuon_momentum,
                beta2=config.normuon_beta2,
            )

        optimizer = normuon(dynamics)
        if actor is not None:
            # Routed as they would be inside the actor -- the backbone's names
            # and module types are unchanged by who steps it -- but into groups
            # of their own, so the backbone keeps its own warmup clock.
            for group in normuon(actor.trunk).param_groups:
                optimizer.add_param_group({**group, "role": BACKBONE_ROLE})
        return optimizer
    groups: list[dict[str, Any]] = [{"params": list(dynamics.parameters())}]
    if actor is not None:
        groups.append({"params": backbone_parameters(actor), "role": BACKBONE_ROLE})
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        eps=1e-5,
        weight_decay=0.0,
        fused=owned[0].device.type == "cuda",
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


def _entity_active_batch(staged: dict[str, Tensor], indices: Tensor | slice) -> Tensor:
    """Owned action entities, units then market orders; quantities are not agents."""
    return torch.cat(
        (
            _batch_tensor(staged["unit_active"], indices, torch.bool),
            _batch_tensor(staged["market_active"], indices, torch.bool),
        ),
        dim=-1,
    )


def _fixed_minibatch_positions(
    sample_count: int, minibatch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-shape positions plus genuine row counts for one epoch.

    Only the final minibatch may wrap to the ordering's beginning. Wrapped
    rows keep compiled graph shapes fixed; callers must give them zero weight.
    """
    if sample_count < 1 or minibatch_size < 1:
        raise ValueError("sample count and minibatch size must be positive")
    batch_count = math.ceil(sample_count / minibatch_size)
    positions = np.arange(batch_count * minibatch_size) % sample_count
    counts = np.full(batch_count, minibatch_size, dtype=np.int64)
    counts[-1] = sample_count - (batch_count - 1) * minibatch_size
    return positions.reshape(batch_count, minibatch_size), counts


def _contiguous_run_indices(
    valid_indices: np.ndarray,
    steps_per_trajectory: int,
    run_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute valid states as contiguous same-trajectory runs.

    Every index appears once. Each contiguous validity segment gets an
    independent random partition phase before its bounded runs are shuffled,
    so fixed-period game events are not systematically omitted as sources.
    Short edge runs are retained. ``run_length == 1`` is ``rng.permutation``.
    """
    if run_length < 1:
        raise ValueError("run length must be positive")
    if valid_indices.size == 0:
        raise ValueError("cannot build runs over no valid states")
    if steps_per_trajectory < 1:
        raise ValueError("steps per trajectory must be positive")
    if run_length == 1:
        return rng.permutation(valid_indices)
    indices = np.sort(valid_indices)
    trajectory = indices // steps_per_trajectory
    step = indices % steps_per_trajectory
    opens = np.ones(indices.size, dtype=bool)
    opens[1:] = (trajectory[1:] != trajectory[:-1]) | (step[1:] != step[:-1] + 1)
    positions = np.arange(indices.size)
    segment_starts = positions[opens]
    segment_lengths = np.diff(np.append(segment_starts, indices.size))
    within = positions - np.repeat(segment_starts, segment_lengths)
    phases = np.repeat(rng.integers(run_length, size=segment_starts.size), segment_lengths)
    block_starts = positions[opens | ((within + phases) % run_length == 0)]
    block_lengths = np.diff(np.append(block_starts, indices.size))
    order = rng.permutation(block_starts.size)
    shuffled_starts = block_starts[order]
    shuffled_lengths = block_lengths[order]
    offsets = np.cumsum(shuffled_lengths) - shuffled_lengths
    return indices[np.repeat(shuffled_starts - offsets, shuffled_lengths) + np.arange(indices.size)]


def _detached_belief(
    belief: StructuredBelief | StructuredDecisionBelief | StructuredCriticBelief,
):
    return type(belief)(*(field.detach() for field in belief))


def actor_lr_cooldown_scale(
    iteration: int, total_iterations: int, cooldown_fraction: float
) -> float:
    """Linear decay to zero over the final `cooldown_fraction` of the planned run.

    Polar Express divides its input by that input's Frobenius norm, so a NorMuon
    step's size is whatever the learning rate says and nothing the objective
    says (`optim.py:58-64`); Adam's `m_hat / sqrt(v_hat)` is scale-free for the
    same reason. Measured over three 90-100 wave arms, a 3.3x spread in
    `advantage_std` (0.085 / 0.284 / 0.252) produced a 1.26x spread in realized
    `approx_kl` (0.0030 / 0.0024 / 0.0024): the policy's displacement per step
    is set by the rate, not by how much the return says to move.

    With warmup as the only schedule that displacement is constant for the whole
    run, which is a constant entropy injection into a policy that starts peaked.
    `modded-nanogpt`, where this optimizer comes from, pairs it with a linear
    decay over the final 60% of training plus a momentum warmup and cooldown
    (`train_gpt.py:1992`, `:1995-2007`); this trainer kept the normalization and
    dropped the schedule.

    Actor only. The critic's rate is its tracking rate against data that moves
    every wave, so decaying it would grow the advantage bias being tested here.
    """
    if cooldown_fraction <= 0.0 or total_iterations <= 0:
        return 1.0
    start = (1.0 - min(cooldown_fraction, 1.0)) * float(total_iterations)
    if float(iteration) <= start:
        return 1.0
    span = float(total_iterations) - start
    return max(0.0, (float(total_iterations) - float(iteration)) / span)


def set_lr_cooldown(
    optimizer: torch.optim.Optimizer, scale: float, *, role: str | None = None
) -> None:
    """Multiply each group's warmup-scaled rate by `scale` from the next step.

    With `role`, only the groups carrying it anneal -- the shared backbone in the
    world model's optimizer, which moves the policy through its heads and so
    cools down with the actor while the projector and predictor do not.
    """
    for group in optimizer.param_groups:
        if role is None or group.get("role") == role:
            group["cooldown_scale"] = float(scale)


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    warmup_steps: int,
    found_inf: Tensor | None = None,
    *,
    advance_schedule: bool = True,
    held_role: str | None = None,
) -> None:
    """Advance the warmup schedule and step, optionally skipping on the device.

    `found_inf` follows GradScaler's device-side protocol.  Reading it here
    would serialize every gateable minibatch, so callers that defer the fatal
    read also restore the speculatively advanced Python schedule before raising.

    Groups whose `role` is `held_role` keep their warmup clock: the caller has
    dropped their gradients, so they take no step and their schedule must not
    count one.
    """
    if advance_schedule:
        for group in optimizer.param_groups:
            if held_role is not None and group.get("role") == held_role:
                continue
            step = int(group.get("warmup_step", 0)) + 1
            group["warmup_step"] = step
            group_base_lr = float(group.get("base_lr", base_learning_rate))
            scale = min(step / warmup_steps, 1.0) if warmup_steps else 1.0
            group["lr"] = group_base_lr * scale * float(group.get("cooldown_scale", 1.0))
    if found_inf is None:
        optimizer.step()
    else:
        optimizer.grad_scale = None
        optimizer.found_inf = found_inf
        try:
            optimizer.step()
        finally:
            del optimizer.grad_scale
            del optimizer.found_inf
    # The fused kernels -- `fused=True` AdamW and NorMuon's `_fused_adam_`
    # groups -- write parameters without advancing their version counters,
    # which `behavior_value_key` reads to tell whether values collected in the
    # rollout still describe the weights. Advanced here for every group, held
    # or gated off or not: a spurious advance only costs one replay.
    torch.autograd.graph.increment_version(
        [parameter for group in optimizer.param_groups for parameter in group["params"]]
    )


def _optimizer_schedule_state(
    optimizer: torch.optim.Optimizer,
) -> tuple[tuple[bool, int, float], ...]:
    """Capture enough Python schedule state to undo deferred skipped steps."""
    return tuple(
        (
            "warmup_step" in group,
            int(group.get("warmup_step", 0)),
            float(group["lr"]),
        )
        for group in optimizer.param_groups
    )


def _restore_skipped_optimizer_steps(
    optimizer: torch.optim.Optimizer,
    initial_state: tuple[tuple[bool, int, float], ...],
    skipped_steps: int,
    warmup_steps: int,
) -> None:
    """Remove device-skipped attempts from the Python-owned warmup clock."""
    if skipped_steps == 0:
        return
    for group, (had_step, initial_step, initial_lr) in zip(
        optimizer.param_groups, initial_state, strict=True
    ):
        restored_step = int(group["warmup_step"]) - skipped_steps
        if restored_step < initial_step:
            raise RuntimeError("optimizer warmup rollback crossed its initial state")
        if not had_step and restored_step == initial_step:
            group.pop("warmup_step")
        else:
            group["warmup_step"] = restored_step
        if restored_step == initial_step:
            group["lr"] = initial_lr
        else:
            base_lr = float(group.get("base_lr", initial_lr))
            scale = min(restored_step / warmup_steps, 1.0) if warmup_steps else 1.0
            group["lr"] = base_lr * scale * float(group.get("cooldown_scale", 1.0))


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
) -> tuple[Tensor, ...]:
    """Replay stored actions through the update-path policy forward.

    This defines the current-likelihood side of the PPO importance ratio in
    every actor minibatch. The behavior side is the sampler likelihood stored
    by the rollout. Autocast keeps log_softmax in fp32 by policy.
    """
    return _replayed_policy(
        actor,
        unit_actions,
        market_kinds,
        market_quantities,
        unit_masks,
        kind_masks,
        quantity_masks,
        autocast_enabled,
        *actor_args,
    )[0]


def _replayed_policy(
    actor: Actor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
    autocast_enabled: bool,
    *actor_args: Any,
) -> tuple[tuple[Tensor, ...], ActorOutput, Tensor]:
    """`_replayed_component_logprobs` plus the output and quantity logits it read.

    The logits are what a full-distribution term -- the reference KL -- needs
    beside the sampled likelihoods, from the same forward.
    """
    with torch.autocast(
        device_type=unit_actions.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        actor_output = actor(*actor_args)
        quantity_logits = actor.quantity_logits(actor_output.market_quantity_context, market_kinds)
        components = _output_component_logprobs(
            actor_output,
            quantity_logits,
            unit_actions,
            market_kinds,
            market_quantities,
            unit_masks,
            kind_masks,
            quantity_masks,
        )
    return components, actor_output, quantity_logits


def _output_component_logprobs(
    actor_output: ActorOutput,
    quantity_logits: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
) -> tuple[Tensor, ...]:
    """The replayed likelihoods and entropies of one forward's output."""
    components = component_logprobs(
        actor_output,
        quantity_logits,
        unit_actions,
        market_kinds,
        market_quantities,
        unit_masks,
        kind_masks,
        quantity_masks,
        validate_masks=False,
    )
    if isinstance(actor_output, StrategicOutput):
        return (
            *components[:3],
            actor_output.plan[:, 1:2],
            *components[3:],
            actor_output.plan[:, 2:3],
        )
    return components


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
) -> tuple[Tensor, ...]:
    """Entropy-free `_replayed_component_logprobs` for full-batch replays.

    The parity diagnostic sweeps every valid state but uses only the gathered
    log-likelihoods, so this variant skips the per-head entropy reductions the
    minibatch objective needs for its metrics.
    """
    with torch.autocast(
        device_type=unit_actions.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        actor_output = actor(*actor_args)
        components = component_selected_logprobs(
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
        if isinstance(actor_output, StrategicOutput):
            return (*components, actor_output.plan[:, 1:2])
        return components


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
    """Recompute likelihoods through the update graph for diagnostics.

    The caller supplies the exact row order whose sampling-versus-update drift
    it wants to measure. PPO's behavior distribution remains the likelihoods
    stored by the rollout and must never be replaced by this replay.
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
    if architecture == "strategic-plan":
        replayed["old_plan_logprobs"] = torch.zeros(rows, dtype=torch.float32, device=device)
    ordered = torch.from_numpy(valid_indices).to(device=device)
    positions, counts = _fixed_minibatch_positions(valid_indices.size, minibatch_size)
    staged_positions = torch.from_numpy(positions).to(device=device)
    for batch in range(positions.shape[0]):
        _begin_update_graph_step(compile_mode, device)
        indices = ordered[staged_positions[batch]]
        values_by_factor = replay(
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
        for name, values in zip(replayed, values_by_factor, strict=True):
            count = int(counts[batch])
            if name == "old_plan_logprobs":
                values = values.squeeze(-1)
            replayed[name].index_copy_(0, indices[:count], values[:count].float())
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


def _begin_update_graph_step(mode: str, device: torch.device) -> None:
    """Keep all compiled calls sharing live outputs in one logical iteration."""
    if device.type == "cuda" and mode in ("reduce-overhead", "max-autotune"):
        torch.compiler.cudagraph_mark_step_begin()


def _cached_update_callable(module: torch.nn.Module, attribute: str, function, mode: str):
    """Compile an update computation, cached per module and mode.

    The cache is keyed by mode because a run can measure several in one process
    -- the calibration chain and `update_replay_parity` both do -- and a single
    slot would hand back the first mode's artifact under a later mode's name,
    silently reporting one configuration's cost as another's.

    CUDA-graph modes require explicit minibatch boundaries: actor gradients
    survive the critic's compiled calls until the guarded optimizer step.
    Replay-only chunks own their copied results before advancing the boundary.
    Measure modes on the actual model and schedule; historical launch-overhead
    measurements on other architectures do not establish the best mode here.

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
        compiled = torch.compile(
            function, options=policy_compile_options(mode), fullgraph=True, dynamic=False
        )
        cache[mode] = compiled
    return compiled


def _validate_reference_actor(
    actor: Actor, reference_actor: Actor | None, config: PpoConfig
) -> None:
    """Require the anchor and its coefficient together, on a frozen twin of the actor."""
    if (reference_actor is None) != (config.reference_kl_coefficient == 0.0):
        raise ValueError("reference_kl_coefficient and reference_actor are required together")
    if reference_actor is None:
        return
    if isinstance(actor, (StrategicActor, CausalActor)):
        raise ValueError("the reference KL anchor covers the three component heads only")
    if reference_actor is actor or type(reference_actor) is not type(actor):
        raise ValueError("the reference actor must be a separate copy of the actor's class")
    if reference_actor.training or any(p.requires_grad for p in reference_actor.parameters()):
        raise ValueError("the reference actor must be frozen: eval mode, no gradients")


def _reference_kl_sums(
    reference_actor: Actor | None,
    output: ActorOutput,
    quantity_logits: Tensor,
    market_kinds: Tensor,
    masks: tuple[Tensor, Tensor, Tensor],
    active: tuple[Tensor, Tensor, Tensor],
    sample_weight: Tensor | None,
    *actor_args: Any,
) -> tuple[Tensor, Tensor]:
    """Forward KL from the frozen reference to the policy, over masked decisions.

    Returns the weighted sum over active decisions -- normalized outside, like
    the surrogate -- and, detached, the largest single decision's KL: a mean
    over a wave is exactly what cannot see one decision flip, and a per-state
    total would read a late state's many small divergences as one. The quantity
    distributions are both read at the stored market kinds, so each compares
    the same conditional. Zeros when there is no reference, so the compiled
    graph keeps one output signature.
    """
    zero = market_kinds.new_zeros((), dtype=torch.float32)
    if reference_actor is None:
        return zero, zero
    reference = reference_actor(*actor_args)
    reference_quantity = reference_actor.quantity_logits(
        reference.market_quantity_context, market_kinds
    )
    total = zero
    largest = zero
    for current, anchored, mask, weight in zip(
        (output.unit_logits, output.market_kind_logits, quantity_logits),
        (reference.unit_logits, reference.market_kind_logits, reference_quantity),
        masks,
        active,
        strict=True,
    ):
        anchored_log = mask_logits(anchored.detach(), mask, validate=False).log_softmax(-1)
        current_log = mask_logits(current, mask, validate=False).log_softmax(-1)
        kl = torch.where(mask, anchored_log.exp() * (anchored_log - current_log), 0.0).sum(-1)
        weight = weight.float()
        if sample_weight is not None:
            weight = weight * sample_weight[:, None]
        total = total + (kl * weight).sum()
        largest = torch.maximum(largest, torch.where(weight > 0, kl.detach(), 0.0).amax())
    return total, largest


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
    sample_weight: Tensor | None = None,
    policy_ratio_scope: str = "components",
    policy_objective: str = "clip",
    tpo_eta: float = 1.0,
    reference_actor: Actor | None = None,
) -> tuple[Tensor, ...]:
    """One actor minibatch: update likelihood plus clipped surrogate reductions.

    Returns device-side (policy objective sum, entropy sum, scoped k3 KL sum,
    scoped clipped count, component k3 KL sum, joint k3 KL sum, reference KL
    sum, largest single-decision reference KL). Normalization stays outside so host
    integers never enter the graph.

    Entropy and KL are telemetry, not objective terms. Detaching their sums
    inside this compiled region preserves their exact forward values while
    keeping their softmax-sized derivative branches and saved intermediates
    out of the actor backward.
    """
    replayed, output, quantity_logits = _replayed_policy(
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
    with torch.autocast(
        device_type=unit_actions.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        reference_kl = _reference_kl_sums(
            reference_actor,
            output,
            quantity_logits,
            market_kinds,
            (unit_masks, kind_masks, quantity_masks),
            (unit_active, kind_active, quantity_active),
            sample_weight,
            *actor_args,
        )
    factor_count = len(replayed) // 2
    old = (old_unit, old_kind, old_quantity)
    active = (unit_active, kind_active, quantity_active)
    if factor_count == 4:
        if policy_ratio_scope != "joint":
            raise ValueError("strategic-plan PPO requires joint policy ratios")
        choice = actor_args[1]
        old += (choice.old_logprobs[:, None],)
        active += (choice.active[:, None],)
    return (
        *_policy_sums(
            replayed[:factor_count],
            old,
            active,
            replayed[factor_count:],
            advantages,
            clip_low,
            clip_high,
            sample_weight=sample_weight,
            policy_ratio_scope=policy_ratio_scope,
            policy_objective=policy_objective,
            tpo_eta=tpo_eta,
        ),
        *reference_kl,
    )


class _BalancedSource(torch.autograd.Function):
    """Fork head inputs, balancing main/auxiliary cotangents before one trunk VJP.

    The norm spans every source head input together, not individual tokens.
    Readout and predictor parameter gradients keep their own objectives.
    Forward is zero-copy and bit-identical; Inductor compiles the backward
    reductions and mix along with the existing model backward.
    """

    @staticmethod
    def forward(ctx, *fields: Tensor) -> tuple[Tensor, ...]:
        ctx.count = len(fields)
        return tuple(field.view_as(field) for field in (*fields, *fields))

    @staticmethod
    def backward(ctx, *gradients: Tensor) -> tuple[Tensor, ...]:
        primary = gradients[: ctx.count]
        auxiliary = gradients[ctx.count :]
        primary_norm = torch.stack(tuple(g.float().square().sum() for g in primary)).sum().sqrt()
        auxiliary_norm = (
            torch.stack(tuple(g.float().square().sum() for g in auxiliary)).sum().sqrt()
        )
        has_auxiliary = auxiliary_norm != 0
        primary_scale = torch.where(has_auxiliary, 0.5, 1.0)
        auxiliary_scale = torch.where(
            has_auxiliary,
            0.5
            * primary_norm
            / torch.where(has_auxiliary, auxiliary_norm, torch.ones_like(auxiliary_norm)),
            0.0,
        )
        return tuple(
            (left.float() * primary_scale + right.float() * auxiliary_scale).to(left.dtype)
            for left, right in zip(primary, auxiliary, strict=True)
        )


def _structured_actor_minibatch_terms(
    actor: StructuredActor | EntityActor,
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
    sample_weight: Tensor | None = None,
    policy_ratio_scope: str = "components",
    policy_objective: str = "clip",
    tpo_eta: float = 1.0,
    reference_actor: Actor | None = None,
) -> tuple[Tensor, ...]:
    """PPO terms, the reference KL and the belief from one structured actor forward."""
    (inputs,) = actor_args
    if not isinstance(inputs, StructuredInputs):
        raise TypeError("structured actor minibatches require StructuredInputs")
    with torch.autocast(
        device_type=unit_actions.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        output, belief = actor.forward_with_auxiliary_belief(inputs)
        quantity_logits = actor.quantity_logits(output.market_quantity_context, market_kinds)
        (
            new_unit,
            new_kind,
            new_quantity,
            unit_entropy,
            kind_entropy,
            quantity_entropy,
        ) = component_logprobs(
            output,
            quantity_logits,
            unit_actions,
            market_kinds,
            market_quantities,
            unit_masks,
            kind_masks,
            quantity_masks,
            validate_masks=False,
        )
        reference_kl = _reference_kl_sums(
            reference_actor,
            output,
            quantity_logits,
            market_kinds,
            (unit_masks, kind_masks, quantity_masks),
            (unit_active, kind_active, quantity_active),
            sample_weight,
            inputs,
        )
    return (
        *_policy_sums(
            (new_unit, new_kind, new_quantity),
            (old_unit, old_kind, old_quantity),
            (unit_active, kind_active, quantity_active),
            (unit_entropy, kind_entropy, quantity_entropy),
            advantages,
            clip_low,
            clip_high,
            sample_weight=sample_weight,
            policy_ratio_scope=policy_ratio_scope,
            policy_objective=policy_objective,
            tpo_eta=tpo_eta,
        ),
        *reference_kl,
        *belief,
    )


def _value_objective(
    critic: Critic,
    critic_logits: Tensor,
    value_targets: Tensor,
    *,
    sample_weight: Tensor | None = None,
    head_active: Tensor | None = None,
) -> Tensor:
    """The critic's per-minibatch objective, in whichever parameterization it has.

    The scalar branch reads the prediction back through `critic.value`, which is
    a squeeze there, so both branches spend exactly one readout.
    """
    if head_active is not None:
        # Sanitize before nonlinear losses; multiplying inactive NaNs by zero
        # afterwards does not protect either the objective or its backward.
        valid = head_active.bool()
        if sample_weight is not None:
            valid = valid & sample_weight[:, None].bool()
        critic_logits = torch.where(valid[..., None], critic_logits, 0.0)
        value_targets = torch.where(valid.any(dim=-1), value_targets, 0.0)
        if critic.config.scalar_value:
            per_state = scalar_value_loss(
                critic.value(critic_logits), value_targets[:, None].expand_as(valid)
            )
        else:
            # The return is shared: build the HL label once per state, not
            # once per head. Only the predicted distributions differ.
            projected = hl_gauss_value_targets(
                value_targets.detach(),
                critic.support,
                sigma_ratio=critic.config.value_sigma_ratio,
                validate=False,
            )
            per_state = -(projected[:, None] * critic_logits.float().log_softmax(dim=-1)).sum(
                dim=-1
            )
    elif critic.config.scalar_value:
        per_state = scalar_value_loss(critic.value(critic_logits), value_targets)
    else:
        per_state = distributional_value_loss(
            critic_logits,
            value_targets,
            critic.support,
            sigma_ratio=critic.config.value_sigma_ratio,
            validate=False,
        )
    if head_active is not None:
        per_state = torch.where(valid, per_state, 0.0).sum(dim=-1) / (
            head_active.sum(dim=-1).clamp_min(1)
        )
    if sample_weight is None:
        return per_state.mean()
    return (per_state * sample_weight).sum() / sample_weight.sum().clamp_min(1)


def _critic_logits_and_loss(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
    sample_weight: Tensor | None = None,
    entity_active: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The shared critic forward and value objective."""
    with torch.autocast(
        device_type=value_targets.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        if isinstance(critic, StructuredCritic) and getattr(
            critic.config, "per_entity_critic", False
        ):
            critic_logits, belief = critic.forward_with_belief(*critic_args)
            entity_logits = critic.decode_entity_belief(belief)
        else:
            critic_logits = critic(*critic_args)
            entity_logits = None
    return (
        _critic_readout_objective(
            critic,
            critic_logits,
            entity_logits,
            value_targets,
            sample_weight=sample_weight,
            entity_active=entity_active,
        ),
        critic_logits,
    )


def _critic_readout_objective(
    critic: Critic,
    global_logits: Tensor,
    entity_logits: Tensor | None,
    value_targets: Tensor,
    *,
    sample_weight: Tensor | None,
    entity_active: Tensor | None,
) -> Tensor:
    if entity_logits is None:
        return _value_objective(critic, global_logits, value_targets, sample_weight=sample_weight)
    if entity_active is None:
        raise ValueError("per-entity critic training requires active entity masks")
    head_active = torch.cat((torch.ones_like(entity_active[:, :1]), entity_active), dim=-1)
    return _value_objective(
        critic,
        torch.cat((global_logits[:, None], entity_logits), dim=1),
        value_targets,
        sample_weight=sample_weight,
        head_active=head_active,
    )


def _critic_minibatch_objective(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
    sample_weight: Tensor | None = None,
    entity_active: Tensor | None = None,
) -> Tensor:
    """One critic minibatch without the value telemetry scoring epochs need."""
    loss, _critic_logits = _critic_logits_and_loss(
        critic,
        value_targets,
        autocast_enabled,
        *critic_args,
        sample_weight=sample_weight,
        entity_active=entity_active,
    )
    return loss


def _critic_minibatch_loss(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
    sample_weight: Tensor | None = None,
    entity_active: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """One critic minibatch: the distributional loss and the mean it implies.

    The predicted mean rides along because the forward that produced the logits
    is the only place it is free. Callers that do not score this prediction use
    `_critic_minibatch_objective`, avoiding the softmax and support reduction.
    """
    loss, critic_logits = _critic_logits_and_loss(
        critic,
        value_targets,
        autocast_enabled,
        *critic_args,
        sample_weight=sample_weight,
        entity_active=entity_active,
    )
    return loss, critic.value(critic_logits).detach()


def _critic_minibatch_fit_terms(
    critic: Critic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
    sample_weight: Tensor | None = None,
    entity_active: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Critic loss plus float64 target/residual moments for a scoring epoch.

    Keeping the moment computation inside the compiled region lets Inductor
    consume the predicted values where they are produced instead of returning
    a full minibatch and materializing five eager float64 intermediates.
    """
    loss, predictions = _critic_minibatch_loss(
        critic,
        value_targets,
        autocast_enabled,
        *critic_args,
        sample_weight=sample_weight,
        entity_active=entity_active,
    )
    targets = value_targets.double()
    residuals = targets - predictions.double()
    moments = _value_fit_moments(targets, residuals, sample_weight)
    return loss, moments


def _forecast_critic_minibatch_fit_terms(
    critic: EntityCritic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
    sample_weight: Tensor,
    forecast_targets: Tensor,
    forecast_valid: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    market_active: Tensor,
    market_quantity_active: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One state encoding, separate V(s) and action-conditioned forecasts."""
    with torch.autocast(
        device_type=value_targets.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        logits, forecasts = critic.forward_with_forecasts(
            *critic_args,
            unit_actions=unit_actions,
            market_kinds=market_kinds,
            market_quantities=market_quantities,
            market_active=market_active,
            market_quantity_active=market_quantity_active,
        )
    value_loss = _value_objective(critic, logits, value_targets, sample_weight=sample_weight)
    targets = value_targets.double()
    residuals = targets - critic.value(logits).detach().double()
    moments = _value_fit_moments(targets, residuals, sample_weight)
    forecast_loss = economic_forecast_loss(
        forecasts, forecast_targets, forecast_valid, sample_weight
    )
    with torch.no_grad():
        persistence_loss = economic_forecast_loss(
            torch.zeros_like(forecasts), forecast_targets, forecast_valid, sample_weight
        )
    return value_loss, moments, forecast_loss, persistence_loss


def _structured_critic_minibatch_fit_terms(
    critic: StructuredCritic | EntityCritic,
    value_targets: Tensor,
    autocast_enabled: bool,
    *critic_args: Any,
    sample_weight: Tensor | None = None,
    balance_auxiliary: bool = False,
    entity_active: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """Critic loss, fit moments, and belief from one forward."""
    with torch.autocast(
        device_type=value_targets.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        belief = critic.encode_belief(*critic_args)
        primary_belief = belief
        if getattr(critic.config, "per_entity_critic", False):
            belief = StructuredCriticBelief(belief.value_decision[:, :1])
        if balance_auxiliary:
            if getattr(critic.config, "per_entity_critic", False):
                primary_belief_entities = primary_belief.value_decision[:, 1:]
            primary, auxiliary = _BalancedSource.apply(belief.value_decision)
            primary_belief = StructuredCriticBelief(primary)
            if getattr(critic.config, "per_entity_critic", False):
                primary_belief = StructuredCriticBelief(
                    torch.cat((primary, primary_belief_entities), dim=1)
                )
            belief = StructuredCriticBelief(auxiliary)
        critic_logits = critic.decode_belief(primary_belief)
        entity_logits = (
            critic.decode_entity_belief(primary_belief)
            if getattr(critic.config, "per_entity_critic", False)
            else None
        )
    loss = _critic_readout_objective(
        critic,
        critic_logits,
        entity_logits,
        value_targets,
        sample_weight=sample_weight,
        entity_active=entity_active,
    )
    predictions = critic.value(critic_logits).detach()
    targets = value_targets.double()
    residuals = targets - predictions.double()
    moments = _value_fit_moments(targets, residuals, sample_weight)
    return (loss, moments, *belief)


def _value_fit_moments(targets: Tensor, residuals: Tensor, sample_weight: Tensor | None) -> Tensor:
    values = (targets, targets.square(), residuals, residuals.square())
    return torch.stack(
        tuple(
            value.sum() if sample_weight is None else (value * sample_weight).sum()
            for value in values
        )
    )


# `torch.no_grad` rather than inference mode keeps every explicit replay
# diagnostic on one Dynamo specialization of `_replayed_selected_logprobs`.
# Inference mode would compile a second entry per minibatch shape and can exhaust
# the per-code-object cache under `fullgraph=True`. Sharing the context also
# means `replay_behavior_logprobs` and this audit execute literally the same
# diagnostic graph.
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

    Runs the same staging, minibatch slicing, and policy forward as the PPO
    update and compares its log-likelihoods with those of the distribution that
    actually sampled the rollout. The replay is diagnostic only: the surrogate
    itself uses the stored sampler likelihoods, so this divergence is also
    present in its importance ratio. Pass the production `use_bfloat16` flag so
    the audited path is the deployed one.

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
    replay = _cached_update_callable(
        actor,
        "_kaggriculture_logprob_replay",
        _replayed_selected_logprobs,
        _device_compile_mode(compile_mode, device),
    )
    components = ("unit", "kind", "quantity")
    if rollout.architecture == "strategic-plan":
        components += ("plan",)
    maximum_logprob_error = dict.fromkeys(components, 0.0)
    maximum_ratio_error = dict.fromkeys(components, 0.0)
    active_counts = dict.fromkeys(components, 0)
    # Accumulated in float64 on the host: the per-head sums run to hundreds of
    # thousands of terms whose individual magnitudes are near the float32
    # rounding floor, which is precisely the regime where a float32 running sum
    # loses the quantity being measured.
    kl_sums = dict.fromkeys(components, 0.0)
    tail_counts = dict.fromkeys(components, 0)
    # Worst component-mean KL in a production-shaped minibatch.
    worst_minibatch_kl = 0.0
    joint_kl_sum = 0.0
    positions, counts = _fixed_minibatch_positions(valid_indices.size, minibatch_size)
    staged_positions = torch.from_numpy(positions).to(device=device)
    for batch in range(positions.shape[0]):
        _begin_update_graph_step(compile_mode, device)
        indices = ordered[staged_positions[batch]]
        # The replay runs the padded shape so it compiles once, but this is an
        # audit: the wrapped tail's repeated rows are excluded from every sum so
        # the reported means and tail counts stay per-state exact.
        rows = int(counts[batch])
        joint_log_ratio = torch.zeros(rows, device=device, dtype=torch.float64)
        minibatch_component_kl_sum = 0.0
        minibatch_components = 0
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
        factor_metadata = (
            ("unit", replayed[0][:rows], "old_unit_logprobs", "unit_active"),
            ("kind", replayed[1][:rows], "old_market_kind_logprobs", "market_active"),
            (
                "quantity",
                replayed[2][:rows],
                "old_market_quantity_logprobs",
                "market_quantity_active",
            ),
        )
        if rollout.architecture == "strategic-plan":
            factor_metadata += (("plan", replayed[3][:rows], "old_plan_logprobs", "plan_active"),)
        for name, new_logprobs, old_key, active_key in factor_metadata:
            active = _batch_tensor(staged[active_key], indices, torch.bool)[:rows]
            old_logprobs = _batch_tensor(staged[old_key], indices, torch.float32)[:rows]
            if name == "plan":
                active = active[:, None]
                old_logprobs = old_logprobs[:, None]
            row_difference = torch.where(
                active,
                new_logprobs.float() - old_logprobs,
                0.0,
            )
            joint_log_ratio += row_difference.double().sum(-1)
            active_count = int(active.sum())
            active_counts[name] += active_count
            minibatch_components += active_count
            if not active_count:
                continue
            difference = new_logprobs[active].float() - old_logprobs[active]
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
            minibatch_component_kl_sum += head_kl_sum
            tail_counts[name] += int((widened.abs() > UPDATE_REPLAY_TAIL_LOGPROB).sum())
        joint_kl_sum += float((torch.expm1(joint_log_ratio) - joint_log_ratio).sum())
        worst_minibatch_kl = max(
            worst_minibatch_kl, minibatch_component_kl_sum / max(1, minibatch_components)
        )
    first_minibatch_kl, mean_first_minibatch_kl = _replay_to_update_minibatch_kl(
        actor,
        rollout,
        staged,
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
        "update_replay_component_kl": total_kl_sum / total_active if total_active else 0.0,
        "update_replay_joint_kl": joint_kl_sum / valid_indices.size,
        # Active-component mean k3, independent of the selected PPO ratio scope.
        "update_replay_minibatch_kl": worst_minibatch_kl,
        # The fixed-shuffle sampling-versus-update statistic
        # `MAX_FIRST_MINIBATCH_KL` bounds. Unlike the no-grad replay numbers
        # above, this executes the grad-tracking graph used by the surrogate.
        "update_replay_first_minibatch_kl": first_minibatch_kl,
        # The same comparison averaged over every minibatch instead of
        # maximized.
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
    """Worst and mean per-minibatch KL from sampler to update forward.

    This is the quantity `MAX_FIRST_MINIBATCH_KL` bounds. The behavior side is
    the rollout-stored sampling likelihood; the comparison runs
    `_actor_minibatch_terms` over shuffled minibatches through the same
    grad-tracking graph as the optimizer. Advantages are zero because the k3
    sum does not depend on them, and every minibatch of one epoch is measured
    rather than only the first. The epoch mean is weighted by active components.
    """
    device = next(actor.parameters()).device
    resolved_mode = _device_compile_mode(compile_mode, device)
    # The k3 sum ignores both the advantages and the clip bounds, so the
    # defaults stand in for a config this audit is not otherwise given.
    clip = PpoConfig()
    # Keep the audit's compiled callable separate from the optimizer's so fixed
    # diagnostic modes and shapes cannot specialize or evict the production
    # update cache.
    terms = _cached_update_callable(
        actor, "_kaggriculture_update_audit_terms", _actor_minibatch_terms, resolved_mode
    )
    # A fixed shuffle makes the sampled per-minibatch maximum reproducible
    # between audit waves.
    shuffled = np.random.default_rng(_REPLAY_AUDIT_SHUFFLE_SEED).permutation(valid_indices)
    shuffled_device = torch.from_numpy(shuffled).to(device=device)
    flat_component_counts = (
        rollout.unit_active.reshape(rollout.valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_active.reshape(rollout.valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_quantity_active.reshape(rollout.valid.size, -1).sum(axis=1, dtype=np.int64)
    )
    if rollout.architecture == "strategic-plan":
        flat_component_counts += rollout.states["plan_active"].reshape(-1).astype(np.int64)
    zero_advantages = torch.zeros(minibatch_size, dtype=torch.float32, device=device)
    worst = 0.0
    total = 0.0
    measured = 0
    positions, counts = _fixed_minibatch_positions(shuffled.size, minibatch_size)
    staged_positions = torch.from_numpy(positions).to(device=device)
    sample_weights = torch.from_numpy(
        (np.arange(minibatch_size)[None, :] < counts[:, None]).astype(np.float32)
    ).to(device=device)
    for batch in range(positions.shape[0]):
        _begin_update_graph_step(resolved_mode, device)
        indices = shuffled_device[staged_positions[batch]]
        component_count = int(
            flat_component_counts[shuffled[positions[batch, : counts[batch]]]].sum()
        )
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
            policy_sum, entropy_sum, kl_sum, clipped_sum, component_kl_sum, *_unused = terms(
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
                sample_weight=sample_weights[batch],
                policy_ratio_scope="joint"
                if rollout.architecture == "strategic-plan"
                else "components",
            )
        minibatch_kl_sum = float(component_kl_sum.detach().double())
        minibatch_kl = minibatch_kl_sum / max(1, component_count)
        del policy_sum, entropy_sum, kl_sum, clipped_sum, component_kl_sum, _unused
        worst = max(worst, minibatch_kl)
        total += minibatch_kl_sum
        measured += component_count
    return worst, total / measured if measured else 0.0


def _component_policy_sums(
    new_logprobs: tuple[Tensor, Tensor, Tensor],
    old_logprobs: tuple[Tensor, Tensor, Tensor],
    active: tuple[Tensor, Tensor, Tensor],
    entropies: tuple[Tensor, Tensor, Tensor],
    advantages: Tensor,
    clip_low: float,
    clip_high: float,
    *,
    sample_weight: Tensor | None = None,
    policy_objective: str = "clip",
    tpo_eta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sum independent component objectives and expose joint KL telemetry."""
    objective = torch.zeros((), device=advantages.device)
    entropy_sum = torch.zeros_like(objective)
    kl = torch.zeros_like(objective)
    clipped = torch.zeros_like(objective)
    joint_log_ratio = advantages.new_zeros(new_logprobs[0].shape[0])
    if advantages.ndim == 1:
        component_advantages = (advantages, advantages, advantages)
    else:
        unit_count = new_logprobs[0].shape[-1]
        unit_advantages = advantages[:, :unit_count]
        order_advantages = advantages[:, unit_count:]
        component_advantages = (unit_advantages, order_advantages, order_advantages)
    for new, old, mask, entropy, advantage in zip(
        new_logprobs, old_logprobs, active, entropies, component_advantages, strict=True
    ):
        weight = mask.float()
        if sample_weight is not None:
            weight = torch.where(sample_weight[:, None].bool(), weight, 0.0)
            weight = weight * sample_weight[:, None]
        component_objective, component_kl, component_clipped = _decision_objective_sums(
            new, old, advantage, weight, clip_low, clip_high, policy_objective, tpo_eta
        )
        objective = objective + component_objective
        kl = kl + component_kl
        clipped = clipped + component_clipped
        entropy_sum = entropy_sum + (torch.where(weight.bool(), entropy, 0.0) * weight).sum()
        log_ratio = torch.where(weight.bool(), new.float() - old.float(), 0.0)
        joint_log_ratio = joint_log_ratio + log_ratio.sum(dim=-1)
    joint_state_weight = (
        torch.ones_like(joint_log_ratio) if sample_weight is None else sample_weight
    )
    joint_kl = ((torch.expm1(joint_log_ratio) - joint_log_ratio) * joint_state_weight).sum()
    return objective, entropy_sum.detach(), kl.detach(), clipped, kl.detach(), joint_kl.detach()


def _policy_sums(
    new_logprobs: tuple[Tensor, ...],
    old_logprobs: tuple[Tensor, ...],
    active: tuple[Tensor, ...],
    entropies: tuple[Tensor, ...],
    advantages: Tensor,
    clip_low: float,
    clip_high: float,
    *,
    sample_weight: Tensor | None = None,
    policy_ratio_scope: str = "components",
    policy_objective: str = "clip",
    tpo_eta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Keep component KL and joint product-ratio KL independently observable."""
    if policy_ratio_scope == "components":
        terms = _component_policy_sums(
            new_logprobs,
            old_logprobs,
            active,
            entropies,
            advantages,
            clip_low,
            clip_high,
            sample_weight=sample_weight,
            policy_objective=policy_objective,
            tpo_eta=tpo_eta,
        )
        return terms
    if policy_ratio_scope != "joint":
        raise ValueError("policy ratio scope must be 'components' or 'joint'")
    if advantages.ndim != 1:
        raise ValueError("joint policy ratios require scalar state advantages")

    state_weight = (
        torch.ones_like(advantages, dtype=torch.float32) if sample_weight is None else sample_weight
    )
    valid_states = state_weight.bool()
    joint_log_ratio = torch.zeros_like(advantages, dtype=torch.float32)
    entropy_sum = torch.zeros((), device=advantages.device)
    component_kl_sum = torch.zeros_like(entropy_sum)
    for new, old, mask, entropy in zip(new_logprobs, old_logprobs, active, entropies, strict=True):
        valid = mask.bool() & valid_states[:, None]
        # Mask before any state reduction: STOP descendants, inactive quantities,
        # and wrapped padding never enter the action's likelihood ratio.
        log_ratio = torch.where(valid, new.float() - old.float(), 0.0)
        joint_log_ratio = joint_log_ratio + log_ratio.sum(dim=-1)
        weight = valid.float() * state_weight[:, None]
        component_kl_sum = component_kl_sum + ((torch.expm1(log_ratio) - log_ratio) * weight).sum()
        entropy_sum = entropy_sum + (torch.where(valid, entropy, 0.0) * weight).sum()
    # The product ratio is represented in log space until the surrogate chooses
    # its one-sided clipping bound. KL retains the unclamped total log ratio.
    # The surrogate needs only that ratio; the TPO target also anchors to the
    # joint sampled likelihood, so it receives the summed old log-likelihood.
    joint_old = torch.zeros_like(joint_log_ratio)
    if policy_objective == "tpo":
        for old, mask in zip(old_logprobs, active, strict=True):
            valid = mask.bool() & valid_states[:, None]
            joint_old = joint_old + torch.where(valid, old.float(), 0.0).sum(dim=-1)
    objective, kl, clipped = _decision_objective_sums(
        (joint_old + joint_log_ratio)[:, None],
        joint_old[:, None],
        advantages,
        state_weight[:, None],
        clip_low,
        clip_high,
        policy_objective,
        tpo_eta,
    )
    return (
        objective,
        entropy_sum.detach(),
        kl.detach(),
        clipped,
        component_kl_sum.detach(),
        kl.detach(),
    )


def _clipped_surrogate_sums(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    active: Tensor,
    clip_low: float,
    clip_high: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return weighted per-component PPO objective, k3 KL, and clipped count."""
    valid = active.bool()
    log_ratio = torch.where(valid, new_logprobs.float() - old_logprobs.float(), 0.0)
    expanded_advantage = torch.where(
        valid, advantages.float()[:, None] if advantages.ndim == 1 else advantages.float(), 0.0
    )
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


def _tpo_target_sums(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    active: Tensor,
    eta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted single-sample TPO objective, k3 KL, and a zero clipped count.

    Per decision the target is `logit q = logit p_old + A / eta` on the sampled
    action and the objective is minus the Bernoulli KL from `q` to the current
    sampled-action likelihood, so it is zero exactly at the target and its
    logit gradient is `p - q`. Saturated decisions (`_TPO_SATURATED_LOGPROB`)
    have no target and contribute nothing; the KL telemetry still counts them
    at their (zero) log-ratio like the clipped surrogate does.
    """
    valid = active.bool()
    old = old_logprobs.float()
    new = new_logprobs.float()
    log_ratio = torch.where(valid, new - old, 0.0)
    approximate_kl_sum = ((torch.expm1(log_ratio) - log_ratio) * active).sum()
    informative = valid & (old < _TPO_SATURATED_LOGPROB)
    # Clamp only inside the informative rows so the masked ones never see the
    # log of a nonpositive number; their value is discarded by the `where`.
    old = torch.where(informative, old, _TPO_SATURATED_LOGPROB)
    new = torch.where(informative, new.clamp_max(_TPO_SATURATED_LOGPROB), _TPO_SATURATED_LOGPROB)
    expanded_advantage = advantages.float()[:, None] if advantages.ndim == 1 else advantages.float()
    target_logit = old - torch.log(-torch.expm1(old)) + expanded_advantage / eta
    q = torch.sigmoid(target_logit)
    divergence = q * (F.logsigmoid(target_logit) - new) + (1.0 - q) * (
        F.logsigmoid(-target_logit) - torch.log(-torch.expm1(new))
    )
    objective_sum = -(torch.where(informative, divergence, 0.0) * active).sum()
    return objective_sum, approximate_kl_sum, torch.zeros_like(objective_sum)


def _decision_objective_sums(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    active: Tensor,
    clip_low: float,
    clip_high: float,
    policy_objective: str,
    tpo_eta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """The configured per-decision objective, in the clipped surrogate's contract."""
    if policy_objective == "tpo":
        return _tpo_target_sums(new_logprobs, old_logprobs, advantages, active, tpo_eta)
    if policy_objective != "clip":
        raise ValueError(f"unsupported policy objective {policy_objective!r}")
    return _clipped_surrogate_sums(
        new_logprobs, old_logprobs, advantages, active, clip_low, clip_high
    )


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


def _r_squared(targets: np.ndarray, predictions: np.ndarray, valid: np.ndarray) -> float:
    """Bias-sensitive pre-update fit on valid owned rows, unlike centered EV."""
    selected_targets = targets[valid]
    if selected_targets.size < 2:
        return 0.0
    variance = float(np.var(selected_targets, dtype=np.float64))
    if variance < 1e-12:
        return 0.0
    residual = np.subtract(selected_targets, predictions[valid], dtype=np.float64)
    return 1.0 - float(np.mean(np.square(residual))) / variance


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
    if "tile_categorical" in staged:
        # The tile embedder derives these columns from the token slot; a stored
        # observation that disagrees would be silently re-labelled.
        tiles = staged["tile_categorical"]
        slots = torch.from_numpy(TILE_SLOT_CATEGORICAL).to(device=tiles.device, dtype=tiles.dtype)
        flags.append((tiles[..., 2:6] != slots).any())
        messages.append("tile farm/row/column/quadrant columns do not match their token slots")
    # One aggregated host readback replaces per-field synchronizing checks.
    failures = torch.stack(flags).cpu()
    for failed, message in zip(failures.tolist(), messages, strict=True):
        if failed:
            raise ValueError(message)


_STRUCTURED_AUXILIARY_METRICS = (
    "latent",
    "decision",
    "decision_one",
    "decision_final",
    "decision_unit",
    "decision_market_kind",
    "decision_market_quantity",
    "eligible",
    "residual_ratio",
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
    actor: StructuredActor | EntityActor,
    dynamics: ActorDynamics,
    staged: dict[str, Tensor],
    indices: Tensor,
    *,
    steps_per_trajectory: int,
    config: PpoConfig,
    autocast_enabled: bool,
    model_grad: bool,
    complete_windows: bool,
    belief_indices: Tensor | None = None,
    belief_inverse: Tensor | None = None,
    belief: StructuredDecisionBelief | None = None,
    plan: StructuredHorizonPlan | None = None,
) -> tuple[Tensor, ActorDynamicsTerms]:
    """Predict actor head inputs, reusing the single PPO forward.

    Both source head-input fields carry auxiliary gradients; successor targets
    and final policy projections are stop-gradient.
    ``model_grad`` controls source encoding when no belief is supplied.
    """
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
    latent_horizon = (
        config.structured_decision_horizon if config.structured_latent_coefficient else 0
    )
    with torch.autocast(
        device_type=indices.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        if belief is not None:
            if belief_indices is not None or belief_inverse is not None:
                raise ValueError("a supplied belief cannot also name unique rows")
        elif model_grad:
            if belief_indices is not None or belief_inverse is not None:
                raise ValueError("actor-gradient auxiliary cannot reuse unique-row beliefs")
            belief = actor.auxiliary_belief(inputs)
        elif belief_indices is not None and belief_inverse is not None:
            (belief_inputs,) = _actor_batch_args(STRUCTURED, staged, belief_indices)
            if not isinstance(belief_inputs, StructuredInputs):
                raise TypeError("structured auxiliary requires StructuredInputs")
            with torch.no_grad():
                unique_belief = actor.auxiliary_belief(belief_inputs)
            belief = StructuredDecisionBelief(*(value[belief_inverse] for value in unique_belief))
        elif belief_indices is None and belief_inverse is None:
            with torch.no_grad():
                belief = actor.auxiliary_belief(inputs)
        else:
            raise ValueError("belief indices and inverse must be supplied together")
        loss_function = actor_window_loss if complete_windows else actor_horizon_loss
        terms = loss_function(
            dynamics,
            actor,
            belief,
            inputs,
            factors,
            decision_horizon=decision_horizon,
            latent_horizon=latent_horizon,
            **({} if complete_windows else {"plan": plan}),
        )
        loss = (
            config.structured_latent_coefficient * terms.latent
            + config.structured_decision_coefficient * terms.decision
        )
    return loss, terms


def _structured_auxiliary_horizon(config: PpoConfig) -> int:
    if config.jepa_active:
        return config.jepa_horizon
    return config.structured_decision_horizon if config.structured_actor_auxiliary_active else 0


def _structured_run_length(config: PpoConfig) -> int:
    """States per contiguous run: one more than the longest active horizon."""
    horizon = _structured_auxiliary_horizon(config)
    if config.structured_critic_auxiliary_active:
        horizon = max(horizon, config.structured_critic_horizon)
    return max(horizon, 0) + 1 if horizon else 1


_STRUCTURED_CRITIC_AUXILIARY_METRICS = ("latent", "value", "eligible", "residual_ratio")


def _structured_critic_auxiliary_terms(
    critic: StructuredCritic | EntityCritic,
    dynamics: StructuredCriticDynamics,
    staged: dict[str, Tensor],
    indices: Tensor,
    *,
    steps_per_trajectory: int,
    config: PpoConfig,
    autocast_enabled: bool,
    model_grad: bool,
    complete_windows: bool,
    belief_indices: Tensor | None = None,
    belief_inverse: Tensor | None = None,
    belief: StructuredCriticBelief | None = None,
    plan: StructuredHorizonPlan | None = None,
) -> tuple[Tensor, StructuredCriticDynamicsTerms]:
    """Evaluate critic NextLat on run-length batches or complete windows."""
    critic_args = _critic_batch_args(STRUCTURED, staged, indices)
    inputs = critic_args[0]
    if not isinstance(inputs, StructuredInputs):
        raise TypeError("structured critic auxiliary requires StructuredInputs")
    factors = {
        "unit_actions": _batch_tensor(staged["unit_actions"], indices, torch.long),
        "market_kinds": _batch_tensor(staged["market_kinds"], indices, torch.long),
        "market_quantities": _batch_tensor(staged["market_quantities"], indices, torch.long),
        "episode_index": torch.div(indices, steps_per_trajectory, rounding_mode="floor"),
        "step": indices.remainder(steps_per_trajectory),
    }
    with torch.autocast(
        device_type=indices.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        if belief is not None:
            if belief_indices is not None or belief_inverse is not None:
                raise ValueError("a supplied belief cannot also name unique rows")
        elif model_grad:
            if belief_indices is not None or belief_inverse is not None:
                raise ValueError("critic-gradient auxiliary cannot reuse unique-row beliefs")
            _, belief = critic.forward_with_belief(*critic_args)
        elif belief_indices is not None and belief_inverse is not None:
            unique_args = _critic_batch_args(STRUCTURED, staged, belief_indices)
            with torch.no_grad():
                _, unique_belief = critic.forward_with_belief(*unique_args)
            belief = type(unique_belief)(*(value[belief_inverse] for value in unique_belief))
        elif belief_indices is None and belief_inverse is None:
            with torch.no_grad():
                _, belief = critic.forward_with_belief(*critic_args)
        else:
            raise ValueError("belief indices and inverse must be supplied together")
        if getattr(critic.config, "per_entity_critic", False):
            belief = StructuredCriticBelief(belief.value_decision[:, :1])
        loss_function = (
            structured_critic_window_loss if complete_windows else structured_critic_horizon_loss
        )
        terms = loss_function(
            dynamics,
            belief,
            inputs,
            factors,
            value_head=critic.value_head,
            horizon=config.structured_critic_horizon,
            **({} if complete_windows else {"plan": plan}),
        )
        loss = (
            config.structured_critic_latent_coefficient * terms.latent
            + config.structured_critic_value_coefficient * terms.value
        )
    return loss, terms


def _jepa_factors(
    staged: dict[str, Tensor], indices: Tensor, steps_per_trajectory: int
) -> dict[str, Tensor]:
    """The executed joint action, the run metadata, and the collected reward.

    `episode_index` and `step` are derived from the flat row index rather than
    stored, exactly as the detached auxiliaries derive them: a minibatch of
    contiguous episode runs is what makes a row's successor already present in
    the batch, and these two fields are how the loss proves it.
    """
    return {
        "unit_actions": _batch_tensor(staged["unit_actions"], indices, torch.long),
        "market_kinds": _batch_tensor(staged["market_kinds"], indices, torch.long),
        "market_quantities": _batch_tensor(staged["market_quantities"], indices, torch.long),
        "episode_index": torch.div(indices, steps_per_trajectory, rounding_mode="floor"),
        "step": indices.remainder(steps_per_trajectory),
        "rewards": _batch_tensor(staged["rewards"], indices, torch.float32),
    }


def _jepa_auxiliary_terms(
    actor: EntityActor,
    objective: JepaObjective,
    staged: dict[str, Tensor],
    indices: Tensor,
    *,
    steps_per_trajectory: int,
    config: PpoConfig,
    autocast_enabled: bool,
    model_grad: bool,
    complete_windows: bool,
    belief_indices: Tensor | None = None,
    belief_inverse: Tensor | None = None,
    belief: JepaBelief | None = None,
    plan: StructuredHorizonPlan | None = None,
    sample_weight: Tensor | None = None,
) -> tuple[Tensor, JepaTerms]:
    """The LeJEPA world-model objective, reusing the single PPO forward's belief.

    The belief it receives is attached, and so, unless the actor is the detached
    ablation, is the one the same forward's `ActorOutput` was decoded from. That
    is the whole shape of the family in one minibatch: one encoder pass, two
    consumers, and one backward that brings both gradients into the trunk.

    Structurally the twin of `_structured_auxiliary_terms`, and deliberately so:
    the belief-resolution ladder below is the same one, because the choice of
    where the source encoding comes from -- the live PPO forward, a fresh
    gradient-carrying pass, a deduplicated no-grad pass, or a plain no-grad pass
    -- is a property of the trainer's phase, not of the objective.

    What differs is everything downstream. There is no stop-gradient anywhere in
    the returned loss: the successor's embedding carries gradient by design, and
    SIGReg is the term that makes that safe.
    """
    if complete_windows:
        raise ValueError("the LeJEPA objective has no complete-window variant")
    (inputs,) = _actor_batch_args(LEJEPA, staged, indices)
    if not isinstance(inputs, StructuredInputs):
        raise TypeError("the LeJEPA auxiliary requires StructuredInputs")
    factors = _jepa_factors(staged, indices, steps_per_trajectory)
    with torch.autocast(
        device_type=indices.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        if belief is not None:
            if belief_indices is not None or belief_inverse is not None:
                raise ValueError("a supplied belief cannot also name unique rows")
        elif model_grad:
            if belief_indices is not None or belief_inverse is not None:
                raise ValueError("actor-gradient auxiliary cannot reuse unique-row beliefs")
            belief = actor.auxiliary_belief(inputs)
        elif belief_indices is not None and belief_inverse is not None:
            (belief_inputs,) = _actor_batch_args(LEJEPA, staged, belief_indices)
            if not isinstance(belief_inputs, StructuredInputs):
                raise TypeError("the LeJEPA auxiliary requires StructuredInputs")
            with torch.no_grad():
                unique_belief = actor.auxiliary_belief(belief_inputs)
            belief = JepaBelief(*(value[belief_inverse] for value in unique_belief))
        elif belief_indices is None and belief_inverse is None:
            with torch.no_grad():
                belief = actor.auxiliary_belief(inputs)
        else:
            raise ValueError("belief indices and inverse must be supplied together")
        terms = jepa_horizon_loss(
            objective,
            belief,
            inputs,
            factors,
            horizon=config.jepa_horizon,
            plan=plan,
            sample_weight=sample_weight,
            score_reward=config.jepa_reward_coefficient > 0.0,
        )
        loss = (
            config.jepa_prediction_coefficient * terms.prediction
            + config.jepa_sigreg_coefficient * terms.sigreg
            + config.jepa_reward_coefficient * terms.reward
        )
    return loss, terms


def _credit_quality_metrics(
    rollout: RolloutBatch,
    monte_carlo: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    gamma: float,
    groups: Mapping[str, np.ndarray] | None,
) -> dict[str, float | int]:
    """Preupdate terminal prediction, removing the known shaping potential.

    G_t = gamma**(T-t-1) U_T - Phi_t. Recover Phi only for these
    diagnostics; neither the terminal outcome nor this reconstruction is staged.
    """
    reward_mode = rollout.reward_mode
    remaining = rollout.valid.sum(axis=1)[:, None] - np.arange(valid.shape[1])[None, :]
    if reward_mode == "terminal-outcome":
        # Completed trajectories contain the exact native outcome in their last
        # valid reward; comparing stored f32 banks can turn close wins into ties.
        terminal = rollout.rewards[np.arange(valid.shape[0]), remaining[:, 0] - 1].astype(
            np.float64
        )
    else:
        own = rollout.final_money.astype(np.float64)
        opponent = rollout.opponent_money.astype(np.float64)
        terminal = (own - opponent) / (own + opponent + 2.0 * STARTING_MONEY)
    target = np.power(gamma, np.maximum(remaining - 1, 0)) * terminal[:, None]
    potential = target - monte_carlo if reward_mode == "shaped" else 0.0
    prediction = values + potential
    partitions = {"all": np.ones(valid.shape[0], dtype=bool), **(groups or {})}
    bins = (
        ("all", remaining > 0),
        ("ttg_1_32", (remaining >= 1) & (remaining <= 32)),
        ("ttg_33_128", (remaining > 32) & (remaining <= 128)),
        ("ttg_129_512", (remaining > 128) & (remaining <= 512)),
        ("ttg_513_plus", remaining > 512),
    )
    metrics: dict[str, float | int] = {}
    for group, rows in partitions.items():
        rows = np.asarray(rows, dtype=bool)
        if rows.shape != (valid.shape[0],):
            raise ValueError("credit diagnostic groups must select trajectory rows")
        for horizon, horizon_mask in bins:
            selected = valid & rows[:, None] & horizon_mask
            count = int(selected.sum())
            prefix = f"credit_preupdate_{group}_{horizon}"
            metrics[f"{prefix}_states"] = count
            if not count:
                continue
            residual = target[selected] - prediction[selected]
            variance = float(target[selected].var())
            metrics.update(
                {
                    f"{prefix}_mc_mse": float(
                        np.square(monte_carlo[selected] - values[selected]).mean()
                    ),
                    f"{prefix}_potential_only_mse": float(np.square(target[selected]).mean()),
                    f"{prefix}_terminal_residual_mse": float(np.square(residual).mean()),
                    f"{prefix}_terminal_target_variance": variance,
                    f"{prefix}_terminal_residual_explained_variance": (
                        1.0 - float(residual.var()) / variance if variance > 1e-12 else 0.0
                    ),
                }
            )
    return metrics


def _auxiliary_branch_belief(belief, squared_norms: list[Tensor] | None = None):
    """Fork auxiliary edges, optionally observing their raw source cotangents.

    Always use the same zero-copy views, including unobserved minibatches: changing
    input view/autograd metadata only for diagnostics can specialize AOTAutograd.
    Hooks exclude the primary objective and reduce immediately to one scalar per
    connected source field; actor side context is already detached.
    """

    def observe(gradient: Tensor) -> None:
        assert squared_norms is not None
        squared_norms.append(
            torch.linalg.vector_norm(gradient.detach(), dtype=torch.float32).square()
        )

    fields = []
    for field in belief:
        branch = field.view_as(field)
        if squared_norms is not None and branch.requires_grad:
            branch.register_hook(observe)
        fields.append(branch)
    return type(belief)(*fields)


def _auxiliary_preupdate_metrics(
    prefix: str, loss: Tensor, terms, names: tuple[str, ...]
) -> dict[str, Tensor]:
    """Snapshot compact scalars without retaining a compiled callable's outputs."""
    values = torch.stack((loss.detach(), *(getattr(terms, name).detach() for name in names)))
    keys = (f"{prefix}_combined", *(f"{prefix}_{name}" for name in names))
    return dict(zip(keys, values.unbind(), strict=True))


def _elapsed_cuda_seconds(start: torch.cuda.Event | None, end: torch.cuda.Event | None) -> float:
    if start is None or end is None:
        return 0.0
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def _structured_actor_belief(
    actor: StructuredActor | EntityActor, autocast_enabled: bool, inputs: StructuredInputs
) -> StructuredDecisionBelief | JepaBelief:
    with torch.autocast(
        device_type=inputs.tile_categorical.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        return actor.auxiliary_belief(inputs)


def _zero_world_model_grads(actor: Actor, dynamics: torch.nn.Module | None) -> None:
    """Clear the world model's gradients: the predictor's, and the backbone's.

    `nn.Module.zero_grad` on the objective reaches the projector and the
    predictor and stops there, because under `lejepa` the backbone is not a
    submodule of the objective -- it is a member of its optimizer's parameter
    groups and nothing more. Nothing else would clear it either: the actor's
    optimizer owns the heads alone. Without this the encoder's gradient would
    accumulate across minibatches, epochs and waves while the objective's was
    recomputed each time, so one loss would reach the two halves of one model
    at two different effective step sizes -- and the frozen warm pass, which
    promises to leave the actor bit-identical, would hand its discarded
    backward to the release wave.
    """
    if dynamics is not None:
        dynamics.zero_grad(set_to_none=True)
    for parameter in backbone_parameters(actor):
        parameter.grad = None


def _warm_actor_update_graphs(
    actor: Actor,
    *,
    actor_terms: Callable[..., tuple[Tensor, ...]],
    actor_optimizer: torch.optim.Optimizer,
    structured_dynamics: torch.nn.Module | None,
    structured_terms_fn: Callable[..., Any] | None,
    auxiliary_active: bool,
    architecture: str,
    staged: dict[str, Tensor],
    indices: Tensor,
    plan: StructuredHorizonPlan | None,
    steps_per_trajectory: int,
    config: PpoConfig,
    autocast_enabled: bool,
    sample_weight: Tensor | None = None,
    auxiliary_sample_weight: Tensor | None = None,
    reference_actor: Actor | None = None,
) -> None:
    """Compile the released actor's forward and backward while it is frozen.

    A warmup wave runs `actor_epochs=0`, so the actor's update graphs -- and
    above all their backwards, which Inductor compiles on the first `.backward()`
    and not with the forward -- first trace on the release wave itself: measured
    18.0 s and 6.0 s of compile landing in one 17 s wave. Nothing about them
    depends on the actor being unfrozen, so one minibatch runs here with
    gradients enabled and the flags the released path will use, and the result
    is discarded: no optimizer steps, no schedule advance, no metrics, and both
    gradient buffers cleared on the way out. The actor stays bit-identical.

    Each callable/configuration/shape is warmed once per actor, not once per
    frozen wave. The marker records successful discarded backward execution,
    not the existence of a lazily compiled forward wrapper.

    `auxiliary_sample_weight` is separate from `sample_weight` because only the
    LeJEPA auxiliary takes the row weight -- the detached predictor's callable has
    no such parameter. `None` against a tensor is a Dynamo guard, so warming the
    wrong one of the two would trace a branch the release wave does not use and
    put back exactly the compile this function exists to move.
    """
    auxiliary_weight = (
        {} if auxiliary_sample_weight is None else {"sample_weight": auxiliary_sample_weight}
    )
    key = (
        actor_terms,
        structured_terms_fn,
        config,
        auxiliary_active,
        autocast_enabled,
        auxiliary_sample_weight is None,
        reference_actor is None,
        tuple(indices.shape),
        tuple(staged["advantages"].shape[1:]),
        None if plan is None else (tuple(plan.indices.shape), tuple(plan.eligible.shape)),
    )
    warmed = getattr(actor, "_kaggriculture_warmed_update_graphs", None)
    if warmed is not None and key in warmed:
        return
    _begin_update_graph_step(config.update_compile_mode, indices.device)
    actor_optimizer.zero_grad(set_to_none=True)
    _zero_world_model_grads(actor, structured_dynamics)
    actor_args = _actor_batch_args(architecture, staged, indices)
    actor_pack = actor_terms(
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
        _batch_tensor(staged["advantages"], indices, torch.float32),
        config.clip_low,
        config.clip_high,
        autocast_enabled,
        *actor_args,
        sample_weight=sample_weight,
        policy_ratio_scope=config.policy_ratio_scope,
        policy_objective=config.policy_objective,
        tpo_eta=config.tpo_eta,
        reference_actor=reference_actor,
    )
    if config.policy_loss_reduction == "states":
        policy_denominator = (
            indices.numel() if sample_weight is None else sample_weight.sum().clamp_min(1)
        )
    else:
        policy_denominator = sum(
            (
                _batch_tensor(staged[name], indices, torch.float32)
                * (sample_weight[:, None] if sample_weight is not None else 1.0)
            ).sum()
            for name in ("unit_active", "market_active", "market_quantity_active")
        ).clamp_min(1)
    loss = (-actor_pack[0] + config.reference_kl_coefficient * actor_pack[6]) / policy_denominator
    if structured_terms_fn is not None:
        assert architecture_of_config(actor.config).structured_inputs
        assert structured_dynamics is not None
        belief = architecture_of_config(actor.config).actor_belief_class(*actor_pack[8:])
        auxiliary_loss, _ = structured_terms_fn(
            actor,
            structured_dynamics,
            staged,
            indices,
            steps_per_trajectory=steps_per_trajectory,
            config=config,
            autocast_enabled=autocast_enabled,
            model_grad=auxiliary_active,
            complete_windows=False,
            belief=_auxiliary_branch_belief(
                belief if auxiliary_active else _detached_belief(belief), None
            ),
            plan=plan,
            **auxiliary_weight,
        )
        (loss + auxiliary_loss).backward()
    else:
        loss.backward()
    actor_optimizer.zero_grad(set_to_none=True)
    _zero_world_model_grads(actor, structured_dynamics)
    if warmed is None:
        actor._kaggriculture_warmed_update_graphs = warmed = set()
    warmed.add(key)


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
    structured_dynamics: ActorDynamics | JepaObjective | None = None,
    structured_dynamics_optimizer: torch.optim.Optimizer | None = None,
    structured_actor_auxiliary: bool | None = None,
    structured_critic_dynamics: StructuredCriticDynamics | JepaObjective | None = None,
    structured_critic_dynamics_optimizer: torch.optim.Optimizer | None = None,
    structured_critic_auxiliary: bool | None = None,
    auxiliary_generator: np.random.Generator | None = None,
    diagnostic_groups: Mapping[str, np.ndarray] | None = None,
    diagnostic_gradients: bool = False,
    reference_actor: Actor | None = None,
) -> dict[str, float | int]:
    """Replay one rollout with asymmetric, per-component clipped policy updates.

    `actor_epochs` overrides the actor's participation for this call only —
    used by the warm-start critic-first phase, where a freshly initialized
    critic must fit before its advantages are allowed to push a pretrained
    actor. Zero runs a critic-only refit and does not consume sampler
    likelihoods.

    `rows` restricts the update to those trajectory rows -- a population's
    per-member partition. Every statistic below is then that subset's own. The
    partition is a row index rather than a sliced rollout because a game's two
    rows belong to two different members, so no storage order makes one
    member's rows a contiguous block and slicing would copy the wave's state
    arrays.

    `reference_actor` is the frozen policy `config.reference_kl_coefficient`
    anchors to; the two are required together.
    """
    _validate_config(config)
    _validate_reference_actor(actor, reference_actor, config)
    if isinstance(actor, (StrategicActor, CausalActor)):
        if config.policy_ratio_scope != "joint":
            raise ValueError("strategic and causal PPO require joint policy ratios")
        if config.structured_actor_auxiliary_active or structured_dynamics is not None:
            raise ValueError("strategic and causal PPO require actor NextLat disabled")
    entity_critic = isinstance(critic, StructuredCritic) and getattr(
        critic.config, "per_entity_critic", False
    )
    forecast_critic = isinstance(critic, EntityCritic) and critic.forecast_heads is not None
    if forecast_critic and (
        config.structured_critic_auxiliary_active or structured_critic_dynamics is not None
    ):
        raise ValueError("forecast critic replaces critic NextLat; disable critic NextLat")
    if forecast_critic and config.economic_forecast_coefficient <= 0:
        raise ValueError("forecast critic requires a positive economic forecast coefficient")
    if entity_critic:
        if config.actor_gae_lambda != 1.0 or config.critic_gae_lambda != 1.0:
            raise ValueError("per-entity critic requires actor and critic GAE lambda one")
        if config.policy_ratio_scope != "components":
            raise ValueError("per-entity critic requires component policy ratio scope")
    if not rollout.learner_stochastic:
        raise ValueError("PPO updates require stochastic learner collection")
    _validate_structured_auxiliary_modules(actor, structured_dynamics, config)
    _validate_structured_critic_auxiliary_modules(critic, structured_critic_dynamics, config)
    actor_predictor_active = structured_dynamics is not None
    critic_predictor_active = structured_critic_dynamics is not None
    if actor_predictor_active != (structured_dynamics_optimizer is not None):
        raise ValueError(
            "active structured actor auxiliary requires exactly one predictor optimizer"
        )
    if critic_predictor_active != (structured_critic_dynamics_optimizer is not None):
        raise ValueError(
            "active structured critic auxiliary requires exactly one predictor optimizer"
        )
    predictor_active = actor_predictor_active or critic_predictor_active
    # One dispatch for the whole update: which self-predictive objective each arm
    # runs, what its journal columns are called, and which matched baseline the
    # preupdate diagnostic compares it against. Resolved from the module that was
    # handed in rather than from the config, because the module is what the
    # optimizer owns and what `_validate_structured_*_auxiliary_modules` has just
    # proved consistent with both the coefficients and the model family.
    jepa_actor_arm = isinstance(structured_dynamics, JepaObjective)
    actor_auxiliary_metrics = JEPA_METRICS if jepa_actor_arm else _STRUCTURED_AUXILIARY_METRICS
    critic_auxiliary_metrics = _STRUCTURED_CRITIC_AUXILIARY_METRICS
    actor_auxiliary_function = (
        _jepa_auxiliary_terms if jepa_actor_arm else _structured_auxiliary_terms
    )
    critic_auxiliary_function = _structured_critic_auxiliary_terms
    # The matched baselines, journaled under `<prefix>_<label>`. The LeJEPA arms
    # carry a second one: persistence alone cannot separate "the predictor is
    # idle" from "the encoder went constant along the trajectory", because the
    # second sends the baseline to zero along with the loss. The shuffled control
    # scores the live predictor against a neighbouring row's action, so a
    # transition that is not action-conditioned shows up as the two agreeing.
    # It rolls the transition sources by the run length because each contiguous
    # same-trajectory run contributes `run_length - horizon` consecutive ones:
    # rolling by one would mostly hand a source its own trajectory's
    # neighbouring action, which is not a control.
    control_stride = _structured_run_length(config)
    actor_controls = (
        (
            ("persistence", JepaPersistenceControl(structured_dynamics)),
            ("shuffled", JepaShuffledControl(structured_dynamics, control_stride)),
        )
        if jepa_actor_arm
        else (("persistence", PersistenceDynamics()),)
    )
    critic_controls = (("persistence", PersistenceDynamics()),)
    actor_belief_class = architecture_of_config(actor.config).actor_belief_class
    critic_belief_class = architecture_of_config(critic.config).critic_belief_class
    if predictor_active != (auxiliary_generator is not None):
        raise ValueError(
            "active structured auxiliary requires exactly one independent auxiliary generator"
        )
    if structured_actor_auxiliary is None:
        structured_actor_auxiliary = actor_predictor_active
    if structured_critic_auxiliary is None:
        structured_critic_auxiliary = critic_predictor_active
    if structured_actor_auxiliary and not actor_predictor_active:
        raise ValueError("actor-side structured auxiliary requires an active actor predictor")
    if structured_critic_auxiliary and not critic_predictor_active:
        raise ValueError("critic-side structured auxiliary requires an active critic predictor")
    # The actor is claimed by its parameter partition rather than as a module:
    # under `lejepa` the shared backbone lives inside it but is owned by the
    # world model's optimizer below. For every other family the partition is the
    # whole module and this reads exactly as it did.
    actor_owned = policy_owned_parameters(actor)
    world_model = backbone_parameters(actor)
    ownership_pairs: list[tuple[str, Any, torch.optim.Optimizer]] = [
        ("actor", actor_owned, actor_optimizer),
        ("critic", critic, critic_optimizer),
    ]
    if structured_dynamics is not None and structured_dynamics_optimizer is not None:
        ownership_pairs.append(
            (
                "actor predictor",
                list(structured_dynamics.parameters()) + world_model,
                structured_dynamics_optimizer,
            )
        )
    elif world_model:
        raise ValueError("the lejepa backbone has no optimizer; its objective is what trains it")
    if structured_critic_dynamics is not None and structured_critic_dynamics_optimizer is not None:
        ownership_pairs.append(
            (
                "critic predictor",
                structured_critic_dynamics,
                structured_critic_dynamics_optimizer,
            )
        )
    _validate_optimizer_ownership(tuple(ownership_pairs))
    if actor_epochs is None:
        actor_epochs = config.epochs
    elif not 0 <= actor_epochs <= config.epochs:
        raise ValueError("actor epoch override must lie within the configured epochs")
    device = next(actor.parameters()).device
    if next(critic.parameters()).device != device:
        raise ValueError("actor and critic must use the same device")
    phase_started = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    def mark_phase(name: str) -> None:
        # Each boundary below already synchronizes with the device (a `.cpu()`
        # readback or an explicit synchronize), so these are wall-clock phase
        # attributions rather than launch-only measurements.
        nonlocal phase_started
        now = time.perf_counter()
        phase_seconds[name] = now - phase_started
        phase_started = now

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

    if rollout.architecture == "strategic-plan":
        flat_component_counts += rollout.states["plan_active"].reshape(-1).astype(np.int64)

    # The complete rollout is reused for several PPO epochs. Stage every array
    # on the accelerator once; repeated NumPy advanced indexing otherwise makes
    # a new host copy and host-to-device transfer for every field/minibatch.
    architecture = rollout.architecture
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
            # The LeJEPA reward target. Every other consumer of the rewards works
            # on the host arena during advantage estimation; the world model needs
            # them per staged row on the device like any other minibatch field.
            "rewards",
        )
    }
    # Stored categorical support is validated in one staged pass; repeated
    # NumPy sweeps over the multi-gigabyte host rollout would stall the update.
    _validate_staged_action_masks(staged, torch.from_numpy(flat_valid).to(device))
    if forecast_critic:
        forecast_targets = build_economic_forecast_targets(
            rollout.states | {"unit_active": rollout.unit_active}, owned_valid
        )
        staged["economic_features"] = _stage_tensor(forecast_targets.features, device)
        staged["economic_future_indices"] = _stage_tensor(forecast_targets.future_indices, device)
        staged["economic_forecast_valid"] = _stage_tensor(forecast_targets.valid, device)
        del forecast_targets
    mark_phase("update_staging_seconds")
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
    # The update forward uses the fused MLP's stateless BF16 path, but the
    # following optimizer steps must leave both its BF16 and FP8 projections
    # ready for the next forward/collection phase. Bootstrap missing FP8 state
    # once, then refresh it from committed master weights after every step.
    refresh_fused_mlp_fp8(actor)
    critic.train()
    refresh_fused_mlp_fp8(critic)
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    predictor_metrics: dict[str, Tensor] = {}
    gradient_metrics: dict[str, Tensor] = {}
    structured_terms_fn: Any = None
    structured_critic_terms_fn: Any = None
    if actor_predictor_active:
        assert architecture_of_config(actor.config).structured_inputs
        assert structured_dynamics is not None
        assert structured_dynamics_optimizer is not None
        structured_terms_fn = _cached_update_callable(
            actor,
            "_kaggriculture_jepa_terms" if jepa_actor_arm else "_kaggriculture_aux_terms",
            actor_auxiliary_function,
            compile_mode,
        )
        _zero_world_model_grads(actor, structured_dynamics)
    if critic_predictor_active:
        assert architecture_of_config(critic.config).structured_inputs
        assert structured_critic_dynamics is not None
        assert structured_critic_dynamics_optimizer is not None
        structured_critic_terms_fn = _cached_update_callable(
            critic,
            "_kaggriculture_critic_aux_terms",
            critic_auxiliary_function,
            compile_mode,
        )
        structured_critic_dynamics.zero_grad(set_to_none=True)
    behavior_values = _owned_behavior_values(
        critic,
        architecture,
        staged,
        rollout,
        rows,
        compile_mode=compile_mode,
        autocast_enabled=autocast_enabled,
        include_entities=entity_critic,
    )
    # A carried wave hands back the collector's own array; any replay is new.
    behavior_values_carried = behavior_values is rollout.behavior_values
    entity_values = behavior_values[..., 1:] if entity_critic else None
    if entity_critic:
        behavior_values = behavior_values[..., 0]
    mark_phase("update_behavior_replay_seconds")
    if structured_dynamics is not None:
        structured_dynamics.train()
        refresh_fused_mlp_fp8(structured_dynamics)
    if structured_critic_dynamics is not None:
        structured_critic_dynamics.train()
        refresh_fused_mlp_fp8(structured_critic_dynamics)
    prepared = prepare_advantages(rollout, behavior_values, config, rows=rows)
    valid_value_targets = prepared.value_targets[owned_valid]
    if not np.isfinite(valid_value_targets).all():
        raise ValueError("value targets must be finite")
    if critic.config.scalar_value:
        # No support, so nothing to clip and nothing to escape: the reference's
        # scalar critic regresses on the return as it stands. The saturation
        # reading stays in the telemetry layout and reads zero.
        saturated = 0
        value_targets = prepared.value_targets
    else:
        value_support = critic.support.detach().float().cpu().numpy()
        support_widths = np.diff(value_support)
        support_width = (value_support[-1] - value_support[0]) / (value_support.size - 1)
        rounding = 2 * np.finfo(value_support.dtype).eps * np.abs(value_support).max()
        if (
            not np.isfinite(value_support).all()
            or not (support_widths > 0).all()
            or not (np.abs(support_widths - support_width) <= rounding).all()
        ):
            raise ValueError("critic value support must be finite, increasing, and evenly spaced")
        # The critic target is the decoupled lambda-one return. A bounded
        # categorical mean still cannot represent a target outside its atoms, so
        # saturation is measured rather than fatal: it is error escaping the
        # support, and the fraction over time is the signal.
        saturated = np.count_nonzero(
            (valid_value_targets < value_support[0]) | (valid_value_targets > value_support[-1])
        )
        value_targets = np.clip(prepared.value_targets, value_support[0], value_support[-1])
    entity_metrics: dict[str, float] = {}
    if entity_values is not None:
        entity_active = np.concatenate((rollout.unit_active, rollout.market_active), axis=-1)
        entity_active = entity_active & owned_valid[..., None]
        entity_advantages, entity_mean, entity_std = prepare_entity_advantages(
            prepared.monte_carlo_returns, entity_values, entity_active
        )
        staged["advantages"] = _stage_tensor(entity_advantages, device)
        entity_metrics = {
            "entity_advantage_mean": entity_mean,
            "entity_advantage_std": entity_std,
            "entity_value_std": float(entity_values[entity_active].std()),
        }
    else:
        staged["advantages"] = torch.from_numpy(prepared.advantages.reshape(-1)).to(device)
    staged["value_targets"] = torch.from_numpy(value_targets.reshape(-1)).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mark_phase("update_advantage_seconds")
    totals = {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "component_kl",
            "joint_kl",
            "reference_kl",
            "clip_fraction",
            "actor_gradient_norm",
            "critic_gradient_norm",
            "critic_trunk_gradient_norm",
            "critic_head_gradient_norm",
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
    first_minibatch_component_kl = 0.0
    first_minibatch_joint_kl = 0.0
    max_component_kl = 0.0
    max_joint_kl = 0.0
    max_reference_decision_kl = torch.zeros((), device=device, dtype=torch.float32)
    actor_auxiliary_active = bool(
        actor_predictor_active and structured_actor_auxiliary and actor_epochs
    )
    critic_auxiliary_active = bool(critic_predictor_active and structured_critic_auxiliary)
    actor_auxiliary_totals = (
        {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in actor_auxiliary_metrics
        }
        if actor_predictor_active
        else {}
    )
    critic_auxiliary_totals = (
        {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in critic_auxiliary_metrics
        }
        if critic_predictor_active
        else {}
    )
    actor_auxiliary_updates = 0
    actor_auxiliary_seconds = 0.0
    actor_predictor_updates = 0
    actor_predictor_states = 0
    actor_auxiliary_loss_total = (
        torch.zeros((), device=device, dtype=torch.float64) if actor_predictor_active else None
    )
    actor_combined_loss_total = (
        torch.zeros((), device=device, dtype=torch.float64) if actor_predictor_active else None
    )
    actor_combined_gradient_norm_total = (
        torch.zeros((), device=device, dtype=torch.float64) if actor_predictor_active else None
    )
    critic_auxiliary_updates = 0
    critic_auxiliary_seconds = 0.0
    critic_predictor_updates = 0
    critic_auxiliary_loss_total = (
        torch.zeros((), device=device, dtype=torch.float64) if critic_predictor_active else None
    )
    critic_combined_loss_total = (
        torch.zeros((), device=device, dtype=torch.float64) if critic_predictor_active else None
    )
    critic_combined_gradient_norm_total = (
        torch.zeros((), device=device, dtype=torch.float64) if critic_predictor_active else None
    )
    critic_auxiliary_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = (
        [] if critic_predictor_active and device.type == "cuda" else None
    )
    actor_terms = _cached_update_callable(
        actor,
        (
            "_kaggriculture_structured_update_terms"
            if actor_predictor_active
            else "_kaggriculture_update_terms"
        ),
        _structured_actor_minibatch_terms if actor_predictor_active else _actor_minibatch_terms,
        compile_mode,
    )
    critic_objective_fn = _cached_update_callable(
        critic,
        "_kaggriculture_update_objective",
        _critic_minibatch_objective,
        compile_mode,
    )
    critic_fit_terms_fn = _cached_update_callable(
        critic,
        (
            "_kaggriculture_structured_update_fit_terms"
            if critic_predictor_active
            else "_kaggriculture_update_fit_terms"
        ),
        (
            _structured_critic_minibatch_fit_terms
            if critic_predictor_active
            else _critic_minibatch_fit_terms
        ),
        compile_mode,
    )
    forecast_fit_terms_fn = (
        _cached_update_callable(
            critic,
            "_kaggriculture_forecast_update_fit_terms",
            _forecast_critic_minibatch_fit_terms,
            compile_mode,
        )
        if forecast_critic
        else None
    )
    forecast_loss_total = torch.zeros((), device=device, dtype=torch.float64)
    forecast_persistence_total = torch.zeros_like(forecast_loss_total)
    stop_for_kl = False
    critic_head_parameters = tuple(critic.value_head.parameters())
    critic_head_parameter_ids = {id(parameter) for parameter in critic_head_parameters}
    critic_trunk_parameters = tuple(
        parameter
        for parameter in critic.parameters()
        if id(parameter) not in critic_head_parameter_ids
    )
    # Share one activation allocation pool on the caller's CUDA stream.
    # Separate actor/critic streams strand each branch's cached blocks; at
    # production shape their combined reservations force eviction every batch.
    # Ordered execution lets the critic reuse the actor's released activations.
    guard_host = torch.empty(
        6 if actor_predictor_active else 5,
        dtype=torch.float64,
        pin_memory=device.type == "cuda",
    )
    guard_event = torch.cuda.Event() if device.type == "cuda" else None
    actor_step_is_gateable = getattr(actor_optimizer, "supports_found_inf", False) or any(
        group.get("fused", False) for group in actor_optimizer.param_groups
    )
    critic_step_is_gateable = getattr(critic_optimizer, "supports_found_inf", False) or any(
        group.get("fused", False) for group in critic_optimizer.param_groups
    )
    critic_nonfinite = torch.zeros((), device=device, dtype=torch.float32)
    critic_skipped = torch.zeros_like(critic_nonfinite)
    critic_schedule_state = _optimizer_schedule_state(critic_optimizer)
    critic_predictor_schedule_state = (
        _optimizer_schedule_state(structured_critic_dynamics_optimizer)
        if structured_critic_dynamics_optimizer is not None
        else None
    )

    critic_epochs = config.epochs if config.critic_epochs is None else config.critic_epochs
    epoch_marks: list[tuple[Tensor, int]] = []
    minibatch_positions, minibatch_counts = _fixed_minibatch_positions(
        valid_indices.size, config.minibatch_size
    )
    actor_minibatches_intended = actor_epochs * minibatch_positions.shape[0]
    actor_zero = torch.zeros((), device=device, dtype=torch.float32)
    # Epoch-invariant: the partition is fixed for the wave, only the permutation
    # it indexes changes, so this transfer happens once rather than per epoch.
    staged_positions = torch.from_numpy(minibatch_positions).to(device=device)
    sample_weights = torch.from_numpy(
        (np.arange(config.minibatch_size)[None, :] < minibatch_counts[:, None]).astype(np.float32)
    ).to(device=device)
    for epoch_index in range(critic_epochs):
        # NextLat shuffles contiguous runs; plain PPO shuffles individual states.
        # Both orders and their successor plans belong to this epoch only.
        if predictor_active:
            assert auxiliary_generator is not None
            shuffled = _contiguous_run_indices(
                valid_indices,
                rollout.valid.shape[1],
                _structured_run_length(config),
                auxiliary_generator,
            )
        else:
            shuffled = generator.permutation(valid_indices)
        horizon_plans: list[StructuredHorizonPlan] = []
        if predictor_active:
            for batch_positions, count in zip(minibatch_positions, minibatch_counts, strict=True):
                episodes, steps = np.divmod(shuffled[batch_positions], rollout.valid.shape[1])
                # Wrapped rows are neither sources nor successors. Sentinels
                # preserve the full graph shape without creating duplicate
                # windows or changing the weight of any genuine target.
                episodes[count:] = -1
                steps[count:] = -1
                host_plan = structured_horizon_plan(
                    episodes, steps, _structured_run_length(config) - 1
                )
                horizon_plans.append(
                    StructuredHorizonPlan(
                        *(
                            value.pin_memory().to(device=device, non_blocking=True)
                            if device.type == "cuda"
                            else value
                            for value in host_plan
                        )
                    )
                )
        shuffled_device = torch.from_numpy(shuffled).to(device=device)
        if actor_epochs == 0 and epoch_index == 0:
            _warm_actor_update_graphs(
                actor,
                actor_terms=actor_terms,
                actor_optimizer=actor_optimizer,
                structured_dynamics=structured_dynamics,
                structured_terms_fn=structured_terms_fn if actor_predictor_active else None,
                auxiliary_active=bool(actor_predictor_active and structured_actor_auxiliary),
                architecture=architecture,
                staged=staged,
                indices=shuffled_device[staged_positions[0]],
                plan=horizon_plans[0] if horizon_plans else None,
                steps_per_trajectory=rollout.valid.shape[1],
                config=config,
                autocast_enabled=autocast_enabled,
                sample_weight=sample_weights[0],
                auxiliary_sample_weight=sample_weights[0] if jepa_actor_arm else None,
                reference_actor=reference_actor,
            )

        for batch_number in range(minibatch_positions.shape[0]):
            # Actor gradients and metrics remain live through critic backward
            # and both optimizer steps, not merely one compiled invocation.
            _begin_update_graph_step(compile_mode, device)
            host_indices = shuffled[minibatch_positions[batch_number]]
            indices = shuffled_device[staged_positions[batch_number]]
            horizon_plan = horizon_plans[batch_number] if horizon_plans else None
            # Redraw the LeJEPA slice directions and tile sample for this
            # minibatch. Buffers are overwritten in place, so the compiled update
            # graph sees the same tensors at the same shapes and nothing
            # recompiles; only the values move. Drawing here rather than inside
            # the step is what keeps the compiled region free of RNG, and drawing
            # from the trainer's auxiliary stream is what keeps a wave replayable.
            if jepa_actor_arm:
                assert structured_dynamics is not None and auxiliary_generator is not None
                structured_dynamics.refresh_slices(auxiliary_generator)
            run_actor = epoch_index < actor_epochs and not stop_for_kl
            actor_args = (
                _actor_batch_args(architecture, staged, indices)
                if run_actor or actor_predictor_active
                else None
            )
            critic_args = _critic_batch_args(architecture, staged, indices, actor_args=actor_args)
            value_targets = _batch_tensor(staged["value_targets"], indices, torch.float32)
            states = int(minibatch_counts[batch_number])
            sample_weight = sample_weights[batch_number]
            # `_fixed_minibatch_positions` wraps the epoch's last minibatch back
            # to the head of the ordering; the LeJEPA objective scores SIGReg and
            # the reward over every staged row, not only the plan's sources, so it
            # is the one auxiliary that has to be told which rows are duplicates.
            actor_auxiliary_weight = {"sample_weight": sample_weight} if jepa_actor_arm else {}
            component_count = 0
            batch_kl = actor_zero
            batch_joint_kl = actor_zero
            policy_loss = actor_zero
            entropy_mean = actor_zero
            clipped_sum = actor_zero
            actor_gradient_norm = actor_zero
            combined_actor_loss = actor_zero
            actor_auxiliary_loss = actor_zero
            actor_auxiliary_terms: ActorDynamicsTerms | JepaTerms | None = None
            actor_auxiliary_started = 0.0
            actor_auxiliary_start_event: torch.cuda.Event | None = None
            actor_auxiliary_end_event: torch.cuda.Event | None = None
            actor_skip: Tensor | None = None
            if run_actor:
                assert actor_args is not None
                # Component activity is immutable rollout metadata. Reducing it
                # on the host avoids a CUDA synchronization in every minibatch
                # merely to recover a denominator already known before staging.
                component_count = int(flat_component_counts[host_indices[:states]].sum())
                component_denominator = max(1, component_count)
                policy_denominator = (
                    states if config.policy_loss_reduction == "states" else component_denominator
                )
                diagnostic_count = (
                    states if config.policy_ratio_scope == "joint" else component_count
                )
                diagnostic_denominator = max(1, diagnostic_count)
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
                _zero_world_model_grads(actor, structured_dynamics)
                actor_gradient_diagnostic = (
                    diagnostic_gradients and actor_predictor_active and not actor_predictor_updates
                )
                actor_pack = actor_terms(
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
                    sample_weight=sample_weight,
                    policy_ratio_scope=config.policy_ratio_scope,
                    policy_objective=config.policy_objective,
                    tpo_eta=config.tpo_eta,
                    reference_actor=reference_actor,
                )
                (
                    policy_sum,
                    entropy_sum,
                    kl_sum,
                    clipped_sum,
                    component_kl_sum,
                    joint_kl_sum,
                    reference_kl_sum,
                    reference_decision_kl,
                ) = actor_pack[:8]
                batch_kl = kl_sum.detach().double() / diagnostic_denominator
                batch_joint_kl = joint_kl_sum.detach().double() / max(1, states)
                batch_component_kl = (
                    batch_kl
                    if config.policy_ratio_scope == "components"
                    else component_kl_sum.detach().double() / component_denominator
                )
                policy_loss = (
                    -policy_sum + config.reference_kl_coefficient * reference_kl_sum
                ) / policy_denominator
                entropy_mean = entropy_sum / component_denominator
                if actor_predictor_active:
                    assert architecture_of_config(actor.config).structured_inputs
                    assert structured_dynamics is not None
                    assert structured_dynamics_optimizer is not None
                    assert structured_terms_fn is not None
                    actor_belief = actor_belief_class(*actor_pack[8:])
                    source_belief = (
                        actor_belief if actor_auxiliary_active else _detached_belief(actor_belief)
                    )
                    if device.type == "cuda":
                        actor_auxiliary_start_event = torch.cuda.Event(enable_timing=True)
                        actor_auxiliary_end_event = torch.cuda.Event(enable_timing=True)
                        actor_auxiliary_start_event.record()
                    else:
                        actor_auxiliary_started = time.perf_counter()
                    actor_source_squares: list[Tensor] = []
                    source_belief = _auxiliary_branch_belief(
                        source_belief,
                        actor_source_squares if actor_gradient_diagnostic else None,
                    )
                    actor_auxiliary_loss, actor_auxiliary_terms = structured_terms_fn(
                        actor,
                        structured_dynamics,
                        staged,
                        indices,
                        steps_per_trajectory=rollout.valid.shape[1],
                        config=config,
                        autocast_enabled=autocast_enabled,
                        model_grad=actor_auxiliary_active,
                        complete_windows=False,
                        belief=source_belief,
                        plan=horizon_plan,
                        **actor_auxiliary_weight,
                    )
                    if "structured_preupdate_combined" not in predictor_metrics:
                        predictor_metrics.update(
                            _auxiliary_preupdate_metrics(
                                "structured_preupdate",
                                actor_auxiliary_loss,
                                actor_auxiliary_terms,
                                actor_auxiliary_metrics,
                            )
                        )
                        # Fresh-wave controls are diagnostic only and
                        # never control source-gradient admission.
                        with torch.no_grad():
                            for label, control in actor_controls:
                                control_loss, control_terms = actor_auxiliary_function(
                                    actor,
                                    control,
                                    staged,
                                    indices,
                                    steps_per_trajectory=rollout.valid.shape[1],
                                    config=config,
                                    autocast_enabled=autocast_enabled,
                                    model_grad=False,
                                    complete_windows=False,
                                    belief=_detached_belief(actor_belief),
                                    plan=horizon_plan,
                                    **actor_auxiliary_weight,
                                )
                                predictor_metrics.update(
                                    _auxiliary_preupdate_metrics(
                                        f"structured_preupdate_{label}",
                                        control_loss,
                                        control_terms,
                                        actor_auxiliary_metrics,
                                    )
                                )
                    combined_actor_loss = policy_loss + actor_auxiliary_loss
                    combined_actor_loss.backward()
                    if actor_gradient_diagnostic:
                        gradient_metrics["structured_gradient_source_norm"] = (
                            torch.stack(actor_source_squares).sum().sqrt()
                            if actor_source_squares
                            else actor_zero
                        )
                    # The policy's own parameters, which under `lejepa` are the
                    # actor minus the backbone: whatever reaches the backbone is
                    # the world model's optimizer to measure, clip and gate on.
                    actor_gradient_norm = torch.nn.utils.get_total_norm(
                        [parameter.grad for parameter in actor_owned if parameter.grad is not None]
                    ).detach()
                    # The backbone clips with the objective that owns it: one
                    # summed gradient, the objective's and the policy's, and one
                    # norm to bound it by.
                    torch.nn.utils.clip_grad_norm_(
                        list(structured_dynamics.parameters()) + world_model,
                        config.nextlat_max_gradient_norm,
                    )
                    if actor_auxiliary_end_event is not None:
                        actor_auxiliary_end_event.record()
                    else:
                        actor_auxiliary_seconds += time.perf_counter() - actor_auxiliary_started
                else:
                    combined_actor_loss = policy_loss
                    combined_actor_loss.backward()
                    # The policy's own parameters, which under `lejepa` are the
                    # actor minus the backbone: whatever reaches the backbone is
                    # the world model's optimizer to measure, clip and gate on.
                    actor_gradient_norm = torch.nn.utils.get_total_norm(
                        [parameter.grad for parameter in actor_owned if parameter.grad is not None]
                    ).detach()
                guard_values = [
                    batch_kl,
                    policy_loss.detach().double(),
                    actor_gradient_norm.double(),
                    batch_joint_kl,
                    batch_component_kl,
                ]
                if actor_predictor_active:
                    guard_values.insert(2, combined_actor_loss.detach().double())
                guard_tensor = torch.stack(guard_values)
                actor_finite = torch.isfinite(guard_tensor[:-1]).all()
                actor_skip = (~actor_finite).float() if actor_step_is_gateable else None
                guard_host.copy_(guard_tensor, non_blocking=True)
                if guard_event is not None:
                    guard_event.record()
            elif actor_predictor_active:
                assert architecture_of_config(actor.config).structured_inputs
                assert structured_dynamics is not None
                assert structured_dynamics_optimizer is not None
                assert structured_terms_fn is not None
                assert actor_args is not None
                (inputs,) = actor_args
                if not isinstance(inputs, StructuredInputs):
                    raise TypeError("structured actor minibatches require StructuredInputs")
                _zero_world_model_grads(actor, structured_dynamics)
                # The policy is frozen here -- a warmup wave, a KL stop, or the
                # critic's extra epochs -- so the encoder is frozen with it and
                # only the predictor fits. Under `lejepa` that holds although
                # the world model owns the encoder: the heads read it, so every
                # encoder step is a policy step, and one taken past the stop is
                # a policy change the trust region has already refused. Warmup
                # waves are where a warm start's fresh projector and predictor
                # fit against the cloned backbone before either moves it.
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ),
                ):
                    actor_belief = _cached_update_callable(
                        actor,
                        "_kaggriculture_frozen_decision_belief",
                        _structured_actor_belief,
                        compile_mode,
                    )(actor, autocast_enabled, inputs)
                actor_belief = _detached_belief(actor_belief)
                if device.type == "cuda":
                    actor_auxiliary_start_event = torch.cuda.Event(enable_timing=True)
                    actor_auxiliary_end_event = torch.cuda.Event(enable_timing=True)
                    actor_auxiliary_start_event.record()
                else:
                    actor_auxiliary_started = time.perf_counter()
                actor_auxiliary_loss, actor_auxiliary_terms = structured_terms_fn(
                    actor,
                    structured_dynamics,
                    staged,
                    indices,
                    steps_per_trajectory=rollout.valid.shape[1],
                    config=config,
                    autocast_enabled=autocast_enabled,
                    model_grad=False,
                    complete_windows=False,
                    belief=actor_belief,
                    plan=horizon_plan,
                    **actor_auxiliary_weight,
                )
                actor_auxiliary_loss.backward()
                if "structured_preupdate_combined" not in predictor_metrics:
                    predictor_metrics.update(
                        _auxiliary_preupdate_metrics(
                            "structured_preupdate",
                            actor_auxiliary_loss,
                            actor_auxiliary_terms,
                            actor_auxiliary_metrics,
                        )
                    )
                    with torch.no_grad():
                        for label, control in actor_controls:
                            control_loss, control_terms = actor_auxiliary_function(
                                actor,
                                control,
                                staged,
                                indices,
                                steps_per_trajectory=rollout.valid.shape[1],
                                config=config,
                                autocast_enabled=autocast_enabled,
                                model_grad=False,
                                complete_windows=False,
                                belief=actor_belief,
                                plan=horizon_plan,
                                **actor_auxiliary_weight,
                            )
                            predictor_metrics.update(
                                _auxiliary_preupdate_metrics(
                                    f"structured_preupdate_{label}",
                                    control_loss,
                                    control_terms,
                                    actor_auxiliary_metrics,
                                )
                            )
                # The backbone clips with the objective that owns it, on this
                # branch as on the other: one summed gradient and one norm to
                # bound it by, whichever branch of the trainer took the step.
                torch.nn.utils.clip_grad_norm_(
                    list(structured_dynamics.parameters()) + world_model,
                    config.nextlat_max_gradient_norm,
                )
                if actor_auxiliary_end_event is not None:
                    actor_auxiliary_end_event.record()
                else:
                    actor_auxiliary_seconds += time.perf_counter() - actor_auxiliary_started

            if jepa_actor_arm:
                # Both branches above encoded this minibatch through the shared
                # backbone, and no optimizer has stepped since. The `lejepa`
                # critic reads that very module on these very public inputs,
                # so hand it the encoding rather than a second trunk forward
                # at identical weights. Detached, as the critic would have it;
                # the tensors are already retained until the minibatch ends.
                critic_args = (*critic_args, _detached_belief(actor_belief))

            # Target KL constrains only the actor. Keep fitting the critic for
            # every configured epoch after policy updates stop.
            critic_auxiliary_loss = actor_zero
            combined_critic_loss = actor_zero
            critic_auxiliary_terms: StructuredCriticDynamicsTerms | None = None
            critic_auxiliary_start_event: torch.cuda.Event | None = None
            critic_auxiliary_end_event: torch.cuda.Event | None = None
            critic_auxiliary_started = 0.0
            critic_optimizer.zero_grad(set_to_none=True)
            if structured_critic_dynamics is not None:
                structured_critic_dynamics.zero_grad(set_to_none=True)
            critic_belief: StructuredCriticBelief | None = None
            critic_gradient_diagnostic = (
                diagnostic_gradients and critic_predictor_active and not critic_predictor_updates
            )
            minibatch_entity_active = (
                _entity_active_batch(staged, indices) if entity_critic else None
            )
            forecast_loss = actor_zero
            if forecast_critic:
                assert forecast_fit_terms_fn is not None
                future_indices = staged["economic_future_indices"][indices]
                economic_features = staged["economic_features"]
                forecast_labels = (
                    economic_features[future_indices].float()
                    - economic_features[indices, None].float()
                )
                value_loss, fit_moments, forecast_loss, persistence_loss = forecast_fit_terms_fn(
                    critic,
                    value_targets,
                    autocast_enabled,
                    *critic_args,
                    sample_weight=sample_weight,
                    forecast_targets=forecast_labels,
                    forecast_valid=staged["economic_forecast_valid"][indices],
                    # Critic-only fitting continues after actor KL stops; these
                    # current actions must be gathered independently of it.
                    unit_actions=_batch_tensor(staged["unit_actions"], indices, torch.long),
                    market_kinds=_batch_tensor(staged["market_kinds"], indices, torch.long),
                    market_quantities=_batch_tensor(
                        staged["market_quantities"], indices, torch.long
                    ),
                    market_active=_batch_tensor(staged["market_active"], indices, torch.bool),
                    market_quantity_active=_batch_tensor(
                        staged["market_quantity_active"], indices, torch.bool
                    ),
                )
                forecast_loss_total += forecast_loss.detach().double() * states
                forecast_persistence_total += persistence_loss.detach().double() * states
                if epoch_index == 0:
                    first_fit_sums += fit_moments
                    first_fit_states += states
                if epoch_index == critic_epochs - 1:
                    last_fit_sums += fit_moments
                    last_fit_states += states
            elif critic_predictor_active or epoch_index == 0 or epoch_index == critic_epochs - 1:
                critic_pack = critic_fit_terms_fn(
                    critic,
                    value_targets,
                    autocast_enabled,
                    *critic_args,
                    sample_weight=sample_weight,
                    entity_active=minibatch_entity_active,
                    **(
                        {
                            "balance_auxiliary": critic_auxiliary_active
                            and config.structured_critic_gradient_balance
                        }
                        if critic_predictor_active
                        else {}
                    ),
                )
                if critic_predictor_active:
                    value_loss, fit_moments = critic_pack[0], critic_pack[1]
                    critic_belief = critic_belief_class(*critic_pack[2:])
                else:
                    value_loss, fit_moments = critic_pack
                if epoch_index == 0:
                    first_fit_sums += fit_moments
                    first_fit_states += states
                if epoch_index == critic_epochs - 1:
                    last_fit_sums += fit_moments
                    last_fit_states += states
            else:
                value_loss = critic_objective_fn(
                    critic,
                    value_targets,
                    autocast_enabled,
                    *critic_args,
                    sample_weight=sample_weight,
                    entity_active=minibatch_entity_active,
                )
            if critic_predictor_active:
                assert architecture_of_config(critic.config).structured_inputs
                assert structured_critic_dynamics is not None
                assert structured_critic_dynamics_optimizer is not None
                assert structured_critic_terms_fn is not None
                assert critic_belief is not None
                source_belief = (
                    critic_belief if critic_auxiliary_active else _detached_belief(critic_belief)
                )
                if device.type == "cuda":
                    critic_auxiliary_start_event = torch.cuda.Event(enable_timing=True)
                    critic_auxiliary_end_event = torch.cuda.Event(enable_timing=True)
                    critic_auxiliary_start_event.record()
                else:
                    critic_auxiliary_started = time.perf_counter()
                critic_source_squares: list[Tensor] = []
                source_belief = _auxiliary_branch_belief(
                    source_belief, critic_source_squares if critic_gradient_diagnostic else None
                )
                critic_auxiliary_loss, critic_auxiliary_terms = structured_critic_terms_fn(
                    critic,
                    structured_critic_dynamics,
                    staged,
                    indices,
                    steps_per_trajectory=rollout.valid.shape[1],
                    config=config,
                    autocast_enabled=autocast_enabled,
                    model_grad=critic_auxiliary_active,
                    complete_windows=False,
                    belief=source_belief,
                    plan=horizon_plan,
                )
                if "structured_critic_preupdate_combined" not in predictor_metrics:
                    predictor_metrics.update(
                        _auxiliary_preupdate_metrics(
                            "structured_critic_preupdate",
                            critic_auxiliary_loss,
                            critic_auxiliary_terms,
                            critic_auxiliary_metrics,
                        )
                    )
                    with torch.no_grad():
                        for label, control in critic_controls:
                            control_loss, control_terms = critic_auxiliary_function(
                                critic,
                                control,
                                staged,
                                indices,
                                steps_per_trajectory=rollout.valid.shape[1],
                                config=config,
                                autocast_enabled=autocast_enabled,
                                model_grad=False,
                                complete_windows=False,
                                belief=_detached_belief(critic_belief),
                                plan=horizon_plan,
                            )
                            predictor_metrics.update(
                                _auxiliary_preupdate_metrics(
                                    f"structured_critic_preupdate_{label}",
                                    control_loss,
                                    control_terms,
                                    critic_auxiliary_metrics,
                                )
                            )
                combined_critic_loss = value_loss + critic_auxiliary_loss
                combined_critic_loss.backward()
                if critic_gradient_diagnostic:
                    gradient_metrics["structured_critic_gradient_source_norm"] = (
                        torch.stack(critic_source_squares).sum().sqrt()
                        if critic_source_squares
                        else actor_zero
                    )
                critic_trunk_gradient_norm = torch.nn.utils.get_total_norm(
                    [
                        parameter.grad
                        for parameter in critic_trunk_parameters
                        if parameter.grad is not None
                    ]
                ).detach()
                torch.nn.utils.clip_grad_norm_(
                    structured_critic_dynamics.parameters(),
                    config.nextlat_max_gradient_norm,
                )
                critic_head_gradient_norm = torch.nn.utils.get_total_norm(
                    [
                        parameter.grad
                        for parameter in critic_head_parameters
                        if parameter.grad is not None
                    ]
                ).detach()
                critic_gradient_norm = torch.linalg.vector_norm(
                    torch.stack((critic_trunk_gradient_norm, critic_head_gradient_norm))
                )
                if critic_auxiliary_end_event is not None:
                    critic_auxiliary_end_event.record()
                else:
                    critic_auxiliary_seconds += time.perf_counter() - critic_auxiliary_started
            else:
                combined_critic_loss = (
                    value_loss + config.economic_forecast_coefficient * forecast_loss
                )
                combined_critic_loss.backward()
                critic_trunk_gradient_norm = torch.nn.utils.get_total_norm(
                    [
                        parameter.grad
                        for parameter in critic_trunk_parameters
                        if parameter.grad is not None
                    ]
                ).detach()
                critic_head_gradient_norm = torch.nn.utils.get_total_norm(
                    [
                        parameter.grad
                        for parameter in critic_head_parameters
                        if parameter.grad is not None
                    ]
                ).detach()
                critic_gradient_norm = torch.linalg.vector_norm(
                    torch.stack((critic_trunk_gradient_norm, critic_head_gradient_norm))
                )

            if run_actor:
                # Read only the actor guard; the already queued critic work
                # can continue while the host enforces the trust region.
                if guard_event is not None:
                    guard_event.synchronize()
                guard_values_list = guard_host.tolist()
                batch_kl_value = guard_values_list[0]
                policy_loss_value = guard_values_list[1]
                if actor_predictor_active:
                    combined_actor_loss_value = guard_values_list[2]
                    actor_gradient_norm_value = guard_values_list[3]
                else:
                    combined_actor_loss_value = policy_loss_value
                    actor_gradient_norm_value = guard_values_list[2]
                batch_component_kl_value = guard_values_list[-1]
                batch_joint_kl_value = guard_values_list[-2]
                actor_auxiliary_seconds += _elapsed_cuda_seconds(
                    actor_auxiliary_start_event, actor_auxiliary_end_event
                )
                if updates == 0:
                    first_minibatch_kl = batch_kl_value
                    first_minibatch_component_kl = batch_component_kl_value
                    first_minibatch_joint_kl = batch_joint_kl_value
                max_approx_kl = max(max_approx_kl, batch_kl_value)
                max_component_kl = max(max_component_kl, batch_component_kl_value)
                max_joint_kl = max(max_joint_kl, batch_joint_kl_value)
                nonfinite_message: str | None = None
                if not math.isfinite(policy_loss_value):
                    nonfinite_message = "non-finite policy loss"
                elif not math.isfinite(combined_actor_loss_value):
                    nonfinite_message = "non-finite structured auxiliary loss"
                elif not math.isfinite(actor_gradient_norm_value):
                    nonfinite_message = "non-finite actor gradient norm"
                elif not math.isfinite(batch_kl_value):
                    nonfinite_message = "non-finite policy KL"
                if nonfinite_message is not None:
                    if actor_step_is_gateable:
                        assert actor_skip is not None
                        _optimizer_step(
                            actor_optimizer,
                            config.actor_learning_rate,
                            config.lr_warmup_steps,
                            found_inf=actor_skip,
                            advance_schedule=False,
                        )
                    raise FloatingPointError(nonfinite_message)
                # The KL is against the actual sampler distribution, so the
                # trust region applies from the first minibatch onward.
                if batch_kl_value > config.target_kl:
                    stop_for_kl = True
                else:
                    _optimizer_step(
                        actor_optimizer,
                        config.actor_learning_rate,
                        config.lr_warmup_steps,
                    )
                    refresh_fused_mlp_fp8(actor, bootstrap_down=False)
                    totals["policy_loss"] -= policy_sum.detach().double()
                    totals["reference_kl"] += reference_kl_sum.detach().double()
                    torch.maximum(
                        max_reference_decision_kl,
                        reference_decision_kl,
                        out=max_reference_decision_kl,
                    )
                    totals["entropy"] += entropy_mean.detach().double() * component_count
                    totals["approx_kl"] += batch_kl * diagnostic_count
                    totals["component_kl"] += batch_component_kl * component_count
                    totals["joint_kl"] += batch_joint_kl * states
                    totals["clip_fraction"] += clipped_sum.detach().double()
                    totals["actor_gradient_norm"] += actor_gradient_norm * states
                    total_components += component_count
                    actor_states += states
                    actor_updates += 1
                    if actor_auxiliary_active:
                        actor_auxiliary_updates += 1
                if actor_predictor_active and actor_auxiliary_terms is not None:
                    assert structured_dynamics is not None
                    assert structured_dynamics_optimizer is not None
                    # The heads read the backbone, so the minibatch that trips
                    # the trust region must not move it either: the encoder
                    # steps exactly when the policy does. Nor does it step when
                    # the auxiliary is off: the objective then took no gradient,
                    # and the backbone is not the policy's optimizer to step.
                    backbone_steps = (
                        bool(world_model) and actor_auxiliary_active and not stop_for_kl
                    )
                    if world_model and not backbone_steps:
                        # Dropped rather than zeroed, because the optimizer
                        # skips a missing gradient outright while a zero one
                        # would still move the weights by momentum.
                        for parameter in world_model:
                            parameter.grad = None
                    _optimizer_step(
                        structured_dynamics_optimizer,
                        config.resolved_structured_learning_rate,
                        config.lr_warmup_steps,
                        held_role=None if backbone_steps else BACKBONE_ROLE,
                    )
                    refresh_fused_mlp_fp8(structured_dynamics, bootstrap_down=False)
                    if backbone_steps:
                        # The backbone stepped after the actor's own refresh
                        # above, so its fused MLP mirrors are refreshed again.
                        refresh_fused_mlp_fp8(actor.trunk, bootstrap_down=False)
                    for name in actor_auxiliary_metrics:
                        actor_auxiliary_totals[name] += (
                            getattr(actor_auxiliary_terms, name).detach().double()
                        )
                    assert actor_auxiliary_loss_total is not None
                    actor_auxiliary_loss_total += actor_auxiliary_loss.detach().double() * states
                    if not stop_for_kl:
                        assert actor_combined_loss_total is not None
                        assert actor_combined_gradient_norm_total is not None
                        actor_combined_loss_total += combined_actor_loss.detach().double() * states
                        actor_combined_gradient_norm_total += actor_gradient_norm.double() * states
                    actor_predictor_updates += 1
                    actor_predictor_states += states
            elif actor_predictor_active and actor_auxiliary_terms is not None:
                assert structured_dynamics is not None
                assert structured_dynamics_optimizer is not None
                actor_auxiliary_seconds += _elapsed_cuda_seconds(
                    actor_auxiliary_start_event, actor_auxiliary_end_event
                )
                if not math.isfinite(float(actor_auxiliary_loss.detach())):
                    raise FloatingPointError("non-finite structured auxiliary loss")
                # The backbone is frozen on this path -- its gradients were
                # never produced -- so neither its clock nor its fp8 mirrors move.
                _optimizer_step(
                    structured_dynamics_optimizer,
                    config.resolved_structured_learning_rate,
                    config.lr_warmup_steps,
                    held_role=BACKBONE_ROLE,
                )
                refresh_fused_mlp_fp8(structured_dynamics, bootstrap_down=False)
                for name in actor_auxiliary_metrics:
                    actor_auxiliary_totals[name] += (
                        getattr(actor_auxiliary_terms, name).detach().double()
                    )
                assert actor_auxiliary_loss_total is not None
                actor_auxiliary_loss_total += actor_auxiliary_loss.detach().double() * states
                actor_predictor_updates += 1
                actor_predictor_states += states
            critic_finite = torch.isfinite(
                torch.stack(
                    (
                        combined_critic_loss.detach(),
                        critic_trunk_gradient_norm,
                        critic_head_gradient_norm,
                        critic_gradient_norm,
                    )
                )
            ).all()
            critic_nonfinite_message = (
                "non-finite combined critic loss or gradient norm"
                if critic_predictor_active
                else "non-finite critic loss or gradient norm"
            )
            if critic_step_is_gateable:
                critic_nonfinite += (~critic_finite).float()
                # Keep all work after the first poisoned minibatch
                # device-skipped until the single end-of-update readback.
                critic_skip = critic_nonfinite.ne(0).float()
                critic_skipped += critic_skip
            else:
                if not bool(critic_finite):
                    raise FloatingPointError(critic_nonfinite_message)
                critic_skip = None
            _optimizer_step(
                critic_optimizer,
                config.critic_learning_rate,
                config.lr_warmup_steps,
                found_inf=critic_skip,
            )
            if critic_predictor_active:
                assert structured_critic_dynamics is not None
                assert structured_critic_dynamics_optimizer is not None
                _optimizer_step(
                    structured_critic_dynamics_optimizer,
                    config.resolved_structured_critic_learning_rate,
                    config.lr_warmup_steps,
                    found_inf=critic_skip,
                )
                refresh_fused_mlp_fp8(structured_critic_dynamics, bootstrap_down=False)
            # A skipped optimizer step leaves the master weights unchanged,
            # so this preserves cache contents while keeping every later
            # fused forward coherent after a finite step.
            refresh_fused_mlp_fp8(critic, bootstrap_down=False)
            totals["value_loss"] += value_loss.detach().double() * states
            totals["critic_gradient_norm"] += critic_gradient_norm * states
            totals["critic_trunk_gradient_norm"] += critic_trunk_gradient_norm * states
            totals["critic_head_gradient_norm"] += critic_head_gradient_norm * states
            if critic_auxiliary_terms is not None:
                for name in critic_auxiliary_metrics:
                    critic_auxiliary_totals[name] += (
                        getattr(critic_auxiliary_terms, name).detach().double()
                    )
                assert critic_auxiliary_loss_total is not None
                assert critic_combined_loss_total is not None
                assert critic_combined_gradient_norm_total is not None
                critic_auxiliary_loss_total += critic_auxiliary_loss.detach().double() * states
                critic_combined_loss_total += combined_critic_loss.detach().double() * states
                critic_combined_gradient_norm_total += critic_gradient_norm.double() * states
                critic_predictor_updates += 1
                if critic_auxiliary_active:
                    critic_auxiliary_updates += 1
                if (
                    critic_auxiliary_events is not None
                    and critic_auxiliary_start_event is not None
                    and critic_auxiliary_end_event is not None
                ):
                    critic_auxiliary_events.append(
                        (
                            critic_auxiliary_start_event,
                            critic_auxiliary_end_event,
                        )
                    )
            total_states += states
            updates += 1
            # Drop Python-owned inputs and returned beliefs before the next
            # gather/forward; backward has already released its saved tensors.
            actor_args = critic_args = inputs = None
            actor_pack = critic_pack = None
            actor_belief = critic_belief = source_belief = None
            if forecast_critic:
                forecast_labels = None
        epoch_marks.append((totals["value_loss"].clone(), total_states))
        completed_epochs += 1

    # All observations are detached scalars. Pack before the existing
    # synchronization instead of interrupting the first minibatch
    # with one host readback per preupdate/persistence field.
    diagnostic_metrics = predictor_metrics | gradient_metrics
    diagnostic_values = (
        torch.stack(tuple(diagnostic_metrics.values())) if diagnostic_metrics else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mark_phase("update_minibatch_seconds")
    skipped_critic_steps = int(critic_skipped.item()) if critic_step_is_gateable else 0
    if skipped_critic_steps:
        _restore_skipped_optimizer_steps(
            critic_optimizer,
            critic_schedule_state,
            skipped_critic_steps,
            config.lr_warmup_steps,
        )
        if structured_critic_dynamics_optimizer is not None:
            assert critic_predictor_schedule_state is not None
            _restore_skipped_optimizer_steps(
                structured_critic_dynamics_optimizer,
                critic_predictor_schedule_state,
                skipped_critic_steps,
                config.lr_warmup_steps,
            )
        critic_nonfinite_message = (
            "non-finite combined critic loss or gradient norm"
            if critic_predictor_active
            else "non-finite critic loss or gradient norm"
        )
        raise FloatingPointError(critic_nonfinite_message)

    if critic_auxiliary_events is not None:
        for start_event, end_event in critic_auxiliary_events:
            critic_auxiliary_seconds += _elapsed_cuda_seconds(start_event, end_event)

    first_epoch_value_loss, last_epoch_value_loss = _epoch_value_losses(epoch_marks)
    diagnostic_total = actor_states if config.policy_ratio_scope == "joint" else total_components
    metrics: dict[str, float | int] = {
        "updates": updates,
        "actor_minibatches_intended": actor_minibatches_intended,
        "actor_updates": actor_updates,
        "epochs": completed_epochs,
        # The valid states this update trained on, which `rows` restricts to
        # one member's share of the wave.
        "states": valid_indices.size,
        # Whether GAE read the collector's values instead of replaying them.
        "update_behavior_values_carried": int(behavior_values_carried),
        "policy_loss": float(
            totals["policy_loss"]
            / max(1, actor_states if config.policy_loss_reduction == "states" else total_components)
        ),
        "value_loss": float(totals["value_loss"] / max(1, total_states)),
        **(
            {
                "economic_forecast_loss": float(forecast_loss_total / max(1, total_states)),
                "economic_forecast_persistence_loss": float(
                    forecast_persistence_total / max(1, total_states)
                ),
                "economic_forecast_coefficient": config.economic_forecast_coefficient,
            }
            if forecast_critic
            else {}
        ),
        "value_loss_first_epoch": first_epoch_value_loss,
        "value_loss_last_epoch": last_epoch_value_loss,
        "entropy": float(totals["entropy"] / max(1, total_components)),
        "approx_kl": float(totals["approx_kl"] / max(1, diagnostic_total)),
        "component_kl": float(totals["component_kl"] / max(1, total_components)),
        "joint_kl": float(totals["joint_kl"] / max(1, actor_states)),
        # Per active decision, and the worst single state's total: the drift
        # this anchor exists for is invisible in the first and plain in the second.
        "reference_kl": float(totals["reference_kl"] / max(1, total_components)),
        "max_reference_decision_kl": float(max_reference_decision_kl),
        "max_component_kl": max_component_kl,
        "max_joint_kl": max_joint_kl,
        "first_minibatch_component_kl": first_minibatch_component_kl,
        "first_minibatch_joint_kl": first_minibatch_joint_kl,
        "max_approx_kl": max_approx_kl,
        "first_minibatch_approx_kl": first_minibatch_kl,
        "kl_early_stop": int(stop_for_kl),
        "clip_fraction": float(totals["clip_fraction"] / max(1, diagnostic_total)),
        "actor_gradient_norm": float(totals["actor_gradient_norm"] / max(1, actor_states)),
        "critic_gradient_norm": float(totals["critic_gradient_norm"] / max(1, total_states)),
        "critic_trunk_gradient_norm": float(
            totals["critic_trunk_gradient_norm"] / max(1, total_states)
        ),
        "critic_head_gradient_norm": float(
            totals["critic_head_gradient_norm"] / max(1, total_states)
        ),
        "advantage_mean": prepared.raw_advantage_mean,
        "advantage_std": prepared.raw_advantage_std,
        **entity_metrics,
        "value_target_mean": float(prepared.value_targets[owned_valid].mean()),
        "value_target_std": float(prepared.value_targets[owned_valid].std()),
        "value_target_min": float(prepared.value_targets[owned_valid].min()),
        "value_target_max": float(prepared.value_targets[owned_valid].max()),
        # Every target statistic here, the suffix-return EV and R-squared,
        # and the correlation below use the unclipped critic target
        # (the lambda-one return). `value_loss` and the two critic-fit explained
        # variances are the exceptions by necessity, since the critic regresses
        # on the saturated copy.
        "value_target_saturated_fraction": float(saturated) / float(valid_value_targets.size),
        "actor_gae_lambda": config.actor_gae_lambda,
        "critic_gae_lambda": config.critic_gae_lambda,
        "gamma": config.gamma,
        "actor_learning_rate": float(actor_optimizer.param_groups[0]["lr"]),
        "critic_learning_rate": float(critic_optimizer.param_groups[0]["lr"]),
        "critic_head_learning_rate": float(
            next(
                group["lr"]
                for group in critic_optimizer.param_groups
                if any(parameter is critic.value_head.weight for parameter in group["params"])
            )
        ),
        # Against the discounted suffix return, which decoupled GAE also uses
        # as the critic target. Pre-update predictions, so this is not a fit.
        "monte_carlo_explained_variance": _explained_variance(
            prepared.monte_carlo_returns, behavior_values, owned_valid
        ),
        # Retain constant bias in the residual: readiness must reject a critic
        # with the right ordering but the wrong value baseline.
        "monte_carlo_r_squared": _r_squared(
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
    if diagnostic_values is not None:
        metrics.update(zip(diagnostic_metrics, diagnostic_values.cpu().tolist(), strict=True))
    metrics.update(
        _credit_quality_metrics(
            rollout,
            prepared.monte_carlo_returns,
            behavior_values,
            owned_valid,
            config.gamma,
            diagnostic_groups,
        )
    )
    if actor_predictor_active:
        metrics["structured_actor_auxiliary_enabled"] = int(actor_auxiliary_active)
        metrics["structured_actor_predictor_updates"] = actor_predictor_updates
        metrics["structured_actor_auxiliary_updates"] = actor_auxiliary_updates
        metrics["structured_actor_auxiliary_seconds"] = actor_auxiliary_seconds
        metrics.update(
            {
                f"structured_actor_{name}": float(total / max(1, actor_predictor_updates))
                for name, total in actor_auxiliary_totals.items()
            }
        )
        assert actor_auxiliary_loss_total is not None
        assert actor_combined_loss_total is not None
        assert actor_combined_gradient_norm_total is not None
        metrics["structured_actor_auxiliary_loss"] = float(
            actor_auxiliary_loss_total / max(1, actor_predictor_states)
        )
        metrics["structured_actor_combined_loss"] = float(
            actor_combined_loss_total / max(1, actor_states)
        )
        metrics["structured_actor_combined_gradient_norm"] = float(
            actor_combined_gradient_norm_total / max(1, actor_states)
        )
    if critic_predictor_active:
        metrics["structured_critic_auxiliary_enabled"] = int(critic_auxiliary_active)
        metrics["structured_critic_predictor_updates"] = critic_predictor_updates
        metrics["structured_critic_auxiliary_updates"] = critic_auxiliary_updates
        metrics["structured_critic_auxiliary_seconds"] = critic_auxiliary_seconds
        metrics.update(
            {
                f"structured_critic_{name}": float(total / max(1, critic_predictor_updates))
                for name, total in critic_auxiliary_totals.items()
            }
        )
        assert critic_auxiliary_loss_total is not None
        assert critic_combined_loss_total is not None
        assert critic_combined_gradient_norm_total is not None
        metrics["structured_critic_auxiliary_loss"] = float(
            critic_auxiliary_loss_total / max(1, total_states)
        )
        metrics["structured_critic_combined_loss"] = float(
            critic_combined_loss_total / max(1, total_states)
        )
        metrics["structured_critic_combined_gradient_norm"] = float(
            critic_combined_gradient_norm_total / max(1, total_states)
        )
    mark_phase("update_finalize_seconds")
    metrics.update(phase_seconds)
    return metrics
