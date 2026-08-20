# Next-Latent Prediction Auxiliary Loss: Assessment and Plan

Source: "Next-Latent Prediction Transformers Learn Compact World Models"
(Teoh et al., Microsoft Research, arXiv 2511.05963v4). Verified against the
full PDF, not just an abstract summary.

## Status

**Wired, measured, and the verdict splits the paper's two terms.** On main the
auxiliary is live in `scripts/train_bc.py` behind `--latent-dynamics-coefficient`
(the reference's `lambda_mse`), `--latent-decode-coefficient` (`lambda_kl`) and
`--latent-horizon` (`mtp_horizon`), all defaulting to zero, so the shipped clone
is the path it was. One trunk pass serves both objectives;
`src/kaggriculture/latent_dynamics.py` carries `LatentDynamics`,
`latent_dynamics_loss`, `latent_dynamics_halves`, `latent_decode_kl`,
`consecutive_rows`, `latent_horizon_loss` and `belief_spread` with 16 pins.

The load-bearing pin is `test_the_decode_reproduces_the_actors_own_heads_exactly`:
decoding the belief with detached head weights reproduces the actor's own logits
at rtol=0, atol=0. It fails on a wrong split index, on a second `market_norm`,
and on any per-row broadcast, which is what the first implementation did.

### Measured outcome

Four arms, identical corpora (the four `public-v16` sets at
`--seeds-per-dataset 64`), identical 12-epoch trapezoid, identical
`--run-length 4` sampler so only the objective differs. Play measured in the
official `kaggle_environments` engine, 16 games per opponent per seed. The two
decision arms ran **sixteen training seeds each**; `dyn` and `dyn-kl` stopped at
three, having already failed.

| arm | lambda_mse | lambda_kl | learning-phase NLL | final NLL | vs `starter` | vs `public-v16` | beats `public-v27` |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 0.00674 +- 0.00051 (n=16) | 0.00143 | 1.000 | 0.875 | **0/16** |
| dyn | 1.0 | 0 | 0.00666 (n=3) | 0.00153 | 1.000 | 0.854 | 0/3 |
| dyn-kl | 1.0 | 0.5 | 0.00629 (n=3) | 0.00151 | 1.000 | 0.875 | 0/3 |
| **kl** | 0 | 0.5 | **0.00508 +- 0.00051 (n=16)** | **0.00107** | 1.000 | 0.875 | **5/16** |

**The KL distillation term improves imitation decisively and makes a strong basin
reachable that is otherwise not reached at all. The SmoothL1 latent regression
does neither.** The first half inverts the paper's emphasis, where `lambda_mse` is
the headline term at 1.0-3.0 and the KL is the optional extra.

Five readings, each with its own evidence:

1. *Imitation sample efficiency: overwhelming.* Learning-phase holdout NLL (epochs
   2-7) is 25% lower, 0.00508 +- 0.00051 against 0.00674 +- 0.00051 at identical
   spread. Of the 256 cross-arm seed pairs, 255 favour the auxiliary, so
   Mann-Whitney gives z = +4.8, p ~= 1e-6. Final NLL 0.00107 against 0.00143.
2. *Play: a reachability effect, not a mean shift.* Five of sixteen KL seeds beat
   `public-v27` (0.625 to 1.000) and **none of sixteen** baseline seeds do -- best
   baseline 0.125, and none of the six `dyn`/`dyn-kl` runs either. Fisher
   one-sided p = 0.022. But on the continuous objective -- `_relative_score`, the
   bank margin the engine actually scores -- the rank test is a wash (U = 144
   against 128 expected, z = +0.60) even though mean money is 78% higher (34,515
   against 19,411). The auxiliary does not move the average member; it makes a
   basin reachable roughly a third of the time that is otherwise unreachable.
   Winners match the anchor exactly: `runs/bc6-v16-mixed`, 4x the corpus and 20
   epochs, scores 0.9375 with money 80,904 to 71,427, and the KL winners hold
   79,051-84,183 against 71,459-72,424. A KL seed that lands well buys what 4x the
   demonstration data buys. An earlier revision claimed p ~= 0.045 from 2-of-3
   against 0-of-9; that was small-sample luck and is withdrawn in favour of the
   n=16 figure.
3. *All the signal is off-distribution.* Against `starter` every arm saturates at
   1.000 and against its own teacher every arm sits at 0.875. Only `public-v27`
   -- stronger, and not the teacher of these corpora -- separates them, which is
   what a world-model auxiliary is supposed to buy and is exactly the failure
   this project already diagnosed: a clone gated on opponent wealth channels
   that collapses against opponents it never saw.
4. *The two terms are not proxies for each other.* In the KL-only arm the
   unsupervised SmoothL1 climbs 0.059 -> 0.85 while its decode KL falls
   0.077 -> 0.068. The belief becomes less latently predictable and more
   decision-preserving at the same time. Latent coordinates are not the quantity
   that matters; the action distribution they induce is.
5. *No training-time metric knows which seed won.* Across sixteen KL seeds the
   five winners and eleven losers are indistinguishable on every number a run
   records, and what difference there is points the wrong way: learning-phase NLL
   0.00540 for winners against 0.00493 for losers, final NLL 0.00102 against
   0.00109, the auxiliary's own KL identical to four decimals, belief dispersion
   0.511 against 0.524. Holdout likelihood here is not merely saturated, it is
   uninformative about off-distribution play -- and slightly anti-correlated with
   it. Selection has to be by play, which is exactly what `build_submission.py`'s
   strength gate already refuses to ship without.

Against the failure direction: `dyn` seeds finish 16 games against `public-v27`
holding 311 and 390 money, having burned a 3,000 bank. Regressing raw latent
coordinates does not merely fail to help, it competes with the clone for trunk
capacity and makes the policy catastrophically fragile off-distribution.

The bimodality is a property of the training seed, not of the evaluation. Every
one of the six baseline and KL artifacts was replayed against `public-v27` on a
disjoint game-seed block (6000-6007 against 5000-5007) and each reproduced its
own verdict: both dead baselines stayed at 0.000, the dead KL seed stayed at
0.000, and the two live KL seeds went 0.938 -> 1.000 and 1.000 -> 1.000, winning
16 of 16 on states they had never seen. Which basin training lands in is what
varies; where a trained artifact ends up is stable across 32 games.

Honest limits. Cost is +10.6% per epoch for either term. Neither `lambda_kl` nor
the horizon was swept; both sit at first-choice values (0.5 and 2). The whole A/B
ran at `--seeds-per-dataset 64`, a deliberately data-starved regime; whether the
term still pays at 256, where the plain baseline already reaches 0.9375, is
untested. The play half rests on a 5-in-16 basin rate, so the recipe is "train K
seeds with the auxiliary and select on play", not "turn the auxiliary on"; without
the selection step the expected member is no better on the objective the engine
scores.

Contributing measurement, since it changes what a run costs: compiling the clone
step is 2.03x per epoch at `--compile-mode default`, now the shipped value. It
is deliberately NOT `max-autotune-no-cudagraphs`, which reaches the same steady
state for 473s of compilation against 39s and only breaks even past 16 epochs.

## What the paper shows

NextLat augments next-token prediction with a self-supervised latent loss: a
small dynamics model p_psi learns to predict the transformer's own next hidden
state h_{t+1} from (h_t, X_{t+1}), with a stop-gradient on the target and
Smooth L1 regression, plus an optional KL term distilling the frozen token
head's distribution at the predicted latent. Theorem 3.2 proves that jointly
optimizing next-token consistency and transition consistency forces h_t to be
a belief state — a sufficient statistic of the history. Empirically this
yields dramatically better learned world models (Manhattan taxi: effective
latent rank 52.7 vs 160.1 for GPT), better lookahead planning (Path-Star,
Countdown), and equal-or-better language modeling, all at negligible training
overhead for d = 1. The belief-state guarantee already holds at d = 1;
deeper rollout horizons only enrich the gradient signal.

The authors explicitly place the method in the self-predictive RL lineage
(SPR; Tang et al. 2023; Ni et al. 2024): representation learning by
minimizing prediction error of the model's own future latents.

## Fit to this pipeline: the mechanism transfers, the recipe does not

The paper's setting is a temporal sequence model: a transformer attending
over history tokens, which has no architectural pressure to compress that
history. Our `FarmActor` is different in a load-bearing way: it is a
**per-step entity transformer** (state token + board tokens + unit tokens +
market query tokens, all from one observation) with **no temporal memory at
all**. There is no history inside the model to compress, so the paper's exact
objective — predict the next hidden state of a sequence position — has no
direct graft point.

What does transfer is the RL form the paper descends from: **self-predictive
representation learning across environment steps**. Train a latent dynamics
model p_psi(h_t, a_t) -> h_{t+1} on rollout transitions, where h_t is the
actor's post-transformer state-token latent and a_t embeds the executed joint
action. This pressures h_t toward a compact predictive summary of the
controllable dynamics — exactly the belief-state property, transplanted from
"position t in a token sequence" to "step t in an episode".

Why this is plausible here and not a reflexive patch:

- The game is **partially observable**: the policy sees only the acting
  player's private state; opponent intent and future market pressure are
  hidden. Belief-shaped latents are the principled response to partial
  observability, even for a memoryless policy (dense next-step supervision
  still shapes what the encoder extracts from the current observation).
- Reward is a shaped bank delta — informative but scalar and slow. The latent
  loss adds a **dense vector-valued learning signal per transition** at
  near-zero parameter cost, the same data-efficiency argument the paper
  validates in low-data regimes (Path-Star, 200k samples).
- It touches representation quality, which our external calibration says is
  not the binding constraint today (see below) — so it is scheduled behind
  the fix that is.

## What it will not fix

The measured gap to public-v27 (~5.6x bank at iteration 388-415) is
behavioral: land purchase, animals, DIG, and FERTILIZE sit at exactly zero in
the action mix. That is an exploration/curriculum failure, and the
behavior-cloning warm start (BC_WARMSTART_PLAN.md) attacks it directly. A
representation auxiliary cannot conjure unvisited subsystems out of on-policy
data that never contains them. NextLat-style latents are worth having, but as
an accelerant for the post-BC RL fine-tune, not as the headline fix.

## Concrete design

1. **Latent**: the 26 head-input tokens of `FarmActor` -- 16 unit tokens
   concatenated with 10 market tokens, `(rows, 26, model_dim)`, exposed by
   `forward_with_belief` and pinned by
   `test_belief_is_fp32_one_token_per_unit_and_order_slot_under_bf16_autocast`.

   This plan first specified the post-transformer **state token** on the grounds
   that it "is not consumed by any action head, making it a clean belief-summary
   site". Measured (`src/kaggriculture/model.py:676-695`): the claim is true and
   it is a **disqualification**, not a qualification. The head slices begin at
   index 101; nothing reads index 0. A latent no head reads cannot be regressed
   against a decode of itself through those heads -- the norm and projection
   would be applied to a vector their weights have never seen, and the decode
   term (item 5), which is the term that makes the latent decision-relevant, is
   then measuring a distribution the policy never computes. Collapsing 16 unit
   slots to one row-level distribution also cannot express "unit 3 harvests
   while unit 7 walks", which is what a step of this policy consists of.

   In the reference the regressed latent IS the decode vector: `teacher_logits =
   lm_head(hidden)` reads exactly the tensor the dynamics model predicts. Our
   analog is therefore the tensor the policy heads read, and p_psi predicts
   **per token with shared weights** -- action embedding broadcast across the 26
   tokens, MLP applied to the last dimension, residual per token. Flattening the
   26 into one 2496-wide regression would invent a per-position parameterization
   the reference does not have and multiply the parameter count by 26.

   One asymmetry is ours to carry, not the reference's: the unit half is raw
   trunk output while `market_hidden = self.market_norm(...)` is post-RMSNorm.
   Measured at production config, unit RMS 0.563 against market 1.000 -- a 1.78x
   gap, not the 10x an earlier survey assumed. A SmoothL1 over the concatenation
   is therefore mildly dominated by the market half. Reported per half rather
   than silently rescaled, so a play difference can be attributed.
2. **Action embedding**: the executed joint action is (per-unit action ids,
   market orders as kind/quantity pairs). Embed with the same factorization
   the policy heads use: sum of unit-action embeddings + sum of market
   (kind embedding * quantity-bin embedding), projected to model_dim. All
   components already exist as vocabularies; no new action space design.
3. **Dynamics model**: `h_hat_{t+1} = h_t + MLP(RMSNorm(concat(a_embed, h_t)))`,
   two hidden layers at 4x model_dim. Two details are the reference's, not
   ours, and both were missing from this plan's first draft
   (`NextLat/models/model_nextlat.py:47-92`): the prediction is a **residual
   delta** on the current latent, and the concat is **normalized** before the
   MLP. A plain `MLP(concat) -> h_hat` has to relearn the identity map that the
   residual gives for free.
4. **Regression term** (`lambda_mse` in the reference, 1.0-3.0 in its shipped
   configs): `smooth_l1_loss(h_hat_{t+1}, sg(h_{t+1}), reduction="none")`,
   masked, divided by the masked ELEMENT count. Despite the name it is SmoothL1,
   not MSE (`model_nextlat.py:303`). Skip terminal transitions: step 719 has no
   successor.
5. **Decode term** (`lambda_kl`, 0.1-1.0 in the reference's configs). This
   plan's first draft claimed the paper's KL term "has no analog here (no token
   head over observations)". That is wrong, and it matters. The reference's KL is
   not over observations: it decodes the PREDICTED latent through the output
   head's own weights, detached, and matches the model's own next-step output
   distribution, also detached (`model_nextlat.py:313-328`). Our output head is
   the policy, so the analog is exact and needs nothing invented: decode
   `h_hat_{t+1}` through the frozen unit/kind/quantity heads and take
   `KL(sg(pi_{t+1}) || pi(h_hat_{t+1}))` under the same legality masks the
   policy uses at t+1. Without it the latent only has to be self-predictable;
   with it, it has to be *decision-relevant*, which is the property we want. The
   critic-distillation substitute this plan proposed instead is strictly worse:
   it supervises a different head than the one the aux is trying to help.
   `lambda_ce` (the reference's third term) is 0 in every shipped config and is
   not implemented.
6. **Horizon** (`mtp_horizon`, 1 by default and 4/8 swept in the reference's
   sweeps, `model_nextlat.py:369-408`): recursively feed the prediction back in,
   `h_hat_{t+k} = p_psi(h_hat_{t+k-1}, a_{t+k})`, accumulating both terms and
   averaging over k. Each extra step tightens eligibility by one row.
7. **Data plumbing, BC first.** The reference's setting is sequence modelling,
   which our BC pretraining resembles far more closely than our RL update does,
   and BC is where a 14-minute run can answer the question. But BC batches are a
   flat shuffle over rows, so `(h_t, h_{t+1})` pairs do not co-occur. Staging
   therefore carries `episode_index` and `step` per row and the sampler draws
   **contiguous runs** (`--run-length`), so successors land adjacent in the same
   minibatch and the aux costs one MLP rather than a second trunk pass.
   Eligibility is `episode_index[j] == episode_index[j+1] and step[j+1] ==
   step[j] + 1`, tightened by the horizon. RL plumbing is easier and comes
   second: rollout storage is already contiguous `[game, step]`.
8. **Objective**: `latent_dynamics_coefficient` (regression) and
   `latent_decode_coefficient` (decode KL), both 0.0 = term absent, plus
   `latent_horizon`. In BC they are trainer arguments; in RL they join
   `PpoConfig` and the actor loss only, critic untouched. p_psi joins the actor
   optimizer and NEVER the actor's state dict: league snapshots, inference
   bundles and the frozen-ensemble stack all consume that dict whole.
9. **Evidence gate** (no smoke runs; every number below is a measurement or the
   claim is withdrawn):
   - **BC A/B, the primary gate.** Same corpora (the four `public-v16` sets at
     `--seeds-per-dataset 256`), same seed, same 12-epoch trapezoid, same
     `--run-length`; the only difference is the two coefficients. Report holdout
     NLL and per-head accuracy, and then the number that actually matters:
     money and score rate in the OFFICIAL engine against `starter`, `pass`,
     `random`, `public-v27` and `public-v16`. A clone that fits the corpus
     better while playing no better has not earned the term.
   - **Held-out-opponent generalization.** The paper's claim is compactness, so
     the interesting cell is the opponent the corpus does NOT contain: train the
     A/B on three corpora and evaluate against the fourth opponent. This is the
     measurement that can distinguish a better world model from a better fit.
   - **Collapse diagnostics in the journal**, not inferred afterwards: the
     auxiliary loss per term, and belief dispersion. The failure mode is a loss
     falling to zero because every latent became the same vector; the
     stop-gradient makes that unlikely, not impossible.
   - **Cost.** Wall clock per epoch before and after. The aux adds one MLP over
     rows already in the batch, so the budget is 3%; the run-sampler change is
     measured separately, since correlated batches are a real change to the
     baseline's own gradient statistics.
   - **RL, only after the BC A/B has an answer.** `benchmark_ppo_iteration`
     before/after at <= 3% throughput cost, then the population A/B.

## Sequencing

1. BC warm-start: done (`runs/bc6-normuon-conv`, holdout NLL 0.00182).
2. **This auxiliary in BC pretraining, measured, before any RL run.** That is a
   change from this document's first draft, which put it in the RL fine-tune.
   Two reasons: BC is the setting the paper's objective was designed for, and a
   12-epoch clone answers in minutes what a 500-iteration league run answers in
   days. If the term cannot improve a supervised clone whose targets are exactly
   the demonstrated actions, its case for surviving a noisy policy-gradient is
   weak.
3. Then RL: the same two coefficients on the actor loss, attributed by the
   population A/B rather than bundled into a relaunch.
