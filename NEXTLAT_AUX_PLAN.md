# Next-Latent Prediction Auxiliary Loss: Assessment and Plan

Source: "Next-Latent Prediction Transformers Learn Compact World Models"
(Teoh et al., Microsoft Research, arXiv 2511.05963v4). Verified against the
full PDF, not just an abstract summary.

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

1. **Latent**: the post-transformer state token of `FarmActor` (model_dim,
   currently 96). It already aggregates board/unit/market context through
   attention and is not consumed by any action head, making it a clean
   belief-summary site.
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
