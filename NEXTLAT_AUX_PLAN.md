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
3. **Dynamics model**: 2-layer MLP p_psi(concat(h_t, a_embed)) -> h_hat_{t+1},
   hidden 2x model_dim. Loss SmoothL1(sg(h_{t+1}), h_hat_{t+1}), d = 1.
   Skip terminal transitions (step 719 has no successor). The paper's KL
   distillation term has no analog here (no token head over observations);
   the principled substitute, if regression alone under-constrains, is
   distilling the frozen critic's value distribution at h_hat — hold that
   back unless d = 1 regression shows representational collapse, which the
   stop-gradient is there to prevent.
4. **Data plumbing**: rollout storage is contiguous [game, step], so the
   transition pair (t, t+1) is index arithmetic on the existing arrays. The
   actor already re-forwards every stored state during update epochs; cache
   the state-token latent in that same forward (one extra tensor slice, no
   extra pass) and gather successor latents by shifted index within the same
   minibatch's episodes. Transitions whose successor falls outside the
   minibatch are supervised in a second gather against latents computed under
   no_grad — the stop-gradient target needs no fresh graph.
5. **Objective**: add `latent_dynamics_coefficient` to `VapoConfig`
   (0.0 = term absent). Add the term to the actor loss only; the critic
   tower stays untouched. p_psi parameters join the actor optimizer.
6. **Evidence gate** (per machine-learning skill, no smoke runs):
   - `benchmark_vapo_iteration` before/after: accept <= 3% throughput cost.
   - Full calibration A/B at production config: the auxiliary must not
     degrade money_mean progression at matched wall-clock; success criterion
     is a measurable improvement in early-phase sample efficiency
     (iterations to 30k money_mean) or final strength.
   - Journal the auxiliary loss so collapse (loss -> 0 with cosine
     similarity of latents -> 1) is visible.

## Sequencing recommendation

1. First: BC warm-start (already scoped) — targets the actual measured gap.
2. With it: this auxiliary in the RL fine-tune phase, where its
   data-efficiency benefit is largest and the fresh-critic phase already
   re-forwards everything.
3. Attribution: the calibration A/B above isolates the auxiliary's effect;
   do not bundle it untested into a relaunch whose gains BC will dominate.

Effort estimate: ~1 day implementation + tests (action embedding reuse makes
this small), plus one calibration cycle for the gate.
