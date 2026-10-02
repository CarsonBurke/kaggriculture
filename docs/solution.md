# Current solution

Technical reference for the production neural solution as of October 1, 2026.
Historical checkpoints use their recorded configurations, which may differ from these defaults.

## Training pipeline

- The production model family is `lejepa`.
- Training starts from an actor behavior-cloned on replays of hosted leaderboard games between agents rated at least 2600. The BC artifact also contains its trained JEPA objective.
- PPO initializes a fresh critic and optimizer state. Loading a BC actor is initialization; resuming a training checkpoint restores the full training state.
- The actor and shared backbone remain frozen during critic warmup.
- Warmup lasts at least 10 iterations. Actor training starts when every learner's previous fresh-wave, pre-update Monte Carlo return R² reaches 0.10. Training fails if the critic is still unready at iteration 40.

## Reward and value targets

- The default reward is `terminal-outcome`: zero on intermediate transitions, then +1 for a win, 0 for a draw, or -1 for a loss.
- Winning is determined by final bank balance. Unsold assets do not contribute to the official score.
- The two players receive opposite rewards. The magnitude of the bank margin does not affect the default reward.
- Discount factor, actor GAE lambda and critic GAE lambda are all 1. The critic target is the completed-game outcome at each valid state; the actor advantage is that outcome minus the behavior-state value estimate.
- Advantage normalization is disabled. The optional group-relative own-bank advantage has coefficient 0.
- The production critic predicts three classes in loss/draw/win order, using unsmoothed cross-entropy. Its scalar value is `P(win) - P(loss)`; expected match score is `(value + 1) / 2`.
- The WDL head requires terminal-outcome rewards, discount factor 1 and critic lambda 1.
- Alternative reward modes remain available: `terminal-bank` pays `(bank0 - bank1) / (bank0 + bank1 + 6000)` at termination; `shaped` uses potential differences with a terminal bank-utility correction.
- With `wdl_value=false`, the alternative critic uses HL-Gauss with 255 bins on [-2.2, 2.2] and Gaussian sigma of three bin widths. The scalar-value option uses half squared error. Production uses neither alternative, symlog nor a critic EMA.

## Temperature and exploration

- Training samples at temperature 1 for both the learner and frozen neural opponents. PPO rejects other learner temperatures.
- The entropy bonus coefficient is 0.
- Argmax evaluation and temperature-1 evaluation measure different behavior and are reported separately.
- Earlier runs used lower temperatures for league opponents. That made identical frozen snapshots better executors of the learner's policy and distorted league results; the current configuration uses a common temperature.

## Observations

- The actor sees both public farms, both bank balances, the market and town, and its own private seeds, shed contents and carried inventory.
- The centralized critic additionally receives the opponent's private seeds, shed contents and unit inventories. These features enter the private critic tower, not the shared actor encoder.
- Farm tokens are ordered own farm first, opponent farm second. Tile order is `y * 10 + x`. Unit slots follow engine execution order, with the farmer first.
- The production model consumes observation schema 8, the tokenizer's current schema. Older artifacts record lower schemas; embedders select the prefix corresponding to the model's recorded schema.
- Tile inputs contain 200 tokens, each with six categorical and 20 continuous features. Categories describe tile kind, occupant, farm ownership, row, column and quadrant. Continuous features describe yield, age, care, decay, lifespan, harvest readiness, shed access and public unit occupancy.
- Own-unit inputs contain 16 slots with four categorical and 26 continuous features, an active mask, and local tile indices for HERE/NORTH/SOUTH/EAST/WEST. Off-board neighbors have separate validity masks.
- Unit continuous features include twelve carried-item counts, total cargo, shed access and twelve inventory insertion ranks. Insertion order matters because DROP fills the shed in that order and discards overflow.
- Economy inputs contain 20 tokens: nine products, three animals, five crops, two farm summaries and one town/clock token.
- Schema-4 base economy feature widths are five per product, three per animal, six per crop, five per farm and fourteen for the town.
- Bank values use signed log1p divided by 12. Schema 4 also includes the unscaled difference between signed-log bank values, computed before low-precision staging to preserve small leads.
- Later schemas add shop unlock order (5), held-sale value and liquidation (6), supply/demand outlook and forecast prices (7), and crop/animal payback (8). The production model consumes all of them.

## Shared backbone

- Model width is 96. Attention uses four query heads and two key/value heads. Feed-forward width is twice the model width.
- Layers use RMS normalization, ReLU-squared feed-forward activations, gated residuals and economy-conditioned scale/shift modulation.
- Two farm-local transformer blocks encode each farm independently with shared weights and axial 2D rotary position embeddings.
- Source memory contains 200 encoded tiles and 20 economy tokens. It remains static during entity reasoning; key/value projections are separate for each round by default.
- The model maintains 26 decision states: sixteen unit states and ten market-order states.
- Four reasoning rounds apply entity self-attention, cross-attention to source memory, and a feed-forward update.
- Inactive units are excluded as attention keys and zeroed after each branch. All market-order states are initially valid; STOP is handled during action sampling.
- The backbone exposes unit states, market-order states, economy tokens and tile tokens as its belief representation.
- The production model has no recurrent state across turns, generic scratch latents or source-memory writeback.

## Actor and critic

- The actor applies one global cross-attention readout round over the shared belief before producing action logits.
- Policy and BC gradients pass through the actor readout into the backbone (`policy_shapes_backbone=true`).
- The critic reads a detached copy of the backbone belief.
- Its private tower embeds sixteen opponent-unit slots and twenty economy tokens enriched with opponent-private features, then applies one cross-attention round into the shared belief.
- A learned query pools the shared context and private tower. Source reading is enabled, so pooling covers 246 shared slots and 36 private slots, with inactive units masked.
- The pooled value representation has shape `[batch, 1, 96]`. A critic readout feed-forward layer is enabled before the WDL head.
- The standalone `entity-attention` family has separate actor and critic encoders and different critic defaults. It is a distinct architecture from production LeJEPA.

## Optimizer ownership

- The actor optimizer owns policy readout and action-head parameters.
- The world-model optimizer owns the shared backbone and JEPA projector/predictor parameters. It receives the combined policy and JEPA gradients, clipped to gradient norm 1.
- The critic optimizer owns the private embeddings, critic tower, pooling and value head. Critic gradients do not update the backbone.
- Actor-head and critic-tower gradients have no global gradient clipping.
- The backbone is registered under the actor; the critic holds an unregistered reference. Critic checkpoints therefore do not contain another copy of the backbone.
- Backbone device and train/eval mode follow the actor. Independently copied actor/critic pairs must be reattached through the registry or pair builder.
- A PPO KL stop also stops backbone and JEPA optimizer steps. The critic tower can continue fitting without moving the policy representation.

## Unit actions

- Each of sixteen unit slots selects from 68 primitives. These include passing, cardinal movement, dropping, quantified pickups, placing products or animals, planting, watering, harvesting, fertilizing, digging, building, feeding and care operations.
- Pickup quantities cover wheat 1–16, fertilizer 1–8 and each animal 1–4.
- Legality masks account for earlier units' effects on shed contents, tiles and seeds in engine execution order.
- Local unit initialization and the unit affordance scorer are enabled. The scorer uses the relevant tile and resource features for each primitive. The optional additional local attention readout is disabled.
- Target navigation is enabled. Destination scores are marginalized into the four movement probabilities using the next step of a vertical-first Manhattan route. The destination prior corrects for unequal directional region sizes.
- Navigation is reconsidered each turn. BC and PPO score the executed movement action, without a separate sampled destination or committed route.

## Market actions

- The actor emits up to ten ordered market slots. Each selects a kind followed by a quantity when the kind requires one; STOP ends the order list.
- There are 22 kinds: STOP, hiring, land purchase, five seed purchases, wheat/fertilizer purchases, three animal purchases and nine product sales.
- Production uses action interface 2: integer quantities 1–100 plus an ALL category. ALL resolves to the current feasible maximum, and its probability mass is merged into that integer before sampling and scoring.
- Quantity heads use rank-32 context and a selected-kind gate. Quantity arithmetic runs in FP32.
- The market resource ledger includes preceding unit effects and updates after every order. It tracks cash, seeds, shed contents, market inventory, hiring and land purchases.
- Trade prices are computed per unit as inventory changes. Multiplying the initial quote by the quantity does not reproduce engine proceeds.
- Market resource conditioning is enabled. Two initially zero linear residuals adjust kind logits and quantity context from 29 pre-order resource features: money, twelve shed counts, nine market inventories, hiring, land and five seed counts.
- The main encoder runs once per turn. Resource residuals run per order; exact pre-order features are recorded for PPO and reconstructed from demonstrations for BC.
- Action masks, ALL marginalization, resource features and sampled log-probabilities must agree across native sampling, Python inference, BC and PPO.

## LeJEPA objective

- The training-only world model predicts the next projected belief conditioned on the executed action. Prediction horizon is one transition.
- Successor targets remain attached to the gradient graph. There is no EMA teacher.
- Unit, market, economy and tile representations use separate typed projector groups. Projector/predictor hidden width is 384.
- Prediction coefficient is 1; SIGReg coefficient is 0.09; reward-prediction coefficient is 0.
- SIGReg uses 128 random directions and up to 1,024 rows. Tile supervision samples 32 tiles per minibatch. Inactive unit slots are handled with occupancy weighting.
- The residual predictor is initialized to persistence. Minibatches preserve contiguous episode groups for successor supervision and exclude cross-episode or padded transitions.
- Actor/critic NextLat objectives and economic forecasting are disabled.
- Projectors and predictors are excluded from deployment. Low prediction loss can coexist with a representation that barely changes along a trajectory; latent motion, persistence controls, shuffled-action controls and gameplay provide additional diagnostics.

## PPO configuration

- Actor and critic each receive one epoch per rollout wave, using minibatches of 4,096 states. The final minibatch is padded with zero-weight rows.
- PPO clips each active decision's importance ratio independently to [0.80, 1.28], using the asymmetric DAPO band.
- Policy loss sums active component surrogates within each state and averages over genuine states.
- The KL target is 0.03, measured as a component average. Joint action KL is reported separately because a small component average can conceal a larger change across a whole turn.
- Importance-ratio denominators use the log-probabilities stored by the sampler. Replayed likelihoods are numerical diagnostics, not replacement behavior probabilities.
- Hidden matrices use NorMuon; gains, biases, embeddings and logit heads use Adam.
- Base learning rates are 5e-5 for the actor, 1.5e-4 for the critic and 1.5e-5 for the backbone/objective. Adam groups use 0.35 times their base rate; the production value head has an explicit rate of 4.375e-4.
- Learning rates warm up over 32 optimizer steps.

## League system

- A single-learner wave contains 128 mirror self-play games, 64 snapshot-league games and 40 reference-agent games: 232 physical games in total.
- Both mirror seats contribute learner trajectories; only the learner seat contributes in league and reference-agent games. This produces 360 trajectories (about 71% mirror), with 258,840 valid states for complete default games.
- The default selector is `hardness`. Budgets of two active and six historical snapshots are pooled into eight distinct selections rather than enforced as separate strata.
- Matchup evidence comes from current-learner native terminal results, scored as win 1, draw 0.5 and loss 0. A Beta(1,1) prior handles sparse evidence; effective counts decay by 0.98 per wave.
- Hardness lanes select the opponents with the lowest posterior learner score. Exact ties are randomized.
- Two screen lanes spend their games two at a time on stale opponents with the largest (age + 1) × posterior standard deviation, so a forgotten matchup reaches hardness selection within a few waves. While unseen opponents remain, a discovery lane tests the newest untested snapshot.
- Selected opponents receive approximately equal game counts with balanced learner seats.
- Actor snapshots are immutable, match the learner's full model configuration, and are saved initially and after every actor-active iteration. The archive keeps the latest sixteen, thins older ones to a spacing that doubles with age, and always keeps any snapshot the learner scores under 0.6 against.
- Ten public Kaggle agents that react to the game state are split in two (`kaggriculture.opponents`). Five league agents (demand-timing, hybrid-2965, harvest-ledger, master-engine-v53 and bronze-v31) are fixed native lanes. Each gets two games per wave, and the rest are allocated in proportion to (1 − estimated score)². The five held-out agents (demand-preserving, demand-advance4, idle-seller, shepherds-ledger and kaito-v48) are never trained against.
- Engine built-ins (`pass`, `random`, `starter` and `scripted-v27`) remain available as league lanes but are disabled by default.
- The older `stratified` selector uses age strata and PFSP weights; it remains an explicit alternative.
- Population mode uses equal counts of ordered learner pairings, with `N * (N - 1) * 13` physical games by default. All seats are learners, with a distinct BC initializer per member and no frozen, built-in or script lanes.

## Simulation and execution

- Rollouts use the Rust `BatchEnv`; the pinned `kaggle-environments==1.32.7` engine is the parity oracle and final evaluator.
- Default games contain 720 states and 719 action transitions, with two 10×10 farms, 24 turns per day, starting bank 3,000, shed capacity 100 and up to sixteen units.
- Native simulation preserves engine ordering, inventory insertion order, price rounding and private-state behavior. Outcome signs are determined before float32 bank rounding to avoid false ties.
- Collection uses one physical-game shard, with Rayon parallelism across games and sequential transitions within each game.
- The default collection mode is `inductor_graph`: a collector-owned CUDA graph over Inductor-fused learner and frozen-policy forwards. Sampling and native stepping occur outside the graph.
- Training weights are FP32; collection uses a BF16 inference replica and updates use BF16 autocast. Quantity heads remain FP32.
- Update compilation uses `default`. LeJEPA retains actor update activations at minibatch size 4,096.
- Recovery checkpoints are saved every 420 seconds, on panel events and at clean termination. This cadence is separate from per-iteration league snapshots.
- Recovery restores model configuration, optimizers, RNG, league evidence and provenance. Actor-only export supports some older artifacts that cannot resume training.
- Local GPU work runs through `mlq`. The current core campaign uses a thirty-minute job cap and a twenty-seven-minute trainer budget; these are campaign settings rather than universal CLI defaults.

## Evaluation

- Compatible single-learner runs receive a native development panel every 25 actor-active waves. Recovery checkpoints also trigger a separate external panel against the held-out reference agents.
- Final selection uses held-out responsive games, paired maps and seats, explicit decoding modes and exact artifact identities.
- Warmup iterations, actor-active waves and applied optimizer updates are separate exposure measures.
- Critic pre-update expected-score calibration measures value fitting; it does not establish held-out policy strength or three-class draw calibration.
- Evaluators inspect all historical statuses because a final DONE status can overwrite earlier errors.
- September 27 comparisons supported Monte Carlo actor credit and the slower actor learning rate. September 28 frontier results still showed losses against demand-advance4; saturated wins against older references do not establish frontier progress.

## Implementation references

- Defaults and launch contract: `src/kaggriculture/production.py`, `scripts/train_ppo.py`.
- PPO, targets and optimization: `src/kaggriculture/ppo.py`, `src/kaggriculture/outcome_value.py`.
- Architecture and auxiliary objective: `src/kaggriculture/lejepa_model.py`, `src/kaggriculture/lejepa.py`, `src/kaggriculture/entity.py`.
- Observations and actions: `src/kaggriculture/tokens.py`, `src/kaggriculture/actions.py`, `src/kaggriculture/policy.py`, `src/kaggriculture/resource_conditioning.py`, `src/kaggriculture/navigation.py`.
- League and simulation: `src/kaggriculture/league.py`, `src/kaggriculture/rollout.py`, `rust/kagg_env/src/core.rs`.
- Experiment history: `docs/experiments/core-model-2026-09-26.md`, `docs/experiments/jepa-runs.md`, `docs/experiments/runs.md`.
