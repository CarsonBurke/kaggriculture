# Structured Transformer Plan for Kaggriculture

## Decision

Do not replace the current spatial U-Net with a conventional image-classification
ViT. Kaggriculture is a typed, partially observable economic game on two small
grids, not a natural-image problem. The target architecture should be a
full-resolution, topology-aware entity transformer with semantic economic tokens,
query-based outputs, and—only after the static model is proven—conditional action
decoding and temporal belief state.

The work should proceed as a sequence of controlled ablations. A larger end-to-end
rewrite would not reveal whether gains came from the spatial encoder, tokenization,
action factorization, memory, or a changed optimization regime.

## Objectives

The model should:

- preserve exact 1x1 tile state for movement and tile actions;
- represent the two farms as distinct spaces rather than channel-stacking
  unrelated coordinates;
- align products, inventories, prices, crops, units, and actions semantically;
- model local geometry and global economic interactions;
- support the exact masked categorical likelihoods required by PPO;
- improve deterministic deployed play, not merely sampled self-play behavior;
- remain practical for 719 actor calls per game, batched self-play, CPU submission
  inference, checkpointing, and frozen-league evaluation;
- make centralized critic inputs lossless with respect to both players' per-unit
  private inventories;
- provide evidence through complete, replicated training runs rather than reduced
  smoke runs.

## Non-objectives

- Do not use 2x2, 5x5, or larger image patches. Patching destroys action-relevant
  tile resolution on a 10x10 board.
- Do not import an ImageNet-pretrained ViT. The input semantics and channel
  structure are unrelated to natural images.
- Do not combine a new spatial encoder, new policy objective, autoregressive
  decoding, recurrent memory, and new reward scheme in one experiment.
- Do not preserve checkpoint compatibility with the old architecture. New model
  formats should fail clearly when given old weights.
- Do not accept self-play score, training entropy, or a short run as evidence of
  submission strength.

## Current System and Confounders

### Existing architecture

The current actor is already a hybrid transformer, not a CNN-only policy:

- input board: `[B, 58, 10, 10]`, comprising 29 own-farm and 29 opponent-farm
  channels;
- spatial U-Net: 612,336 actor parameters;
- entity transformer: 779,280 actor parameters;
- actor sequence: one state token, 100 spatial tokens, 16 unit tokens, and 10
  market tokens;
- total actor parameters: 1,420,361;
- separate actor and centralized critic backbones.

Relevant implementation:

- `src/kaggriculture/encoding.py`
- `src/kaggriculture/model.py`
- `src/kaggriculture/policy.py`
- `src/kaggriculture/rollout.py`
- `src/kaggriculture/ppo.py`

### Misaligned inductive biases

The existing U-Net immediately mixes own `(x, y)` with opponent `(x, y)` as
channels, although those locations have no direct spatial relationship. It also
imposes translation equivariance before explicit position is available, even
though shed access, board boundaries, quadrant purchase order, and unit spawning
have absolute semantics.

The high-resolution U-Net skip means downsampling is not necessarily destructive,
and local convolution remains a useful prior. Removing it is therefore a
hypothesis to test, not an established improvement.

### Larger known bottlenecks

The architecture comparison is confounded by issues that a ViT cannot fix:

1. Training samples actions while `CheckpointAgent` deploys deterministic argmax.
2. The initial market policy heavily favors `STOP`.
3. The selected iteration-30 checkpoint lost all 256 paired-seat games against
   public v27, with mean margin approximately -151,472.
4. The newer run approached 100% market `STOP`, approximately 99% ties, and no
   planting or harvesting by iteration 50.
5. All unit and market logits are computed in one model forward. Earlier sampled
   actions update legality, but later logits do not condition on what was sampled.
6. The critic receives aggregate carried inventory for both players and no unit
   tokens. It therefore loses item-to-unit assignments for both sides, even though
   those assignments determine spatially local feed, fertilize, drop, and shed
   actions.
7. Seventy-two heterogeneous global actor features are projected into one state
   token, obscuring product-aligned economic relationships.

Before architecture results are trusted, the control training pipeline must
demonstrate non-passive deterministic behavior on an external opponent panel.
Actor-only pretraining on legality-projected v27 trajectories is the preferred
cold-start path. RL should then use a fresh critic and optimizer and no persistent
behavior-cloning or KL term.

## Target Architecture

The target is a structured latent farm transformer. It consists of semantic input
tokens, topology-aware fusion, and task-specific output queries.

### Semantic input tokens

#### Own-farm tiles

Create 100 tokens, one per tile. Each token should contain:

- tile kind;
- crop or animal identity;
- yield, age, water, feed, care, fertilizer, and decay state;
- unlocked/locked state;
- row and column identity;
- quadrant identity;
- edge and corner indicators;
- Manhattan distance to the nearest shed-access tile;
- own-farm identity.

Categorical fields should use learned embeddings. Continuous fields should be
normalized using known mechanic bounds and projected through a small MLP. Do not
encode unrelated categorical alternatives as undifferentiated continuous scalars.

#### Opponent-farm tiles

Encode the opponent's 100 public tiles with the same shared tile tokenizer and an
opponent-farm identity. Do not place all 200 tiles into every dense self-attention
block by default. Compress opponent tiles into 8-16 public-farm summary latents by
cross-attention, while retaining the raw tokens as optional keys for output queries
or diagnostic ablations.

This reflects the game topology: the actor cannot act on the opponent's farm, and
cross-player interaction is mainly economic.

#### Unit tokens

Create up to 16 own-unit tokens containing:

- active/main-farmer identity;
- execution-order slot;
- exact position;
- exact per-unit inventory;
- gathered current-tile features;
- gathered north, south, east, and west neighbor features.

Inactive units must be excluded from attention rather than zeroed while remaining
in the softmax denominator. Preserve execution-order identity because unit order is
semantically meaningful.

The actor may include opponent public-unit tokens. The centralized critic must
include both players' units with exact private per-unit inventories.

#### Economic tokens

Create one token for each of the nine products. Align within each token:

- item identity and known base-price/mechanic identity;
- shared market inventory and current price;
- own shed quantity;
- own aggregate carried quantity;
- relevant town demand;
- opponent quantity for the critic only.

Create five seed/crop tokens containing seed count, seed cost, crop timing, current
planted area, and aggregate expected harvest state. Animal identities can either
receive their own tokens or share the corresponding product token when the mapping
is unambiguous.

Add two farm-summary tokens and one clock token for money, unlocked land, current
hires, day/hour phase, elapsed horizon, and remaining horizon. Remove the stationary
runtime-overage feature from learned input unless live inference can vary it.

### Positional and relational structure

Retain axial 2D RoPE for tile and unit coordinates. Non-spatial tokens must use a
distinct non-spatial positional treatment rather than pretending to occupy `(0, 0)`.

Add learned embeddings or attention relations for:

- token type;
- own versus opponent farm;
- same farm;
- same tile;
- relative row and column displacement;
- Manhattan-distance bucket;
- same quadrant;
- shed-access membership;
- unit-on-tile;
- matching product/crop/action identity.

Arbitrary additive attention bias can disable the current Flash-Attention path.
Implement and profile both:

1. Flash-friendly structural embeddings folded into queries and keys.
2. Explicit relation bias when the quality gain justifies its measured cost.

Do not assume theoretical FLOP reduction predicts wall-clock performance.

### Latent fusion core

Recommended starting configuration:

- model dimension: 128;
- attention heads: 4;
- global latent tokens: 32;
- farm-local relational depth: 2 shared-weight blocks per farm;
- latent transformer depth: 8;
- pre-RMSNorm and Q/K normalization;
- ReLU-squared FFN initially, to avoid conflating the encoder test with an
  activation change;
- gated residual or LayerScale initialized near the identity;
- no dropout unless evidence shows overfitting;
- separate actor and critic cores.

Processing flow:

1. Tokenize all observations without convolutional spatial mixing.
2. Run two farm-local relational transformer blocks over each 100-tile farm
   independently, with shared weights and distinct farm identity. Tile-to-tile
   relative displacement, distance, quadrant, and shed relations operate here.
3. Cross-attend the opponent summary latents to the encoded opponent tiles.
4. Cross-attend 32 global latents to encoded own tiles, opponent summaries, units,
   farm summaries, and economic tokens. Token-type, farm, unit-on-tile, and matching
   item relations operate in this input-to-latent attention.
5. Process the global latents with the transformer core.
6. Decode unit queries using global latents plus relation-aware cross-attention to
   their current and neighboring own tiles.
7. Decode market queries using global latents plus economic tokens.
8. Decode critic value atoms from a centralized value query.

The direct local path prevents latent compression from erasing immediate tile
legality or navigation context. The latent core prevents every transformer layer
from paying dense attention cost over roughly 250 raw tokens. The farm-local stage
is required: a single input-to-latent cross-attention followed only by latent
self-attention cannot express the stated raw tile-to-tile topology.

### Actor outputs

The first structured-transformer experiment must preserve the current output
contracts:

- unit logits: `[B, 16, 59]`;
- market-kind logits: `[B, 10, 22]`;
- market quantity context: `[B, 10, quantity_rank]`;
- `market_quantity_kind_gate` with the current selected-kind lookup contract;
- `market_quantity_value` with the current exact-quantity lookup contract;
- `market_quantity_bias` with shape `[22, 100]`;
- exact Rust and Python selected-kind computation for the masked quantity
  categorical over integers 1 through 100.

The quantity parameters are sampler API, not merely internal model details.
`policy.py`, native rollout, frozen ensembles, replay, and submission inference
extract them directly. A later decoder may replace this API, but Stage 1 must not do
so silently.

After the spatial/token architecture is proven, replace unrelated dense action
columns with compositional action embeddings. Unit-action representations should
share opcode, direction, product/crop/animal, and quantity sub-embeddings. Market
actions should share item embeddings with the economic tokens.

### Centralized critic

The critic should use the same token schema but a distinct parameterized encoder.
It must receive:

- both farms' public tiles;
- both players' exact private shed and seed state;
- both players' exact per-unit inventories;
- shared market and town state;
- the same temporal features as the actor.

Retain the bounded HL-Gauss categorical value head until an independent value-head
ablation proves something better. Fixing critic state aliasing is part of the
structured input change, not a value-loss change.

## Controlled Implementation Stages

### Stage 0A: Build the demonstration-pretraining subsystem

Actor pretraining is currently a design note, not an implemented capability. Build
it before using pretraining as a control requirement:

1. Collect complete public-v27 trajectories with observation, raw action, seat,
   episode seed, step, terminal result, and source/opponent provenance.
2. Project every raw action through the exact sequential legality ledger. Invalid
   unit actions become `PASS`; invalid market orders are removed and compacted;
   excessive quantities use the largest legal integer.
3. Store projected factor actions, active flags, legality masks, and the exact actor
   observation fields needed to retokenize data for different architectures.
4. Split training and held-out episodes by environment seed, never by individual
   state, to avoid trajectory leakage.
5. Implement supervised masked categorical losses for unit action, market kind,
   and exact quantity factors.
6. Save an actor-only pretrained artifact bound to dataset, projection code,
   architecture config, source identity, and training schedule.
7. Add explicit loading of a pretrained actor into a new RL run while always
   constructing a fresh critic and optimizer.

Every architecture in an encoder comparison must train on the same projected
episodes with the same number of optimizer examples and the same selection rule.
Do not compare a pretrained U-Net against a from-scratch direct-token model.

### Stage 0B: Build cross-architecture evaluation support

The current mixed rollout requires every frozen opponent to share the learner's
model configuration, and checkpoint selection assumes a common source/calibrated
run. Before Stage 1:

1. Add an explicit architecture identifier and architecture-specific model config
   to actor artifacts.
2. Add an evaluator-side actor registry capable of loading the current hybrid and
   candidate transformer without pretending their configs are equal.
3. Support heterogeneous actor pairs in external CPU evaluation and cross-play.
4. Preserve independent source and calibration provenance for both competitors
   instead of requiring one shared identity.
5. Extend result schemas and selection tooling to compare paired results across
   architecture families without weakening within-run provenance checks.

Training leagues may remain architecture-homogeneous: frozen opponents within one
run must use that run's architecture. Matched experiments should preserve league
categories, counts, sampling rules, and seed schedules, not attempt to load an old
U-Net policy into a structured-transformer frozen ensemble.

### Stage 0C: Establish a valid control

Before comparing encoders:

1. Make deterministic external-panel behavior a first-class training diagnostic.
2. Pretrain the actor using the Stage 0A dataset and fixed matched recipe.
3. Start RL with a fresh critic and optimizer.
4. Complete at least three control seeds with the current U-Net architecture.
5. Evaluate every seed on identical, training-disjoint paired-seat panels.

Do not continue to architecture selection if the control still converges to a
deterministic pass/`STOP` policy.

### Stage 1: Direct tile-encoder ablation

Purpose: test whether the learned U-Net spatial mixing is beneficial.

Implement a no-convolution, per-tile encoder while preserving:

- the current policy heads;
- the current critic loss;
- the current action masks and factor likelihoods;
- the current optimizer and rollout protocol;
- the same Stage 0A projected demonstrations, pretraining example count, and
  pretrained-model selection rule as the control;
- approximately matched trainable parameters;
- approximately matched full-iteration wall-clock or an explicitly reported
  compute mismatch.

Run two variants if resources permit:

- `direct-paired`: 100 coordinate-paired board tokens with separate own/opponent
  projections. This keeps sequence length fixed and isolates the U-Net removal,
  although the coordinate pairing remains semantically imperfect.
- `direct-split`: 100 own tiles plus compressed opponent tokens with explicit farm
  identity. This tests the better representation but changes sequence topology.

Do not interpret `direct-split` alone as a clean CNN-versus-ViT result.

### Stage 2: Semantic economics and global latents

Add product, crop/seed, farm, and clock tokens. Replace the single global feature
projection. Add the 32-latent core and direct local unit path.

Ablate separately:

1. semantic economic tokens;
2. opponent-farm compression;
3. latent core versus dense full attention;
4. explicit structural relations versus Flash-friendly embeddings.

### Stage 3: Conditional action decoder

This is a separate, high-upside hypothesis. Current later action factors do not
condition on earlier sampled choices despite sequential execution.

Implement first for the market queue, then for units:

- causal action-prefix self-attention;
- cross-attention to the frozen state encoding;
- embeddings of previously selected action components;
- parallel teacher forcing during PPO replay;
- cached keys and values during rollout and CPU inference;
- exact storage of every conditional component log-probability;
- exact reproduction of behavior likelihoods during update replay.

A naive implementation can require up to 26 neural passes per turn and invalidate
the current one-forward Rust rollout design. The stage is not complete until the
native sampler, rollout storage, replay parity, compilation behavior, and full
iteration throughput are handled.

The policy-objective choice must be explicit. Compare:

- existing per-component clipping applied to conditional log-probabilities;
- a joint turn-level likelihood ratio;
- a principled grouped ratio for the unit sequence and market sequence.

Do not silently change clipping semantics while claiming an architecture-only
gain.

### Stage 4: Temporal belief state, only if justified

Opponent private inventory makes the problem partially observable, but recurrent
memory is expensive and invasive. First train a supervised probe that predicts
opponent hidden aggregates from current public state versus a short public history.

Only add memory if history materially improves held-out belief prediction or policy
quality. A candidate design is 4-8 recurrent memory tokens updated with a GRU-style
gate. It requires:

- sequence-preserving PPO minibatches;
- episode resets;
- stored pre-step memory;
- burn-in or full unroll semantics;
- recurrent frozen-league snapshots;
- stateful submission inference with reliable reset at step zero;
- explicit tests preventing centralized opponent-private leakage into the actor.

## Experiment Protocol

### Training design

For every promoted comparison:

- use at least three independent full training seeds;
- use identical environment-seed schedules within matched pairs;
- use the same number of collected states and report wall-clock time separately;
- freeze source snapshots and calibration decisions;
- queue all GPU work through `mlq`;
- run complete production-horizon games and complete planned training budgets;
- preserve opponent sampling and league composition unless that is the named
  ablation, using architecture-homogeneous opponents and matched composition rules;
- record parameter count, model FLOPs estimate, peak CUDA memory, rollout
  throughput, update throughput, and total iteration time;
- do not use a reduced model, shortened horizon, or shortened training run as
  quality evidence.

### Evaluation panels

Evaluate each seed using identical paired seats and training-disjoint seeds against:

1. public v27;
2. a fixed panel of historical learner checkpoints;
3. the current baseline architecture from all training seeds;
4. other candidate architectures from all training seeds;
5. deterministic decoding, which is the deployed behavior;
6. stochastic decoding as a diagnostic, not a substitute for deployment results.

Use at least 128 environment seeds per finalist, both seats, with no invalid game
excluded. Report score rate, mean and median margin, paired confidence intervals,
seat splits, strategy coverage, and per-opponent results.

### Primary selection criterion

Use the paired per-seed difference between candidate and baseline on the fixed
external panel. Promote an architecture when the aggregate confidence interval and
replicated seed results support a real improvement, not merely a higher point
estimate from one training seed.

Secondary criteria are:

- worst-opponent score;
- margin lower confidence bound;
- absence of deterministic `STOP`/tie collapse;
- sample efficiency;
- strength per training wall-clock;
- stability across seats and training seeds.

### Deployment and systems gates

Every candidate must pass:

- batch-one, one-thread CPU inference benchmarking;
- action-time p50, p99, and maximum reporting over complete games;
- the one-second submission action timeout with substantial measured headroom;
- full actor bundle build and isolated validation;
- full mixed self-play/league rollout throughput measurement;
- eager and compiled numerical parity;
- behavior-logprob replay parity;
- finite forward and backward tests;
- actual SDPA/Flash dispatch inspection where intended;
- checkpoint, frozen-league, and resume validation for the new model format;
- memory use within the target RTX 5090's available capacity.

## Test Plan

### Tokenization

- exact actor and critic tensor shapes;
- own/opponent farm identity cannot be swapped silently;
- every board tile appears exactly once per farm;
- categorical and continuous bounds are finite;
- inactive unit masking is exact;
- critic receives exact unit-private assignments for both players;
- actor never receives opponent private information;
- coordinate, quadrant, shed-distance, and token-type encodings are correct.

### Architecture

- output contracts match the policy sampler;
- architecture registry loads each supported artifact into the correct actor class;
- heterogeneous external actor pairs preserve independent provenance;
- local unit gathers select the correct current and neighboring tiles;
- non-spatial tokens do not alias board coordinate `(0, 0)`;
- relation encodings are symmetric or directional as intended;
- padded entities cannot change valid-token outputs;
- actor and critic parameter sets remain separate;
- eager and compiled forwards agree within established precision tolerances;
- CUDA attention dispatch matches the intended implementation.

### Policy and replay

- demonstration projection matches the online sequential ledger;
- held-out demonstration splits contain no training episode seeds;
- every compared architecture consumes the same pretrained examples;
- legality masks remain identical to the existing ledger oracle;
- selected conditional log-probabilities replay exactly at unchanged weights;
- autoregressive teacher forcing matches sequential inference;
- cached and uncached decoding produce the same logits;
- deterministic decoding is stable across bundle and local inference;
- PPO ratio grouping and clipping are covered by explicit tests.

### Recurrent extension

- memory resets exactly at episode start;
- hidden state cannot leak between games or seats;
- rollout-time and replay-time memory agree;
- sequence shuffling never breaks temporal order;
- truncated unroll boundaries have explicit gradient semantics;
- actor memory never consumes centralized private fields.

## Repository Changes by Stage

Likely modules affected:

- a new demonstration dataset/projection module and collection/pretraining scripts:
  complete v27 trajectory collection, legality projection, storage, supervised
  losses, held-out evaluation, provenance, and pretrained-actor loading;
- `src/kaggriculture/encoding.py`: semantic token encoding and critic-private
  completeness;
- `rust/kagg_env/src/core.rs` and `rust/kagg_env/src/python.rs`: native encoded
  buffers and eventual conditional decoding support;
- `src/kaggriculture/model.py`: tokenizers, latent transformer, query decoders, and
  model configuration;
- `src/kaggriculture/policy.py`: new output contracts and conditional sampling;
- `src/kaggriculture/rollout.py`: storage, compiled rollout, cached decoding, and
  recurrent state;
- `src/kaggriculture/ppo.py`: replay, conditional likelihoods, possible ratio
  grouping, and sequence minibatches;
- `src/kaggriculture/inference.py`: architecture registry, new artifact format, and
  eventual recurrent state;
- `src/kaggriculture/league.py`: new-format, architecture-homogeneous frozen
  policy pools;
- evaluation and selection tooling: heterogeneous actor-pair loading and
  architecture-aware provenance;
- training, calibration, and bundle scripts;
- corresponding unit, parity, throughput, and end-to-end tests.

Do not add compatibility shims throughout the model. Introduce an explicit new
artifact/checkpoint format and remove obsolete architecture-specific paths once the
new system is selected.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Pure attention loses useful local inductive bias | Preserve direct local tile gathers; test U-Net versus direct tokens before removal. |
| Splitting farms multiplies attention cost | Compress opponent tiles; use a latent core; measure dense Flash attention as a competing implementation. |
| Relation bias disables Flash Attention | Compare query/key structural embeddings with explicit bias using full iteration benchmarks. |
| Better stochastic policy remains passive under argmax | Make deterministic panel performance and strategy coverage promotion gates. |
| Autoregression destroys rollout throughput | Encode state once, cache decoder K/V, begin with market-only decoding, and measure the complete native loop. |
| New critic gains are mistaken for actor gains | Run actor-encoder and critic-input ablations separately. |
| Recurrent memory destabilizes PPO | Require a history-value probe first; preserve sequences and explicit reset/burn-in semantics. |
| One lucky seed drives selection | Use at least three complete seeds and paired external evaluation. |
| Architecture work masks objective/exploration failure | Establish a productive deterministic control before selecting a model. |

## Research Basis

- Relational Deep Reinforcement Learning: <https://arxiv.org/abs/1806.01830>
- Perceiver IO: <https://arxiv.org/abs/2107.14795>
- Graphormer: <https://arxiv.org/abs/2106.05234>
- Stabilizing Transformers for Reinforcement Learning: <https://arxiv.org/abs/1910.06764>
- Data-efficient Image Transformers: <https://arxiv.org/abs/2012.12877>

These papers motivate relational structure, latent structured input/output
processing, explicit topology, recurrent gating, and caution around training pure
ViTs without an appropriate data regime. They do not prove that this design will
beat the current hybrid in Kaggriculture; only the controlled experiments above can
establish that.

## Recommended Order of Work

1. Build and validate projected-v27 demonstration pretraining.
2. Build architecture-aware heterogeneous external evaluation and cross-play.
3. Establish three productive deterministic control runs.
4. Pretrain and compare the parameter-matched direct tile encoder using the same
   demonstrations and schedule.
5. Split farms and add semantic economic tokens.
6. Add and compare the farm-local relational stage and latent fusion core.
7. Promote the strongest static actor/critic only after full paired evaluation.
8. Test market-prefix conditional decoding as an independent ablation.
9. Extend conditional decoding to unit actions if its strength-per-wall-clock is
   favorable.
10. Add temporal memory only after a history-based belief probe establishes value.
11. Recalibrate compilation, train finalists, run the full external panel, and build
   a submission only from an evaluated selected checkpoint.
