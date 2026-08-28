# Structured VIT Regression Campaign

## Purpose

This file is the execution ledger for `VIT_NEXTLAT_PROPOSAL.md`. The campaign improves the structured actor without losing attribution and then applies structured NextLat to the best architecture.

The loop is:

1. Freeze a champion.
2. Change one named mechanism.
3. Run the complete diagnostic budget.
4. Evaluate every seed on a fixed, training-disjoint play panel.
5. Promote only a Pareto improvement in play quality and wall time.
6. Build the next challenger from the promoted champion, not from an accumulating unmeasured branch.

All accelerator jobs must be submitted through `mlq`.

## Current champion

`V0` is the selected structured configuration:

```text
architecture       structured
model_dim          80
attention_heads    4
ffn_multiplier     4
farm_blocks        2
opponent_latents   8
latents            32
core_layers        8
quantity_rank      32
actor parameters   1,319,225
critic parameters  1,078,701
```

Evidence anchors:

- BC actor: `runs/ab-structured-s1/bc-actor.pt`
- PPO run: `runs/econ-pastself-100-structured/checkpoint-000100.pt`
- BC holdout NLL: 0.0008223874
- PPO intra-league score: 0.51736
- PPO public-v16 score rate: 0.84375
- BC steady epoch median: 8.86 seconds
- PPO actor-active median iteration: 27.98 seconds
- PPO actor-active rollout median: 40,910 states/second
- PPO actor-active update median: 21.96 seconds

The entity-CNN remains the speed reference, not the quality champion:

- BC steady epoch median: 7.82 seconds
- PPO actor-active median iteration: 21.25 seconds
- PPO actor-active rollout median: 55,156 states/second
- PPO actor-active update median: 16.85 seconds

Every benchmark must be rerun on one frozen source revision after the exact systems work; historical measurements are anchors, not substitutes.

## Standard budgets

### B0: exact systems benchmark

No retraining. Measure both eager and compiled execution at:

- actor forward batch 1;
- actor forward/backward batch 2,048;
- complete BC epoch after compilation;
- complete mixed PPO iteration;
- complete candidate action path in official evaluation.

Record cold compilation, steady median, p95, peak allocated/reserved CUDA memory, and action parity. Use at least 10 steady samples after warmup. An exact change is accepted only when eager and compiled outputs remain within the established BF16 tolerance, fixed-RNG legal actions are identical, and complete wall time improves.

### B1: fast learning regression

Use the existing complete BC schedule, never a shortened epoch count:

```text
datasets           bc-v16-mirror-512
                   bc-v16-vs-starter-512
                   bc-v16-vs-pass-256
                   bc-v16-vs-random-256
seeds/dataset      64
holdout seeds      8
epochs             12
batch size         2048
run length         4
matrix LR          4.2e-3
matrix WD          1.2
Adam WD            0.005
compile             default
architecture       structured
model dim          80
```

Initial architecture screens use training seeds `1,2,3,4`. NextLat screens use the same four first and extend finalists to `1..8`; a promoted NextLat recipe receives a final 16-seed BC replication because its prior effect was basin-dependent.

Evaluate every artifact in the official engine on the screening block `10000..10015`, both seats, against public-v27 and public-v16. Use starter/pass/random only as collapse diagnostics. This is 32 complete 720-step games per principal opponent and seed.

A B1 arm advances when:

- at least half of matched seeds improve public-v27 margin;
- median public-v27 margin improves;
- public-v16 behavior and strategy coverage do not collapse;
- holdout NLL remains finite and within 5% of the champion unless play improves materially;
- steady BC time is reported, including auxiliary overhead;
- no selection uses holdout NLL to choose a NextLat seed.

B1 is a screening gate, not final evidence.

### B2: confirmation BC and disjoint play

For a B1 finalist:

- extend architecture arms to eight training seeds when variance warrants it;
- extend the selected NextLat recipe to 16 seeds;
- evaluate on `20000..20031`, both seats, against public-v27, public-v16, and the current champion;
- select any best-of-K artifact only on the screening block and report its result on this untouched confirmation block.

Promotion requires a replicated play improvement or a clear speed-quality Pareto improvement. A lucky screening seed that fails the confirmation block is rejected.

### P100: PPO gate

Run training seeds `20260813, 20260814, 20260815` with matched rollout/environment seeds:

```text
iterations                    100
critic-only warmup            20
actor epochs                  1
critic epochs                 champion setting
self-play games/iteration     144
league games/iteration        36
minibatch size                2048
episode steps                 720
optimizer                     NorMuon
rollout compile               default
update compile                default
```

Use identical BC-selection rules for candidate and champion. Evaluate checkpoint 100 on `30000..30031`, both seats, against public-v27, public-v16, all matched champion seeds, and all matched candidate seeds.

Report score, mean/median margin, paired intervals, seat split, strategy coverage, total wall time, rollout throughput, update throughput, and time-to-score. Promote on aggregate matched evidence, not the best individual run.

### P500: finalist continuation

Continue the exact P100 checkpoints to iteration 500. Do not restart them. Final evaluation uses `40000..40127`, both seats, against the full fixed panel and cross-play population.

## Phase S: exact performance work

These changes do not alter the learning hypothesis. Implement and benchmark them before launching a large regression matrix.

| ID | Change | Budget | Gate | Status |
|---|---|---|---|---|
| S00 | Stage-profile V0 tokenizer, trunk, decoders, heads, rollout, and update | B0 | Durable per-stage baseline | Pending |
| S01 | Batch own/opponent farm encoding as `[2B,100,D]` | B0 | Parity and lower farm-stage time | Pending |
| S02 | Expand canonical RoPE buffers without per-forward index gather | B0 | Parity and lower trunk time | Pending |
| S03 | Fuse offset categorical embedding tables | B0 | Forward/backward parity and lower tokenizer time | Pending |
| S04 | Cache architecture/provenance-bound BC encoded shards | Cold/warm corpus benchmark | Byte-identical staged tensors and lower launch time/RSS | Pending |
| S05 | Cache immutable quantity sampler arrays | Official action-path benchmark | Identical actions and lower batch-one latency | Pending |
| S06 | Lockstep-batch official accelerator evaluation | 32/64/256 complete games | Identical games and lower panel wall time | Pending |

Adopt passing S01–S06 into `V0-fast`. If an alleged exact change fails parity, move it to a named learning regression or reject it; do not weaken the parity gate.

## Phase G: global memory

All arms start from `V0-fast`. Only the promoted arm becomes the base of the next phase.

| ID | Change from champion | B1 decision | Next step | Status |
|---|---|---|---|---|
| G10 | One economy/farm/town global refresh after core layer 4 | Compare quality and latency to V0-fast | Promote or reject | Pending |
| G11 | One full-global refresh: economy plus units plus opponent summaries | Run as alternative to G10 | Choose G10, G11, or neither | Pending |
| G12 | Two refreshes after layers 2 and 5 using the winning memory composition | Run only if G10/G11 wins | Compare shared versus separate refresh weights by profile first | Blocked on G10/G11 |
| G13 | Split clock/phase from the town token | Run only on the winning refresh topology | Keep only if addressability improves play | Blocked on G10/G11 |
| G14 | DiT-style clock/farm modulation of core residual gates | Alternative to G12, not bundled | Compare cheap conditioning with repeated cross-attention | Blocked on G10/G11 |

The initial global memory excludes raw 100-patch farms. Adding them is allowed only after attention/probe evidence shows the central latents lost spatial information that the direct local path cannot recover.

## Phase R: residual transport and initialization

Use the winner of Phase G, or `V0-fast` if no global arm wins.

| ID | Change from champion | Budget | Status |
|---|---|---|---|
| R10 | Static `latent_read` output reinjection at core layers 3 and 6 | B1 | Pending |
| R11 | One U-shaped layer-2 to layer-6 skip instead of R10 | B1 | Pending |
| R12 | Combine the winning static skip with the winning global refresh | B1 then B2 | Blocked on G/R winner |
| R13 | Zero attention-output and FFN-down projections as an alternative to current 0.1 gates | B1 | Pending |
| R14 | One late MUDD-lite route over `{x0, layer2, layer5, current}` | B1 then B2 | Blocked until a static skip wins |

Do not run R14 if neither R10 nor R11 improves play. Dynamic routing is not a rescue for a useless static path.

## Phase Q: attention projection and block efficiency

Run only after S00 identifies material time in attention projections or block launches.

| ID | Change from champion | Budget | Status |
|---|---|---|---|
| Q10 | Contiguous fused self-attention QKV bank with explicit NorMuon slice semantics | B0 plus B1 | Blocked on S00 |
| Q11 | Fuse market latent and economy contexts into one decoder block | B1 | Pending |
| Q12 | One combined unit context containing central latents and five local patches | B0 profile plus B1 | Pending |
| Q13 | Custom fused ReLU-squared FFN kernel | B0 | Blocked until S00 proves FFN launch/activation material |

Q10 is a learning regression unless optimizer behavior is mathematically preserved. Parameter count alone does not make it exact.

## Phase E: actor speed-quality frontier

Use the strongest architecture after G/R/Q. Each arm changes one capacity variable.

| ID | Change from champion | Budget | Status |
|---|---|---|---|
| E10 | Reduce central latents from 32 to 24 | B1 | Pending |
| E11 | Reduce core depth from 8 to 6 | B1 | Pending |
| E12 | Six core layers plus the winning global refresh | B1 then B2 | Blocked on G winner |
| E13 | Combine the best latent count and core depth only if both independent arms pass | B1 then B2 | Blocked on E10/E11 |

Select a Pareto champion. A faster arm with a small statistically unresolved score change may advance to P100; a slower arm must show a clear play improvement.

## Phase N: structured NextLat and future patches

Implement `StructuredBelief`, factored action entity tokens, and the training-only transition predictor on the Phase E actor champion.

Prediction targets:

- 100 own-farm post-local patch tokens;
- 16 unit decision tokens;
- 10 market decision tokens;
- later, economy tokens and eight opponent summaries as independent additions.

Use separate decision and patch horizons. Keep raw latent-coordinate SmoothL1 disabled.

| ID | Objective | Training seeds | Gate | Status |
|---|---|---:|---|---|
| N00 | Contiguous sampler, every auxiliary coefficient zero | 1–4 | Confirms structured sampler control | Existing V0 recipe covers run-length 4; rebaseline after API change |
| N10 | Decision-decode KL 0.5, horizon 2 | 1–4 | Port the known useful term to structured decisions | Pending |
| N11 | Own-patch normalized feature L1 only, horizon 1 | 1–4 | Tests future patches without decision KL | Pending |
| N12 | Decision KL from N10 plus own-patch loss from N11 | 1–4 | Tests complementarity | Pending |
| N13 | Winning decision/patch objective with patch horizon 2 | 1–4 | Tests recursive spatial dynamics | Blocked on N10–N12 |
| N14 | Add economy-entity future prediction | 1–4 | Independent global-dynamics contribution | Blocked on N10–N13 |
| N15 | Add opponent-summary future prediction | 1–4 | Independent uncertain-opponent contribution | Blocked on N10–N13 |
| N16 | Add all 100 opponent post-local patch targets without opponent-action conditioning | 1–4 | Tests spatial opponent dynamics separately from summaries | Blocked on N15 |
| N17 | EMA target trunk for patch targets | 1–4 | Run only for unstable/collapsed online targets | Conditional |

Before N11, record a fixed calibration batch and select one patch coefficient whose initial patch-loss trunk-gradient norm is 10–30% of the clone-loss trunk-gradient norm. Store the batch identity, both norms, and selected coefficient. Use that coefficient unchanged in N11–N17.

After seeds 1–4:

1. Evaluate every seed; do not rank by training loss.
2. Extend the top two objective recipes, not the top two individual artifacts, to seeds 1–8.
3. Run B2 on the best recipe.
4. Replicate that recipe over 16 BC seeds before claiming a basin-rate improvement.
5. Apply the same best-of-K selection budget to the no-auxiliary champion so selection compute is matched.

Required diagnostics per run:

- unit/kind/quantity decode KL;
- patch L1 over all, changed, and unchanged patches;
- one-step and recursive losses;
- per-type variance, effective rank, cosine similarity, dispersion, and residual magnitude;
- BC holdout metrics and complete epoch timing;
- official-engine off-distribution play for every training seed.

Reject identity copying when unchanged-patch loss falls but changed-patch loss does not. Reject collapse when feature variance/effective rank falls materially with the auxiliary.

## Phase NP: PPO-active NextLat

Run only after one structured NextLat BC recipe passes B2.

| ID | Change | Budget | Status |
|---|---|---|---|
| NP10 | Winning NextLat recipe in BC only; no PPO auxiliary | P100 | Blocked on N winner |
| NP11 | Same BC initialization with decision and patch losses active throughout PPO | P100 | Blocked on N winner |
| NP12 | Predeclared early-only PPO auxiliary schedule | P100 | Run only if NP11 helps early and harms late across matched seeds |

NP10 and NP11 use the same BC seed-selection rule. NP11 draws a separate contiguous auxiliary transition batch alongside each ordinary PPO actor minibatch and combines losses in one optimizer step. PPO policy minibatch ordering remains unchanged.

Measure auxiliary update overhead separately. The predictor remains absent from league actors and inference, so rollout inference throughput should remain identical for NP10 and NP11; only PPO update time may change.

## Phase C: critic efficiency

Critic work is independent of actor representation and starts from the selected actor champion.

| ID | Change | Budget | Status |
|---|---|---|---|
| C10 | Three critic epochs during iterations 0–19, one afterward | P100 | Pending |
| C11 | Critic core depth 6 instead of 8 | Value holdout probe then P100 | Pending |
| C12 | Critic core depth 4 instead of the C11 winner | Value holdout probe then P100 | Blocked on C11 |
| C13 | Critic central latents 16 instead of 32 | Value holdout probe then P100 | Pending |
| C14 | Combine winning critic depth/latents with winning epoch schedule | P100 | Blocked on C10–C13 |

Retain centralized private unit assignments and the distributional value head. Compare next-wave value loss/explained variance and actor time-to-score; same-wave critic fit is not a promotion metric.

## Phase F: final integration

Combine only independently promoted mechanisms:

```text
F0 = exact systems winner
   + global-memory winner, if any
   + static/dynamic residual winner, if any
   + actor efficiency winner
   + structured NextLat winner
   + critic efficiency winner
```

Run one B2 confirmation after combination to detect interactions. Then run P100 with three matched seeds. Only a passing F0 continues to P500.

Do not add deferred modded-nanogpt features during final integration. Every final component must have its own result row and promotion decision in this file.

## Results ledger

Append one row immediately when a run family completes.

| ID | Source revision | Run paths | Seeds | BC NLL | Public-v27 score/margin | Public-v16 score/margin | BC s/epoch | PPO s/iter | Decision | New champion |
|---|---|---|---:|---:|---|---|---:|---:|---|---|
| V0 | Recorded in artifacts | `runs/ab-structured-s1`, `runs/econ-pastself-100-structured` | 1 / 20260813 | 0.0008224 | Recorded external panel | 0.84375 score rate after PPO | 8.86 steady | 27.98 steady | Current anchor | V0 |

Promotion decisions must name the evidence and the rejected tradeoff. “Lower loss” or “faster” alone is not a decision.
