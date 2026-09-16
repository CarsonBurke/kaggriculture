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

## Production contract

Executable defaults are authoritative. Production architecture and PPO settings
live in `src/kaggriculture/production.py`; BC settings live in
`scripts/train_bc.py`. Use the entrypoints' defaults, selecting
`--production-model` for a production-compatible BC initializer.

This ledger records experiments and measured results, not an alternative
configuration. Explicit overrides belong to named experiments and must be
recorded with their artifacts; historical recipes must not become launch defaults.

Historical schema-v1 BC evidence (not a valid schema-v2 initializer):

- BC actor: `runs/vit-gqa-ffn2/n16/bc-actor.pt`
- BC terminal NLL: 0.0007995702

The following PPO and throughput measurements belong to the retired V0
configuration with FFN multiplier 4. They remain historical comparison anchors,
not evidence for the production contract above:

- PPO run: `runs/econ-pastself-100-structured/checkpoint-000100.pt`
- PPO intra-league score: 0.51736
- PPO public-v16 score rate: 0.84375
- BC steady epoch median: 8.86 seconds
- PPO actor-active median iteration: 27.98 seconds
- PPO actor-active rollout median: 40,910 states/second
- PPO actor-active update median: 21.96 seconds
- Entity-CNN BC steady epoch median: 7.82 seconds
- Entity-CNN PPO actor-active median iteration: 21.25 seconds
- Entity-CNN PPO actor-active rollout median: 55,156 states/second
- Entity-CNN PPO actor-active update median: 16.85 seconds

Every benchmark must be rerun on one frozen source revision after the exact
systems work; historical measurements are anchors, not substitutes.

## Schema-v2 RL repair verification

The clean cutover uses predictor gate v3 and observation schema v2. Deployment
executes sampled/selected actions verbatim; no standing-weed rewrite remains.
Seed-domain and finite-sample evaluation rules are documented in `README.md`.

Verification:

- CPU suite: 956 passed, 16 CUDA tests deselected.
- CUDA suite: 16 passed in queued job 4874.
- Rust suites: 42 passed; Clippy passes with warnings denied.
- Native oracle: exact state, structured/conv encoding, potential, utility and
  reward parity over 8 full games / 5,752 joint transitions.
- Native binding safety rejects malformed, aliased and stale-schema buffers.
- Task-scoped Python lint passes. Independent static reviews covered learning
  math, gradients, schema privacy/parity, inference and evaluation admission.

Queued source: `b11fce311ed34b6e68ffca2fe31c7513c027b81ae2600669f086e0edf2564590`.
All jobs use normal priority and `maxParallelRuns=1`; unrelated workloads are
not preempted. Queued work is not yet learning or throughput evidence:

| MLQ job | Workload | Output |
| --- | --- | --- |
| 4874 | Full CUDA regression selection, 45-minute deadline — passed | MLQ logs |
| 4875 | Completed 12-epoch BC exception, retained for this RL run | `runs/rl-repair-schema2-bc/` |
| 4876–4878 | Aux off/predictor/enabled, 128+64 games, 11 repeats each, 2-hour deadlines | `artifacts/benchmarks/rl-repair-schema2-*.jsonl` |
| 4879 | Standard P100, seed 20260812, production gates, 8-hour deadline | `runs/rl-repair-schema2-p100/` |
| 4880–4881 | Matched 32-map development panels, both seats, public v27, 2-hour deadlines | `evaluations/rl-repair-schema2-*-development.json` |
| 4884 | Standard P100 retry, seed 20260812, production gates, 8-hour deadline, gated on 4880 success (bypasses OOM-failed bench 4878 that skipped 4879) — failed: mixed-tree launch (snapshot script + live package refused by launcher guard) | MLQ logs |
| 4886 | Standard P100 retry of 4884 with `PYTHONPATH` pinned to the frozen snapshot `src` so launcher and package agree | `runs/rl-repair-schema2-p100/` |
| 4890–4892 | Screening/finalist/starter panels of frozen `checkpoint-000079.pt` — 4890 cancelled by request, 4891/4892 skipped; superseded by 4901–4903 | — |
| 4901 | Screening 32-cluster panel of frozen `checkpoint-000079.pt` vs public v27 (selection evidence for the finalist) | `evaluations/rl-repair-schema2-p100-ckpt79-screening.json` |
| 4902 | Finalist panel of `checkpoint-000079.pt` vs public v27, gated on 4901 | `evaluations/rl-repair-schema2-p100-ckpt79-finalist-v27.json` |
| 4903 | Builtin 16-cluster panel of `checkpoint-000079.pt` vs starter, gated on 4901 | `evaluations/rl-repair-schema2-p100-ckpt79-starter.json` |
| 4901–4903 | Superseded before start by the finished iteration-100 chain below (no attempts ran) — 4901 cancelled, 4902/4903 skipped | — |
| 4909–4911 | Checkpoint-100 chain without snapshot `PYTHONPATH` — 4909 failed the provenance gate (live workspace tree `64758a2e` vs bound `b11fce31`), 4910/4911 skipped | MLQ logs |
| 4917 | Screening 32-cluster panel of frozen `checkpoint-000100.pt` vs public v27 with snapshot `PYTHONPATH` — passed, 64/64 games, valid | `evaluations/rl-repair-schema2-p100-ckpt100-screening.json` |
| 4918/4921 | Finalist attempts with the raw screening report as selection evidence — failed, report carries no `best_output_sha256` binding | MLQ logs |
| 4919 | Builtin 16-cluster panel of `checkpoint-000100.pt` vs starter (screening domain) — passed, 1.0, valid, but not admissible for packaging | `evaluations/rl-repair-schema2-p100-ckpt100-starter.json` |
| 4920 | Selection report without `--best-output` — succeeded but unusable (no frozen-bytes attestation); superseded by 4922 | `evaluations/rl-repair-schema2-p100-ckpt100-selection.json` (overwritten) |
| 4922 | Selection with `--best-output` freezing `ckpt100-selected.pt` (`54681e7b`) — passed, valid | `evaluations/rl-repair-schema2-p100-ckpt100-selection.json` |
| 4923 | Finalist panel of frozen `ckpt100-selected.pt` vs public v27 — passed, 1.0 over 64 games, valid | `evaluations/rl-repair-schema2-p100-ckpt100-finalist-v27.json` |
| 4924 | Builtin development-domain panel of frozen bytes vs starter — passed, 1.0, valid | `evaluations/rl-repair-schema2-p100-ckpt100-starter-dev.json` |
| 4925 | Isolated full-horizon validation of `submission-ckpt100.tar.gz` (`674df801`) — passed | `runs/rl-repair-schema2-p100/submission-ckpt100-validation.json` |
| — | Kaggle submission of `submission-ckpt100.tar.gz` to `kaggriculture` — accepted | `runs/rl-repair-schema2-p100/submission-ckpt100.tar.gz` |
| 5038 | P100 from e2 BC — failed: retain-graph NextLat clip OOM, then incomplete CUDA event timing | MLQ logs |
| 5039 | Retry of 5038 — failed iter 8: CUDA fragmentation OOM during critic persistence diagnostic | `runs/rl-repair-schema2-e2-p100/` |
| 5041 | Retry: expandable CUDA segments, persistence diagnostic only on `diagnostic_gradients`, snapshot `6eed3b76`, 12-hour deadline | `runs/rl-repair-schema2-e2-p100/` |

BC uses the four current v16 64-seed corpora, an 8-seed holdout per corpus,
batch 2,048, run length 4, compiled BF16 and the complete standard optimizer
schedule. PPO keeps the production critic-readiness deadline; an unready critic
fails rather than relaxing the gate. Development panels are diagnostics, not
screening/finalist certification. The benchmark forces enabled auxiliaries only
to measure their cost; production still requires readiness.

The completed 12-epoch job 4875 is a one-off retained initializer for this campaign
by explicit user decision. Future BC runs inherit the CLI epoch default; do not
repeat this job's override or retrain it for this RL run.

## Standard budgets

### B0: exact systems benchmark

Use `scripts/benchmark_ppo_iteration.py` with its production defaults. Override
only the execution mode being compared; hold the remaining settings fixed.
Record cold compilation, steady timings, memory and action parity from the
complete workload, not an isolated forward pass.

### B1: fast learning regression

Run `scripts/train_bc.py --production-model` with the selected input corpora and
output directory. Inherit the training defaults rather than copying hyperparameters
from this ledger. Compare candidate and champion with the same seed and data.
Evaluation domains and panel defaults come from the evaluator; choose the
screening role explicitly when selecting candidates.

A B1 arm advances when:

- matched public-v27 play improves;
- public-v16 behavior and strategy coverage do not collapse;
- holdout NLL remains finite and within 5% of the champion unless play improves materially;
- steady BC time is reported, including auxiliary overhead;
- no selection uses holdout NLL to choose a NextLat seed.

B1 is a screening gate, not final evidence.

### B2: confirmation BC and disjoint play

Freeze the screening-selected artifact before evaluating with
`scripts/evaluate_checkpoint.py --seed-domain finalist --selection-report ...`.
Use the evaluator's reserved domain and panel defaults, not a separate seed range
from this ledger. Reusing screening maps is not confirmation.

### P100: PPO gate

P100 is the named 100-iteration experiment: use
`scripts/launch_production.py --iterations 100` and inherit all other production
defaults. Reuse the chosen BC artifact. Compare candidate and champion with
matched rollout seeds and the evaluator's development panel.

Report score, mean/median margin, paired intervals, seat split, strategy coverage, total wall time, rollout throughput, update throughput, and time-to-score. Promote on aggregate matched evidence, not the best individual run.

### P500: finalist continuation

Continue the exact P100 checkpoint through `scripts/launch_production.py --resume`.
Inherit its continuation budget and production defaults. Do not restart or invent
a separate evaluation seed range; use the screening/finalist workflow above.

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
| VRAM-20260908 | BC `1a92bd83`; RL `2f669120` | `runs/production-vram-bc-20260908`, `runs/production-vram-p100-20260908-r3` | BC default / 20260812 | 0.00178258 | Not evaluated | Not evaluated | 222.02 cold; 26.36 second | Not a matched throughput comparison | BC complete; P100 interrupted by kernel global OOM after iteration 56, recovery at 39; no promotion | — |
| Joint-NextLat-VRAM-20260908 | Dense `dc651682` + corrected critic mask; compact `dc803236` | `artifacts/benchmarks/perf-joint-20260908-production-{dense,compact}.jsonl` | 20260812 | Existing BC initializer | Not evaluated | Not evaluated | — | 17.21 dense → 15.04 compact, warm median | Keep execution changes and user-selected ungated joint learning; no play-strength promotion | — |

Promotion decisions must name the evidence and the rejected tradeoff. “Lower loss” or “faster” alone is not a decision.

### VRAM run recovery, 2026-09-08

Current production residual transports required a new compatible BC artifact
(MLQ 5636, two default epochs over the four current v16 corpora). The CUDA-stream
reuse change itself did not require BC. PPO kept the production 4096-row
minibatch ceiling, BF16, compiled collection/update, learning gates and seed.
The CPU-only external evaluation sidecar was disabled; no play-strength claim
is made from self-play metrics.

Real runs exposed two repaired update blockers: fresh-wave persistence scores
were incorrectly restricted to gradient-diagnostic iterations, and retained
gradient diagnostics were incompatible with donating compiled backward graphs.
The final diagnostic-only compiler variants disable both AOT donation and
Inductor in-place reuse; ordinary minibatches retain their existing policy.
MLQ 5654 passed 138 tests, and 5655 passed the fresh-process repeated-backward
cache regression. Independent review found no remaining actionable issue.

MLQ 5656 passed real diagnostic iterations 26 and 51 and completed iteration 56
before the kernel killed its process in a global host-memory OOM. The workstation
had exhausted approximately 60 GiB RAM and 60 GiB swap, with active swapping
and 10–15% CPU I/O wait; browser/Orca processes were also OOM-killed. This is
not a completed P100 or a clean throughput measurement. The last reported
Monte Carlo-return explained variance was 0.8452.

Verified recovery artifact:
`runs/production-vram-p100-20260908-r3/checkpoint-000039.pt`
(SHA-256 `97d41d8dd5c6746beaae303e103efb96db04db0dfacd7365ba454920dda631e4`).
Resume it under the original frozen `2f669120` source after sustained host-memory
headroom is available; do not retrain BC or bypass checkpoint source identity.
There are 61 iterations remaining from that durable checkpoint. The full
source digest, job chain, validation, and exact continuation command are in
`artifacts/probes/vram-20260907/training-run.json`. No automatic retry remains queued.

### Joint NextLat and execution comparison, 2026-09-08

The user selected ungated joint representation learning. Actor NextLat joins
accepted actor PPO updates after critic warmup; critic NextLat joins critic
updates throughout. Persistence is diagnostic only. Successors and auxiliary
readouts remain stop-gradient. Gradient observation now uses auxiliary-only
branch views during the same combined backward, not repeated parameter VJPs.
Recovery format 14 removes quality-gate state; old format 13 remains actor-exportable
but must not be resumed under the changed training contract.

Fixed critic KL eligibility broadcasting from `[B,1] * [B]` to a per-row masked
mean. The original-code witness (MLQ 5691) measured loss 36.23235 instead of
0.23235 and four nonzero invalid-row gradients. MLQ 5674 passed 333 targeted
tests, including corrected masking, compact/dense losses and gradients, compiled
single-backward observation, CUDA rollout parity, runner warmup transitions,
and recovery. Two independent reviews found no actionable defect.

Matched production benchmarks 5721/5722 each completed six full 192-game,
720-step waves with joint auxiliaries, BF16, compilation, the existing
4096-row ceiling, and `deterministic_training=false` as recorded in the prior
production run. Dense control changes only the critic mask defect; it does not
receive the execution optimizations.

| Measurement | Dense control | Compact/pipelined |
|---|---:|---:|
| Warm iteration median | 17.213 s | 15.043 s |
| Warm update median | 13.096 s | 11.876 s |
| Warm rollout median | 3.774 s | 3.167 s |
| Diagnostic iteration | 41.749 s | 18.789 s |
| Peak live CUDA allocation | 18.449 GiB | 15.189 GiB |
| Peak CUDA reservation | 25.000 GiB | 24.844 GiB |
| First iteration, including setup/compilation | 100.156 s | 109.813 s |

This is 17.7% less peak live allocation and 12.6% less warm iteration time
(1.14x throughput), not a comparable reduction in driver-reserved VRAM.
Caches still reserve nearly 25 GiB; allocator mapping warnings persist in both
arms. Host RSS was approximately 8 GiB in both. No cold-start improvement,
host-RAM-pressure resolution, or play-strength improvement is established.
The actor's unchanged KL stop gives counts `[42,25,24,22,21,22]` versus
`[42,25,23,23,20,22]` as numerical trajectories diverge; every wave performs
57 critic auxiliary updates. Configured minibatch/epoch budgets are unchanged.
Replay KL stays below 0.00114; worst tail fraction stays below 3.64e-6.
The candidate diagnostic measured live source-belief norms 0.08127 (actor)
and 9.67e-6 (critic).

Earlier arms incorrectly added strict deterministic algorithms and are excluded
from production throughput claims. A bounded warm profile (5710) identified
2.69 s in deterministic indexing-backward kernels in one sampled update
minibatch, plus allocator retries. The dense strict arm was allowed to run
too long; later strict arms 5690/5699 were automatically cancelled after their
diagnostic wave exceeded 120 s. Subsequent experiments enforce 300 s cold,
120 s first diagnostic, 90 s ordinary-wave hard limits and reject two ordinary
waves above 45 s, with no automatic retry. Queue waiting is not runtime.

Full source identities, commands, measurements, harness source and rejected-arm
provenance are preserved in
`artifacts/probes/vram-20260907/performance-joint-comparison.json`.
That comparison phase did not launch RL or terminate unrelated processes.

### Cheap diagnostics and completed P100, 2026-09-09

Diagnostics now preserve identical observed/unobserved auxiliary input views,
reduce source cotangents without full-gradient float/square intermediates, and
defer compact preupdate/persistence readback until the final stream join.
Matched full-production jobs 5753/5754 forced gradient diagnostics off/on for
all six waves. Warm update medians were 11.926/12.025 seconds (+0.099 seconds,
0.83%); whole-wave medians were 15.098/15.326 seconds. This is a workload-level
comparison, not isolated kernel overhead: accepted actor-step counts differ
slightly under nondeterministic production execution. Observed actor/critic
source norms were 0.08137/9.73e-6. Production observes every 25 waves.

MLQ 5763 passed 269 contract tests; targeted Ruff passed. Independent review
found one incorrect plateau-guard metric name, fixed before training and
re-reviewed. Temporary execution harnesses were removed after completion;
their exact sources and setup failures remain in the evidence artifact.

MLQ 5769 completed all 100 iterations under frozen source
`5279bdb8fccefdfe921df720521c135ddfac2881a7178bf60b701db3290e21a0`,
using the current canonical production configuration: 128 self-play plus
64 league games, 720 steps, 4096-row ceiling, compiled BF16, seed 20260812,
and the compatible `dadfd6ce` BC initializer. Actor/critic clipping follows
the current repository contract; only predictors retain the configured
NextLat norm ceiling. The actor released at iteration 17, producing 84
actor-active waves, 3313 actor auxiliary steps and 5700 critic auxiliary steps.
The run took 0.498 hours; warm actor-active iteration median was 16.965 seconds.
Recovery checkpoints were scheduled every 300 seconds. No unrelated process
was terminated.

This is a failed learning result, not a promotion. Online mean money fell from
35875 at release to 11.82 at iteration 100, with median zero. Critic value loss
fell from 3.732 to 1.533 while shaped-return explained variance rose to 0.912.
The optional both-signal EMA guard did not cull: improving value loss reset its
patience, leaving 21 stale waves at completion against a threshold of 30.
Lower value loss therefore did not protect against catastrophic policy loss.

Official Python evaluations 5770–5773 used compiled CUDA BF16, fixed batch 32,
the same development seeds 4000000–4000031, and both seats. All 256 games
completed with zero invalid games; these are not CPU submission-parity reports.

| Opponent | Initial wins / games | Final wins / games | Initial mean money | Final mean money |
|---|---:|---:|---:|---:|
| starter | 64 / 64 | 0 / 64 | 149439.05 | 4.69 |
| public-v27 Python reference | 53 / 64 | 0 / 64 | 83465.13 | 5.92 |

Seed-cluster bootstrap 95% intervals for paired money changes are
[-161536, -136530] against starter and [-93874, -73262] against public-v27.
The pinned public-v16 file was unavailable; no native opponent was substituted.
Reject the final RL policy and retain the initializer. Read-only diagnosis
found no demonstrated reward-sign/mask defect; weak early-horizon terminal
credit and a moving current-policy NextLat target remain causal hypotheses,
not established explanations from these losses alone.

Final recovery: `runs/production-joint-p100-20260909/checkpoint-000100.pt`,
SHA-256 `1f2ddbe95ac78ff2e332f4520f80e05452a5ada56bc52e3fc8080b8c8aefd21a`.
Complete launch, diagnostic measurements, paired results, culling state,
reviews and limitations:
`artifacts/probes/vram-20260907/cheap-diagnostics-fullrun.json`.
Per-game reports: `evaluations/production-joint-p100-20260909-{before,after}-{starter,public-v27}.json`.

### Head-only LR and 4800-row batches, 2026-09-09

The equal-rate trial 5783 never released the actor: at iteration 40,
Monte Carlo-return EV was 8.12e-6 and prediction std 2.87e-5 versus target
std 0.467. The readiness deadline stopped it; redundant evaluations of the
unchanged BC actor were cancelled.

The next configuration keeps actor/critic trunk LR at 3e-5 and raises only
the critic value-head Adam LR to 8.75e-5. Other critic Adam parameters and
predictor rates remain unchanged. The user selected a 4800-row physical
minibatch ceiling with headroom rather than pushing the 5120 boundary.
Each full critic epoch now covers all 230080 states in 48 balanced batches.

Two exact memory-lifetime/work reductions accompany it: release prior
minibatch inputs and returned beliefs after last use/existing stream joins;
refine compact NextLat buckets using unused shape slots, retaining every
old boundary so padding never increases and the eight-shape cap remains.
No precision, model-capacity, sample-budget, or accumulation change was made.
MLQ 5794 passed 318 tests, including dense/compact loss and gradient parity
at a non-power-of-two batch size, head-only update isolation, and optimizer
resume. Targeted Ruff and two independent reviews passed.

First launch 5796 was killed by kernel global host-RAM OOM before its first
completed iteration; a CUDA allocation warning preceded termination.
After explicit user approval, only language servers 403230, 1219214, and
1495681 were terminated. Recovery job 5815 resumed the intact iteration-zero
checkpoint and completed repeated full waves with 48 critic updates and
the intended separate head rate. It stopped at the 40-iteration critic-readiness
deadline: EV 0.03701 remained below 0.10; value loss was 3.71237. The actor never
updated. This improves on the equal-rate trial's near-zero EV but does not
establish a successful RL setup. Final panels depend on successful training,
so the unchanged BC policy is not evaluated again. No automatic retry is queued.

Run: `runs/production-head-lr4800-p100-20260909`.
Frozen source: `a1ae27c818f9db9e75119edc48cf16821f6ed7ec66714aad350d210bd8e17df8`.
Evidence and job records:
`artifacts/probes/vram-20260907/head-lr-larger-batch.json`.

### 5e-5 trunks and 5120-row full-run attempt, 2026-09-09

Requested 100 iterations from the same BC initializer, with actor/critic trunk
LR 5e-5, ordinary Adam LR 1.75e-5, and value-head Adam LR 1.458333e-4.
The run-specific 5120 ceiling gives 45 balanced batches; the default stays 4800.
Initial job 5822 suffered kernel-confirmed host-RAM OOM before iteration one.
Recovery 5825 rejected the `latest.pt` alias at the initial-checkpoint safety
gate. Job 5828 resumed the exact `checkpoint-000000.pt`, with two Inductor
compiler workers and recompile diagnostics, without changing training semantics.

Critic return EV reached 0.10443 at wave 30; actor updates began at 31.
During warmup the actor weights were frozen, but its predictor took 45 detached
predictor-only updates per wave. At actor release, PPO and NextLat source
gradients both became active on the actor. Rollout mean entropy rose from
0.17373 at wave 31 to 0.39000 at 40, while mean money fell from 39210.63 to
228.79. Critic combined gradient norm peaked at 318.169 at wave 46, almost
entirely in the trunk. These are pre-step parameter-gradient norms, not NorMuon
update magnitudes; existing diagnostics do not identify the responsible loss.

Job 5828 was cancelled by request after 49 completed waves; the watchdog did
not emit a no-progress cancellation. Last mean money was 42.35 and return EV
0.44347. Latest durable numbered checkpoint is `checkpoint-000046.pt`.
Success-gated final panels 5829/5830 were skipped. No automatic retry is queued.
Joint execution at 5120 completed but repeatedly hit allocator mapping warnings,
so this is not evidence of comfortable VRAM headroom.

Runtime recorded 19 recompile events for first-use batch/gradient-phase and
league-lane variants, not a repeated compile on every wave. League growth
introduced additional compilations later in training; startup-only compilation
is not the current contract. Rollout CUDA graphs are also captured per wave,
separately from Inductor kernel compilation.

Run: `runs/production-lr5e5-b5120-p100-20260909`.
Frozen source: `04be752fd0fe16bb820299a71c6f3c73c0ba491e0b0634ba07b3d2b4f9133159`.
Evidence: `artifacts/probes/vram-20260907/lr5e5-b5120-fullrun.json` and
`artifacts/probes/vram-20260907/lr5e5-gradient-entropy-diagnosis.json`.

### Head-only NextLat contract correction, 2026-09-09

Reference audit found fixed-phase two-state runs nearly excluded daily rollovers:
three production-sized metadata shuffles retained 0, 0, and 1 of 9280 available
rollover transitions. Runs now receive an independent random partition phase
per contiguous validity segment before shuffling, with every primary state
still appearing exactly once. Multi-step eligibility now requires a valid
same-trajectory chain, not merely matching endpoints.

PPO NextLat now supervises only normalized actor unit/market head inputs and the
critic's normalized value-head input. Production enables latent SmoothL1 and
decoded KL at coefficient 1 and horizon 1 for each model. World-state objectives
and their CLI/configuration controls are removed from PPO; BC-only world-feature
experiments remain outside this contract. Checkpoint format 15 rejects old
critic/predictor recovery state while retaining legacy actor extraction.

MLQ 5839 collected a full 230080-state wave and exercised both auxiliary
forward/backward paths in compiled CUDA BF16 on a 5113-row balanced minibatch.
Both source and predictor gradients were finite and nonzero; final readout
parameters received no auxiliary gradients. The sampled wave retained 4930 of
9280 daily-rollover transitions. This check used a BC actor and a fresh critic,
with zero optimizer steps: it is execution evidence, not a learning comparison
or evidence that the previous gradient spikes are resolved. MLQ 5841 passed the
production-shaped temporal coverage regression. No training restart was queued.

Final compiled contract recheck MLQ 5847 also passed. Focused regressions in
5846 passed 95 cases; two inactive-loss dtype mismatches were corrected.
The follow-up 5850 passed all 13 window/legacy-actor cases, and 5848 passed the
predictor/optimizer/auxiliary-RNG recovery round trip. Independent review found
one complete-window keyword-default omission, corrected and covered by the
window checks. All focused failures are resolved; affected Python files pass
Ruff. No learning-performance claim is made.

Evidence: `artifacts/probes/vram-20260907/nextlat-head-contract.json`.

### Corrected head-only NextLat full run, 2026-09-09

User authorized a fresh 100-iteration run from the same BC actor after the
contract corrections. MLQ 5884 used frozen source
`d92c4646ea6510c056ac56b8a148a58a8284d4654f2c6b80976ab43c019cfd1e`.
It retains the previous 5120-row ceiling, 5e-5 actor/critic trunk rates,
1.458333e-4 value-head rate, full 128-self-play/64-league wave, and compiled
CUDA BF16. Critic and independent actor/critic predictors start fresh;
no incompatible training checkpoint is resumed.

The job is exclusive, priority zero, with a two-hour execution deadline,
one attempt, and two Inductor compiler workers. Existing critic-readiness
and both-signal EMA plateau guards remain enabled. Final 32-seed, both-seat
development panels versus Starter and public-v27 are success-gated MLQ
5885 and 5886, each exclusive with a 30-minute deadline and one attempt.
MLQ 5884 failed after its first update, before logging iteration one: the
actor persistence diagnostic still required the removed world metric
`structured_preupdate_patch`. Final panels 5885/5886 were skipped.

Run: `runs/production-nextlat-heads-b5120-p100-20260909`.
Launch command, environment, safety policy, and evaluation jobs:
`artifacts/probes/vram-20260907/nextlat-heads-fullrun.json`.

The diagnostic now consumes `combined`, `latent`, and `decision`. Its focused
regression reproduced the exact failure before the fix; both fresh-wave
persistence tests and targeted Ruff pass afterward. Replacement full run
5901 uses source
`339f014d7d227862f46d971802aa4035baa3684ec50ba22ba8522cd59cd70d74`,
the same training settings and BC initializer, and a fresh run directory:
`runs/production-nextlat-heads-b5120-p100-r2-20260909`.
Success-gated final Starter/public-v27 panels are 5902/5903.
MLQ 5901 was cancelled by request after 30 recorded waves. Actor updates began
at wave 11 (warmup return EV 0.2062 at wave 10). Mean money fell from 37910.01
at wave 11 to 3.5969 at wave 30; critic gradient norm peaked at 53.4293 at
wave 26. Latest durable numbered checkpoint is `checkpoint-000014.pt`.
Final panels 5902/5903 were skipped. Head-only NextLat therefore did not resolve
the observed collapse, although critic readiness occurred earlier.

Rollout spikes at waves 2, 4, and 15 coincided with compilation of new stacked
league-model counts (16–17.5 s versus roughly 2.9 s normally). Actor release
also compiled the gradient-enabled auxiliary graph (47.4 s update versus
roughly 10.7 s afterward). Expandable segments were already enabled; repeated
20 MiB mapping warnings show VRAM pressure remained at the 5120-row ceiling.
Their contribution to individual GPU-idle intervals is not quantified.
The next authorized comparison disables both actor and critic NextLat after
behavior-preserving compilation work; other learning settings stay matched.

### No-NextLat ablation and training freeze, 2026-09-09

MLQ 5927 disabled both actor and critic NextLat, retaining the BC initializer,
schema v2, 5120-row ceiling, learning rates, optimizer, full wave, and PPO schedule.
Source `0dc383188c54c250edbb3f9768bb6668a70792277afe1e5ccfc9447d51f24f6e`
also prewarms balanced league layouts; no ongoing extra padding is introduced.
Fused predictor lifecycle fixes are inactive in this non-fused, no-predictor run.
Matched full-wave compilation evidence is in `league-prewarm-evidence.json`
under `artifacts/probes/vram-20260907/`.

Training was stopped after 51 recorded waves following the user's direction to
investigate persistent collapse and launch no new runs until resolved.
Mean money was 16503.32 versus 38278.76 at actor release (wave 11);
critic gradient norm rose from 2.5674 to 209.0152, almost entirely before the
value head. Final panels 5928/5929 were cancelled before starting. Retained
numbered checkpoints are 000000, 000011, 000029, and 000048.
Run: `runs/production-no-nextlat-b5120-p100-20260909`.
Full provenance and outcomes: `artifacts/probes/vram-20260907/no-nextlat-fullrun.json`.

Source and saved-state audits do not yet establish the shared collapse cause.
Critic CE and Monte Carlo MSE fall while raw trunk gradients rise. Saved Adam
moments localize large gradient magnitudes to early reinjection/residual gates,
but their reconstructed last-step magnitudes decrease rather than explode.
The head's centered effective spectral norm grows 2.28 to 5.74 between
checkpoints 11 and 48; total critic parameter norm stays near 118.
User-authorized read-only checkpoint diagnostics subsequently measured both
activation conditioning and actor-credit direction; all used compiled CUDA BF16
with zero optimizer steps and no checkpoint mutation. No causal fix or restart
is claimed.

Separate source corrections add schema-v3 inventory insertion ranks, repair
fused predictor projection refresh, correct zero-mass categorical rounding, and
use bias-sensitive R-squared for readiness (checkpoint format 16).
The frozen ablation includes none of the schema/sampling/readiness changes.
At release, the observed R-squared would still pass 0.10 in both compared runs;
the readiness blind spot is not the observed release explanation.

Crossed critic diagnostics 5938/5939 used a fixed 5113-state sample from each
230080-state checkpoint-11/checkpoint-48 wave. With checkpoint-11 data, critic
11/48 raw gradient norms were 3.2542/147.2426; with checkpoint-48 data, they were
49.8452/199.6529. The late critic has roughly 9–10x larger initial latent-read
activation gradients. The early critic on late-policy data instead has much
stronger batch alignment (latent-read coherence 0.0206 to 0.3645), despite a
slightly smaller activation-gradient norm. Both learned conditioning and the
data/target distribution contribute. No normalization input collapsed to zero.
Instrumentation matched baseline parameter gradients within 0.33% relative
error. Results: `critic-checkpoint-data11.json` and
`critic-checkpoint-data48.json` in the probe directory.

Full-wave actor-credit diagnostics 5943/5944 compared GAE and Monte Carlo
advantages on the same saved-policy trajectories, accumulating gradients over
all 230080 states without updates. Whole-actor gradient cosines were 0.8849
(checkpoint 11) and 0.7470 (checkpoint 48); market-head cosines were 0.6021 and
0.6158. These measurements do not support a wholesale gradient-sign reversal.
Individual action-credit differences are not causal estimates: state selection
and a finite 320-trajectory Monte Carlo sample confound that interpretation.
The same mixed-play collection's mean money fell 38114.16 to 16228.98.
Results: `actor-credit-checkpoint11.json` and `actor-credit-checkpoint48.json`.

Paired fixed-opponent diagnostics 5954/5955 used the same 64 seeds per opponent,
32 games per seat. Against BC, checkpoint 11 to 48 mean money fell 38882.92 to
14252.47 and terminal log-ratio utility fell +0.1423 to -0.4523. Against Starter,
money fell 56440.09 to 19043.14 and utility fell 1.7016 to 1.1704. Paired
20,000-resample bootstrap 95% intervals for the utility differences were
[-1.0415, -0.1482] and [-0.8399, -0.2181], respectively. These descriptive
panels establish reward deterioration against fixed opponents, not its cause.
BC score fell 0.53125 to 0.359375; Starter score rose 0.90625 to 0.96875, so
win rate and reward magnitude must not be conflated. Results:
`paired-outcomes-checkpoint11.json` and `paired-outcomes-checkpoint48.json`.
Initial launch attempts 5952/5953 failed before model execution because `uv`
tried to create an environment inside the read-only snapshot; corrected jobs
used the existing absolute Python interpreter.

Functional direction diagnostics 5957/5958 computed the NorMuon/Adam arithmetic
on full-wave gradients at the default `highest` float32 matmul precision,
without applying any update or mutating saved state.
Using saved optimizer history, the GAE-derived direction's first-order Monte
Carlo loss changes were -1.3864e-5 at checkpoint 11 and -7.0980e-6 at checkpoint
48. Both are descent directions on those samples; their proposed parameter
displacement norms were 0.0046753 and 0.0045990. This rules out a gross direction
reversal in these specific checks, not minibatch noise, finite-step curvature,
or historical failure. Results: `actor-direction-checkpoint11.json` and
`actor-direction-checkpoint48.json`. Training remains frozen; a shared causal
correction has not been established.

The user then authorized a bounded finite-update diagnostic, not resumed
training: one PPO cycle per disposable GAE/Monte Carlo branch, followed by
fixed-opponent checks, with no checkpoint writes. Final production-precision
job 5973 used checkpoint 11, the same 230080-state rollout and minibatch RNG,
independent CUDA optimizer state, and 45 actor minibatches per branch.
Both began with actor/critic Adam counters 45/495; source counters and rollout
hashes stayed unchanged. The original frozen optimizer was retained to isolate
actor lambda. A held-out panel used 512 games per opponent, balanced across seats.

| Terminal log-ratio utility | Baseline | GAE cycle | Monte Carlo cycle |
|---|---:|---:|---:|
| BC | 0.11712 | 0.13046 | 0.15653 |
| Starter | 1.64969 | 1.70902 | 1.67862 |

All paired 95% bootstrap utility-difference intervals included zero. Fixed-data
Monte Carlo policy loss improved from 0.0105175 to 0.0101985 with GAE and
0.0098041 with Monte Carlo. This one-cycle experiment did not reproduce the
multi-wave collapse or establish a GAE correction. Critic end parameters still
differed by at most 0.00113 between branches; no bitwise determinism is claimed.
Results: `bounded-update-checkpoint11-production-high.json`.
Earlier 5965 used `highest` rather than production's `high` precision and is
retained separately. Attempt 5964 is explicitly invalidated because inherited
PyTorch optimizer loading aliased CPU step counters between disposable branches.
Attempts 5961/5963 failed in diagnostic staging before any optimizer update.

### Verified optimizer normalization defect, 2026-09-09

Read-only saved-momentum probe 5967 found that production's `high` versus
`highest` matmul precision changes Polar Express output by median 0.14–0.20%
and at most 3.20%. That alone does not establish the collapse cause. More
importantly, the additive `1e-6` normalization denominator changed the nominally
scale-independent result by up to 25.79% when an actual saved momentum matrix
was multiplied by 1000, even at highest precision. PPO produces momentum norms
of only a few `1e-6`, so this is an exercised range, not an extreme synthetic case.

`optim.py` now substitutes a denominator only when its norm is zero, preserving
the original 2% spectral safety factor without perturbing nonzero spectra.
The strengthened five-shape numerical regression failed before the fix
(13.5–14.6% relative errors); all 18 Polar Express numerical tests passed after
it, including mixed zero/nonzero batches. The coefficient-count structural
assertion was replaced by observable zero-matrix behavior.

Compiled CUDA probe 5969 checked 436 saved actor/critic matrices across
checkpoints 11/48. Worst rescaling error fell from 25.79% to 0.00516% at
`highest` precision, and from 25.79% to 0.8724% under production's `high`
precision; zero matrices remained exactly zero. Residual TF32 rounding is not
hidden by the correction. Probe 5966 first exceeded Dynamo's eight-entry cache
while alternating precision modes; 5967/5969 retained both compiled variant
sets with a diagnostic-only limit of 32, with no eager fallback.
Results: `optimizer-precision-checkpoints.json` and
`optimizer-normalization-fix-checkpoints.json`.

The normalization defect is fixed and numerically verified. Its contribution
to the money collapse, and prevention of future collapse, remain unproven.
No learning-rate, clipping, GAE, or production matmul-precision change was made.
Full training remains frozen.

### Corrected NextLat run and executable nanogpt investigation, 2026-09-09

The authorized schema-3 initialization (job 6005) completed two BC epochs.
Full NextLat job 6006 used frozen source `22fde8e4ed2d`, both actor/critic
predictors, the normalization correction, and the 4800-row production ceiling.
Actor updates began at wave 12. The correction did not prevent collapse:
36 waves were recorded, with final training-wave mean money 51.83. The job
was cancelled; checkpoint 29 is the latest retained full checkpoint.
Official Starter panel 6013 returned 128 losses in 128 valid games, with mean
candidate reward 0.0625. Baseline panels 6008/6009 were cancelled, so no matched
baseline comparison is claimed. Public-v27 panel 6014 was cancelled to unblock
the requested reference investigation and remains deferred.

Jobs 6015/6020 executed the actual `modded-nanogpt` Polar Express and Triton
Gram/polynomial kernels, variance reduction, Adam, and mantissa-preserving
matrix updates on disposable tensors. Training-module top-level code was not
executed. Saved local rates, moments, and parameter routing were held fixed;
this compares reference kernels, not the complete language-model training
recipe. Reference source SHA-256 identities and executable extraction are in
`artifacts/probes/nanogpt-reference-20260909/`.

Stationary-moment surrogate matrix updates differed by median 4–11%, with an
early-critic maximum near 81%; these are not historical minibatch gradients.
A separate reference AST copy changed only its fixed-epsilon denominator.
That control and the remaining BF16/kernel-path differences both matter;
their errors are not additive. On the same FP32 polar-factor inputs,
local/reference variance reduction differed by less than 2.2e-7 relatively.
No separate variance-reduction arithmetic defect was found.

Frozen-model jobs 6018/6019 differentiated four objectives over a complete
230080-state wave each, using 48 production-sized minibatches and compiled CUDA
BF16 model execution. No model optimizer step or checkpoint mutation occurred.

| Actor gradient quantity | Pre-release checkpoint 4 | Collapsed checkpoint 29 |
|---|---:|---:|
| PPO gradient norm | 0.02915 | 0.01698 |
| NextLat latent gradient norm | 0.46489 | 0.13135 |
| NextLat decision gradient norm | 11.53218 | 1.67090 |
| Decision/PPO gradient norm ratio | 395.62 | 98.43 |
| PPO versus sampled MC gradient cosine | 0.85763 | 0.43164 |
| Combined versus sampled MC gradient cosine | -0.10037 | 0.20951 |

Job 6021 used those actual gradients and saved optimizer history to compute
local/reference updates on disposable parameter tensors. At checkpoint 4,
PPO alone predicted MC-loss changes of -9.7817e-5/-9.7588e-5; adding the
unchanged NextLat terms reversed both to +4.5110e-6/+4.5073e-6. Thus replacing
the optimizer with the actual reference kernels does not remove this sampled
uphill counterfactual. At checkpoint 29, both objectives were descent
directions under both optimizers; the conflict is not universal.

Checkpoint 4 precedes the actual actor release at wave 12. These are
finite-wave, first-order diagnostics at saved rates, not replayed historical
steps, held-out evaluations of changed policies, or proof of multi-wave
causality. Combined raw gradients were reconstructed from separately
differentiated BF16 components, without claiming bitwise combined-backward
identity. The reference NextLat unit weights accompany supervised token
cross-entropy, whereas the local primary objective is advantage-weighted PPO;
matching coefficients does not establish comparable gradient balance.
The no-NextLat ablation already exhibited the same gradient escalation and
collapse. The auxiliary conflict is therefore a separate observation, not a
lead on their shared cause. The proposed auxiliary-weight experiment is
retracted; NextLat remains fixed. Further investigation must explain the
critic/trunk gradient amplification observed without NextLat, using those
existing checkpoints and the actual optimizer reference. No new training,
optimizer transplant, or loss-weight change has been launched.
Full evidence and limitations:
`artifacts/probes/nanogpt-reference-20260909/reference-investigation.json`.

### Full reference-contract audit without NextLat attribution, 2026-09-09

The earlier kernel comparisons imposed local parameter routing, schedules,
moment history, and matrix grouping on both sides. They therefore did not
validate the complete supplied nanogpt optimization contract.

A primary-source audit found a concrete routing defect in both current and
historical no-NextLat code. `optim.py:84-144` declares lookup parameters and
learned queries Adam-managed, but its name-only allowlist omits all sixteen
structured trunk embedding tables, `opponent_queries`, `latent_queries`, and
the critic's `value_query`. Job 6025 reconciled actual model ownership with
saved optimizer state at no-NextLat checkpoints 11/48: eighteen actor tensors
and nineteen critic tensors have NorMuon history instead. This is not an
auxiliary-specific path. MuddLite input/output weights also use NorMuon,
whereas the supplied reference assigns its MUDD controls to Adam; this is
recorded separately as a reference-domain divergence.

With actual reference kernels, a zero-history independent-lookup control
kept one saved-moment surrogate row fixed and changed only the other rows.
That row's NorMuon direction changed by 70.1%/65.5% for checkpoints 11/48;
the Adam row was unchanged. This demonstrates the update coupling introduced
by spectral treatment of lookup tables, not historical training causality.
The Adam control uses explicit .9/.999 betas and 1e-8 epsilon; it exercises
row independence, not the complete reference embedding recipe.

Other material contracts were checked rather than silently transplanted:
the reference retains Adam gradients across alternate matrix steps and steps
Adam only on odd steps; it has per-role betas/rates, .85-to-.95 momentum
warmup, per-head-pair Q/K matrix banks, and a 2x down-projection multiplier.
Local per-minibatch updates and the .008/.023 rate ratio do not reproduce
that whole recipe. No Nesterov sign, second-moment aliasing, or gradient-clear
defect was established. Compiled RMSNorm controls matched reference defaults:
BF16 uses FP32 opmath epsilon, not BF16 storage epsilon.

The matching CleanRL implementation has a non-affine RMS readout followed
by width^-1/2 scaling, unlike the direct learned-RMS local critic readout.
The named historical run reward-normalizes but leaves advantages raw.
Its recorded critic clipping count is zero in 15250 updates, so clipping was
not actively maintaining its critic stability. No erroneous categorical
atom-count factor or self-bootstrapping lambda-one critic target was found.
The exact historical CleanRL shared-source revision remains unverified;
its run settings are recorded, but no immutable source snapshot was found.

Job 6027 extended the earlier crossed-data probe by holding cotangents fixed
across no-NextLat checkpoints. It recreated each complete 230080-state wave
and the archived 5113-row diagnostic batch. Both models ran compiled CUDA
BF16, with highest FP32 matmul precision matching the archived diagnostic.
There were no model optimizer steps or checkpoint mutations.

| Fixed batch | Cotangent source | Late/early logit-VJP norm | Late/early hidden-VJP norm |
|---|---:|---:|---:|
| Checkpoint 11 data | 11 | 16.916 | 5.380 |
| Checkpoint 11 data | 48 | 3.217 | 1.035 |
| Checkpoint 48 data | 11 | 7.413 | 2.446 |
| Checkpoint 48 data | 48 | 10.766 | 3.623 |

Identical logit cotangents remove loss-residual differences; identical hidden
cotangents additionally remove head-weight differences. These results prove
changed directional sensitivity below the head, not a complete condition
number or a causal link from the routing defect to collapse. Diagonal VJP
norms matched the archived ordinary backwards within 2.93e-5 relatively.

Semantic routing is the next concrete correction to isolate, not another
NextLat sweep or indiscriminate clipping/normalization change. A corrected
routing comparison must account for fresh Adam state: historical NorMuon
moments do not contain Adam second-moment history. Production source and
training remain unchanged. Detailed source audits, runtime controls, and
limitations: `artifacts/probes/nanogpt-reference-20260909/shared-reference-audit.json`.

### Semantic lookup routing correction and fresh-state control, 2026-09-10

`route_parameters()` now assigns `nn.Embedding` weights to Adam by module
ownership, including tied weights whose projection alias is registered first.
Direct opponent/latent/value queries also use Adam. Hidden projections remain
on NorMuon. MUDD routing, learning rates, optimizer arithmetic, and objectives
were not changed for this correction. The historical structured model changes
exactly 18 actor and 19 critic tensors from NorMuon to Adam.

The production-role and independent-row/tied-weight regressions both failed
before the fix; all 53 optimizer tests pass afterward. The affected optimizer,
PPO, BC, and PPO-runner suites pass: 257 passed, 17 CUDA-marked deselected.
The BC latent/decision-only test had an obsolete positivity assertion for its
inactive own-patch residual; that assertion was removed, not repinned to zero.
Ruff passes. Independent implementation and experimental-control reviews found
no material issue.

MLQ **6030 succeeded on its sole attempt**. The disposable comparison uses
no-NextLat checkpoint 11 and its schema-compatible historical model/PPO runtime.
Both branches use identical current optimizer arithmetic, model weights,
rollout, RNG/shuffle state, and empty optimizer histories with fresh warmup.
Only parameter routing differs. Both complete one full cycle of **45 actor
and 45 critic updates on 230,080 states**, with no KL early stop.

Each unchanged/baseline/corrected policy receives the same 1,024-game native
panel: 512 BC and 512 Starter games, matched by seed, seat, opponent, and
sampling seed. The following are one-cycle outcomes, not resumed histories:

| Observable | Unchanged | Baseline routing | Corrected routing |
|---|---:|---:|---:|
| BC win rate | 53.320% | 53.906% | 52.930% |
| Starter win rate | 90.234% | 92.383% | 92.969% |
| MC policy-loss diagnostic | 0.010518 | 0.010200 | 0.010178 |
| Critic gradient norm during update | — | 3.082072 | 3.176275 |
| Critic fit explained variance | — | 0.284631 | 0.285068 |

Corrected-minus-baseline paired-seed bootstrap intervals (4,096 replicates,
95% percentile intervals) do not establish a gameplay difference:

- BC win difference: **−0.977 percentage points**, interval [−4.492, 2.734].
- Starter win difference: **+0.586 points**, interval [−2.148, 3.516].
- BC log-money utility difference: −0.02429, interval [−0.10876, 0.06216].
- Starter utility difference: −0.03065, interval [−0.09139, 0.02945].

These intervals describe evaluation uncertainty for the fixed snapshots, not
training-seed uncertainty. Compilation/cache ordering confounds branch timings;
this comparison makes no throughput claim. The corrected critic gradient norm
is not lower. The semantic defect is repaired, but neither immediate gameplay
improvement nor prevention of the historical collapse is established.

Checkpoint and rollout hashes remain unchanged, no model checkpoint was
written, and no continuing training was launched. Historical optimizer state
is not converted: NorMuon history lacks Adam second moments and cannot populate
the corrected partition. Exact historical resumes retain their original source;
adopting this correction requires fresh optimizer state. README records this
compatibility boundary.

Reproducible launch, frozen optimizer implementations, diagnostic harness, raw
outcomes, paired analysis, regression results, and reviews are retained under
`artifacts/probes/nanogpt-reference-20260909/`: `fresh-routing-launch.json`,
`fresh-routing-comparison.json`, `fresh-routing-analysis.json`,
`routing-fix-validation.json`, and `routing-fix-reviews.json`.

### Frozen critic radius and feedback attribution, 2026-09-10

The investigation continues without another NextLat ablation or training run.
The actual `../modded-nanogpt/train_gpt.py` and `triton_kernels.py` match the
previously archived reference byte-for-byte. Source review distinguishes
structural gradient concentration from temporal sensitivity growth: actor and
critic have separate trunks, actor decoders also have non-latent paths, and
large raw norms do not by themselves imply large Adam or NorMuon updates.

MLQ **6037 was rejected**: its arithmetic no-op wrapper changed one parameter
gradient vector by 2.116%, beyond the predeclared 2% instrumentation tolerance.
The replacement uses an exact-identity custom forward and captures reference
radii through the same compiled graph used for interventions. The tolerance
was not loosened. **6039 and 6044 succeeded**, each on its sole attempt, with
exclusive queue admission and a 30-minute bound. They completed 64 broad and
72 fine-grained frozen-checkpoint controls respectively.

Each experiment uses the historical schema-compatible runtime, compiled CUDA
BF16, and the same precision convention as the archived crossed VJPs. A fixed
5,113-state sample from each full 230,080-state early/late rollout is crossed
with both checkpoints and both fixed hidden cotangents. No optimizer steps
occur. Checkpoint hashes are unchanged.

At each selected RMS input, the intervention multiplies only the input
pullback by `r_late / r_early`, using paired per-row/token/head radii including
the actual epsilon. It preserves the late weights, activation directions,
local affine-weight derivative, and forward values between controls. It
therefore tests the inverse-radius factor without replacing the model's
forward function or changing the incoming hidden cotangent. These artificial
backward rules are diagnostics, not proposed training gradients.

All 136 accepted controls preserve their instrumented forward values exactly.
Identity parameter gradients differ from the untouched compiled model by at
most **0.302%** relatively. Instrumentation does change compilation/fusion:
the maximum hidden-feature discrepancy versus the untouched forward is
0.0859375; exact forward equivalence is between intervention cases, not between
the instrumented and untouched kernels. The same identity and all-radius
effects reproduce in the fine-grained experiment.

| Data / cotangent checkpoint | Early norm | Late norm | Late with early radii | Late/early ratio before → after |
|---|---:|---:|---:|---:|
| 11 / 11 | 3.253 | 17.496 | 7.971 | 5.379 → 2.450 |
| 11 / 48 | 142.303 | 147.248 | 66.751 | 1.035 → 0.469 |
| 48 / 11 | 49.841 | 121.912 | 36.673 | 2.446 → 0.736 |
| 48 / 48 | 55.111 | 199.619 | 52.021 | 3.622 → 0.944 |

Restoring early inverse-radius factors reduces the late gradient norm by
**54.4–73.9%** across these crossings. The following single-family effects
interact and must not be added:

| Early-radius restoration location | Late total-norm reduction |
|---|---:|
| Final `value_norm` alone | 19.6–32.7% |
| Value-decoder pre-FFN norm alone | 12.6–22.9% |
| Core output norm alone | 16.2–24.2% |
| Core pre-attention/pre-FFN norms | 25.1–39.3% |

Decoder query-input and QK-radius changes are not the major total-norm
contributors. Nor is a shrinking initial query or reinjection input:
latent-query radius stays near 0.01961–0.01966, while reinjection input radius
stays near 0.0357–0.0360. Those small radii describe baseline geometry, not
the observed temporal growth on their own.

Removing only the normalized-input reinjection feedback into `x0` changes
late total norms by **−0.083% to +0.526%**. Latent-query parameter norms
actually rise by about 0.34–0.98% under that removal. Gradient vectors change
by about 7–9%, so the route exists, but these results do not support it as the
main total-norm amplifier. Large direct gate gradients are a different
derivative and do not measure the strength of feedback through gate values.
Removing only MUDD coefficient-generator input feedback changes late total
norms by less than 0.03%; its direct source-mixing path and forward effects on
stream radii were not removed.

The head measurement is now disambiguated: centered `W` spectral norm grows
**2.26449 → 5.60467**, while folding in the final RMS affine gain gives
**2.28256 → 5.73595**. The previously reported 2.28 → 5.74 already includes
that gain; multiplying it by the learned gain again double-counts it.
Spectral norm remains a possible-gain bound, not the gain of every cotangent.

Two corrections to the supplied mathematical claims are consequential:

- Affine RMS input pullback uses
  `diag(gamma)/r - x (gamma*x)^T / (d*r^3)`, the transpose of the forward
  Jacobian. A nonuniform-gain finite-difference check confirms this.
- Latent-read coherence is 0.0206 → 0.1412 across early/late models on early
  data, but 0.3645 → 0.1922 on late data. The 0.3645 observation belongs to
  the early model, not the late model; no monotonic coherence increase is shown.

Relative to the healthy nanoGPT reference, relevant differences include its
unit-RMS initial residual stream, non-affine norms, zero-initialized MLP down
projections, and softcapped-logit CE. Its `resid_lambdas` multiply the shortcut;
`post_lambdas` multiply the branch. It also uses repeated input injections and
Adam-controlled gates/embeddings. Neither a claimed 100x-per-layer gate effect
nor unqualified `(batch*latents)^2` norm scaling follows from these architectures.

**Conclusion:** inverse-RMS factors in the core and value decoder make a large,
directly measured contribution to the late parameter sensitivity. Final
`value_norm` alone is not the whole explanation; reinjection and MUDD
coefficient-feedback claims are substantially weaker than the radius evidence.
This does not identify which training updates shrank the streams, establish
harmful function-space optimizer steps, or prove the cause/prevention of policy
collapse. Production source and continuing training remain unchanged.

Evidence under `artifacts/probes/nanogpt-reference-20260909/`:
`critic-conditioning-analysis.json`, `critic-conditioning-detail-analysis.json`,
their raw results and launch manifests, `structural-claim-review.json`,
`extended-claim-review.json`, and `affine-rms-formula-check.json`. The rejected
6037 source/results and accepted 6039 source are retained separately.

### Critic unit-RMS forward-model experiment, 2026-09-10

The user authorized a full training run with a residual/RMS architecture change,
not the diagnostic backward rule. `StructuredConfig.critic_unit_rms` is an
explicit opt-in, default false. It changes only the centralized critic:
initialize central latent/value query rows at unit RMS, and normalize the
latent-read output with a non-affine RMSNorm before `x0` and the core. All
remaining norms, branch gates, MUDD, readout, losses, and rates are unchanged.
This targets initial core/decoder residual scale; learned queries and downstream
residual additions can still shrink or cancel. It is not a proven collapse fix.

Actor-only BC loading now allows differences in the three explicitly
critic-only configuration fields while retaining strict actor-field validation.
Full training resume still requires exact model configuration. The run uses
the existing schema-3 BC actor (`production-schema3-bc-20260909/bc-actor.pt`,
SHA-256 `51daf5b72028f29890273aa5089289375651b69ba3a7d102cd475c2d58f3791d`),
a fresh critic and both fresh optimizers, and the corrected lookup routing.
The historical schema-2 collapse is not an isolated causal control.

Compiled CUDA BF16 verification **6063 passed** on a fixed 5,113-state sample
from a complete 230,080-state wave, using production `high` matmul precision.
All three BC actor output tensors match exactly with the critic flag changed.
Critic forward/backward values are finite. Mean radii change as follows at
fresh initialization:

| Location | Default critic | Unit-RMS critic |
|---|---:|---:|
| First core pre-attention input | 0.03562 | 1.00000 |
| Core output norm input | 0.10043 | 1.00507 |
| Value decoder pre-FFN input | 0.03769 | 1.00079 |
| Final value norm input | 0.04558 | 1.00407 |

With the same random hidden cotangent, parameter-VJP norm is 63.5497 versus
1.8428. This is an initialization diagnostic, not a learning result or a
reproduction of the earlier late-checkpoint radius intervention. The head
remains zero-initialized; its categorical-loss gradient is finite and nonzero.
No optimizer updates or model checkpoint writes occurred in verification.

Earlier probe attempts are retained as failures, not evidence: 6057 supplied
different actor configurations to a collector that correctly requires identical
ones; 6059 passed centralized critic inputs to the actor parity forward; 6061
mixed initialization devices and failed exact actor parity. The accepted probe
matches production initialization, checks parameters and nonpersistent buffers
before parity, and executes every model forward/backward on compiled CUDA.
Dependencies 6058/6060 were skipped. Focused regression job **6062 passed all
three tests**, including compiled BC policy preservation and rejection of an
actor-affecting warm-start mismatch. Scoped Ruff passes. Two independent static
reviews found no blocker and explicitly distinguished initial geometry from
guaranteed long-run conditioning.

Full training **6064** requests **100 waves**, 128 self-play plus 64 league games,
720 episode steps, seed 20260812, and a 5120-row minibatch ceiling. Actor and
critic each use one epoch. Both NextLat objectives remain disabled, matching the
no-NextLat investigation; this is not another auxiliary ablation. Collection is
Inductor CUDA-graph BF16 and updates are compiled BF16. Existing safeguards are
unchanged: minimum ten critic-only waves, prior-wave MC R-squared >=0.10 with a
40-wave readiness deadline, replay-parity gates, and the online-proxy autocull
(20 actor-active warmup waves, alpha 0.1 EMA, 30-wave patience; money +1000 or
value loss -0.01 resets patience). The value-loss proxy can improve while the
policy deteriorates, so it is not an external-strength guarantee.

Run directory: `runs/production-critic-unit-rms-p100-20260910`.
Frozen source digest:
`0911562a8efac6bfcffbdd5811c077967ae6e3ecd3856ba438bb57350891fc7f`.
MLQ uses exclusive admission (`--max-parallel-runs 1`), one attempt, default
priority, and a two-hour training bound. Four matched initial/final policy panels
are queued after terminal training state: **6065–6068**, Starter and public-v27,
32 development seeds in both seats per panel, compiled CUDA BF16, each exclusive
with a 30-minute bound. Per-checkpoint external workers are disabled; these
panels run through MLQ even if training is culled.

Launch manifests, accepted and rejected probe sources/results, regression logs,
and independent reviews are under `artifacts/probes/critic-unit-rms-20260910/`.

Startup verified: wave **1 completed all 230,080 states and 45 critic updates**.
The actor is correctly frozen during the minimum warmup (`actor_updates=0`).
Critic gradient norm is 1.37804 (trunk 0.01656), value CE 4.51277, and the
replay-parity breach flag is zero. This establishes live full-wave training,
not actor readiness or policy improvement. Initial cold compilation is included
in the 268.71-second first wave. `startup-proof.json` preserves the complete
record; the 100-wave training job continues under its existing guards.

**Outcome: failed critic fitting, not successful stabilization.** Job 6064
stopped at the existing 40-wave readiness deadline, with MC-return R-squared
-0.0002493 against the required 0.10. All 40 recorded waves had the actor frozen:
**zero actor optimizer updates**. The absence of policy collapse therefore
describes retention of the BC policy, not stability under policy learning.

Value CE was 3.65169 at wave 5, 3.64618 at wave 10, and 3.64402 at wave 40.
At wave 40, scalar value prediction standard deviation was only 1.0744e-5
against target standard deviation 0.41609. Total/trunk critic gradient norms
were 0.09789/0.00603. The old no-auxiliary critic already had prediction standard
deviation 0.18670 at wave 10 and released the actor at wave 11; the earlier
schema-3 auxiliary run released at wave 12. Those are contextual comparisons,
not matched causal controls. Raw CE across the runs also sees different target
distributions and cannot by itself establish relative fit quality.

The assistant's subsequent suggestion that lower gradients/no collapse were
encouraging was made without inspecting this trajectory and is retracted.
The candidate failed to learn useful state-dependent scalar values. A plausible,
unproven mechanism is that unit-scale learned query shortcuts overwhelm the
state-dependent branch contributions with unchanged 0.1 branch gates; the
initialization VJP reduction did not distinguish conditioning improvement from
loss of useful sensitivity. Do not treat this as a validated fix, raise rates
merely to recover the old norm, or bypass readiness to train the actor.

All evaluation jobs are terminal: initial panels 6065/6066 failed, final panels
6067/6068 succeeded. They cannot demonstrate policy improvement with zero actor
updates. No replacement training was launched. Quantitative trajectory summary
and the readiness failure traceback: `value-fit-failure.json` in the probe
directory.

### Frozen cause attribution for failed unit-RMS critic, 2026-09-10

The user requested determination of the failed critic's cause, not another
training run. **The supported proximal mechanism is dominance by constant
learned-query residuals:** the architecture attenuates state-dependent features
at both the central latent read and the value read. Its head subsequently fits
an almost state-independent target distribution. Equal unit RMS is not equal
state information: the actual nanoGPT reference starts its residual from
`embed(input_seq)` (line 1531) before normalization (1549), whereas these critic
queries are identical across batch states.

MLQ **6069** completed 30 frozen controls on checkpoints 0/5/40. An independent
review correctly identified that its global tracing tolerance was inappropriate
for a nearly constant representation. **6076** therefore repeated the factorial
without tracing and explicitly audited centered tracing error. The initial
confirmation job 6073 was cancelled before admission so uninstrumented precision
controls could be included in the same replacement experiment.

Both successful jobs used one complete 230,080-state wave from the unchanged BC
policy and its fixed 5,113-state diagnostic sample, compiled CUDA BF16 with
production `high` matmul precision. All checkpoint hashes remained unchanged.
There were zero optimizer steps and no model checkpoint writes. Queue admission
was exclusive, default priority, one attempt, with 40-/30-minute limits for
6069/6076. No replacement training was launched.

**Accepted uninstrumented factorial:** independently scale central and value
queries by 1 or 0.02 and retain/remove core-entry normalization, with every other
saved parameter and input fixed. These rescalings preserve query directions;
they are not exactly the original Gaussian initialization or retrained models.
At checkpoint zero, with core-entry normalization retained:

| Query magnitude intervention | Across-state hidden RMS variation | Conditional head-gradient covariance norm |
|---|---:|---:|
| Neither: both unit scale | 0.00058223 | 0.00008276 |
| Central queries ×0.02 only | 0.00473537 | 0.00086780 |
| Value query ×0.02 only | 0.00693091 | 0.00114679 |
| Both queries ×0.02 | 0.15858161 | 0.02889257 |

The combined query change raises state variation **272× at initialization**,
252× at checkpoint 5, and 233× at checkpoint 40; corresponding conditional
head-gradient covariance grows **349×, 339×, and 259×**. Removing only core-entry
normalization at unit query scale leaves the representation nearly constant.
Thus the two query magnitudes interact strongly; the extra core normalization
alone does not explain the failure.

For hidden features `h` and categorical logit residuals `delta`, the head's
real-arithmetic mean gradient decomposes as
`mean(delta) outer mean(h) + Cov(delta, h)`. Initially, the state-dependent term
is only **0.00564%** of the common term in norm, versus **2.001%** with both queries
scaled down. These are FP32 arithmetic diagnostics, not exact BF16 backward or
Adam-update decompositions. Sixteen label-permutation controls establish a
descriptive alignment comparison, not independent-sample confidence intervals;
the reported hundreds-fold change is absolute covariance magnitude, not a
hundreds-fold signal-to-noise improvement.

The fitted centered value-head matrix has **99.9963% / 99.9934%** of squared
singular-value energy in one component at checkpoints 5/40. Its weight energy
along `sign(mean(h))` is **99.9679% / 99.8484%**. This matches learning primarily
from common features under coordinate-wise Adam. Rank one alone is not a bug:
here the direction is nearly constant across states and scalar predictions
remain nearly constant. On the fixed sample, checkpoint-40 CE is **3.56408**;
the best constant categorical prediction has CE **3.55085**. The critic mostly
learned the marginal distribution rather than conditioning on observations.

**Alternatives tested or excluded:**

- All 1,800 critic optimizer updates ran; saved query, branch, gate, and head
  parameters changed. Static tracing found no actor-warmup skip/detach, zero-LR,
  target/state-index mismatch, or optimizer-ownership defect on the critic path.
- Uninstrumented FP32 residual-path controls retain BF16 projection GEMMs but
  also change downstream normalization precision. They do not rescue scalar
  predictions or R-squared. At checkpoint 40, prediction std remains about
  4.6e-5 on this fixed sample; precision alone is not the explanation.
- Final RMS's continuous pullback retains about 61% of the corresponding
  unprojected norm in the preliminary arithmetic diagnostic, not approximately
  zero. Nearly perfect alignment with `sign(mean(h))` must not be confused with
  radial alignment to `h`; final-RMS radial annihilation is not supported.

**Measurement validity:** the original tracer failed state-sensitive parity:
centered hidden-vector errors were **94.6%, 87.9%, and 71.6%** at 0/5/40.
Its small-signal estimates are discarded, not excused by the passing global
norm tolerance. All factorial numbers above instead use uninstrumented forwards.
Whole-model logits and a separate readout of the uninstrumented hidden features
agree exactly in all 24 factorial cases. The second independent review accepted
this replacement evidence and the bounded proximal conclusion.

The architectural error was treating a large input-independent query shortcut
as equivalent to a healthy unit-scale input-dependent residual. A corrective
design should preserve substantial state information at the read boundaries,
not merely reduce raw gradients. Frozen rescaling does not retrain the already
marginal-fitting head, prove future readiness, establish long-run policy
stability, or isolate every optimizer-history contribution to the 40-wave
failure. No production source was changed during this investigation.

Evidence: `failure-analysis.json`, `failure-confirmation.json`,
`failure-confirmation-complete-launch.json`, `failure-localization.json`, and
`failure-reviews.json` under `artifacts/probes/critic-unit-rms-20260910/`.
Preserved executable sources: `localize_failure_6069.py` and
`confirm_failure_6076.py`; the former is retained as historical instrumentation,
not accepted small-signal evidence.

### State-dependent critic reads and full training, 2026-09-10

The user authorized full implementation, training, and outcome evaluation of the
state-carrying residual correction. The active `critic_unit_rms` experiment is
replaced by opt-in `critic_state_read`. Only the critic's central latent read and
value decoder use `Block(state_read=True)`: queries address context, but there is
no query residual shortcut or attention gate. Non-affine RMSNorm of the attention
output supplies the residual content before the existing gated FFN. Core-entry
normalization is retained, and central/value query addresses still initialize at
unit RMS. The read's attention is an input projection, so `zero_init_branches`
cannot zero it; FFN residual zero-initialization remains honored. Actor and other
blocks, objectives, learning rates, optimizer routing, and safeguards are unchanged.
Historical experiment checkpoints retain their original frozen runtimes; no alias
silently reinterprets the obsolete model contract.

Compiled CUDA BF16 verification **6081 passed**, using a full 230,080-state
BC-policy wave and its fixed 5,113-state sample. Initialization now explicitly
matches the runner's actor-then-critic RNG order. Uninstrumented state-dependent
value-hidden RMS variation is **0.27842**, versus **0.17159** for the original
critic; hidden RMS is 0.99998/0.99778. The FP32 diagnostic conditional/common
head-gradient norm ratio is **0.03773 / 0.02398**. These pass the predeclared
0.05 state-variation and 0.005 covariance-ratio floors, but are not learning
evidence. Whole-model and split-readout logits agree exactly. The BC actor's
three output tensors also match exactly with the critic-only flag changed.

A disposable nonzero value head exercised actual compiled HL-Gauss CE backward:
all observed gradients are finite, and gradients reach both query addresses,
the state encoder, central read, core, and value read. This is a connectivity
check, not optimizer training; the real run retains its zero-initialized head.
No optimizer updates or model checkpoint writes occurred in verification.
Regression job **6082 passed five tests**, including single-context invariance
to the query vector, preserved context dependence with both zero-init settings,
compiled actor warm-start parity, and configuration parsing. Scoped Ruff passes;
two independent static reviews found no blocker.

Training **6083** requests the full **100-wave** schedule: 128 self-play plus
64 league games, 720 episode steps, seed 20260812, a 5120-row minibatch ceiling,
one actor and one critic epoch per wave, and no NextLat objectives. The existing
schema-3 BC actor initializes only the actor; critic and optimizer states are
fresh. Collection uses Inductor CUDA graphs and BF16; updates use compiled BF16
with production `high` matmul precision. Minimum ten critic-only waves, MC
R-squared >=0.10 by the 40-wave readiness deadline, replay-parity checks, and
the existing online-proxy autocull are retained without bypasses. MLQ admission
is exclusive, default priority, one attempt, with a two-hour bound.

Run: `runs/production-critic-state-read-p100-20260910`.
Frozen source:
`e52c06abebd648288763d832728400cbe1d8d17d2374313454d47100cb4533c7`.
The run will be followed through terminal outcome, including confirmation of
actual actor updates; startup or frozen-policy retention is not stability proof.
Official initial/final panels will use this run's immutable checkpoint zero,
not the older BC artifact, so both policies are evaluated under their bound
source identity with matched seeds/opponents. Earlier 6065/6066 failures were
source-identity mismatches, not game outcomes.

Evidence under `artifacts/probes/critic-state-read-20260910/`:
`launch.json`, `verification.json`, `verify_state_read_6081.py`,
`regressions.json`, and `reviews.json`.

Actor release is confirmed at **wave 23**, with **45 actual actor updates**.
The preceding wave's MC R-squared was **0.120264**, above the unchanged 0.10
readiness threshold. Wave 23 reports MC R-squared 0.108046, value prediction
standard deviation 0.112907, and critic CE 3.477076. This clears the failed
unit-RMS run's frozen-actor readiness failure; it does not yet establish policy
improvement or resistance to later collapse. Full release metrics are preserved
in `actor-release.json`.

Training **6083 completed all 100 waves**, exit 0, in approximately **29.2
minutes** of runner time. It performed **3,510 actor updates** over waves
23–100 and **4,500 critic updates**. Final checkpoint:
`checkpoint-000100.pt`; terminal metrics/job details:
`artifacts/probes/critic-state-read-20260910/training-outcome.json`.

This is **not a resolved learning-stability result**. Final MC R-squared is
0.724100 and critic CE is 2.115759, but the critic trunk gradient norm grows
from 0.718574 at actor release to 63.445101 at wave 100, peaking at 74.452112
on wave 99. Head gradient norm remains 0.116397 at wave 100. These are
state-weighted averages of per-minibatch pre-step norms, not cumulative
gradient sums. Online mean money falls from approximately 48k before actor
release to 10,234 at wave 100; changing seeds/opponents prevent treating that
as a matched evaluation. Improving critic loss kept the existing autocull
from stopping the economic regression.

Official matched evaluation jobs **6086–6089** compare immutable checkpoints
0 and 100 against starter and public-v27, each on development seeds
4,000,000–4,000,031 in both seats (64 games per panel). They retain compiled
CUDA BF16, exclusive admission, default priority, one attempt, and a 30-minute
limit each. `evaluation-launch.json` preserves the exact commands. Frozen
diagnostic **6091** compares uninstrumented compiled CE gradients on the same
full-wave-derived sample across checkpoints 0, 15, 34, 70, and 100, with no
optimizer steps or checkpoint mutations, to localize the user's reported
late trunk-gradient growth.

**Official evaluations 6086–6089 all completed successfully**, with zero
invalid games across all 256 games. Both policies beat starter 64/64 and lose
to public-v27 64/64, but score saturation conceals a severe economic regression:

| Opponent | Initial mean money | Final mean money | Initial mean margin | Final mean margin |
| --- | ---: | ---: | ---: | ---: |
| starter | 147,247.55 | 7,338.70 | 143,764.14 | 3,850.41 |
| public-v27 | 17,143.48 | 7,358.28 | -112,543.34 | -131,820.13 |

Matched seed-cluster mean money changes (final minus initial), with approximate
paired normal 95% intervals over 32 independent seeds, are **-139,908.84
[-152,020.60, -127,797.09]** against starter and **-9,785.20
[-11,895.35, -7,675.06]** against public-v27. These are development diagnostics,
not held-out model-selection or CPU submission-admission evidence. Exact panel
paths, artifact identities, paired changes, and methods are preserved in
`evaluation-comparison.json`; job outcomes in `evaluation-diagnostic-outcomes.json`.
The state-read experiment recovered critic readiness but **did not prevent
actor economic collapse**. Do not promote its final policy on critic-loss or
unchanged saturated win-rate evidence.

**Frozen gradient diagnostic 6091 passed.** Uninstrumented, compiled CUDA BF16
backward on the same 5,113 states from one full 230,080-state BC-policy wave
reproduces late gradient growth without optimizer steps or changing data:
total norm is 0.386780 at checkpoint 15, 7.712196 at checkpoint 34, 88.987339
at checkpoint 70, and 127.640598 at checkpoint 100. All observed gradients
are finite and checkpoint SHA-256 identities remain unchanged.

At checkpoint 100, `trunk.opponent_queries` has norm 89.578346 and accounts
for **49.25% of squared total gradient norm**. Central-read attention
`key_value.weight` (47.525230) and `output.weight` (41.104233) contribute another
**24.23%**. These parameter scales remain broadly stable: opponent-query RMS
is 0.020025 initially and 0.018990 finally; both central-read matrix RMS
values stay near 0.065. Thus gross parameter shrinkage does not explain the
growth. This localizes the affected paths; it does **not** measure
pre-normalization activation cancellation or prove a particular Jacobian
amplification mechanism.

Fixed BC-data CE worsens from **3.512578** at checkpoint 15 to **5.174014** at
checkpoint 100, despite the improving changing-policy training CE. Critic
forgetting/distribution shift and actor regression therefore coexist with the
gradient growth; this diagnostic does not establish their causal direction.
Evidence: `late-gradient-localization.json`, `late-gradient-analysis.json`,
`late-gradient-launch.json`, and archived `localize_late_gradients_6091.py`.

### Critic gradient growth and money collapse are two separate mechanisms, 2026-09-10

The user asked for an analytic investigation of the huge critic gradient norms
and the money collapse across the last runs, against `../NextLat` and
`../modded-nanogpt`. **They are not the same failure and neither causes the
other.** The gradient growth is an inverse-radius readout that the optimizers
discard; the money collapse is a sign-definite reward-shaping/credit-window
defect in the actor objective. Three frozen diagnostics were run; no production
source was changed and no training was launched.

#### Gradient magnitude does not set the step size -- but the rate ratio does

**Superseded in part.** The three controlled runs below show the missing
per-parameter Adam rate multiplier was a real and harmful misalignment, worth
about 20% of the growth exponent. "Not a step-size hazard" was too strong.

Both critic optimizers discard gradient magnitude. NorMuon pre-scales each
matrix by its own norm (`src/kaggriculture/optim.py:129-163`) and the Adam group
runs `eps=1e-10` (`optim.py:250-251`); nothing clips
(`optim.py:1-9`, `src/kaggriculture/ppo.py:582-586`). `../modded-nanogpt` makes
the same choice deliberately -- no clipping anywhere, NorMuon plus Adam, a token
**sum** loss -- and instead bounds scale by construction: unit-RMS residual
entry, non-affine `norm()` before every read, bounded gates (`2*sigmoid`,
`sigmoid`, `tanh`) with zero-init gate weights, zero-init branch output
projections, QK RMSNorm, and softcapped-logit CE `23*sigmoid((z+5)/7.5)`
(`modded-nanogpt/train_gpt.py:952,1106,1156,1291-1308,1549,1594,1690`;
`triton_kernels.py:1208-1213`). `../NextLat` takes the opposite route --
affine norms, std-0.02 init, no softcap, no gates -- and buys stability with
`grad_clip 1.0`, Huber against detached targets, and `wd 0.1` on matrices
(`NextLat/defaults.yaml:117-131`, `models/model_base.py:206,363-381`,
`models/model_nextlat.py:303`). This repo has adopted *neither* discipline:
no clipping **and** no softcap, no weight decay on any critic group, a
sub-unit-RMS residual seed, and unbounded gates.

#### What actually produces 330x (jobs 6096, 6098)

Eager autocast BF16 backward on the same fixed 5,113-state sample from one
complete 230,080-state BC-policy wave reproduces job 6091's compiled numbers to
2%: total gradient norm **0.3823 -> 126.5** across checkpoints 15/34/70/100
versus the compiled 0.38678 -> 127.64. The growth factors exactly:

| Factor | ckpt15 -> ckpt100 |
|---|---:|
| Head cotangent `dL/d value_hidden` | x6.580 |
| of which `value_head.weight` norm 3.3106 -> 18.9159 | x5.714 |
| of which logit residual norm 39.93 -> 43.05 | x1.078 |
| Trunk parameter-VJP at one fixed random cotangent | x3.023 |
| Cotangent alignment with the amplified subspace | x16.629 |
| **Product** | **x330.8** |

The dominant factor is the alignment term, and it is now localized. Per-token
radii at `trunk.latent_context_norm` (160 tokens, tensor RMS a flat 2.02):

| Read-context segment | Tokens | Radius ckpt15 | Radius ckpt100 | Cotangent mass ckpt15 | ckpt100 |
|---|---:|---:|---:|---:|---:|
| `own_tiles` | 100 | 2.32111 | 2.31767 | 0.57299 | 0.05484 |
| `opponent_summary` | 8 | **0.03829** | **0.03334** | 0.10517 | **0.44266** |
| `own_units` | 16 | 1.29783 | 1.34379 | 0.06321 | 0.00465 |
| `economy` | 20 | 1.04053 | 1.03933 | 0.19426 | 0.49188 |
| `opponent_units` | 16 | 0.91814 | 0.94637 | 0.06437 | 0.00597 |

RMSNorm normalizes each token separately, so the realized backward gain is the
cotangent-mass-weighted mean inverse radius. Measured gain **8.541 -> 19.175**
matches that weighted quantity **8.593 -> 19.356** to 1%: the growth is the
critic's error migrating off the radius-2.32 tile tokens onto the radius-0.033
opponent-summary tokens (mass 0.105 -> 0.443, peaking 0.606 at checkpoint 70).
The 17.25-of-160 tokens at exactly zero radius are masked unit slots; they carry
no cotangent mass and drive nothing.

That channel then hits a second amplifier. `trunk.opponent_queries` is
`torch.randn(8, 80) * 0.02` (`structured.py:936-939`), and `opponent_summary` is
a default `Block`, so the queries are both the attention query and the residual
seed. Its own affine `attention_norm` sees fp32 input at radius **0.02020 ->
0.01899** with resolved eps 1.19e-7, giving a measured backward gain of
**50.51 -> 55.17** -- essentially constant, so it sets the *level*, not the
growth. Composite path gain to the queries is therefore about 30 x 52 relative
to a unit-RMS token/query path, and `trunk.opponent_queries` rises from
**1.26% to 49.16%** of squared total gradient norm.

`critic_state_read` converted `latent_queries` and `value_query` to unit RMS
(`structured.py:941-948`, `1279-1285`) and left `opponent_queries` on the old
0.02 path. The two treated reads behave: `latent_read.read_norm` gain
7.099 -> 8.720, `core_input_norm` 1.006 -> 1.006, `value_norm` 0.871 -> 0.994.
The still-untreated one is the 49% contributor.

A real, separate defect follows from the same line. Adam is scale-invariant
per element, so the Adam group's `1.75e-5` moves `opponent_queries` by
`1.75e-5 / 0.019 = 9.2e-4` of its own RMS per step against `1.75e-5` for
`latent_queries` and `value_query`: a **52.6x larger effective learning rate**
on the one input-independent query in the critic, for 4,500 steps, with no
weight decay. Sub-batch gradients on that shared tensor also become collinear:
mean pairwise cosine **0.444 -> 0.980**, alignment ratio 0.718 -> 0.991. The
critic's late error is a common-mode, state-independent direction concentrated
on the opponent channel -- the same pathology class as the failed unit-RMS
critic, now on the untreated tensor.

Two secondary inverse-radius terms are real but smaller:
`value_decoder.read_norm` radius **0.32179 -> 0.11489** with gain
**4.135 -> 9.891**, and `core_norm` input radius 1.00374 -> 0.85874. These are
genuine activation cancellation; they are not the 49% term.

**Conclusion on gradients:** the 330x is a faithful readout of a critic becoming
ill-conditioned (consistent with fixed-BC-data CE regressing 3.513 -> 5.174),
not a cause of divergence. It cannot move a parameter, and no critic parameter
norm explodes. Raising a clip or lowering a rate to make the number smaller
would treat the readout.

#### The money collapse is a shaped-reward credit-window defect (job 6097)

The reward is potential shaping on log-relative liquid assets:
`r_t = gamma*Phi(s_{t+1}) - Phi(s_t)` with terminal `U - Phi`
(`src/kaggriculture/encoding.py:302-363`, `rollout.py:1616-1626`), where `Phi`
counts **only bank money plus product liquidation value**
(`encoding.py:273-299`; `PRODUCTS` at `constants.py:9-19`). Land, placed
animals, shed animals, structures, seeds in the ground, and hired labour are
**not assets**. Every purchase is therefore an immediate, deterministic `Phi`
loss whose recovery arrives only at harvest or sale.

The actor's credit window is shorter than every payback in the game.
`actor_gae_lambda = 0.972183588` and `gamma = 0.997` give
`1/(1-gamma*lambda) = 32.5` turns = 1.36 days at 24 turns/day. The fraction of a
payoff at lag `L` that the lambda-return realizes rather than delegating to `V`
is `lambda^L`:

| Investment | Cost | Payback | L (turns) | Realized `lambda^L` | Delegated to `V` |
|---|---:|---:|---:|---:|---:|
| Wheat / carrot seed | $10 / $20 | 2 d | 48 | 25.8% | 74.2% |
| Goose | $300 | 4 d | 96 | 6.7% | 93.3% |
| Sheep | $500 | 6 d | 144 | 1.7% | 98.3% |
| Cow / tomato | $400 / $50 | 8 d | 192 | 0.4% | 99.6% |
| Strawberry / melon | $100 / $80 | 10 d | 240 | 0.1% | 99.9% |
| Land quadrant | $1000/$2000/$4000 | rest of season | -- | ~0% | ~100% |

The cost enters at lag 0 at full weight; the payoff enters at 0.1-26%. The
critic cannot supply the remainder: at checkpoint 15, eight waves before
release, monte-carlo R-squared on a fresh wave is 0.058 with prediction std
0.048 against return std 0.390, and at wave 23 itself
`credit_preupdate_all_ttg_513_plus_terminal_residual_explained_variance` is
**-0.408** -- worse than the mean for exactly the early-game states where
investment happens.

Measured directly, one fresh production-shaped wave per checkpoint with that
checkpoint's own actor and critic, grouping every active action component by the
action the behavior policy chose, trajectory-clustered standard errors:

| Checkpoint 15 group | Share | `A_gae` | `A_mc` | `A_mc - A_gae` (paired) | `W32` |
|---|---:|---:|---:|---:|---:|
| `buy_land` | 0.00125 | **-0.08385** +/- 0.00674 | **+0.05382** +/- 0.01810 | **+0.13767** +/- 0.01866 | -0.06994 |
| `buy_animal` | 0.00633 | **-0.02593** +/- 0.00460 | **+0.05482** +/- 0.01353 | **+0.08075** +/- 0.01418 | -0.05508 |
| `buy_seed` | 0.03849 | **-0.00531** +/- 0.00161 | **+0.02196** +/- 0.00978 | **+0.02727** +/- 0.00896 | -0.01329 |
| `place_animal` | 0.00271 | +0.01555 +/- 0.00313 | +0.10921 +/- 0.01581 | +0.09366 +/- 0.01369 | +0.02037 |
| `stop` (reference) | 0.60937 | +0.00524 +/- 0.00126 | +0.01240 +/- 0.00991 | +0.00716 | +0.00308 |

`A_gae` is exactly what the surrogate multiplies; `A_mc` shares its baseline but
uses lambda-one credit; `W32` is the shaped reward inside a 32-turn window. All
three purchase families have **significantly negative surrogate advantage and
significantly positive full-episode advantage**: the paired, baseline-free
truncation gap is +7.4, +5.7, and +3.0 standard errors. Relative to `stop` the
per-decision pressure is -0.0891 on `buy_land`, -0.0312 on `buy_animal`, and
-0.0105 on `buy_seed`. The best action in the game by lambda-one credit,
`place_animal` (+0.109), is understated 7x by the surrogate.

The pressure ordering is the dollar cost divided by `3000 + money`, because
`dPhi/d$ = 1/(STARTING_MONEY + m)` (`encoding.py:302-308`). That predicts the
observed extinction order and rate in `runs/production-critic-state-read-p100-20260910/metrics.jsonl`,
wave 22 -> wave 100:

| Action | Typical cost | Fraction w22 | Fraction w100 |
|---|---:|---:|---:|
| `buy_land` | $1000-4000 | 1.234e-3 | **0.0** |
| `buy_animal` | $300-500 | 6.519e-3 | 1.779e-5 |
| `hire` | $1-89 Fibonacci | 0.1851 | 0.1201 |
| `buy_seed` | $10-100 | 3.77e-2 | 4.37e-2 (quantity mean 4.36 -> 2.35) |
| `build` (free) | $0 | 5.07e-3 | 1.38e-2 |

Everything else is a precondition cascade. `place_animal`, `feed`, `care`, and
`collect_fertilizer` all had **positive** `A_gae` at checkpoint 15 (+0.0156,
+0.0176, +0.0197, +0.0159) and still fell to about 3e-5, because their action
masks require animals that `buy_animal` no longer supplies. `water`
(0.1085 -> 0.0563) and `plant` (0.0284 -> 0.0171) follow the seed quantity and
land. Terminal `A_mc` for `place_animal`/`feed`/`care` at checkpoint 70 is
+0.303/+0.202/+0.215 -- the largest long-horizon advantages in the action set,
on actions the policy had already extinguished.

The mechanism is self-accelerating, and the acceleration is quantitative. As
money falls the per-dollar potential cost rises as `1/(3000+m)`. From checkpoint
15 to 34 money went 50,077 -> 14,488, predicting a **3.035x** stronger penalty;
measured `A_gae(buy_land)` went -0.08385 -> -0.26415, a factor of **3.150**, and
`W32` a factor of 3.322. Measured `W32` for `buy_land` also tracks
`-log((3000+m)/(3000+m-4000))`: -0.0699 against -0.0784 at m=50,077 and -0.2323
against -0.2597 at m=14,488.

#### Why every guard missed it

- **Self-play is exactly zero-sum.** `shaped_pair_reward` returns
  `(r, -r)` (`encoding.py:362`), and 256 of 320 trajectories are self-play, so
  `self_play_score_rate` is pinned at 0.5000 in every wave of every run. A
  mutual economic collapse is invisible to 80% of the training signal.
- **The critic's loss improves as the economy dies.** `critic_gae_lambda = 1`,
  so the target is `gamma^(T-1-t) U - Phi_t`, whose variance collapses with the
  economy: `terminal_target_variance` 0.677 -> 0.0446 and `advantage_std`
  0.134 -> 0.057 while monte-carlo R-squared *rises* 0.108 -> 0.724. Passive
  play minimizes the critic's loss and the advantage magnitude simultaneously.
- **The autocull's disjunction is therefore unsatisfiable.**
  `scripts/train_ppo.py:248-253` resets patience when **either** the money EMA
  gains 1000 **or** the value-loss EMA drops 0.01. Replaying the real journals
  through that exact rule, the stale counter reaches a maximum of **1 against a
  patience of 30** in all three collapsing runs: the value-loss EMA clears its
  0.01 threshold on nearly every wave. The 20-observation warmup anchor is also
  set at wave 42 in the state-read run, by which point money had already fallen
  to about 8,500, so the money reference it compares against is the collapsed
  value.
- **The KL trust region bounds distance, not direction** (`ppo.py:3690-3692`,
  `target_kl = 0.03`): measured `approx_kl` stayed 0.002-0.005 per wave, and
  3,510 small correctly-bounded steps in a consistently wrong direction did the
  damage. There is no entropy coefficient at all (`ppo.py:552-568`), so nothing
  opposed the extinction of the purchase families.

#### The same signature in every run whose actor trained

| Run | Actor release | Dispersion ratio at release | Money before -> final | Entropy | `buy_animal` | Trunk grad |
|---|---:|---:|---:|---:|---:|---:|
| `production-no-nextlat-b5120-p100-20260909` | wave 11 | 0.199/0.487 = 0.410 | 39,607 -> 16,503 (x0.417) | 0.173 -> 0.260 | 8.1e-3 -> 1.2e-4 | 2.46 -> 209.0 |
| `production-critic-state-read-p100-20260910` | wave 23 | 0.113/0.385 = 0.294 | 50,417 -> 10,234 (x0.203) | 0.142 -> 0.262 | 6.5e-3 -> 1.8e-5 | 0.53 -> 63.4 |
| `production-nextlat-normalization-fix-p100-20260909` | wave 12 | 0.080/0.366 = 0.219 | 48,628 -> **52** (x0.001) | 0.150 -> 0.507 | 6.2e-3 -> 1.5e-4 | 1.86 -> 152.2 |
| `production-critic-unit-rms-p100-20260910` | **never** | -- | 48,628 -> 48,604 (x1.000) | 0.145 -> 0.143 | 6.2e-3 -> 6.4e-3 | 0.017 -> 0.006 |

Terminal money ratio is monotone in the critic's prediction-dispersion ratio at
release across the three runs that released, and the run whose actor never
updated is the only one with no collapse, no entropy rise, no purchase
extinction, and no gradient growth. The dispersion ordering is three points and
the two readiness gates differ (`monte_carlo_explained_variance` for the older
run, `monte_carlo_r_squared` for the newer), so this is a consistency check, not
a controlled dose-response. The unit-RMS run is also not a clean control: its
critic never fit anything, so its small gradients are degenerate rather than
healthy.

#### What this does and does not establish

Established by measurement: the multiplicative sources of the 330x; that
`opponent_queries` sits behind two stacked inverse-radius amplifiers and carries
a 52.6x effective learning rate; that the surrogate advantage for all three
purchase families is significantly negative while their lambda-one advantage is
significantly positive; that the extinction order follows dollar cost over
`3000 + money` and accelerates as `1/(3000+m)` with a 3.0-predicted/3.2-measured
factor; that the collapsed policy is worse on its **own** shaped objective
against fixed external opponents (terminal utility 3.098 -> 0.446 versus
`starter`, -1.931 -> -2.616 versus `public-v27`, from the matched 6086-6089
panels).

Not established: that fixing either mechanism prevents the collapse. `A_mc` is
not an unbiased causal advantage -- purchases correlate with rich states, so its
positive sign is partly selection; only the paired `A_mc - A_gae` gap is
baseline-free. `W32` is a whole-window reward conditioned on the action, not
that action's isolated cost, which is why `buy_animal`'s -0.055 exceeds one
animal's -0.0057 to -0.0095. No counterfactual training was run at a longer
`actor_gae_lambda`, with an asset-inclusive potential, with a unit-RMS
`opponent_queries`, or with an autocull conjunction. Nothing here proves the
gradient growth is harmless to learning -- only that it cannot change a step
size under NorMuon and Adam.

Evidence under `artifacts/probes/critic-collapse-20260910/`:
`norm-gain-localization.json` (job 6096), `advantage-attribution.json` (6097),
`context-radius-localization.json` (6098), and the executable sources
`measure_norm_gains.py`, `attribute_advantages.py`, `measure_context_radii.py`.
All three jobs used exclusive admission, one attempt, frozen weights, zero
optimizer steps, and verified checkpoint SHA-256 identity before and after.

### Optimizer alignment against the references: measured, not resolved, 2026-09-10

The user rejected reward shaping, self-play, and weight decay as causes and
directed that the optimizer claims be checked against `../modded-nanogpt` and
`../NextLat`, and, if genuinely misaligned, be tested aligned with the critic
run past its warmup gate. Three 100-wave runs, identical seed 20260812,
identical 140-argument launch, identical BC warm start, only the named change:

| run | change | peak trunk grad | w90-100 median | final money |
|---|---|---|---:|---:|
| `production-critic-state-read-p100-20260910` | baseline | 74.45 @w99 | 44.85 | 10,234 |
| `production-adam-rate-align-p100-20260910` | per-parameter Adam rate | 53.92 @w100 | 21.60 | 14,902 |
| `production-opponent-state-read-p100-20260910` | + opponent state read | 16.74 @w100 | 12.95 | 10,011 |

#### The misalignment is real

`modded-nanogpt` attaches an explicit `lr_mul` to every Adam-routed role in
its parameter table, spanning **0.01** on `smear_gate` to **75** on the
embedding tables, with 0.1-0.25 on the MUDD coefficient generators
(`train_gpt.py:2026-2050`). `../NextLat` exposes the same idea as the
`_get_param_lr_overrides` hook returning absolute per-parameter rates
(`models/model_base.py:150-159,199-214,253,288-306`). This trainer had **no
such mechanism**: one `adam_learning_rate` for every Adam parameter plus a
single hand-placed exception for `value_head`
(`ppo.py:1215-1236,1260-1274`). Measured on the baseline critic's own
checkpoint zero, that left `trunk.opponent_queries` at RMS 0.02002 taking
`1.75e-5 / 0.02002 = 8.74e-4` of its own RMS per step against `1.75e-5` for
the unit-RMS `latent_queries` and `value_query` -- a **50x** relative step, on
the one input-independent query bank in the critic.

Two other reference practices are *not* misalignments and were left alone:
no gradient clipping (`modded-nanogpt` has none anywhere; `grep clip_grad`
returns nothing) and Adam `eps=1e-10` (identical to `train_gpt.py:2065-2069`).
This repo's only clips are on the NextLat auxiliary predictors
(`ppo.py:3381,3482,3609`), which is deliberate.

#### What aligning it did

`optim.py` now collects per-parameter Adam rate multipliers that modules
declare through `adam_learning_rate_multipliers`, and builds one Adam group
per distinct multiplier (`optim.py:103-181,298-396`). `StructuredTrunk` and
`StructuredCritic` declare `SMALL_QUERY_INITIAL_SCALE = 0.02` for exactly the
query banks still initialized at that scale (`structured.py:349-357,946-971,
1302-1312`), so every Adam parameter moves by the same fraction of itself.

That alone moved the w90-100 median trunk gradient norm from **44.85 to
21.60** and the peak from 74.45 to 53.92. Extending `critic_state_read` to the
last remaining sub-unit-RMS residual seed -- `opponent_summary` becomes a
normalized state read with a unit-RMS query bank, matching what
`latent_read` and `value_decoder` already were (`structured.py:946-953`) --
took the peak to **16.74**, a **4.4x** reduction against baseline.

#### What it did not do

The growth is still there, with the same shape. Log-linear fits of the trunk
gradient norm against wave number over waves 24-100:

| run | slope per wave | x per 10 waves | doubling | R-squared |
|---|---:|---:|---:|---:|
| baseline | 0.0532 | 1.70 | 13.0 waves | 0.949 |
| Adam rate aligned | 0.0435 | 1.55 | 15.9 waves | 0.913 |
| + opponent state read | 0.0374 | 1.45 | 18.5 waves | 0.908 |

Three clean exponentials. The interventions bought a **30% smaller exponent**
and delayed the crossing of trunk norm 10 from wave 63 to wave 93; they did
not change the character. Growth from wave 1 is still **73x** in the best arm.

Money is untouched: 10,234 baseline versus 10,011 with both changes, from the
same 48,628 start. `market_buy_animal_fraction` still goes extinct
(6.4e-3 -> 2.8e-5). Whatever collapses the economy is not in the optimizer and
not in the opponent-summary geometry.

#### The invariant driver

Per-parameter RMS across every checkpoint of all three runs, log-linear from
wave 15, keeping only parameters with R-squared above 0.85 in all three:

| parameter | slope B | slope A | slope C | RMS first -> last |
|---|---:|---:|---:|---|
| `value_head.bias` | 0.0261 | 0.0241 | 0.0255 | 0 -> 0.2768 |
| `value_head.weight` | 0.0199 | 0.0168 | 0.0197 | 0 -> 0.2104 |
| `trunk.core.*.modulation.weight` | ~0.017 | ~0.014 | ~0.018 | 0 -> 0.0046 |
| everything else | < 0.0022 | < 0.0022 | < 0.0022 | flat |

Only the zero-initialized readout grows, and it grows at **the same rate in
every arm** while the total exponent differs by 30%. Every other critic
parameter is flat to within 0.2% over 100 waves: `latent_queries` 1.0000 ->
1.0002, `value_query` 1.0000 -> 0.9998, `core_norm` 1.0000 -> 1.0017. The two
interventions removed run-specific amplifiers stacked on top of a driver they
do not touch.

Nothing else in the metrics is exponential. Scanning every numeric metric for
a log-linear fit over waves 24-100, `critic_trunk_gradient_norm` is the only
cleanly *increasing* one (R-squared 0.91-0.95); the credit-diagnostic MSEs are
cleanly *decreasing* at 0.023-0.055 per wave. Regressing log gradient norm on
log target variance, log target std, log value loss, log MC R-squared, or log
money pools badly across the three runs (best pooled R-squared 0.67 against
0.91-0.95 for wave number alone). The growth is a function of training time,
not of any data statistic.

`value_head` is zero-initialized (`structured.py:1315-1317`), carries the
8.33x `critic_head_lr` group, has no weight decay, and feeds an HL-Gauss
cross-entropy with **no logit bound**. Its reference counterpart, `lm_head`,
is the one tensor `modded-nanogpt` both softcaps -- `23*sigmoid((z+5)/7.5)`,
`train_gpt.py:1690`, `triton_kernels.py:1208-1213` -- and decays hardest
(`wd_mul: 150`, betas `(0.5, 0.95)`, `train_gpt.py:2033`). This repo has
neither. That is the next thing to test and it has not been tested.

Evidence: `runs/production-adam-rate-align-p100-20260910` (job 6106),
`runs/production-opponent-state-read-p100-20260910` (jobs 6110 + 6115 resume
after an out-of-memory kill from a foreign GPU tenant at wave 67),
`artifacts/probes/critic-collapse-20260910/norm-gain-aligned.json` (job 6109).
Both source changes are kept: they cost nothing and remove 4.4x of peak
gradient norm. Neither is a fix.

#### Where the residual exponent actually lives

Re-running the norm-gain localization on both new arms (jobs 6109, 6122) shows
the opponent path is fully closed and the exponent barely moved.
`trunk.opponent_queries` carries gradient **88.68** at baseline checkpoint 100,
49.95 with the Adam rate aligned, and **0.154** with the opponent state read --
a 577x reduction -- while `trunk.latent_context_norm`'s backward gain drops
19.17 -> 1.04. Total gradient at checkpoint 100 still only falls 126.5 -> 64.3.

| factor, first fitted checkpoint -> 100 | baseline | Adam rate | + opp read |
|---|---:|---:|---:|
| `value_head.weight` norm | x5.71 | x4.00 | x5.60 |
| logit residual | x1.08 | x1.06 | x1.08 |
| trunk VJP at one fixed random cotangent | x3.02 | x2.45 | x3.09 |
| cotangent alignment with the amplified subspace | x17.77 | x7.96 | x9.30 |
| **total** | **x330.8** | **x82.8** | **x173.8** |

As exponents per wave: `value_head` 0.0205 / 0.0176 / 0.0203, trunk VJP 0.0130 /
0.0114 / 0.0132, alignment 0.0338 / 0.0263 / 0.0262. The head term is the one
that does not move between arms, and alignment is the largest single term in
all three.

The stacked non-affine `read_norm`s that `critic_state_read` itself introduces
are a large amplifier but a **saturating** one. Their input radii shrink and
plateau, and their gain product grows only 2.3-4.8x across a run:

| arm | read-norm gain product | per wave | total gradient | per wave |
|---|---:|---:|---:|---:|
| baseline (2 reads) | 29.4 -> 86.2 | +0.0127 | x330.8 | +0.0683 |
| Adam rate (2 reads) | 40.9 -> 94.8 | +0.0106 | x82.8 | +0.0559 |
| + opp read (3 reads) | 146.0 -> 699.5 | +0.0184 | x173.8 | +0.0607 |

At checkpoint 100 the three radii are 0.1404, 0.1351, 0.1122 with gains 7.26,
8.99, 10.73 -- a **700x** constant pullback amplifier through the critic, and
the price the state-read design pays. It explains the level, not the trend.

#### The value-head rate cannot simply be aligned

`critic_head_lr` is 1.4583e-4, **8.33x** the shared Adam rate, with no
`lr_mul` counterpart in either reference; `modded-nanogpt` runs `lm_head` at
the base Adam rate and controls it with betas `(0.5, 0.95)` and `wd_mul: 150`
instead (`train_gpt.py:2033`). Since `value_head` growth is the one
arm-invariant exponent term, arm D
(`runs/production-head-rate-align-p100-20260910`, job 6124) reran arm C with
`--critic-head-lr 1.75e-05`, the shared rate, changing nothing else.

It **failed the warmup contract**: MC R-squared reached only **0.036446** by
wave 40 against the required 0.10, and `_critic_warmup_decision`
(`scripts/train_ppo.py:1370-1376`) aborted the run with zero actor updates.
Against arm C at the same waves: 0.0018 vs 0.1102 at wave 25, 0.0364 vs 0.1111
at wave 40; value loss 3.627 vs 3.122. The 8.33x head rate is load-bearing --
the configuration comment at `ppo.py:517-520` is correct, and lowering it
starves the critic rather than taming it. The reference's actual control for
this tensor is a bounded logit plus heavy decay, neither of which exists here
and neither of which was tested.

Net for the optimizer question: one real misalignment found and fixed (the
missing per-parameter Adam rate, 20% of the exponent), two claimed
misalignments dismissed against the references (no clipping, `eps=1e-10`, both
identical to `modded-nanogpt`), and one that cannot be aligned without
breaking critic fitting. The exponential trend survives all of it.

#### Arm E: the reference's softcapped readout

`model.py:33-58` now bounds every categorical value readout the way
`modded-nanogpt` bounds its own -- `23 * sigmoid((z + 5) / 7.5)`, the
reference's exact constants (`train_gpt.py:1690`,
`triton_kernels.py:1208-1213`) -- applied at both critic heads
(`structured.py:1356`, `model.py:815`). The cap is provably non-binding for
this objective: fitting a single HL-Gauss target at `sigma = 0.75` bins to
convergence gives a capped cross-entropy floor of **1.200341** against
**1.200322** uncapped and an irreducible target entropy of **1.200312** -- a
2.8e-5 nat gap -- with fitted peak probability 0.482636 against the target's
0.482655 and `value()` recovering 0.299991 from a 0.300000 target. The
zero-initialized head still starts exactly uniform (capped logit 15.197396 on
every atom, softmax 0.00990099 = 1/101).

Arm E is arm C plus the cap, same seed, same launch, head rate unchanged.
Actor released at wave **29** (arm C: 23), so the cap costs a little critic
fitting speed but clears the warmup contract that arm D failed.

| arm | change | slope/wave | peak | w90-100 median | final money |
|---|---|---:|---:|---:|---:|
| B | baseline | +0.0532 | 74.45 | 44.85 | 10,234 |
| A | + per-parameter Adam rate | +0.0435 | 53.92 | 21.60 | 14,902 |
| C | + opponent state read | +0.0374 | 16.74 | 12.95 | 10,011 |
| D | head rate aligned to 1x | -- | aborted | -- | warmup failed |
| E | + softcapped value logits | **+0.0368** | **10.33** | **7.83** | 12,515 |

**The cap buys level, not trend.** Peak falls 16.74 -> 10.33 and the w90-100
median 12.95 -> 7.83, a cumulative **7.2x** off baseline's peak and **5.7x**
off its late median. The exponent moves 0.0374 -> 0.0368 per wave: nothing.
R-squared of the log-linear fit is 0.937; it is still the same exponential.

The mechanism check explains why. `value_head.weight` RMS still grows at
**+0.0184/wave** under the cap against +0.0197 uncapped, and ends *higher*
(0.2401 against 0.2077) because the sigmoid compresses, so the pre-cap logits
must travel further for the same distribution. The cap bounds the head's
*effect* on the backward field without stopping its growth -- which is exactly
a constant-factor intervention. Head-weight growth is therefore **not causal
for the exponent**; it was a correlate.

#### Standing after four interventions

| exponent term, per wave | B | A | C |
|---|---:|---:|---:|
| `value_head.weight` | 0.0205 | 0.0176 | 0.0203 |
| trunk VJP at fixed cotangent | 0.0130 | 0.0114 | 0.0132 |
| stacked `read_norm` gain product | 0.0127 | 0.0106 | 0.0184 |
| **cotangent alignment** | **0.0338** | **0.0263** | **0.0262** |

Everything that has been removed -- a 50x mis-scaled Adam rate, a 50x
inverse-radius residual seed carrying 49% of the gradient, and an unbounded
readout -- was a multiplicative *level*. The surviving term is the alignment
one: the critic's per-sample cotangents become progressively more common-mode
and progressively better aligned with the trunk's most amplified directions.
That is a conditioning property of the representation under a non-stationary
target, and no optimizer or readout change addresses it.

Money is untouched across every arm: 10,234 / 14,902 / 10,011 / 12,515 from
the same 48,628 start, with `buy_animal` extinct in all of them. The two
failures remain independent, and only the gradient one has been reduced.

All three source changes are kept -- per-parameter Adam rates, the opponent
state read, and the softcapped readout. Together they cost nothing measurable,
remove 7.2x of peak critic gradient norm, and bring the trainer into line with
the reference on the three points where it genuinely departed. None of them is
a fix for the trend.

Evidence: `runs/production-value-softcap-p100-20260910` (job 6155),
`runs/production-head-rate-align-p100-20260910` (job 6124, aborted by the
warmup contract).

## Credit window, and the Adam epsilon floor (2026-09-10, second pass)

### Arms F and G: the actor's credit window

Two arms changed only `--actor-gae-lambda` from the VAPO constant 0.972183588
against the same 140-argument launch and seed 20260812.

| arm | actor lambda | slope/wave (24-100) | R2 | peak grad | final money | trough money | adv std at w25 |
|---|---|---:|---:|---:|---:|---:|---:|
| B baseline | 0.972184 | +0.0532 | 0.949 | 74.45 | 10,234 | 7,930 | 0.128 |
| E + softcap | 0.972184 | +0.0368 | 0.937 | 10.33 | 12,515 | 8,447 | 0.127 |
| F | 0.996044 | +0.0306 | 0.920 | 6.13 | 15,473 | 11,151 | 0.262 |
| G | 1.0 | -- | -- | 3.05 | 23,901 (w92) | 22,377 | 0.423 |

Arm G (`runs/production-actor-lambda1-p100-20260910`, job 6173, cancelled by the
queue at wave 92) is the best play this campaign has produced: money settles near
24k instead of 10-12k, and the critic gradient never leaves single digits.

The mechanism is not the credit window as such. Advantage standard deviation at
wave 25 goes 0.128 -> 0.262 -> 0.423 across B/F/G, which is a 3.3x change in the
size of the actor's gradient, and that is what the next section makes matter.

### The Adam epsilon floor

`modded-nanogpt`'s Adam epsilon is 1e-10 and this trainer copied it. Epsilon is
only negligible against the second moments a trainer actually produces, and a
PPO surrogate's are nothing like a language model's. Read directly out of the
actor optimizer state of `runs/production-value-softcap-p100-20260910`
(39,582 Adam-managed elements, `sqrt(v_hat)` with bias correction at step 3240):

* Tenth percentile `sqrt(v_hat)` is 1.9e-9, only 19x above epsilon.
* `market_quantity_bias` has a median of 1.07e-11 at wave 100, two orders BELOW
  epsilon. Its mean attenuation `sqrt(v_hat) / (sqrt(v_hat) + eps)` is 0.38.
* The floor tightens monotonically as the critic fits and advantages shrink.
  `market_quantity_bias` attenuation runs 0.69 / 0.59 / 0.42 / 0.38 at waves
  41 / 59 / 78 / 100, its median `sqrt(v_hat)` falling 1.10e-9 -> 1.07e-11.
  Adam parameters under 0.95 attenuation go 1 -> 2 -> 5 -> 7 over the same waves;
  `trunk.units.slot.weight` 1.00 -> 0.68, `unit_head.1.weight` 1.00 -> 0.90.
* The critic side is clean: every critic Adam parameter stays at 1.0000, because
  its gradients grow rather than shrink.

A floored element is not Adam-stepped at all. Its update is `m_hat / eps`,
proportional to the gradient instead of normalized by it, so its effective
learning rate is the advantage scale. The rarely-sampled action rows are hit
first and hardest, which makes an action's disappearance self-sealing: sampled
less, smaller second moment, more attenuation, updated less. That is a ratchet,
and it is why `buy_animal` never came back in any arm.

Measured end to end
(`tests/test_ppo.py::test_actor_update_is_invariant_to_a_uniform_advantage_rescale`):
scaling every advantage by 32 must leave the update alone, because Polar Express
divides its input by that input's Frobenius norm. At epsilon 1e-10 the update
cosine is 0.998279 and its norm ratio 1.010436, with individual parameters off by
15-20% and `spatial.input.bias` off by 3993%. At 1e-20 the same readings are
0.999975 and 0.999971. `src/kaggriculture/optim.py` now ships 1e-20.

This also settles advantage whitening, removed from `prepare_advantages` by
`15e869c` (a performance commit with an empty body). The scale half of whitening
reaches the update only through this epsilon; with the floor gone it is a
per-update constant the optimizer divides back out. The mean half is not inert,
but it is small: mean advantage over advantage std averages -0.005 to -0.010 over
the second half of B/E/F and is positive in only 15-20 of 50 waves.
`PpoConfig.normalize_advantages` and `--normalize-advantages` exist to test it.

### What remains of the critic gradient

The cotangent decomposition now runs the trainer's own target pipeline
(`_owned_behavior_values` then `prepare_advantages`, clipped to the support) on
5113 sampled states per checkpoint, and splits the parameter gradient into
`G_common = sum_i J_i^T cbar` and the rest
(`artifacts/probes/critic-collapse-20260910/cotangent-softcap.json`, job 6175):

| iteration | grad norm | common | fluctuation | cosine | common share of cotangent energy | gain at cbar over random |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.974 | 0.000 | 0.974 | 0.000 | -- | 0.00 |
| 17 | 0.281 | 0.195 | 0.277 | 0.370 | 0.058 | 0.73 |
| 41 | 2.923 | 3.003 | 0.502 | 0.986 | 0.116 | 1.36 |
| 59 | 7.539 | 8.023 | 4.341 | 0.846 | 0.090 | 2.06 |
| 78 | 28.318 | 24.048 | 4.890 | 0.996 | 0.170 | 2.33 |
| 100 | 73.862 | 61.999 | 12.180 | 0.999 | 0.192 | 3.32 |

Only 19% of the cotangent field's energy is common-mode at wave 100, yet it
produces 84% of the gradient norm: shared cotangents sum as B while fluctuations
sum as sqrt(B), and the trunk amplifies the shared direction 3.3x more than a
random one. The mean distributional residual itself stays small and does not
drift monotonically -- its norm goes 0.0195 -> 0.0636 between waves 17 and 100
and its first moment changes sign -- so the growth factorizes as common-mode
residual (3.3x) times directional anisotropy (4.5x) times trunk and head gain
(about 18x), not as a runaway prediction error.

### Update cost

At production shape a wave is 17.86 s
(`runs/production-actor-lambda1-p100-20260910` wave 92, 230,080 states): rollout
3.35 s at 68.8k states/s, behavior replay 1.07 s at 214k states/s, minibatches
12.998 s, staging 0.24 s. 45 actor plus 45 critic minibatches at 5120 states
take 144 ms each.

An earlier version of this section called roughly 6.5 s of that launch overhead,
by scaling the behavior-replay path's states per second up to a forward plus
backward. That inference is withdrawn: the replay is a forward-only critic pass
at whole-wave chunk width and is not a unit of update work, and the claim
contradicts a measurement already in the tree. `_cached_update_callable`'s
docstring reports wall clock equal to summed device time within 0.4% at 2048 and
4096 rows in eager and both compiled modes, with `reduce-overhead` removing 97.5%
of launch submissions (906 per actor minibatch down to 23) for a 0.2% change in
wall clock: a compiled actor minibatch is 906 kernels over 55.4 ms, about 61 us
each, so the launches hide behind the device. This phase shortens only by doing
less device work -- fusion, precision, fewer minibatches, smaller model, fewer
states -- and `artifacts/probes/update-backends-20260910.json` (job 6204)
re-measures the mode table at the production 5120-row shape to confirm that the
conclusion still holds there.

The one-time costs are large and were being paid inside measured waves. Across
the four arms wave 1 runs 134.7-180.2 s against a steady 8.1-8.8 s: rollout
84.2-89.6 s against 3.0-3.4 s, update 25.8-70.1 s against 5.1-5.4 s. The actor
release wave pays again -- 47.8 s against the next wave's 18.1 s in arm B, and
+1.1 to +3.3 s in the others -- because the actor-side update graphs compile
only once the actor first steps. The rollout share is dominated by
`_warmup_balanced_league`, which compiles one Inductor specialization per league
lane count from 1 to `max_lanes`, plus the collector's own forward and the
behavior replay. Inductor's FX graph cache and the AOTAutograd cache are both on
by default in torch 2.13 with a stable cache directory
(`/var/tmp/torchinductor_marvin`), so this is cold-cache cost that every arm in
this campaign paid afresh only because every arm edited the source the graphs
hash over.

### Scalar critic option

`scalar_value` on both model configurations (`--scalar-value true`) replaces the
categorical readout with CleanRL's: `nn.Linear(model_dim, 1)`, loss
`0.5 * (prediction - return)^2` (`model.py::scalar_value_loss`), no softcap on
the readout, and no clipping of the value target to a support in `update_ppo`.
`value_atoms`, `value_min`, `value_max` and `value_sigma_ratio` go inert. The
categorical path is untouched and remains the default, so the two are one flag
apart on the same launch; checkpoints are not interchangeable between them,
since the head's width differs (101 versus 1, 1,662,117 versus 1,649,217
parameters on the production structured critic).

Rationale from the measurement above: the categorical objective is what makes
the critic's gradient a growing quantity at all. The readout must sharpen a
101-way softmax onto an HL-Gauss target whose width is fixed in support units
while the targets themselves concentrate (`value_target_std` 0.39 -> 0.17 over
waves 17-100), and 84% of the resulting parameter gradient is common-mode. A
scalar head has no sharpening to do: its gradient is the residual itself.

Queued arms: `kragg-scalar-critic-p100` (job 6196) and `kragg-scalar-lam1-p100`
(job 6197), plus `kragg-adameps-p100` (6194) and `kragg-adameps-lam1-p100` (6195)
for the epsilon change alone, and `kragg-profile-update` (6193) for the update
cost.

## One compile, in wave one (2026-09-11)

`--fail-on-late-compile` (default on) aborts a run when a settled wave compiles
**anything at all**. `compilewatch.py` reads Dynamo's own `CompilationMetrics`
and separates the two kinds by `cache_size`, the entry count that frame already
had: positive is a guard failure -- an input varied that was meant to be
constant, with the reason text taken from Dynamo's `recompiles` artifact log
rather than reconstructed, so the abort names the tensor and the extent that
moved; zero is a first compile arriving late, which means a warmup did not
reach a frame the wave then paid for. Every wave records `dynamo_compiles`,
`dynamo_recompiles`, `dynamo_compile_seconds` and `dynamo_shapes_settled`, and
prints one `{"event": "compilations"}` line naming each frame it paid for.

Fatality waits for two facts: the actor was already unfrozen for the *previous*
wave (its forward and backward are warmed while it is frozen, but the release
wave is still the first to step it, so it keeps a one-wave grace), and this
wave's league lane layout repeats the previous one (the collector's only
legitimately moving shape). The guard began as recompile-only; that was too
weak, and the two sections below are what it was missing.

### What it found, and what each cost

A 6-wave production-shape run under the guard reported recompiles in six
frames. Three distinct causes, all real:

1. `_actor_minibatch_terms`, `_critic_minibatch_fit_terms`,
   `_replayed_selected_logprobs`: `size mismatch at index 0, expected 5113,
   actual 5112`. `_balanced_minibatch_slices` produced near-equal minibatches,
   so each wave ran two row counts whose values moved with the wave's valid
   state count.
2. `_replayed_value_chunk`: `expected 4096, actual 704` -- the short final
   replay chunk, whose size is the wave's row count modulo 4096.
3. `_StackedActorEnsemble._forward`: `self.params['market_quantity_bias'] size
   mismatch at index 0, expected 3, actual 2` -- the league stack width. The
   selection count moves all run: 2,3,4,5,...,11 with 10 changes after wave 25
   in `production-value-softcap-p100`.

### Fixes, measured

`_fixed_minibatch_positions` replaces the balanced partitioner everywhere (one
partitioner, 11 files): every minibatch is exactly `minibatch_size` rows and the
final one wraps to the epoch's leading positions, which costs under 5120
duplicated rows of ~230,000 in one minibatch of 45, drawn fresh each epoch.
`replay_behavior_values` wraps its final chunk the same way and trims the
result. The parity audit and the predictor evaluation slice `row[:count]` so no
state is scored twice.

The stacked ensemble is now keyed by lane shape, not by model identity. Keying
on identity built fresh stacked tensors nearly every wave, and the compiled
forward bakes their addresses through `mark_static_address`, so each new
instance re-traced inside its own wave; one instance per shape, refilled by
`load`, keeps the addresses fixed for the process. Each `(mode, width)` also
compiles through its own code object, so Dynamo's per-code recompile budget is
no longer the binding constraint and `recompile_limit` is no longer raised.
`_warmup_league_layout` warms that same instance instead of a throwaway stack;
warming a throwaway compiled against addresses the wave never used.

Evidence: `runs/production-adameps-p100-20260910` (job 6248), 50 waves, release
at wave 29.

|                | before            | after |
|----------------|------------------:|------:|
| wave 1         | 176.2-180.2 s     | 55.1 s |
| wave 1 rollout | 84.2-89.6 s       | 24.0 s |
| steady wave, frozen actor | 8.07-8.79 s | 8.45-8.58 s |
| steady wave, released actor | 17.0-18.1 s | 16.6 s |
| recompiles after wave 1 | per wave | 0 over 50 waves |

### The two intermittent compiles, and their warmups

What remained after those fixes was two kinds of *first* compile arriving in
later waves, 131 s of it spread across a 50-wave run:

1. **League lane layouts.** One new per-layout code object at waves 4, 16, 32,
   46-50 (`rollout.py:1300`, 7.2-7.8 s each) as the historical pool filled.
   `_balanced_assignments` (`train_ppo.py:928`) gives each selected opponent
   `ceil(games / lanes)` or `floor(games / lanes)` games, so the padded width is
   exactly `ceil(league_games / lanes)` and the reachable set is the lane counts
   one through the league's maximum -- 11 shapes, nothing data-dependent.
   `_warmup_reachable_layouts` compiles all of them in the first wave on the
   persistent ensembles later waves acquire and refill. This is the shape set
   the deleted `warmup_league_lanes` had right; what it got wrong was warming
   throwaway stacks, so nothing it compiled was ever replayed.
2. **The released actor's backward.** 18.0 s and 6.0 s landing in the release
   wave itself: Inductor compiles a backward on its first `.backward()`, not
   with its forward, and a warmup wave runs `actor_epochs=0` so the actor's
   forward traced in wave 1 (via the parity audit, under `no_grad`) while its
   backward could not. Nothing about those graphs needs the actor unfrozen, so
   `_warm_actor_update_graphs` runs one minibatch with gradients on and the
   released path's flags, then throws the result away: no optimizer step, no
   schedule advance, no metrics, both gradient buffers cleared. The actor stays
   byte-identical and `actor_optimizer.state` stays empty, pinned by
   `test_zero_actor_epochs_runs_a_critic_only_warmup_update`.

Measured, 42 waves at production shape, warmup floor 2, release at wave 29
(`runs/scratch-compile-once-20260911`, job 6275):

|wave|compiles|recompiles|compile s|wave s|
|---|---:|---:|---:|---:|
|1|38|3|122.7|130.6|
|2-28 (frozen)|0|0|0|8.6 median|
|29 (release)|0|0|0|17.4|
|30-42 (released)|0|0|0|16.6 median|

The guard was armed on 10 of those waves (`dynamo_shapes_settled`, waves 33-42)
under the broadened rule and never fired. Total compile is now 122.7 s once,
against 131 s previously spread across the run, and the release wave costs what
a steady released wave costs (17.4 s vs 16.6 s; it was 24.1 s). Wave 33's 39.9 s
is the external evaluation subprocess sharing the device -- rollout 20.1 s
against a 3.3 s steady rollout, with zero compiles.

Wave 1's three recompiles are `_polar_express_wide_batch` (`optim.py:216`)
specializing on rank and stride, 0.56 s for four cache entries; they are
bounded, they never recur, and they are why the guard waits for a settled wave.

What this does not touch: the released-actor wave is 13.4 s of update against
3.15 s of rollout, and the update is device-bound (`launch_bound_gap_ratio`
0.957 at the production 5120-row shape), so compile-mode changes cannot move it
-- measured 11.93 s default, 12.09 s `reduce-overhead`, 11.61 s
`max-autotune-no-cudagraphs` for 601 s of compile.

### Where the update's 13.4 s actually goes

`scripts/profile_update_phases.py` (job 6231) at production shape, 4793-row
minibatches, device time over wall time in every section: actor
forward+backward 79.8 ms (0.999), critic forward+backward 74.0 ms (0.999),
actor optimizer step 7.3 ms, critic optimizer step 5.9 ms, gradient norms 1.2 ms
combined, gathers under 0.3 ms. Nothing in the update is launch-bound, so the
lever is device work -- 48 actor plus 48 critic minibatches is 8.1 s of the
12.8 s `update_minibatch_seconds`, and the remainder is not in these sections.

The profiler also had a real bug: it called `requires_grad_(False)` on the actor
and critic while the sections were being *defined*, so every later backward
section saw a graphless surrogate and the run died at `actor_forward_backward`.
The freeze was unnecessary in the first place -- the predictor sections pass
`model_grad=False`, and that path takes its source belief under `no_grad`
(`ppo.py:2574-2576`).

### Scalar-critic arms need no separate clone

`scalar_value` lives on the shared model configuration, so `--scalar-value true`
also changed the *actor's* recorded config and `_load_initial_actor` rejected
every BC artifact. Only the critic reads the field (`structured.py:1321,1364,1386`,
`model.py:789,823,828`), so it joins the critic-only exclusions already listed
there beside `critic_core_layers`, `critic_latents` and `critic_state_read`.

### Four arms under the guard (12-minute budget each, truncated)

All four ran the same 140-argument launch, seed 20260812, warmup floor 20,
`--max-hours 0.2`, with `--fail-on-recompile` live. Every arm recorded exactly
**3 recompiles, all in wave 1**, and the guard never fired.

|arm|waves|release|final money|final R^2|critic trunk grad norm|
|---|---:|---:|---:|---:|---:|
|`adameps` (epsilon 1e-20 only)|50|29|9,355|0.214|2.26|
|`adameps` + `--actor-gae-lambda 1.0`|51|29|28,260|0.329|1.02|
|`scalar-critic`|43|21|10,275|0.277|0.24|
|`scalar-critic` + `--actor-gae-lambda 1.0`|47|21|25,664|0.397|0.28|

Read as direction only: the budget cuts every arm near wave 50, so there is no
doubling time here, and wall times in these four are contended by a foreign
tenant. Both scalar-critic arms release at 21 rather than 29 -- the scalar head
reaches the Monte Carlo R-squared gate faster -- and carry a critic gradient
norm an order of magnitude below the categorical arms, which is the prediction
the cotangent decomposition made. The two lambda-one arms are the only ones
holding money in the 25-28k range at truncation.

## The money collapse is entropy inflation, and it was already in telemetry

Every intervention in this file treated the money collapse as a downstream
consequence of the critic's gradient growth. It is not. It is visible in
`rollout_entropy` and the action-mix fractions that every run has always
recorded, and it begins at the wave the actor is released.

`production-value-softcap-p100`, release at wave 29:

|wave|money|sell frac|harvest frac|qty mean|entropy|approx KL|clip frac|
|---:|---:|---:|---:|---:|---:|---:|---:|
|25-28 (frozen)|44.7k|0.089|0.0311|4.29|0.147|0|0|
|29 (release)|46.0k|0.090|0.0311|4.27|0.147|0.0015|0.012|
|35|36.5k|0.079|0.0263|3.81|0.170|0.0024|0.018|
|41|9.2k|0.036|0.0165|3.40|0.241|0.0034|0.025|
|45|8.4k|0.035|0.0164|3.40|0.246|0.0034|0.024|

Entropy rises monotonically from the BC clone's 0.147 while the two
money-producing factors halve. Per-wave KL stays at 0.003 and the clip fraction
at 2%: there is no instability to find, which is why every stability-flavoured
intervention missed it. **There is no entropy bonus anywhere in this
codebase** -- no `entropy_coefficient`, no temperature schedule (`temperature`
is 1.0) -- so the inflation is the update itself moving mass off a peaked
policy, and the rare, precise, productive actions are what it costs.

### Two 90-100 wave arms, and what lambda actually buys

All three ran under the guard, snapshot `23b1086126`, seed 20260812, warmup
floor 20, zero compiles after wave 1 (90, 100 and 100 waves). The control was
drained at wave 90 by the queue admitting a waiting tenant, not by any failure.
The scalar arm compiled 202 s in wave one rather than 123 s because its
Inductor cache had just been cleared; the invariant held cold.

|arm|waves|release|entropy release -> end|reaches 0.20|sell|harvest|money|target corr|
|---|---:|---:|---|---:|---|---|---|---:|
|`--actor-gae-lambda 0.9722` (control)|90|29|0.147 -> 0.233 (peak 0.253 at w43)|w36|0.090 -> 0.050|0.0311 -> 0.0099|46.0k -> 7.0k -> 14.6k|0.673|
|`--actor-gae-lambda 1.0`|100|29|0.147 -> 0.233|w61|0.090 -> 0.077|0.0311 -> 0.0181|46.0k -> 23.4k|0.720|
|`--scalar-value true --actor-gae-lambda 1.0`|100|21|0.140 -> 0.234|w47|0.091 -> 0.085|0.0322 -> 0.0182|50.0k -> 24.7k|0.832|

All three converge on **the same entropy fixed point, 0.233-0.234**, from three
different advantage estimators and two different value parameterizations. The
fixed point is a property of the update, not of the critic. What differs is
*which* actions pay for it: the control spends its entropy on harvest (down
3.1x) and selling, while both lambda-one arms hold harvest at 0.018 and selling
near 0.08 and end 1.6-1.7x richer.

The scalar arm is the informative one, because it isolates critic quality.
It fits far better -- target correlation 0.832 against the control's 0.673, and
it clears the release gate at wave 21 instead of 29 -- and it buys **1.3k of
money over lambda one alone** (24.7k against 23.4k) and no change at all in the
entropy fixed point. So the earlier reading of this as bad-critic credit
routing is too strong: under lambda one the critic is a pure baseline, its
quality only shrinks advantage variance, and that variance is not what is
moving money. Lambda is the whole effect, and it acts on the credit path for
delayed payoffs -- a harvest pays off only through a later sale -- not on the
amount of probability mass the update moves.

### What it is not

`ppo.py:597-602` already suspected the common-mode advantage offset -- a
persistently negative mean advantage pushes every *sampled* action's logprob
down, which is an entropy force with no counterpart in the clipped surrogate.
Measured across five runs, the offset is real but far too small and far too
uncorrelated to be the driver:

|run|released waves|median `advantage_mean/advantage_std`|waves negative|corr with next wave's entropy change|
|---|---:|---:|---:|---:|
|`lamdefault` 90w|62|-0.0047|38/62|-0.034|
|`lam1` 100w|72|-0.0024|39/72|-0.004|
|`adameps` 50w|22|-0.0038|13/22|-0.002|
|`adameps-lam1` 51w|23|-0.0072|14/23|-0.070|
|`softcap` 100w|72|-0.0044|42/72|-0.061|

Sign-persistent in only 60% of waves and correlated with the entropy step at
|r| <= 0.07. Rejected.

Also rejected as primary: the asymmetric clip. `clip_high` is 1.28 against
`clip_low` 0.8 (`ppo.py:543`, no rationale recorded anywhere in this file),
which is DAPO's clip-higher and exists precisely to stop entropy *collapse*.
It only binds on 1.2-2.5% of tokens here, so it cannot account for a 1.7x
entropy rise -- but it is the wrong sign for this failure and deserves an arm.

### The learner loses to its own past

Per-opponent league telemetry at wave 100 of the lambda-one arm, mean money
margin from the learner's seat:

|opponent|category|games|mean margin|score rate|
|---|---|---:|---:|---:|
|`00000053`|historical|6|-36,571|0.333|
|`00000062`|historical|6|-18,025|0.167|
|`00000000`|historical|6|-10,779|0.500|
|`00000098`|active|6|-10,159|0.167|
|`00000090`|active|6|+391|0.500|
|`00000033`|historical|6|+4,444|0.667|
|`00000077`|historical|5|+6,825|0.600|

It loses to seven of ten, including three of its own earlier snapshots, and
loses worst to its wave-53 self. The regression is head to head, not an
artifact of reading absolute money.

## What the policy actually observes (schema v3 audit)

The dense 29-channel board in `encoding.py` is not the training path;
`Game::encode_player_structured` (`core.rs:1010`) is, and it emits tokens:

|group|count|content|
|---|---:|---|
|tiles|200 (both farms)|6 categorical -- kind, occupant, farm, row, column, quadrant -- and 20 continuous|
|units|16|role, slot, row, column; 12 carried counts, carried total, shed-access flag, 12 FIFO ranks; a 5-tile HERE/NSEW gather|
|products|9|market inventory offset, price/(2 base), base/max base, own shed, own carried|
|animals|3|cost, shed, carried|
|crops|5|seed cost, seeds held, first-yield day, max-yield day, max held, ongoing flag|
|farms|2|money (log), unlocked quadrants/4, hands, hires today|
|town|1|6 clock scalars plus 8 shop counts|

Settled by reading the encoder rather than the wrapper: the step is present
four ways (`core.rs:1174-1177`); crops and animals are fully distinguishable
(`kind` spans Empty/Locked/Weed/Plant/Coop/Pasture and `occupant` is 0 none,
1-5 crop, 6-8 animal, `core.rs:3044`, `:3080`, so an empty coop is distinct
from a stocked one and from a pasture); tile magnitudes are species-normalized,
so absolute stock is recoverable only jointly with the occupant embedding; and
there are no shrubs -- `TileKind` has exactly six variants (`core.rs:123-131`),
weeds are the only nuisance tile, and a weed tile carries no age at all
(`core.rs:3041` writes nothing for it).

Unit-to-tile geometry is the weak axis. Axial RoPE is applied only inside the
two farm-local tile blocks (`structured.py:1009`, `:1022-1025`), so tile-to-tile
relative position is proper. Units are in no rotated attention: a unit gets
absolute row/column embeddings, its 5-tile gather, and then the unit decoder
attends to **only the 32 latents plus those 5 tiles** (`structured.py:1201-1235`)
-- never to the 100 own-tile tokens. "Nearest harvest-ready tile, three east"
has to survive a 144-token to 32-latent bottleneck. Half of all unit actions
are moves (`unit_move_fraction` 0.50, `unit_pass_fraction` 0.24).

### Gaps, ranked by how hard the quantity is to reconstruct

1. **Shed room is nowhere.** The cap is a *shared* 100 across all 12 items
   (`shed_total`, `core.rs:610`; enforced at `:572`, `:578`, `:746`, `:756`),
   and the end-of-day auto-drop deposits `min(carried, room)` and then zeroes
   the carried stack, so overflow is **destroyed** (`core.rs:747-749`, run for
   every unit at `:2417`). The observation gives 12 per-item fractions in 12
   separate tokens and never their sum; the dense encoder it replaced did
   provide it (`encoding.py:196-202`). Deposit actions are masked away at the
   cap, so fullness reaches the policy only through the mask, and the
   destructive path -- harvest, then midnight -- has no mask. Low incidence at
   today's skill (`unit_place_fraction` 0.003, `unit_shed_fraction` 0.025), so
   this is a ceiling, not the current bottleneck.
2. **No marginal revenue.** Selling q yields the sum of `market_price(inv+k)`
   over k, and each product has its own curve shape (Linear/Square/Sqrt/Log).
   The policy sees the current price and `(inv - 10000)/500` only, and picks
   `market_quantity_mean` near 4 out of 100 available. Price at a few
   quantities, or the local slope, is four floats on a token that exists.
3. **The consumption clock has no phase.** The town drains market inventory at
   `step % 4 == 0` and `step % 24 == 0` (`core.rs:2365-2379`), so prices
   ratchet on a fixed comb. The clock is six continuous scalars carrying only
   the fundamental daily harmonic; `step % 4` is a frequency-six function of a
   linear ramp. Every spatial axis got an embedding (row, column, quadrant) and
   the clock got scalars. Midnight is drastic -- all hands fired, all carried
   goods force-dropped, positions reset to spawn (`core.rs:2405-2428`).
4. **Animal tokens carry 3 fields against crops' 6** -- no first-yield day, no
   max held, no required structure -- and nothing anywhere links an animal to
   its product (goose to egg) or a crop to its product index. All constants, so
   memorizable; lowest priority.
5. **The critic cannot see who it is playing.** `CriticExtras`
   (`structured.py:209-217`) is the opponent's private columns and unit tokens:
   state, never identity. League seats are drawn from 2 active plus 6
   historical snapshots plus 3 built-in lanes including `pass` and `random`, and
   the per-opponent mean margin at wave 100 spans -36,571 to +6,825. At step 0
   every opponent's state is identical, so the critic must predict the mixture
   mean and its R-squared is structurally capped. Opponent identity is free
   under CTDE -- the critic is discarded at inference -- and is the cheapest
   available attack on the 0.21-0.40 R-squared that the collapse analysis above
   makes load-bearing.

Not a missing feature but a missing dependency: all 10 market slots are decoded
in one parallel forward (`structured.py:1239-1260`), so slot 5's kind logits
cannot know slot 1 already sold 100 wheat; only the sequentially updated mask
couples them (`core.rs:1555-1588`), and quantity conditions solely on its own
slot's kind. The 16 unit actions are the same. That caps deliberate order
splitting and duplicate avoidance.

Every item in 1-4 bumps `OBSERVATION_SCHEMA_VERSION`, which hard-rejects every
existing BC artifact (`structured.py:113-116`), so they belong in one v4 rather
than four.

## HL-Gauss bandwidth revisit (2026-09-14)

Actor NextLat was abandoned; its coefficients remain zero. The starting
categorical arm is job 6914, `production-hlgauss-dreamer-p500-20260913`.
Despite its name this is raw-return HL-Gauss, not a full Dreamer critic:
255 atoms on `[-2.2,2.2]`, no symlog, no two-hot targets, and no EMA.

The [HL-Gauss paper](https://arxiv.org/pdf/2403.03950), section 5.1.2,
motivates tuning smoothing in return units independently of discretization.
Moving from 101 to 255 atoms while retaining sigma/bin `0.75` narrowed the
Gaussian from sigma `0.033` to `0.0129921`. The implementation otherwise
matches integrated, normalized Gaussian labels and categorical cross-entropy.
Its sigmoid logit cap is a separate repository choice, not prescribed by the
paper; historical cap benefits do not establish current saturation.

Job **6944** tests only sigma/bin **0.75 -> 3.0** (raw sigma **0.0519685**).
The new shared `--value-sigma-ratio` flag avoids editing defaults for trials;
actor-only BC loading permits the critic-only override. Actor, critic and
head learning rates, optimizer, softcap, zero initialization, critic auxiliary
1/1 plain sum, seed, BC checkpoint, 6400 minibatch, league and compilation
modes remain matched. Current source explicitly records
`actor_opponent_farm=True`, equivalent to the old always-on path.

Frozen source: `ff0c0357999f15178673e70b86c2e13dab5357454b7a42c809e5320bed518f68`.
Run: `runs/production-hlgauss-bandwidth3-p500-20260914`.
Both trials had a 25-minute queue cap, concurrency one, and native autocull.
The new trial completed **96 waves**, versus **119** for the old arm, then
hit the cap without a numerical failure. Actor release moved **17 -> 16**.

Matched waves **77-96**, arithmetic means of per-wave metrics:

| Metric | Original HL | Wider HL | Change |
|---|---:|---:|---:|
| Pre-update Monte Carlo EV | 0.696181 | 0.713530 | +0.017349 |
| Pre-update Monte Carlo MSE | 0.005698 | 0.005630 | -1.20% |
| Combined critic gradient norm | 53.154845 | 52.720756 | -0.82% |
| Online money | 51,216.59 | 52,069.70 | +1.67% |
| Rollout entropy | 0.163572 | 0.156776 | lower |
| Seconds/wave | 12.150965 | 14.142334 | +16.39% |
| Value cross-entropy | 2.775750 | 3.025229 | different label entropy |

The first ten frozen-actor waves remain nearly flat: mean rollout EV
`0.001459 -> 0.001659`. Thus bandwidth alone does not explain the early
mean-learning delay or solve the large critic gradients. Those gradients
combine CE and the critic auxiliary; neither their norm nor small auxiliary
losses identify objective interference.

At the same wall-clock cap, final-20-wave EV is **0.713530 versus 0.753328**.
Compilation costs differ and an unmanaged external GPU workload was present;
both rollout and update were slower, so the entire slowdown cannot be
attributed to smoothing. The realized compute-budget result nevertheless
does not establish a win.

Last external evaluations, wider checkpoint 83 versus original checkpoint
102, scored starter **4/4 vs 4/4**, public-v27 **3/4 vs 4/4**, and public-v16
**3/4 vs 0/4**. Each opponent uses only two paired seeds; these mixed,
high-uncertainty results do not establish stronger play.

**Not promoted.** Scalar remains the default and categorical sigma/bin stays
`0.75`. This was a modest per-update improvement, not an optimal HL-Gauss
configuration. Before another readout/optimizer change, measure marginal
sharpening, raw cap derivatives, and separate CE/auxiliary parameter gradients;
the present evidence cannot select the mechanism.

Verification job **6943** passed three parser/warm-start regressions, including
bit-identical compiled BF16 actor outputs after a critic smoothing override.
Compiled Gaussian checks over 8193 targets in `[-2,2]` measured maximum mean
bias `6.70e-6` at sigma/bin 3.0. Interior label entropy rises from approximately
**1.2003 to 2.5222 nats**; CE includes that floor and conditional uncertainty,
so raw CE is not a comparable scalar-error metric across the arms.

Evidence: `artifacts/probes/hlgauss-bandwidth-20260914/{experiment,analysis,comparison,numerics}.json`.
Latest recovery checkpoint is wave 83; the final actor snapshot is wave 96.
The temporary numerical verifier was removed after success; job logs retain
its measured output. No automatic retry or further trial was launched.

## Promoted HL-Gauss state-mean baseline (2026-09-14)

The user subsequently promoted the wider HL-Gauss critic and the active
state-mean run as the new baseline. This supersedes the non-promotion decision
above: defaults are now categorical HL-Gauss, 255 atoms on `[-2.2, 2.2]`,
sigma/bin `3.0`, and state-mean component-clipped PPO. Actor NextLat remains
off; critic latent and decoded-value auxiliaries remain coefficient 1 each,
plain sum, horizon 1. The critic and actor have independent weights.

Job **6946**, frozen source
`ce9f9fb0d05dca755f8587522398bfcfef4b42344b76efecb81b50e3479063a5`,
completed **97 waves** at its **25-minute cap**, with actor release at wave
**16**. Relative to job 6944, only the PPO policy-loss denominator changed
from active components to valid states. Final 20-wave arithmetic means:

| Metric | State-mean baseline |
|---|---:|
| Value-target correlation | 0.835445 |
| Pre-update Monte Carlo EV | 0.697190 |
| Pre-update Monte Carlo MSE | 0.005665 |
| Combined critic gradient norm | 44.889176 |
| Online money | 52,816.78 |
| Rollout entropy | 0.156943 |
| Seconds/wave | 13.946455 |

At matched waves 77-96, component/state means were correlation
**0.845115 / 0.835309**, MC EV **0.713530 / 0.696841**, and money
**52,069.70 / 52,798.89**. These are single-seed online comparisons,
not evidence of an external win-rate improvement. Promotion is the user's
baseline choice, not a claim that every measured metric improved.

Verification job **6945** passed **50 regressions**, including compiled BF16
warm-start compatibility and critic auxiliary gradients. A separate compiled
gradient check on a trained critic checkpoint confirmed nonzero value-head,
latent-query, and first/last ViT gradients, with no actor gradient or shared
parameters. Its artificial targets establish attachment, not the magnitude
or direction of training gradients.

Run: `runs/production-hlgauss-state-mean-p500-20260914`.
Evidence: `artifacts/probes/hlgauss-state-mean-20260914/`, including
`experiment.json`, `comparison.json`, and `verification.json`.
Latest external evaluations used checkpoint 82: starter **4/4**, public-v27
**4/4**, public-v16 **2/4**. Each used two paired seeds; the final actor
snapshot is wave 97, not the externally evaluated checkpoint.

## Priority threefold learning-rate ablation (2026-09-14)

Job **6947** used the exact frozen baseline source `ce9f9fb0d05dca755f8587522398bfcfef4b42344b76efecb81b50e3479063a5`,
with actor and critic rates **0.00005 -> 0.00015**, critic-head rate
**0.0001458333333 -> 0.0004375**, and the inherited critic-predictor rate
tripled accordingly. Actor auxiliary remained off. Priority **1**, concurrency
**1**, hard **25-minute cap**, no retry. The run completed **98 waves**, with
actor release at **11**, versus baseline release **16**.

| Final-20-wave mean | Baseline | 3x LR |
|---|---:|---:|
| Value-target correlation | 0.835445 | 0.938391 |
| Monte Carlo EV | 0.697190 | 0.879801 |
| Monte Carlo MSE | 0.005665 | 0.002677 |
| Combined critic gradient norm | 44.889176 | 154.276475 |
| Online money | 52,816.78 | 36,257.20 |
| Entropy | 0.156943 | 0.360701 |

The critic fits its on-policy targets better, but observed play is worse.
Latest external checkpoint **83** scored starter **4/4**, public-v27 **0/4**,
public-v16 **0/4**, versus baseline checkpoint 82's **4/4, 4/4, 2/4**.
Each opponent still has only two paired seeds. The value metrics use each
policy's own changing state distribution, not a shared held-out dataset.
This arm is not promoted; all other ablations retain the baseline rates.

Run: `runs/production-hlgauss-state-mean-lr3-p500-20260914`.
Evidence: `artifacts/probes/hlgauss-state-mean-lr3-20260914/experiment.json`.

## Promoted dense VAPO temporal defaults (2026-09-14)

The user identified dense-reward trial **7010** as the winner and promoted its
temporal settings for future runs. This supersedes the gamma/lambda defaults
of baseline 6946; the rest of that baseline remains unchanged.

| Setting | Previous baseline | Promoted default |
|---|---:|---:|
| Reward | Dense bounded-margin potential shaping | Unchanged |
| Gamma | 0.997 | 1.0 |
| Actor GAE lambda | 1.0 | 0.972183588317107 |
| Critic GAE lambda | 1.0 | 1.0 |

Actor lambda is `1 - 1 / (0.05 * 719)`. The critic learns undiscounted Monte
Carlo shaped returns; the actor uses a shorter GAE trace and the critic's
intermediate predictions for lower-variance credit. Terminal utility remains
the normalized final-bank margin, not binary win/loss. Base learning rates,
HL-Gauss sigma/bin 3, state-mean component clipping, actor auxiliary off, and
critic auxiliaries at 1 each remain unchanged.

Promotion was the user's decision during the trial. Existing queued ablations
retain their frozen commands; future launchers inherit the shared defaults unless explicitly
overridden. The per-entity critic experiment still requires an explicit
`--actor-gae-lambda 1`; it is not silently combined with VAPO's shorter trace.

Run: `runs/production-hlgauss-vapo-dense-p500-20260914`.
Trial command and source provenance:
`artifacts/probes/actor-head-joint-ablation-20260914/vapo-trials.json`.

The trial subsequently finished **101 waves** at its **25-minute cap**.
Exit 143 is the intended MLQ timeout, not a numerical crash. Actor release
was wave **13**, versus **16** in baseline 6946. Final 20-wave means:

| Metric | Previous baseline | Dense VAPO |
|---|---:|---:|
| Online money | 52,816.78 | 66,031.57 |
| Entropy | 0.156943 | 0.135017 |
| Seconds/wave | 13.946455 | 12.860979 |
| Value-target correlation | 0.835445 | 0.745910 |
| Monte Carlo EV | 0.697190 | 0.555210 |
| Monte Carlo MSE | 0.005665 | 0.042179 |
| Critic gradient norm | 44.889176 | 32.396110 |

Online money increased **25.02%**. Critic target scales change with gamma, and
each policy visits its own state distribution: raw MSE is not a matched
critic-quality comparison. Latest external checkpoint **85** scored starter
**100%**, public-v27 **100%**, and public-v16 **75%**, each over four games
(two paired seeds). Baseline checkpoint 82 scored **100%, 100%, 50%**.
The final actor snapshot is wave **101**, not the externally evaluated checkpoint.

Final evidence:
`artifacts/probes/actor-head-joint-ablation-20260914/vapo-dense-results.json`.

### Joint and terminal-only migrated onto VAPO

The user subsequently requested the full VAPO temporal settings for both queued
ablations, superseding the earlier decision to retain their original commands.
Cancelled joint trial **6986** is replaced by fresh trial **7072**; queued
terminal-only trial **7004** was cancelled before start and replaced by **7073**.
Both use gamma **1**, actor lambda **0.972183588317107**, and critic lambda **1**.
Joint retains dense reward; terminal-only retains sparse final-bank margin.
The prior requirement to first demonstrate learning with terminal-only at the
old temporal settings no longer applies.

Both replacements retain the original validated frozen source for their arm,
BC initialization, seed, learning rates, workload, and autocull policy. Each has
an exclusive **25-minute cap**, priority **0**, and one attempt. Frozen-source
CLI parsing and the accepted MLQ commands were checked before recording them.
These jobs were queued, not completed, when this entry was written.

Commands, source digests, environment, and exact argument differences:
`artifacts/probes/actor-head-joint-ablation-20260914/vapo-migrated-ablations.json`.

### Threefold learning rates on the promoted dense VAPO base

Requested trial **7074** changes only learning rates relative to dense VAPO
**7010**: actor/critic **0.00005 -> 0.00015**, critic head
**0.0001458333333 -> 0.0004375**, with the critic predictor inheriting the
tripled critic rate. Gamma **1**, actor lambda **0.972183588317107**, critic
lambda **1**, dense shaping, and actor auxiliary off remain unchanged.
It uses the exact frozen source and BC initialization of 7010, a fresh run
directory, priority **0**, exclusive GPU use, and a hard **25-minute cap**.
Existing autocull is unchanged; no retries. This is distinct from old trial
6947, which used gamma 0.997 and actor lambda 1.

After the requested comparisons finish, the next candidate is two critic
epochs per wave with one actor epoch at base VAPO rates. It trades fewer
rollout waves for better-fitted intermediate values; external play and
time-to-go credit diagnostics must justify the extra compute. It has not
been queued ahead of the requested trials.

Command, exact argument differences, parsed configuration, and followup rationale:
`artifacts/probes/actor-head-joint-ablation-20260914/vapo-lr3.json`.

### Per-entity critic result

Trial **7003** completed **60 waves** at its hard **25-minute cap**; exit 143
was the intended timeout. Actor release was wave **28**. It retained the
original gamma **0.997**, actor/critic lambda **1/1**, and active-entity
advantage normalization; it was not a VAPO-configured trial.

Final 20-wave means: online money **45,029.64**, value-target correlation
**0.624903**, Monte Carlo EV **0.386550**, and seconds/wave **16.214902**.
Original baseline 6946 completed 97 waves, released at 16, and averaged
money **52,816.78** over its final 20 waves. These are equal-cap comparisons,
not matched training ages or state distributions.

Latest external checkpoint **55** scored **100%** against starter, **100%**
against public-v27, and **0%** against public-v16, each over four games.
The final actor snapshot is wave **60**. No promotion: this arm did not
improve the observed equal-budget results. Dense VAPO remains the default.
Evidence: `artifacts/probes/actor-head-joint-ablation-20260914/per-entity-results.json`.

## Neural league compilation buckets (2026-09-14)

Compiled mixed-play inference now rounds neural lane counts and per-lane widths
to powers of two. Padding duplicates existing inputs and actor weights, and its
outputs are discarded before sampling. Opponent selection, physical games,
stored training rows, and recovery checkpoint cadence are unchanged. The late
compile guard now compares physical buckets, not raw assignment counts.

The full-production-model benchmark replayed the 31-layout sequence from
terminal-bank job **7073**, with fixed checkpoint weights/inputs and BF16 autocast.
Baseline **7083** versus bucketed **7086**, each with separate cold caches:

| Measurement | Exact layouts | Bucketed layouts |
|---|---:|---:|
| Compiled graphs | 12 | 6 |
| Reported compilation time | 203.30 s | 105.04 s |
| Isolated benchmark wall time | 237.73 s | 129.02 s |
| Projected frozen-forward GPU time over 31 x 719 steps | 13.79 s | 14.62 s |
| Generated compiler cache footprint after matched validation workloads | 525.93 MB | 284.81 MB |

All **558** forward/captured-replay head comparisons were bitwise identical,
including changed-weight refills. Full-wave jobs **7087/7088** and **7090/7091**
also matched every stored array exactly: 128 self-play plus 64 league games,
719 decisions, and 230,080 stored states per wave. The latter pair exercised
four layouts and then four cached waves; cached totals were **10.31 versus
10.14 seconds**, with zero compilation events in either arm. This single paired
measurement establishes no steady-state speedup, but showed no rollout slowdown.
No PPO update or learning-quality improvement is claimed.

The cache figures are apparent generated bytes in isolated tmpfs caches, not
physical SSD-write measurements. Production job 7073 used `TMPDIR=/var/tmp`,
which is NVMe-backed here. Checkpoint writes were not changed.

Regression job **7089** passed eight focused tests. Disabling bucketing in a
separate negative-control process made the new regression fail on repeated
compilation as intended. A symbolic-shape prototype was rejected after native
PyTorch vmap batching rules specialized dynamic sizes; no eager fallback or
relaxed compiler checks remain.

MLQ remains globally paused. Training jobs **7074**, **7081**, and **7082** were
held only during benchmark admission and restored to queued state without any
attempt starting. Their immutable source snapshots are unchanged and do not
automatically receive this working-tree optimization.

Evidence: `artifacts/probes/league-layout-compile/summary.json` and its linked raw
benchmark, full-wave, and regression reports.

### Queued adoption

Replaced **7074 → 7093** (dense LR3), **7081 → 7094** (terminal-bank,
resume iteration 28), and **7082 → 7095** (joint clip/KL, resume iteration 33).
Each replacement freezes its original experimental source plus only the
compile-layout backport in `rollout.py` and `train_ppo.py`; model architecture,
reward, optimization settings, 25-minute limits, dependencies, and output
directories are preserved. Parsed CLI configurations match apart from the
explicitly migrated resume paths.

Original checkpoints and source snapshots remain unchanged. Migrated checkpoint
copies record original source identities and checkpoint hashes, candidate source
identities, and the compile evidence. Every original non-source payload field
compares exactly after serialization, including optimizer and RNG state; all
league sidecar hashes match their checkpoint manifests. Checkpoint format and
candidate source identity checks pass.

During preparation, another queue operation held the original jobs and started
job 7092. Replacement jobs preserve those individual holds with zero attempts.
Admission was paused only for replacement and restored to its observed unpaused
state; job 7092 was not interrupted. No training was launched by this adoption.

Evidence and exact source identities:
`artifacts/probes/queued-compile-adoption-20260914/manifest.json`.

## Joint and terminal reward trials at threefold learning rates (2026-09-14)

Current dense VAPO defaults remain unchanged. Three fresh BC-initialized trials
use actor/critic learning rates **0.00015** and critic-head rate **0.0004375**;
the critic predictor inherits the tripled critic rate.

| MLQ job | Policy ratio / KL scope | Reward |
|---|---|---|
| 7120 | Joint | Dense shaped bank margin |
| 7121 | Components | Terminal-only bank margin |
| 7122 | Components | Terminal-only win/loss/draw: +1/-1/0 |

All three use the same frozen source
`b7b54cedfab2e314e84401e01469c4e01ae4040f514c3b5befbfc40ed81979c9`,
including league compilation buckets. Gamma **1**, actor lambda
**0.972183588317107**, critic lambda **1**, architecture, seed, BC initialization,
critic warmup/readiness, compiled BF16 execution, and external evaluation are
held constant. These are fresh trials, not checkpoint continuations. Each has
an exclusive **25-minute cap**, priority **0**, and one attempt.

Existing autocull remains unchanged: after 20 actor-active warmup waves, either
money EMA +1000 or value-loss EMA improvement of min(0.01, 1% of reference loss)
resets 30-wave patience, with EMA alpha 0.1. Money is not the new outcome
objective, and easier value fitting is not evidence of stronger play; use
external results to judge the win/loss/draw arm.

Verification: eight focused reward, native full-horizon, credit-diagnostic,
and production-parser checks passed; affected Python files passed Ruff.
A separate frozen-source native scenario exercised all 719 transitions:
terminal scores (520,3000), (3000,520), and (3000,3000) produced outcome rewards
(-1,+1), (+1,-1), and (0,0), with all earlier outcome rewards zero.
All frozen launch commands parsed and MLQ accepted the declared limits.
Job 7120 was running and 7121/7122 queued when this entry was recorded;
no learning result is claimed yet.

Commands and provenance:
`artifacts/probes/actor-head-joint-ablation-20260914/vapo-lr3-reward-trials.json`.
Native scenario:
`artifacts/probes/actor-head-joint-ablation-20260914/terminal-outcome-verification.json`.

### Next architecture direction (proposal, not implemented)

SAM3's image decoder repeatedly evolves object queries through self-attention,
prompt cross-attention, image cross-attention, and an FFN while retaining fixed
encoded image memory. Our default actor instead reads observation memory into
32 generic latents once; eight core layers reinject the same initial encoding,
not fresh map evidence, before separate farmer and market decoding.

The proposed controlled sequence is actor-only repeated full-memory reads,
then a matched 32-slot workspace containing 16 units, ten market-order slots,
and six scratch slots. Keep the spatial encoder, local farmer detail, explicit
economy tokens, opponent summary, global modulation, and independent centralized
critic unchanged. Dedicated exogenous conditioning, removing scratch slots, or
map writeback are later hypotheses, not simultaneous changes.

This requires deliberate architecture-specific initialization: strict BC loading
cannot preserve a changed entity workspace merely because tensor sizes match.
Compare equal wall time and equal environment steps, accounting for extra KV
projections/FFNs. Static memory does not make projected KV reusable across
independent layers. No architecture experiment has been queued here.
Evidence and design constraints:
`artifacts/probes/actor-head-joint-ablation-20260914/architecture-direction.json`.

## Promoted terminal-outcome LR3 recipe and entity-only direction (2026-09-14)

The user selected **7122**, HL-Gauss VAPO terminal-outcome LR3, as the best run
so far. That selection is accepted directly; no new comparison was run to
reconfirm it. Luna promoted the new-launch defaults:

- Reward: terminal-only win/loss/draw **+1/-1/0**.
- Actor/critic base LR: **0.00015**; ordinary Adam rate **0.0000525**.
- Production/direct-CLI critic-head LR: **0.0004375**, via the existing formula.
- Gamma **1**, actor lambda **0.972183588317107**, critic lambda **1** retained.
- Minimum fresh-BC critic warmup **10 waves**. The readiness gate and 40-wave
  deadline remain; this is distinct from 32 optimizer-step LR warmup.
- HL-Gauss, component PPO ratios/KL, actor auxiliaries off and production critic
  auxiliaries 1/1, architecture, and existing autocull remain unchanged.

The production launcher's stale help advertising five warmup waves was corrected.
Omitted fresh-BC warmup resolves to ten; explicit overrides remain usable.
Resume restores its checkpoint's warmup state and receives no implicit warmup flag.
Historical immutable jobs/snapshots are unchanged, and old reward modes remain
explicit options. Checkpoint-following critic probes use their recorded reward
mode and gamma rather than silently inheriting the new reward default.

Verification: **11** focused reward/diagnostic/checkpoint-metadata tests passed;
Ruff passed on twelve changed Python files. A separate non-model proof exercised
direct and production CLI defaults, benchmark reward defaults, explicit old-mode
and LR overrides, the ten-wave warmup floor/readiness gate, resume flag omission,
and all 719 native game transitions. Default rewards were zero before terminal,
then (-1,+1), (+1,-1), and (0,0) for the three win/loss/draw scenarios.

### Revised architecture proposal: only 26 persistent entities

This supersedes the earlier 32-token/six-scratch proposal. Keep exactly **16 unit
and ten market-order tokens** as evolving state. Start with four rounds of
entity self-attention, cross-attention to static memory, and one FFN per round,
then feed actor heads directly from their corresponding tokens. Memory contains
all 200 encoded farm tiles plus 20 economy tokens; remove generic core latents,
opponent summary compression, and the separate output cross-decoders.
Retain local farmer features at initialization and explicit ownership/position.

A separate centralized critic follows the entity-workspace design and attention
pools its final tokens into the normalized global value belief. Preserve the
existing critic NextLat target on that pooled belief, not 26 new auxiliary
targets. Keeping the existing sixteen private opponent-unit context tokens gives
236 static critic-memory tokens while retaining only 26 evolving query tokens.

For hardware alignment, compare 80/96/128 embedding widths, not extra entities.
Four query heads give widths 20 (currently padded to 24), 24, and 32 respectively.
An especially relevant candidate keeps the expensive 200-tile encoder at width
80 and makes only the entity stream width 128. Its cross-attention can project
80-wide memory directly to entity K/V without a separate widened-memory tensor.
Current code explicitly selects memory-efficient SDPA; width 128 alone does not
enable Flash. A historical isolated Flash win did not improve full update time.

Static arithmetic, not runtime evidence: with four rounds and deliberately shared
memory K/V, encoder-plus-reasoning MACs versus the current D80 actor are about
0.816x for D80 throughout, 1.117x for D96 throughout, 1.859x for D128 throughout,
and 1.104x for D80 tiles/D128 entities. Counts omit embeddings, normalization,
gates, heads, layout/padding and backward. Shared K/V changes parameter sharing;
per-round projections are a separate capacity control. Measure compiled BF16
whole-update and rollout time, memory, and equal-budget learning rather than
inferring speed from utilization or attention-pair counts.

At this decision point the architecture was still a proposal; the implementation
and measured execution are recorded below. Evidence and calculation assumptions:
`artifacts/probes/actor-head-joint-ablation-20260914/terminal-outcome-default-promotion.json`.

## D96 entity-attention implementation and campaign (2026-09-14)

The user approved **96 for both memory and entities**, not a width sweep.
`entity-attention` is now the production family. Actor and critic independently
encode both farms with two shared-weight local blocks and evolve exactly
16 unit plus ten market-order states through four self/cross/FFN rounds.
Actor static memory has 220 tokens; centralized critic memory has 236, including
16 private opponent units. RMS normalization and memory K/V projection are shared
across rounds and evaluated once per forward, without detaching their gradients.
Four query heads use two KV heads of width 24. Economy-conditioned RMS scale/shift,
local unit tile features, ownership/position, and inactive-unit masking are retained.
There are no generic/scratch latents, compressed opponent summaries, reinjection,
MUDD, or output cross-decoders. Critic attention pooling retains one normalized
96-wide NextLat value belief, with the existing coefficient-1/1 losses.

Registry capabilities integrate BC, native rollout, physical frozen ensembles,
PPO, optional actor dynamics, critic NextLat, checkpoint inference, and submission
packaging. Historical families remain separately registered for old artifacts;
new BC initialization is required. Full seven-field legacy BC auxiliary beliefs
are intentionally unsupported for this family.

Source: `057c9c480101ef1260ef606f9fac1ab0178f3abb432196bed0eacc018ceb99e0`.
Launches, environment, artifact paths and benchmark commands:
`artifacts/probes/entity-attention-d96-20260914/`.
All jobs use MLQ, exclusive parallel limit 1, priority 0, one attempt and a
25-minute cap. No CPU model execution or eager performance fallback is used.
Eager BF16 is only the GPU benchmark's numerical correctness reference.

Verification: 16 configuration/schema cases passed without constructing models;
MLQ **7127** passed all six CUDA contract tests in 215.11 seconds, including
inactive/private-state isolation, shared-KV gradients from all four rounds,
compiled combined backward, strict artifact loading, full 719-transition native
collection, replay gates and an actual PPO update. Ruff formatting/checks passed.

MLQ **7128** completed the full matched two-epoch BC recipe: four provenance-matched
v16 corpora, 299,104 training rows, 69,024 held-out rows, twelve held-out seeds per
corpus, CUDA BF16 compilation and batch size 1024. Final held-out NLL was
**0.00130696**, versus the historical BC artifact's **0.00172686**; unit/kind/
quantity accuracy was **99.9889% / 99.8643% / 99.9656%**. This establishes compatible
initialization, not improved game strength. Artifact:
`runs/entity-attention-d96-bc-20260914/bc-actor.pt`.

Matched fixed-state benchmark **7129**, six-repeat legacy/entity whole-iteration
benchmarks **7130/7131**, and production PPO **7132** use that frozen source.
PPO retains the promoted terminal-outcome LR3 recipe and full 128-self/64-league
wave, ten-wave minimum critic warmup, readiness gate and existing autocull.
Its CPU asynchronous external evaluator is disabled; official evaluation is
performed separately with the compiled BF16 CUDA checkpoint evaluator.
This difference must be retained when interpreting historical wall-time results.

### Whole-iteration timing

Both six-repeat full-shape benchmarks completed with all 36 actor, critic and
predictor updates per wave. Medians below exclude the first cold iteration:

| Metric | Legacy structured D80 | Entity-attention D96 |
| --- | ---: | ---: |
| Rollout | 2.66978 s | 2.28002 s |
| PPO update | 9.31682 s | 8.47432 s |
| Whole iteration | 12.09513 s | 10.80719 s |
| Peak live CUDA memory | 21.10 GiB | 16.33 GiB |
| Peak reserved CUDA memory | 21.75 GiB | 16.66 GiB |
| First iteration, including cold work | 38.22934 s | 53.04079 s |

Measured whole-iteration throughput improves **1.119x** (10.65% less time);
rollout improves 1.171x and update 1.099x. This is a modest speedup, not the
anticipated large multiplier. Live memory falls about 22.6%. Both arms use
compatible BC artifacts, eight frozen opponents, identical production wave
sizes/objectives and no skipped updates; their policy-generated states differ.
The timing harness does not include production's evolving PFSP/built-in mix.
Replay gates pass with zero tail fraction. Raw records and the extracted
comparison are in `whole-iteration-comparison.json` beside the campaign launches.

Fixed-state job 7129 stopped on its legacy raw-logit closeness assertion:
maximum absolute difference 1.09375, including inactive/illegal outputs.
Raw logits are not the categorical consumer contract. The corrected benchmark
retains those diagnostics and gates legal-policy KL and tail probability mass
using existing replay tolerances, without changing model code or precision.
MLQ **7136** uses benchmark-only source revision
`8e62a793406be015fe16211688920007578dc6086ac919a665e0e9515119f8ba`;
the failed report remains preserved rather than overwritten.

### Full training control and GPU activity

PPO **7132** stopped through the unchanged autocull policy at **iteration 115**
(exit 75), not a crash or an abbreviated training probe. Training elapsed time
was **1283.51 seconds**. The readiness gate released actor training at iteration
20; 96 actor-active waves completed. The final wave accepted all 36 actor,
critic and critic-predictor updates. After 30 stale actor-active observations,
EMA money was 34,898.51 versus its 48,897.63 reference and EMA value loss was
3.00548 versus its 2.98493 reference.

Final MC R² was **0.35354**, critic gradient norm **1.63808**, and money mean
**34,388.90**. At the same 115-wave point, historical 7122 recorded MC R²
0.35734, critic gradient norm 104.62317 and money mean 58,279.65.
Different evolving matchups prevent treating training money as a matched external
score. Lower critic gradient norms alone do not establish better prediction or
policy quality. Full summaries, last-ten medians and caveats are saved in
`training-comparison.json`. Final actor:
`runs/production-entity-attention-d96-outcome-lr3-p500-20260914/league/league-actor-00000115.pt`.
The first official evaluation submissions 7137/7138/7139 rejected the internal
league-snapshot format before running games. Corrected MLQ **7140/7141/7142**
evaluated the final recovery `checkpoint-000115.pt` with two paired development
seeds each, using compiled BF16 CUDA inference and the pinned official engine.

The additional `data/bc-v16-current-v27-64` corpus was used by **neither** this BC
run nor its historical production BC initialization. Both used the same four
mirror/starter/pass/random corpora to keep the architecture comparison controlled.

An eight-second live GPU sample during training recorded seven observations at
100% activity (mostly 473–478 W) and one at 59%; activity is not achieved FLOPs.
Steady benchmark medians attribute 7.343 s to minibatch work, 1.037 s to behavior
replay, 0.066 s to staging, 0.022 s to advantages and 0.010 s to finalization.
The separate 2.087 s replay-parity audit is diagnostic benchmark overhead, not
part of the timed PPO update or its 10.807 s whole-iteration total.
The measured bottleneck is training computation, not host staging. No D96 kernel
profile yet separates GEMM, attention, normalization and optimizer contributions.
Evidence: `gpu-efficiency-observation.json`.

### Fixed-state and official evaluation results

Corrected fixed-state benchmark **7136** completed both architectures. It used
the same 6,400 states/actions/masks from one full 230,080-state native wave:

| Warm median wall time | Legacy D80 | Entity D96 |
| --- | ---: | ---: |
| Actor PPO forward/backward | 108.648 ms | 94.921 ms |
| Critic HL-Gauss + NextLat forward/backward | 99.750 ms | 96.736 ms |
| Behavior replay | 33.035 ms | 26.550 ms |

These exclude optimizer work. Critic forward/backward improves only 3.1%, limiting
the whole-update gain. Actor parameters fall from 975,358 to 800,810; critic
parameters from 888,699 to 808,215. Legal-policy KL against the eager-BF16
correctness reference stayed below 0.000895 for legacy and 0.000439 for entity,
with zero tail probability mass. Compiled losses and gradients passed their
checks. Small forward-only measurements are not the physical ensemble/CUDA-graph
collector; use the whole-iteration pair for rollout throughput claims.
Evidence: `fixed-benchmark-policy-parity.json` and `fixed-state-summary.json`.

Final control official evaluation, four games per opponent:

| Opponent | Wins / draws / losses | Candidate mean bank |
| --- | ---: | ---: |
| starter | 4 / 0 / 0 | 62,257.75 |
| public-v27 | 0 / 0 / 4 | 44,942.00 |
| public-v16 | 0 / 0 / 4 | 32,234.50 |

All twelve games completed, with zero invalid games. This two-seed development
panel is too small for a broad strength claim; it does not establish improved
learning over historical 7122. Evidence: `official-evaluation-summary.json` and
the three complete `evaluation-*.json` reports.

## Independent per-head actor NextLat trial (2026-09-14)

After the D96 control completed, the user requested actor NextLat again, one
predictor per actor head, following `../NextLat`. This is not merely enabling
the previous shared-attention predictor.

`ActorDynamics` now has independent unit-action, market-kind and market-quantity
residual MLPs. Each consumes `[joint_action_embedding, normalized_head_state]`,
applies the reference's bias-free RMS normalization (eps 1e-5), then three
bias-free linear layers with two GELUs, and adds the predicted delta to the
source state. Reference projection-factor 1 gives hidden width **256** at D96.
The reference calls its wrapper `LayerNorm`, but `bias=False` actually executes
RMSNorm; the implementation follows that behavior. Linear/embedding
initialization uses normal std 0.02.

The RL adaptation encodes the complete valid joint action through the existing
typed action embeddings and one fixed-slot flattened projection to D96. Inactive
units and post-STOP slots are excluded; STOP itself remains, and quantities on
non-quantified kinds are inert. One action code conditions all three independent
predictors. Kind and quantity share the initial normalized market source but
have separate recurrent predictions. The quantity auxiliary freezes the entire
D96-to-rank32 factorized readout, conditioned on the successor selected kind.

Each head has its own masked-coordinate SmoothL1 mean and masked-decision
teacher-to-student KL mean. Both three-head sums receive coefficient **1**;
horizon **1**, optional CE **0**, and no source-gradient balancing or implicit
division by three. Quantity targets require successor quantity activity, not
merely an active market slot. Unit survival and trajectory boundaries remain
enforced. Sources and predictors train; successor teachers and auxiliary readout
weights are detached. Actor/critic inference weights and critic NextLat are
unchanged; old shared-attention predictor states are intentionally incompatible.

MLQ **7143** passed **28 tests** in 103.91 seconds, including three-field
recurrence, independent head gradients, frozen matching quantity decode,
successor activity/ancestry, inert ignored actions, strict entity artifacts,
full native rollout/PPO update, and combined fullgraph BF16 PPO plus actor and
critic auxiliary backward. Ruff passed on the five affected Python files.

Full trial **7144** uses immutable source
`ba2e45d5dbd7438a05d145a26323c12916d993b6dc2f3023555de823fd2dcdd7`,
the same D96 BC artifact, four corpora, seed, production wave geometry, LR3
terminal-outcome recipe, critic NextLat and autocull as the control. Actor
coefficients/horizon are explicitly 1/1/1; production actor-auxiliary defaults
remain off. The additional v16-versus-v27 corpus is still excluded to isolate
this change. MLQ limit 1, priority 0, one attempt, 25-minute cap.

Reference file hashes and the exact contract are recorded in
`artifacts/probes/entity-attention-d96-20260914/actor-head-nextlat-design.json`;
launch/source evidence is in `actor-head-nextlat-trial-job.json`.
Output: `runs/production-entity-attention-d96-actor-head-nextlat-p500-20260914`.

### Cancelled trial: early policy collapse

MLQ **7144** received a cancellation request and exited 143/SIGTERM after
iteration **32**. Main did not issue that cancellation and did not restart it.
This is a **partial, cancelled result**, not a completed 25-minute trial or
autocull. The last logged training elapsed time was **409.58 seconds**, with
13 actor-active waves starting at iteration 20.

| Iteration 32 | Actor auxiliary off control | Three-head actor NextLat 1/1 |
| --- | ---: | ---: |
| Mean training bank | 53,873.11 | 15.17 |
| MC R² | 0.18376 | -0.00124 |
| Actor gradient norm | 1.44931 | 6.79757 |
| Accepted actor minibatches | 36 / 36 | 20 / 36 |

The actor auxiliary loss was **0.96666** (latent 0.31845, decoded 0.64811);
the separately logged PPO loss was **-0.09904**. Its maximum minibatch KL was
0.03142 and triggered the 0.03 actor-update stop. Separate combined/auxiliary
metric aggregations must not be subtracted to reconstruct PPO loss, and scalar
loss magnitudes alone do not establish gradient dominance.

All 36 critic and predictor updates ran in the final wave, but fewer actor
updates and radically different policy states make its ~10.68-second iteration
unsuitable evidence of a compute speedup. The run shows rapid policy collapse
under this unbalanced per-head 1/1 recipe, despite passing implementation
contract tests. Actor NextLat remains **off by default**; no coefficient search,
gradient-balancing change, or automatic restart was performed.

Actor-only league snapshots through iteration 32 are preserved. The only full
recovery checkpoint is initialization (`checkpoint-000000.pt`, also `latest.pt`);
there is no trained optimizer/critic recovery checkpoint to resume at iteration
32. No post-cancellation GPU evaluation was launched. Exact partial trajectory,
per-head losses, equal-wave control and cancellation status:
`artifacts/probes/entity-attention-d96-20260914/actor-head-nextlat-partial-results.json`.

The user subsequently ended this actor-auxiliary direction and requested work
on D96 execution efficiency and larger minibatches. Actor NextLat stays off;
the experimental implementation is retained without enabling it in production.

## 8192 default and learning-bottleneck analysis (2026-09-14)

The user selected **8192** as the new default. `PpoConfig.minibatch_size` now
owns that value for production/direct launchers and benchmarks; the forward
traffic probe follows it. Explicit overrides and historical fixed-shape probes
retain their declared sizes. Actor NextLat remains off, critic NextLat remains
1/1, and learning rates/objectives/precision are unchanged.

Six-repeat full production benchmarks **7145/7146** completed:

| Steady median | 6400 | 8192 |
| --- | ---: | ---: |
| Whole iteration | 10.95245 s | 11.19621 s |
| PPO update | 8.39137 s | 8.73563 s |
| Actor/critic/predictor updates per wave | 36 each | 29 each |
| Peak live VRAM | 16.33 GiB | 19.99 GiB |
| Peak reserved VRAM | 16.66 GiB | 20.53 GiB |

All intended updates and replay gates passed. This probe does **not** establish
a speed advantage for 8192; the selection is explicit user direction.
The 9600/11200 jobs **7147/7148** were cancelled before starting and not retried.
No new full learning run was launched after the user switched to analysis.

The partition is fixed-shape, not balanced: 28 full 8192 batches plus 704 genuine
states in the last batch, padded with 7488 zero-weight rows. Every genuine state
is used once. The last update normalizes over its genuine rows, so smaller-tail
variance and 36→29 optimizer steps are prospective learning changes, not free
throughput. Both historical architecture controls used 6400 and its 6080-row
tail, so this new batching detail cannot explain their observed difference.
Stale comments calling the partition balanced were corrected without changing
partition behavior.

Verification: five focused configuration/launcher tests passed; a non-model CLI
proof checked inherited8192, explicit6400 and complete 230080-state coverage.
The rewritten production profiler **7151** also completed cold, steady and
instrumented full fresh-wave updates using the unflagged8192 default: all29
actor/critic/predictor steps each time, zero allocator retries, ~19.99 GiB live.
Steady unprofiled update was8.7015s. Actual CUDA kernel attribution recorded
109364 launches; the largest attention-backward kernel consumed17.8% of summed
kernel time, largest attention-forward kernel7.4%, custom RMS forward5.7%.
These are instrumentation-derived kernel shares, not throughput or learning
measurements. Source/runtime evidence:
`artifacts/probes/entity-attention-performance-20260914/`.

### Matched learning evidence

The closest wall-time pair is entity115 at1283.507s and legacy104 at1283.207s,
only0.300s apart. Entity completed3456 actor updates versus legacy3384, despite
requiring19 rather than10 critic-only warmup waves. Delayed warmup alone does
not explain the later gap.

Trailing-ten medians ending at those checkpoints:

| Metric | Legacy D80 | Entity D96 |
| --- | ---: | ---: |
| Training money mean | 51,061.23 | 34,721.13 |
| Pre-update MC R² | 0.41973 | 0.33506 |
| Critic gradient norm | 69.685 | 1.391 |
| Actor component KL | 0.003744 | 0.001820 |
| Terminal residual EV, last32 transitions | 0.63498 | 0.30197 |
| Terminal residual EV,33–128 transitions left | 0.65218 | 0.37854 |
| Terminal residual EV,513+ transitions left | 0.15974 | 0.14211 |

These are different evolving policy-state distributions, not common-state
critic evaluation. Nevertheless the deficit is especially late-game, not just
long-horizon credit assignment. Lower gradient norms do not imply better fit.
Neither control had KL early stops, target saturation or an entropy collapse.
Entity policy drift was smaller under the same optimizer settings.

Entity starts with more training money (64,308.66 versus48,540.81 on wave1),
falls to25,589 at60, and stalls around35k; legacy also dips but recovers beyond50k.
This is not simply economically weaker initial BC. Entity's late action mix
has more hiring (21.5% versus17.7%), less selling (8.1% versus9.7%) and more
harvesting (3.58% versus3.10%). These fractions suggest investigating spending
and cash conversion, but do not establish profit mechanisms without per-step
cash/inventory/price accounting.

### Win rate and signed margin

Official panels use the same two development seeds/both seats and hash-matched
public opponents. Both models win4/12:4/4 starter,0/4 v27,0/4 v16.
Legacy104 mean signed final margins versus starter/v27/v16 are
90,203 / -51,107.25 / -67,193.75; entity115 gives
58,790.75 / -76,110.50 / -94,497.00. Thus every margin is worse for entity.
Summed final margin across the12 games is **-112,392 legacy versus-447,267 entity**.
There is no observed hidden win-rate or final-margin advantage.

This sum is not within-game time-integrated bank advantage, which was not logged.
Legacy per-game banks were not retained, so its mean normalized terminal margin
cannot be recovered from mean banks. Training `margin_abs_mean` is unsigned:
larger losses increase it. Self-play signed margins cancel across both seats,
and overall training score is mechanically0.4+0.2×league score.

Only two independent seed clusters/opponent and repeated development exposure
limit strength claims. Historical inference was portableCPU; entity evaluation
used compiledBF16GPU, so the panel is not an inference-controlled architecture
experiment. Training leagues also contain different learned opponents.

### Architecture hypotheses, not demonstrated bugs

The strongest readout hypothesis is the critic's scalar attention pool over
action-shaped entity states versus legacy's dedicated learned value-query
attention/FFN decoder over32 generic states. Global outcome information must
now be computed inside those entity states before pooling; a normalized linear
value head performs no subsequent relational computation. Active workspace is
ten market slots plus active units—only11 states with one farmer—not always26.
Four reasoning rounds, shared memory K/V and direct heads also change capacity;
D96 is not uniformly larger than D80.

Sparse ±1/0 outcome rewards and actor lambda0.97218 make reliable values crucial:
the direct terminal contribution scales as lambda^distance (~0.060 at100,
0.000211 at300 transitions). Accurate intermediate TD predictions can still
carry long-range credit; this is not a hard36-step planning limit. One shared
state advantage credits every sampled component, leaving a coordination problem.

Critic NextLat additionally shapes the same pooled belief using own actions and
detached successor targets. Possible interference with opponent-sensitive value
information needs source-gradient norm/cosine and persistence diagnostics;
coefficient1/1 or healthy total norms do not prove either benefit or conflict.

Best next discriminating measurement, if authorized: fit both critic architectures
to one frozen state/terminal-return dataset under the same continuation policy,
with held-out episodes and time-to-go strata, especially the last32/128 transitions.
Simply comparing existing critics against another policy's returns would confound
representation quality with the different value functions they were trained to fit.
Best isolated architecture hypothesis to test afterward: a critic-only learned
value-query attention/FFN readout over the unchanged entity states. Keep actor,
memory, rounds,8192,objectives and LR fixed. Consider more rounds only after
locating remaining actor/critic capacity error; do not automatically raise LR
because KL is smaller. No such architecture change was made during this analysis.
Verified numeric evidence:
`artifacts/probes/entity-attention-d96-20260914/learning-analysis-quantitative.json`.

## Single-query GQA critic and attention backends (2026-09-15)

The critic now reads its 26 entity states with one learned projected
four-query-head/two-KV-head query, followed by RMS normalization and the existing
HL255 head / critic NextLat consumer. No extra FFN, residual query shortcut, or
persistent core token was added. The standalone readout must not inherit residual
output-projection zero initialization: an explicit regression reproduced zero
context gradients before that initialization exception was fixed.

Actor parameters and BC initialization remain compatible. Scalar-pool critic
checkpoints are deliberately incompatible; resume those with their frozen source
or initialize the new critic. Production entity configurations reject MHA, and
critic NextLat also uses GQA. The training dependency floor is now PyTorch2.13,
matching the already locked and measured version.

### Measurement contract

CUDA BF16, compiled rollout/updates, D96/Hq4/Hkv2/head24, minibatch8192,
128 self-play plus64 league games, eight frozen BC opponents, and all719 steps.
Each wave retains230,080 genuine learner states and29 optimizer updates, including
the704-state tail with7,488 zero-weight padding entries. Actor NextLat remains off;
critic NextLat coefficients remain1/1. The actor initialization is
`runs/entity-attention-d96-bc-20260914/bc-actor.pt`.

After the user's runtime-budget correction, every performance/diagnostic job has
a hard **120-second MLQ cap**, exclusive max-parallel-runs1, priority0, one attempt.
Production probes use three full waves: one cold and two warm. Compilation,
startup and diagnostic parity checks count against the cap; queue wait does not.
Cold/profiled waves and diagnostic parity time are not steady production timing.
Earlier six-wave results remain historical evidence, not the new probe policy.

| Variant / MLQ job | Warm whole-wave median, s | Warm update median, s |
|---|---:|---:|
| Scalar-pool control /7157, five warm waves |11.140|8.658|
| New GQA readout, folded efficient /7159, five warm waves |11.765|8.832|
| Folded Flash unmasked + cuDNN masked /7185, five warm waves |10.919|8.500|
| Same hybrid, short control /7192 |10.876|8.463|
| Native CUDA embedding backward /7193 |11.501|8.661|
| Joint positional-embedding gradient /7198 |10.876|8.308|
| Integrated Flash/cuDNN dispatch /7201 |10.963|8.488|
| Folded GQA Flex TRITON masked updates /7202 |10.438|8.064|
| Native GQA Flex TRITON masked updates /7203 |11.003|8.262|
| Folded GQA Flex AUTO masked updates /7207 |11.345|8.582|
| Native GQA Flex AUTO masked updates /7208 |12.606|8.659|

Flex rows retain Flash/cuDNN rollout; only masked gradient-enabled calls change.
Their rollout fluctuations therefore cannot be attributed to Flex update kernels.
The folded TRITON candidate improves whole-wave latency by4.79% against7201 and
11.28% against the initial GQA efficient run. These are short single-seed
measurements, not a claim of statistical certainty or better learning.

### Why native Flash initially looked worse

The original native-Flash-unmasked rollout timings were
3.510,2.270,12.976,9.612,7.755 seconds. The apparent multi-second penalty was not
reproduced; its cause is unestablished. Subsequent full-wave traces showed the
same390,011 rollout kernels and719 CUDA-graph launches across efficient, native
Flash and folded Flash: no observed eight-opponent loop fallback.

Flash itself helps. Matched-layout CUDA-graph diagnostics measured farm attention
forward at2.457ms efficient versus1.007ms folded Flash for B16384,Q100,K100.
Native Flash GQA backward, however, expands dK/dV to query-head count and then
reduces groups. At B8192,K220,Hq4,head24 those two BF16 intermediates total660MiB.
Folding query groups avoids that expansion without repeating K/V. In profiled
updates, Flash main-backward kernel time was1,587.7ms native versus552.3ms folded.
Profiled timings establish attribution, not unprofiled throughput.

### Masks and FlexAttention

Production masks are key-validity vectors broadcast across heads and queries:
26 keys for entity/readout attention and236 keys for critic memory. Their storage
is small, but removing them changes the attention distribution. Inactive entity
states are explicitly zeroed after each reasoning branch; merely deleting masks
does not create persistent latent workspace. No mask-removal ablation was run.

The selected Flex candidate fuses boolean validity into dense attention scores,
without constructing dynamic sparse-block metadata. Standard TRITON beat AUTO
and folded GQA beat native GQA in these update measurements. Masked no-grad rollout
retains cuDNN: a compiled eight-lane Flex probe failed with
`KeyError(TransformType.Vmap)` in the installed PyTorch2.13 higher-order operator.
The rollout was not replaced with a loop or an eager/FP32 fallback.

Correctness evidence:

- 7166 reproduced the zero-initialized standalone readout gradient failure;
 7167 passed all13 entity/config contracts after the fix.
- 7176 passed compiled Flash forward/gradient oracles; maximum relative gradient
 L2 error0.003396 versus an independent FP32 GPU numerical reference.
- 7196 passed cuDNN production-shape oracles, including Q1/K26 readout and
 Q1/K27 critic NextLat; masked K/V gradients and perturbation invariance passed.
- 7200 passed15 generic CUDA SDPA contracts, including head/query-specific masks,
 fully masked rows, noncontiguous projections and padded head widths.
- 7206 passed native and folded Flex oracles for entity, private-memory and value
 attention, including fully masked rows and inactive-key perturbations. Maximum
 relative gradient L2 error was0.003176 overall,0.002969 for folded Flex.
 The first oracle7204 hit the test's specialization limit; resetting each
 deliberately distinct case fixed the harness without relaxing model limits.

Rejected codegen experiments were removed from working source; their immutable
snapshots remain reproducible. Native embedding backward was slower. Joint
positional gradients preserved forward outputs bit-for-bit and matched gradients
within relative L2 error0.00001625, but produced no whole-wave gain. Pattern
matching7194 reached the120-second cap before a measured wave. Long autotuning
and superseded long probes were cancelled rather than extended.

Evidence and exact frozen-source/job identities:
`artifacts/probes/critic-gqa-readout-20260915/contract-verification.json`,
`flash-kernel-diagnostic.json`, `flash-trace-analysis.json`,
`short-probe-evidence.json`, and `flex-investigation-findings.json` in that same
directory. These are performance and correctness experiments; they do not
establish a learning improvement, and no new full training run is represented.

### Integrated rerun and final contracts

The unwrapped production benchmark7212 completed under the120-second cap:
warm waves10.6084/10.5678 seconds, updates8.1770/8.1988 seconds. Medians are
**10.5881 seconds/wave and8.1879 seconds/update**, a3.42% whole-wave latency
reduction against the integrated Flash/cuDNN control7201. This reproduces the
direction of the candidate result, not its exact4.79% magnitude.

The final entity/config test batch7213 passed five cases before its120-second
cap. Remaining contracts were partitioned rather than extending the cap:
7215 passed the combined compiled PPO/actor-NextLat/critic-NextLat backward
contract;7216 passed seven artifact, critic-checkpoint, full719-step PPO replay,
and registry checks. Generic CUDA mask suite7214 passed all15 cases against
the actual default dispatch. Two independent read-only reviews reported no
concrete findings. Scoped Ruff checks and dependency lock resolution passed.

A fresh efficient-only control7217 on the **same final frozen source** measured
10.9083/10.9703 seconds/wave and8.4875/8.5154 seconds/update: medians10.9393
and8.5015 seconds. Against that control the retained dispatch reduces whole-wave
latency by **3.21%**, not the10–11% suggested by comparing against the earlier
efficient run. The source of that earlier timing drift is unestablished; use
the final matched control for the headline gain. This is a modest improvement,
not the requested substantially faster whole-model result, so the conditional
full training/comparison has not been launched.

### CUDA-graph follow-up (2026-09-15)

The active learner was already inside the mixed-wave inference CUDA graph,
together with the eight-lane frozen-policy forward, gather/scatter, and stream
join. A wave records once and replays for all719 real steps. Native stepping,
preference generation, and policy-statistics transfer remain outside that graph.

All follow-up GPU probes use exclusive MLQ admission, priority0, one attempt and
a hard120-second limit. They retain192 physical games,320 learner trajectories,
230080 genuine states, B8192,29 optimizer updates, actor auxiliary off and critic
auxiliary1/1. Unless explicitly identified as an operator or inference-only
diagnostic, timings include the full rollout and PPO workload.

Enabling `reduce-overhead` initially failed:7218 and regression7221 reproduced
actor gradient storage being recycled by a subsequent critic graph before the
guarded actor optimizer consumed it. One explicit graph iteration now spans the
entire PPO minibatch. Replay-only loops advance independently and behavior-value
replay copies each result into owned output storage. Regression7222 passed;7220
completed three full waves without the graph-lifetime warnings. Its warm median
was10.5519s total and8.1134s update, versus10.5881s/8.1879s for the earlier
compiled-default control. This is not evidence of a large graph speedup.

Additional experiments were tested, then removed from production source:

| Experiment | Evidence | Decision |
|---|---|---|
| Separate preference CUDA graph |7223:2 exact RNG contracts passed;7228:all rollout fields except elapsed time matched across the full230080-state wave | No repeatable whole-wave win |
| Interleaved preference capture |7241:three warm samples per arm,2.3336s captured versus2.3982s uncaptured median; isolated7231/7232 full-wave comparison favored uncaptured10.3874s versus10.7330s | Timing variation exceeds a convincing gain |
| Compiled NorMuon variance normalization |7233:72 production-shape numerical cases passed, maximum relative L2 error1.414e-7;7239:6 CUDA grouping/gating cases passed |7240 update8.2820s versus8.1105s sampler-only control; no accepted speedup |
| Larger RMS row blocks |7235:update8.1516s, no improvement;7238:12 of6144000 BF16 outputs differ from the old reduction layout | Reject |
| Re-enabled compiler pattern matcher |7242 reached120s before completing a measured wave, with cast emulation retained | No qualified result |

The optimizer experiment first exhausted Dynamo's specialization cache.
Guard logs7236 identified view `_base` ancestry, singleton batches, optional
gates and square-shape equalities. Normalizing layout and detaching no-grad
optimizer views fixed the test and full workload without raising cache limits
or falling back to eager model execution. The implementation was nevertheless
discarded because the measured performance did not justify it. The strengthened
CUDA optimizer contract remains useful independently of that experiment.

Independent read-only reviews of the retained PPO graph changes and the
experimental sampling graph found no actionable defects. The permanent native
graph regression exercises the original failure within a minibatch; the full
production benchmark additionally exercises repeated graph replay over29
minibatches per wave. Scoped Ruff checks passed.

Exact commands, immutable source identities and detailed evidence are in
`artifacts/probes/critic-gqa-readout-20260915/cuda-graph-investigation.json`
and its referenced MLQ jobs. The preference equivalence hashes are in
`full-rollout-sampling-equivalence.json`; interleaved inference timings are in
`sampling-graph-interleaved.json`. These results alone make no learning claim.

### Retained short-attention tuning and learning launch

Flex's Blackwell backward defaults use128-row/column tiles for narrow heads.
Uniform32×32 and64×64 candidates retained all masks and passed15 CUDA
forward/backward contracts each (7245/7246). Six complete alternating waves in
7247 separated the first invocation of each configuration from warm measurements:

| Warm update | Backend defaults |64×64 tiles |
|---|---:|---:|
| First measured wave |8.1319s |7.9169s |
| Second measured wave |8.0725s |7.9268s |
| Median |8.1022s |7.9218s |

That is a2.23% update-time reduction. Whole-wave medians were10.3273s versus
10.2252s (0.99%), with rollout variation working against the tuned arm. The
retained override applies only to padded head widths<=32, folded query
lengths<=64 and key lengths<=256; other shapes use backend heuristics.

The final unwrapped production benchmark7250 measured10.4204s total and8.1522s
update medians, with no graph warnings. Final-source mask contracts7251 passed
all15 cases, and native719-step graph/PPO regression7252 passed. Before the tile
change,7249 also passed the native regression plus six CUDA optimizer
grouping/gating cases. No large inference-graph speedup is claimed.

The latest request authorized a full learning run once the execution path was
credible. Following that validation, job7254 launched the500-iteration budget
at `runs/production-entity-gqa-d96-outcome-graphs-b8192-20260915`, using immutable
source digest `0a290af7ebb05af23043d857d1e3150b781853a6ce30e194bc4b42541c85ddf3`.
It uses the existing BC actor, a fresh single-query critic, B8192, actor
NextLat0/0, critic NextLat1/1, and `reduce-overhead` updates. Learning rates remain
actor0.00015, critic0.00015 and critic head0.0004375; objective and GAE settings
are unchanged. Each chunk has a25-minute MLQ cap and0.33-hour graceful boundary,
with420-second checkpoints and the existing checkpointed plateau guard.

The first launch7253 failed argument validation before training because an
added optional source-digest flag requires a calibration-decision file. The
corrected direct launch omits that unpaired flag, like the historical run;
immutable-source provenance remains recorded. Job7254 completed19 iterations
and released the actor at iteration18. Its first replay audit had maximum
KL8.5335e-11 and zero tail violations. These establish live training, not a
learning improvement. The historical comparison uses B6400 and scalar pooling,
so it is descriptive rather than a controlled critic-only ablation.

### Full learning outcome and held-out check

Job7254 plateau-culled at iteration70, rather than exhausting its500-iteration
budget. Reported training time was823.245s (13.72 minutes), and MLQ process time
was834.054s. Exit75 is the configured intentional prune, not an execution crash.
The actor had53 active iterations after release at18. The guard recorded30
stale observations, with EMA money30085.37 and value loss2.99698 versus its
reference39034.22/2.97701. No automatic restart was launched.

The historical complete iteration immediately preceding that elapsed time was
iteration73 at815.414s. Last20-iteration means were:

| Online proxy | New run | Historical matched-time window |
|---|---:|---:|
| Money |28234.52 |25743.58 |
| Value loss |2.99780 |3.00483 |
| Monte Carlo R-squared |0.34575 |0.34560 |

Those proxies do not establish a playing-strength improvement. Job7256 loaded
the saved actors and evaluated each against the identical BC initializer on256
reserved development seeds4000000–4000255. Both used CUDA BF16 compiled
`inductor_graph` inference, temperature1, the full719-step native engine and
balanced seats by seed parity—not both seats of each seed. Initializer SHA256
matched between training runs. The nearest available historical recovery
checkpoint was iteration76 at850.730s,27.485s beyond the new checkpoint.

| Held-out native result versus BC | New iteration70 | Historical iteration76 |
|---|---:|---:|
| Wins / games |96 /256 |92 /256 |
| Win rate |37.50% |35.94% |
|95% Hoeffding interval |29.01–45.99% |27.45–44.43% |

The paired score difference was+1.56 percentage points, with a conservative
95% interval of−15.41 to+18.54 points: no demonstrated advantage over the
historical policy. Both policies underperformed BC in this check. This is native
development evidence, not Kaggle evaluator or submission-admission evidence.
Do not promote the new checkpoint over BC on these results.

The saved recovery artifact is
`runs/production-entity-gqa-d96-outcome-graphs-b8192-20260915/checkpoint-000070.pt`
(also `latest.pt`), SHA256
`8ef0a2b050c2cd59fd6353ae52b36023d9deccb86a00306e329055a7ea5c87a4`.
7255 read the version17 recovery payload;7256 strictly loaded critic and
predictor state, confirmed optimizer/RNG and cull-state presence, and exercised
the restored actor in complete games. A resumed optimizer step was not tested.
Detailed results are in `learning-comparison.json` and
`learning-native-evaluation.json` beside the graph investigation artifact.

### User-directed continuation and initial-decline diagnosis

The user requested continued training because the new entity policy is
recovering better than its predecessor. The prior plateau-pruned outcome is not
a conclusion that recovery had stopped: money EMA rose from28096.11 at
iteration65 to30085.37 at70, but the guard still compared it against39034.22
and therefore incremented patience throughout that recovery.

Job7257 produced `checkpoint-000070-no-autocull.pt` beside the original
checkpoint, changing only `training_data_config.autocull` to null. A recursive
round-trip comparison verified exact equality of every other recovery field,
including all model, optimizer, RNG, league and warm-start state. The original
checkpoint remains unchanged. `continuation-checkpoint.json` records both
digests and the explicit user authorization.

Job7258 initially resumed that state into
`runs/production-entity-gqa-d96-outcome-graphs-b8192-20260915-continued`.
It keeps the500-total-iteration target, disables the per-process hours cutoff
and the online plateau rule, and retains all other training/numerical guards.
MLQ admits one job at a time, priority0, with a2-hour hard cap. Periodic recovery
checkpoints remain420 seconds apart. The first resumed iteration71 completed
29actor updates with maximum replay KL2.92452e-11. Reported elapsed time resets
for each continuation process; add the original823.245s to measure accumulated
training along a successfully continued checkpoint lineage.

The matched iteration61–70 windows substantiate better recovery:

| Metric | New entity | Previous entity |
|---|---:|---:|
| Self-play money |30227.06 |26696.15 |
| League money |29178.82 |25827.22 |
| League score |0.44844 |0.41406 |

These are approximately13% improvements in both money measures. They remain
online, moving-opponent measurements, not a controlled architecture ablation.

The leading initial-decline hypothesis is immature long-horizon credit, rather
than a numerical failure or entropy collapse:

- Actor release at18 used prior-wave aggregate Monte Carlo R-squared0.10929.
  During the first10actor-active waves, aggregate R-squared averaged0.20861,
  yet explained variance with more than512steps remaining averaged only0.02107.
- Actor GAE lambda0.972183588 gives a35.95-step geometric horizon in719-step
  games. The direct terminal coefficient at719steps is1.59687e-9; early action
  credit consequently depends heavily on successor-value estimates.
- The32-step actor LR warmup counts optimizer minibatches, not rollout waves.
  It ends on the third minibatch of the second actor-active wave. Critic-only
  warmup does not consume the actor schedule, so this is not a shared-clock bug.
- Entropy rises into the money trough and falls during recovery. The decline
  also exists in self-play, so changing league composition cannot fully explain
  it. Aggregate value accuracy alone cannot prove harmful action-conditional
  advantage bias.

Do not alter this continuation to test the hypothesis. The discriminating
follow-up is a same-rollout, per-time-to-go comparison of current GAE versus
Monte Carlo actor-gradient alignment, followed by a separate lambda1 ablation
if warranted. That removes successor-value bootstrapping bias but increases
variance; merely increasing critic warmup or lowering LR is not yet justified.
Numeric windows, source interpretation and launch details are preserved in
`initial-decline-diagnosis.json` and `continuation-launch.json`.

The first continuation exposed a runtime-accounting defect: at iteration72,
`CompileWatch` called three CUDA-graph runtime records late compilations. Torch
stores runtime overhead in the same `CompilationMetrics` stream and marks those
records `is_runtime=True`; they had no compiled frame or guard failure. The
source fix excludes runtime records from compilation classification rather than
disabling the late-compile guard. Job7259 reproduces the exact pre-fix failure
using Torch's own runtime timer; job7260 passes all4compile-watch contracts,
including genuine late-first-compile and guard-failure rejection.

Only `src/kaggriculture/compilewatch.py` differs in frozen runtime source
`10eccba60a93577b72390000e1a217902773758877e58a4b0edabf8d992ca489`.
Job7261 explicitly rebound the uncullable iteration70 checkpoint to that source,
verifying every other recovery field remained exactly equal. Job7262 then
correctly rejected the prior continuation directory's conflicting old-source
checkpoint70; no checkpoint was overwritten to bypass that check.

Job7263 starts in a fresh directory,
`runs/production-entity-gqa-d96-outcome-graphs-b8192-20260915-continued-runtime-fixed`,
from `checkpoint-000070-no-autocull-compilewatch.pt`. It retains the same
500-iteration target,2-hour MLQ cap, exclusive admission, BF16 compiled execution
and disabled plateau stopping. The original and failed continuation records are
preserved. Exact submission and source-rebind evidence are in
`continuation-runtime-fixed-clean-launch.json` and
`continuation-runtime-fix-checkpoint.json`.

Verified job7263 through iteration74: iteration71 completed cold setup and its
fresh replay audit; iterations72–74 each applied29actor updates with settled
shapes and zero compiler events. Training continues beyond that verification
boundary. The runtime fix changes neither gradients nor the actual compile
guard's treatment of new frames.

## Rejected critic-only inverted memory attention (2026-09-15)

**Rejected by the user as worse.** Job7286 was cancelled by request (exit143).
The option, operator, special compatibility handling and experiment-only tests
have been removed from live source. The following is historical evidence for
the frozen experiment, not a supported architecture or a future trial.

This ablation follows *Inverted-Attention Transformers can Learn Object
Representations: Insights from Slot Attention*. Only the entity critic's four
static-memory reads change. Per query head, memory tokens compete across valid
entity queries, then each query normalizes its allocated memory weights.
Actor attention, critic self-attention, FFNs/residuals, shared memory projection,
and the learned one-query GQA value pool remain unchanged. The flag
`--critic-inverted-attention true` is off by default.

Frozen implementation:
`30f6f00255f6be0521646a3a12c8bc2ac0af0946f849693e4522e2b7253ba8c0`.
The Triton operator differentiates both normalizations, keeps query-head
competition separate under GQA, and avoids full attention matrices and expanded
K/V gradient buffers. Independent math review found a CUDA grid-Y overflow:
job7271 reproduces the old launch failure at131072 lanes; flattened grid-X
launches pass that same large-batch, zero-stride-input regression.

Verification on the frozen implementation:

- Job7272: six numerical/masking/gradient regressions pass, including GPU
  float64-oracle comparisons for FP32 and BF16, per-head GQA competition,
  singleton mean-pool identity, common-key-logit cancellation and extreme
  competition without underflow.
- Job7273: changing-mask fullgraph CUDA-graph replay and gradients pass.
- Job7276: four inactive-unit/privacy/live-gradient contracts pass.
- Job7274: complete719-step native games, replay parity and a real
  BF16 `reduce-overhead` actor/critic/critic-NextLat PPO update pass.
- Isolated fixture failures in7275 and7280 are preserved: the first omitted
  the entrypoint's sibling `autocull_hook`; the second synthetic untrained
  actor omitted required `seed_usage`. The fixture now links the complete
  frozen scripts directory and declares empty exposure; production admission
  checks were not relaxed.

The user subsequently authorized replacing the running normal-attention
continuation. Job7263 was cancelled by request after preserving its full
iteration420 recovery checkpoint in the original run directory:
`checkpoint-000420.pt`, SHA256
`2b6479155012aeb77bf7055625b765f8a059d523a509d9c7256cd8bc7d072156`.
The file was CPU-deserialized to verify actor, critic, optimizer and RNG state.
Metrics reached440; those later20iterations are not checkpointed. Resume the
preserved checkpoint with its original frozen source `10eccba60a93577b…`;
do not reinterpret its critic weights as an inverted-attention recovery.

### Measured final value pooling

Job7279 uses that iteration420 baseline, compiled CUDA BF16 inference and
256 complete development games against the same BC opponent. It samples32
evenly spaced states per game:8192 states from184064 genuine transitions.
The final pool is selective, not approximately uniform:

| Metric | Four learned query heads |
| --- | --- |
| Mean normalized attention entropy; uniform=1 | 0.681,0.684,0.717,0.757 |
| Mean largest attention weight | 0.268,0.365,0.330,0.249 |
| Mean uniform weight over valid states | 0.05565 |
| Effective slot fraction | 0.399,0.402,0.444,0.495 |

Replacing only the final pooling weights with valid-state uniform weights,
while retaining projected values, output projection, normalization and HL head,
changes scalar value predictions by0.5993 on average and1.5711 at p90 in model
value units. This measures representation/value sensitivity, not PPO learning
or playing strength under a different critic. A learned one-query pool is not
guaranteed optimal, but inverting its query axis would necessarily remove its
selective weighting.

Commands, preserved failures, checkpoint identity and stratified pooling
measurements: `artifacts/probes/critic-inverted-attention-20260915/`.

### Verified throughput and authorized replacement

Job7283 passes all six actor-identity, historical warm-start, frozen-opponent
and strict-recovery contracts. Together the focused GPU selection passes18
checks; two metadata-only registry/CLI checks also pass.

Jobs7284/7285 complete the matched128-self-play-plus64-league production
benchmark, with230080 genuine states and29 actor/critic/NextLat minibatches
per iteration. Both use the same frozen source and BC bytes
`a1f152531b3eea6361e1549636a7bb9a9d2a6e8ce24063a3a71dd1a9dcde99ee`.
Each has one cold and two warm repeats:

| Median warm time | Normal critic | Inverted critic |
| --- | ---: | ---: |
| Complete iteration | 10.5966s | 11.4546s |
| PPO update | 8.2047s | 9.1266s |
| Rollout | 2.3198s | 2.2632s |

The observed complete-iteration overhead is8.10%; two warm repeats do not
establish a precise long-run speed distribution or any learning advantage.

MLQ7286 ran in
`runs/production-entity-gqa-d96-critic-inverted-outcome-graphs-b8192-20260915`.
It starts from the same BC actor with a fresh critic and optimizers, retains
seed20260812, D96/4Q/2KV, HL255/sigma3, B8192, the same learning rates,
actor/critic lambdas and critic-only NextLat recipe, and changes only the
critic internal attention operator. Target500, no autocull, `max_hours=0`,
two-hour queue cap, priority0, exclusive parallel limit1, one attempt.

Initial startup verification covered iteration6:29 critic updates per wave, actor still frozen
under the existing critic-warmup gate, latest HL loss3.28137. The actual
recorded configuration confirms the inverted flag, strict frozen source,
B8192 and disabled autocull. This is startup/update evidence, not an outcome
claim; the run was subsequently cancelled rather than completing500 iterations. See `learning-launch.json`,
`learning-startup-verification.json` and `validation-results.json` in the
campaign artifact directory above.

## Entity cross-attention and intermediate FFN cost (2026-09-15)

Ordinary attention is restored in source
`ed6361247fcce4e2b45aa83eb5ee895e597087c37b7c9ff865f615235d2a66b1`.
Jobs7287/7288 pass five existing native-PPO, privacy/masking and artifact
contracts; scoped Ruff checks pass. No rejected operator/configuration references
remain in live source, scripts, tests or README.

Each actor entity reads all220 memory tokens:200 farm tiles and20 economy
tokens. The critic adds16 masked private opponent-unit tokens. HERE/N/S/E/W
gathering seeds the unit state; it does not restrict entity cross-attention.
The two farm-encoder blocks themselves attend across each entire100-tile farm.
All four entity rounds share the projected memory K/V, with independent
query/output projections and evolving entity states.

A runtime-only timing prototype inserts an independently parameterized,
pre-RMS, gated, economy-conditioned96→192→96 FFN between SA and CA in each
of four rounds, for both actor and critic. Added output projections are zeroed
for identity-preserving BC initialization. It adds223872 parameters per network
including conditioning, and7.667712 million FFN matrix FLOPs per state/network.
That timing-only prototype and its compatibility loader were not retained.
The subsequent campaign below implements an independently configurable branch
with fresh matching BC, rather than the prototype's identity-preserving warm start.

All timing workloads retain230080 genuine states and29 actor/critic updates
per iteration, compiled BF16, and the production league layout. Jobs7289/7291
complete the initial matched three-iteration workloads. Job7290 emitted all
three iterations and `benchmark_complete`, but subsequently hit its120-second
process deadline; its record is preserved rather than called a successful job.

To investigate variable post-update rollout time, jobs7292/7293 each complete
five full iterations. The comparison window was specified before submission:
discard repeats0/1 in both arms and use repeats2/3/4. PPO-update medians are
8.09831s ordinary versus8.43001s with the extra FFNs: **+4.10%**.
The earlier successful pair found+5.51% update overhead.

Whole-iteration timing is not resolved cleanly: rollout medians varied from
roughly2.4s to3.7s across arms, reversing the apparent complete-iteration
comparison. Do not interpret that reversal as a speedup from adding FFNs, or
attribute its cause without a separate measurement. A5–10% whole-iteration
overhead is a planning estimate, not a measured conclusion. Learning benefit
was not tested; no new learning run was launched.

Commands, arithmetic, individual timings, complete raw summaries and the
predeclared comparison window:
`artifacts/probes/entity-cross-attention-review-20260915/`.

## Piecewise entity architecture ablations (2026-09-15)

Five independently selectable flags retain the production defaults: round-local
projected K/V, an FFN between entity SA and CA, a final actor-unit local readout,
a critic FFN after learned-query pooling, and removal of local unit initialization.
The last contrast compares local-readout/init-off against local-readout/init-on;
all other contrasts use the unchanged control. Common source normalization remains
shared even when projected K/V are untied. No flags are automatically combined or
promoted.

Final learning source:
`02cfc15b85047c3cde4cfcb6af0775c8083384184fac7e9bdbad5aa3600b4421`.
Run roots: `runs/entity-piecewise-20260915/<arm>/`.
Commands, gates, source identities and analysis protocol:
`artifacts/probes/entity-piecewise-ablation-20260915/campaign.json`.

Each actor-changing arm receives the original full two-epoch BC recipe over the
same four corpora, 299104 training rows and69024 held-out rows. The critic-only
FFN reuses the control BC actor. BC jobs7305/7307/7308/7309/7335 succeeded;
the earlier untied initializer7306 is superseded, not reused.

PPO retains the recorded production hyperparameters, seed20260812,
128 self-play plus64 league games,720 ticks,230080 genuine states per wave,
B8192, compiled BF16, and the existing auxiliary objectives. The user-specified
budget is enforced as `--max-hours 0.4` with a25-minute MLQ hard cap, exclusive
parallel limit1 and one attempt. The500-iteration argument is only an upper
bound. Benchmark and evaluation jobs have120-second caps. Each learning arm is
followed by its evaluation before the next arm; terminal dependencies retain
ordering without allowing a skipped intermediate job to bypass earlier work.

Full-size benchmark gates all succeeded. Each contains one cold and two warm
iterations, with29 actor and29 critic-auxiliary updates per iteration:

| Arm | Benchmark job | Warm update median | Relative to reference |
| --- | ---: | ---: | ---: |
| Control |7312|8.18429s|—|
| Untied K/V |7336|9.07567s|+10.89%|
| Intermediate FFN |7314|8.97406s|+9.65%|
| Final local readout |7315|8.35147s|+2.04%|
| Critic FFN |7316|8.14386s|−0.49%|
| Local readout without local initialization |7317|8.52603s|+2.09% versus local readout|

These are two-sample warm timing observations, not precise speed distributions;
small differences do not establish a speedup or slowdown. Benchmark-only
numerical replay audits are outside the production-iteration timer.

The local decoder exposed a real production-size CUDA grid-Y failure at
8192×16 unit rows. Masked FlexAttention now uses32768-row attention views above
65535 batch rows, preserving the full PPO batch, BF16, masks and gradients.
Regression7300 fails before the fix with grid_y65537 and an invalid launch;
7301 passes the large-batch gradient and no-grad cases, and7302 passes the actual
131072-row decoder forward/backward and inference replay. Feature contracts
7295–7298 and the final shared-normalization/untied-projection equivalence test7334
also pass. The initial untied benchmark7313 hit its120-second deadline and is
preserved as a failed, superseded gate, not counted as successful evidence.

The fixed development panel uses256 full games each against native `starter`
and `scripted-v27`, seeds4500000–4500255, balanced seed-modulo-two seats and
sampling seed20260917. Seats are paired across arms, not both seats on every map.
Primary comparison is mean score rate, matching the terminal win/loss/draw
objective; money, lower-tail money and change from each arm's BC are secondary.
Paired95% intervals resample whole seed clusters containing both opponents.
These quantify game-seed uncertainty, not training-seed uncertainty, and are
exploratory without multiple-comparison adjustment. This is a native development
panel, not the pinned Kaggle final evaluator. Protocol gate7319 reproduces
identical money when the same artifact is evaluated twice.

Control7339 completed134 iterations in24.1058 training minutes and evaluation7340
succeeded. Score rate fell from60.55% for its BC actor to49.61% after PPO:
98.83% against starter and0.39% against scripted-v27. Aggregate mean money was
55664 versus66411 for BC. Higher moving-league money is therefore not sufficient
evidence of stronger fixed-opponent play.

The production-continuation comparison found identical PPO configuration and
simulator source, but different BC weights and reset critic, optimizers and league
state. Run variation is plausible, not causally established. The user accepted
that possibility; preserved production checkpoints107 and420 receive the same
fixed panel rather than attributing differences in moving-league curves to a new
architecture. Detailed evidence is in `production-discrepancy.json`.

### Intermittent GPU-idle intervals in untied K/V

The live run's compile watcher identifies late first compilation of newly reached
stacked league-inference layouts, not guard-triggered recompilation. At
iterations34/36/37/39 these cost6.04/6.39/4.56/6.60 seconds respectively and extend
rollout to6.87–9.53 seconds while PPO updates remain about9 seconds. Normal
post-start medians through iteration61 are2.364s rollout,8.640s update and11.167s
total; staging is0.069s. Post-start compilation totals48.769s through that point,
with no compilation in iterations40–61. The largest unaccounted iteration-boundary
gap is0.227s, so multi-second pauses are not explained by unmeasured checkpoint
or logging work in this window.

`_stacked_actor_ensemble` caches by physical lane count and `_compiled` by
per-lane width. Previously unseen power-of-two layout combinations therefore
compile as league assignments change. Repeated `_w16` names need not denote the
same frame: separate lane-count ensembles own separate code objects. The frozen
trials retain this behavior and count its cost against their wall budget; no
opponent-distribution or batch-size change is used to conceal it.
Evidence: `untied-gpu-idle-diagnosis.json` in the campaign artifact directory.

### Untied K/V outcome

Learning7344 and evaluation7345 succeeded. The arm completed124 iterations
and3103 actor updates in23.9995 recorded training minutes, versus134 iterations
and3364 actor updates for control. Fixed-panel score is49.02% versus49.61%;
the paired difference is−0.586 percentage points,95% interval
[−1.758,+0.586]. Mean money is46719 versus55664: difference−8945,
95% seed-cluster interval[−12191,−5770]. Starter score is98.05% and scripted-v27
score is0%. Its early moving-league lead did not survive the fixed panel.
This single matched-budget trial does not support promoting untied projections.

### Separate action API corrections

The architecture trials retain their original frozen source. Separately,
`MarketLedger` and its existing quote/mutation routines move from `policy.py`
to `actions.py`; public market helpers now use those same dynamic quotes.
All sampler, demonstration and probe imports migrate without compatibility
re-exports. The unit snapshot-mask docstring no longer claims sequentiality.
Supplied nondefault `shedCapacity` is rejected by `CheckpointAgent` and the
generated submission entrypoint; observation-only and batched inference still
assume the default capacity100. No primitive vocabulary, movement economics,
DROP behavior, native game rules, model defaults or current training recipe changes.

Scoped Ruff checks pass. Model-free action, demonstration, market-ledger and
configuration regressions:194 passed in15.12s. MLQ7357 passes real compiled CUDA
BF16 inference with omitted/default configuration, executes the generated agent
function against the real GPU agent, rejects capacities1 and200, and produces
an action accepted by the reference engine. This checks the changed entrypoint,
not a new full submission bundle or CPU model execution. Validation source:
`e7a45327751fbf0ab977978fda564292ff3ec34f70046bf2d3cbe2b7cfae4588`.

Action-space recommendations remain separate research decisions. Partial wheat
placement is genuinely absent and can preserve feed stock; larger fertilizer or
animal pickups can enable routes that their current bins exclude. Oversized
clamped pickup aliases add no new transfer where smaller exact bins already
exist. A future target-tile head should marginalize into primitive movement
probabilities rather than teleport, silently route multiple turns, or invent
ambiguous BC target labels. DROP retains its actual destructive semantics and
useful multi-item deposit. No blanket strategy-pruning mask is added.

A model-free native counterfactual at final action718 confirms that a hire
creates one hand and spends1 money while both games immediately terminate at719;
this terminal boundary is distinct from the ordinary end-of-day boundary.
That establishes a zero-future-action case, not its frequency or economic
importance in learned play. A later useful-hire restriction must be a separate
sampler-support experiment, applied before sampling in both implementations and
reflected in stored masks and BC handling. Decisions and proof:
`action-recommendations.json`, `final-action-hire-probe.json`,
`action-api-validation.json` and `action-api-smoke.json`.

### Intermediate FFN outcome

Learning7346 and evaluation7347 succeeded:128 iterations,3219 actor updates,
24.1655 recorded training minutes. Fixed-panel score is49.41% versus49.61%
control; difference−0.195 percentage points,95% interval[−1.172,+0.586].
Mean money is59265 versus55664: difference+3601,95% interval[+544,+6474].
The money10th percentile rises from11801 to21194. Against scripted-v27,
mean money improves by10146,95% interval[+6941,+13310], but score remains0%.
This is a secondary money/distribution signal, not an observed win-rate benefit
or a default promotion. Warm update overhead in the full-size gate was9.65%.

### Final local unit readout outcome

Learning7348 and evaluation7349 succeeded:127 iterations,3219 actor updates,
24.1260 recorded training minutes. Fixed-panel score is48.44% versus49.61%
control; difference−1.172 percentage points,95% interval[−2.344,−0.195].
Mean money is29281 versus55664: difference−26383,95% interval
[−29369,−23444]. Starter score is96.88% and scripted-v27 score is0%.
The matching BC actor already had weaker fixed-panel score (53.71% versus
60.55% control BC) despite slightly lower held-out NLL. This is therefore an
end-to-end fixed-BC-recipe comparison, not an isolated effect on PPO from
behaviorally identical initial policies. The trial does not support promotion.

### Critic readout FFN outcome

Learning7350 and evaluation7351 succeeded:139 iterations,3596 actor updates,
24.0300 recorded training minutes. This arm reuses the exact control BC actor.
Fixed-panel score is49.61%, identical to control; paired score difference0.000
percentage points,95% interval[−0.977,+0.977]. Mean money is61208 versus55664:
difference+5544,95% interval[+2277,+8763]. Starter score is99.22% and
scripted-v27 score is0%. The full-size warm update gate was effectively flat
(−0.49%, within two-sample timing noise). This is another secondary money
improvement without an observed win-rate benefit; defaults remain unchanged.

### Removing local initialization with local readout enabled

Learning7352 and evaluation7353 succeeded:135 iterations,3161 actor updates,
24.0610 recorded training minutes. The first actor update occurs at iteration27,
versus17 for local readout with initialization; these are fixed-time comparisons,
not equal-optimizer-work comparisons. Matching BC score is56.05%, versus53.71%
for local readout with initialization.

Against the required local-readout reference, final score is48.83% versus48.44%:
difference+0.391 percentage points,95% interval[−0.977,+1.758]. Mean money
improves from29281 to36888: difference+7607,95% interval[+4765,+10397].
Both policies score0% against scripted-v27. The no-initialization arm remains
below the original control in mean money by18776,95% interval[−22235,−15258];
its score difference from control is−0.781 points,95% interval[−2.148,+0.391].
This recovers some money relative to the weak local-readout arm, but does not
justify replacing the original architecture.

### Original six-arm pilot disposition

All six learning jobs and six fixed-panel evaluations succeeded on their first
attempt, with25-minute learning and2-minute evaluation hard caps. Queue proof:
`original-campaign-terminal-jobs.json`. Complete paired comparisons and BC/PPO
distributions are in `outcome-summary.json`.

| Arm | Iterations | Fixed-panel score | Mean money | Warm full-size update |
| --- | ---: | ---: | ---: | ---: |
| Control | 134 | 49.61% | 55664 | 8.184s |
| Untied projected K/V | 124 | 49.02% | 46719 | 9.076s |
| Intermediate FFN | 128 | 49.41% | 59265 | 8.974s |
| Final local readout | 127 | 48.44% | 29281 | 8.351s |
| Critic readout FFN | 139 | 49.61% | 61208 | 8.144s |
| Local readout, no local initialization | 135 | 48.83% | 36888 | 8.526s |

No default promotion. Intermediate and critic FFNs show secondary money signals,
not win-rate improvements. The local-readout variants remain weaker than control.
All five modified PPO actors score0% against scripted-v27; control wins one of
256 games. Their BC actors won materially more often against that opponent.
These are one-training-seed, native-development-panel results, not official
Kaggle finalist evidence; game-seed intervals do not measure training variance.

The subsequently requested tile cross-attention RoPE is a separate extension.
Its learning source differs from the matched control source only in
`src/kaggriculture/entity.py`, excluding the separate action API corrections.

### Tile cross-attention RoPE extension

The user requested 2D RoPE for tile cross-attention after the original arms.
`--tile-cross-rope true` is independently off by default. Unit queries use
their `(column,row)` coordinates; both tile-memory grids use local farm
coordinates. Market queries, economy/private-unit keys, and all values remain
unrotated. Mixed scores still change on their spatial side; this is not a
pairwise exemption that preserves every mixed score. Entity SA, critic pooling,
and the optional local decoder are unchanged.

Learning/benchmark source:
`13d56d6086b61ed6bbd8dd37dd0b0cc79974c90861956aaa3a3d7ca93e61518f`.
Its sole file difference from control source02cfc15b is `entity.py`.
Live-code validation sourcead9ece04 has identical entity implementation and
also contains the separately verified action API corrections.

CUDA correctness jobs7386–7388 passed within their2-minute caps: a full-trunk
dense spatial forward/gradient oracle including both farm grids and nonspatial
memory; shared/untied equivalence with RoPE; combined-feature inactive-unit
isolation and backward behavior. Seven model-free CLI/artifact metadata checks
passed, as did scoped Ruff. Matching two-epoch BC7389 succeeded; held-out NLL
is0.001261689 versus control0.001380757. This alone is not gameplay evidence.

Cold full-size benchmark7390 hit its120s cap after complete repeats0 and1.
Both had230080 genuine states and29 actor plus29 critic-auxiliary updates.
Cold repeat0 reported74.201s inside its rollout/update timer, plus19.381s
for the separate replay-parity audit; warm repeat1 reported8.469s update and
10.838s total, plus2.206s parity audit. No numerical/runtime error was logged.
The failed cold attempt is retained. Job7394 repeats the identical three-repeat
workload with the populated compiler cache, still capped at120s; it is not a
BC/PPO retry or a cap extension.

Warm-cache gate7394 succeeded with all three full-size repeats and all29 actor
and critic-auxiliary updates per repeat. Its two warm medians are8.281s update
and10.825s total, versus8.184s and10.785s control. The update difference is
+1.19%, not a precise timing-distribution estimate. Actor/critic parameter counts
remain800810/835911. Learning7400 and fixed-panel evaluation7401 are admitted
under the unchanged24-minute trainer/25-minute hard-cap protocol.

Learning7400 and evaluation7401 succeeded on their first attempts. RoPE reached
131 iterations and3335 actor updates in24.1346 recorded training minutes.
Fixed-panel score is49.41% versus49.61% control: difference−0.195 percentage
points,95% interval[−1.172,+0.781]. Mean money is55627 versus55664:
difference−38,95% interval[−2979,+2792]. Money10th percentile improves to19794
from11801, but aggregate money and score do not establish a standalone benefit.
Scripted-v27 score is0%, versus28.52% for the matching BC actor.
No default promotion. Complete results are in `outcome-summary.json`.

### Requested four-feature combination

The user selected intermediate FFN, untied projected K/V, critic readout FFN,
and tile cross-attention RoPE together. The exact overrides are
`--inter-attention-ffn true --shared-memory-kv false
--critic-readout-ffn true --tile-cross-rope true`.
Local initialization remains enabled; final local readout remains disabled.
This is an interaction trial against the original control, not an attribution
experiment for any individual feature and not a default promotion.

Use the same frozen13d56d60 source as isolated RoPE, the same two-epoch BC
recipe, full-size230080-state/B8192 gate, training seed,24-minute trainer
budget/25-minute hard cap, and fixed native development panel. BC7402 feeds
full-size benchmark7403. Existing CUDA7386–7388 establish spatial gradients,
shared/untied equivalence, and combined-feature mask/backward correctness;
the exact selected combination must also pass its full-size gate before PPO.
All jobs remain priority0, maxParallelRuns1, maxAttempts1; probes/evals are
capped at2 minutes. Details are in `combined-four-campaign.json`.

BC7402 completed both epochs, held-out NLL0.001544537. Cold full-size gate7403
hit120s after complete repeats0 and1, with no stderr error; warm repeat1 update
was10.274s. The unchanged three-repeat warm-cache gate7408 then succeeded.
Warm medians are9.781s update and12.474s total; update overhead is19.51%
versus control. Every repeat contained230080 genuine states,29 actor updates,
and29 critic-auxiliary updates. Parameter counts are1052402 actor and1124847
critic. The cold failed attempt remains recorded, not relabeled successful.

Before PPO submission, the user explicitly permitted this combined run to exceed
25 minutes. Learning7413 therefore uses a55-minute trainer budget
(`--max-hours 0.9166666666666666`) and60-minute MLQ hard cap. Seed, training
settings, and architecture flags are otherwise unchanged. Evaluation7414 retains
the same fixed native panel and2-minute cap. The older24-minute control is
historical context, not a matched-compute architecture comparison for this
longer trial; any observed gain confounds architecture and additional training.
There is still one learning attempt, no automatic retries or default promotion.

Learning7413 and evaluation7414 succeeded on their first attempts. Actual queue
wall times were55.3514 minutes and0.5374 minutes, within their60-minute and
2-minute caps. The trainer recorded55.1962 minutes,271 iterations,62351680
states, and7366 actor updates; first actor update was at iteration18.
Saved configs verify exactly the four requested flags, local initialization on,
final local readout off, identical training settings except the authorized
runtime budget, and the expected frozen source.

| Fixed native panel | Combined BC | Combined PPO | Historical24-minute control PPO |
| --- | ---: | ---: | ---: |
| Overall score | 55.86% | 50.20% | 49.61% |
| Mean money | 57322 | 89467 | 55664 |
| Money10th percentile | 1445 | 47493 | 11801 |
| Starter score | 93.36% | 100.00% | 98.83% |
| Scripted-v27 score | 18.36% | 0.39% | 0.39% |

The combined actor wins256/256 starter games and1/256 scripted-v27 games.
Versus historical control, paired overall score difference is+0.586 percentage
points,95% interval[−0.195,+1.563]; mean-money difference is+33802,
95% interval[+30703,+36993]. The larger money result is real on this panel,
but cannot be attributed solely to architecture with the unequal budgets.
Versus its own BC actor, overall score decreases5.664 points,95% interval
[−8.594,−2.734], while money increases32145. The BC actor won47/256
scripted-v27 games. Additional money has not translated into strong-opponent
wins. No default promotion; no further learning run or retry was launched.
Complete per-game evidence, matched-configuration checks, paired seed-cluster
intervals, and terminal queue records are in `combined-four-campaign.json`,
`combined-four-evaluation.json`, and `outcome-summary.json`.
