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
