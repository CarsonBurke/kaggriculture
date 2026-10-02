# LeJEPA runs and ablation queue

Working list for the `lejepa` family (shared entity-attention backbone trained
by the LeJEPA objective and the policy's loss; the critic reads it detached). The
design itself is described in [the training reference](../training-reference.md#lejepa-world-model).

## Market rules moved to kaggle-environments 1.32.7 (2026-09-23)

Kaggle has scored with 1.32.7 since Aug 15. That version gives carrot, tomato
and egg a "hinge" scarcity curve, `u + 8 * max(0, u - 1)^2` with `u = x / T`,
at 1.00 / 0.40 / 0.40 of base at the target. At 1.32.6 they had log 0.2, linear
0.4 and linear 0.4. A scarce carrot (inventory 9000) now quotes $531, not ~$42.
b72d797 had pinned us back to 1.32.6. Every run and evaluation above trained
and scored against the old curves. The leaderboard slide (1118 -> 715) and
public episode 112327315 (our farm stalled at ~25-30 tiles and sold ~360
units, against ~3,500 for the opponent; 20k vs 121k) are what old-rules policies
look like on the new market. Results above compare runs with each other on the
old rules. They don't transfer as absolute numbers.

- The engine, `constants.py` and the pin are all on 1.32.7. Our quotes match the
  official ones for every product at every inventory from 0 to 20000, plus 30k
  random probes. Engine and v27 parity audits are exact.
- The scripted v27 opponent keeps its own frozen 1.32.6 table, as the public
  script does (`V27_MARKET_PARAMS`).
- Loading the native engine checks its quotes against the installed
  kaggle-environments. A frozen snapshot can no longer train on one rule set
  while official evaluation plays another. That happened to 9289, whose
  snapshot predates the upgrade.
- BC rejects demonstrations whose quotes disagree with the current rules.
  Every `data/bc-*` corpus was recorded at 1.32.6. The matched v16 corpora
  moved to `data/bc-v16-1326-*`, and `scripts/extract_matched_bc_datasets.py`
  re-extracts `data/bc-v16-current-*` under 1.32.7. The v16 teacher never
  sells carrot, tomato or egg: in 64 mirror games no seller raised their
  inventory while quotes reached $724 / $786 / $284, so its money is identical
  under both rule sets. v27 sold none either (64 games against v16). A clone
  starts blind to the trade that dominates the new market; PPO must find it.
- The int32 policy price table saturates the hinge quotes it can't reach
  (tomato below -659k inventory; only wheat and fertilizer can be bought).
  Before this change, building the table failed.
- The price feature (`price / (2 * base)`) now reaches ~3-4 in native games.
  At 1.32.6 it stayed near 1, so it is out of distribution for any old
  checkpoint.

## Established (2026-09-22)

- **Per-slot heads cannot clone the teacher.** With each decision slot decoded
  from itself alone (a gated feed-forward per slot), a 12-epoch clone plateaued
  at 0.76 market-kind accuracy (holdout NLL 0.18); the entity actor reaches
  0.999. The world model keeps the economy in its own tokens, not in market
  slots. Run: `runs/lejepa-bc12-20260922/bc`.
- **One readout round fixes it.** `policy_readout_layers=1` (decision slots
  cross-attend to the whole detached belief) clones to 1.000 / 0.997 / 1.000
  unit / kind / quantity accuracy, holdout NLL 0.0033, at unchanged epoch time
  (~11 s). Objective healthy at the end: dispersion 1.17, SIGReg 0.058, motion
  0.166. Run: `runs/lejepa-readout-bc12-20260922/bc`.
- **The detached-readout lejepa clone converged later than the entity clone.**
  Market kind reached 0.90 after two epochs and 0.997 after twelve. Later
  attached-backbone PPO evidence favored the epoch-2 initializer (9248 below),
  so future campaigns inherit the two-epoch BC default.
- **A detached backbone collapses under sampling (9199).** The readout clone
  started PPO at self-play money ~50-560 from iteration 1 (entity: ~58k), lost
  to `pass` 69% of games, v27 panel 0.0; stopped. On CPU the same clone played
  88.8k/102k/70k greedy (identical to entity) but bankrupt in all four
  temperature-1 self-play games (seeds 101/202/303/404). The first sampled
  departures leave the teacher's trajectory and the farm is never recovered:
  6 weeds by day 3, 13 by day 5, animals 4 -> 0. Refuted as causes: STOP hazard
  on untrained post-STOP slots, and distribution flatness (entity is less sharp
  in its own sampled states and survives). The world-model-only features carry
  what predicts the teacher, not what a decision needs off-trajectory.
- **Letting the policy reach the backbone fixes it** (`policy_shapes_backbone`,
  now the default; job 9201, `runs/lejepa-attached-bc12-20260922/bc`). Holdout
  NLL 0.0005 (vs 0.0033), kind accuracy 0.9999; objective healthier, not
  weaker: motion 0.44 (vs 0.17), dispersion 1.24, SIGReg 0.076. Sampled CPU
  self-play 88.8k/102k/125.6k/70.6k, teacher level; greedy the same.
- **PPO then drifts the opening purchase off the teacher's (9207).** Healthy
  through iteration 24: self-play money ~89k, v27 panel 0.54-0.69, critic R²
  0.13 by iteration 15, joint KL ~0.001. At iteration 25, money fell to 4k,
  pass fraction rose 0.16 -> 0.35, entropy rose 0.004 -> 0.14 and motion fell
  to 0.28. Snapshots 14 and 23, played sampled on CPU, equal the clone
  exactly. Snapshot 24 is bankrupt, and its only greedy difference is the
  step-0 purchase. Grafting snapshot 23's step-0 move onto snapshot 24
  restores 50-126k. Across snapshots 14/20/23/24 the cow quantity logit for
  bin 1 went 13.4 -> 10.9 -> 7.95 -> -0.34 while bin 2 rose 0.28 -> 3.89. The
  opening flipped from cow 1 / sheep 4 / melon 2 to cow 2 / sheep 2 / melon 7.
  Mechanism: every game shares the step-0 state, and the probability of the
  opening there is ~1, so the surrogate has no restoring gradient. Shared
  parameters (quantity bias, value embeddings) move those logits through
  updates on other states. A mean KL cannot see one state flip, and most
  unmasked quantity mass sits on illegal bins, so the legal choice lives in
  the tail.
- **Fix: a reference KL anchor** (`--reference-kl-coefficient`): forward
  KL(clone || policy) over the masked unit, kind and quantity distributions.
  The quantity term is read at the stored kinds. The clone is reloaded from
  the warm-start artifact with a digest check. `actor/max_reference_decision_kl`,
  the largest single-decision KL in a wave, is the signal that sees one decision
  flip. Coefficient 0.05: the regularized optimum is proportional to
  clone x exp(A / 0.05). With raw advantage std ~0.13, a 1-std advantage can
  move a choice e^2.6, so the policy can still improve.
  **Removed since** (after 9275 below): the anchor kept the policy on the
  clone exactly where it was going bankrupt. The flag no longer exists; runs
  9236/9248/9267 reproduce only from source snapshot `c10dea49`.
- **With the anchor the opening holds (9236).** 116 iterations at
  coefficient 0.05. Snapshots 16, 24 and 37 keep the clone's choice among the
  legal bins. By snapshot 37 unmasked cow mass had moved toward bin 15, which
  is illegal there, so it never plays. One run with other changes alongside,
  so the anchor is the likely cause, not a proven one. It is stable but not
  improving yet: self-play money ~88-93k flat, league money 125k -> 80-95k,
  critic explained variance 0.13 at warmup's end, 0.01-0.03 over
  iterations 25-40, 0.16 at the end (entity baseline: 0.50).
  Panel at iteration 38: v27 argmax 1.0, sampled 0.47, starter 1.0. Mean
  reference KL ~0.0005, but `max_reference_decision_kl` sits at 4-11. The
  outlier decisions are not in self-play states (CPU replay in fp32 and bf16
  both under 0.05), so they are most likely in league games; not yet found.
- **The 12-epoch clone starved the critic; 2 epochs fixes it (9248).** The
  12 epochs were chosen for the detached readout clone. The attached clone
  reaches kind accuracy 0.9989 after 2 epochs, with holdout kind/quantity
  entropy 0.012/0.013 against 0.0004/0.0001 after 12, the same as the entity
  clone at its 2 epochs. PPO from `bc-actor-epoch-0002.pt`, anchored to it,
  same snapshot and arguments as 9236 otherwise:
  - Rollout entropy 0.16 falling to 0.12, against 0.003 for 9236.
  - Critic explained variance 0.45-0.55 from iteration ~60; panel R^2 0.50 ->
    0.68. 9236 reached 0.16 and a panel R^2 of -0.18 -> 0.29 with the same
    critic, so its weak critic came from near-deterministic data, not the
    critic.
  - Self-play money 29k -> 55-60k, still rising at iteration 118; league
    money 39k -> 50-60k. No KL early stops.
  - It starts far weaker than the 12-epoch clone: sampled v27 panel 0.00-0.05
    (9236: 0.62-0.86), sampled starter 0.86 -> 1.00. Argmax v27 0.83, 0.00,
    0.86, 0.92 at panels 38/63/88/113, where the entity baseline decayed to
    0.22 by 114.
  - Resumed for two more hours as 9267.
- **The anchor was the wrong fix; without it the epoch-2 run is the best yet
  (9275).** The anchor pulled every state back toward the clone, including
  the ones the policy was going bankrupt in, and the opening drift it guarded
  against came from the 12-epoch clone's saturation, which the epoch-2 clone
  lacks. Same start, snapshot and arguments as 9248, no anchor, 225
  iterations:
  - 10th-percentile money 19k at iteration 100, 32k at 175, 31-45k over
    214-225 (anchored 3.6k / 6.2k; component-control 5.5k / 18.8k; scalar 8.3k
    at 100).
  - Self-play money 62k at 100, 72k at 175, 74-81k at the end (component
    control 54k / 62k).
  - Argmax v27 panel 0.00 through iteration 88, then 0.20 / 0.23 / 0.48 / 0.56 /
    0.67 at 113-213; sampled v27 0.20-0.23 at 188-213; starter 1.00.
  - Entropy 0.19 at 50 falling to 0.095 (component-control rises to ~0.18);
    critic explained variance 0.51 at 100 falling to 0.30-0.39.
  - No sign of the opening drift. Cancelled by request at iteration 225.
- **A frozen reference must not use the custom embedding backward.**
  `_TinyVocabularyEmbedding.apply` with no input requiring grad fails Dynamo
  (`'Function' object is not subscriptable`, gate 9233). `TileEmbedder` now
  takes the plain gather unless a table receives a gradient.
- **GPU idle blocks are rollout stalls, not compiles.** The update is a steady
  ~7.5 s. Rollout is ~2.1 s, but 28-38 s at iterations 26-28 and 35-36 with no
  compile, at load average 48 with other sessions building and testing. The
  Rust environment forks and joins a 24-thread rayon pool every step, so one
  descheduled worker stalls the step, and mlq jobs get half the CPU weight
  while another scope is busy. League recompiles (4-5.5 s each at iterations
  16-19 and 23) happen only when a new power-of-two (lanes, width) bucket
  appears while the historical pool fills. Members themselves reuse the
  stacked ensemble's buffers, so iterations 24-116 compiled nothing.

## In flight

| Job | What | Status |
| --- | --- | --- |
| 9198 | Gate, detached readout clone | done: 5.9 s/it (rollout 2.05, update 3.71), replay parity exact, peak reserved 25.2 GiB (over budget), KL early stop every iteration |
| 9199 | 25-minute PPO, detached, `runs/lejepa-full-20260922/ppo` | stopped: collapsed from iteration 1 (above) |
| 9206 | Gate, attached clone (9201) | done: money 88k -> 92k, no KL early stops, joint KL ~0.001, motion 0.43, 8.2 s/it, peak reserved 25.6 GiB (over budget) |
| 9207 | 25-minute PPO, attached, `runs/lejepa-attached-20260922/ppo` | stopped at iteration 25: opening-purchase drift (above) |
| 9233 | Gate, anchor 0.05 | failed: Dynamo on the frozen reference's embedding (fixed, above) |
| 9235 | Gate, anchor 0.05, default compile | done: 9.48 s/it (update 7.30), peak reserved 23.89 GiB |
| 9236 | 25-minute PPO, anchor 0.05, `runs/lejepa-anchored-20260922/ppo` | done, 116 iterations: no collapse, but flat money and weak critic (above) |
| 9248 | 25-minute PPO from the epoch-2 clone, `runs/lejepa-bc2-anchored-20260922/ppo` | done, 118 iterations: improving, critic healthy (above) |
| 9267 | 9248 resumed, two hours | cancelled by request after ~30 min at iteration 231: still improving (sampled v27 panel 0.22, argmax 0.94 at 213) |
| 9275 | 9248 without the anchor, `runs/lejepa-bc2-free-20260922/ppo` | cancelled by request at iteration 225: best lejepa run, ahead of the entity controls on money (above) |
| 9281 | 9275 with the backbone at 1.5e-5, `runs/lejepa-bc2-free-bblr-20260922/ppo` | cancelled by request at iteration 246: new best, v27 argmax 1.00 from 138; now the default (below) |
| 9289 | 9281 with a stop-gradient target (`--jepa-detach-target`), `runs/lejepa-bc2-stopgrad-bblr-20260922/ppo` | cancelled at iteration ~250 by the rules move: a tie. v27 sampled 0.70 vs 0.64 at 238, 0.61 vs 0.59 at 188 (64 games, within noise); argmax 1.00 from 138 in both; critic MSE 0.22 vs 0.21 |
| 9300 | New rules: epoch-2 lejepa clone on the re-extracted `data/bc-v16-current-*` corpora, `runs/lejepa-1327-20260923/bc`; its epoch times stand in for the cancelled remat probe (9293) | succeeded: holdout NLL 0.0026, unit/kind/quantity accuracy 1.000/0.999/0.999 |
| 9301 | Original eight-hour PPO after 9300 | cancelled before start; replaced by 9302 |
| 9302 | New rules: PPO from 9300, backbone 1.5e-5, originally 35-minute trainer budget and 40-minute hard limit, `runs/lejepa-1327-20260923/ppo` | cancelled by request at iteration 173; checkpoint 162 retained |
| 9303 | New rules: schema-v4 money-margin clone, `runs/lejepa-margin-1327-20260923/bc` | succeeded after two epochs; holdout NLL 0.00228 |
| 9304 | Money-margin PPO from 9303, `runs/lejepa-margin-1327-20260923/ppo` | stopped at the 30-minute queue limit at iteration 89; checkpoint 85 retained |
| 9305 | Product-sales diagnostic from v3 checkpoint 87, 64 full sampled native self-play games | succeeded: 141 carrot, 7 tomato, 0 egg units sold |
| 9298 | New rules: PPO fine-tune of 9281 iteration 238 (`runs/lejepa-hinge-20260923/init/actor-9281-238.pt`, fresh critic) | cancelled by request before start, in favour of a fresh clone and PPO run on the new rules |

Baseline for comparison at matched iterations: entity component-control,
`runs/structural-gae-20260918/component-control/ppo` (v27 argmax panel 0.906 at
wave 39, decaying to 0-0.28; starter ~1.0 throughout).

Stop criteria for 9199: critic-warmup R² not reaching 0.10 by wave ~30,
`first_minibatch_component_kl` near 0.11, `replay_parity_breached`,
`structured_actor_motion` or `_dispersion` falling toward 0, persistence
prediction ratio rising toward 1, shuffled prediction ratio near 1, v27 panel
decaying faster than the baseline. If healthy, resume the same run directory
with a longer `--max-hours` rather than restarting.

## Ablation candidates

**Backbone LR 10x below the policy's won (9281 vs 9275)** and is now every
`lejepa` PPO run's default (`JEPA_BACKBONE_LEARNING_RATE` in
`scripts/queue_structural_campaign.py`). Same snapshot, clone and recipe; only
`--structured-learning-rate 1.5e-5` differs. One seed, but the gap is large and
monotone:

| iteration | v27 sampled / argmax (1.5e-5) | v27 sampled / argmax (1.5e-4) | critic MSE (1.5e-5 / 1.5e-4) |
|---|---|---|---|
| 63 | 0.11 / 0.91 | 0.00 / 0.00 | 0.39 / 0.42 |
| 113 | 0.36 / 0.98 | 0.03 / 0.20 | 0.31 / 0.54 |
| 163 | 0.50 / 1.00 | 0.12 / 0.48 | 0.28 / 0.33 |
| 213 | 0.58 / 1.00 | 0.20 / 0.67 | 0.26 / 0.40 |
| 238 | 0.64 / 1.00 | - | 0.21 / - |

Self-play money mean runs ahead in every 25-iteration window (81.8k vs 75.6k
over 200-224) at similar 10th percentile (36.8k vs 36.2k), with rollout entropy
lower (0.046 vs 0.097) and panel critic MSE falling instead of rising. Panel
score rates are over 64 games per cell.

**Stop-gradient target tied the attached one (9289 vs 9281, above)** on the
old rules, so the attached target stays.

**Remaining arms, from the 1.5e-5 baseline, now on the 1.32.7 rules:**

- Money margin: observation schema v4 adds each farm's log-money margin over
  the other beside the absolute feature, which bf16 resolves only to ~4.8%.
  Its matched epoch-2 clone succeeded as job 9303; PPO job 9304 ran to the
  30-minute queue limit, retaining checkpoint 85.

The money-margin arm uses the same 1.32.7 source snapshot and four current-rules
corpora as 9300/9302. Only `--observation-schema-version 4` changes the model's
input; the clone output and PPO run directory are separate. Job 9303 uses the
two-epoch BC budget and a 30-minute queue limit. Job 9304 depends on 9303's
success, uses the same seed and PPO recipe as 9302, and has a 35-minute trainer
argument; its running queue limit was reduced from 40 to 30 minutes by
decision. Both use normal priority and
`maxParallelRuns=1`. The baseline's schedule is unchanged. Compare sampled and
argmax v27 panels, self-play money and its lower tail, critic fit, and executed
sales by product before attributing any gain to the margin feature.

At matched actor wave 25, the margin arm improves panel critic MSE (0.337
versus 0.497) and self-play money's 10th percentile (~5.8k versus ~1.4k),
but loses ground against scripted v27: sampled score 0.078 versus 0.188 and
argmax score 0/64 versus 61/64. Starter sampled score is 0.938 versus 0.906.
This early result is mixed. By decision, fresh LeJEPA runs now inherit
schema v4 immediately for its stronger critic and self-play economics; the
v27 panel regression remains a measured risk to resolve, not a claimed win.
Existing v3 artifacts still load with their saved schema. Promote later clear
winners into the next fresh run without waiting for a whole campaign to end.

At actor wave 50, scripted-v27 scores remain below the matched v3 baseline:
sampled **0.094 versus 0.250** and argmax **0/64 versus 64/64**. Starter sampled
is 0.969 versus 0.938. Monitor this opponent result closely even though the
fresh-run default was promoted by explicit decision.

At wave 75, margin critic MSE is **0.211** versus **0.529** for v3; self-play
money mean is **69.0k** versus **64.0k**, and its 10th percentile is **10.0k**
versus **3.3k**. Sampled scripted-v27 score is **0.125** versus **0.250**;
argmax is **0/64** versus **63/64**. Job 9304 reached the 30-minute queue
deadline at iteration 89 and exited on SIGTERM. Checkpoint 85 is the latest
durable one. The promotion is for economics and critic learning, with the
opponent-strength regression unresolved.

Future jobs launched for this line of work have a **30-minute maximum wall
time**. To allow a clean final checkpoint, give the trainer at most 27 minutes
and leave the rest to the queue deadline. Longer studies need separate resume
chunks, each within the same per-job limit.

**Queued action follow-up (2026-09-23).** Jobs **9315/9316** are a matched
two-epoch BC and 27-minute PPO ablation of `policy_readout_layers=2` versus
the margin baseline's one round. They reuse the exact 1.32.7 frozen source and
schema-v4 training data; the readout depth is the only model change, while the
trainer budget is shorter under the new per-job limit. Jobs **9319/9320**
will compare sampled and argmax policies with margin checkpoint 85 on 256
common development seeds per opponent after PPO terminates. Separately,
**9317/9318** measure v3/v4 carrot, tomato and egg sales, HIRE timing,
pickup-cap use, wheat placement and feeding on the same 64 native self-play
seeds. Both manifests are under `artifacts/probes/lejepa-{readout2,action-audit}-1327-20260923/`.
Results are pending shared GPU admission. Each job is bounded by 30 minutes.
The separate joint-ratio jobs **9321–9323** were cancelled before start after
the priority was clarified to be action *decoding*. The readout-depth
arm remains queued. The dedicated execution-order causal decoder comparison,
with compiled gates and a flat joint-PPO control, is recorded in
[docs/experiments/runs.md](runs.md); corrected jobs **9336–9347** are each bounded by 30
minutes. The first submission **9324–9335** was cancelled or skipped before
starting because its GPU gate omitted the tests' required enable flags.

The older list follows.

From `../cleanrl/cleanrl/ppo_continuous_action_jepa_ngpt_residual_v1.py`
(HalfCheetah-v4, 50M steps, one seed: final 12.3k vs 11.2k for the no-JEPA
control; trivial MLP encoder, so weak evidence for an attention backbone).
Ordered by expected value here.

1. **Stop-gradient target instead of attached.** cleanrl detaches the
   next-step embedding (no EMA); its docstring (:195-198) reports an attached
   target "shrank the backbone and doubled coordinate drift". Ours is attached
   (LeWM). Largest open question; our reward head may offset some of the
   pressure. Readouts: motion, dispersion, backbone scale, drift, panel.
2. **Backbone LR well below the policy LR, annealed with it.** cleanrl's SSL LR
   is ~50x below PPO's and follows its schedule (:116, :561-562), so the actor's
   input barely moves (drift ~9e-5 per rollout). Ours steps at the actor LR
   inside the KL trust region. No code needed: `--structured-learning-rate`.
3. **Hypersphere-normalized residual heads** with per-step matrix
   renormalization (:154-171, :321-331). Permits large sustained PPO steps;
   independent of the encoder, so applies to the readout round and heads.
4. **AdaLN-zero action-conditioned predictor** (:174-189): shift/scale/gate
   modulation from the action with zero-initialized modulation weights.
5. **Representation-drift metric on fixed probe rows** (:536, :637-638). Cheap
   diagnostic; worth adding to telemetry regardless of the ablations.

Already covered or not transferable:

- Scaled unit-vector input to the heads (`sqrt(d) * normalize(z)`, :335): our
  RMSNorm on the belief slots is the same map.
- Critic on raw observations (cleanrl, and its best geometry arm): our critic
  needs privileged opponent inputs and reads them through its private tower;
  a raw-observation critic would be a separate encoder, not a head change.

Carried over from the design review, awaiting a decision:

6. SIGReg scope: whether the tile and economy latents should carry the floor
   (economy is the group nearest it: SIGReg 0.20 at the end of BC vs 0.01-0.015
   for the others).
7. Predictor conditioned on the opponent's action as well as the seat's own.
8. Dense outcome target for the reward head. Under terminal-outcome rewards
   about one row in 720 is nonzero, so at coefficient 0.1 the reward term
   barely shapes the backbone.
9. Collapse gate on `structured_actor_motion`.
10. `policy_readout_layers` 2 vs 1, and 0 (linear probe) as the control.
