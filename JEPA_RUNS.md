# LeJEPA runs and ablation queue

Working list for the `lejepa` family (shared entity-attention backbone trained
by the LeJEPA objective and the policy's loss; the critic reads it detached). The
design itself is described in the README, section "LeJEPA world model".

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
- **The lejepa clone converges later than the entity clone.** Market kind 0.90
  after two epochs, 0.997 after twelve; the campaign now uses `BC_EPOCHS = (2, 12)`.
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
- **With the anchor the opening holds (9236).** 116 iterations at
  coefficient 0.05. Snapshots 16, 24 and 37 keep the clone's choice among the
  legal bins. By snapshot 37 unmasked cow mass had moved toward bin 15, which
  is illegal there, so it never plays. One run with other changes alongside,
  so the anchor is the likely cause, not a proven one. It is stable but not
  improving yet: self-play money ~88-93k flat, league money 125k -> 80-95k,
  critic explained variance 0.02-0.06 after warmup (ended ~iteration 14).
  Panel at iteration 38: v27 argmax 1.0, sampled 0.47, starter 1.0. Mean
  reference KL ~0.0005, but `max_reference_decision_kl` sits at 4-11. The
  outlier decisions are not in self-play states (CPU replay in fp32 and bf16
  both under 0.05), so they are most likely in league games; not yet found.
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
