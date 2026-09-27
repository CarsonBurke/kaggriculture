# Neural core selection and training

The task is to identify the best features of our models from September 12–26,
create a coherent core, train it, and submit the resulting neural policy.
The earlier public-script submissions were outside that intended objective.
They are not candidates in this campaign.

## Evidence and proposed core

| Feature | Evidence | Decision |
| --- | --- | --- |
| LeJEPA with policy gradients into its backbone | Detached readout collapsed off the teacher trajectory; the attached version restored sampled farming | Retain |
| One global policy readout | Per-slot readout plateaued at 76% market-kind accuracy; one global round reached 99.7% | Retain one round |
| Two global readout rounds | Completed checkpoint-85 matched panel: sampled V27 3/256 versus margin-control 44/256; both argmax 0/256 | Do not promote the deeper readout |
| Slower backbone optimizer | Historical matched 9281/9275 favored backbone LR 1.5e-5 with policy LR 1.5e-4; those results used old market rules | Retain as the conservative fresh-training recipe; measure under current rules |
| Categorical quantities with ALL | Percentage heads failed, but the matched ALL PPO arm OOMed; its later unpaired success does not establish superiority over legacy absolute quantities | Compare legacy and ALL directly |
| Local unit affordance scorer | A8 checkpoint gains replicated, but additional PPO confounded architectural attribution | Compare scorer on/off with matched training |
| Schema-v4 money-margin input | Better critic diagnostics but worse early matched gameplay; later strong policies do not isolate its contribution | Compare schema 3/4 with matched training |
| Source-read critic and hardness league | Useful critic/operational evidence; historical policy comparisons had confounds | Retain; do not claim isolated policy wins |
| Soft terminal outcome and V16 league lane | V1 and S1 argument dictionaries differ only in reward mode; S1 improves V16 gameplay, with some sampled-V27 tradeoff | Common current-rule objective for the new comparison |
| PPO objective | TPO eta1 ended at 0/64 against starter and V27 in both decoding modes; the later joint-ratio control also collapsed from strong BC to zero V27 wins | Retain component-ratio clipped PPO |
| Extra NextLat, writeback, strategic plans, causal market decoding, percentage quantities | No stronger validated policy at useful compute cost; causal arm was previously closed | Exclude from the core |

Primary sources: `JEPA_RUNS.md`, `RUN_COMPARISON_20260918.md`, `RUNS.md`,
`ACTION_INTERFACE_ABLATIONS.md`, and the actual action-stack configuration and
evaluation JSON files. Historical results under 1.32.6 are not current-rule
absolute performance claims.

## Why B1 and B2 looked worse

B1 was weak before PPO: frozen-actor self-play began around $1,146, with BC
quantity accuracy 77.16%. Its one-epoch run also changed the entire BC learning
rate and momentum schedule, so it was not simply a prefix of the two-epoch run.

B2 started around $54,748 and released its actor at iteration 15 around $56,447.
By iteration 27, self-play money fell to $4,434 while entropy rose from 0.096 to
0.291. It used a fresh clone, a fresh critic, and actor LR 0.0015. S1 used the
same rate but transferred a trained A8 actor and critic and remained healthy
over this interval. The learning rate alone is therefore not an established
cause. The core comparison uses the previously supported fresh-training rate
0.00015 rather than copying the continuation rate.

The PPO target KL is component-average KL, not joint turn KL. B2 iteration 27's
component KL 0.00395 was below 0.03 even though joint KL was 0.0319. Absence of a
KL stop is consistent with the implemented objective, not evidence the gate
failed.

## Matched full-training comparison

The executable recipe is `scripts/queue_core_campaign.py`; exact commands and
job IDs are in `artifacts/probes/core-model-20260926/campaign-feature-chain.json`.

- Legacy: schema 3, absolute quantities (interface 1), affordance disabled.
- Quantity change: schema 3, ALL quantities (interface 2), affordance disabled.
- Scorer change: schema 3, ALL quantities, affordance enabled.
- Margin change/control: schema 4, ALL quantities, affordance enabled.

All arms use the same four current-rule V16 demonstration datasets, two BC
epochs, batch 1024, seed 20260812, and BC optimizer schedule. Each receives the
same PPO workload: 128 self-play games plus 64 league games, 720-step episodes,
minibatch 4096, BF16 compiled execution, and one training seed, 20800000.
The budget is at most 500 iterations or two training hours, with a 150-minute
MLQ hard limit. Report common actor-update milestones as well as wall time;
cold-critic release times need not match. These are full learning runs, not
short pilot runs used as strength evidence.

The native architecture guard may cull after 150 actor waves when the panel
score EMA is at least five points below initialization and neither score EMA
nor critic MSE has materially improved for 100 waves. The best evaluated
checkpoint is retained. Weak but non-deteriorating runs require a separate
evidence-based review; the guard does not claim to catch every plateau.

PPO jobs depend on both full-game BC reports and execute
`scripts/admit_core_training.py` before starting training. Admission requires
argmax starter score >=0.9 and V27 score >=0.5, sampled starter score >=0.75,
and at most 25% of sampled games below $1,000 against each opponent. Reports
must contain at least 256 complete matched game records per opponent and bind
the exact clone and source hashes. These are initialization-adequacy floors,
not architecture-significance tests. A rejected clone leaves that arm's PPO
comparison unresolved; it does not refute the architecture. Six admission
tests cover adequate clones, sampled collapse, and invalid evidence.

Identical seeds control the run recipe but cannot
guarantee identical shared initial tensors when architecture shapes differ.
The four arms form a one-change chain. They measure conditional feature effects,
not all interactions. Legacy architecture is retrained with the common current
recipe; it is not a reproduction of every historical optimizer/reward choice.

The source is the immutable
`9a60ea219fffd8735305ebedb3e32751b1a8010772d7cfe819aa72d05aa61fd8`
snapshot. It contains a correction to the architecture panel: deployment scores
use the averaged actor, while critic calibration uses separate sampled rollouts
from the live actor. Previously, averaged-policy returns were compared with the
live-policy critic, confounding its calibration and the culling signal. The fix
passed 22 panel tests in both working trees, including regression coverage for
different deployment/live-policy outcomes and RNG/mode restoration.

The native extension is reused from `stack-1f45a59ef6063290` only after comparing
its Rust source and build-input bytes with the new snapshot. The earlier JEPA
dynamic-width compilation problem was already repaired with fixed row capacity
and masked padding; B1/B2 crossed the old failure point without recompilation.
Trainers record complete source and data identities.
This avoids silently mixing the dirty main and action-stack working trees,
whose inference, critic transfer, and actor-averaging implementations differ.
No unrelated working-tree changes are reverted.

## Selection and submission

Compare our existing A8, S1, R1 average, P2W, current-rule schema-v3 LeJEPA and
historical entity/source-read checkpoints on identical fresh development maps.
The current native panel balances seats across maps; it is not both seats of
every map. Final official-engine evaluation will use both seats per map.

Select trained core candidates using deployment argmax gameplay, retaining
sampled robustness and opponent-specific outcomes. Evaluate both live and
averaged policies where relevant; do not silently select the last checkpoint.
Keep screening and final selection seeds separate from training and reused
development seeds. Compare the best new core against the strongest existing
neural policy before promotion. Replace both active public-script submissions
with validated neural candidates; submitting another public controller is out
of scope.

Build a neural-only bundle with the actual selected weights and matching
source. Validate exact exported CPU inference, game completion, action timing,
seed provenance, and official-engine outcomes before Kaggle submission. CPU
validation matches Kaggle deployment; training and model development remain
queued CUDA BF16 work.

## Queue and current results

- 10225: completed current-checkpoint argmax panel, 512 fresh games per opponent.
- 10226: historical evaluation failed after completing schema-v3 built-in games;
  native frozen neural opponents require identical model configurations.
- 10241: replacement historical panel against common built-in opponents, queued.
- 10232 / 10233: control/schema3 BC completed successfully.
- 10234–10239 and 10243: canceled or dependency-skipped before execution when the
  unproven ALL feature was added to the matched comparison. Earlier superseded
  source/panel jobs 10229–10231 and 10240 were also canceled before execution.
- 10244 / 10245: legacy and schema3-no-affordance BC; 20-minute limits.
- 10246 / 10247: four-arm argmax/sampled BC panels; 20-minute limits.
- 10248–10251: admission-gated PPO in chain order; 150-minute limits.
- 10252: built-in-opponent selection panel after all four PPO jobs terminate;
  includes each endpoint and best checkpoint, historical schema3 and R1 references;
  50-minute limit. Cross-architecture head-to-head requires the official engine.
- All campaign jobs use priority 0, maximum parallel jobs 1, and one attempt.
- Other-project job 10242 currently occupies the GPU with a 12-hour cap; it has
  not been interrupted or reprioritized.

The fresh panel uses seeds 4700000–4700511. R1's average scored 1.000 against
S1 and 0.2285 against V16; S1 scored 0.0039 against R1 and 0.2207 against V16.
A8 scored 0.0039 against S1 and 0 against R1. Each candidate won every starter
game; R1/S1/P2W won every V27 game. These results select later checkpoints over
A8 but do not independently attribute their advantage to architecture features.

The historical schema3 checkpoint (`lejepa-1327-20260923`, iteration 162)
won 423/512 against V16 (82.62%), versus R1's 117/512 (22.85%) on these maps.
It also won 512/512 starter and 509/512 V27 games. These completed partial
results are recorded in attempt 8135 stdout; job 10241 will produce the complete
structured historical report. This is a strong candidate, not proof that any
one architectural difference caused the improvement.

Status: two BC arms completed; remaining core comparison queued; full PPO, final selection,
neural submission, and hosted validation remain unfinished.

## Step-budget and functional audit correction

The preceding 500-total-iteration / two-hour recipe is superseded. PPO jobs
10248–10251 are held before execution. Wall-clock endpoints and the current
`best_checkpoint` field are not valid architecture-selection criteria.

Actual logged history confirms that a wave is not universally equivalent:
at 50 actor-active waves, the old schema3 run had applied 1,450 actor minibatch
updates, while B2, S1 and R1 each applied 2,850. S1/R1 also inherit training from
previous checkpoints. Their aggregate historical panel scores cannot establish
causal feature advantages. Extracted evidence is in
`artifacts/probes/core-model-20260926/historical-step-panels.json`.

Revised comparison requirement: identical rollout workload, minibatch size and
one actor epoch; 500 actor-active waves excluding critic warmup; full checkpoint
and evaluation every 25 waves. Use the existing persisted actor-wave counter,
not a second unpersisted counter. Disable the internal wall-clock stop, retain
a generous operational timeout, and label timeout/cull results incomplete at
unreached milestones. Report cumulative environment transitions, actual actor
optimizer updates and skipped minibatches. Compare common achieved milestones;
do not force extra updates through a KL stop to equalize the counter. Historical
comparisons must additionally account for inherited BC/PPO exposure.

The current panel averages starter/V27 and both decoding modes. It omits V16,
can saturate on easy opponents, and mixes deployment with exploration quality.
Primary architecture evidence must use matched step checkpoints against V16,
V27 and a frozen strong neural reference, with opponent results separate.
Sampled robustness, live/averaged actors, collapse tails and learning curves are
secondary diagnostics. Keep held-out final selection separate from this panel.

Provisional core: attached LeJEPA backbone, one global readout, discrete integer
quantities, source-read private critic, component-ratio PPO and conservative
backbone rate. Schema3/absolute/no-scorer is the baseline, not a proven optimum.
Source-read, soft reward, hardness league and auxiliary losses have weaker
isolated policy evidence than attachment/global readout.

Code-supported limitations and next experiments:

1. The actor emits all decisions from the pre-action observation; quantity
   conditions on current kind, not the earlier action prefix. Feasibility masks
   repair legality but cannot update preferences using cash/inventory consumed
   by earlier choices (`lejepa_model.py:581`, `model.py:697`). First measure
   clipping, resource contention and unused capacity; if material, test one
   zero-initialized resource-ledger residual while preserving integer decoding.
2. Horizon-one LeJEPA predicts residual persistence and has no direct long-term
   economic target (`lejepa.py:347`). From an identical BC checkpoint, compare
   normal PPO auxiliaries with all auxiliary coefficients zero, preserving the
   attached backbone optimizer and policy gradients. Select on gameplay, not
   prediction loss. A shuffled-action/persistence probe diagnoses whether the
   predictor uses actions meaningfully.
3. The reward head averages MSE over rows although only terminal outcome rows
   have nonzero targets (`lejepa.py:879`). Separately test reward coefficient
   0.1 versus zero; sparse supervision is a limitation, not a confirmed bug.
4. Critic warmup and actor release deserve a same-actor initialization test:
   cold/readiness-gated critic versus separately fitted critic. Continuation
   results cannot isolate this because the actor is also pretrained longer.
5. Component-average KL can hide a large joint-turn change. Joint PPO previously
   failed, so do not simply switch objectives. Measure action-family KL, clipping,
   greedy action drift and shared-backbone movement around release; use this
   evidence to choose a targeted trust-region experiment.

Sequence: first the three conditional architecture differences (ALL, scorer,
margin); then auxiliary contribution and critic initialization on the selected
core. Test hard versus soft reward from the same initializer before calling
soft reward optimal. Avoid an indiscriminate Cartesian sweep. A failed BC gate
requires resolving initialization, not declaring the model feature inferior.
